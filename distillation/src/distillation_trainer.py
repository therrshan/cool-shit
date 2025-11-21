import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel
from transformers import get_linear_schedule_with_warmup
import json
from pathlib import Path
from tqdm import tqdm
import wandb
from datetime import datetime
import os

class DistillationDataset(Dataset):
    """
    Custom dataset for knowledge distillation
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        full_text = item['full_text']
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

class KnowledgeDistillationTrainer:
    """
    Trainer for knowledge distillation from teacher to student model
    """
    def __init__(
        self,
        teacher_model,
        student_model,
        tokenizer,
        train_dataloader,
        val_dataloader,
        device='cuda',
        temperature=2.0,
        alpha=0.5,
        learning_rate=5e-5,
        num_epochs=3,
        output_dir='models/',
        use_wandb=False
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Move models to device
        self.student = self.student.to(device)
        # Teacher is already on device from quantization
        
        # Set teacher to eval mode
        self.teacher.eval()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Scheduler
        total_steps = len(train_dataloader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Loss function
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        # Tracking
        self.best_val_loss = float('inf')
        self.training_stats = []
    
    def distillation_loss(self, student_logits, teacher_logits, labels, attention_mask):
        """
        Compute distillation loss combining:
        1. Soft targets (KL divergence with teacher)
        2. Hard targets (cross-entropy with labels)
        """
        # Mask out padding tokens
        active_loss = attention_mask.view(-1) == 1
        active_logits = student_logits.view(-1, student_logits.size(-1))[active_loss]
        active_labels = labels.view(-1)[active_loss]
        
        # Hard target loss (standard cross-entropy)
        hard_loss = self.ce_loss(active_logits, active_labels)
        
        # Soft target loss (KL divergence with teacher)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Only compute KL loss on active tokens
        student_soft_active = student_soft.view(-1, student_soft.size(-1))[active_loss]
        teacher_soft_active = teacher_soft.view(-1, teacher_soft.size(-1))[active_loss]
        
        soft_loss = self.kl_loss(student_soft_active, teacher_soft_active) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss, hard_loss.item(), soft_loss.item()
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        """
        self.student.train()
        total_loss = 0
        total_hard_loss = 0
        total_soft_loss = 0
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Shift for language modeling
            labels = input_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            # Get teacher outputs (no gradient)
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits
            
            # Get student outputs
            student_outputs = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            student_logits = student_outputs.logits
            
            # Compute distillation loss
            loss, hard_loss, soft_loss = self.distillation_loss(
                student_logits, teacher_logits, labels, attention_mask
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            total_hard_loss += hard_loss
            total_soft_loss += soft_loss
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'hard': f'{hard_loss:.4f}',
                'soft': f'{soft_loss:.4f}'
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/hard_loss': hard_loss,
                    'train/soft_loss': soft_loss,
                    'train/lr': self.scheduler.get_last_lr()[0]
                })
        
        avg_loss = total_loss / len(self.train_dataloader)
        avg_hard_loss = total_hard_loss / len(self.train_dataloader)
        avg_soft_loss = total_soft_loss / len(self.train_dataloader)
        
        return avg_loss, avg_hard_loss, avg_soft_loss
    
    def validate(self):
        """
        Validate the model
        """
        self.student.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                # Teacher outputs
                teacher_outputs = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits
                
                # Student outputs
                student_outputs = self.student(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                student_logits = student_outputs.logits
                
                # Compute loss
                loss, _, _ = self.distillation_loss(
                    student_logits, teacher_logits, labels, attention_mask
                )
                
                total_loss += loss.item()
        
        avg_val_loss = total_loss / len(self.val_dataloader)
        return avg_val_loss
    
    def train(self):
        """
        Main training loop
        """
        print("\n" + "="*60)
        print("STARTING DISTILLATION TRAINING")
        print("="*60 + "\n")
        
        for epoch in range(self.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.num_epochs} ---")
            
            # Train
            train_loss, hard_loss, soft_loss = self.train_epoch(epoch)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"  Hard Loss: {hard_loss:.4f}")
            print(f"  Soft Loss: {soft_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            print(f"Val Loss: {val_loss:.4f}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_loss,
                    'val/loss': val_loss
                })
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f'best_model_epoch_{epoch+1}.pt')
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Save checkpoint every epoch
            self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
            
            # Track stats
            self.training_stats.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'hard_loss': hard_loss,
                'soft_loss': soft_loss
            })
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        
        # Save training stats
        stats_path = self.output_dir / 'training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        print("\n" + "="*60)
        print("✓ TRAINING COMPLETE!")
        print("="*60 + "\n")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Models saved in: {self.output_dir}")
    
    def save_checkpoint(self, filename):
        """
        Save model checkpoint
        """
        checkpoint_path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_stats': self.training_stats
        }, checkpoint_path)

def main():
    """
    Main function to run distillation training
    """
    # Configuration
    CONFIG = {
        'teacher_model': 'mistralai/Mistral-7B-Instruct-v0.2',
        'train_data': 'data/train.json',
        'val_data': 'data/val.json',
        'batch_size': 4,
        'max_length': 512,
        'temperature': 2.0,
        'alpha': 0.5,
        'learning_rate': 5e-5,
        'num_epochs': 3,
        'output_dir': 'models/',
        'use_wandb': False  # Set to True if you want to use W&B
    }
    
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Load teacher model
    print("\nLoading teacher model...")
    from setup_models import setup_teacher_model
    teacher, tokenizer = setup_teacher_model(CONFIG['teacher_model'])
    
    # Create student model
    print("\nCreating student model...")
    from setup_models import setup_student_model
    student, _ = setup_student_model(vocab_size=len(tokenizer))
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = DistillationDataset(
        CONFIG['train_data'],
        tokenizer,
        max_length=CONFIG['max_length']
    )
    val_dataset = DistillationDataset(
        CONFIG['val_data'],
        tokenizer,
        max_length=CONFIG['max_length']
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=2
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # Initialize W&B if requested
    if CONFIG['use_wandb']:
        wandb.init(
            project='knowledge-distillation',
            config=CONFIG
        )
    
    # Initialize trainer
    trainer = KnowledgeDistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        temperature=CONFIG['temperature'],
        alpha=CONFIG['alpha'],
        learning_rate=CONFIG['learning_rate'],
        num_epochs=CONFIG['num_epochs'],
        output_dir=CONFIG['output_dir'],
        use_wandb=CONFIG['use_wandb']
    )
    
    # Train
    trainer.train()
    
    if CONFIG['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    main()