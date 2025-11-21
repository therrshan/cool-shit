"""
Configuration file for Knowledge Distillation project
"""

# Model Configuration
MODEL_CONFIG = {
    'teacher_model': 'mistralai/Mistral-7B-Instruct-v0.2',
    
    # Student model architecture
    'student_hidden_size': 768,
    'student_num_layers': 12,
    'student_num_heads': 12,
    'student_max_length': 2048,
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 4,  # Adjust based on GPU memory
    'gradient_accumulation_steps': 4,  # Effective batch size = 16
    'max_length': 512,
    'num_epochs': 3,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
}

# Distillation Configuration
DISTILLATION_CONFIG = {
    'temperature': 2.0,  # Temperature for soft targets
    'alpha': 0.5,  # Weight for soft loss (1-alpha for hard loss)
}

# Data Configuration
DATA_CONFIG = {
    'train_data': 'data/train.json',
    'val_data': 'data/val.json',
    'test_data': None,  # Optional
}

# Output Configuration
OUTPUT_CONFIG = {
    'output_dir': 'models/',
    'logs_dir': 'logs/',
    'checkpoint_every_n_epochs': 1,
    'save_best_only': False,
}

# Weights & Biases Configuration
WANDB_CONFIG = {
    'use_wandb': False,  # Set to True to enable W&B logging
    'project_name': 'knowledge-distillation',
    'run_name': None,  # Auto-generated if None
}

# Hardware Configuration
HARDWARE_CONFIG = {
    'device': 'cuda',
    'mixed_precision': True,  # Use FP16 training
    'num_workers': 2,
}