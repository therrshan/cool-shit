import torch
import time
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

class ModelEvaluator:
    """
    Comprehensive evaluation framework for teacher and student models
    """
    def __init__(self, teacher_model, student_model, tokenizer, device='cuda'):
        self.teacher = teacher_model
        self.student = student_model
        self.tokenizer = tokenizer
        self.device = device
        
        # Set models to eval mode
        self.teacher.eval()
        self.student.eval()
        
        self.results = {
            'teacher': {},
            'student': {},
            'comparison': {}
        }
    
    def measure_inference_speed(self, model, num_samples=100, max_length=128):
        """
        Measure inference latency and throughput
        """
        print(f"\nMeasuring inference speed ({num_samples} samples)...")
        
        test_prompts = [
            "Explain quantum computing in simple terms:",
            "What are the benefits of exercise?",
            "Write a short story about a robot:",
            "How does photosynthesis work?",
            "What is machine learning?"
        ] * 20  # Repeat to get 100 samples
        
        latencies = []
        
        for prompt in tqdm(test_prompts[:num_samples], desc="Speed test"):
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            end_time = time.time()
            
            latency = end_time - start_time
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        throughput = 1.0 / avg_latency
        
        return {
            'avg_latency_seconds': avg_latency,
            'std_latency_seconds': std_latency,
            'throughput_samples_per_sec': throughput,
            'p50_latency': np.percentile(latencies, 50),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99)
        }
    
    def measure_memory_usage(self, model):
        """
        Measure GPU memory usage
        """
        print("\nMeasuring memory usage...")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Run inference
        test_input = "Test memory usage with a longer prompt to simulate real usage scenarios."
        inputs = self.tokenizer(test_input, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=100, pad_token_id=self.tokenizer.pad_token_id)
        
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
        max_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        
        return {
            'memory_allocated_gb': memory_allocated,
            'memory_reserved_gb': memory_reserved,
            'max_memory_gb': max_memory
        }
    
    def measure_model_size(self, model):
        """
        Calculate model size in parameters and disk space
        """
        print("\nMeasuring model size...")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate disk size (FP32)
        size_mb = total_params * 4 / (1024**2)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'size_mb_fp32': size_mb,
            'size_mb_fp16': size_mb / 2,
            'parameters_readable': f"{total_params/1e6:.2f}M" if total_params < 1e9 else f"{total_params/1e9:.2f}B"
        }
    
    def evaluate_perplexity(self, model, test_data_path='data/val.json', max_samples=500):
        """
        Calculate perplexity on test set
        """
        print(f"\nEvaluating perplexity on {max_samples} samples...")
        
        # Load test data
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)[:max_samples]
        
        total_loss = 0
        total_tokens = 0
        
        for item in tqdm(test_data, desc="Computing perplexity"):
            text = item['full_text']
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                # Count tokens (excluding padding)
                num_tokens = (inputs['attention_mask'] == 1).sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'total_tokens_evaluated': total_tokens
        }
    
    def evaluate_generation_quality(self, model, num_samples=20):
        """
        Qualitative evaluation of generation quality
        """
        print(f"\nEvaluating generation quality ({num_samples} samples)...")
        
        test_prompts = [
            "Explain what machine learning is:",
            "Write a creative story about a dragon:",
            "What are the health benefits of meditation?",
            "How do solar panels work?",
            "Describe the water cycle:",
            "What is the theory of relativity?",
            "Give me a recipe for chocolate chip cookies:",
            "Explain blockchain technology:",
            "What causes the seasons to change?",
            "How does the immune system work?",
            "Write a poem about nature:",
            "What is photosynthesis?",
            "Explain supply and demand:",
            "How do airplanes fly?",
            "What is artificial intelligence?",
            "Describe the process of cell division:",
            "What causes earthquakes?",
            "How does the internet work?",
            "Explain the carbon cycle:",
            "What is DNA?"
        ]
        
        generations = []
        
        for prompt in tqdm(test_prompts[:num_samples], desc="Generating samples"):
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            generations.append({
                'prompt': prompt,
                'generated_text': generated_text,
                'length': len(generated_text.split())
            })
        
        avg_length = np.mean([g['length'] for g in generations])
        
        return {
            'num_samples': num_samples,
            'avg_generation_length_words': avg_length,
            'samples': generations[:5]  # Save first 5 for inspection
        }
    
    def evaluate_mmlu_subset(self, model, num_samples=100):
        """
        Evaluate on MMLU (Massive Multitask Language Understanding) subset
        Tests reasoning and knowledge
        """
        print(f"\nEvaluating on MMLU subset ({num_samples} samples)...")
        
        try:
            # Load MMLU dataset (using a subset)
            dataset = load_dataset("cais/mmlu", "all", split="test", streaming=True)
            
            correct = 0
            total = 0
            
            for i, item in enumerate(dataset):
                if i >= num_samples:
                    break
                
                question = item['question']
                choices = item['choices']
                correct_answer_idx = item['answer']
                
                # Format prompt
                prompt = f"Question: {question}\n"
                for idx, choice in enumerate(choices):
                    prompt += f"{chr(65+idx)}) {choice}\n"
                prompt += "Answer:"
                
                inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=5,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Check if correct (simple matching)
                if chr(65 + correct_answer_idx) in generated[:20]:
                    correct += 1
                
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            
            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
        except Exception as e:
            print(f"MMLU evaluation failed: {e}")
            return {'accuracy': None, 'error': str(e)}
    
    def run_full_evaluation(self):
        """
        Run comprehensive evaluation on both models
        """
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        # Teacher evaluation
        print("\n>>> Evaluating TEACHER model...")
        self.results['teacher']['speed'] = self.measure_inference_speed(self.teacher, num_samples=50)
        self.results['teacher']['memory'] = self.measure_memory_usage(self.teacher)
        self.results['teacher']['size'] = self.measure_model_size(self.teacher)
        self.results['teacher']['perplexity'] = self.evaluate_perplexity(self.teacher, max_samples=200)
        self.results['teacher']['generation'] = self.evaluate_generation_quality(self.teacher, num_samples=10)
        self.results['teacher']['mmlu'] = self.evaluate_mmlu_subset(self.teacher, num_samples=50)
        
        # Student evaluation
        print("\n>>> Evaluating STUDENT model...")
        self.results['student']['speed'] = self.measure_inference_speed(self.student, num_samples=50)
        self.results['student']['memory'] = self.measure_memory_usage(self.student)
        self.results['student']['size'] = self.measure_model_size(self.student)
        self.results['student']['perplexity'] = self.evaluate_perplexity(self.student, max_samples=200)
        self.results['student']['generation'] = self.evaluate_generation_quality(self.student, num_samples=10)
        self.results['student']['mmlu'] = self.evaluate_mmlu_subset(self.student, num_samples=50)
        
        # Compute comparisons
        self.compute_comparisons()
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def compute_comparisons(self):
        """
        Compute comparison metrics between teacher and student
        """
        print("\n>>> Computing comparisons...")
        
        t_speed = self.results['teacher']['speed']['avg_latency_seconds']
        s_speed = self.results['student']['speed']['avg_latency_seconds']
        
        t_size = self.results['teacher']['size']['total_parameters']
        s_size = self.results['student']['size']['total_parameters']
        
        t_memory = self.results['teacher']['memory']['max_memory_gb']
        s_memory = self.results['student']['memory']['max_memory_gb']
        
        t_perplexity = self.results['teacher']['perplexity']['perplexity']
        s_perplexity = self.results['student']['perplexity']['perplexity']
        
        self.results['comparison'] = {
            'speedup_factor': t_speed / s_speed,
            'size_reduction_factor': t_size / s_size,
            'memory_reduction_factor': t_memory / s_memory,
            'perplexity_ratio': s_perplexity / t_perplexity,
            'size_reduction_percent': (1 - s_size/t_size) * 100,
            'memory_reduction_percent': (1 - s_memory/t_memory) * 100,
            'speed_improvement_percent': (1 - s_speed/t_speed) * 100
        }
    
    def save_results(self, output_dir='outputs/'):
        """
        Save evaluation results to JSON
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_file}")
    
    def print_summary(self):
        """
        Print evaluation summary
        """
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        print("\n>>> PERFORMANCE METRICS:")
        print(f"  Teacher Speed: {self.results['teacher']['speed']['avg_latency_seconds']:.4f}s per sample")
        print(f"  Student Speed: {self.results['student']['speed']['avg_latency_seconds']:.4f}s per sample")
        print(f"  Speedup: {self.results['comparison']['speedup_factor']:.2f}x faster")
        
        print("\n>>> MODEL SIZE:")
        print(f"  Teacher: {self.results['teacher']['size']['parameters_readable']}")
        print(f"  Student: {self.results['student']['size']['parameters_readable']}")
        print(f"  Size Reduction: {self.results['comparison']['size_reduction_percent']:.1f}%")
        
        print("\n>>> MEMORY USAGE:")
        print(f"  Teacher: {self.results['teacher']['memory']['max_memory_gb']:.2f} GB")
        print(f"  Student: {self.results['student']['memory']['max_memory_gb']:.2f} GB")
        print(f"  Memory Reduction: {self.results['comparison']['memory_reduction_percent']:.1f}%")
        
        print("\n>>> QUALITY METRICS:")
        print(f"  Teacher Perplexity: {self.results['teacher']['perplexity']['perplexity']:.2f}")
        print(f"  Student Perplexity: {self.results['student']['perplexity']['perplexity']:.2f}")
        print(f"  Quality Retention: {(1/self.results['comparison']['perplexity_ratio'])*100:.1f}%")
        
        if self.results['teacher']['mmlu']['accuracy'] is not None:
            print(f"\n>>> MMLU ACCURACY:")
            print(f"  Teacher: {self.results['teacher']['mmlu']['accuracy']*100:.1f}%")
            print(f"  Student: {self.results['student']['mmlu']['accuracy']*100:.1f}%")

def load_trained_student(checkpoint_path, tokenizer):
    """
    Load a trained student model from checkpoint
    """
    from transformers import GPT2Config, GPT2LMHeadModel
    
    # Create model with same config
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=2048,
        n_embd=768,
        n_layer=12,
        n_head=12,
    )
    
    student = GPT2LMHeadModel(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    student.load_state_dict(checkpoint['model_state_dict'])
    
    return student

def main():
    """
    Main evaluation function
    """
    print("Loading models for evaluation...")
    
    # Load teacher
    from setup_models import setup_teacher_model
    teacher, tokenizer = setup_teacher_model()
    
    # Load trained student
    student_checkpoint = 'models/best_model_epoch_3.pt'  # Adjust as needed
    student = load_trained_student(student_checkpoint, tokenizer)
    student = student.to('cuda')
    
    # Run evaluation
    evaluator = ModelEvaluator(teacher, student, tokenizer)
    results = evaluator.run_full_evaluation()
    
    print("\n✓ Evaluation complete!")

if __name__ == "__main__":
    main()