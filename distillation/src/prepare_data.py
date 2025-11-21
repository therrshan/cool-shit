from datasets import load_dataset
import json
import os
from pathlib import Path

def prepare_alpaca_dataset(num_samples=10000, save_path="data/"):
    """
    Prepare Alpaca instruction dataset
    Great for general instruction following
    """
    print("Loading Alpaca dataset...")
    
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    # Take subset if specified
    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    print(f"✓ Loaded {len(dataset)} samples")
    
    # Format for training
    formatted_data = []
    for item in dataset:
        instruction = item['instruction']
        input_text = item['input']
        output = item['output']
        
        # Combine instruction and input
        if input_text:
            prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
        else:
            prompt = f"Instruction: {instruction}\nResponse:"
        
        formatted_data.append({
            'prompt': prompt,
            'response': output,
            'full_text': f"{prompt} {output}"
        })
    
    # Save to disk
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = save_dir / "alpaca_formatted.json"
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"✓ Saved formatted dataset to {output_file}")
    return formatted_data

def prepare_dolly_dataset(num_samples=10000, save_path="data/"):
    """
    Prepare Dolly dataset
    High-quality, diverse instruction dataset
    """
    print("Loading Dolly dataset...")
    
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    
    # Take subset if specified
    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    print(f"✓ Loaded {len(dataset)} samples")
    
    # Format for training
    formatted_data = []
    for item in dataset:
        instruction = item['instruction']
        context = item.get('context', '')
        response = item['response']
        
        # Combine instruction and context
        if context:
            prompt = f"Instruction: {instruction}\nContext: {context}\nResponse:"
        else:
            prompt = f"Instruction: {instruction}\nResponse:"
        
        formatted_data.append({
            'prompt': prompt,
            'response': response,
            'full_text': f"{prompt} {response}",
            'category': item.get('category', 'general')
        })
    
    # Save to disk
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = save_dir / "dolly_formatted.json"
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"✓ Saved formatted dataset to {output_file}")
    return formatted_data

def prepare_openorca_dataset(num_samples=5000, save_path="data/"):
    """
    Prepare OpenOrca dataset
    High-quality explanations and reasoning
    """
    print("Loading OpenOrca dataset...")
    
    # Use a subset of OpenOrca (full dataset is very large)
    dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    
    # Take first N samples
    formatted_data = []
    for i, item in enumerate(dataset):
        if i >= num_samples:
            break
        
        system_prompt = item.get('system_prompt', '')
        question = item.get('question', '')
        response = item.get('response', '')
        
        prompt = f"{system_prompt}\n{question}\nResponse:"
        
        formatted_data.append({
            'prompt': prompt,
            'response': response,
            'full_text': f"{prompt} {response}"
        })
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1} samples...")
    
    print(f"✓ Loaded {len(formatted_data)} samples")
    
    # Save to disk
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = save_dir / "openorca_formatted.json"
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"✓ Saved formatted dataset to {output_file}")
    return formatted_data

def create_train_val_split(data, val_ratio=0.1, save_path="data/"):
    """
    Split data into train and validation sets
    """
    import random
    
    print(f"\nCreating train/val split (val_ratio={val_ratio})...")
    
    # Shuffle data
    random.seed(42)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Split
    split_idx = int(len(shuffled_data) * (1 - val_ratio))
    train_data = shuffled_data[:split_idx]
    val_data = shuffled_data[split_idx:]
    
    print(f"✓ Train samples: {len(train_data)}")
    print(f"✓ Val samples: {len(val_data)}")
    
    # Save splits
    save_dir = Path(save_path)
    
    train_file = save_dir / "train.json"
    val_file = save_dir / "val.json"
    
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"✓ Saved train data to {train_file}")
    print(f"✓ Saved val data to {val_file}")
    
    return train_data, val_data

def main():
    """
    Main function to prepare datasets
    """
    print("\n" + "="*60)
    print("DATASET PREPARATION")
    print("="*60 + "\n")
    
    print("Using Dolly dataset (high quality, diverse instructions)")
    print("Preparing 10,000 samples...\n")
    
    # Use Dolly dataset - best balance of quality and size
    all_data = prepare_dolly_dataset(num_samples=10000)
    
    # Create train/val split
    train_data, val_data = create_train_val_split(all_data, val_ratio=0.1)
    
    print("\n" + "="*60)
    print("✓ DATASET PREPARATION COMPLETE!")
    print("="*60 + "\n")
    
    print("Dataset statistics:")
    print(f"  Total samples: {len(all_data)}")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"\nFiles saved in: ./data/")

if __name__ == "__main__":
    main()