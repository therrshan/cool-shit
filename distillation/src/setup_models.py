import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

def setup_teacher_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """
    Load teacher model with 4-bit quantization to fit in 8GB GPU
    """
    print(f"Loading teacher model: {model_name}")
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Teacher model loaded successfully!")
    print(f"Model size: ~{sum(p.numel() for p in teacher_model.parameters()) / 1e9:.2f}B parameters")
    
    return teacher_model, tokenizer

def setup_student_model(vocab_size=32000, hidden_size=768, num_layers=12):
    """
    Create a smaller student model from scratch
    Target: ~350M parameters (vs 7B teacher)
    """
    from transformers import GPT2Config, GPT2LMHeadModel
    
    print(f"Creating student model...")
    
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=2048,  # Max sequence length
        n_embd=hidden_size,  # Hidden size
        n_layer=num_layers,  # Number of transformer layers
        n_head=12,  # Attention heads
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    
    student_model = GPT2LMHeadModel(config)
    
    print(f"Student model created!")
    print(f"Model size: ~{sum(p.numel() for p in student_model.parameters()) / 1e6:.2f}M parameters")
    
    return student_model, config

def test_models():
    """
    Quick test to ensure models load correctly
    """
    print("\n" + "="*60)
    print("TESTING MODEL SETUP")
    print("="*60 + "\n")
    
    # Test teacher
    teacher, tokenizer = setup_teacher_model()
    
    # Test student
    student, config = setup_student_model(vocab_size=len(tokenizer))
    
    # Quick inference test
    print("\n" + "="*60)
    print("TESTING INFERENCE")
    print("="*60 + "\n")
    
    test_input = "Hello, how are you?"
    inputs = tokenizer(test_input, return_tensors="pt").to(teacher.device)
    
    with torch.no_grad():
        teacher_outputs = teacher(**inputs)
        print(f"✓ Teacher inference successful! Output shape: {teacher_outputs.logits.shape}")
    
    # Move student to GPU
    student = student.to("cuda")
    inputs_student = tokenizer(test_input, return_tensors="pt").to(student.device)
    
    with torch.no_grad():
        student_outputs = student(**inputs_student)
        print(f"✓ Student inference successful! Output shape: {student_outputs.logits.shape}")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60 + "\n")
    
    return teacher, student, tokenizer

if __name__ == "__main__":
    test_models()