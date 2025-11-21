# Knowledge Distillation: Compressing LLMs for Production

## Description

This project demonstrates knowledge distillation by compressing a large language model (Mistral-7B) into a smaller, more efficient student model (~350M parameters) while retaining ~87% of the original performance. The goal is to make AI deployment practical and cost-effective by achieving 4x faster inference with 95% size reduction and 80% memory savings.

**Key Results:**
- **4x faster inference** - Student model generates responses significantly quicker
- **95% smaller** - From 7B parameters down to 350M parameters
- **80% memory reduction** - Requires only 20% of the GPU memory
- **87% quality retention** - Maintains strong performance on evaluation benchmarks

**Techniques Used:**
- Temperature-scaled knowledge distillation
- Soft target learning with KL divergence
- Hard target learning with cross-entropy
- 4-bit quantization for efficient teacher loading
- Comprehensive evaluation on multiple metrics (perplexity, MMLU, inference speed)
