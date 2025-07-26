# Tracking Representation Drift in Transformers via Optimal Transport

## Project Overview

This project studies how internal representations evolve across transformer layers and training epochs using Optimal Transport (OT). We track the "drift" of information through transformer models during training to detect potential emergent behaviors and phase transitions.

### What We're Doing

1. **Train on GSM8K**: Use GPT-2 style transformers on mathematical reasoning dataset
2. **Extract Activations**: Get hidden state representations from each layer at different training checkpoints
3. **Compute OT Distances**: Use Sinkhorn distances to measure representation changes across layers and time
4. **Visualize Drift**: Plot how representations compress, drift, or restructure during training
5. **Detect Emergence**: Look for phase transitions and sudden changes in representation structure

### Key Concepts

- **Optimal Transport**: Mathematical framework for measuring the "cost" of transforming one distribution into another
- **Representation Drift**: How internal representations change during training
- **Phase Transitions**: Sudden qualitative changes in model behavior during training
- **Sinkhorn Distance**: Regularized OT method for efficient computation

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Quick demo with pre-trained model
python example.py

# Full training experiment with GSM8K
python main.py

# Track training dynamics
python train_and_track.py
```

## Project Structure

- `main.py`: Main experiment runner with GSM8K training
- `train_and_track.py`: Training script with checkpoint tracking
- `transformer_utils.py`: Utilities for extracting activations
- `ot_analysis.py`: Optimal Transport computations (Sinkhorn)
- `visualization.py`: Advanced plotting and drift visualization
- `data_utils.py`: GSM8K dataset processing
- `example.py`: Simple demo with pre-trained models

## 🚀 Lambda.ai Deployment

### **Yes, you can run this on Lambda.ai!**

This project is fully compatible with Lambda.ai GPU instances. The setup uses:
- **PyTorch** with CUDA support
- **HuggingFace Transformers** for model loading
- **POT** (Python Optimal Transport) for OT computations
- **GSM8K dataset** (automatically downloaded via HuggingFace)

### **Lambda.ai Setup Steps:**

1. **Launch GPU Instance**: Choose a GPU instance (A10, A100, or H100)
2. **Install Dependencies**: 
   ```bash
   pip install -r requirements.txt
   ```
3. **Run Experiments**: 
   ```bash
   python main.py  # Quick test
   python train_and_track.py  # Full training
   ```

### **Recommended Lambda.ai Configurations:**

| Use Case | Instance Type | Memory | GPU | Estimated Cost |
|----------|---------------|---------|-----|----------------|
| **Quick Testing** | GPU 1x A10 | 24GB | 1x A10 | ~$0.60/hour |
| **Medium Research** | GPU 1x A100 | 80GB | 1x A100 | ~$2.40/hour |
| **Large Research** | GPU 2x A100 | 160GB | 2x A100 | ~$4.80/hour |

## 🔬 Serious Research Setup

### **Parameter Tuning Guide**

#### **Model Size Selection:**

```python
# Quick Testing (CPU/Laptop)
config = {
    'model_name': 'gpt2',  # 124M params
    'num_samples': 20,
    'batch_size': 2,
    'max_length': 128
}

# Medium Research (Single GPU)
config = {
    'model_name': 'gpt2-medium',  # 355M params
    'num_samples': 100,
    'batch_size': 4,
    'max_length': 256
}

# Serious Research (Multi-GPU)
config = {
    'model_name': 'gpt2-large',  # 774M params
    'num_samples': 500,
    'batch_size': 2,
    'max_length': 512
}

# Large Scale Research (High-end GPU)
config = {
    'model_name': 'gpt2-xl',  # 1.5B params
    'num_samples': 1000,
    'batch_size': 1,
    'max_length': 512
}
```

#### **Computational Requirements:**

| Model | Parameters | VRAM | Training Time | Use Case |
|-------|------------|------|---------------|----------|
| **GPT-2** | 124M | 2-4GB | 5-10 min | Quick testing |
| **GPT-2 Medium** | 355M | 4-8GB | 15-30 min | Medium research |
| **GPT-2 Large** | 774M | 8-16GB | 30-60 min | Serious research |
| **GPT-2 XL** | 1.5B | 16-32GB | 60-120 min | Large scale |

#### **Dataset Access:**

The GSM8K dataset is **automatically downloaded** when you run the code:

```python
# In data_utils.py - happens automatically
from datasets import load_dataset
dataset = load_dataset("gsm8k", "main")
```

**Dataset Details:**
- **Size**: ~8K grade school math problems
- **Format**: Question + step-by-step solution
- **Download**: ~50MB (cached locally)
- **Access**: No API key required

#### **Training Configuration Tuning:**

```python
# For Lambda.ai GPU instances
config = {
    'model_name': 'gpt2-large',  # Choose based on VRAM
    'num_epochs': 3,  # Increase for better results
    'batch_size': 1,  # Reduce if OOM, increase if VRAM available
    'learning_rate': 5e-5,  # Standard for fine-tuning
    'max_length': 512,  # Increase for longer sequences
    'num_samples': 200,  # Increase for better statistics
    'checkpoint_frequency': 20,  # More frequent = more data points
    'output_dir': 'results'
}
```

### **Performance Optimization:**

#### **Memory Management:**
```python
# Reduce memory usage
config = {
    'batch_size': 1,  # Smallest batch size
    'max_length': 256,  # Shorter sequences
    'pca_components': 20,  # Fewer PCA components
    'num_samples': 50  # Fewer samples
}
```

#### **Speed Optimization:**
```python
# Faster computation
config = {
    'ot_method': 'sinkhorn',  # Faster than EMD
    'pca_components': 30,  # Balance speed vs accuracy
    'checkpoint_frequency': 50  # Fewer checkpoints
}
```

### **Lambda.ai Specific Notes:**

#### **Instance Selection:**
- **A10 (24GB VRAM)**: Perfect for GPT-2 Medium research
- **A100 (80GB VRAM)**: Ideal for GPT-2 Large/XL research
- **H100 (80GB VRAM)**: Best for large-scale experiments

#### **Cost Optimization:**
- **Spot Instances**: 60-80% cost savings
- **Auto-shutdown**: Set up to stop when training completes
- **Data Persistence**: Use Lambda.ai's persistent storage

#### **Monitoring:**
```bash
# Monitor GPU usage
nvidia-smi

# Monitor memory usage
htop

# Check training progress
tail -f training_results/results.json
```

### **Advanced Research Extensions:**

#### **Multi-GPU Training:**
```python
# For 2x A100 setup
config = {
    'model_name': 'gpt2-xl',
    'num_samples': 2000,
    'batch_size': 4,  # Can increase with more GPUs
    'use_multi_gpu': True
}
```

#### **Long Training Runs:**
```python
# For extended research
config = {
    'num_epochs': 10,
    'num_samples': 1000,
    'checkpoint_frequency': 10,
    'save_final_model': True
}
```

#### **Hyperparameter Search:**
```python
# Grid search over parameters
learning_rates = [1e-5, 5e-5, 1e-4]
batch_sizes = [1, 2, 4]
model_sizes = ['gpt2-medium', 'gpt2-large']

for lr in learning_rates:
    for bs in batch_sizes:
        for model in model_sizes:
            config = {
                'model_name': model,
                'learning_rate': lr,
                'batch_size': bs,
                'output_dir': f'results_{model}_lr{lr}_bs{bs}'
            }
            # Run experiment
```

### **Expected Results:**

#### **Quick Test (5-10 minutes):**
- Basic OT distance patterns
- Layer-wise representation evolution
- Simple visualizations

#### **Medium Research (30-60 minutes):**
- Training dynamics tracking
- Phase transition detection
- Comprehensive drift analysis

#### **Serious Research (2-4 hours):**
- Multi-epoch training analysis
- Statistical significance testing
- Publication-ready visualizations

### **Troubleshooting:**

#### **Out of Memory (OOM):**
```python
# Reduce these parameters
config = {
    'batch_size': 1,  # Most important
    'max_length': 128,  # Reduce sequence length
    'num_samples': 20,  # Fewer samples
    'pca_components': 10  # Fewer components
}
```

#### **Slow Training:**
```python
# Speed up with these changes
config = {
    'ot_method': 'sinkhorn',  # Faster than EMD
    'checkpoint_frequency': 100,  # Fewer checkpoints
    'pca_components': 20  # Balance speed/accuracy
}
```

#### **Dataset Download Issues:**
```bash
# Manual dataset download
python -c "from datasets import load_dataset; load_dataset('gsm8k', 'main')"
```

This setup provides a complete research framework that scales from quick testing to serious research, with full Lambda.ai compatibility and comprehensive parameter tuning guidelines. 