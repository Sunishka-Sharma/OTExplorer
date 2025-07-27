# Tracking Representation Drift in Transformers via Optimal Transport

## Project Overview

This project studies how internal representations evolve across transformer layers using Optimal Transport (OT). We analyze the "drift" of information through transformer models to detect potential emergent behaviors and phase transitions in pre-trained models.

### What We're Doing

1. **Static Analysis**: Analyze pre-trained GPT-2 models on GSM8K mathematical reasoning dataset
2. **Extract Activations**: Get hidden state representations from each layer
3. **Compute OT Distances**: Use Sinkhorn distances to measure representation changes across layers
4. **Visualize Drift**: Plot how representations compress, drift, or restructure across layers
5. **Detect Emergence**: Look for phase transitions and sudden changes in representation structure

### Key Concepts

- **Optimal Transport**: Mathematical framework for measuring the "cost" of transforming one distribution into another
- **Representation Drift**: How internal representations change across transformer layers
- **Phase Transitions**: Sudden qualitative changes in representation structure between layers
- **Sinkhorn Distance**: Regularized OT method for efficient computation

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### **Main Experiment (Static Analysis)**
```bash
# Run static representation drift analysis
python main.py
```

### **Quick Demo**
```bash
# Simple example with pre-trained model
python example.py
```

## Project Structure

### **Core Modules:**

- **`main.py`**: Main experiment runner for static representation drift analysis
- **`example.py`**: Simple demo with pre-trained models
- **`transformer_utils.py`**: Utilities for extracting activations from transformer layers
- **`ot_analysis.py`**: Optimal Transport computations (Sinkhorn algorithm)
- **`visualization.py`**: Advanced plotting and drift visualization
- **`data_utils.py`**: GSM8K dataset processing and preparation

### **Module Explanations:**

#### **`main.py` - Static Analysis Runner**
- **Purpose**: Analyze representation drift in pre-trained models
- **What it does**: 
  - Loads pre-trained GPT-2 model
  - Extracts activations from all layers
  - Computes OT distances between adjacent layers
  - Detects phase transitions
  - Generates visualizations
- **Output**: `results/` directory with analysis results and plots

#### **`transformer_utils.py` - Activation Extraction**
- **Purpose**: Extract hidden state activations from transformer layers
- **Key Features**:
  - Hook-based activation capture
  - Batch processing for efficiency
  - Memory management for large models
  - Support for different model architectures

#### **`ot_analysis.py` - Optimal Transport Analysis**
- **Purpose**: Compute OT distances and analyze representation flow
- **Algorithms**:
  - Sinkhorn algorithm (fast, regularized)
  - EMD (Earth Mover's Distance)
  - Phase transition detection
  - Entropy analysis

#### **`visualization.py` - Results Visualization**
- **Purpose**: Create publication-ready visualizations
- **Plots**:
  - OT distance heatmaps
  - Layer-wise drift plots
  - Phase transition detection
  - Entropy evolution

#### **`data_utils.py` - Dataset Management**
- **Purpose**: Load and prepare GSM8K dataset
- **Features**:
  - Automatic dataset download
  - Text preprocessing
  - Tokenization
  - Batch creation

## 🔬 Current Research Configuration

### **Production Configuration (Used in Results):**
```python
config = {
    'model_name': 'gpt2-large',  # 774M parameters
    'num_samples': 2000,         # Large dataset for statistical significance
    'max_length': 256,           # Balanced sequence length
    'batch_size': 4,             # Optimized for GPU memory
    'ot_method': 'sinkhorn',     # Fast, regularized OT
    'normalize': True,           # Normalize activations
    'pca_components': 50,        # Dimensionality reduction
    'transition_threshold': 2.0, # Phase transition sensitivity
    'save_results': True,
    'save_plots': True,
    'output_dir': 'results'
}
```

### **Memory Management:**
- **GPU VRAM**: 23GB+ recommended for GPT-2 Large
- **System RAM**: 64GB+ for large datasets
- **Batch Processing**: Automatic memory optimization
- **Gradient Checkpointing**: Enabled for large models

## 🚀 Lambda.ai Deployment

### **Recommended Instance Configuration:**
- **Instance**: GPU 1x A100 (80GB VRAM)
- **Memory**: 160GB+ system RAM
- **Storage**: 500GB+ for results and models

### **Setup Commands:**
```bash
# Clone and setup
git clone https://github.com/Sunishka-Sharma/OTExplorer.git
cd OTExplorer
git checkout main

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run experiment
python main.py
```

## 📊 Expected Results

### **Analysis Output:**
- **OT Distances**: Layer-wise transport costs
- **Phase Transitions**: Detected at specific layers
- **Entropy Evolution**: Information compression patterns
- **Visualizations**: Publication-ready plots

### **Performance Metrics:**
- **Model**: GPT-2 Large (774M parameters)
- **Dataset**: 2000 GSM8K samples
- **Layers**: 36 transformer layers analyzed
- **Processing Time**: 30-60 minutes on A100

## 🔧 Troubleshooting

### **Memory Issues:**
```python
# Reduce memory usage
config = {
    'model_name': 'gpt2-medium',  # Smaller model
    'num_samples': 500,           # Fewer samples
    'batch_size': 1,              # Smaller batches
    'max_length': 128,            # Shorter sequences
}
```

### **Performance Optimization:**
```python
# Faster computation
config = {
    'ot_method': 'sinkhorn',      # Faster than EMD
    'pca_components': 30,         # Fewer components
    'normalize': True,            # Better numerical stability
}
```

## 📈 Research Significance

This project provides:
- **Novel Analysis Method**: OT-based representation drift detection
- **Emergence Detection**: Identification of phase transitions in transformers
- **Scalable Framework**: Applicable to various model architectures
- **Mathematical Rigor**: Formal geometric analysis of neural representations

## 🔗 Related Work

- **Optimal Transport**: Sinkhorn algorithm for efficient computation
- **Representation Learning**: Analysis of learned features
- **Emergence in AI**: Detection of emergent behaviors
- **Geometric Deep Learning**: Geometric analysis of neural networks 
