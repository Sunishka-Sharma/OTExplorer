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