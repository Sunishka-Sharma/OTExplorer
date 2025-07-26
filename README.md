# Layer by Layer: Detecting Emergence via Optimal Transport

## Project Overview

This project explores how internal representations evolve across transformer layers using Optimal Transport (OT). We track the "flow" of information through a small transformer model to detect potential emergent behaviors or phase transitions.

### What We're Doing

1. **Extract Activations**: Get hidden state representations from each layer of a transformer
2. **Compute OT Distances**: Measure how representations change between adjacent layers
3. **Visualize Evolution**: Plot how these distances evolve across layers
4. **Detect Emergence**: Look for sudden changes that might indicate emergent behavior

### Key Concepts

- **Optimal Transport**: A mathematical framework for measuring the "cost" of transforming one distribution into another
- **Emergence**: Sudden qualitative changes in model behavior that aren't predictable from individual components
- **Phase Transitions**: Sharp changes in representation flow that might signal learning shifts

## Setup

```bash
pip install torch transformers python-ot matplotlib seaborn numpy pandas
```

## Usage

```bash
python main.py
```

## Project Structure

- `main.py`: Main experiment runner
- `transformer_utils.py`: Utilities for extracting activations
- `ot_analysis.py`: Optimal Transport computations
- `visualization.py`: Plotting and visualization functions
- `data_utils.py`: Dataset and data processing utilities 