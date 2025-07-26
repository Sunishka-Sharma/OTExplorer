# Layer by Layer: Detecting Emergence via Optimal Transport

## Project Overview

This project explores how internal representations evolve across transformer layers using Optimal Transport (OT). We track the "flow" of information through a small transformer model to detect potential emergent behaviors or phase transitions.

### Key Questions
- How do representations change as information flows through transformer layers?
- Can we detect sudden qualitative changes (emergence) in representation space?
- What insights can Optimal Transport provide about the learning dynamics?

## Methodology

### 1. **Optimal Transport Primer**
Optimal Transport measures the "cost" of transforming one distribution into another. Think of it as measuring how much work it takes to move sand from one pile to another, considering the shape and distribution of both piles.

**Why OT for Emergence Detection?**
- Captures the full distributional structure, not just averages
- Provides a principled way to measure "distance" between representations
- Can detect subtle changes in representation geometry

### 2. **Experimental Setup**
- **Model**: GPT-2 (12 layers, 768 hidden dimensions)
- **Tasks**: Copy task, parity task, arithmetic task
- **Data**: Synthetic sequences of varying complexity
- **Analysis**: Compare representations between adjacent layers

### 3. **Pipeline**
```
Input Text → Tokenization → Layer-wise Activations → OT Analysis → Visualization
```

## Key Findings

### 1. **Representation Evolution Pattern**
From our experiment with the copy task:
- **Early layers (0-2)**: Large OT distances (0.7-0.3) - rapid transformation
- **Middle layers (3-9)**: Small, stable distances (0.09-0.17) - refinement
- **Final layer (10-11)**: Large spike (1.28) - output preparation

### 2. **Entropy Evolution**
- **High entropy in early layers**: Raw, unprocessed information
- **Decreasing entropy in middle layers**: Information compression and organization
- **Increasing entropy in final layers**: Preparation for output generation

### 3. **No Phase Transitions Detected**
- Smooth, gradual evolution rather than sudden jumps
- Suggests the model learns incrementally rather than through discrete phases

## Technical Implementation

### Core Components

1. **ActivationExtractor** (`transformer_utils.py`)
   - Loads GPT-2 model
   - Extracts hidden states from each layer
   - Handles batching and padding

2. **OTAnalyzer** (`ot_analysis.py`)
   - Computes Earth Mover's Distance between layer representations
   - Detects phase transitions using statistical outlier detection
   - Calculates representation statistics (entropy, sparsity, etc.)

3. **OTVisualizer** (`visualization.py`)
   - Creates comprehensive visualizations
   - Plots OT distances, phase transitions, entropy evolution
   - Generates heatmaps and comparison charts

4. **EmergenceExperiment** (`main.py`)
   - Orchestrates the complete pipeline
   - Handles configuration and results saving
   - Provides experiment summary

### Key Functions

```python
# Extract activations from all layers
activations = extractor.extract_activations_batch(input_texts)

# Compute OT distances between adjacent layers
distances = ot_analyzer.compute_layer_distances(layer_representations)

# Detect phase transitions
transitions = ot_analyzer.detect_phase_transitions(distances)

# Create visualizations
visualizer.create_comprehensive_dashboard(analysis_results)
```

## Results Interpretation

### 1. **What the OT Distances Tell Us**
- **Large distances**: Significant transformation of representations
- **Small distances**: Fine-tuning and refinement
- **Spikes**: Potential emergence points or output preparation

### 2. **Entropy as Information Measure**
- **High entropy**: Diverse, unorganized representations
- **Low entropy**: Compressed, organized representations
- **Pattern**: U-shaped curve suggests information compression then expansion

### 3. **Phase Transition Detection**
- Uses statistical outlier detection on distance differences
- Threshold-based approach (configurable)
- Currently no transitions detected in our experiments

## Extensions and Future Work

### 1. **Alternative Tasks**
- **Parity task**: Test mathematical reasoning emergence
- **Arithmetic task**: Test numerical computation emergence
- **Language tasks**: Test linguistic structure emergence

### 2. **Advanced Analysis**
- **Information Bottleneck**: Compare with IB theory
- **Singular Learning Theory**: Analyze loss landscape geometry
- **Attention analysis**: Combine with attention weight analysis

### 3. **Training Dynamics**
- **During training**: Track emergence over training epochs
- **Different architectures**: Compare with other transformer variants
- **Scaling laws**: Study emergence as model size increases

### 4. **Visualization Enhancements**
- **3D trajectory plots**: Visualize representation flow in 3D
- **Interactive dashboards**: Real-time exploration of results
- **Comparative analysis**: Side-by-side task comparisons

## Usage Guide

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run simple example
python example.py

# Run full experiment
python main.py
```

### Configuration
```python
config = {
    'model_name': 'gpt2',
    'task_type': 'copy',  # 'copy', 'parity', 'arithmetic'
    'num_samples': 100,
    'ot_method': 'emd',  # 'emd' or 'sinkhorn'
    'transition_threshold': 2.0,
    'save_plots': True
}
```

### Output Files
- `results/analysis_results.json`: Complete analysis results
- `results/config.json`: Experiment configuration
- `results/plots/`: All visualization plots

## Key Insights

1. **Gradual Learning**: Transformers learn incrementally rather than through discrete phases
2. **Information Compression**: Middle layers compress and organize information
3. **Output Preparation**: Final layers prepare representations for generation
4. **Task Dependence**: Different tasks may show different emergence patterns

## Conclusion

This project provides a framework for studying emergence in transformer models using Optimal Transport. While we didn't detect dramatic phase transitions in our initial experiments, the methodology reveals interesting patterns in representation evolution and provides a foundation for more sophisticated emergence detection.

The key contribution is demonstrating how OT can be used to track representation flow and potentially detect emergent behaviors in neural networks. Future work could explore more complex tasks, larger models, and training dynamics to better understand when and how emergence occurs. 