# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run a Simple Example
```bash
python example.py
```
This will:
- Load GPT-2 model
- Extract activations from 5 sample texts
- Compute OT distances between layers
- Create visualizations
- Show results summary

### 3. Run Full Experiment
```bash
python main.py
```
This will:
- Generate 50 synthetic samples
- Run comprehensive OT analysis
- Create all visualizations
- Save results to `results/` directory

### 4. Compare Different Tasks
```bash
python task_comparison.py
```
This will:
- Run experiments on copy, parity, and arithmetic tasks
- Compare emergence patterns across tasks
- Generate comparison plots

## 📊 Understanding the Results

### Key Metrics
- **OT Distance**: Measures how much representations change between layers
- **Entropy**: Measures information compression/organization
- **Phase Transitions**: Sudden changes that might indicate emergence

### What to Look For
1. **Large OT distances**: Significant representation changes
2. **Entropy patterns**: U-shaped curve suggests compression then expansion
3. **Spikes**: Potential emergence points
4. **Task differences**: Different tasks may show different patterns

## 🔧 Customization

### Change Task Type
```python
config = {
    'task_type': 'parity',  # 'copy', 'parity', 'arithmetic'
    'num_samples': 100,
    # ... other settings
}
```

### Change Model
```python
config = {
    'model_name': 'gpt2-medium',  # or other HuggingFace models
    # ... other settings
}
```

### Adjust Analysis Parameters
```python
config = {
    'ot_method': 'sinkhorn',  # 'emd' or 'sinkhorn'
    'transition_threshold': 1.5,  # Lower = more sensitive
    'pca_components': 30,  # Dimensionality reduction
    # ... other settings
}
```

## 📁 Output Files

After running experiments, you'll find:

```
results/
├── analysis_results.json    # Complete analysis data
├── config.json             # Experiment configuration
└── plots/
    ├── ot_distances.png     # OT distance visualization
    ├── phase_transitions.png # Phase transition detection
    ├── entropy.png          # Entropy evolution
    ├── heatmap.png          # Distance heatmap
    ├── statistics.png       # Layer statistics
    └── distance_comparison.png # OT vs simple distances
```

## 🎯 Key Insights from Our Experiments

### Copy Task Results
- **Early layers (0-2)**: Large distances (0.7-0.3) - rapid transformation
- **Middle layers (3-9)**: Small distances (0.09-0.17) - refinement
- **Final layer (10-11)**: Large spike (1.28) - output preparation

### No Phase Transitions Detected
- Smooth, gradual evolution rather than sudden jumps
- Suggests incremental learning rather than discrete phases

### Entropy Pattern
- High in early layers (raw information)
- Low in middle layers (compression)
- High in final layers (output preparation)

## 🔬 Next Steps

### For Beginners
1. Try different sample texts in `example.py`
2. Experiment with different task types
3. Adjust the transition threshold to see more/fewer transitions

### For Advanced Users
1. Implement custom tasks in `data_utils.py`
2. Add new analysis metrics in `ot_analysis.py`
3. Create custom visualizations in `visualization.py`
4. Compare with other models (BERT, T5, etc.)

### Research Extensions
1. **Training Dynamics**: Track emergence during training
2. **Scaling Laws**: Study emergence as model size increases
3. **Information Bottleneck**: Compare with IB theory
4. **Attention Analysis**: Combine with attention weight analysis

## 🐛 Troubleshooting

### Common Issues

**"Model not found"**
- Check internet connection for model download
- Try different model names

**"Memory error"**
- Reduce `num_samples` or `batch_size`
- Use smaller model (e.g., `gpt2` instead of `gpt2-medium`)

**"No phase transitions detected"**
- Lower the `transition_threshold`
- Try different tasks or more samples
- This is normal - emergence may be gradual

**"Slow performance"**
- Use GPU if available (set `device='cuda'`)
- Reduce `num_samples` or `pca_components`
- Use `ot_method='sinkhorn'` for faster computation

## 📚 Learn More

- **Optimal Transport**: [POT Documentation](https://pythonot.github.io/)
- **Transformers**: [HuggingFace Documentation](https://huggingface.co/docs)
- **Emergence in AI**: Research papers on phase transitions in neural networks
- **Information Theory**: Information Bottleneck and related concepts

## 🤝 Contributing

Feel free to:
- Add new analysis methods
- Implement additional tasks
- Improve visualizations
- Extend to other model architectures
- Add training dynamics analysis

Happy exploring! 🚀 