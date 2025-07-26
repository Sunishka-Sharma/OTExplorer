"""
Simple example demonstrating the OT analysis pipeline.
"""

import torch
import numpy as np
from transformer_utils import ActivationExtractor
from ot_analysis import OTAnalyzer
from visualization import OTVisualizer

def simple_example():
    """Run a simple example with a few sample texts."""
    
    print("Simple OT Analysis Example")
    print("=" * 40)
    
    # Sample texts
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Optimal transport measures the cost of moving mass",
        "Transformers have revolutionized natural language processing",
        "Emergence occurs when complex behaviors arise from simple rules"
    ]
    
    print(f"Using {len(sample_texts)} sample texts")
    
    # Initialize components
    print("Loading GPT-2 model...")
    extractor = ActivationExtractor(model_name="gpt2", device="cpu")
    
    print("Extracting activations...")
    activations = extractor.extract_activations_batch(sample_texts)
    
    print("Preparing representations...")
    # Get all layer representations
    all_reps = extractor.get_all_layer_representations(activations)
    
    # Convert to numpy and prepare for analysis
    layer_representations = []
    for i, reps in enumerate(all_reps):
        reps_np = reps.numpy()
        batch_size, seq_len, hidden_dim = reps_np.shape
        
        # Reshape to combine batch and sequence dimensions
        reps_reshaped = reps_np.reshape(-1, hidden_dim)
        
        # Remove padding tokens (simple heuristic)
        non_padding_mask = np.any(reps_reshaped != 0, axis=1)
        reps_clean = reps_reshaped[non_padding_mask]
        
        layer_representations.append(reps_clean)
        print(f"Layer {i}: {reps_clean.shape}")
    
    # Run OT analysis
    print("Running OT analysis...")
    ot_analyzer = OTAnalyzer(method="emd", normalize=True)
    analysis_results = ot_analyzer.analyze_representation_flow(layer_representations)
    
    # Create visualizations
    print("Creating visualizations...")
    visualizer = OTVisualizer()
    
    # Plot OT distances
    visualizer.plot_ot_distances(analysis_results['ot_distances'])
    
    # Plot phase transitions
    visualizer.plot_phase_transitions(
        analysis_results['ot_distances'], 
        analysis_results['phase_transitions']
    )
    
    # Plot entropy evolution
    visualizer.plot_entropy_evolution(analysis_results['entropies'])
    
    # Print summary
    print("\nResults Summary:")
    print(f"Number of layers: {analysis_results['num_layers']}")
    print(f"OT distances: {[f'{d:.4f}' for d in analysis_results['ot_distances']]}")
    print(f"Phase transitions: {analysis_results['phase_transitions']}")
    print(f"Entropies: {[f'{e:.4f}' for e in analysis_results['entropies']]}")
    
    # Cleanup
    extractor.cleanup()
    
    print("\nExample completed!")

def compare_methods_example():
    """Compare different OT methods."""
    
    print("\nComparing OT Methods")
    print("=" * 40)
    
    # Sample texts
    sample_texts = [
        "Hello world",
        "Python programming",
        "Data science",
        "Artificial intelligence"
    ]
    
    # Initialize components
    extractor = ActivationExtractor(model_name="gpt2", device="cpu")
    activations = extractor.extract_activations_batch(sample_texts)
    
    # Prepare representations
    all_reps = extractor.get_all_layer_representations(activations)
    layer_representations = []
    
    for reps in all_reps:
        reps_np = reps.numpy()
        batch_size, seq_len, hidden_dim = reps_np.shape
        reps_reshaped = reps_np.reshape(-1, hidden_dim)
        non_padding_mask = np.any(reps_reshaped != 0, axis=1)
        reps_clean = reps_reshaped[non_padding_mask]
        layer_representations.append(reps_clean)
    
    # Compare methods
    methods = ["emd", "sinkhorn"]
    results = {}
    
    for method in methods:
        print(f"Testing {method} method...")
        ot_analyzer = OTAnalyzer(method=method, normalize=True)
        distances = ot_analyzer.compute_layer_distances(layer_representations)
        results[method] = distances
    
    # Print comparison
    print("\nMethod Comparison:")
    for i in range(len(results["emd"])):
        print(f"L{i}-L{i+1}: EMD={results['emd'][i]:.4f}, Sinkhorn={results['sinkhorn'][i]:.4f}")
    
    # Cleanup
    extractor.cleanup()

if __name__ == "__main__":
    # Run simple example
    simple_example()
    
    # Run method comparison
    compare_methods_example() 