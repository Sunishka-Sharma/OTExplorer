"""
Compare emergence patterns across different tasks.
"""

import torch
import numpy as np
from main import EmergenceExperiment
from visualization import OTVisualizer
import matplotlib.pyplot as plt

def compare_tasks():
    """Compare OT analysis across different tasks."""
    
    tasks = ['copy', 'parity', 'arithmetic']
    results = {}
    
    print("Comparing Emergence Patterns Across Tasks")
    print("=" * 50)
    
    for task in tasks:
        print(f"\nRunning experiment for task: {task}")
        print("-" * 30)
        
        config = {
            'model_name': 'gpt2',
            'task_type': task,
            'num_samples': 30,
            'max_length': 15,
            'batch_size': 4,
            'ot_method': 'emd',
            'normalize': True,
            'pca_components': 20,
            'transition_threshold': 2.0,
            'save_results': True,
            'save_plots': True,
            'output_dir': f'results_{task}'
        }
        
        experiment = EmergenceExperiment(config)
        experiment.run()
        results[task] = experiment.results
    
    # Create comparison plots
    create_comparison_plots(results)
    
    return results

def create_comparison_plots(results):
    """Create comparison plots across tasks."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: OT Distances comparison
    ax1 = axes[0, 0]
    for task, result in results.items():
        distances = result['ot_distances']
        ax1.plot(range(len(distances)), distances, 'o-', label=task, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Layer Pairs')
    ax1.set_ylabel('OT Distance')
    ax1.set_title('OT Distances Across Tasks')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Entropy comparison
    ax2 = axes[0, 1]
    for task, result in results.items():
        entropies = result['entropies']
        ax2.plot(range(len(entropies)), entropies, 's-', label=task, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Entropy Evolution Across Tasks')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Phase transitions
    ax3 = axes[1, 0]
    for task, result in results.items():
        transitions = result['phase_transitions']
        if transitions:
            for trans in transitions:
                ax3.scatter(trans, 0, s=100, marker='*', label=f'{task} transition' if trans == transitions[0] else "")
        else:
            ax3.scatter(-1, 0, s=100, marker='x', label=f'{task} (no transitions)')
    
    ax3.set_xlabel('Layer Pair')
    ax3.set_ylabel('Phase Transitions')
    ax3.set_title('Phase Transitions Across Tasks')
    ax3.legend()
    ax3.set_ylim(-0.5, 0.5)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    tasks = list(results.keys())
    mean_distances = [np.mean(result['ot_distances']) for result in results.values()]
    max_distances = [np.max(result['ot_distances']) for result in results.values()]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    ax4.bar(x - width/2, mean_distances, width, label='Mean OT Distance', alpha=0.7)
    ax4.bar(x + width/2, max_distances, width, label='Max OT Distance', alpha=0.7)
    
    ax4.set_xlabel('Task')
    ax4.set_ylabel('OT Distance')
    ax4.set_title('Distance Statistics Across Tasks')
    ax4.set_xticks(x)
    ax4.set_xticklabels(tasks)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('task_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n" + "=" * 50)
    print("TASK COMPARISON SUMMARY")
    print("=" * 50)
    
    for task, result in results.items():
        print(f"\n{task.upper()} TASK:")
        print(f"  Mean OT Distance: {np.mean(result['ot_distances']):.4f}")
        print(f"  Max OT Distance: {np.max(result['ot_distances']):.4f}")
        print(f"  Phase Transitions: {result['phase_transitions']}")
        print(f"  Entropy Range: {np.min(result['entropies']):.4f} - {np.max(result['entropies']):.4f}")

def analyze_specific_patterns(results):
    """Analyze specific patterns in the results."""
    
    print("\n" + "=" * 50)
    print("PATTERN ANALYSIS")
    print("=" * 50)
    
    for task, result in results.items():
        distances = result['ot_distances']
        entropies = result['entropies']
        
        # Find the layer with maximum OT distance
        max_dist_idx = np.argmax(distances)
        max_dist = distances[max_dist_idx]
        
        # Find the layer with minimum entropy
        min_entropy_idx = np.argmin(entropies)
        min_entropy = entropies[min_entropy_idx]
        
        print(f"\n{task.upper()} TASK PATTERNS:")
        print(f"  Peak transformation at layer pair {max_dist_idx} (distance: {max_dist:.4f})")
        print(f"  Maximum compression at layer {min_entropy_idx} (entropy: {min_entropy:.4f})")
        
        # Check if there's a correlation between distance and entropy changes
        if len(distances) > 1 and len(entropies) > 1:
            distance_changes = np.diff(distances)
            entropy_changes = np.diff(entropies)
            correlation = np.corrcoef(distance_changes, entropy_changes)[0, 1]
            print(f"  Distance-Entropy correlation: {correlation:.4f}")

if __name__ == "__main__":
    # Run task comparison
    results = compare_tasks()
    
    # Analyze patterns
    analyze_specific_patterns(results)
    
    print("\nTask comparison completed! Check 'task_comparison.png' for visualizations.") 