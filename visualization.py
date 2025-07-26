"""
Visualization functions for OT analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class OTVisualizer:
    """Visualize Optimal Transport analysis results."""
    
    def __init__(self, figsize: tuple = (12, 8)):
        self.figsize = figsize
    
    def plot_ot_distances(self, distances: List[float], 
                         title: str = "OT Distances Between Adjacent Layers",
                         save_path: Optional[str] = None):
        """
        Plot OT distances between adjacent layers.
        
        Args:
            distances: List of OT distances
            title: Plot title
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create layer pairs for x-axis labels
        layer_pairs = [f"L{i}-L{i+1}" for i in range(len(distances))]
        
        # Plot distances
        bars = ax.bar(range(len(distances)), distances, alpha=0.7, color='skyblue')
        
        # Add value labels on bars
        for i, (bar, dist) in enumerate(zip(bars, distances)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{dist:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Layer Pairs')
        ax.set_ylabel('OT Distance')
        ax.set_title(title)
        ax.set_xticks(range(len(distances)))
        ax.set_xticklabels(layer_pairs, rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_phase_transitions(self, distances: List[float], 
                             transitions: List[int],
                             title: str = "OT Distances with Phase Transitions",
                             save_path: Optional[str] = None):
        """
        Plot OT distances with highlighted phase transitions.
        
        Args:
            distances: List of OT distances
            transitions: List of transition indices
            title: Plot title
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot distances
        layer_pairs = [f"L{i}-L{i+1}" for i in range(len(distances))]
        ax.plot(range(len(distances)), distances, 'o-', linewidth=2, markersize=8, 
               color='blue', label='OT Distance')
        
        # Highlight phase transitions
        if transitions:
            transition_distances = [distances[i] for i in transitions]
            ax.scatter(transitions, transition_distances, s=100, c='red', 
                      marker='*', zorder=5, label='Phase Transitions')
            
            # Add annotations
            for i, trans_idx in enumerate(transitions):
                ax.annotate(f'Transition\nat L{trans_idx}', 
                           xy=(trans_idx, transition_distances[i]),
                           xytext=(trans_idx + 0.5, transition_distances[i] + 0.1),
                           arrowprops=dict(arrowstyle='->', color='red'),
                           fontsize=10, ha='center')
        
        ax.set_xlabel('Layer Pairs')
        ax.set_ylabel('OT Distance')
        ax.set_title(title)
        ax.set_xticks(range(len(distances)))
        ax.set_xticklabels(layer_pairs, rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_heatmap(self, distances: List[float], 
                    title: str = "OT Distance Heatmap",
                    save_path: Optional[str] = None):
        """
        Create a heatmap of OT distances.
        
        Args:
            distances: List of OT distances
            title: Plot title
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a matrix for the heatmap
        n_layers = len(distances) + 1
        heatmap_matrix = np.zeros((n_layers, n_layers))
        
        # Fill the matrix with distances
        for i, dist in enumerate(distances):
            heatmap_matrix[i, i+1] = dist
            heatmap_matrix[i+1, i] = dist  # Make it symmetric
        
        # Create heatmap
        sns.heatmap(heatmap_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=[f'L{i}' for i in range(n_layers)],
                   yticklabels=[f'L{i}' for i in range(n_layers)],
                   ax=ax, cbar_kws={'label': 'OT Distance'})
        
        ax.set_title(title)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Layer')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_entropy_evolution(self, entropies: List[float],
                             title: str = "Representation Entropy Across Layers",
                             save_path: Optional[str] = None):
        """
        Plot entropy evolution across layers.
        
        Args:
            entropies: List of entropy values for each layer
            title: Plot title
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        layers = [f'L{i}' for i in range(len(entropies))]
        
        ax.plot(range(len(entropies)), entropies, 'o-', linewidth=2, markersize=8,
               color='green', label='Entropy')
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Entropy')
        ax.set_title(title)
        ax.set_xticks(range(len(entropies)))
        ax.set_xticklabels(layers)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_statistics_comparison(self, statistics: Dict,
                                 title: str = "Layer Statistics Comparison",
                                 save_path: Optional[str] = None):
        """
        Plot comparison of different statistics across layers.
        
        Args:
            statistics: Dictionary with statistics for each layer
            title: Plot title
            save_path: Path to save the plot
        """
        # Extract data
        layers = []
        mean_norms = []
        std_norms = []
        sparsities = []
        
        for layer_name, stats in statistics.items():
            layers.append(layer_name)
            mean_norms.append(stats['mean_norm'])
            std_norms.append(stats['std_norm'])
            sparsities.append(stats['sparsity'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot mean norms
        axes[0, 0].plot(range(len(mean_norms)), mean_norms, 'o-', color='blue')
        axes[0, 0].set_title('Mean Norm')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Mean Norm')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot std norms
        axes[0, 1].plot(range(len(std_norms)), std_norms, 'o-', color='red')
        axes[0, 1].set_title('Std Norm')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Std Norm')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot sparsity
        axes[1, 0].plot(range(len(sparsities)), sparsities, 'o-', color='green')
        axes[1, 0].set_title('Sparsity')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Sparsity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot all together
        axes[1, 1].plot(range(len(mean_norms)), mean_norms, 'o-', label='Mean Norm', color='blue')
        axes[1, 1].plot(range(len(std_norms)), std_norms, 's-', label='Std Norm', color='red')
        axes[1, 1].plot(range(len(sparsities)), sparsities, '^-', label='Sparsity', color='green')
        axes[1, 1].set_title('All Statistics')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_distance_comparison(self, comparison_results: Dict,
                               title: str = "OT vs Simple Distance Metrics",
                               save_path: Optional[str] = None):
        """
        Compare OT distances with simple distance metrics.
        
        Args:
            comparison_results: Dictionary with OT and simple distances
            title: Plot title
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ot_distances = comparison_results['ot_distances']
        simple_distances = comparison_results['simple_distances']
        
        x = range(len(ot_distances))
        
        # Plot OT distances
        ax.plot(x, ot_distances, 'o-', linewidth=2, markersize=8, 
               color='blue', label='OT Distance')
        
        # Plot simple distances
        colors = ['red', 'green', 'orange']
        for i, (metric, distances) in enumerate(simple_distances.items()):
            ax.plot(x, distances, 's-', linewidth=2, markersize=6,
                   color=colors[i], label=f'{metric.capitalize()} Distance')
        
        ax.set_xlabel('Layer Pairs')
        ax.set_ylabel('Distance')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{i}-L{i+1}" for i in range(len(ot_distances))], rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_comprehensive_dashboard(self, analysis_results: Dict,
                                     save_dir: str = "plots"):
        """
        Create a comprehensive dashboard with all visualizations.
        
        Args:
            analysis_results: Dictionary with all analysis results
            save_dir: Directory to save plots
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Extract results
        distances = analysis_results['ot_distances']
        transitions = analysis_results['phase_transitions']
        statistics = analysis_results['statistics']
        entropies = analysis_results['entropies']
        
        # Create all plots
        self.plot_ot_distances(distances, save_path=f"{save_dir}/ot_distances.png")
        self.plot_phase_transitions(distances, transitions, save_path=f"{save_dir}/phase_transitions.png")
        self.plot_heatmap(distances, save_path=f"{save_dir}/heatmap.png")
        self.plot_entropy_evolution(entropies, save_path=f"{save_dir}/entropy.png")
        self.plot_statistics_comparison(statistics, save_path=f"{save_dir}/statistics.png")
        
        print(f"All plots saved to {save_dir}/")

def plot_sample_activations(activations: Dict[str, torch.Tensor], 
                          sample_idx: int = 0,
                          title: str = "Sample Activations Across Layers"):
    """
    Plot activations for a single sample across layers.
    
    Args:
        activations: Dictionary of layer activations
        sample_idx: Index of sample to plot
        title: Plot title
    """
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    layer_names = sorted(activations.keys())
    
    for i, layer_name in enumerate(layer_names):
        if i >= len(axes):
            break
            
        layer_acts = activations[layer_name][sample_idx]  # Shape: (seq_len, hidden_dim)
        
        # Plot heatmap of activations
        im = axes[i].imshow(layer_acts.T, aspect='auto', cmap='RdBu_r')
        axes[i].set_title(f'{layer_name}')
        axes[i].set_xlabel('Token Position')
        axes[i].set_ylabel('Hidden Dimension')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i])
    
    # Hide unused subplots
    for i in range(len(layer_names), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show() 