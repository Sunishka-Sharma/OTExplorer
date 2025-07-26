"""
Compare geometric transformation paths between trained and untrained models.
"""

import torch
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from data_utils import GSM8KDataset
from transformer_utils import ModelComparison
from visualization import OTVisualizer

class TrainedVsUntrainedExperiment:
    """Compare trained vs untrained models for geometric transformation analysis."""
    
    def __init__(self, config: dict = None):
        self.config = config or self._get_default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model_comparison = None
        self.visualizer = None
        self.results = {}
        
    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'model_name': 'gpt2-medium',  # Use medium for reasonable comparison
            'num_samples': 30,  # Number of GSM8K samples for comparison
            'max_length': 256,  # Shorter sequences for efficiency
            'output_dir': 'trained_vs_untrained_results',
            'save_plots': True,
            'save_results': True
        }
    
    def setup(self):
        """Setup the experiment components."""
        print("Setting up trained vs untrained comparison...")
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Initialize model comparison
        self.model_comparison = ModelComparison(
            model_name=self.config['model_name'],
            device=self.device
        )
        
        # Setup models
        self.model_comparison.setup_models()
        
        # Initialize visualizer
        self.visualizer = OTVisualizer()
        
        print("Setup complete!")
    
    def load_comparison_data(self) -> list:
        """Load GSM8K data for comparison."""
        print("Loading GSM8K data for comparison...")
        
        dataset = GSM8KDataset(
            model_name=self.config['model_name'],
            max_length=self.config['max_length']
        )
        
        # Get sample data
        data = dataset.get_sample_data(self.config['num_samples'])
        
        # Extract input texts for activation extraction
        input_texts = [item['input_text'] for item in data]
        
        print(f"Loaded {len(input_texts)} GSM8K samples for comparison")
        return input_texts
    
    def run_comparison(self, input_texts: list) -> dict:
        """Run the trained vs untrained comparison."""
        print("Running trained vs untrained comparison...")
        
        # Extract activations from both models
        activations = self.model_comparison.extract_comparison_activations(input_texts)
        
        # Compare geometric transformations
        comparison_results = self.model_comparison.compare_geometric_transformations(activations)
        
        return comparison_results
    
    def create_comparison_visualizations(self, comparison_results: dict):
        """Create visualizations comparing trained vs untrained models."""
        print("Creating comparison visualizations...")
        
        # Extract data
        trained_distances = comparison_results['trained']['ot_distances']
        untrained_distances = comparison_results['untrained']['ot_distances']
        differences = comparison_results['differences']
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: OT Distances Comparison
        ax1 = axes[0, 0]
        layer_pairs = [f"L{i}-L{i+1}" for i in range(len(trained_distances))]
        x = range(len(trained_distances))
        
        ax1.plot(x, trained_distances, 'o-', linewidth=2, markersize=8, 
                color='blue', label='Trained Model')
        ax1.plot(x, untrained_distances, 's-', linewidth=2, markersize=8,
                color='red', label='Untrained Model')
        
        ax1.set_xlabel('Layer Pairs')
        ax1.set_ylabel('OT Distance')
        ax1.set_title('OT Distances: Trained vs Untrained')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layer_pairs, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Difference Analysis
        ax2 = axes[0, 1]
        differences_array = np.array(differences['ot_distances']['difference'])
        relative_differences = np.array(differences['ot_distances']['relative_difference'])
        
        ax2.bar(x, differences_array, alpha=0.7, color='green', label='Absolute Difference')
        ax2.set_xlabel('Layer Pairs')
        ax2.set_ylabel('Difference (Trained - Untrained)')
        ax2.set_title('Absolute Differences in OT Distances')
        ax2.set_xticks(x)
        ax2.set_xticklabels(layer_pairs, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Relative Differences
        ax3 = axes[1, 0]
        ax3.bar(x, relative_differences, alpha=0.7, color='orange', label='Relative Difference')
        ax3.set_xlabel('Layer Pairs')
        ax3.set_ylabel('Relative Difference')
        ax3.set_title('Relative Differences in OT Distances')
        ax3.set_xticks(x)
        ax3.set_xticklabels(layer_pairs, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Entropy Comparison
        ax4 = axes[1, 1]
        if 'entropies' in differences:
            trained_entropies = differences['entropies']['trained']
            untrained_entropies = differences['entropies']['untrained']
            
            ax4.plot(range(len(trained_entropies)), trained_entropies, 'o-', 
                    color='blue', label='Trained Model')
            ax4.plot(range(len(untrained_entropies)), untrained_entropies, 's-',
                    color='red', label='Untrained Model')
            
            ax4.set_xlabel('Layer')
            ax4.set_ylabel('Entropy')
            ax4.set_title('Entropy Comparison')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'trained_vs_untrained_comparison.png'), dpi=300)
        plt.show()
        
        # Create heatmap of differences
        self._create_difference_heatmap(comparison_results)
        
        # Create transformation type comparison
        self._create_transformation_comparison(comparison_results)
    
    def _create_difference_heatmap(self, comparison_results: dict):
        """Create heatmap showing differences across layers."""
        differences = comparison_results['differences']
        
        if 'ot_distances' not in differences:
            return
        
        # Create matrix for heatmap
        trained_distances = np.array(differences['ot_distances']['trained'])
        untrained_distances = np.array(differences['ot_distances']['untrained'])
        
        # Stack for heatmap
        comparison_matrix = np.vstack([trained_distances, untrained_distances])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(comparison_matrix, 
                   xticklabels=[f'L{i}-L{i+1}' for i in range(len(trained_distances))],
                   yticklabels=['Trained', 'Untrained'],
                   cmap='RdBu_r',
                   center=0,
                   ax=ax,
                   annot=True,
                   fmt='.3f')
        
        ax.set_title('OT Distance Comparison Heatmap')
        ax.set_xlabel('Layer Pairs')
        ax.set_ylabel('Model Type')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'comparison_heatmap.png'), dpi=300)
        plt.show()
    
    def _create_transformation_comparison(self, comparison_results: dict):
        """Create visualization of transformation type comparison."""
        differences = comparison_results['differences']
        
        if 'geometric_characterization' not in differences:
            return
        
        geo_comp = differences['geometric_characterization']['transformation_type_comparison']
        
        # Create bar chart comparing transformation types
        fig, ax = plt.subplots(figsize=(10, 6))
        
        transformation_types = ['compression', 'spreading', 'stable']
        trained_scores = []
        untrained_scores = []
        
        for trans_type in transformation_types:
            trained_type = geo_comp['trained']
            untrained_type = geo_comp['untrained']
            
            trained_scores.append(1 if trained_type == trans_type else 0)
            untrained_scores.append(1 if untrained_type == trans_type else 0)
        
        x = np.arange(len(transformation_types))
        width = 0.35
        
        ax.bar(x - width/2, trained_scores, width, label='Trained Model', alpha=0.7)
        ax.bar(x + width/2, untrained_scores, width, label='Untrained Model', alpha=0.7)
        
        ax.set_xlabel('Transformation Type')
        ax.set_ylabel('Type Match')
        ax.set_title('Geometric Transformation Type Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(transformation_types)
        ax.legend()
        ax.set_ylim(0, 1.2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'transformation_type_comparison.png'), dpi=300)
        plt.show()
    
    def save_results(self, comparison_results: dict):
        """Save comparison results to files."""
        if not self.config['save_results']:
            return
        
        print("Saving comparison results...")
        
        # Save detailed results
        results_file = os.path.join(self.config['output_dir'], 'comparison_results.json')
        
        # Convert numpy types to Python types for JSON serialization
        def make_json_serializable(obj):
            if isinstance(obj, dict):
                return {key: make_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)
        
        serializable_results = make_json_serializable(comparison_results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save configuration
        config_file = os.path.join(self.config['output_dir'], 'config.json')
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Results saved to {self.config['output_dir']}/")
    
    def print_summary(self, comparison_results: dict):
        """Print a summary of the comparison results."""
        print("\n" + "="*60)
        print("TRAINED VS UNTRAINED COMPARISON SUMMARY")
        print("="*60)
        
        print(f"Model: {self.config['model_name']}")
        print(f"Number of samples: {self.config['num_samples']}")
        
        # OT Distance comparison
        differences = comparison_results['differences']
        if 'ot_distances' in differences:
            trained_distances = differences['ot_distances']['trained']
            untrained_distances = differences['ot_distances']['untrained']
            diff_values = differences['ot_distances']['difference']
            
            print(f"\nOT Distance Comparison:")
            print(f"  Trained model mean distance: {np.mean(trained_distances):.4f}")
            print(f"  Untrained model mean distance: {np.mean(untrained_distances):.4f}")
            print(f"  Mean difference (trained - untrained): {np.mean(diff_values):.4f}")
            
            # Find layers with biggest differences
            diff_array = np.array(diff_values)
            max_diff_idx = np.argmax(np.abs(diff_array))
            print(f"  Layer pair with biggest difference: L{max_diff_idx}-L{max_diff_idx+1} ({diff_values[max_diff_idx]:.4f})")
        
        # Geometric transformation comparison
        if 'geometric_characterization' in differences:
            geo_comp = differences['geometric_characterization']['transformation_type_comparison']
            print(f"\nGeometric Transformation Types:")
            print(f"  Trained model: {geo_comp['trained']}")
            print(f"  Untrained model: {geo_comp['untrained']}")
        
        # Entropy comparison
        if 'entropies' in differences:
            trained_entropies = differences['entropies']['trained']
            untrained_entropies = differences['entropies']['untrained']
            entropy_diff = differences['entropies']['difference']
            
            print(f"\nEntropy Comparison:")
            print(f"  Trained model mean entropy: {np.mean(trained_entropies):.4f}")
            print(f"  Untrained model mean entropy: {np.mean(untrained_entropies):.4f}")
            print(f"  Mean entropy difference: {np.mean(entropy_diff):.4f}")
        
        print("="*60)
    
    def run(self):
        """Run the complete trained vs untrained comparison."""
        print("Starting Trained vs Untrained Model Comparison")
        print("="*60)
        
        try:
            # Setup
            self.setup()
            
            # Load comparison data
            input_texts = self.load_comparison_data()
            
            # Run comparison
            comparison_results = self.run_comparison(input_texts)
            
            # Create visualizations
            if self.config['save_plots']:
                self.create_comparison_visualizations(comparison_results)
            
            # Save results
            self.save_results(comparison_results)
            
            # Print summary
            self.print_summary(comparison_results)
            
            # Store results
            self.results = comparison_results
            
            print("\nComparison completed successfully!")
            
        except Exception as e:
            print(f"Error during comparison: {e}")
            raise
        finally:
            # Cleanup
            if self.model_comparison:
                self.model_comparison.cleanup()

def main():
    """Main comparison function."""
    config = {
        'model_name': 'gpt2-medium',  # Use medium for reasonable comparison
        'num_samples': 20,  # Start with fewer samples for testing
        'max_length': 128,  # Shorter sequences for faster processing
        'output_dir': 'trained_vs_untrained_results',
        'save_plots': True,
        'save_results': True
    }
    
    experiment = TrainedVsUntrainedExperiment(config)
    experiment.run()
    
    return experiment

if __name__ == "__main__":
    main() 