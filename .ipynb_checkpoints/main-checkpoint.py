"""
Main experiment runner for Tracking Representation Drift in Transformers via Optimal Transport.
"""

import torch
import numpy as np
import os
import json
from datetime import datetime
from typing import List, Dict

# Import our modules
from data_utils import GSM8KDataset
from transformer_utils import ActivationExtractor
from ot_analysis import OTAnalyzer, compare_ot_vs_simple_distances
from visualization import OTVisualizer

class RepresentationDriftExperiment:
    """Main experiment class for tracking representation drift via Optimal Transport."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the experiment.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.extractor = None
        self.ot_analyzer = None
        self.visualizer = None
        self.results = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'model_name': 'gpt2-medium',  # Larger model for better performance
            'num_samples': 50,  # Number of GSM8K samples
            'max_length': 256,  # Shorter sequences for efficiency
            'batch_size': 4,
            'ot_method': 'sinkhorn',  # Use Sinkhorn for efficiency
            'normalize': True,
            'pca_components': 50,
            'transition_threshold': 2.0,
            'save_results': True,
            'save_plots': True,
            'output_dir': 'results'
        }
    
    def setup(self):
        """Setup the experiment components."""
        print("Setting up experiment...")
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Initialize activation extractor
        print("Loading transformer model...")
        self.extractor = ActivationExtractor(
            model_name=self.config['model_name'],
            device=self.device
        )
        
        # Initialize OT analyzer
        self.ot_analyzer = OTAnalyzer(
            method=self.config['ot_method'],
            normalize=self.config['normalize']
        )
        
        # Initialize visualizer
        self.visualizer = OTVisualizer()
        
        print("Setup complete!")
    
    def load_gsm8k_data(self) -> List[str]:
        """Load GSM8K dataset and extract sample questions."""
        print("Loading GSM8K dataset...")
        
        dataset = GSM8KDataset(
            model_name=self.config['model_name'],
            max_length=self.config['max_length']
        )
        
        # Get sample data
        data = dataset.get_sample_data(self.config['num_samples'])
        
        # Extract input texts for activation extraction
        input_texts = [item['input_text'] for item in data]
        
        print(f"Loaded {len(input_texts)} GSM8K samples")
        return input_texts
    
    def extract_activations(self, input_texts: List[str]) -> Dict[str, torch.Tensor]:
        """Extract activations from all layers."""
        print("Extracting activations from transformer layers...")
        
        activations = self.extractor.extract_activations_batch(input_texts)
        
        print(f"Extracted activations from {len(activations)} layers")
        return activations
    
    def prepare_representations(self, activations: Dict[str, torch.Tensor]) -> List[np.ndarray]:
        """Prepare layer representations for OT analysis."""
        print("Preparing layer representations...")
        
        layer_representations = []
        
        # Get all layer representations in order
        all_reps = self.extractor.get_all_layer_representations(activations)
        
        for i, reps in enumerate(all_reps):
            # Convert to numpy and reshape: (batch_size, seq_len, hidden_dim) -> (batch_size * seq_len, hidden_dim)
            reps_np = reps.numpy()
            batch_size, seq_len, hidden_dim = reps_np.shape
            
            # Reshape to combine batch and sequence dimensions
            reps_reshaped = reps_np.reshape(-1, hidden_dim)
            
            # Remove padding tokens (assuming padding token ID is 0)
            # This is a simple heuristic - you might need to adjust based on your tokenizer
            non_padding_mask = np.any(reps_reshaped != 0, axis=1)
            reps_clean = reps_reshaped[non_padding_mask]
            
            layer_representations.append(reps_clean)
            print(f"Layer {i}: {reps_clean.shape}")
        
        return layer_representations
    
    def run_ot_analysis(self, layer_representations: List[np.ndarray]) -> Dict:
        """Run Optimal Transport analysis."""
        print("Running Optimal Transport analysis...")
        
        # Comprehensive analysis
        analysis_results = self.ot_analyzer.analyze_representation_flow(layer_representations)
        
        # Compare with simple distance metrics
        comparison_results = compare_ot_vs_simple_distances(layer_representations)
        
        # Combine results
        results = {
            **analysis_results,
            'comparison': comparison_results
        }
        
        print("OT analysis complete!")
        return results
    
    def create_visualizations(self, analysis_results: Dict):
        """Create visualizations of the results."""
        print("Creating visualizations...")
        
        # Create comprehensive dashboard
        self.visualizer.create_comprehensive_dashboard(
            analysis_results, 
            save_dir=f"{self.config['output_dir']}/plots"
        )
        
        # Create distance comparison plot
        self.visualizer.plot_distance_comparison(
            analysis_results['comparison'],
            save_path=f"{self.config['output_dir']}/plots/distance_comparison.png"
        )
        
        print("Visualizations complete!")
    
    def _make_json_serializable(self, obj):
        """Recursively convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def save_results(self, analysis_results: Dict):
        """Save results to files."""
        if not self.config['save_results']:
            return
        
        print("Saving results...")
        
        # Save analysis results
        results_file = f"{self.config['output_dir']}/analysis_results.json"
        
        # Convert all numpy types to Python types
        serializable_results = self._make_json_serializable(analysis_results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save configuration
        config_file = f"{self.config['output_dir']}/config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Results saved to {self.config['output_dir']}/")
    
    def print_summary(self, analysis_results: Dict):
        """Print a summary of the results."""
        print("\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        
        print(f"Model: {self.config['model_name']}")
        print(f"Dataset: GSM8K")
        print(f"Number of layers: {analysis_results['num_layers']}")
        print(f"Number of samples: {self.config['num_samples']}")
        print(f"OT Method: {self.config['ot_method']}")
        
        print(f"\nOT Distances between adjacent layers:")
        for i, dist in enumerate(analysis_results['ot_distances']):
            print(f"  L{i}-L{i+1}: {dist:.4f}")
        
        if analysis_results['phase_transitions']:
            print(f"\nPhase transitions detected at layers: {analysis_results['phase_transitions']}")
        else:
            print("\nNo phase transitions detected")
        
        print(f"\nEntropy evolution:")
        for i, entropy in enumerate(analysis_results['entropies']):
            print(f"  L{i}: {entropy:.4f}")
        
        print("="*50)
    
    def run(self):
        """Run the complete experiment."""
        print("Starting Representation Drift Detection Experiment")
        print("="*60)
        
        try:
            # Setup
            self.setup()
            
            # Load GSM8K data
            input_texts = self.load_gsm8k_data()
            
            # Extract activations
            activations = self.extract_activations(input_texts)
            
            # Prepare representations
            layer_representations = self.prepare_representations(activations)
            
            # Run OT analysis
            analysis_results = self.run_ot_analysis(layer_representations)
            
            # Create visualizations
            if self.config['save_plots']:
                self.create_visualizations(analysis_results)
            
            # Save results
            self.save_results(analysis_results)
            
            # Print summary
            self.print_summary(analysis_results)
            
            # Store results
            self.results = analysis_results
            
            print("\nExperiment completed successfully!")
            
        except Exception as e:
            print(f"Error during experiment: {e}")
            raise
        finally:
            # Cleanup
            if self.extractor:
                self.extractor.cleanup()

def run_quick_experiment():
    """Run a quick experiment with default settings."""
    config = {
        'model_name': 'gpt2-medium',
        'num_samples': 20,  # Small number for quick testing
        'max_length': 256,  # Shorter sequences
        'batch_size': 2,
        'ot_method': 'sinkhorn',
        'normalize': True,
        'pca_components': 30,
        'transition_threshold': 2.0,
        'save_results': True,
        'save_plots': True,
        'output_dir': 'results'
    }
    
    experiment = RepresentationDriftExperiment(config)
    experiment.run()
    
    return experiment

def run_large_model_experiment():
    """Run experiment with larger model."""
    config = {
        'model_name': 'gpt2-large',
        'num_samples': 2000,
        'max_length': 512,
        'batch_size': 4,  # Smaller batch size for larger model
        'ot_method': 'sinkhorn',
        'normalize': True,
        'pca_components': 200,
        'transition_threshold': 2.0,
        'save_results': True,
        'save_plots': True,
        'output_dir': 'results_large_model'
    }
    
    experiment = RepresentationDriftExperiment(config)
    experiment.run()
    
    return experiment

if __name__ == "__main__":
    print("Tracking Representation Drift in Transformers via Optimal Transport")
    print("=" * 60)
    
    # Run quick experiment
    #experiment = run_quick_experiment()
    
    # Uncomment to run with larger model (requires more memory)
    experiment = run_large_model_experiment() 