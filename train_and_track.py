"""
Training script with representation tracking via Optimal Transport.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime

# Import our modules
from data_utils import GSM8KDataset
from transformer_utils import TrainingTracker, get_model_info
from ot_analysis import OTAnalyzer
from visualization import OTVisualizer

class GSM8KTrainer:
    """Train GPT-2 on GSM8K with representation tracking."""
    
    def __init__(self, config: dict = None):
        self.config = config or self._get_default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.tracker = None
        self.ot_analyzer = None
        self.visualizer = None
        
    def _get_default_config(self) -> dict:
        """Get default training configuration."""
        return {
            'model_name': 'gpt2',  # Use base GPT2 to avoid NaN issues
            'num_epochs': 3,
            'batch_size': 2,  # Smaller batch size for larger models
            'learning_rate': 5e-5,
            'max_length': 512,
            'num_samples': 100,  # Number of GSM8K samples to use
            'checkpoint_frequency': 50,  # Save checkpoint every N steps
            'sample_texts': [
                "Question: If there are 15 trees in the grove, and the grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nAnswer:",
                "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nAnswer:",
                "Question: Sam bought a dozen boxes of cookies, with 10 cookies in each box. If 10 of the cookies were eaten, how many cookies are left?\nAnswer:"
            ],
            'output_dir': 'training_results',
            'save_plots': True,
            'use_wandb': False
        }
    
    def setup(self):
        """Setup training components."""
        print("Setting up training...")
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Load model and tokenizer
        print(f"Loading {self.config['model_name']}...")
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        if self.config['model_name'] == 'gpt2':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        elif self.config['model_name'] == 'gpt2-medium':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        elif self.config['model_name'] == 'gpt2-large':
            self.model = GPT2LMHeadModel.from_pretrained('gpt2-large')
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        
        # Print model info
        model_info = get_model_info(self.model)
        print(f"Model: {self.config['model_name']}")
        print(f"Layers: {model_info['num_layers']}")
        print(f"Hidden size: {model_info['hidden_size']}")
        print(f"Parameters: {model_info['total_parameters']:,}")
        
        # Setup dataset
        print("Loading GSM8K dataset...")
        self.dataset = GSM8KDataset(
            model_name=self.config['model_name'],
            max_length=self.config['max_length']
        )
        
        # Setup training tracker
        self.tracker = TrainingTracker(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        # Setup OT analyzer
        self.ot_analyzer = OTAnalyzer(method="sinkhorn", normalize=True)
        
        # Setup visualizer
        self.visualizer = OTVisualizer()
        
        print("Setup complete!")
    
    def prepare_data(self):
        """Prepare training data."""
        print("Preparing training data...")
        
        # Get GSM8K data
        raw_data = self.dataset.preprocess_gsm8k(
            split="train", 
            num_samples=self.config['num_samples']
        )
        
        # Tokenize data
        tokenized_data = self.dataset.tokenize_data(raw_data)
        
        # Create batches
        batches = self.dataset.create_dataloader(
            tokenized_data, 
            batch_size=self.config['batch_size']
        )
        
        print(f"Prepared {len(batches)} batches from {len(tokenized_data)} samples")
        return batches
    
    def train_epoch(self, batches, epoch: int, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(batches)
        
        progress_bar = tqdm(batches, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            # Prepare batch
            input_ids = torch.cat([item['input_ids'] for item in batch]).to(self.device)
            attention_mask = torch.cat([item['attention_mask'] for item in batch]).to(self.device)
            labels = torch.cat([item['labels'] for item in batch]).to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at step {step}, skipping...")
                continue
                
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Save checkpoint
            if step % self.config['checkpoint_frequency'] == 0:
                self.tracker.save_checkpoint(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    sample_texts=self.config['sample_texts'],
                    checkpoint_dir=os.path.join(self.config['output_dir'], 'checkpoints')
                )
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def analyze_representation_drift(self):
        """Analyze representation drift across checkpoints."""
        print("Analyzing representation drift...")
        
        if not self.tracker.checkpoints:
            print("No checkpoints found!")
            return
        
        # Get activations from different checkpoints
        checkpoint_activations = {}
        
        for checkpoint in self.tracker.checkpoints:
            epoch = checkpoint['epoch']
            step = checkpoint['step']
            key = f"epoch_{epoch}_step_{step}"
            
            activations = self.tracker.get_activations_at_checkpoint(epoch, step)
            if activations:
                checkpoint_activations[key] = activations
        
        # Analyze drift between checkpoints
        drift_results = {}
        checkpoint_keys = sorted(checkpoint_activations.keys())
        
        for i in range(len(checkpoint_keys) - 1):
            checkpoint1 = checkpoint_keys[i]
            checkpoint2 = checkpoint_keys[i + 1]
            
            print(f"Analyzing drift: {checkpoint1} → {checkpoint2}")
            
            # Get layer representations for both checkpoints
            activations1 = checkpoint_activations[checkpoint1]
            activations2 = checkpoint_activations[checkpoint2]
            
            # Prepare representations for OT analysis
            layer_reps1 = self._prepare_layer_representations(activations1)
            layer_reps2 = self._prepare_layer_representations(activations2)
            
            # Compute OT distances between corresponding layers
            layer_distances = []
            for layer_idx in range(len(layer_reps1)):
                if layer_idx < len(layer_reps2):
                    result = self.ot_analyzer.compute_ot_distance(
                        layer_reps1[layer_idx], 
                        layer_reps2[layer_idx]
                    )
                    # compute_ot_distance returns (distance, plan), we only need distance
                    distance = result[0] if result is not None else None
                    if distance is not None:  # Only add valid distances
                        layer_distances.append(distance)
            
            if layer_distances:  # Only compute statistics if we have valid distances
                drift_results[f"{checkpoint1}_to_{checkpoint2}"] = {
                    'layer_distances': layer_distances,
                    'mean_drift': np.mean(layer_distances),
                    'max_drift': np.max(layer_distances),
                    'drift_std': np.std(layer_distances)
                }
            else:
                print(f"Warning: No valid OT distances for {checkpoint1} → {checkpoint2}")
        
        return drift_results
    
    def _prepare_layer_representations(self, activations: dict) -> list:
        """Prepare layer representations for OT analysis."""
        layer_reps = []
        
        # Get all layer representations in order
        num_layers = len(activations)
        for i in range(num_layers):
            layer_name = f"layer_{i}"
            if layer_name in activations:
                reps = activations[layer_name]
                
                # Convert to numpy and reshape
                reps_np = reps.numpy()
                batch_size, seq_len, hidden_dim = reps_np.shape
                reps_reshaped = reps_np.reshape(-1, hidden_dim)
                
                # Remove padding tokens
                non_padding_mask = np.any(reps_reshaped != 0, axis=1)
                reps_clean = reps_reshaped[non_padding_mask]
                
                layer_reps.append(reps_clean)
        
        return layer_reps
    
    def create_drift_visualizations(self, drift_results: dict):
        """Create visualizations of representation drift."""
        if not drift_results:
            return
        
        print("Creating drift visualizations...")
        
        # Create drift heatmap
        checkpoint_pairs = list(drift_results.keys())
        num_layers = len(drift_results[checkpoint_pairs[0]]['layer_distances'])
        
        drift_matrix = np.zeros((len(checkpoint_pairs), num_layers))
        
        for i, pair in enumerate(checkpoint_pairs):
            distances = drift_results[pair]['layer_distances']
            drift_matrix[i, :len(distances)] = distances
        
        # Plot drift heatmap
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(drift_matrix, 
                   xticklabels=[f'L{i}' for i in range(num_layers)],
                   yticklabels=[pair.replace('_to_', ' → ') for pair in checkpoint_pairs],
                   cmap='YlOrRd',
                   ax=ax)
        ax.set_title('Representation Drift Across Training Checkpoints')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Checkpoint Transition')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'drift_heatmap.png'), dpi=300)
        plt.show()
        
        # Plot drift statistics
        mean_drifts = [drift_results[pair]['mean_drift'] for pair in checkpoint_pairs]
        max_drifts = [drift_results[pair]['max_drift'] for pair in checkpoint_pairs]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mean drift
        ax1.plot(range(len(checkpoint_pairs)), mean_drifts, 'o-', color='blue')
        ax1.set_title('Mean Representation Drift')
        ax1.set_xlabel('Checkpoint Transition')
        ax1.set_ylabel('Mean OT Distance')
        ax1.grid(True, alpha=0.3)
        
        # Max drift
        ax2.plot(range(len(checkpoint_pairs)), max_drifts, 's-', color='red')
        ax2.set_title('Maximum Representation Drift')
        ax2.set_xlabel('Checkpoint Transition')
        ax2.set_ylabel('Max OT Distance')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'drift_statistics.png'), dpi=300)
        plt.show()
    
    def train(self):
        """Run the complete training process."""
        print("Starting GSM8K training with representation tracking")
        print("=" * 60)
        
        try:
            # Setup
            self.setup()
            
            # Prepare data
            batches = self.prepare_data()
            
            # Setup training
            optimizer = optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            training_losses = []
            
            for epoch in range(self.config['num_epochs']):
                print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
                
                avg_loss = self.train_epoch(batches, epoch, optimizer, criterion)
                training_losses.append(avg_loss)
                
                print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
            
            # Save final checkpoint
            self.tracker.save_checkpoint(
                epoch=self.config['num_epochs'] - 1,
                step=len(batches) - 1,
                loss=avg_loss,
                sample_texts=self.config['sample_texts'],
                checkpoint_dir=os.path.join(self.config['output_dir'], 'checkpoints')
            )
            
            # Save training history
            self.tracker.save_training_history(
                os.path.join(self.config['output_dir'], 'training_history.json')
            )
            
            # Analyze representation drift
            drift_results = self.analyze_representation_drift()
            
            # Create visualizations
            if self.config['save_plots']:
                self.create_drift_visualizations(drift_results)
            
            # Save results
            results = {
                'training_losses': training_losses,
                'drift_results': drift_results,
                'config': self.config
            }
            
            with open(os.path.join(self.config['output_dir'], 'results.json'), 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\nTraining completed! Results saved to {self.config['output_dir']}/")
            
        except Exception as e:
            print(f"Error during training: {e}")
            raise

def main():
    """Main training function."""
    config = {
        'model_name': 'gpt2',  # Use base GPT2 to avoid NaN issues
        'num_epochs': 2,  # Start with fewer epochs for testing
        'batch_size': 1,  # Small batch size for memory constraints
        'learning_rate': 1e-6,  # Much lower learning rate to prevent NaN
        'max_length': 128,  # Shorter sequences for faster training
        'num_samples': 20,  # Start with fewer samples
        'checkpoint_frequency': 5,  # More frequent checkpoints
        'sample_texts': [
            "Question: If there are 15 trees in the grove, and the grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nAnswer:",
            "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nAnswer:",
            "Question: Sam bought a dozen boxes of cookies, with 10 cookies in each box. If 10 of the cookies were eaten, how many cookies are left?\nAnswer:"
        ],
        'output_dir': 'training_results',
        'save_plots': True,
        'use_wandb': False
    }
    
    trainer = GSM8KTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main() 