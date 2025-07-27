"""
Utilities for loading transformer models and extracting hidden state activations.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import os
import json

class ActivationExtractor:
    """Extract hidden state activations from transformer layers."""
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto", model_type: str = "trained"):
        self.device = self._get_device(device)
        self.model_name = model_name
        self.model_type = model_type  # "trained" or "untrained"
        self.model = None
        self.tokenizer = None
        self.hooks = []
        self.activations = {}
        
        self._load_model()
        self._register_hooks()
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load the transformer model and tokenizer."""
        print(f"Loading {self.model_type} model: {self.model_name}")
        print(f"Using device: {self.device}")
        
        if self.model_type == "trained":
            # Load pre-trained model
            if self.model_name == "gpt2":
                self.model = GPT2LMHeadModel.from_pretrained("gpt2")
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            elif self.model_name == "gpt2-medium":
                self.model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
            elif self.model_name == "gpt2-large":
                self.model = GPT2LMHeadModel.from_pretrained("gpt2-large")
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
            elif self.model_name == "gpt2-xl":
                self.model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
            else:
                # For other models, try AutoModel
                self.model = AutoModel.from_pretrained(self.model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        else:
            # Load untrained model (random weights)
            if self.model_name == "gpt2":
                config = GPT2Config.from_pretrained("gpt2")
                self.model = GPT2LMHeadModel(config)
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            elif self.model_name == "gpt2-medium":
                config = GPT2Config.from_pretrained("gpt2-medium")
                self.model = GPT2LMHeadModel(config)
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
            elif self.model_name == "gpt2-large":
                config = GPT2Config.from_pretrained("gpt2-large")
                self.model = GPT2LMHeadModel(config)
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
            elif self.model_name == "gpt2-xl":
                config = GPT2Config.from_pretrained("gpt2-xl")
                self.model = GPT2LMHeadModel(config)
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
            else:
                # For other models
                config = AutoModel.from_pretrained(self.model_name).config
                self.model = AutoModel.from_config(config)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"{self.model_type.capitalize()} model loaded with {self.model.config.num_hidden_layers} layers")
        print(f"Hidden size: {self.model.config.hidden_size}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _register_hooks(self):
        """Register hooks to capture activations from each layer."""
        self.activations = {}
        self.hooks = []
        
        # Clear any existing hooks
        for hook in self.hooks:
            hook.remove()
        
        # Register hooks for each transformer layer
        for i, layer in enumerate(self.model.transformer.h):
            def make_hook(layer_idx):
                def hook(module, input, output):
                    # output is typically a tuple, we want the hidden states
                    if isinstance(output, tuple):
                        hidden_states = output[0]  # First element is usually hidden states
                    else:
                        hidden_states = output
                    
                    # Store activations for this layer
                    self.activations[f"layer_{layer_idx}"] = hidden_states.detach().cpu()
                return hook
            
            hook = layer.register_forward_hook(make_hook(i))
            self.hooks.append(hook)
    
    def extract_activations(self, input_text: str) -> Dict[str, torch.Tensor]:
        """Extract activations for a given input text."""
        # Clear previous activations
        self.activations = {}
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass (this will trigger the hooks)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return self.activations
    
    def extract_activations_batch(self, input_texts: List[str]) -> Dict[str, torch.Tensor]:
        """Extract activations for a batch of input texts."""
        all_activations = {}
        
        for text in tqdm(input_texts, desc=f"Extracting activations ({self.model_type})"):
            activations = self.extract_activations(text)
            
            # Concatenate activations across batch
            for layer_name, layer_acts in activations.items():
                if layer_name not in all_activations:
                    all_activations[layer_name] = []
                all_activations[layer_name].append(layer_acts)
        
        # Stack activations from all samples
        for layer_name in all_activations:
            # Pad sequences to same length before stacking
            max_seq_len = max(acts.shape[1] for acts in all_activations[layer_name])
            padded_activations = []
            
            for acts in all_activations[layer_name]:
                batch_size, seq_len, hidden_dim = acts.shape
                if seq_len < max_seq_len:
                    # Pad with zeros
                    padding = torch.zeros(batch_size, max_seq_len - seq_len, hidden_dim)
                    acts_padded = torch.cat([acts, padding], dim=1)
                else:
                    acts_padded = acts
                padded_activations.append(acts_padded)
            
            all_activations[layer_name] = torch.cat(padded_activations, dim=0)
        
        return all_activations
    
    def get_layer_representations(self, activations: Dict[str, torch.Tensor], 
                                layer_idx: int) -> torch.Tensor:
        """Get representations from a specific layer."""
        layer_name = f"layer_{layer_idx}"
        if layer_name not in activations:
            raise ValueError(f"Layer {layer_idx} not found in activations")
        
        # activations[layer_name] shape: (batch_size, seq_len, hidden_dim)
        return activations[layer_name]
    
    def get_all_layer_representations(self, activations: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Get representations from all layers in order."""
        num_layers = len(activations)
        layer_reps = []
        
        for i in range(num_layers):
            layer_name = f"layer_{i}"
            if layer_name in activations:
                layer_reps.append(activations[layer_name])
        
        return layer_reps
    
    def cleanup(self):
        """Remove hooks and clean up."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

class ModelComparison:
    """Compare trained vs untrained models for geometric transformation analysis."""
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto"):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.trained_extractor = None
        self.untrained_extractor = None
        
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def setup_models(self):
        """Setup both trained and untrained models."""
        print("Setting up trained and untrained models for comparison...")
        
        # Load trained model
        self.trained_extractor = ActivationExtractor(
            model_name=self.model_name,
            device=self.device,
            model_type="trained"
        )
        
        # Load untrained model
        self.untrained_extractor = ActivationExtractor(
            model_name=self.model_name,
            device=self.device,
            model_type="untrained"
        )
        
        print("Both models loaded successfully!")
    
    def extract_comparison_activations(self, input_texts: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Extract activations from both models using the same input texts."""
        print("Extracting activations from both models...")
        
        # Extract from trained model
        trained_activations = self.trained_extractor.extract_activations_batch(input_texts)
        
        # Extract from untrained model
        untrained_activations = self.untrained_extractor.extract_activations_batch(input_texts)
        
        return {
            "trained": trained_activations,
            "untrained": untrained_activations
        }
    
    def compare_geometric_transformations(self, activations: Dict[str, Dict[str, torch.Tensor]]) -> Dict:
        """Compare geometric transformations between trained and untrained models."""
        from ot_analysis import OTAnalyzer
        
        print("Comparing geometric transformations...")
        
        ot_analyzer = OTAnalyzer(method="sinkhorn", normalize=True)
        
        # Prepare representations for both models
        trained_reps = self._prepare_layer_representations(activations["trained"])
        untrained_reps = self._prepare_layer_representations(activations["untrained"])
        
        # Analyze both models
        trained_analysis = ot_analyzer.analyze_representation_flow(trained_reps)
        untrained_analysis = ot_analyzer.analyze_representation_flow(untrained_reps)
        
        # Compare results
        comparison = {
            "trained": trained_analysis,
            "untrained": untrained_analysis,
            "differences": self._compute_differences(trained_analysis, untrained_analysis)
        }
        
        return comparison
    
    def _prepare_layer_representations(self, activations: Dict[str, torch.Tensor]) -> List[np.ndarray]:
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
    
    def _compute_differences(self, trained_analysis: Dict, untrained_analysis: Dict) -> Dict:
        """Compute differences between trained and untrained model analyses."""
        differences = {}
        
        # Compare OT distances
        if 'ot_distances' in trained_analysis and 'ot_distances' in untrained_analysis:
            trained_distances = np.array(trained_analysis['ot_distances'])
            untrained_distances = np.array(untrained_analysis['ot_distances'])
            
            differences['ot_distances'] = {
                'trained': trained_distances.tolist(),
                'untrained': untrained_distances.tolist(),
                'difference': (trained_distances - untrained_distances).tolist(),
                'relative_difference': ((trained_distances - untrained_distances) / (untrained_distances + 1e-8)).tolist()
            }
        
        # Compare entropies
        if 'entropies' in trained_analysis and 'entropies' in untrained_analysis:
            trained_entropies = np.array(trained_analysis['entropies'])
            untrained_entropies = np.array(untrained_analysis['entropies'])
            
            differences['entropies'] = {
                'trained': trained_entropies.tolist(),
                'untrained': untrained_entropies.tolist(),
                'difference': (trained_entropies - untrained_entropies).tolist()
            }
        
        # Compare geometric characterizations
        if 'geometric_characterization' in trained_analysis and 'geometric_characterization' in untrained_analysis:
            trained_geo = trained_analysis['geometric_characterization']
            untrained_geo = untrained_analysis['geometric_characterization']
            
            differences['geometric_characterization'] = {
                'trained': trained_geo,
                'untrained': untrained_geo,
                'transformation_type_comparison': {
                    'trained': trained_geo.get('transformation_type', 'unknown'),
                    'untrained': untrained_geo.get('transformation_type', 'unknown')
                }
            }
        
        return differences
    
    def cleanup(self):
        """Clean up both models."""
        if self.trained_extractor:
            self.trained_extractor.cleanup()
        if self.untrained_extractor:
            self.untrained_extractor.cleanup()

class TrainingTracker:
    """Track model training and extract activations at checkpoints."""
    
    def __init__(self, model: nn.Module, tokenizer, device: str = "auto"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = self._get_device(device)
        self.model.to(self.device)
        
        self.checkpoints = []
        self.activations_history = {}
        
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def save_checkpoint(self, epoch: int, step: int, loss: float, 
                       sample_texts: List[str], checkpoint_dir: str = "checkpoints"):
        """Save model checkpoint and extract activations."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model state
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{step}.pt")
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        
        # Extract activations for sample texts using the existing model
        # Create a simple activation extractor that uses the existing model
        class SimpleActivationExtractor:
            def __init__(self, model, tokenizer, device):
                self.model = model
                self.tokenizer = tokenizer
                self.device = device
                self.hooks = []
                self.activations = {}
                self._register_hooks()
            
            def _register_hooks(self):
                """Register hooks to capture activations from each layer."""
                self.activations = {}
                self.hooks = []
                
                # Register hooks for each transformer layer
                for i, layer in enumerate(self.model.transformer.h):
                    def make_hook(layer_idx):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                hidden_states = output[0]
                            else:
                                hidden_states = output
                            self.activations[f"layer_{layer_idx}"] = hidden_states.detach().cpu()
                        return hook
                    
                    hook = layer.register_forward_hook(make_hook(i))
                    self.hooks.append(hook)
            
            def extract_activations_batch(self, input_texts: List[str]) -> Dict[str, torch.Tensor]:
                """Extract activations for a batch of input texts."""
                all_activations = {}
                
                for text in tqdm(input_texts, desc="Extracting activations"):
                    # Clear previous activations
                    self.activations = {}
                    
                    # Tokenize input
                    inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Forward pass (this will trigger the hooks)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    # Concatenate activations across batch
                    for layer_name, layer_acts in self.activations.items():
                        if layer_name not in all_activations:
                            all_activations[layer_name] = []
                        all_activations[layer_name].append(layer_acts)
                
                # Stack activations from all samples
                for layer_name in all_activations:
                    # Pad sequences to same length before stacking
                    max_seq_len = max(acts.shape[1] for acts in all_activations[layer_name])
                    padded_activations = []
                    
                    for acts in all_activations[layer_name]:
                        batch_size, seq_len, hidden_dim = acts.shape
                        if seq_len < max_seq_len:
                            # Pad with zeros
                            padding = torch.zeros(batch_size, max_seq_len - seq_len, hidden_dim)
                            acts_padded = torch.cat([acts, padding], dim=1)
                        else:
                            acts_padded = acts
                        padded_activations.append(acts_padded)
                    
                    all_activations[layer_name] = torch.cat(padded_activations, dim=0)
                
                return all_activations
            
            def cleanup(self):
                """Remove hooks and clean up."""
                for hook in self.hooks:
                    hook.remove()
                self.hooks = []
        
        # Use the simple extractor
        extractor = SimpleActivationExtractor(self.model, self.tokenizer, self.device)
        activations = extractor.extract_activations_batch(sample_texts)
        
        # Save activations
        activations_path = os.path.join(checkpoint_dir, f"activations_epoch_{epoch}_step_{step}.pt")
        torch.save(activations, activations_path)
        
        # Store checkpoint info
        checkpoint_info = {
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'checkpoint_path': checkpoint_path,
            'activations_path': activations_path
        }
        self.checkpoints.append(checkpoint_info)
        
        # Store activations in memory
        self.activations_history[f"epoch_{epoch}_step_{step}"] = activations
        
        extractor.cleanup()
        
        print(f"Checkpoint saved: epoch {epoch}, step {step}, loss {loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
    
    def get_activations_at_checkpoint(self, epoch: int, step: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get activations for a specific checkpoint."""
        key = f"epoch_{epoch}_step_{step}"
        return self.activations_history.get(key)
    
    def save_training_history(self, output_path: str = "training_history.json"):
        """Save training history to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.checkpoints, f, indent=2)

def load_pretrained_model(model_name: str = "gpt2", device: str = "auto"):
    """Load a pretrained transformer model."""
    if model_name == "gpt2":
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    elif model_name == "gpt2-medium":
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    elif model_name == "gpt2-large":
        model = GPT2LMHeadModel.from_pretrained("gpt2-large")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    elif model_name == "gpt2-xl":
        model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    else:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() and device == "auto" else device
    model.to(device)
    model.eval()
    
    return model, tokenizer

def load_untrained_model(model_name: str = "gpt2", device: str = "auto"):
    """Load an untrained (randomly initialized) transformer model."""
    if model_name == "gpt2":
        config = GPT2Config.from_pretrained("gpt2")
        model = GPT2LMHeadModel(config)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    elif model_name == "gpt2-medium":
        config = GPT2Config.from_pretrained("gpt2-medium")
        model = GPT2LMHeadModel(config)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    elif model_name == "gpt2-large":
        config = GPT2Config.from_pretrained("gpt2-large")
        model = GPT2LMHeadModel(config)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
    elif model_name == "gpt2-xl":
        config = GPT2Config.from_pretrained("gpt2-xl")
        model = GPT2LMHeadModel(config)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    else:
        config = AutoModel.from_pretrained(model_name).config
        model = AutoModel.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() and device == "auto" else device
    model.to(device)
    model.eval()
    
    return model, tokenizer

def get_model_info(model):
    """Get basic information about the model."""
    info = {
        'num_layers': model.config.num_hidden_layers,
        'hidden_size': model.config.hidden_size,
        'num_attention_heads': model.config.num_attention_heads,
        'intermediate_size': getattr(model.config, 'intermediate_size', None),
        'vocab_size': model.config.vocab_size,
        'total_parameters': sum(p.numel() for p in model.parameters())
    }
    return info

def sample_from_model(model, tokenizer, prompt: str, max_length: int = 100, temperature: float = 0.7) -> str:
    """Generate text from the model given a prompt."""
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text 