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
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto"):
        self.device = self._get_device(device)
        self.model_name = model_name
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
        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")
        
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
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded with {self.model.config.num_hidden_layers} layers")
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
        
        for text in tqdm(input_texts, desc="Extracting activations"):
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
        
        # Extract activations for sample texts
        extractor = ActivationExtractor(model_name="custom", device=self.device)
        extractor.model = self.model
        extractor.tokenizer = self.tokenizer
        extractor._register_hooks()
        
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