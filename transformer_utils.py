"""
Utilities for loading transformer models and extracting hidden state activations.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

class ActivationExtractor:
    """Extract hidden state activations from transformer layers."""
    
    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.hooks = []
        self.activations = {}
        
        self._load_model()
        self._register_hooks()
    
    def _load_model(self):
        """Load the transformer model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        
        if self.model_name == "gpt2":
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            # For other models, try AutoModel
            self.model = AutoModel.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded with {self.model.config.num_hidden_layers} layers")
    
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

def load_pretrained_model(model_name: str = "gpt2", device: str = "cpu"):
    """Load a pretrained transformer model."""
    if model_name == "gpt2":
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
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
        'vocab_size': model.config.vocab_size
    }
    return info

def sample_from_model(model, tokenizer, prompt: str, max_length: int = 50) -> str:
    """Generate text from the model given a prompt."""
    inputs = tokenizer(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text 