"""
Data utilities for generating toy datasets and processing data.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
import random

class ToyDataset:
    """Generate toy datasets for transformer experiments."""
    
    def __init__(self, task_type: str = "copy", vocab_size: int = 100, max_length: int = 20):
        self.task_type = task_type
        self.vocab_size = vocab_size
        self.max_length = max_length
        
    def generate_copy_task_data(self, num_samples: int = 1000) -> List[Tuple[str, str]]:
        """Generate copy task: input -> output (same as input)."""
        data = []
        for _ in range(num_samples):
            # Generate random sequence
            length = random.randint(5, self.max_length)
            sequence = [random.randint(0, self.vocab_size - 1) for _ in range(length)]
            input_seq = " ".join(map(str, sequence))
            output_seq = input_seq  # Copy task: output = input
            data.append((input_seq, output_seq))
        return data
    
    def generate_parity_task_data(self, num_samples: int = 1000) -> List[Tuple[str, str]]:
        """Generate parity task: count even numbers in sequence."""
        data = []
        for _ in range(num_samples):
            length = random.randint(5, self.max_length)
            sequence = [random.randint(0, self.vocab_size - 1) for _ in range(length)]
            even_count = sum(1 for x in sequence if x % 2 == 0)
            input_seq = " ".join(map(str, sequence))
            output_seq = f"even_count: {even_count}"
            data.append((input_seq, output_seq))
        return data
    
    def generate_arithmetic_task_data(self, num_samples: int = 1000) -> List[Tuple[str, str]]:
        """Generate simple arithmetic task: sum of sequence."""
        data = []
        for _ in range(num_samples):
            length = random.randint(3, 8)
            sequence = [random.randint(1, 20) for _ in range(length)]
            total = sum(sequence)
            input_seq = " ".join(map(str, sequence))
            output_seq = f"sum: {total}"
            data.append((input_seq, output_seq))
        return data
    
    def generate_data(self, num_samples: int = 1000) -> List[Tuple[str, str]]:
        """Generate data based on task type."""
        if self.task_type == "copy":
            return self.generate_copy_task_data(num_samples)
        elif self.task_type == "parity":
            return self.generate_parity_task_data(num_samples)
        elif self.task_type == "arithmetic":
            return self.generate_arithmetic_task_data(num_samples)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

def tokenize_data(data: List[Tuple[str, str]], tokenizer) -> List[Dict]:
    """Tokenize the data using the provided tokenizer."""
    tokenized_data = []
    
    for input_text, output_text in data:
        # Tokenize input
        input_tokens = tokenizer.encode(input_text, return_tensors='pt')
        
        # Tokenize output
        output_tokens = tokenizer.encode(output_text, return_tensors='pt')
        
        tokenized_data.append({
            'input_ids': input_tokens,
            'output_ids': output_tokens,
            'input_text': input_text,
            'output_text': output_text
        })
    
    return tokenized_data

def create_batch(tokenized_data: List[Dict], batch_size: int = 8) -> List[Dict]:
    """Create batches from tokenized data."""
    batches = []
    
    for i in range(0, len(tokenized_data), batch_size):
        batch = tokenized_data[i:i + batch_size]
        
        # Pad sequences to same length
        max_input_len = max(len(item['input_ids'][0]) for item in batch)
        max_output_len = max(len(item['output_ids'][0]) for item in batch)
        
        padded_batch = []
        for item in batch:
            input_padding = max_input_len - len(item['input_ids'][0])
            output_padding = max_output_len - len(item['output_ids'][0])
            
            padded_input = torch.cat([item['input_ids'][0], 
                                    torch.zeros(input_padding, dtype=torch.long)])
            padded_output = torch.cat([item['output_ids'][0], 
                                     torch.zeros(output_padding, dtype=torch.long)])
            
            padded_batch.append({
                'input_ids': padded_input.unsqueeze(0),
                'output_ids': padded_output.unsqueeze(0),
                'input_text': item['input_text'],
                'output_text': item['output_text']
            })
        
        batches.append(padded_batch)
    
    return batches

def get_sample_activations(model, tokenizer, sample_text: str = "1 2 3 4 5") -> Dict:
    """Get activations for a single sample text."""
    # Tokenize input
    inputs = tokenizer(sample_text, return_tensors='pt')
    
    # Get activations (this will be implemented in transformer_utils.py)
    # For now, return placeholder
    return {
        'input_text': sample_text,
        'input_ids': inputs['input_ids'],
        'activations': None  # Will be filled by transformer_utils
    } 