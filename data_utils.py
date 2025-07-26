"""
Data utilities for GSM8K dataset and mathematical reasoning tasks.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from datasets import load_dataset
import re
from transformers import AutoTokenizer
import random

class GSM8KDataset:
    """Handle GSM8K dataset for mathematical reasoning."""
    
    def __init__(self, model_name: str = "gpt2", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load GSM8K dataset
        self.dataset = load_dataset("gsm8k", "main")
        
    def preprocess_gsm8k(self, split: str = "train", num_samples: Optional[int] = None) -> List[Dict]:
        """
        Preprocess GSM8K dataset for training.
        
        Args:
            split: Dataset split ('train' or 'test')
            num_samples: Number of samples to use (None for all)
        
        Returns:
            List of processed examples
        """
        data = self.dataset[split]
        
        if num_samples:
            data = data.select(range(min(num_samples, len(data))))
        
        processed_data = []
        
        for example in data:
            question = example['question']
            answer = example['answer']
            
            # Extract the final answer from the solution
            final_answer = self._extract_final_answer(answer)
            
            # Format as instruction-following task
            formatted_input = f"Question: {question}\nAnswer:"
            formatted_output = f" {answer}\nThe answer is {final_answer}."
            
            processed_data.append({
                'input_text': formatted_input,
                'output_text': formatted_output,
                'question': question,
                'answer': answer,
                'final_answer': final_answer
            })
        
        return processed_data
    
    def _extract_final_answer(self, answer: str) -> str:
        """Extract the final numerical answer from the solution."""
        # Look for patterns like "The answer is X" or "Therefore, X"
        patterns = [
            r"The answer is (\d+(?:\.\d+)?)",
            r"Therefore, (\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, answer)
            if match:
                return match.group(1)
        
        # If no pattern found, return the last number in the text
        numbers = re.findall(r'\d+(?:\.\d+)?', answer)
        if numbers:
            return numbers[-1]
        
        return "unknown"
    
    def tokenize_data(self, data: List[Dict]) -> List[Dict]:
        """Tokenize the processed data."""
        tokenized_data = []
        
        for example in data:
            # Tokenize input and output
            input_tokens = self.tokenizer.encode(
                example['input_text'], 
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )
            
            output_tokens = self.tokenizer.encode(
                example['output_text'],
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )
            
            # Combine input and output for training
            full_text = example['input_text'] + example['output_text']
            full_tokens = self.tokenizer.encode(
                full_text,
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            )
            
            tokenized_data.append({
                'input_ids': input_tokens,
                'output_ids': output_tokens,
                'full_ids': full_tokens,
                'input_text': example['input_text'],
                'output_text': example['output_text'],
                'question': example['question'],
                'answer': example['answer'],
                'final_answer': example['final_answer']
            })
        
        return tokenized_data
    
    def create_dataloader(self, tokenized_data: List[Dict], batch_size: int = 4) -> List[Dict]:
        """Create batches for training."""
        batches = []
        
        for i in range(0, len(tokenized_data), batch_size):
            batch = tokenized_data[i:i + batch_size]
            
            # Pad sequences to same length
            max_len = max(len(item['full_ids'][0]) for item in batch)
            
            padded_batch = []
            for item in batch:
                current_len = len(item['full_ids'][0])
                if current_len < max_len:
                    padding = torch.zeros(max_len - current_len, dtype=torch.long)
                    padded_ids = torch.cat([item['full_ids'][0], padding])
                else:
                    padded_ids = item['full_ids'][0]
                
                padded_batch.append({
                    'input_ids': padded_ids.unsqueeze(0),
                    'attention_mask': (padded_ids != 0).unsqueeze(0),
                    'labels': padded_ids.unsqueeze(0),
                    'input_text': item['input_text'],
                    'output_text': item['output_text'],
                    'question': item['question'],
                    'answer': item['answer'],
                    'final_answer': item['final_answer']
                })
            
            batches.append(padded_batch)
        
        return batches
    
    def get_sample_data(self, num_samples: int = 10) -> List[Dict]:
        """Get a small sample of data for quick testing."""
        return self.preprocess_gsm8k(split="train", num_samples=num_samples)

class ToyEmbeddingSimulator:
    """Simulate OT between toy embeddings for intuition building."""
    
    def __init__(self, dim: int = 64, num_points: int = 100):
        self.dim = dim
        self.num_points = num_points
    
    def generate_clusters(self, num_clusters: int = 3, cluster_std: float = 0.5) -> np.ndarray:
        """Generate clustered embeddings."""
        embeddings = []
        
        for i in range(num_clusters):
            # Generate cluster center
            center = np.random.randn(self.dim)
            center = center / np.linalg.norm(center)
            
            # Generate points around center
            cluster_points = np.random.randn(self.num_points // num_clusters, self.dim) * cluster_std
            cluster_points += center
            
            embeddings.append(cluster_points)
        
        return np.vstack(embeddings)
    
    def generate_shifted_distributions(self, shift_magnitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate two distributions with a known shift."""
        # Generate base distribution
        base = np.random.randn(self.num_points, self.dim)
        base = base / np.linalg.norm(base, axis=1, keepdims=True)
        
        # Generate shifted distribution
        shift = np.random.randn(self.dim)
        shift = shift / np.linalg.norm(shift) * shift_magnitude
        
        shifted = base + shift
        shifted = shifted / np.linalg.norm(shifted, axis=1, keepdims=True)
        
        return base, shifted
    
    def generate_compressed_distribution(self, compression_factor: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Generate distributions with different compression levels."""
        # Generate base distribution
        base = np.random.randn(self.num_points, self.dim)
        base = base / np.linalg.norm(base, axis=1, keepdims=True)
        
        # Generate compressed distribution
        compressed = base * compression_factor
        compressed = compressed / np.linalg.norm(compressed, axis=1, keepdims=True)
        
        return base, compressed

def format_gsm8k_prompt(question: str, answer: str = "") -> str:
    """Format GSM8K question for training/inference."""
    if answer:
        return f"Question: {question}\nAnswer: {answer}"
    else:
        return f"Question: {question}\nAnswer:"

def extract_math_answer(text: str) -> Optional[float]:
    """Extract numerical answer from text."""
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None 