"""
Optimal Transport analysis for comparing layer representations.
"""

import torch
import numpy as np
import ot
from typing import List, Tuple, Dict, Optional
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class OTAnalyzer:
    """Analyze layer representations using Optimal Transport."""
    
    def __init__(self, method: str = "emd", normalize: bool = True):
        """
        Initialize OT analyzer.
        
        Args:
            method: OT method ('emd' for Earth Mover's Distance, 'sinkhorn' for Sinkhorn)
            normalize: Whether to normalize representations before computing OT
        """
        self.method = method
        self.normalize = normalize
    
    def compute_ot_distance(self, X1: np.ndarray, X2: np.ndarray, 
                          method: str = None) -> float:
        """
        Compute Optimal Transport distance between two sets of points.
        
        Args:
            X1: First set of points (n_samples, n_features)
            X2: Second set of points (n_samples, n_features)
            method: OT method to use (overrides self.method if provided)
        
        Returns:
            OT distance
        """
        if method is None:
            method = self.method
        
        # Normalize if requested
        if self.normalize:
            X1 = self._normalize_representations(X1)
            X2 = self._normalize_representations(X2)
        
        # Compute cost matrix (Euclidean distances)
        M = euclidean_distances(X1, X2)
        
        # Compute OT distance
        if method == "emd":
            # Earth Mover's Distance
            a = np.ones(X1.shape[0]) / X1.shape[0]  # Uniform weights
            b = np.ones(X2.shape[0]) / X2.shape[0]  # Uniform weights
            distance = ot.emd2(a, b, M)
        elif method == "sinkhorn":
            # Sinkhorn distance (regularized OT)
            a = np.ones(X1.shape[0]) / X1.shape[0]
            b = np.ones(X2.shape[0]) / X2.shape[0]
            distance = ot.sinkhorn2(a, b, M, reg=0.1)
        else:
            raise ValueError(f"Unknown OT method: {method}")
        
        return distance
    
    def _normalize_representations(self, X: np.ndarray) -> np.ndarray:
        """Normalize representations to unit norm."""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return X / norms
    
    def compute_layer_distances(self, layer_representations: List[np.ndarray]) -> List[float]:
        """
        Compute OT distances between adjacent layers.
        
        Args:
            layer_representations: List of layer representations, each of shape (n_samples, n_features)
        
        Returns:
            List of OT distances between adjacent layers
        """
        distances = []
        
        for i in range(len(layer_representations) - 1):
            X1 = layer_representations[i]
            X2 = layer_representations[i + 1]
            
            # Subsample if too many points (for computational efficiency)
            if X1.shape[0] > 1000:
                indices = np.random.choice(X1.shape[0], 1000, replace=False)
                X1 = X1[indices]
                X2 = X2[indices]
            
            distance = self.compute_ot_distance(X1, X2)
            distances.append(distance)
        
        return distances
    
    def detect_phase_transitions(self, distances: List[float], 
                               threshold: float = 2.0) -> List[int]:
        """
        Detect phase transitions based on sudden changes in OT distances.
        
        Args:
            distances: List of OT distances between adjacent layers
            threshold: Threshold for detecting sudden changes (in standard deviations)
        
        Returns:
            List of layer indices where phase transitions occur
        """
        if len(distances) < 3:
            return []
        
        # Compute differences between consecutive distances
        diffs = np.diff(distances)
        
        # Compute rolling mean and std for adaptive thresholding
        window_size = min(5, len(diffs))
        
        # Use simple approach for small datasets
        if len(diffs) <= window_size:
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            z_scores = np.abs(diffs - mean_diff) / (std_diff + 1e-8)
        else:
            # Compute rolling statistics
            rolling_mean = np.convolve(diffs, np.ones(window_size)/window_size, mode='same')
            rolling_std = np.array([np.std(diffs[max(0, i-window_size//2):min(len(diffs), i+window_size//2+1)]) 
                                   for i in range(len(diffs))])
            z_scores = np.abs(diffs - rolling_mean) / (rolling_std + 1e-8)
        
        # Detect outliers
        transition_indices = np.where(z_scores > threshold)[0]
        
        return transition_indices.tolist()
    
    def compute_representation_statistics(self, representations: List[np.ndarray]) -> Dict:
        """
        Compute basic statistics for each layer's representations.
        
        Args:
            representations: List of layer representations
        
        Returns:
            Dictionary with statistics for each layer
        """
        stats = {}
        
        for i, reps in enumerate(representations):
            layer_stats = {
                'mean_norm': np.mean(np.linalg.norm(reps, axis=1)),
                'std_norm': np.std(np.linalg.norm(reps, axis=1)),
                'mean_activation': np.mean(reps),
                'std_activation': np.std(reps),
                'sparsity': np.mean(reps == 0),  # Fraction of zero activations
                'rank': np.linalg.matrix_rank(reps)
            }
            stats[f'layer_{i}'] = layer_stats
        
        return stats
    
    def reduce_dimensionality(self, representations: List[np.ndarray], 
                            n_components: int = 50) -> List[np.ndarray]:
        """
        Reduce dimensionality of representations using PCA.
        
        Args:
            representations: List of layer representations
            n_components: Number of PCA components to keep
        
        Returns:
            List of reduced representations
        """
        reduced_reps = []
        
        for reps in representations:
            if reps.shape[1] > n_components:
                # Adjust n_components if we have fewer samples
                actual_n_components = min(n_components, reps.shape[0] - 1, reps.shape[1])
                if actual_n_components < 1:
                    actual_n_components = 1
                
                pca = PCA(n_components=actual_n_components)
                reduced = pca.fit_transform(reps)
                reduced_reps.append(reduced)
            else:
                reduced_reps.append(reps)
        
        return reduced_reps
    
    def compute_entropy(self, representations: List[np.ndarray]) -> List[float]:
        """
        Compute entropy of each layer's representations.
        
        Args:
            representations: List of layer representations
        
        Returns:
            List of entropy values for each layer
        """
        entropies = []
        
        for reps in representations:
            # Normalize to create a probability distribution
            norms = np.linalg.norm(reps, axis=1)
            probs = norms / np.sum(norms)
            probs = probs[probs > 0]  # Remove zeros
            
            # Compute entropy
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            entropies.append(entropy)
        
        return entropies
    
    def analyze_representation_flow(self, layer_representations: List[np.ndarray]) -> Dict:
        """
        Comprehensive analysis of representation flow across layers.
        
        Args:
            layer_representations: List of layer representations
        
        Returns:
            Dictionary with comprehensive analysis results
        """
        # Reduce dimensionality for computational efficiency
        reduced_reps = self.reduce_dimensionality(layer_representations)
        
        # Compute OT distances
        distances = self.compute_layer_distances(reduced_reps)
        
        # Detect phase transitions
        transitions = self.detect_phase_transitions(distances)
        
        # Compute statistics
        stats = self.compute_representation_statistics(layer_representations)
        
        # Compute entropy
        entropies = self.compute_entropy(layer_representations)
        
        return {
            'ot_distances': distances,
            'phase_transitions': transitions,
            'statistics': stats,
            'entropies': entropies,
            'num_layers': len(layer_representations)
        }

def compute_simple_distance(X1: np.ndarray, X2: np.ndarray, 
                          metric: str = "euclidean") -> float:
    """
    Compute simple distance metrics between two sets of points.
    
    Args:
        X1: First set of points
        X2: Second set of points
        metric: Distance metric ('euclidean', 'cosine', 'manhattan')
    
    Returns:
        Distance value
    """
    if metric == "euclidean":
        return np.mean(euclidean_distances(X1, X2))
    elif metric == "cosine":
        from sklearn.metrics.pairwise import cosine_distances
        return np.mean(cosine_distances(X1, X2))
    elif metric == "manhattan":
        from sklearn.metrics.pairwise import manhattan_distances
        return np.mean(manhattan_distances(X1, X2))
    else:
        raise ValueError(f"Unknown metric: {metric}")

def compare_ot_vs_simple_distances(layer_representations: List[np.ndarray]) -> Dict:
    """
    Compare OT distances with simple distance metrics.
    
    Args:
        layer_representations: List of layer representations
    
    Returns:
        Dictionary with comparison results
    """
    ot_analyzer = OTAnalyzer()
    reduced_reps = ot_analyzer.reduce_dimensionality(layer_representations)
    
    # Compute OT distances
    ot_distances = ot_analyzer.compute_layer_distances(reduced_reps)
    
    # Compute simple distances
    simple_distances = {}
    for metric in ["euclidean", "cosine", "manhattan"]:
        distances = []
        for i in range(len(reduced_reps) - 1):
            dist = compute_simple_distance(reduced_reps[i], reduced_reps[i + 1], metric)
            distances.append(dist)
        simple_distances[metric] = distances
    
    return {
        'ot_distances': ot_distances,
        'simple_distances': simple_distances
    } 