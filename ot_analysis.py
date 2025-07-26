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
    
    def __init__(self, method: str = "sinkhorn", normalize: bool = True, reg: float = 0.1):
        """
        Initialize OT analyzer.
        
        Args:
            method: OT method ('emd' for Earth Mover's Distance, 'sinkhorn' for Sinkhorn)
            normalize: Whether to normalize representations before computing OT
            reg: Regularization parameter for Sinkhorn
        """
        self.method = method
        self.normalize = normalize
        self.reg = reg
    
    def compute_ot_distance(self, X1: np.ndarray, X2: np.ndarray, 
                          method: str = None, return_plan: bool = False) -> Tuple[float, Optional[np.ndarray]]:
        """
        Compute Optimal Transport distance between two sets of points.
        
        Args:
            X1: First set of points (n_samples, n_features)
            X2: Second set of points (n_samples, n_features)
            method: OT method to use (overrides self.method if provided)
            return_plan: Whether to return the transport plan (T matrix)
        
        Returns:
            OT distance and optionally the transport plan
        """
        if method is None:
            method = self.method
        
        # Normalize if requested
        if self.normalize:
            X1 = self._normalize_representations(X1)
            X2 = self._normalize_representations(X2)
        
        # Compute cost matrix (squared Euclidean distances as per screenshots)
        M = euclidean_distances(X1, X2, squared=True)
        
        # Compute OT distance and plan
        if method == "emd":
            # Earth Mover's Distance
            a = np.ones(X1.shape[0]) / X1.shape[0]  # Uniform weights
            b = np.ones(X2.shape[0]) / X2.shape[0]  # Uniform weights
            if return_plan:
                T = ot.emd(a, b, M)
                distance = np.sum(T * M)
                return distance, T
            else:
                distance = ot.emd2(a, b, M)
                return distance, None
        elif method == "sinkhorn":
            # Sinkhorn distance (regularized OT)
            a = np.ones(X1.shape[0]) / X1.shape[0]
            b = np.ones(X2.shape[0]) / X2.shape[0]
            if return_plan:
                T = ot.sinkhorn(a, b, M, reg=self.reg)
                distance = np.sum(T * M)
                return distance, T
            else:
                distance = ot.sinkhorn2(a, b, M, reg=self.reg)
                return distance, None
        else:
            raise ValueError(f"Unknown OT method: {method}")
    
    def analyze_map_sparsity(self, T: np.ndarray) -> Dict:
        """
        Analyze the sparsity of the transport plan (T matrix).
        
        Args:
            T: Transport plan matrix
        
        Returns:
            Dictionary with sparsity metrics
        """
        # Calculate sparsity metrics
        total_elements = T.size
        non_zero_elements = np.count_nonzero(T)
        sparsity_ratio = 1 - (non_zero_elements / total_elements)
        
        # Calculate entropy of the transport plan
        T_normalized = T / np.sum(T)
        entropy = -np.sum(T_normalized * np.log(T_normalized + 1e-8))
        
        # Calculate maximum value per row/column (one-to-one mapping indicator)
        max_per_row = np.max(T, axis=1)
        max_per_col = np.max(T, axis=0)
        
        # Calculate how "concentrated" the mapping is
        concentration_score = np.mean(max_per_row) / (np.sum(T) / T.shape[0])
        
        return {
            'sparsity_ratio': sparsity_ratio,
            'entropy': entropy,
            'concentration_score': concentration_score,
            'non_zero_elements': non_zero_elements,
            'total_elements': total_elements,
            'is_sparse': sparsity_ratio > 0.8,  # Threshold for sparsity
            'is_concentrated': concentration_score > 0.5  # Threshold for concentration
        }
    
    def characterize_geometric_transformation(self, layer_distances: List[float]) -> Dict:
        """
        Characterize the geometric transformation based on OT distances.
        
        Args:
            layer_distances: List of OT distances between adjacent layers
        
        Returns:
            Dictionary with geometric characterization
        """
        if len(layer_distances) < 2:
            return {}
        
        # Calculate trends
        distances_array = np.array(layer_distances)
        
        # Check for compression (decreasing distances)
        compression_score = 0
        if len(distances_array) > 1:
            # Calculate how much distances decrease
            decreases = np.diff(distances_array)
            compression_score = -np.sum(decreases[decreases < 0]) / len(decreases)
        
        # Check for spreading/drift (high or increasing distances)
        spreading_score = np.mean(distances_array)
        drift_trend = np.corrcoef(range(len(distances_array)), distances_array)[0, 1]
        
        # Identify transformation phases
        early_layers = distances_array[:len(distances_array)//3] if len(distances_array) >= 3 else distances_array
        middle_layers = distances_array[len(distances_array)//3:2*len(distances_array)//3] if len(distances_array) >= 3 else []
        late_layers = distances_array[2*len(distances_array)//3:] if len(distances_array) >= 3 else []
        
        characterization = {
            'compression_score': compression_score,
            'spreading_score': spreading_score,
            'drift_trend': drift_trend,
            'mean_distance': np.mean(distances_array),
            'std_distance': np.std(distances_array),
            'max_distance': np.max(distances_array),
            'min_distance': np.min(distances_array),
            'early_layer_mean': np.mean(early_layers) if len(early_layers) > 0 else 0,
            'middle_layer_mean': np.mean(middle_layers) if len(middle_layers) > 0 else 0,
            'late_layer_mean': np.mean(late_layers) if len(late_layers) > 0 else 0,
            'is_compressing': compression_score > 0.1,
            'is_spreading': spreading_score > np.median(distances_array),
            'transformation_type': self._classify_transformation(distances_array)
        }
        
        return characterization
    
    def _classify_transformation(self, distances: np.ndarray) -> str:
        """Classify the type of geometric transformation."""
        if len(distances) < 2:
            return "insufficient_data"
        
        # Calculate trends
        early = distances[:len(distances)//3] if len(distances) >= 3 else distances
        late = distances[2*len(distances)//3:] if len(distances) >= 3 else distances
        
        early_mean = np.mean(early)
        late_mean = np.mean(late)
        
        if late_mean < early_mean * 0.7:
            return "compression"
        elif late_mean > early_mean * 1.3:
            return "spreading"
        else:
            return "stable"
    
    def _normalize_representations(self, X: np.ndarray) -> np.ndarray:
        """Normalize representations to unit norm."""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return X / norms
    
    def compute_layer_distances(self, layer_representations: List[np.ndarray], 
                              return_plans: bool = False) -> Tuple[List[float], Optional[List[np.ndarray]]]:
        """
        Compute OT distances between adjacent layers.
        
        Args:
            layer_representations: List of layer representations, each of shape (n_samples, n_features)
            return_plans: Whether to return transport plans
        
        Returns:
            List of OT distances and optionally transport plans
        """
        distances = []
        plans = [] if return_plans else None
        
        for i in range(len(layer_representations) - 1):
            X1 = layer_representations[i]
            X2 = layer_representations[i + 1]
            
            # Subsample if too many points (for computational efficiency)
            if X1.shape[0] > 1000:
                indices = np.random.choice(X1.shape[0], 1000, replace=False)
                X1 = X1[indices]
                X2 = X2[indices]
            
            if return_plans:
                distance, plan = self.compute_ot_distance(X1, X2, return_plan=True)
                distances.append(distance)
                plans.append(plan)
            else:
                distance, _ = self.compute_ot_distance(X1, X2)
                distances.append(distance)
        
        return distances, plans
    
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
                'rank': np.linalg.matrix_rank(reps),
                'condition_number': np.linalg.cond(reps)
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
        
        # Compute OT distances and plans
        distances, plans = self.compute_layer_distances(reduced_reps, return_plans=True)
        
        # Analyze map sparsity for each layer pair
        sparsity_analysis = {}
        if plans:
            for i, plan in enumerate(plans):
                sparsity_analysis[f'layer_{i}_to_{i+1}'] = self.analyze_map_sparsity(plan)
        
        # Detect phase transitions
        transitions = self.detect_phase_transitions(distances)
        
        # Characterize geometric transformation
        geometric_characterization = self.characterize_geometric_transformation(distances)
        
        # Compute statistics
        stats = self.compute_representation_statistics(layer_representations)
        
        # Compute entropy
        entropies = self.compute_entropy(layer_representations)
        
        return {
            'ot_distances': distances,
            'phase_transitions': transitions,
            'statistics': stats,
            'entropies': entropies,
            'num_layers': len(layer_representations),
            'sparsity_analysis': sparsity_analysis,
            'geometric_characterization': geometric_characterization
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
    ot_distances = ot_analyzer.compute_layer_distances(reduced_reps)[0]
    
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