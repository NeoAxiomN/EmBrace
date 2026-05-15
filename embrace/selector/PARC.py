import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import gc
import os
from typing import Tuple
import sklearn
import scipy


class PARCEvaluator(object):
    """
    PARC (Prediction-based Representational Similarity Analysis) evaluator
    for EEG foundation models.
    
    Evaluates transferability of foundation models to downstream tasks.
    """
    
    def __init__(self, n_dims: int = None):
        """
        Args:
            n_dims: Optional dimensionality reduction for features
        """
        self.n_dims = n_dims
    
    def feature_reduce(self, features: np.ndarray) -> np.ndarray:
        """Apply PCA dimensionality reduction if n_dims is specified."""
        if self.n_dims is None or str(self.n_dims) == 'None':
            return features
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.n_dims)
        return pca.fit_transform(features)
    
    def get_lowertri(self, rdm: np.ndarray) -> np.ndarray:
        """Extract lower triangular part of RDM."""
        num_conditions = rdm.shape[0]
        return rdm[np.triu_indices(num_conditions, 1)]
    
    def compute_parc(self, features: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
        """
        Compute PARC correlation score.
        
        Args:
            features: (N, D) feature array
            labels: (N,) label array
            num_classes: Number of classes in the dataset
            
        Returns:
            PARC correlation score (0-100)
        """
        # Reduce dimensionality if specified
        features = self.feature_reduce(features)
        
        # Standardize features
        scaler = sklearn.preprocessing.StandardScaler()
        features = scaler.fit_transform(features)
        
        # Convert labels to one-hot encoding
        if labels.ndim == 1:
            label_matrix = np.eye(num_classes)[labels]
        else:
            label_matrix = labels
        
        # Compute Representational Dissimilarity Matrices (RDMs)
        rdm_features = 1 - np.corrcoef(features)
        rdm_labels = 1 - np.corrcoef(label_matrix)
        
        # Extract lower triangular parts
        lt_rdm_features = self.get_lowertri(rdm_features)
        lt_rdm_labels = self.get_lowertri(rdm_labels)
        
        # Compute Spearman correlation
        correlation, _ = scipy.stats.spearmanr(lt_rdm_features, lt_rdm_labels)
        
        return correlation * 100
    


