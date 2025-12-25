"""
Base Embedding Classes
=======================

Defines the base classes for neural embeddings:
- BaseEmbedding: Abstract base class
- MatrixEmbedding: Standard Euclidean matrix-based embeddings
- PolytopeEmbedding: Polytope-topology-based embeddings
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""
    embedding_dim: int
    num_points: int
    use_normalization: bool = True
    metric: str = "euclidean"  # or "geodesic"
    
    
class BaseEmbedding(ABC):
    """
    Abstract base class for all embedding types.
    
    Embeddings map high-dimensional data to lower-dimensional representations
    while preserving important structural properties.
    """
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedding.
        
        Parameters:
            config: Embedding configuration
        """
        self.config = config
        self.embeddings: Optional[np.ndarray] = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, data: np.ndarray) -> 'BaseEmbedding':
        """
        Fit the embedding to data.
        
        Parameters:
            data: Input data of shape (n_samples, n_features)
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data to embedding space.
        
        Parameters:
            data: Input data of shape (n_samples, n_features)
            
        Returns:
            Embedded data of shape (n_samples, embedding_dim)
        """
        pass
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Parameters:
            data: Input data of shape (n_samples, n_features)
            
        Returns:
            Embedded data of shape (n_samples, embedding_dim)
        """
        self.fit(data)
        return self.transform(data)
    
    @abstractmethod
    def distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Compute distance between two points in embedding space.
        
        Parameters:
            point1: First point
            point2: Second point
            
        Returns:
            Distance between points
        """
        pass
    
    def pairwise_distances(self, points: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute pairwise distances in embedding space.
        
        Parameters:
            points: Points to compute distances for (default: self.embeddings)
            
        Returns:
            Distance matrix of shape (n_points, n_points)
        """
        if points is None:
            if self.embeddings is None:
                raise ValueError("No embeddings available")
            points = self.embeddings
            
        n = len(points)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = self.distance(points[i], points[j])
                distances[i, j] = dist
                distances[j, i] = dist
                
        return distances
    
    def get_neighbors(self, point: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors to a point.
        
        Parameters:
            point: Query point
            k: Number of neighbors
            
        Returns:
            (indices, distances) of k nearest neighbors
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available")
            
        distances = np.array([self.distance(point, emb) for emb in self.embeddings])
        indices = np.argsort(distances)[:k]
        
        return indices, distances[indices]


class MatrixEmbedding(BaseEmbedding):
    """
    Standard matrix-based embedding in Euclidean space.
    
    This is the baseline approach using standard linear algebra techniques
    like PCA, MDS, or random projections.
    """
    
    def __init__(self, config: EmbeddingConfig, method: str = "pca"):
        """
        Initialize matrix embedding.
        
        Parameters:
            config: Embedding configuration
            method: Embedding method ("pca", "mds", "random")
        """
        super().__init__(config)
        self.method = method
        self.projection_matrix: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        
    def fit(self, data: np.ndarray) -> 'MatrixEmbedding':
        """
        Fit embedding using specified method.
        
        Parameters:
            data: Input data of shape (n_samples, n_features)
            
        Returns:
            self
        """
        n_samples, n_features = data.shape
        
        if self.config.use_normalization:
            self.mean_ = np.mean(data, axis=0)
            centered_data = data - self.mean_
        else:
            self.mean_ = np.zeros(n_features)
            centered_data = data
            
        if self.method == "pca":
            # PCA using SVD
            cov = np.cov(centered_data.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            # Take top embedding_dim components
            self.projection_matrix = eigenvectors[:, :self.config.embedding_dim]
            
        elif self.method == "random":
            # Random projection
            self.projection_matrix = np.random.randn(
                n_features, self.config.embedding_dim
            )
            # Normalize columns
            self.projection_matrix /= np.linalg.norm(
                self.projection_matrix, axis=0
            )
            
        elif self.method == "mds":
            # Classical MDS (distance-preserving)
            # Compute distance matrix
            dist_matrix = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    dist = np.linalg.norm(data[i] - data[j])
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
                    
            # Double centering
            n = n_samples
            H = np.eye(n) - np.ones((n, n)) / n
            B = -0.5 * H @ (dist_matrix ** 2) @ H
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(B)
            idx = np.argsort(eigenvalues)[::-1]
            
            # Construct embedding
            top_eigenvalues = eigenvalues[idx[:self.config.embedding_dim]]
            top_eigenvectors = eigenvectors[:, idx[:self.config.embedding_dim]]
            
            self.embeddings = top_eigenvectors @ np.diag(np.sqrt(np.maximum(top_eigenvalues, 0)))
            
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data to embedding space.
        
        Parameters:
            data: Input data of shape (n_samples, n_features)
            
        Returns:
            Embedded data of shape (n_samples, embedding_dim)
        """
        if not self.is_fitted:
            raise ValueError("Embedding must be fitted before transform")
            
        if self.method == "mds":
            # MDS already computed embeddings during fit
            return self.embeddings
            
        if self.config.use_normalization:
            centered_data = data - self.mean_
        else:
            centered_data = data
            
        self.embeddings = centered_data @ self.projection_matrix
        return self.embeddings
    
    def distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Compute Euclidean distance between points.
        
        Parameters:
            point1: First point
            point2: Second point
            
        Returns:
            Euclidean distance
        """
        return np.linalg.norm(point1 - point2)


class PolytopeEmbedding(BaseEmbedding):
    """
    Polytope-topology-based embedding.
    
    Maps data to vertices and surface of a regular polytope, preserving
    geometric relationships through polytope structure.
    """
    
    def __init__(
        self,
        config: EmbeddingConfig,
        polytope_type: str = "tesseract",
        preserve_angles: bool = True,
    ):
        """
        Initialize polytope embedding.
        
        Parameters:
            config: Embedding configuration
            polytope_type: Type of regular polytope to use
            preserve_angles: Whether to preserve angular relationships (CANS)
        """
        super().__init__(config)
        self.polytope_type = polytope_type
        self.preserve_angles = preserve_angles
        self.polytope_vertices: Optional[np.ndarray] = None
        self.vertex_assignments: Optional[np.ndarray] = None
        
    def fit(self, data: np.ndarray) -> 'PolytopeEmbedding':
        """
        Fit embedding by mapping data to polytope structure.
        
        Parameters:
            data: Input data of shape (n_samples, n_features)
            
        Returns:
            self
        """
        # Generate polytope vertices (will be implemented in polytope_generators)
        from .polytope_generators import RegularPolytopeGenerator
        
        generator = RegularPolytopeGenerator(self.config.embedding_dim)
        self.polytope_vertices = generator.generate(self.polytope_type)
        
        # Map data points to polytope vertices using similarity
        n_samples = data.shape[0]
        n_vertices = len(self.polytope_vertices)
        
        # Compute similarity matrix between data and vertices
        # For now, use simple assignment based on distance in PCA space
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.config.embedding_dim)
        data_projected = pca.fit_transform(data)
        
        # Normalize to unit sphere
        data_projected /= np.linalg.norm(data_projected, axis=1, keepdims=True)
        
        # Assign to nearest polytope vertex
        self.vertex_assignments = np.zeros(n_samples, dtype=int)
        self.embeddings = np.zeros((n_samples, self.config.embedding_dim))
        
        for i, point in enumerate(data_projected):
            # Find nearest polytope vertex
            distances = np.linalg.norm(
                self.polytope_vertices - point, axis=1
            )
            nearest_vertex = np.argmin(distances)
            self.vertex_assignments[i] = nearest_vertex
            
            # Interpolate between point and nearest vertex
            alpha = 0.5  # Interpolation factor
            self.embeddings[i] = alpha * point + (1 - alpha) * self.polytope_vertices[nearest_vertex]
        
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data to polytope embedding space.
        
        Parameters:
            data: Input data of shape (n_samples, n_features)
            
        Returns:
            Embedded data of shape (n_samples, embedding_dim)
        """
        if not self.is_fitted:
            raise ValueError("Embedding must be fitted before transform")
            
        # Re-fit for new data (in practice, would use learned mapping)
        return self.fit(data).embeddings
    
    def distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Compute geodesic distance on polytope surface.
        
        Parameters:
            point1: First point
            point2: Second point
            
        Returns:
            Geodesic distance on polytope
        """
        if self.config.metric == "geodesic":
            # Geodesic distance on polytope surface
            # Approximate using graph distance between polytope vertices
            return self._geodesic_distance(point1, point2)
        else:
            # Euclidean distance
            return np.linalg.norm(point1 - point2)
    
    def _geodesic_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Compute geodesic distance on polytope surface.
        
        Uses graph shortest path on polytope edge graph.
        """
        # Simplified: use Euclidean for now
        # Full implementation would compute shortest path on polytope graph
        return np.linalg.norm(point1 - point2)
    
    def get_polytope_structure(self) -> Dict[str, Any]:
        """
        Get polytope structure information.
        
        Returns:
            Dictionary with polytope structure data
        """
        return {
            "type": self.polytope_type,
            "vertices": self.polytope_vertices,
            "dimension": self.config.embedding_dim,
            "num_vertices": len(self.polytope_vertices) if self.polytope_vertices is not None else 0,
        }
