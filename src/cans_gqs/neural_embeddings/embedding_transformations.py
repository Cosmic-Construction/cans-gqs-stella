"""
Embedding Transformations
==========================

Tools for transforming between embedding spaces:
- Matrix to polytope coordinate mappings
- Geodesic distance metrics on polytope surfaces
- Angular relationship preservation using CANS
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path


class EmbeddingTransformer:
    """
    Base class for embedding transformations.
    """
    
    def __init__(self, source_dim: int, target_dim: int):
        """
        Initialize transformer.
        
        Parameters:
            source_dim: Dimension of source embedding
            target_dim: Dimension of target embedding
        """
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.is_fitted = False
        
    def fit(self, source_embeddings: np.ndarray, target_embeddings: np.ndarray):
        """
        Fit transformation between embeddings.
        
        Parameters:
            source_embeddings: Source embedding matrix (n_samples, source_dim)
            target_embeddings: Target embedding matrix (n_samples, target_dim)
        """
        raise NotImplementedError
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings from source to target space.
        
        Parameters:
            embeddings: Embeddings in source space (n_samples, source_dim)
            
        Returns:
            Embeddings in target space (n_samples, target_dim)
        """
        raise NotImplementedError


class MatrixToPolytopeMapper:
    """
    Maps standard matrix embeddings to polytope-structured embeddings.
    
    This transformer takes points in standard Euclidean space and maps them
    to the surface or interior of a regular polytope.
    """
    
    def __init__(
        self,
        polytope_vertices: np.ndarray,
        mapping_method: str = "nearest_vertex",
        preserve_distances: bool = True,
    ):
        """
        Initialize mapper.
        
        Parameters:
            polytope_vertices: Vertices of target polytope (n_vertices, dim)
            mapping_method: Method for mapping ("nearest_vertex", "barycentric", "projection")
            preserve_distances: Whether to preserve pairwise distances
        """
        self.polytope_vertices = polytope_vertices
        self.mapping_method = mapping_method
        self.preserve_distances = preserve_distances
        self.dimension = polytope_vertices.shape[1]
        
    def map_to_polytope(self, points: np.ndarray) -> np.ndarray:
        """
        Map points to polytope structure.
        
        Parameters:
            points: Points in Euclidean space (n_points, dim)
            
        Returns:
            Points mapped to polytope (n_points, dim)
        """
        if self.mapping_method == "nearest_vertex":
            return self._map_nearest_vertex(points)
        elif self.mapping_method == "barycentric":
            return self._map_barycentric(points)
        elif self.mapping_method == "projection":
            return self._map_projection(points)
        else:
            raise ValueError(f"Unknown mapping method: {self.mapping_method}")
    
    def _map_nearest_vertex(self, points: np.ndarray) -> np.ndarray:
        """
        Map each point to its nearest polytope vertex.
        
        Parameters:
            points: Points to map (n_points, dim)
            
        Returns:
            Mapped points (n_points, dim)
        """
        # Normalize points to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized_points = points / norms
        
        # Find nearest vertex for each point
        distances = cdist(normalized_points, self.polytope_vertices)
        nearest_vertices = np.argmin(distances, axis=1)
        
        # Map to polytope vertices with interpolation
        mapped_points = np.zeros_like(points)
        for i, (point, vertex_idx) in enumerate(zip(normalized_points, nearest_vertices)):
            vertex = self.polytope_vertices[vertex_idx]
            
            # Interpolate between original point and vertex
            alpha = 0.3  # Blend factor (0 = vertex, 1 = original)
            mapped_points[i] = alpha * point + (1 - alpha) * vertex
            
        # Restore original norms if not preserving distances
        if not self.preserve_distances:
            mapped_points = mapped_points * norms
            
        return mapped_points
    
    def _map_barycentric(self, points: np.ndarray) -> np.ndarray:
        """
        Map points using barycentric coordinates relative to polytope.
        
        Parameters:
            points: Points to map (n_points, dim)
            
        Returns:
            Mapped points (n_points, dim)
        """
        # Simplified: use nearest k vertices
        k = min(self.dimension + 1, len(self.polytope_vertices))
        
        mapped_points = np.zeros_like(points)
        
        for i, point in enumerate(points):
            # Find k nearest vertices
            distances = np.linalg.norm(
                self.polytope_vertices - point, axis=1
            )
            nearest_k = np.argsort(distances)[:k]
            
            # Compute barycentric weights (inverse distance)
            weights = 1.0 / (distances[nearest_k] + 1e-8)
            weights /= weights.sum()
            
            # Weighted combination
            mapped_points[i] = np.sum(
                self.polytope_vertices[nearest_k] * weights[:, np.newaxis],
                axis=0
            )
        
        return mapped_points
    
    def _map_projection(self, points: np.ndarray) -> np.ndarray:
        """
        Project points onto polytope surface.
        
        Parameters:
            points: Points to map (n_points, dim)
            
        Returns:
            Mapped points (n_points, dim)
        """
        # Simplified: project onto convex hull
        # Full implementation would project onto actual polytope surface
        
        # Normalize to unit sphere (approximation of convex hull)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized_points = points / norms
        
        # Find nearest polytope facet and project
        # For now, use simple normalization
        mapped_points = normalized_points.copy()
        
        return mapped_points
    
    def inverse_map(self, polytope_points: np.ndarray) -> np.ndarray:
        """
        Map points from polytope back to Euclidean space.
        
        Parameters:
            polytope_points: Points on polytope (n_points, dim)
            
        Returns:
            Points in Euclidean space (n_points, dim)
        """
        # For most mappings, the inverse is the identity or projection
        return polytope_points


class GeodesicDistanceMetric:
    """
    Computes geodesic distances on polytope surfaces.
    
    Uses graph-based shortest path on polytope edge structure.
    """
    
    def __init__(self, polytope_vertices: np.ndarray, polytope_edges: Optional[List[Tuple[int, int]]] = None):
        """
        Initialize geodesic metric.
        
        Parameters:
            polytope_vertices: Vertices of polytope (n_vertices, dim)
            polytope_edges: List of edge connections as (i, j) pairs
        """
        self.polytope_vertices = polytope_vertices
        self.n_vertices = len(polytope_vertices)
        
        if polytope_edges is None:
            # Infer edges from proximity
            self.polytope_edges = self._infer_edges()
        else:
            self.polytope_edges = polytope_edges
            
        # Build adjacency graph
        self.adjacency_matrix = self._build_adjacency_matrix()
        
        # Compute all shortest paths
        self.distance_matrix = shortest_path(
            self.adjacency_matrix,
            directed=False,
            method='auto'
        )
    
    def _infer_edges(self) -> List[Tuple[int, int]]:
        """
        Infer edge connections from vertex positions.
        
        Assumes edges connect nearby vertices (k-nearest neighbors).
        
        Returns:
            List of edge pairs
        """
        # Compute pairwise distances
        distances = cdist(self.polytope_vertices, self.polytope_vertices)
        
        # For each vertex, connect to k nearest neighbors
        k = min(6, self.n_vertices - 1)  # Typical polytope connectivity
        edges = []
        
        for i in range(self.n_vertices):
            nearest = np.argsort(distances[i])[1:k+1]  # Exclude self
            for j in nearest:
                if i < j:  # Avoid duplicates
                    edges.append((i, j))
        
        return edges
    
    def _build_adjacency_matrix(self) -> np.ndarray:
        """
        Build weighted adjacency matrix from edges.
        
        Returns:
            Adjacency matrix (n_vertices, n_vertices)
        """
        adj = np.zeros((self.n_vertices, self.n_vertices))
        
        for i, j in self.polytope_edges:
            # Weight by Euclidean distance
            dist = np.linalg.norm(
                self.polytope_vertices[i] - self.polytope_vertices[j]
            )
            adj[i, j] = dist
            adj[j, i] = dist
        
        return adj
    
    def distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Compute geodesic distance between two points on polytope.
        
        Parameters:
            point1: First point (dim,)
            point2: Second point (dim,)
            
        Returns:
            Geodesic distance
        """
        # Find nearest vertices to each point
        distances1 = np.linalg.norm(
            self.polytope_vertices - point1, axis=1
        )
        distances2 = np.linalg.norm(
            self.polytope_vertices - point2, axis=1
        )
        
        vertex1 = np.argmin(distances1)
        vertex2 = np.argmin(distances2)
        
        # Look up precomputed geodesic distance
        geodesic_dist = self.distance_matrix[vertex1, vertex2]
        
        # Add local distances to nearest vertices
        local_dist1 = distances1[vertex1]
        local_dist2 = distances2[vertex2]
        
        return geodesic_dist + local_dist1 + local_dist2
    
    def pairwise_distances(self, points: np.ndarray) -> np.ndarray:
        """
        Compute pairwise geodesic distances.
        
        Parameters:
            points: Points on polytope (n_points, dim)
            
        Returns:
            Distance matrix (n_points, n_points)
        """
        n = len(points)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = self.distance(points[i], points[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances


class AngularRelationshipPreserver:
    """
    Preserves angular relationships using CANS framework during embedding.
    
    Integrates with existing CANS-3D and CANS-nD systems to maintain
    geometric structure in embeddings.
    """
    
    def __init__(self, dimension: int, use_cans: bool = True):
        """
        Initialize preserver.
        
        Parameters:
            dimension: Dimension of embedding space
            use_cans: Whether to use CANS angular computations
        """
        self.dimension = dimension
        self.use_cans = use_cans
        
        if use_cans:
            # Import CANS components
            try:
                from ..cans_nd.nd_angular_system import NDAngularSystem
                from ..cans_nd.nd_primitives import NDVertex, NDEdge
                
                self.angular_system = NDAngularSystem(dimension)
                self.NDVertex = NDVertex
                self.NDEdge = NDEdge
            except ImportError:
                print("Warning: CANS components not available, using fallback")
                self.use_cans = False
    
    def compute_angular_preservation(
        self,
        original_points: np.ndarray,
        embedded_points: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute metrics for angular relationship preservation.
        
        Parameters:
            original_points: Original points (n_points, original_dim)
            embedded_points: Embedded points (n_points, embed_dim)
            
        Returns:
            Dictionary of preservation metrics
        """
        n = len(original_points)
        
        if n < 3:
            return {"error": "Need at least 3 points"}
        
        # Compute angular differences for triplets
        angular_errors = []
        
        for i in range(min(n, 10)):  # Sample triplets
            for j in range(i+1, min(n, 10)):
                for k in range(j+1, min(n, 10)):
                    # Original angle
                    v1_orig = original_points[j] - original_points[i]
                    v2_orig = original_points[k] - original_points[i]
                    angle_orig = self._compute_angle(v1_orig, v2_orig)
                    
                    # Embedded angle
                    v1_emb = embedded_points[j] - embedded_points[i]
                    v2_emb = embedded_points[k] - embedded_points[i]
                    angle_emb = self._compute_angle(v1_emb, v2_emb)
                    
                    # Angular error
                    error = abs(angle_orig - angle_emb)
                    angular_errors.append(error)
        
        if not angular_errors:
            return {"error": "Could not compute angles"}
        
        return {
            "mean_angular_error": np.mean(angular_errors),
            "max_angular_error": np.max(angular_errors),
            "median_angular_error": np.median(angular_errors),
            "angular_preservation_ratio": 1.0 - np.mean(angular_errors) / np.pi,
        }
    
    def _compute_angle(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute angle between two vectors.
        
        Parameters:
            v1: First vector
            v2: Second vector
            
        Returns:
            Angle in radians
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return np.arccos(cos_angle)
    
    def optimize_embedding(
        self,
        embeddings: np.ndarray,
        target_angles: Dict[Tuple[int, int, int], float],
        learning_rate: float = 0.01,
        iterations: int = 100,
    ) -> np.ndarray:
        """
        Optimize embeddings to preserve target angular relationships.
        
        Parameters:
            embeddings: Initial embeddings (n_points, dim)
            target_angles: Dict of (i, j, k) -> angle for triplets
            learning_rate: Learning rate for optimization
            iterations: Number of optimization iterations
            
        Returns:
            Optimized embeddings
        """
        optimized = embeddings.copy()
        
        for _ in range(iterations):
            gradient = np.zeros_like(optimized)
            
            for (i, j, k), target_angle in target_angles.items():
                # Compute current angle
                v1 = optimized[j] - optimized[i]
                v2 = optimized[k] - optimized[i]
                current_angle = self._compute_angle(v1, v2)
                
                # Compute gradient (simplified)
                error = current_angle - target_angle
                
                # Update gradient (simplified gradient descent)
                gradient[i] += error * 0.01 * (v1 + v2)
                gradient[j] -= error * 0.01 * v1
                gradient[k] -= error * 0.01 * v2
            
            # Update embeddings
            optimized -= learning_rate * gradient
        
        return optimized
