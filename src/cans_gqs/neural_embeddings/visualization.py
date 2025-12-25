"""
Embedding Visualization
=======================

Tools for visualizing and comparing embeddings:
- 2D/3D visualization of embeddings
- Comparison between standard and polytope embeddings
- Quality metrics and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any, Tuple
from mpl_toolkits.mplot3d import Axes3D


class EmbeddingVisualizer:
    """
    Visualization tools for embeddings.
    """
    
    @staticmethod
    def plot_embedding_2d(
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "Embedding Visualization",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot 2D embedding.
        
        Parameters:
            embeddings: Embeddings (n_points, 2)
            labels: Optional labels for coloring points
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            scatter = ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                c=labels,
                cmap='viridis',
                alpha=0.6,
                s=50,
            )
            plt.colorbar(scatter, ax=ax, label='Labels')
        else:
            ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                alpha=0.6,
                s=50,
            )
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_embedding_3d(
        embeddings: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "3D Embedding Visualization",
        save_path: Optional[str] = None,
        polytope_vertices: Optional[np.ndarray] = None,
    ) -> plt.Figure:
        """
        Plot 3D embedding.
        
        Parameters:
            embeddings: Embeddings (n_points, 3)
            labels: Optional labels for coloring points
            title: Plot title
            save_path: Optional path to save figure
            polytope_vertices: Optional polytope vertices to overlay
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            scatter = ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                embeddings[:, 2],
                c=labels,
                cmap='viridis',
                alpha=0.6,
                s=50,
            )
            plt.colorbar(scatter, ax=ax, label='Labels')
        else:
            ax.scatter(
                embeddings[:, 0],
                embeddings[:, 1],
                embeddings[:, 2],
                alpha=0.6,
                s=50,
            )
        
        # Overlay polytope structure if provided
        if polytope_vertices is not None:
            ax.scatter(
                polytope_vertices[:, 0],
                polytope_vertices[:, 1],
                polytope_vertices[:, 2],
                c='red',
                marker='^',
                s=100,
                alpha=0.8,
                label='Polytope vertices',
            )
            ax.legend()
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_polytope_wireframe(
        polytope_vertices: np.ndarray,
        polytope_edges: Optional[List[Tuple[int, int]]] = None,
        title: str = "Polytope Structure",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot polytope wireframe structure.
        
        Parameters:
            polytope_vertices: Vertices (n_vertices, 3)
            polytope_edges: Edge connections
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Figure object
        """
        if polytope_vertices.shape[1] != 3:
            raise ValueError("Can only plot 3D polytopes")
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot vertices
        ax.scatter(
            polytope_vertices[:, 0],
            polytope_vertices[:, 1],
            polytope_vertices[:, 2],
            c='red',
            marker='o',
            s=100,
            alpha=0.8,
        )
        
        # Plot edges if provided
        if polytope_edges is not None:
            for i, j in polytope_edges:
                points = np.array([polytope_vertices[i], polytope_vertices[j]])
                ax.plot(
                    points[:, 0],
                    points[:, 1],
                    points[:, 2],
                    'b-',
                    alpha=0.6,
                    linewidth=1,
                )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_distance_preservation(
        original_distances: np.ndarray,
        embedded_distances: np.ndarray,
        title: str = "Distance Preservation",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot how well distances are preserved.
        
        Parameters:
            original_distances: Original pairwise distances
            embedded_distances: Embedded pairwise distances
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Figure object
        """
        # Flatten distance matrices (upper triangle)
        n = len(original_distances)
        orig_flat = []
        emb_flat = []
        
        for i in range(n):
            for j in range(i+1, n):
                orig_flat.append(original_distances[i, j])
                emb_flat.append(embedded_distances[i, j])
        
        orig_flat = np.array(orig_flat)
        emb_flat = np.array(emb_flat)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Scatter plot
        ax1.scatter(orig_flat, emb_flat, alpha=0.5, s=20)
        ax1.plot(
            [orig_flat.min(), orig_flat.max()],
            [orig_flat.min(), orig_flat.max()],
            'r--',
            label='Perfect preservation',
        )
        ax1.set_xlabel('Original Distance')
        ax1.set_ylabel('Embedded Distance')
        ax1.set_title('Distance Correlation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error distribution
        errors = np.abs(orig_flat - emb_flat)
        ax2.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Absolute Distance Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distance Error Distribution')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def compare_embeddings(
    matrix_embedding: np.ndarray,
    polytope_embedding: np.ndarray,
    original_data: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compare matrix and polytope embeddings.
    
    Parameters:
        matrix_embedding: Standard matrix embedding (n_points, dim)
        polytope_embedding: Polytope embedding (n_points, dim)
        original_data: Original high-dimensional data (optional)
        labels: Optional labels for analysis
        
    Returns:
        Dictionary with comparison metrics and visualizations
    """
    results = {}
    
    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    
    matrix_dist = squareform(pdist(matrix_embedding))
    polytope_dist = squareform(pdist(polytope_embedding))
    
    # Distance preservation metrics
    dist_corr = np.corrcoef(matrix_dist.flatten(), polytope_dist.flatten())[0, 1]
    dist_mse = np.mean((matrix_dist - polytope_dist) ** 2)
    
    results['distance_correlation'] = dist_corr
    results['distance_mse'] = dist_mse
    
    # If original data provided, compute stress
    if original_data is not None:
        orig_dist = squareform(pdist(original_data))
        
        # Stress for matrix embedding
        matrix_stress = np.sqrt(
            np.sum((orig_dist - matrix_dist) ** 2) / np.sum(orig_dist ** 2)
        )
        
        # Stress for polytope embedding
        polytope_stress = np.sqrt(
            np.sum((orig_dist - polytope_dist) ** 2) / np.sum(orig_dist ** 2)
        )
        
        results['matrix_stress'] = matrix_stress
        results['polytope_stress'] = polytope_stress
        results['stress_improvement'] = (matrix_stress - polytope_stress) / matrix_stress
    
    # Variance explained (if 2D or 3D)
    dim = matrix_embedding.shape[1]
    if dim <= 3:
        matrix_var = np.var(matrix_embedding, axis=0)
        polytope_var = np.var(polytope_embedding, axis=0)
        
        results['matrix_variance'] = matrix_var
        results['polytope_variance'] = polytope_var
    
    # Neighborhood preservation
    k = min(10, len(matrix_embedding) - 1)
    neighborhood_agreement = []
    
    for i in range(len(matrix_embedding)):
        # K nearest neighbors in matrix embedding
        matrix_neighbors = np.argsort(matrix_dist[i])[1:k+1]
        
        # K nearest neighbors in polytope embedding
        polytope_neighbors = np.argsort(polytope_dist[i])[1:k+1]
        
        # Compute overlap
        overlap = len(set(matrix_neighbors) & set(polytope_neighbors))
        neighborhood_agreement.append(overlap / k)
    
    results['mean_neighborhood_preservation'] = np.mean(neighborhood_agreement)
    results['neighborhood_preservation_std'] = np.std(neighborhood_agreement)
    
    # Create comparison visualizations
    if dim == 2:
        fig1 = EmbeddingVisualizer.plot_embedding_2d(
            matrix_embedding,
            labels=labels,
            title="Matrix Embedding (2D)",
        )
        results['matrix_plot'] = fig1
        
        fig2 = EmbeddingVisualizer.plot_embedding_2d(
            polytope_embedding,
            labels=labels,
            title="Polytope Embedding (2D)",
        )
        results['polytope_plot'] = fig2
        
    elif dim == 3:
        fig1 = EmbeddingVisualizer.plot_embedding_3d(
            matrix_embedding,
            labels=labels,
            title="Matrix Embedding (3D)",
        )
        results['matrix_plot'] = fig1
        
        fig2 = EmbeddingVisualizer.plot_embedding_3d(
            polytope_embedding,
            labels=labels,
            title="Polytope Embedding (3D)",
        )
        results['polytope_plot'] = fig2
    
    # Distance preservation plot
    if original_data is not None:
        fig3 = EmbeddingVisualizer.plot_distance_preservation(
            orig_dist,
            polytope_dist,
            title="Polytope Embedding Distance Preservation",
        )
        results['distance_preservation_plot'] = fig3
    
    return results


def visualize_polytope_embedding_process(
    original_data: np.ndarray,
    matrix_embedding: np.ndarray,
    polytope_embedding: np.ndarray,
    polytope_vertices: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize the embedding process step by step.
    
    Parameters:
        original_data: Original data (n_points, n_features)
        matrix_embedding: Matrix embedding (n_points, 2 or 3)
        polytope_embedding: Polytope embedding (n_points, 2 or 3)
        polytope_vertices: Polytope vertices
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
    dim = matrix_embedding.shape[1]
    
    if dim == 2:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original data (PCA projection if high-dim)
        from sklearn.decomposition import PCA
        if original_data.shape[1] > 2:
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(original_data)
        else:
            data_2d = original_data
        
        axes[0].scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.6)
        axes[0].set_title('Original Data (PCA)')
        axes[0].grid(True, alpha=0.3)
        
        # Matrix embedding
        axes[1].scatter(matrix_embedding[:, 0], matrix_embedding[:, 1], alpha=0.6)
        axes[1].set_title('Matrix Embedding')
        axes[1].grid(True, alpha=0.3)
        
        # Polytope embedding with polytope structure
        axes[2].scatter(polytope_embedding[:, 0], polytope_embedding[:, 1], alpha=0.6, label='Data')
        axes[2].scatter(
            polytope_vertices[:, 0],
            polytope_vertices[:, 1],
            c='red',
            marker='^',
            s=100,
            alpha=0.8,
            label='Polytope vertices',
        )
        axes[2].set_title('Polytope Embedding')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
    elif dim == 3:
        fig = plt.figure(figsize=(18, 6))
        
        # Original data (PCA projection if high-dim)
        from sklearn.decomposition import PCA
        if original_data.shape[1] > 3:
            pca = PCA(n_components=3)
            data_3d = pca.fit_transform(original_data)
        else:
            data_3d = original_data
        
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], alpha=0.6)
        ax1.set_title('Original Data (PCA)')
        
        # Matrix embedding
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(
            matrix_embedding[:, 0],
            matrix_embedding[:, 1],
            matrix_embedding[:, 2],
            alpha=0.6,
        )
        ax2.set_title('Matrix Embedding')
        
        # Polytope embedding
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(
            polytope_embedding[:, 0],
            polytope_embedding[:, 1],
            polytope_embedding[:, 2],
            alpha=0.6,
            label='Data',
        )
        ax3.scatter(
            polytope_vertices[:, 0],
            polytope_vertices[:, 1],
            polytope_vertices[:, 2],
            c='red',
            marker='^',
            s=100,
            alpha=0.8,
            label='Polytope vertices',
        )
        ax3.set_title('Polytope Embedding')
        ax3.legend()
    
    else:
        raise ValueError("Can only visualize 2D or 3D embeddings")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
