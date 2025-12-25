"""
CANS-nD: n-Dimensional Visualization Tools
Part 2 implementation - visualization and plotting

This module implements the NDVisualizer which provides:
- Direct plotting for 2D and 3D systems
- PCA-based visualization for high-dimensional systems
- Interactive plotting capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional

from .nd_primitives import NDVertex, NDEdge, NDFace, NDHyperface
from .nd_angular_system import NDAngularSystem


class NDVisualizer:
    """Visualization tools for n-dimensional geometry"""
    
    @staticmethod
    def plot_nd_system(system: NDAngularSystem, projection: str = 'pca',
                       figsize: tuple = (12, 10)) -> plt.Figure:
        """
        Plot n-dimensional system using dimensionality reduction.
        
        Parameters:
            system: NDAngularSystem to visualize
            projection: Projection method ('pca' or 'direct')
            figsize: Figure size in inches
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        
        if system.dimension <= 3:
            if system.dimension == 3:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)
            NDVisualizer._plot_low_dimensional(system, ax)
        else:
            NDVisualizer._plot_high_dimensional(system, fig, projection)
        
        plt.title(f"{system.dimension}D Geometric System")
        plt.tight_layout()
        return fig
    
    @staticmethod
    def _plot_low_dimensional(system: NDAngularSystem, ax):
        """
        Plot 2D or 3D system directly.
        
        Parameters:
            system: NDAngularSystem (2D or 3D)
            ax: Matplotlib axis
        """
        for label, primitive in system.primitives.items():
            coords = primitive.vertex_coordinates
            
            if system.dimension == 2:
                if isinstance(primitive, NDVertex):
                    ax.plot(coords[0, 0], coords[0, 1], 'ro', markersize=8, label=label)
                elif isinstance(primitive, NDEdge):
                    ax.plot(coords[:, 0], coords[:, 1], 'b-', alpha=0.7, linewidth=2)
                elif isinstance(primitive, NDFace):
                    # Close the polygon
                    closed_coords = np.vstack([coords, coords[0]])
                    ax.fill(closed_coords[:, 0], closed_coords[:, 1], 
                           alpha=0.3, edgecolor='k', linewidth=1)
            
            elif system.dimension == 3:
                if isinstance(primitive, NDVertex):
                    ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2],
                             c='r', s=100, label=label, depthshade=True)
                elif isinstance(primitive, NDEdge):
                    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                           'b-', alpha=0.7, linewidth=2)
                elif isinstance(primitive, NDFace):
                    poly = Poly3DCollection([coords], alpha=0.3, 
                                          facecolor='cyan', edgecolor='k')
                    ax.add_collection3d(poly)
                elif isinstance(primitive, NDHyperface):
                    # In 3D, hyperface is a 2D face
                    poly = Poly3DCollection([coords], alpha=0.3,
                                          facecolor='yellow', edgecolor='k')
                    ax.add_collection3d(poly)
        
        # Set labels
        if system.dimension == 2:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        elif system.dimension == 3:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.grid(True, alpha=0.3)
    
    @staticmethod
    def _plot_high_dimensional(system: NDAngularSystem, fig: plt.Figure,
                              projection: str):
        """
        Plot high-dimensional system using dimensionality reduction.
        
        Parameters:
            system: NDAngularSystem (dimension > 3)
            fig: Matplotlib figure
            projection: Projection method ('pca' or 'direct')
        """
        # Collect all coordinates
        all_coords = []
        labels = []
        types = []
        
        for label, primitive in system.primitives.items():
            coords = primitive.vertex_coordinates
            for coord in coords:
                all_coords.append(coord)
                labels.append(label)
                types.append(primitive.__class__.__name__)
        
        if len(all_coords) == 0:
            # No data to plot
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No primitives to visualize',
                   ha='center', va='center', fontsize=14)
            return
        
        all_coords = np.array(all_coords)
        
        # Apply dimensionality reduction
        if projection == 'pca':
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(3, all_coords.shape[0], all_coords.shape[1]))
                projected = pca.fit_transform(all_coords)
                
                # Pad to 3D if needed
                if projected.shape[1] < 3:
                    pad_width = ((0, 0), (0, 3 - projected.shape[1]))
                    projected = np.pad(projected, pad_width, mode='constant')
                
                xlabel, ylabel, zlabel = 'PC1', 'PC2', 'PC3'
                explained_var = pca.explained_variance_ratio_
                xlabel += f' ({explained_var[0]:.1%})'
                if len(explained_var) > 1:
                    ylabel += f' ({explained_var[1]:.1%})'
                if len(explained_var) > 2:
                    zlabel += f' ({explained_var[2]:.1%})'
            except ImportError:
                # Fallback: use first 3 dimensions
                projected = all_coords[:, :3]
                xlabel, ylabel, zlabel = 'X', 'Y', 'Z'
        else:
            # Direct projection to first 3 dimensions
            projected = all_coords[:, :3]
            xlabel, ylabel, zlabel = 'X', 'Y', 'Z'
        
        # Create 3D plot
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by type
        type_colors = {
            'NDVertex': 'red',
            'NDEdge': 'blue',
            'NDFace': 'green',
            'NDHyperface': 'purple',
        }
        
        # Plot points
        for i, (label, prim_type) in enumerate(zip(labels, types)):
            color = type_colors.get(prim_type, 'gray')
            # Only show label for first 10 points to avoid clutter
            point_label = label if i < 10 else None
            ax.scatter(projected[i, 0], projected[i, 1], projected[i, 2],
                      c=color, s=50, label=point_label, alpha=0.7,
                      depthshade=True)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.grid(True, alpha=0.3)
        
        # Add legend for primitive types
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=ptype)
                          for ptype, color in type_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right')
    
    @staticmethod
    def plot_polytope_4d(vertices_4d: np.ndarray, edges: Optional[list] = None,
                        projection_method: str = 'pca') -> plt.Figure:
        """
        Specialized visualization for 4D polytopes.
        
        Parameters:
            vertices_4d: Array of 4D vertex coordinates (N x 4)
            edges: Optional list of edge pairs [(i, j), ...]
            projection_method: Method for 4D to 3D projection
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(14, 10))
        
        # Project to 3D
        if projection_method == 'pca':
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3)
                vertices_3d = pca.fit_transform(vertices_4d)
                explained_var = pca.explained_variance_ratio_
            except ImportError:
                # Fallback: use first 3 dimensions
                vertices_3d = vertices_4d[:, :3]
                explained_var = [0, 0, 0]
        else:
            # Stereographic projection or other method
            vertices_3d = vertices_4d[:, :3]
            explained_var = [0, 0, 0]
        
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot vertices
        ax.scatter(vertices_3d[:, 0], vertices_3d[:, 1], vertices_3d[:, 2],
                  c='red', s=100, alpha=0.8, depthshade=True)
        
        # Label vertices
        for i, v in enumerate(vertices_3d):
            if i < 16:  # Limit labels
                ax.text(v[0], v[1], v[2], f' V{i}', fontsize=8)
        
        # Plot edges if provided
        if edges is not None:
            for i, j in edges:
                points = vertices_3d[[i, j]]
                ax.plot(points[:, 0], points[:, 1], points[:, 2],
                       'b-', alpha=0.4, linewidth=1)
        
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})' if projection_method == 'pca' else 'X')
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})' if projection_method == 'pca' else 'Y')
        ax.set_zlabel(f'PC3 ({explained_var[2]:.1%})' if projection_method == 'pca' else 'Z')
        ax.set_title('4D Polytope Projection')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_angular_distribution(system: NDAngularSystem, 
                                  angle_type: str = 'k_dihedral',
                                  bins: int = 30) -> plt.Figure:
        """
        Plot distribution of angles in the system.
        
        Parameters:
            system: NDAngularSystem
            angle_type: Type of angle to plot ('k_dihedral', 'solid', etc.)
            bins: Number of histogram bins
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # This is a placeholder - actual implementation would compute
        # angles based on the system's primitives
        angles = []
        
        # Plot histogram
        if len(angles) > 0:
            ax.hist(np.degrees(angles), bins=bins, alpha=0.7, 
                   edgecolor='black', color='skyblue')
            ax.set_xlabel('Angle (degrees)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{angle_type} Angle Distribution')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No angles computed yet',
                   ha='center', va='center', fontsize=14,
                   transform=ax.transAxes)
        
        plt.tight_layout()
        return fig
