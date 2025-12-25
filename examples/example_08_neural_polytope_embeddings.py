"""
Example: Neural Embeddings with Polytope Topology
==================================================

This example demonstrates how to use polytope topology models to extend
neural embeddings from standard matrices.

Inspired by Stella4D software and regular polytope theory, this approach
maps high-dimensional data to polytope structures for better geometric
preservation.
"""

import numpy as np
import matplotlib.pyplot as plt
from cans_gqs.neural_embeddings import (
    MatrixEmbedding,
    PolytopeEmbedding,
    EmbeddingConfig,
    RegularPolytopeGenerator,
    PlatonicSolids,
    RegularPolytopes4D,
    compare_embeddings,
    EmbeddingVisualizer,
    visualize_polytope_embedding_process,
)


def demo_3d_platonic_solids():
    """Demonstrate embeddings using 3D Platonic solids."""
    print("=" * 70)
    print("Demo 1: 3D Embeddings with Platonic Solids")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    data = np.random.randn(n_samples, n_features)
    
    # Add some structure (3 clusters)
    data[:30] += np.array([2, 2, 2, 0, 0, 0, 0, 0, 0, 0])
    data[30:60] += np.array([0, 0, 0, 2, 2, 2, 0, 0, 0, 0])
    data[60:] += np.array([0, 0, 0, 0, 0, 0, 2, 2, 2, 0])
    
    labels = np.array([0]*30 + [1]*30 + [2]*40)
    
    print(f"\nData shape: {data.shape}")
    print(f"Number of samples: {n_samples}")
    print(f"Original dimensionality: {n_features}")
    
    # Configure embeddings
    config = EmbeddingConfig(
        embedding_dim=3,
        num_points=n_samples,
        use_normalization=True,
        metric="euclidean",
    )
    
    # Standard matrix embedding (PCA)
    print("\n1. Creating standard matrix embedding (PCA)...")
    matrix_emb = MatrixEmbedding(config, method="pca")
    matrix_result = matrix_emb.fit_transform(data)
    print(f"   Matrix embedding shape: {matrix_result.shape}")
    
    # Polytope embeddings with different Platonic solids
    polytope_types = ["cube", "octahedron", "dodecahedron", "icosahedron"]
    
    for poly_type in polytope_types:
        print(f"\n2. Creating polytope embedding ({poly_type})...")
        
        # Generate polytope vertices
        generator = RegularPolytopeGenerator(3)
        vertices = generator.generate(poly_type)
        print(f"   Polytope: {poly_type}")
        print(f"   Number of vertices: {len(vertices)}")
        
        # Create polytope embedding
        polytope_emb = PolytopeEmbedding(
            config,
            polytope_type=poly_type,
            preserve_angles=True,
        )
        polytope_result = polytope_emb.fit_transform(data)
        print(f"   Polytope embedding shape: {polytope_result.shape}")
        
        # Visualize
        fig = EmbeddingVisualizer.plot_embedding_3d(
            polytope_result,
            labels=labels,
            title=f"Polytope Embedding ({poly_type})",
            polytope_vertices=vertices,
        )
        plt.show()
        
        # Compare embeddings
        print(f"\n3. Comparing embeddings for {poly_type}...")
        comparison = compare_embeddings(
            matrix_result,
            polytope_result,
            original_data=data,
            labels=labels,
        )
        
        print(f"   Distance correlation: {comparison['distance_correlation']:.4f}")
        print(f"   Matrix stress: {comparison['matrix_stress']:.4f}")
        print(f"   Polytope stress: {comparison['polytope_stress']:.4f}")
        print(f"   Stress improvement: {comparison['stress_improvement']*100:.2f}%")
        print(f"   Neighborhood preservation: {comparison['mean_neighborhood_preservation']:.4f}")


def demo_4d_regular_polytopes():
    """Demonstrate embeddings using 4D regular polytopes."""
    print("\n" + "=" * 70)
    print("Demo 2: 4D Embeddings with Regular Polytopes")
    print("=" * 70)
    
    # Generate synthetic 4D-structured data
    np.random.seed(42)
    n_samples = 80
    n_features = 20
    data = np.random.randn(n_samples, n_features)
    
    # Add 4D structure
    data[:20] += np.array([3, 0, 0, 0] + [0]*16)
    data[20:40] += np.array([0, 3, 0, 0] + [0]*16)
    data[40:60] += np.array([0, 0, 3, 0] + [0]*16)
    data[60:] += np.array([0, 0, 0, 3] + [0]*16)
    
    labels = np.array([0]*20 + [1]*20 + [2]*20 + [3]*20)
    
    print(f"\nData shape: {data.shape}")
    print(f"Embedding to 4D space using 4D polytopes...")
    
    # Configure 4D embeddings
    config = EmbeddingConfig(
        embedding_dim=4,
        num_points=n_samples,
        use_normalization=True,
        metric="euclidean",
    )
    
    # Test different 4D polytopes
    polytope_types_4d = [
        "5-cell",
        "tesseract",
        "16-cell",
        "24-cell",
    ]
    
    for poly_type in polytope_types_4d:
        print(f"\n--- {poly_type.upper()} ---")
        
        # Generate 4D polytope
        generator = RegularPolytopeGenerator(4)
        vertices = generator.generate(poly_type)
        info = generator.get_polytope_info(poly_type)
        
        print(f"Polytope: {info['type']}")
        print(f"Dimension: {info['dimension']}D")
        print(f"Number of vertices: {info['num_vertices']}")
        
        # Create polytope embedding
        polytope_emb = PolytopeEmbedding(
            config,
            polytope_type=poly_type,
            preserve_angles=True,
        )
        polytope_result = polytope_emb.fit_transform(data)
        print(f"Embedding shape: {polytope_result.shape}")
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        dists = pdist(polytope_result)
        print(f"Mean pairwise distance: {np.mean(dists):.4f}")
        print(f"Std pairwise distance: {np.std(dists):.4f}")


def demo_stella4d_inspired_120_cell():
    """
    Demonstrate the 120-cell polytope inspired by Stella4D software.
    
    The 120-cell is one of the six regular 4D polytopes and is the 4D analog
    of the dodecahedron. It has 600 vertices and 120 dodecahedral cells.
    """
    print("\n" + "=" * 70)
    print("Demo 3: Stella4D-Inspired 120-Cell Embeddings")
    print("=" * 70)
    print("\nThe 120-cell is a remarkable 4D polytope:")
    print("- 600 vertices")
    print("- 1200 edges")
    print("- 720 pentagonal faces")
    print("- 120 dodecahedral cells")
    print("- 4D analog of the dodecahedron")
    print("\nReference: https://www.software3d.com/120Cell.php")
    
    # Generate data
    np.random.seed(42)
    n_samples = 150
    n_features = 50
    data = np.random.randn(n_samples, n_features)
    
    # Create clusters
    n_clusters = 5
    samples_per_cluster = n_samples // n_clusters
    for i in range(n_clusters):
        start = i * samples_per_cluster
        end = start + samples_per_cluster
        offset = np.zeros(n_features)
        offset[i*10:(i+1)*10] = 3
        data[start:end] += offset
    
    print(f"\nData shape: {data.shape}")
    
    # Configure 4D embedding
    config = EmbeddingConfig(
        embedding_dim=4,
        num_points=n_samples,
        use_normalization=True,
        metric="geodesic",
    )
    
    # Generate 120-cell
    print("\nGenerating 120-cell structure...")
    generator = RegularPolytopeGenerator(4)
    vertices_120 = generator.generate("120-cell")
    info = generator.get_polytope_info("120-cell")
    
    print(f"\n120-cell properties:")
    print(f"  Vertices: {info['num_vertices']}")
    print(f"  Dimension: {info['dimension']}D")
    print(f"  Regular: {info['is_regular']}")
    
    # Create embedding
    print("\nCreating 120-cell embedding...")
    polytope_emb = PolytopeEmbedding(
        config,
        polytope_type="120-cell",
        preserve_angles=True,
    )
    embedding_result = polytope_emb.fit_transform(data)
    
    print(f"Embedding shape: {embedding_result.shape}")
    print(f"Embedding complete!")
    
    # Analyze structure
    from scipy.spatial.distance import pdist
    dists = pdist(embedding_result)
    print(f"\nEmbedding statistics:")
    print(f"  Mean pairwise distance: {np.mean(dists):.4f}")
    print(f"  Std pairwise distance: {np.std(dists):.4f}")
    print(f"  Min distance: {np.min(dists):.4f}")
    print(f"  Max distance: {np.max(dists):.4f}")


def demo_comparison_matrix_vs_polytope():
    """
    Comprehensive comparison between standard matrix embeddings
    and polytope-based embeddings.
    """
    print("\n" + "=" * 70)
    print("Demo 4: Comprehensive Comparison - Matrix vs Polytope Embeddings")
    print("=" * 70)
    
    # Generate structured data
    np.random.seed(42)
    n_samples = 100
    n_features = 30
    data = np.random.randn(n_samples, n_features)
    
    # Create 4 clusters
    data[:25] += np.array([3, 3, 3] + [0]*27)
    data[25:50] += np.array([0]*3 + [3, 3, 3] + [0]*24)
    data[50:75] += np.array([0]*6 + [3, 3, 3] + [0]*21)
    data[75:] += np.array([0]*9 + [3, 3, 3] + [0]*18)
    
    labels = np.array([0]*25 + [1]*25 + [2]*25 + [3]*25)
    
    print(f"\nData: {n_samples} samples, {n_features} features, 4 clusters")
    
    # Create both embeddings
    config = EmbeddingConfig(
        embedding_dim=3,
        num_points=n_samples,
        use_normalization=True,
    )
    
    print("\nCreating matrix embedding (PCA)...")
    matrix_emb = MatrixEmbedding(config, method="pca")
    matrix_result = matrix_emb.fit_transform(data)
    
    print("Creating polytope embedding (dodecahedron)...")
    polytope_emb = PolytopeEmbedding(
        config,
        polytope_type="dodecahedron",
        preserve_angles=True,
    )
    polytope_result = polytope_emb.fit_transform(data)
    
    # Compare
    print("\nComparing embeddings...")
    comparison = compare_embeddings(
        matrix_result,
        polytope_result,
        original_data=data,
        labels=labels,
    )
    
    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    print(f"Distance correlation: {comparison['distance_correlation']:.4f}")
    print(f"Distance MSE: {comparison['distance_mse']:.4f}")
    print(f"\nMatrix embedding stress: {comparison['matrix_stress']:.4f}")
    print(f"Polytope embedding stress: {comparison['polytope_stress']:.4f}")
    print(f"Stress improvement: {comparison['stress_improvement']*100:.2f}%")
    print(f"\nNeighborhood preservation: {comparison['mean_neighborhood_preservation']:.4f}")
    print(f"  (1.0 = perfect preservation, 0.0 = no preservation)")
    
    # Visualize process
    generator = RegularPolytopeGenerator(3)
    dodeca_vertices = generator.generate("dodecahedron")
    
    fig = visualize_polytope_embedding_process(
        data,
        matrix_result,
        polytope_result,
        dodeca_vertices,
    )
    plt.show()
    
    print("\n" + "="*50)
    print("Comparison complete! Check the visualization.")
    print("="*50)


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print(" NEURAL EMBEDDINGS WITH POLYTOPE TOPOLOGY MODELS")
    print(" Extending Standard Matrix Embeddings Using Regular Polytopes")
    print("="*70)
    print("\nInspired by:")
    print("  - Stella4D: https://www.software3d.com/StellaManual.php?prod=Stella4D")
    print("  - 120-cell: https://www.software3d.com/120Cell.php")
    print("  - PolyNav: https://www.software3d.com/PolyNav/PolyNavigator.php")
    print("\nThis framework integrates:")
    print("  1. CANS (Comprehensive Angular Naming System)")
    print("  2. Regular polytope theory (3D, 4D, nD)")
    print("  3. Neural embedding techniques")
    print("  4. Geometric structure preservation")
    
    # Run demonstrations
    try:
        demo_3d_platonic_solids()
    except Exception as e:
        print(f"\nError in demo 1: {e}")
    
    try:
        demo_4d_regular_polytopes()
    except Exception as e:
        print(f"\nError in demo 2: {e}")
    
    try:
        demo_stella4d_inspired_120_cell()
    except Exception as e:
        print(f"\nError in demo 3: {e}")
    
    try:
        demo_comparison_matrix_vs_polytope()
    except Exception as e:
        print(f"\nError in demo 4: {e}")
    
    print("\n" + "="*70)
    print("All demonstrations complete!")
    print("="*70)


if __name__ == "__main__":
    main()
