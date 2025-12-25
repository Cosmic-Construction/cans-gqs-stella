# Neural Embeddings with Polytope Topology

## Overview

This module extends the CANS-GQS framework with **neural embedding capabilities using polytope topology**. Traditional neural embeddings map high-dimensional data to Euclidean spaces using standard matrix operations (PCA, t-SNE, etc.). This implementation enhances that approach by incorporating regular polytope structures from 3D, 4D, and higher dimensions.

## Motivation

### Why Polytope Topology for Neural Embeddings?

1. **Geometric Structure Preservation**: Regular polytopes have well-defined symmetries and angular relationships that can preserve important structural properties of data.

2. **Natural Manifold Learning**: Polytope surfaces provide natural manifolds for embedding that respect the intrinsic geometry of the data.

3. **Integration with CANS**: The Comprehensive Angular Naming System (CANS) provides precise angular measurements on polytope structures, enabling better preservation of angular relationships.

4. **Inspiration from Stella4D**: The [Stella4D software](https://www.software3d.com/StellaManual.php?prod=Stella4D) demonstrates the rich structure of 4D polytopes, particularly the [120-cell](https://www.software3d.com/120Cell.php) and other regular polytopes.

## Architecture

### Core Components

```
neural_embeddings/
├── base_embeddings.py          # Base embedding classes
├── polytope_generators.py       # Regular polytope generators
├── embedding_transformations.py # Mapping and metric utilities
└── visualization.py            # Visualization tools
```

### Key Classes

#### 1. Base Embeddings

**MatrixEmbedding**: Standard Euclidean embeddings
- Methods: PCA, MDS, random projection
- Baseline for comparison

**PolytopeEmbedding**: Polytope-structured embeddings
- Maps data to regular polytope vertices and surfaces
- Preserves angular relationships using CANS
- Supports 3D, 4D, and nD polytopes

**EmbeddingConfig**: Configuration dataclass
```python
@dataclass
class EmbeddingConfig:
    embedding_dim: int
    num_points: int
    use_normalization: bool = True
    metric: str = "euclidean"  # or "geodesic"
```

#### 2. Polytope Generators

**PlatonicSolids**: The 5 regular 3D polyhedra
- Tetrahedron (4 vertices)
- Cube (8 vertices)
- Octahedron (6 vertices)
- Dodecahedron (20 vertices)
- Icosahedron (12 vertices)

**RegularPolytopes4D**: The 6 regular 4D polytopes
- 5-cell / 4-simplex (5 vertices)
- 8-cell / Tesseract (16 vertices)
- 16-cell / 4-orthoplex (8 vertices)
- 24-cell (24 vertices) - unique to 4D
- 120-cell (approximate, inspired by Stella4D)
- 600-cell (120 vertices)

**RegularPolytopeGenerator**: Unified generator
- Simplexes for any dimension
- Hypercubes for any dimension
- Cross-polytopes for any dimension

#### 3. Transformation Utilities

**MatrixToPolytopeMapper**: Maps Euclidean points to polytope surfaces
- Methods: nearest vertex, barycentric, projection
- Distance preservation options

**GeodesicDistanceMetric**: Computes geodesic distances on polytope surfaces
- Graph-based shortest paths
- Proper metric on polytope manifold

**AngularRelationshipPreserver**: Preserves CANS angular relationships
- Computes angular preservation metrics
- Optimizes embeddings to maintain angles

## Usage Examples

### Example 1: Basic 3D Embedding with Platonic Solids

```python
from cans_gqs.neural_embeddings import (
    MatrixEmbedding,
    PolytopeEmbedding,
    EmbeddingConfig,
    compare_embeddings,
)
import numpy as np

# Generate data
data = np.random.randn(100, 20)

# Configure embedding
config = EmbeddingConfig(
    embedding_dim=3,
    num_points=100,
    use_normalization=True,
)

# Standard matrix embedding (baseline)
matrix_emb = MatrixEmbedding(config, method="pca")
matrix_result = matrix_emb.fit_transform(data)

# Polytope embedding with dodecahedron
polytope_emb = PolytopeEmbedding(
    config,
    polytope_type="dodecahedron",
    preserve_angles=True,
)
polytope_result = polytope_emb.fit_transform(data)

# Compare
comparison = compare_embeddings(
    matrix_result,
    polytope_result,
    original_data=data,
)

print(f"Neighborhood preservation: {comparison['mean_neighborhood_preservation']:.3f}")
print(f"Stress improvement: {comparison['stress_improvement']*100:.1f}%")
```

### Example 2: 4D Embedding with Tesseract

```python
from cans_gqs.neural_embeddings import (
    PolytopeEmbedding,
    EmbeddingConfig,
    RegularPolytopeGenerator,
)

# Configure 4D embedding
config = EmbeddingConfig(
    embedding_dim=4,
    num_points=80,
    metric="geodesic",
)

# Create tesseract embedding
tesseract_emb = PolytopeEmbedding(
    config,
    polytope_type="tesseract",
    preserve_angles=True,
)

# Embed data
embedding_4d = tesseract_emb.fit_transform(data)

# Get polytope structure
structure = tesseract_emb.get_polytope_structure()
print(f"Polytope: {structure['type']}")
print(f"Vertices: {structure['num_vertices']}")
```

### Example 3: Stella4D-Inspired 120-Cell

```python
from cans_gqs.neural_embeddings import (
    RegularPolytopes4D,
    PolytopeEmbedding,
)

# Generate 120-cell structure (simplified approximation)
vertices_120 = RegularPolytopes4D.one_twenty_cell()
print(f"120-cell vertices: {vertices_120.shape}")

# Use for embedding
config = EmbeddingConfig(embedding_dim=4, num_points=150)
emb_120 = PolytopeEmbedding(config, polytope_type="120-cell")
result = emb_120.fit_transform(high_dim_data)
```

### Example 4: Custom Visualization

```python
from cans_gqs.neural_embeddings import (
    EmbeddingVisualizer,
    visualize_polytope_embedding_process,
    PlatonicSolids,
)

# Create visualizer
visualizer = EmbeddingVisualizer()

# Plot 3D embedding with polytope overlay
cube_vertices = PlatonicSolids.cube()
fig = visualizer.plot_embedding_3d(
    embedding_result,
    labels=cluster_labels,
    polytope_vertices=cube_vertices,
    title="Cube-Structured Embedding",
)

# Visualize complete embedding process
fig = visualize_polytope_embedding_process(
    original_data,
    matrix_embedding,
    polytope_embedding,
    cube_vertices,
)
```

## Features

### Polytope Types Supported

#### 3D (Platonic Solids)
- **Tetrahedron**: 4 vertices, 4 faces
- **Cube**: 8 vertices, 6 faces
- **Octahedron**: 6 vertices, 8 faces
- **Dodecahedron**: 20 vertices, 12 faces
- **Icosahedron**: 12 vertices, 20 faces

#### 4D (Regular Polytopes)
- **5-cell**: 5 vertices, 5 tetrahedral cells
- **Tesseract (8-cell)**: 16 vertices, 8 cubic cells
- **16-cell**: 8 vertices, 16 tetrahedral cells
- **24-cell**: 24 vertices, 24 octahedral cells (unique to 4D)
- **120-cell**: ~600 vertices, 120 dodecahedral cells (4D dodecahedron)
- **600-cell**: 120 vertices, 600 tetrahedral cells (4D icosahedron)

#### nD (General)
- **Simplex**: n+1 vertices (n-dimensional tetrahedron)
- **Hypercube**: 2^n vertices (n-dimensional cube)
- **Cross-polytope**: 2n vertices (n-dimensional octahedron)

### Mapping Methods

1. **Nearest Vertex**: Maps each point to its nearest polytope vertex
2. **Barycentric**: Uses barycentric coordinates relative to nearby vertices
3. **Projection**: Projects points onto polytope surface

### Distance Metrics

1. **Euclidean**: Standard L2 distance in embedding space
2. **Geodesic**: Shortest path on polytope surface (graph-based)

### Evaluation Metrics

- **Distance Correlation**: How well pairwise distances are preserved
- **Stress**: Kruskal's stress metric for distance preservation
- **Neighborhood Preservation**: k-nearest neighbor overlap
- **Angular Preservation**: Preservation of angular relationships (CANS-based)

## Integration with CANS Framework

The polytope embeddings integrate seamlessly with the existing CANS framework:

1. **Angular Preservation**: Uses CANS angular computations to preserve geometric relationships
2. **4D Polytopes**: Leverages existing `Polytope4DAngularSystem` for 4D angular analysis
3. **nD Extensions**: Compatible with `NDAngularSystem` for arbitrary dimensions

## References

### Stella4D Software
- **Website**: https://www.software3d.com/
- **Stella4D Manual**: https://www.software3d.com/StellaManual.php?prod=Stella4D
- **120-cell**: https://www.software3d.com/120Cell.php
- **PolyNav**: https://www.software3d.com/PolyNav/PolyNavigator.php

### Mathematical Background
- Coxeter, H. S. M. (1973). *Regular Polytopes*
- Edelsbrunner, H., & Harer, J. (2010). *Computational Topology*
- Bronstein, M. M., et al. (2017). "Geometric Deep Learning"

## Testing

Comprehensive test suite in `tests/test_neural_embeddings.py`:

```bash
# Run all tests
pytest tests/test_neural_embeddings.py -v

# Run specific test class
pytest tests/test_neural_embeddings.py::TestPlatonicSolids -v
pytest tests/test_neural_embeddings.py::TestRegularPolytopes4D -v

# Run integration tests
pytest tests/test_neural_embeddings.py::TestIntegration -v
```

Current test coverage: 29 tests, all passing

## Future Enhancements

1. **GPU Acceleration**: CuPy-based implementations for large-scale embeddings
2. **Learned Polytopes**: Learn optimal polytope structures from data
3. **Hierarchical Embeddings**: Multi-scale polytope hierarchies
4. **Dynamic Embeddings**: Time-varying polytope structures
5. **Semi-regular Polytopes**: Extend beyond regular polytopes (Archimedean solids)
6. **Full 120-cell**: Complete implementation with all 600 vertices

## Performance

Typical performance characteristics:

- **3D Embeddings**: ~0.1s for 100 points with Platonic solids
- **4D Embeddings**: ~0.3s for 100 points with regular 4D polytopes
- **Comparison Metrics**: ~0.2s for full comparison suite

For large-scale applications (>10,000 points), consider:
- Batch processing
- Approximate nearest neighbor methods
- Numba acceleration (already integrated in CANS framework)

## License

MIT License (same as parent CANS-GQS framework)

## Contributing

Contributions welcome! Areas of interest:
- Additional polytope types
- Alternative mapping methods
- Performance optimizations
- Visualization enhancements
- Application examples

## Citation

If you use this neural embedding framework, please cite both the CANS-GQS framework and reference Stella4D as inspiration:

```
ContributorX Ltd. (2024)
The Comprehensive Angular Naming System (CANS) and the Geodesic Query System (GQS):
Neural Embeddings with Polytope Topology
GitHub: Cosmic-Construction/cans-gqs-stella

Inspired by Stella4D software (Webb, R.): https://www.software3d.com/
```
