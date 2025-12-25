"""
Neural Embeddings with Polytope Topology
==========================================

This module extends standard neural embedding techniques by incorporating
polytope topology structures. Traditional neural embeddings use Euclidean
spaces (standard matrices), but this module enables embeddings on polytope
manifolds for better geometric structure preservation.

Key Features:
- Standard matrix-based embeddings (baseline)
- Polytope-structured embeddings (3D, 4D, nD)
- Regular polytope generators (Platonic solids, 4D polytopes, etc.)
- Geodesic distance metrics on polytope surfaces
- Angular relationship preservation using CANS framework

Inspired by:
- Stella4D software for 4D polytope visualization
- Regular polytope theory (120-cell, 600-cell, etc.)
- Geometric deep learning on manifolds
"""

from .base_embeddings import (
    BaseEmbedding,
    MatrixEmbedding,
    PolytopeEmbedding,
    EmbeddingConfig,
)
from .polytope_generators import (
    RegularPolytopeGenerator,
    PlatonicSolids,
    RegularPolytopes4D,
    PolytopeType3D,
    PolytopeType4D,
)
from .embedding_transformations import (
    EmbeddingTransformer,
    MatrixToPolytopeMapper,
    GeodesicDistanceMetric,
    AngularRelationshipPreserver,
)
from .visualization import (
    EmbeddingVisualizer,
    compare_embeddings,
    visualize_polytope_embedding_process,
)

__all__ = [
    "BaseEmbedding",
    "MatrixEmbedding",
    "PolytopeEmbedding",
    "EmbeddingConfig",
    "RegularPolytopeGenerator",
    "PlatonicSolids",
    "RegularPolytopes4D",
    "PolytopeType3D",
    "PolytopeType4D",
    "EmbeddingTransformer",
    "MatrixToPolytopeMapper",
    "GeodesicDistanceMetric",
    "AngularRelationshipPreserver",
    "EmbeddingVisualizer",
    "compare_embeddings",
    "visualize_polytope_embedding_process",
]
