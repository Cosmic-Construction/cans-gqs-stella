"""
Topology Module for CANS-GQS

This module provides topology analysis tools including:
- Circle topology enumeration (from topcyc81)
- Dimensional analysis of topological structures
- Flip transformations for sphere embeddings
- Integration with polytope neural embeddings

The core insight: rooted trees in the plane map to free trees on the sphere,
analogous to how infinite hexagonal tilings in the plane map to finite
12-pentagonal faces on a dodecahedron.
"""

from .circle_topology import CircleTopology
from .flip_transforms import CircleExpression, find_flip_clusters
from .planar_spherical_bridge import (
    PlanarSphericalBridge,
    TopologyEmbeddingIntegrator,
    EulerCharacteristicAnalyzer,
)
from .dynamic_helix_embedding import (
    DynamicHelixEmbedding,
    HelixStrand,
)
from .nested_tree_cells import (
    OffByOneStructure,
    TwinPrimeMedian,
    NineNodeRootedTree,
    DodecahedralCell,
    OneTwentyCellNesting,
)

__all__ = [
    'CircleTopology',
    'CircleExpression',
    'find_flip_clusters',
    'PlanarSphericalBridge',
    'TopologyEmbeddingIntegrator',
    'EulerCharacteristicAnalyzer',
    'DynamicHelixEmbedding',
    'HelixStrand',
    'OffByOneStructure',
    'TwinPrimeMedian',
    'NineNodeRootedTree',
    'DodecahedralCell',
    'OneTwentyCellNesting',
]
