"""
Planar-Spherical Bridge: Connecting Circle Topology to Polytope Embeddings

This module explores the deep mathematical connections between:
1. Rooted trees in the plane ↔ Free trees on the sphere
2. Infinite hexagonal tilings ↔ Finite dodecahedral tessellations
3. Implications for polytope neural embedding tensors

Mathematical Foundation:
========================

The key insight comes from the Euler characteristic χ and curvature:

In the PLANE (χ = 0, zero curvature):
- Hexagonal tilings can extend infinitely
- Rooted trees have a distinguished "root" pointing to infinity
- Catalan number orderings are preserved

On the SPHERE (χ = 2, positive curvature):
- Hexagons cannot tile a sphere - you need exactly 12 pentagons (dodecahedron)
- The "root" has nowhere to point - trees become unrooted/free
- Flip transformations identify equivalent topologies

For NEURAL EMBEDDINGS:
- Planar embeddings (standard): Euclidean distance matrices
- Spherical embeddings (polytope): Geodesic distances on curved manifolds
- The "defect" of 12 pentagons corresponds to embedding capacity constraints
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from functools import lru_cache

from .circle_topology import CircleTopology


class EulerCharacteristicAnalyzer:
    """
    Analyzes topological structures through the lens of Euler characteristic.

    The Euler characteristic χ = V - E + F determines global topology:
    - Plane: χ = 1 (with point at infinity)
    - Sphere: χ = 2
    - Torus: χ = 0
    - n-dimensional polytopes: Euler-Poincaré formula

    This directly affects how trees and tilings behave on each surface.
    """

    # Fundamental polytope data: vertices, edges, faces
    PLATONIC_SOLIDS = {
        'tetrahedron': {'V': 4, 'E': 6, 'F': 4, 'face_type': 3},
        'cube': {'V': 8, 'E': 12, 'F': 6, 'face_type': 4},
        'octahedron': {'V': 6, 'E': 12, 'F': 8, 'face_type': 3},
        'dodecahedron': {'V': 20, 'E': 30, 'F': 12, 'face_type': 5},
        'icosahedron': {'V': 12, 'E': 30, 'F': 20, 'face_type': 3},
    }

    # 4D polytope Euler characteristic components
    POLYTOPES_4D = {
        '5-cell': {'V': 5, 'E': 10, 'F': 10, 'C': 5},        # χ = 0
        'tesseract': {'V': 16, 'E': 32, 'F': 24, 'C': 8},    # χ = 0
        '16-cell': {'V': 8, 'E': 24, 'F': 32, 'C': 16},      # χ = 0
        '24-cell': {'V': 24, 'E': 96, 'F': 96, 'C': 24},     # χ = 0
        '120-cell': {'V': 600, 'E': 1200, 'F': 720, 'C': 120},
        '600-cell': {'V': 120, 'E': 720, 'F': 1200, 'C': 600},
    }

    @classmethod
    def euler_characteristic_3d(cls, polytope_name: str) -> int:
        """Compute χ = V - E + F for 3D polytopes (always 2 for convex)."""
        if polytope_name not in cls.PLATONIC_SOLIDS:
            raise ValueError(f"Unknown 3D polytope: {polytope_name}")
        p = cls.PLATONIC_SOLIDS[polytope_name]
        return p['V'] - p['E'] + p['F']

    @classmethod
    def euler_characteristic_4d(cls, polytope_name: str) -> int:
        """Compute χ = V - E + F - C for 4D polytopes (always 0 for convex)."""
        if polytope_name not in cls.POLYTOPES_4D:
            raise ValueError(f"Unknown 4D polytope: {polytope_name}")
        p = cls.POLYTOPES_4D[polytope_name]
        return p['V'] - p['E'] + p['F'] - p['C']

    @classmethod
    def angular_defect(cls, polytope_name: str) -> float:
        """
        Compute the total angular defect (Descartes' theorem).

        For a convex polyhedron, the sum of angular defects at all vertices
        equals 4π (720 degrees). This is why you can't tile a sphere with
        hexagons alone - each hexagon has zero angular defect!

        The dodecahedron's 12 pentagonal faces provide the necessary
        angular defect that allows for a finite, closed surface.
        """
        if polytope_name not in cls.PLATONIC_SOLIDS:
            return 0.0

        p = cls.PLATONIC_SOLIDS[polytope_name]
        # Angular defect per vertex = 2π - (angle sum at vertex)
        # For regular polyhedra, defect = 4π / V
        return 4 * np.pi / p['V']

    @classmethod
    def curvature_density(cls, polytope_name: str) -> float:
        """
        Compute Gaussian curvature concentrated at vertices.

        On a smooth sphere, curvature is uniformly distributed.
        On a polyhedron, it's concentrated at vertices.
        This affects how embeddings preserve distances.
        """
        if polytope_name not in cls.PLATONIC_SOLIDS:
            return 0.0

        p = cls.PLATONIC_SOLIDS[polytope_name]
        defect_per_vertex = 4 * np.pi / p['V']
        return defect_per_vertex  # Gaussian curvature ≡ angular defect


class PlanarSphericalBridge:
    """
    Bridges planar and spherical topology for circle arrangements.

    Core principle: The transition from plane to sphere is analogous to
    "compactifying" infinity. This:

    1. Removes the distinguished "outside" region
    2. Makes all regions topologically equivalent
    3. Introduces flip symmetries that weren't present in the plane

    For trees:
    - Rooted tree (plane): root points to infinite outside
    - Free tree (sphere): no distinguished vertex, flip equivalence

    For tilings:
    - Hexagonal (plane): can extend infinitely, zero curvature
    - Dodecahedral (sphere): finite, 12 pentagons absorb curvature
    """

    def __init__(self):
        self.topology = CircleTopology()

    def dimensional_reduction_ratio(self, n: int) -> Dict[str, float]:
        """
        Compute the reduction ratios as we increase embedding dimension.

        The reduction represents how many planar topologies collapse into
        one when embedded on higher-dimensional surfaces.

        Args:
            n: Number of circles/nodes

        Returns:
            Dictionary of reduction ratios between dimensions
        """
        catalan = self.topology.catalan_number(n)
        rooted = self.topology.non_intersecting_circles(n)
        unrooted = self.topology.sphere_surface_clusters(n)
        hyper = self.topology.hypersphere_4d_clusters(n)

        return {
            '1D_to_2D': catalan / rooted if rooted > 0 else 1.0,
            '2D_to_3D': rooted / unrooted if unrooted > 0 else 1.0,
            '3D_to_4D': unrooted / hyper if hyper > 0 else 1.0,
            '1D_to_4D': catalan / hyper if hyper > 0 else 1.0,
            'values': {
                'catalan': catalan,
                'rooted': rooted,
                'unrooted': unrooted,
                'hypersphere': hyper,
            }
        }

    def hexagon_to_pentagon_defect(self) -> Dict[str, any]:
        """
        Analyze the hexagon → pentagon transformation on curved surfaces.

        In the plane: hexagons tile perfectly (interior angle = 120°, 3 meet)
        On sphere: need 12 pentagons (interior angle = 108°) to close surface

        This is the Goldberg polyhedron principle - you MUST introduce
        pentagonal "defects" to achieve closure.

        Relevance to embeddings: The 12 pentagons represent irreducible
        "singularities" where the embedding cannot preserve local flatness.
        """
        # Hexagon internal angle: 120° = 2π/3
        hex_angle = 2 * np.pi / 3
        # Pentagon internal angle: 108° = 3π/5
        pent_angle = 3 * np.pi / 5

        # At each vertex where faces meet:
        # Hexagons: 3 × 120° = 360° (flat, no curvature)
        # Pentagons: 3 × 108° = 324° (angular defect of 36°)

        hex_sum = 3 * hex_angle
        pent_sum = 3 * pent_angle

        defect_per_pentagon = 2 * np.pi - pent_sum  # ~36° = π/5

        # Gauss-Bonnet: Total defect = 4π for sphere
        # So we need 4π / (π/5) = 20 vertices with pentagonal defect
        # But dodecahedron has 12 pentagonal FACES meeting at 20 vertices

        return {
            'hexagon_interior_angle_rad': hex_angle,
            'pentagon_interior_angle_rad': pent_angle,
            'hexagon_vertex_sum': hex_sum,  # = 2π, flat
            'pentagon_vertex_sum': pent_sum,  # < 2π, curved
            'angular_defect_per_vertex': defect_per_pentagon,
            'required_pentagons': 12,  # Always exactly 12!
            'explanation': (
                "A hexagonal tiling has zero Gaussian curvature and extends "
                "infinitely. To close on a sphere (total curvature 4π), we must "
                "introduce exactly 12 pentagonal 'defects'. This is the "
                "dodecahedron - the spherical analogue of the hexagonal plane."
            ),
        }

    def tree_root_to_flip_equivalence(self, n: int) -> Dict[str, any]:
        """
        Analyze how rooted trees (plane) map to free trees (sphere).

        The "root" in a planar tree corresponds to the distinguished
        outside/infinite region. On a sphere, there is no "outside" -
        all regions are equivalent, so the root becomes arbitrary.

        The flip transformation explicitly shows this: flipping the
        plane (or viewing from the other side of the sphere) changes
        which region appears to be "outside".

        Args:
            n: Number of nodes in the tree

        Returns:
            Analysis of the root ↔ flip correspondence
        """
        rooted = self.topology.rooted_trees(n)
        unrooted = self.topology.unrooted_trees(n)

        # The ratio tells us how many rooted trees collapse per free tree
        symmetry_factor = rooted / unrooted if unrooted > 0 else 1

        return {
            'n_nodes': n,
            'rooted_trees': rooted,
            'free_trees': unrooted,
            'symmetry_factor': symmetry_factor,
            'interpretation': (
                f"On average, {symmetry_factor:.2f} rooted trees in the plane "
                f"correspond to each free tree on the sphere. This represents "
                f"the 'flip equivalence' from losing the distinguished root."
            ),
        }


class TopologyEmbeddingIntegrator:
    """
    Integrates topological analysis with polytope neural embeddings.

    The key insight for neural embeddings:

    1. PLANAR EMBEDDINGS (e.g., PCA, t-SNE):
       - Preserve rooted/ordered structure
       - Can represent infinite extent
       - No intrinsic curvature constraints
       - Distance preservation is "local" in flat space

    2. POLYTOPE EMBEDDINGS (e.g., dodecahedron, 120-cell):
       - Impose flip-equivalence structure
       - Force finite, closed topology
       - Curvature concentrated at vertices (pentagons)
       - Geodesic distances follow curved manifold

    IMPLICATION: Polytope embeddings naturally quotient by symmetries
    that standard embeddings preserve. This can be DESIRABLE for:
    - Symmetry-invariant representations
    - Compact, finite embedding spaces
    - Natural clustering at high-curvature vertices
    """

    def __init__(self):
        self.bridge = PlanarSphericalBridge()
        self.euler = EulerCharacteristicAnalyzer()

    def embedding_capacity_analysis(self, polytope: str) -> Dict[str, any]:
        """
        Analyze the embedding capacity of a polytope.

        The number of vertices limits distinct embedding locations.
        The topology determines distance relationships.
        The curvature affects how well local structure is preserved.

        Args:
            polytope: Name of the polytope (e.g., 'dodecahedron', 'tesseract')

        Returns:
            Capacity analysis dictionary
        """
        if polytope in self.euler.PLATONIC_SOLIDS:
            data = self.euler.PLATONIC_SOLIDS[polytope]
            dim = 3
            chi = self.euler.euler_characteristic_3d(polytope)
            curvature = self.euler.curvature_density(polytope)
        elif polytope in self.euler.POLYTOPES_4D:
            data = self.euler.POLYTOPES_4D[polytope]
            dim = 4
            chi = self.euler.euler_characteristic_4d(polytope)
            curvature = 0  # Distributed differently in 4D
        else:
            return {'error': f'Unknown polytope: {polytope}'}

        return {
            'polytope': polytope,
            'dimension': dim,
            'vertices': data['V'],
            'edges': data['E'],
            'euler_characteristic': chi,
            'curvature_per_vertex': curvature,
            'embedding_capacity': data['V'],  # Max distinct points
            'unique_distances': data['E'],    # Edge-based distances
            'face_count': data.get('F', data.get('C', 0)),
            'topology_implications': self._topology_implications(polytope, dim),
        }

    def _topology_implications(self, polytope: str, dim: int) -> str:
        """Generate implications text for a polytope embedding."""
        if polytope == 'dodecahedron':
            return (
                "The dodecahedron's 12 pentagonal faces represent the minimal "
                "curvature defects needed to close a hexagonal-like structure. "
                "Points embedded near these faces cluster in high-curvature regions, "
                "analogous to how 12 'root positions' remain after quotienting "
                "by flip transformations."
            )
        elif polytope == 'icosahedron':
            return (
                "The icosahedron is dual to the dodecahedron - its 12 vertices "
                "correspond to the dodecahedron's 12 faces. Embedding here places "
                "points at the curvature concentration sites directly."
            )
        elif polytope == '120-cell':
            return (
                "The 120-cell is the 4D analogue of the dodecahedron: 120 "
                "dodecahedral cells arranged on a hypersphere. This provides "
                "~600 embedding positions with rich symmetry structure, analogous "
                "to how 4D circle topologies collapse under centrosymmetry."
            )
        elif polytope == 'tesseract':
            return (
                "The tesseract (4D hypercube) preserves the regularity of cubic "
                "structure in 4D. Its 16 vertices provide moderate embedding "
                "capacity with high symmetry."
            )
        else:
            return f"Standard {dim}D regular polytope with symmetric structure."

    def tensor_shape_implications(self,
                                  n_points: int,
                                  polytope: str) -> Dict[str, any]:
        """
        Analyze how topology affects tensor shapes in neural embeddings.

        The key tensors affected:
        1. Embedding matrix: (n_points, embedding_dim)
        2. Distance matrix: (n_points, n_points) - but topology constrains values
        3. Adjacency structure: Determined by polytope edges
        4. Curvature weighting: Non-uniform across embedding space

        Args:
            n_points: Number of points to embed
            polytope: Target polytope for embedding

        Returns:
            Tensor shape analysis
        """
        capacity = self.embedding_capacity_analysis(polytope)
        if 'error' in capacity:
            return capacity

        dim = capacity['dimension']
        n_vertices = capacity['vertices']
        n_edges = capacity['edges']

        # How many points per vertex on average?
        points_per_vertex = n_points / n_vertices

        # Distinct pairwise relationships determined by polytope
        unique_distance_classes = self._count_distance_classes(polytope)

        return {
            'embedding_tensor': {
                'shape': (n_points, dim),
                'description': f"{n_points} points in {dim}D polytope space",
            },
            'distance_tensor': {
                'shape': (n_points, n_points),
                'unique_classes': unique_distance_classes,
                'description': (
                    f"Pairwise distances collapse into {unique_distance_classes} "
                    f"distinct classes due to polytope symmetry"
                ),
            },
            'vertex_assignment_tensor': {
                'shape': (n_points,),
                'range': f"[0, {n_vertices-1}]",
                'avg_per_vertex': points_per_vertex,
            },
            'adjacency_tensor': {
                'shape': (n_vertices, n_vertices),
                'nonzero_entries': 2 * n_edges,  # Symmetric
                'sparsity': 1 - (2 * n_edges) / (n_vertices ** 2),
            },
            'topology_quotient': {
                'description': (
                    "The polytope embedding quotients embedding space by its "
                    "automorphism group. This is analogous to how sphere "
                    "embedding quotients rooted trees to free trees."
                ),
            },
        }

    def _count_distance_classes(self, polytope: str) -> int:
        """Count distinct distance classes for a polytope."""
        # For regular polytopes, distinct distances = floor((diameter + 1) / 2)
        distance_classes = {
            'tetrahedron': 1,    # All vertices equidistant
            'cube': 3,          # Edge, face diagonal, space diagonal
            'octahedron': 2,    # Edge, through-center
            'dodecahedron': 5,  # Multiple distance levels
            'icosahedron': 3,   # Three distance classes
            '5-cell': 1,        # All vertices equidistant
            'tesseract': 4,     # Four distance classes
            '16-cell': 2,       # Two distance classes
            '24-cell': 3,       # Three distance classes
            '120-cell': 15,     # Many distance classes
            '600-cell': 8,      # Multiple distance classes
        }
        return distance_classes.get(polytope, 1)

    def rooted_to_free_embedding_analogy(self, n: int) -> Dict[str, any]:
        """
        Explain the rooted→free tree analogy for embeddings.

        Key insight: Just as embedding circles on a sphere collapses
        rooted trees to free trees, embedding points on polytopes
        collapses certain distance configurations into equivalence classes.

        Args:
            n: Number of points/circles

        Returns:
            Analogy explanation with concrete numbers
        """
        analysis = self.bridge.tree_root_to_flip_equivalence(n + 1)
        hex_pent = self.bridge.hexagon_to_pentagon_defect()

        return {
            'n_points': n,
            'tree_analogy': analysis,
            'tiling_analogy': {
                'planar': "Infinite hexagonal tiling - no curvature constraints",
                'spherical': "Finite dodecahedral tiling - 12 pentagonal defects",
                'ratio': "12 pentagons : ∞ hexagons",
            },
            'embedding_implication': {
                'planar_embedding': (
                    f"PCA/t-SNE preserve all {analysis['rooted_trees']} "
                    "distinct configurations (like rooted trees)"
                ),
                'polytope_embedding': (
                    f"Dodecahedron embedding collapses to ~{analysis['free_trees']} "
                    "equivalence classes (like free trees)"
                ),
                'compression_factor': analysis['symmetry_factor'],
            },
            'neural_network_relevance': (
                "For neural networks, this means polytope embeddings provide "
                "natural symmetry-invariant representations. Points that differ "
                "only by 'rotation around infinity' (the planar equivalent of "
                "a flip transform) will be mapped to the same region."
            ),
        }


def demonstrate_bridge():
    """Demonstrate the planar-spherical bridge concepts."""
    print("=" * 70)
    print("PLANAR-SPHERICAL BRIDGE: Topology to Neural Embeddings")
    print("=" * 70)
    print()

    bridge = PlanarSphericalBridge()
    integrator = TopologyEmbeddingIntegrator()

    # 1. Dimensional reduction
    print("1. DIMENSIONAL REDUCTION RATIOS")
    print("-" * 40)
    for n in range(3, 8):
        ratios = bridge.dimensional_reduction_ratio(n)
        print(f"n={n}: Catalan({ratios['values']['catalan']}) → "
              f"Rooted({ratios['values']['rooted']}) → "
              f"Free({ratios['values']['unrooted']}) → "
              f"Hyper({ratios['values']['hypersphere']})")
    print()

    # 2. Hexagon to pentagon
    print("2. HEXAGON → PENTAGON (TILING CLOSURE)")
    print("-" * 40)
    hex_pent = bridge.hexagon_to_pentagon_defect()
    print(f"Hexagon interior angle: {np.degrees(hex_pent['hexagon_interior_angle_rad']):.1f}°")
    print(f"Pentagon interior angle: {np.degrees(hex_pent['pentagon_interior_angle_rad']):.1f}°")
    print(f"Angular defect per vertex: {np.degrees(hex_pent['angular_defect_per_vertex']):.1f}°")
    print(f"Required pentagons: {hex_pent['required_pentagons']}")
    print()

    # 3. Embedding capacity
    print("3. POLYTOPE EMBEDDING CAPACITY")
    print("-" * 40)
    for polytope in ['dodecahedron', 'icosahedron', '120-cell']:
        capacity = integrator.embedding_capacity_analysis(polytope)
        if 'error' not in capacity:
            print(f"{polytope}:")
            print(f"  Vertices: {capacity['vertices']}, Edges: {capacity['edges']}")
            print(f"  χ = {capacity['euler_characteristic']}")
    print()

    # 4. Tensor implications
    print("4. TENSOR SHAPE IMPLICATIONS (100 points)")
    print("-" * 40)
    tensors = integrator.tensor_shape_implications(100, 'dodecahedron')
    print(f"Embedding tensor: {tensors['embedding_tensor']['shape']}")
    print(f"Distance classes: {tensors['distance_tensor']['unique_classes']}")
    print(f"Points per vertex: {tensors['vertex_assignment_tensor']['avg_per_vertex']:.1f}")
    print()

    # 5. Neural embedding analogy
    print("5. ROOTED → FREE TREE ANALOGY FOR EMBEDDINGS")
    print("-" * 40)
    analogy = integrator.rooted_to_free_embedding_analogy(6)
    print(f"Planar configurations: {analogy['tree_analogy']['rooted_trees']}")
    print(f"Spherical equivalence classes: {analogy['tree_analogy']['free_trees']}")
    print(f"Compression factor: {analogy['embedding_implication']['compression_factor']:.2f}x")
    print()


if __name__ == "__main__":
    demonstrate_bridge()
