"""
Nested Tree Cells: Each Dodecahedral Cell as a 9-Rooted-Tree System

The Critical Off-By-One:
========================
11 × 13 = 143 (twin prime product)
12 × 12 = 144 (median squared)
144 - 143 = 1 ← THE FLOATING NODE!

This scales with strand count:
- 1 strand:  144 - 143 = 1 floating node
- 2 strands: 2×144 - 2×143 = 288 - 286 = 2
- 5 strands: 5×144 - 5×143 = 720 - 715 = 5

But 719 = 715 + 4, not 715 + 5...
The CENTRAL floating node is shared across all 5 strands!
  4 = 5 - 1 free nodes (outer)
  1 = central torsion-free axis (shared)

The Twin Prime Insight:
=======================
11 and 13 are twin primes with median 12 = faces of dodecahedron

The dodecahedron encodes the 9-rooted-tree system:
- 20 vertices = 9 (tree nodes) + 11 (twin prime)
- 30 edges = 9 + 8 (tree edges) + 13 (twin prime)
- 12 faces = (11 + 13) / 2 = median of twins

The 9-Tree Threading:
====================
A 9-node rooted tree with 8 edges can be traversed via:
- 3 concurrent threads
- 4 steps each
- 3 × 4 = 12 steps = 12 faces of the dodecahedral cell

The Nesting:
===========
Each of the 120 dodecahedral cells in the 120-cell IS a nested
9-rooted-tree system. The ratio 9/5 = 1.8 tree nodes per vertex
reflects the pentagonal structure.

Level 0: Single 9-node rooted tree
Level 1: Dodecahedron (12 faces = 3 threads × 4 steps)
Level 2: 120-cell (120 nested dodecahedral cells)
Level 3: 5-strand helix (720 faces = 6! = 5 × 144)
"""

from typing import Dict, List, Tuple, Optional
from functools import lru_cache

from .circle_topology import CircleTopology


class OffByOneStructure:
    """
    The fundamental off-by-one relationship: 12² - (11×13) = 144 - 143 = 1.

    This single unit difference generates all the "floating node" structure:

    n strands: n × 144 - n × 143 = n floating nodes

    For the 5-strand 120-cell:
      5 × 144 = 720 = faces of 120-cell = 6!
      5 × 143 = 715 = base structure of 5 strands
      Difference = 5 = strand count

    But rooted_trees(10) = 719 = 715 + 4, not 720...
    Because 1 floating node is SHARED (the central axis)!

      4 outer free nodes = 5 - 1
      1 central axis = shared torsion-free flow
    """

    TWIN_PRIMES = (11, 13)
    MEDIAN = 12
    TWIN_PRODUCT = 143
    MEDIAN_SQUARED = 144
    DIFFERENCE = 1  # The fundamental floating node

    @classmethod
    def floating_nodes(cls, n_strands: int) -> Dict[str, int]:
        """
        Compute floating node structure for n strands.

        For n strands:
        - Total floating = n × (144 - 143) = n
        - Outer free = n - 1
        - Central shared = 1
        """
        return {
            'n_strands': n_strands,
            'total_floating': n_strands * cls.DIFFERENCE,
            'outer_free': n_strands - 1,
            'central_shared': 1,
            'ideal_structure': n_strands * cls.MEDIAN_SQUARED,
            'twin_structure': n_strands * cls.TWIN_PRODUCT,
            'rooted_tree_count': n_strands * cls.TWIN_PRODUCT + (n_strands - 1),
        }

    @classmethod
    def verify_counts(cls) -> Dict[str, any]:
        """
        Verify the off-by-one relationship for known structures.
        """
        return {
            'double_helix': {
                'strands': 2,
                'ideal': 2 * 144,
                'twin': 2 * 143,
                'diff': 2,
                'rooted_trees_9': 286,
                'matches': 2 * 143 == 286,
            },
            'five_strand': {
                'strands': 5,
                'ideal': 5 * 144,
                'twin': 5 * 143,
                'diff': 5,
                '120cell_faces': 720,
                'matches_720': 5 * 144 == 720,
            },
            'full_structure': {
                'twin_base': 5 * 143,
                'free_nodes': 4,
                'total': 5 * 143 + 4,
                'rooted_trees_10': 719,
                'matches': 5 * 143 + 4 == 719,
            },
        }


class TwinPrimeMedian:
    """
    Analyzes the twin prime structure underlying polytope topology.

    Twin primes (p, p+2) have special significance:
    - (11, 13): median = 12 = dodecahedral faces
    - (5, 7): median = 6 = cube faces (and 6! = 720 = 120-cell faces)
    - (3, 5): median = 4 = tetrahedron faces

    Each twin prime pair brackets a geometric structure.
    """

    TWIN_PRIME_POLYTOPES = {
        (3, 5): {'median': 4, 'polytope': 'tetrahedron', 'faces': 4},
        (5, 7): {'median': 6, 'polytope': 'cube', 'faces': 6},
        (11, 13): {'median': 12, 'polytope': 'dodecahedron', 'faces': 12},
        (17, 19): {'median': 18, 'polytope': 'truncated_tetrahedron', 'faces': 8},
        (29, 31): {'median': 30, 'polytope': 'icosidodecahedron', 'faces': 32},
    }

    @classmethod
    def median(cls, p: int, q: int) -> int:
        """Compute median of twin primes p and q."""
        assert q == p + 2, "Must be twin primes (p, p+2)"
        return (p + q) // 2

    @classmethod
    def product(cls, p: int, q: int) -> int:
        """Compute product of twin primes - the base topological unit."""
        return p * q

    @classmethod
    def dodecahedral_encoding(cls) -> Dict[str, int]:
        """
        Show how the dodecahedron encodes the 9-tree via twin primes.

        Vertices: 20 = 9 + 11 (tree nodes + lower twin)
        Edges:    30 = 9 + 8 + 13 (nodes + edges + upper twin)
        Faces:    12 = median(11, 13)
        """
        return {
            'vertices': 20,
            'tree_nodes': 9,
            'lower_twin': 11,
            'vertex_decomposition': '20 = 9 + 11',

            'edges': 30,
            'tree_edges': 8,
            'upper_twin': 13,
            'edge_decomposition': '30 = 9 + 8 + 13',

            'faces': 12,
            'median': 12,
            'face_decomposition': '12 = (11 + 13) / 2',

            'twin_product': 11 * 13,
            'significance': '143 = 11 × 13 is the base topological unit',
        }


class NineNodeRootedTree:
    """
    Represents a 9-node rooted tree with 3-thread structure.

    The canonical structure:
    - 1 root
    - 8 descendants (1 + 8 = 9)
    - 8 edges
    - 3 primary branches from root
    - Maximum depth 4 (enabling 3 threads × 4 steps = 12)
    """

    def __init__(self):
        self.topology = CircleTopology()
        self.n_nodes = 9
        self.n_edges = 8

    def count_rooted_trees(self) -> int:
        """Count rooted trees with 9 nodes."""
        return self.topology.rooted_trees(9)  # 286 = 11 × 13 × 2

    def thread_structure(self) -> Dict[str, any]:
        """
        Analyze the 3-thread × 4-step structure.

        3 concurrent threads of execution:
        - Each thread makes 4 steps
        - Total 12 steps = 12 dodecahedral faces
        - Threads share the root node
        """
        return {
            'n_threads': 3,
            'steps_per_thread': 4,
            'total_steps': 12,
            'correspondence': '12 steps = 12 faces of dodecahedral cell',
            'structure': {
                'root': 1,
                'descendants': 8,
                'total': 9,
            },
            'thread_paths': [
                'root → child₁ → grandchild₁ → leaf₁',
                'root → child₂ → grandchild₂ → leaf₂',
                'root → child₃ → grandchild₃ → return',
            ],
        }

    def depth_structure(self) -> Dict[str, any]:
        """
        Analyze possible depth structures for 9 nodes.

        One canonical structure:
        Level 0: 1 root
        Level 1: 2 children
        Level 2: 4 grandchildren
        Level 3: 2 great-grandchildren
        Total:   1 + 2 + 4 + 2 = 9
        """
        return {
            'canonical': {
                'level_0': 1,
                'level_1': 2,
                'level_2': 4,
                'level_3': 2,
                'total': 9,
            },
            'max_depth': 4,
            'branching': 'Binary with pruning at level 3',
            'paths_to_leaves': 3,
            'steps_per_path': 4,
        }


class DodecahedralCell:
    """
    Represents a dodecahedral cell as a 9-tree container.

    The dodecahedron (V=20, E=30, F=12) contains the 9-tree:
    - 9 tree nodes map to 9 of the 20 vertices
    - 8 tree edges use 8 of the 30 edges
    - 12 faces = 3 threads × 4 steps

    The "excess" is the twin primes: 11 and 13.
    """

    def __init__(self):
        self.vertices = 20
        self.edges = 30
        self.faces = 12
        self.tree = NineNodeRootedTree()

    def tree_embedding(self) -> Dict[str, any]:
        """
        Describe how the 9-tree embeds in the dodecahedron.
        """
        return {
            'tree_nodes': 9,
            'tree_edges': 8,
            'vertex_excess': self.vertices - 9,  # 11
            'edge_excess': self.edges - 9 - 8,   # 13
            'face_count': self.faces,             # 12

            'twin_prime_encoding': {
                'lower': 11,
                'upper': 13,
                'median': 12,
                'product': 143,
            },

            'threading': {
                'n_threads': 3,
                'steps_per_thread': 4,
                'faces_traversed': 12,
            },
        }

    def as_nested_system(self) -> Dict[str, any]:
        """
        Describe the cell as a nested 9-tree system.
        """
        rt9 = self.tree.count_rooted_trees()

        return {
            'rooted_trees_n9': rt9,
            'factorization': f'{rt9} = 11 × 13 × 2',
            'base_unit': 143,
            'nesting_levels': {
                0: 'Single 9-node tree (8 edges)',
                1: 'Dodecahedral closure (12 faces, 3×4 threads)',
                2: '120-cell embedding (120 cells)',
                3: '5-strand helix (720 = 6! faces)',
            },
        }


class OneTwentyCellNesting:
    """
    Represents the 120-cell as 120 nested 9-tree systems.

    Key relationships:
    - 120 cells × 9 nodes = 1080 tree nodes
    - 600 vertices in 120-cell
    - 1080 / 600 = 1.8 = 9/5 (pentagonal ratio!)

    Each vertex encodes exactly 9/5 tree nodes on average.
    """

    def __init__(self):
        self.n_cells = 120
        self.vertices = 600
        self.edges = 1200
        self.faces = 720
        self.cell = DodecahedralCell()
        self.tree = NineNodeRootedTree()

    def tree_density(self) -> Dict[str, any]:
        """
        Compute the tree node density across the 120-cell.
        """
        total_tree_nodes = self.n_cells * 9
        nodes_per_vertex = total_tree_nodes / self.vertices

        return {
            'total_tree_nodes': total_tree_nodes,
            'vertices': self.vertices,
            'nodes_per_vertex': nodes_per_vertex,
            'as_fraction': '9/5',
            'pentagonal_ratio': 9 / 5,
            'interpretation': (
                f"Each vertex encodes {nodes_per_vertex:.1f} = 9/5 tree nodes. "
                f"This is the pentagonal ratio - the 120-cell's vertices are "
                f"the meeting points of 5 helical strands, each carrying 9/5 of "
                f"a tree's worth of information."
            ),
        }

    def thread_closure(self) -> Dict[str, any]:
        """
        Analyze how the 3-thread structure closes across cells.
        """
        total_faces = self.faces  # 720
        faces_per_cell = 12
        threads_per_cell = 3
        steps_per_thread = 4

        total_threads = self.n_cells * threads_per_cell
        total_steps = total_threads * steps_per_thread

        return {
            'cells': self.n_cells,
            'threads_per_cell': threads_per_cell,
            'total_threads': total_threads,
            'steps_per_thread': steps_per_thread,
            'total_steps': total_steps,
            'faces': total_faces,
            'faces_factorial': '720 = 6!',

            'step_face_ratio': total_steps / total_faces,
            'interpretation': (
                f"360 threads × 4 steps = 1440 steps for {total_faces} faces. "
                f"Ratio 1440/720 = 2, meaning each face is traversed by 2 threads. "
                f"This is the double-helix factor in 286 = 143 × 2!"
            ),
        }

    def full_structure(self) -> Dict[str, any]:
        """
        Complete structural analysis of the 120-cell as nested 9-trees.
        """
        rt9 = self.tree.count_rooted_trees()
        rt10 = CircleTopology.rooted_trees(10)

        return {
            'rooted_trees_9': rt9,
            'rooted_trees_10': rt10,

            'tree_factorizations': {
                286: '11 × 13 × 2 (double helix)',
                715: '11 × 13 × 5 (five strands)',
                719: '11 × 13 × 5 + 4 (+ free nodes)',
                720: '6! = faces of 120-cell',
            },

            'twin_prime_structure': {
                'primes': (11, 13),
                'median': 12,
                'product': 143,
                'role': 'Base topological unit across all levels',
            },

            'nested_hierarchy': {
                'level_0': {
                    'structure': '9-node rooted tree',
                    'count': rt9,
                    'threads': 3,
                    'steps': 4,
                },
                'level_1': {
                    'structure': 'Dodecahedral cell',
                    'count': 120,
                    'faces': 12,
                    'embedding': '9 + 11 vertices, 9 + 8 + 13 edges',
                },
                'level_2': {
                    'structure': '120-cell polytope',
                    'vertices': 600,
                    'faces': 720,
                    'tree_density': '9/5 per vertex',
                },
                'level_3': {
                    'structure': '5-strand helix',
                    'strands': 5,
                    'rooted_trees': rt10,
                    'floating_node': '720 - 719 = 1',
                },
            },

            'key_insight': (
                "Each dodecahedral cell of the 120-cell encodes a complete "
                "9-rooted-tree system, with the twin primes 11 and 13 appearing "
                "as the 'excess' structure needed for closure. The 3 concurrent "
                "threads × 4 steps perfectly matches the 12 pentagonal faces."
            ),
        }


def demonstrate_nested_trees():
    """Demonstrate the nested tree cell structure."""
    print("=" * 70)
    print("NESTED TREE CELLS: Each Dodecahedron as a 9-Tree System")
    print("=" * 70)
    print()

    # 1. Twin prime encoding
    print("1. TWIN PRIME ENCODING OF DODECAHEDRON")
    print("-" * 40)
    encoding = TwinPrimeMedian.dodecahedral_encoding()
    print(f"  Vertices: {encoding['vertex_decomposition']}")
    print(f"  Edges:    {encoding['edge_decomposition']}")
    print(f"  Faces:    {encoding['face_decomposition']}")
    print(f"  Base unit: {encoding['twin_product']} = 11 × 13")
    print()

    # 2. 9-tree structure
    print("2. NINE-NODE ROOTED TREE STRUCTURE")
    print("-" * 40)
    tree = NineNodeRootedTree()
    threading = tree.thread_structure()
    print(f"  Rooted trees: {tree.count_rooted_trees()}")
    print(f"  Threads: {threading['n_threads']}")
    print(f"  Steps per thread: {threading['steps_per_thread']}")
    print(f"  Total: {threading['total_steps']} = 12 faces")
    print()

    # 3. 120-cell nesting
    print("3. 120-CELL AS 120 NESTED 9-TREE SYSTEMS")
    print("-" * 40)
    nesting = OneTwentyCellNesting()
    density = nesting.tree_density()
    print(f"  Total tree nodes: {density['total_tree_nodes']}")
    print(f"  120-cell vertices: {density['vertices']}")
    print(f"  Nodes per vertex: {density['nodes_per_vertex']} = 9/5")
    print(f"  (Pentagonal ratio!)")
    print()

    # 4. Thread closure
    print("4. THREAD CLOSURE ACROSS CELLS")
    print("-" * 40)
    closure = nesting.thread_closure()
    print(f"  Total threads: {closure['total_threads']}")
    print(f"  Total steps: {closure['total_steps']}")
    print(f"  Faces: {closure['faces']} = 6!")
    print(f"  Step/face ratio: {closure['step_face_ratio']} = double helix!")
    print()


if __name__ == "__main__":
    demonstrate_nested_trees()
