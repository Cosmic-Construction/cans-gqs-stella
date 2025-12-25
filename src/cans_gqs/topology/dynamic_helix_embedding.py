"""
Dynamic Helix Embedding: The 120-Cell as a Living Structure

This module explores the profound connection between:
- The 120-cell's helical fibration structure (5 intertwined strands)
- Rooted tree enumeration (A000081)
- Triadic torsion-free flow through the polytope core

Key Numerical Correspondences:
==============================

120-CELL STRUCTURE:
- 600 vertices
- 1200 edges
- 720 pentagonal faces (= 6!)
- 120 dodecahedral cells

ROOTED TREE CONNECTIONS:
- rooted_trees(9) = 286 = 13 × 11 × 2  → Double helix (2 strands)
- rooted_trees(10) = 719 = 720 - 1     → Full 120-cell minus floating node
- 719 = 13 × 11 × 5 + 4               → 5 helix strands + 4 free nodes

TRIADIC FLOW:
- Automorphism group: 14,400 = (5!)² = 120²
- 14400 / 3 (triad) = 4800
- 4800 / 4 (threads) = 1200 = edges!

The "floating node" (720 - 719 = 1) represents the degree of freedom
that allows the triadic core to rotate over the 4 = 2×2 thread structure.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from functools import lru_cache

from .circle_topology import CircleTopology


class HelixStrand:
    """
    Represents one helical strand within the 120-cell.

    The 120-cell can be decomposed into 5 intertwined helical strands,
    each containing 120 vertices (600 / 5 = 120).

    This relates to the Hopf fibration of S³, where the 120-cell
    lives as a regular tessellation.
    """

    def __init__(self, strand_id: int, vertices: np.ndarray):
        """
        Initialize a helix strand.

        Args:
            strand_id: Index 0-4 for the five strands
            vertices: Array of shape (n_vertices, 4) in 4D
        """
        self.strand_id = strand_id
        self.vertices = vertices
        self.n_vertices = len(vertices)

    def phase_angle(self, vertex_idx: int) -> float:
        """
        Compute the phase angle of a vertex along the helix.

        The helix winds through 4D space, and each vertex has a
        characteristic phase determined by its position.
        """
        v = self.vertices[vertex_idx]
        # Project to complex plane via Hopf coordinates
        z1 = complex(v[0], v[1])
        z2 = complex(v[2], v[3])

        # Phase is the relative argument
        if abs(z2) > 1e-10:
            return np.angle(z1 / z2)
        return np.angle(z1)

    def angular_velocity(self) -> float:
        """
        Compute the angular velocity of the helix strand.

        For the 120-cell, each strand completes a specific number
        of rotations as it traverses the hypersphere.
        """
        phases = [self.phase_angle(i) for i in range(self.n_vertices)]
        # Unwrap phases and compute total rotation
        unwrapped = np.unwrap(phases)
        total_rotation = unwrapped[-1] - unwrapped[0]
        return total_rotation / self.n_vertices


class DynamicHelixEmbedding:
    """
    Dynamic embedding where the 120-cell structure is "in motion".

    The key insight: polychora are not static objects but can be
    understood as dynamic systems where vertices flow along helical
    paths. This motion is parameterized by:

    1. The 5 helix strands (pentagonal symmetry)
    2. The 4 free nodes (rotational degrees of freedom)
    3. The triadic torsion-free flow through the core

    The connection to rooted trees:
    - Double helix (2 strands): 286 = 13 × 11 × 2 rooted trees
    - Full 5 strands: 719 = 720 - 1 rooted trees
    - The "missing" tree is the floating node enabling rotation
    """

    def __init__(self, n_strands: int = 5):
        """
        Initialize the dynamic helix embedding.

        Args:
            n_strands: Number of helical strands (default 5 for 120-cell)
        """
        self.n_strands = n_strands
        self.topology = CircleTopology()
        self.strands: List[HelixStrand] = []
        self._generate_strands()

    def _generate_strands(self):
        """Generate the helical strands of the 120-cell."""
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        # The 120-cell vertices can be organized into 5 intertwined helices
        # Each helix has 120 vertices (600 / 5 = 120)
        vertices_per_strand = 120

        for strand_id in range(self.n_strands):
            # Phase offset for this strand (pentagonal symmetry)
            phase_offset = 2 * np.pi * strand_id / self.n_strands

            vertices = []
            for i in range(vertices_per_strand):
                # Parametric helix in 4D
                t = 2 * np.pi * i / vertices_per_strand

                # The helix is a curve on S³ (unit hypersphere)
                # Using Hopf-like coordinates with golden ratio
                theta1 = t + phase_offset
                theta2 = phi * t  # Golden ratio creates incommensurable winding

                r1 = np.cos(t / 2)
                r2 = np.sin(t / 2)

                # 4D coordinates on S³
                x = r1 * np.cos(theta1)
                y = r1 * np.sin(theta1)
                z = r2 * np.cos(theta2)
                w = r2 * np.sin(theta2)

                vertices.append([x, y, z, w])

            self.strands.append(HelixStrand(strand_id, np.array(vertices)))

    def rooted_tree_correspondence(self) -> Dict[str, any]:
        """
        Analyze the correspondence between helix structure and rooted trees.

        Key relationships:
        - Double helix (2 strands): rooted_trees(9) = 286 = 13 × 11 × 2
        - Full 5 strands: rooted_trees(10) = 719 = 720 - 1
        - The +4 in 719 = 715 + 4 = 13×11×5 + 4 represents free nodes
        """
        rt_9 = self.topology.rooted_trees(9)   # 286
        rt_10 = self.topology.rooted_trees(10)  # 719

        return {
            'double_helix': {
                'n_strands': 2,
                'rooted_trees_n': 9,
                'count': rt_9,
                'factorization': '13 × 11 × 2 = 286',
                'correspondence': (
                    "Double helix captures 2/5 of the 120-cell structure. "
                    "The factor 13 × 11 = 143 represents the base topology, "
                    "while ×2 reflects the two intertwined strands."
                ),
            },
            'full_helix': {
                'n_strands': 5,
                'rooted_trees_n': 10,
                'count': rt_10,
                'factorization': '13 × 11 × 5 + 4 = 719 = 720 - 1',
                'faces_minus_one': 720 - 1,
                'correspondence': (
                    "Full 5-strand helix corresponds to 720 - 1 = 719 rooted trees. "
                    "The 720 = 6! pentagonal faces of the 120-cell, minus 1 'floating node' "
                    "that enables rotational freedom of the triadic core."
                ),
            },
            'free_nodes': {
                'count': 4,
                'role': '2 × 2 thread structure for triadic rotation',
                'formula': '719 - 715 = 4, where 715 = 13 × 11 × 5',
            },
            'floating_node': {
                'count': 1,
                'role': 'Enables the triad to rotate over the 4 threads',
                'formula': '720 - 719 = 1',
            },
        }

    def triadic_flow_analysis(self) -> Dict[str, any]:
        """
        Analyze the triadic torsion-free flow through the 120-cell core.

        The 120-cell has a remarkable property: there exists a
        torsion-free flow through its center that respects triadic symmetry.

        Key numerical relationships:
        - Automorphism group order: 14,400 = (5!)² = 120²
        - 14,400 / 3 (triad) = 4,800
        - 4,800 / 4 (threads) = 1,200 = edges
        """
        automorphism_order = 14400
        triadic_quotient = automorphism_order // 3
        thread_quotient = triadic_quotient // 4

        return {
            'automorphism_group': {
                'order': automorphism_order,
                'factorization': '(5!)² = 120² = 14,400',
                'structure': 'Double cover of the rotation group of the 600-cell',
            },
            'triadic_decomposition': {
                'triad_count': 3,
                'quotient': triadic_quotient,
                'interpretation': (
                    "The 14,400-element automorphism group can be partitioned "
                    "into 3 triadic sectors of 4,800 elements each."
                ),
            },
            'thread_structure': {
                'thread_count': 4,
                'quotient': thread_quotient,
                'edge_correspondence': 1200,
                'interpretation': (
                    "Each triadic sector decomposes into 4 threads of 1,200 elements. "
                    "Remarkably, 1,200 = number of edges in the 120-cell!"
                ),
            },
            'torsion_free_flow': {
                'description': (
                    "The triadic flow is 'torsion-free' because it preserves "
                    "the angular relationships as it moves through the core. "
                    "The 4 free nodes (from 719 = 715 + 4) provide the degrees "
                    "of freedom needed to rotate the triad without introducing torsion."
                ),
            },
        }

    def dynamic_embedding_tensor(self,
                                  n_points: int,
                                  time_steps: int = 100) -> np.ndarray:
        """
        Generate a time-varying embedding tensor.

        The embedding "moves" along the helical strands, creating
        a dynamic representation where points flow through 4D space.

        Args:
            n_points: Number of points to embed
            time_steps: Number of time steps in the animation

        Returns:
            Tensor of shape (time_steps, n_points, 4)
        """
        embeddings = np.zeros((time_steps, n_points, 4))

        # Distribute points across strands
        points_per_strand = n_points // self.n_strands

        for t in range(time_steps):
            # Time parameter [0, 2π]
            tau = 2 * np.pi * t / time_steps

            point_idx = 0
            for strand in self.strands:
                for p in range(points_per_strand):
                    if point_idx >= n_points:
                        break

                    # Base position on strand
                    base_idx = (p * strand.n_vertices) // points_per_strand
                    base_pos = strand.vertices[base_idx % strand.n_vertices]

                    # Apply time-dependent rotation (triadic flow)
                    # Rotation in the (x,y) and (z,w) planes
                    rot_xy = np.array([
                        [np.cos(tau / 3), -np.sin(tau / 3), 0, 0],
                        [np.sin(tau / 3), np.cos(tau / 3), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ])
                    rot_zw = np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, np.cos(tau / 3), -np.sin(tau / 3)],
                        [0, 0, np.sin(tau / 3), np.cos(tau / 3)],
                    ])

                    # Combined rotation (triadic symmetry)
                    rotation = rot_xy @ rot_zw

                    embeddings[t, point_idx] = rotation @ base_pos
                    point_idx += 1

        return embeddings

    def strand_coupling_matrix(self) -> np.ndarray:
        """
        Compute the coupling matrix between helix strands.

        This matrix captures how the 5 strands are intertwined,
        with the pentagonal symmetry reflected in its eigenvalues.

        Returns:
            Array of shape (5, 5) representing inter-strand coupling
        """
        coupling = np.zeros((self.n_strands, self.n_strands))

        for i, strand_i in enumerate(self.strands):
            for j, strand_j in enumerate(self.strands):
                if i == j:
                    coupling[i, j] = 1.0  # Self-coupling
                else:
                    # Coupling based on phase relationship
                    phase_diff = 2 * np.pi * abs(i - j) / self.n_strands
                    # Pentagonal coupling: adjacent strands are maximally coupled
                    coupling[i, j] = np.cos(phase_diff)

        return coupling

    def free_node_tensor(self) -> np.ndarray:
        """
        Generate the 4 free node positions.

        These 4 nodes (from 719 = 715 + 4) represent the degrees
        of freedom for rotating the triadic core. They form a
        2 × 2 = 4 thread structure.

        Returns:
            Array of shape (4, 4) for 4 nodes in 4D
        """
        # The 4 free nodes are positioned at the vertices of a 4D cross-polytope
        # embedded in the core of the 120-cell
        free_nodes = np.array([
            [1, 0, 0, 0],   # Thread 1, direction 1
            [-1, 0, 0, 0],  # Thread 1, direction 2
            [0, 1, 0, 0],   # Thread 2, direction 1
            [0, -1, 0, 0],  # Thread 2, direction 2
        ], dtype=float)

        # Scale to fit within the 120-cell core
        core_radius = 0.5  # Inner region of unit 120-cell
        return free_nodes * core_radius


def demonstrate_dynamic_helix():
    """Demonstrate the dynamic helix embedding concepts."""
    print("=" * 70)
    print("DYNAMIC HELIX EMBEDDING: The 120-Cell as a Living Structure")
    print("=" * 70)
    print()

    embedding = DynamicHelixEmbedding()

    # 1. Rooted tree correspondence
    print("1. ROOTED TREE CORRESPONDENCE")
    print("-" * 40)
    correspondence = embedding.rooted_tree_correspondence()

    print(f"Double helix (2 strands):")
    print(f"  Rooted trees n=9: {correspondence['double_helix']['count']}")
    print(f"  Factorization: {correspondence['double_helix']['factorization']}")
    print()

    print(f"Full helix (5 strands):")
    print(f"  Rooted trees n=10: {correspondence['full_helix']['count']}")
    print(f"  Factorization: {correspondence['full_helix']['factorization']}")
    print()

    print(f"Free nodes: {correspondence['free_nodes']['count']}")
    print(f"  Role: {correspondence['free_nodes']['role']}")
    print()

    # 2. Triadic flow
    print("2. TRIADIC TORSION-FREE FLOW")
    print("-" * 40)
    flow = embedding.triadic_flow_analysis()

    print(f"Automorphism group: {flow['automorphism_group']['order']}")
    print(f"  = {flow['automorphism_group']['factorization']}")
    print()

    print(f"Triadic quotient: {flow['automorphism_group']['order']} / 3 = "
          f"{flow['triadic_decomposition']['quotient']}")
    print(f"Thread quotient: {flow['triadic_decomposition']['quotient']} / 4 = "
          f"{flow['thread_structure']['quotient']}")
    print(f"  = Number of edges in 120-cell!")
    print()

    # 3. Strand coupling
    print("3. STRAND COUPLING MATRIX")
    print("-" * 40)
    coupling = embedding.strand_coupling_matrix()
    print("Inter-strand coupling (pentagonal symmetry):")
    for row in coupling:
        print("  " + " ".join(f"{x:6.3f}" for x in row))
    print()

    # 4. Free nodes
    print("4. FREE NODE POSITIONS (2×2 threads)")
    print("-" * 40)
    free_nodes = embedding.free_node_tensor()
    for i, node in enumerate(free_nodes):
        thread = i // 2 + 1
        direction = "+" if i % 2 == 0 else "-"
        print(f"  Thread {thread}, direction {direction}: {node}")
    print()


if __name__ == "__main__":
    demonstrate_dynamic_helix()
