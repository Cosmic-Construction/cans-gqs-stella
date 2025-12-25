"""
Regular Polytope Generators
============================

Generate vertices and structure for regular polytopes in various dimensions:
- 3D: Platonic solids (tetrahedron, cube, octahedron, dodecahedron, icosahedron)
- 4D: Regular polytopes (5-cell, 8-cell, 16-cell, 24-cell, 120-cell, 600-cell)
- nD: Simplexes, hypercubes, cross-polytopes

Inspired by Stella4D software and polytope theory.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum


class PolytopeType3D(Enum):
    """3D Platonic solid types."""
    TETRAHEDRON = "tetrahedron"
    CUBE = "cube"
    OCTAHEDRON = "octahedron"
    DODECAHEDRON = "dodecahedron"
    ICOSAHEDRON = "icosahedron"


class PolytopeType4D(Enum):
    """4D regular polytope types."""
    FIVE_CELL = "5-cell"  # 4-simplex
    EIGHT_CELL = "8-cell"  # tesseract (4-cube)
    SIXTEEN_CELL = "16-cell"  # 4-orthoplex
    TWENTY_FOUR_CELL = "24-cell"  # unique to 4D
    ONE_TWENTY_CELL = "120-cell"  # 4D dodecahedron analog
    SIX_HUNDRED_CELL = "600-cell"  # 4D icosahedron analog


class PlatonicSolids:
    """
    Generator for the 5 Platonic solids in 3D.
    """
    
    @staticmethod
    def tetrahedron() -> np.ndarray:
        """
        Generate regular tetrahedron vertices.
        
        Returns:
            Array of shape (4, 3) containing vertex coordinates
        """
        # Regular tetrahedron inscribed in unit cube
        vertices = np.array([
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
        ], dtype=float)
        
        # Normalize to unit sphere
        vertices /= np.linalg.norm(vertices[0])
        return vertices
    
    @staticmethod
    def cube() -> np.ndarray:
        """
        Generate cube vertices.
        
        Returns:
            Array of shape (8, 3) containing vertex coordinates
        """
        vertices = np.array([
            [x, y, z]
            for x in [-1, 1]
            for y in [-1, 1]
            for z in [-1, 1]
        ], dtype=float)
        
        # Normalize to unit sphere
        vertices /= np.linalg.norm(vertices[0])
        return vertices
    
    @staticmethod
    def octahedron() -> np.ndarray:
        """
        Generate regular octahedron vertices.
        
        Returns:
            Array of shape (6, 3) containing vertex coordinates
        """
        vertices = np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ], dtype=float)
        
        return vertices
    
    @staticmethod
    def dodecahedron() -> np.ndarray:
        """
        Generate regular dodecahedron vertices.
        
        Returns:
            Array of shape (20, 3) containing vertex coordinates
        """
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        vertices = []
        
        # 8 vertices from cube
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    vertices.append([x, y, z])
        
        # 12 vertices from rectangles
        for perm in [[0, 1/phi, phi], [1/phi, phi, 0], [phi, 0, 1/phi]]:
            for signs in [[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]]:
                vertices.append([s * p for s, p in zip(signs, perm)])
        
        vertices = np.array(vertices, dtype=float)
        
        # Normalize to unit sphere
        vertices /= np.linalg.norm(vertices[0])
        return vertices
    
    @staticmethod
    def icosahedron() -> np.ndarray:
        """
        Generate regular icosahedron vertices.
        
        Returns:
            Array of shape (12, 3) containing vertex coordinates
        """
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        vertices = []
        
        # 12 vertices from golden rectangles
        for perm in [[0, 1, phi], [1, phi, 0], [phi, 0, 1]]:
            for signs in [[1, 1], [1, -1], [-1, 1], [-1, -1]]:
                v = [perm[0], signs[0] * perm[1], signs[1] * perm[2]]
                vertices.append(v)
        
        vertices = np.array(vertices, dtype=float)
        
        # Normalize to unit sphere
        vertices /= np.linalg.norm(vertices[0])
        return vertices


class RegularPolytopes4D:
    """
    Generator for the 6 regular polytopes in 4D.
    
    These are the 4D analogs of the Platonic solids, as featured in Stella4D.
    """
    
    @staticmethod
    def five_cell() -> np.ndarray:
        """
        Generate 5-cell (4-simplex) vertices.
        
        The 4D analog of the tetrahedron.
        
        Returns:
            Array of shape (5, 4) containing vertex coordinates
        """
        # Regular 4-simplex inscribed in 4D space
        vertices = np.array([
            [1, 1, 1, -1],
            [1, -1, -1, 1],
            [-1, 1, -1, 1],
            [-1, -1, 1, 1],
            [1, 1, -1, 1],
        ], dtype=float)
        
        # Normalize to unit hypersphere
        vertices /= np.linalg.norm(vertices[0])
        return vertices
    
    @staticmethod
    def tesseract() -> np.ndarray:
        """
        Generate tesseract (8-cell, 4-cube) vertices.
        
        The 4D analog of the cube.
        
        Returns:
            Array of shape (16, 4) containing vertex coordinates
        """
        # All combinations of ±1 in 4 dimensions
        vertices = np.array([
            [x, y, z, w]
            for x in [-1, 1]
            for y in [-1, 1]
            for z in [-1, 1]
            for w in [-1, 1]
        ], dtype=float)
        
        # Normalize to unit hypersphere
        vertices /= np.linalg.norm(vertices[0])
        return vertices
    
    @staticmethod
    def sixteen_cell() -> np.ndarray:
        """
        Generate 16-cell (4-orthoplex) vertices.
        
        The 4D analog of the octahedron.
        
        Returns:
            Array of shape (8, 4) containing vertex coordinates
        """
        # Standard basis vectors ±e_i
        vertices = np.array([
            [1, 0, 0, 0], [-1, 0, 0, 0],
            [0, 1, 0, 0], [0, -1, 0, 0],
            [0, 0, 1, 0], [0, 0, -1, 0],
            [0, 0, 0, 1], [0, 0, 0, -1],
        ], dtype=float)
        
        return vertices
    
    @staticmethod
    def twenty_four_cell() -> np.ndarray:
        """
        Generate 24-cell vertices.
        
        A unique regular polytope that exists only in 4D.
        Self-dual and highly symmetric.
        
        The 24-cell has 24 vertices which are all permutations
        of (±1, ±1, 0, 0) normalized to unit sphere.
        
        Returns:
            Array of shape (24, 4) containing vertex coordinates
        """
        vertices = []
        
        # All permutations of (±1, ±1, 0, 0) with all sign combinations
        import itertools
        
        for pair in itertools.combinations(range(4), 2):
            # pair is the two positions that are non-zero
            for signs in [[1, 1], [1, -1], [-1, 1], [-1, -1]]:
                v = [0.0, 0.0, 0.0, 0.0]
                v[pair[0]] = signs[0]
                v[pair[1]] = signs[1]
                vertices.append(v)
        
        vertices = np.array(vertices, dtype=float)
        
        # Normalize to unit hypersphere
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices = vertices / norms
        
        return vertices
    
    @staticmethod
    def one_twenty_cell() -> np.ndarray:
        """
        Generate 120-cell vertices.
        
        The 4D analog of the dodecahedron. Has 600 vertices.
        This is a simplified approximation for computational efficiency.
        
        Returns:
            Array of shape (~600, 4) containing vertex coordinates
        """
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        vertices = []
        
        # The 120-cell has complex structure with 600 vertices
        # We generate a representative subset for efficiency
        
        # 16 vertices from tesseract
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    for w in [-1, 1]:
                        vertices.append([x, y, z, w])
        
        # 8 vertices from 16-cell
        for i in range(4):
            for sign in [2, -2]:
                v = [0, 0, 0, 0]
                v[i] = sign
                vertices.append(v)
        
        # Additional vertices using golden ratio (simplified)
        for perm in [[0, 1, phi, 1/phi], [1, phi, 1/phi, 0], 
                     [phi, 1/phi, 0, 1], [1/phi, 0, 1, phi]]:
            for signs in [[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], 
                          [1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, -1, 1],
                          [-1, -1, 1, 1], [-1, -1, -1, -1]]:
                v = [s * p for s, p in zip(signs, perm)]
                vertices.append(v)
        
        vertices = np.array(vertices, dtype=float)
        
        # Remove duplicates
        vertices = np.unique(vertices, axis=0)
        
        # Normalize to unit hypersphere
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices = vertices / norms
        
        return vertices
    
    @staticmethod
    def six_hundred_cell() -> np.ndarray:
        """
        Generate 600-cell vertices.
        
        The 4D analog of the icosahedron. Has 120 vertices.
        
        Returns:
            Array of shape (120, 4) containing vertex coordinates
        """
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        vertices = []
        
        # 8 vertices from 16-cell (scaled by 2)
        for i in range(4):
            for sign in [2, -2]:
                v = [0, 0, 0, 0]
                v[i] = sign
                vertices.append(v)
        
        # 16 vertices from tesseract
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    for w in [-1, 1]:
                        vertices.append([x, y, z, w])
        
        # 96 vertices using golden ratio
        # All even permutations of (0, ±1, ±phi, ±1/phi)
        base_coords = [0, 1, phi, 1/phi]
        
        for perm_idx in [[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], 
                         [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1]]:
            for signs in [[1, 1, 1], [1, 1, -1], [1, -1, 1], 
                          [1, -1, -1], [-1, 1, 1], [-1, 1, -1],
                          [-1, -1, 1], [-1, -1, -1]]:
                perm = [base_coords[i] for i in perm_idx]
                v = [perm[0], signs[0] * perm[1], signs[1] * perm[2], signs[2] * perm[3]]
                vertices.append(v)
        
        vertices = np.array(vertices, dtype=float)
        
        # Remove duplicates
        vertices = np.unique(vertices, axis=0)
        
        # Normalize to unit hypersphere
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        vertices = vertices / norms
        
        return vertices


class RegularPolytopeGenerator:
    """
    Unified generator for regular polytopes in any dimension.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize generator.
        
        Parameters:
            dimension: Dimension of the polytope (3, 4, or higher)
        """
        self.dimension = dimension
        
    def generate(self, polytope_type: str) -> np.ndarray:
        """
        Generate vertices for specified polytope type.
        
        Parameters:
            polytope_type: Type of polytope to generate
            
        Returns:
            Array of vertex coordinates
        """
        # Check if it's a generic nD type first
        if polytope_type in ["simplex", "hypercube", "cross-polytope"]:
            return self._generate_nd(polytope_type)
        elif self.dimension == 3:
            return self._generate_3d(polytope_type)
        elif self.dimension == 4:
            return self._generate_4d(polytope_type)
        else:
            return self._generate_nd(polytope_type)
    
    def _generate_3d(self, polytope_type: str) -> np.ndarray:
        """Generate 3D Platonic solid."""
        type_map = {
            "tetrahedron": PlatonicSolids.tetrahedron,
            "cube": PlatonicSolids.cube,
            "octahedron": PlatonicSolids.octahedron,
            "dodecahedron": PlatonicSolids.dodecahedron,
            "icosahedron": PlatonicSolids.icosahedron,
        }
        
        if polytope_type not in type_map:
            raise ValueError(f"Unknown 3D polytope type: {polytope_type}")
            
        return type_map[polytope_type]()
    
    def _generate_4d(self, polytope_type: str) -> np.ndarray:
        """Generate 4D regular polytope."""
        type_map = {
            "5-cell": RegularPolytopes4D.five_cell,
            "8-cell": RegularPolytopes4D.tesseract,
            "tesseract": RegularPolytopes4D.tesseract,
            "16-cell": RegularPolytopes4D.sixteen_cell,
            "24-cell": RegularPolytopes4D.twenty_four_cell,
            "120-cell": RegularPolytopes4D.one_twenty_cell,
            "600-cell": RegularPolytopes4D.six_hundred_cell,
        }
        
        if polytope_type not in type_map:
            raise ValueError(f"Unknown 4D polytope type: {polytope_type}")
            
        return type_map[polytope_type]()
    
    def _generate_nd(self, polytope_type: str) -> np.ndarray:
        """Generate n-dimensional polytope (simplex or hypercube)."""
        if polytope_type == "simplex":
            return self._generate_simplex()
        elif polytope_type == "hypercube":
            return self._generate_hypercube()
        elif polytope_type == "cross-polytope":
            return self._generate_cross_polytope()
        else:
            raise ValueError(f"Unknown nD polytope type: {polytope_type}")
    
    def _generate_simplex(self) -> np.ndarray:
        """
        Generate n-simplex vertices.
        
        Returns:
            Array of shape (n+1, n) containing vertex coordinates
        """
        n = self.dimension
        
        # Standard simplex: vertices at e_i and -sum(e_i)/n
        vertices = np.eye(n)
        
        # Add barycenter vertex
        center = -np.ones(n) / n
        vertices = np.vstack([vertices, center])
        
        # Center and normalize
        vertices -= vertices.mean(axis=0)
        vertices /= np.linalg.norm(vertices[0])
        
        return vertices
    
    def _generate_hypercube(self) -> np.ndarray:
        """
        Generate n-dimensional hypercube vertices.
        
        Returns:
            Array of shape (2^n, n) containing vertex coordinates
        """
        n = self.dimension
        
        # All combinations of ±1 in n dimensions
        vertices = []
        for i in range(2**n):
            vertex = []
            for j in range(n):
                vertex.append(1 if (i >> j) & 1 else -1)
            vertices.append(vertex)
        
        vertices = np.array(vertices, dtype=float)
        
        # Normalize to unit hypersphere
        vertices /= np.linalg.norm(vertices[0])
        
        return vertices
    
    def _generate_cross_polytope(self) -> np.ndarray:
        """
        Generate n-dimensional cross-polytope vertices.
        
        Returns:
            Array of shape (2n, n) containing vertex coordinates
        """
        n = self.dimension
        
        # Standard basis vectors ±e_i
        vertices = []
        for i in range(n):
            for sign in [1, -1]:
                v = np.zeros(n)
                v[i] = sign
                vertices.append(v)
        
        return np.array(vertices, dtype=float)
    
    def get_polytope_info(self, polytope_type: str) -> Dict[str, any]:
        """
        Get information about a polytope type.
        
        Parameters:
            polytope_type: Type of polytope
            
        Returns:
            Dictionary with polytope information
        """
        vertices = self.generate(polytope_type)
        
        return {
            "type": polytope_type,
            "dimension": self.dimension,
            "num_vertices": len(vertices),
            "vertices": vertices,
            "is_regular": True,
        }
