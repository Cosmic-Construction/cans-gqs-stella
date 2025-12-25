"""
CANS-nD: n-Dimensional Primitive Hierarchy
Part 2 implementation - Geometric primitives for arbitrary dimensions

This module provides the foundational geometric primitives for CANS-nD:
- NDVertex (0-dimensional)
- NDEdge (1-dimensional)
- NDFace (2-dimensional)
- NDHyperface ((n-1)-dimensional)
- NDPolytope (n-dimensional)
"""

import numpy as np
from typing import List, Tuple
from scipy.linalg import null_space


class NDGeometryError(Exception):
    """Custom exception for n-dimensional geometry errors"""
    pass


class NDPrimitive:
    """Base class for n-dimensional geometric primitives"""
    
    def __init__(self, dimension: int, vertices: List[np.ndarray], label: str = None):
        """
        Initialize an n-dimensional primitive.
        
        Parameters:
            dimension: Ambient space dimension
            vertices: List of nD points
            label: Optional label for the primitive
        """
        self.dimension = dimension
        self.vertices = vertices
        self.label = label
        self.validate()
    
    def validate(self):
        """Validate the primitive's geometric consistency"""
        if not self.vertices:
            raise NDGeometryError(f"{self.__class__.__name__} must have vertices")
        
        for v in self.vertices:
            if len(v) != self.dimension:
                raise NDGeometryError(
                    f"Vertex dimension {len(v)} != system dimension {self.dimension}"
                )
    
    @property
    def vertex_coordinates(self) -> np.ndarray:
        """Return vertices as a matrix (k x n) where k is number of vertices"""
        return np.array(self.vertices)
    
    def affine_dimension(self) -> int:
        """Compute the affine dimension of this primitive"""
        if len(self.vertices) < 2:
            return 0
        centered = self.vertex_coordinates - self.vertex_coordinates[0]
        return np.linalg.matrix_rank(centered)
    
    def contains_point(self, point: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if point lies in the affine hull of this primitive"""
        if len(self.vertices) == 1:
            return np.allclose(self.vertices[0], point, atol=tol)
        
        # Use barycentric coordinates
        A = (self.vertex_coordinates[1:] - self.vertex_coordinates[0]).T
        b = point - self.vertex_coordinates[0]
        
        try:
            coefficients, residual, rank, s = np.linalg.lstsq(A, b, rcond=None)
            residual_norm = np.linalg.norm(A @ coefficients - b) if len(A @ coefficients - b) > 0 else 0
            
            return (
                residual_norm < tol
                and np.all(coefficients >= -tol)
                and np.sum(coefficients) <= 1 + tol
            )
        except np.linalg.LinAlgError:
            return False


class NDVertex(NDPrimitive):
    """0-dimensional primitive (point)"""
    
    def __init__(self, coordinates: np.ndarray, label: str = None):
        """
        Create a vertex (point) in n-dimensional space.
        
        Parameters:
            coordinates: nD coordinate array
            label: Optional label
        """
        super().__init__(len(coordinates), [coordinates], label)
        self.coordinates = coordinates


class NDEdge(NDPrimitive):
    """1-dimensional primitive (line segment)"""
    
    def __init__(self, vertices: List[np.ndarray], label: str = None):
        """
        Create an edge (line segment) in n-dimensional space.
        
        Parameters:
            vertices: List of exactly 2 vertices
            label: Optional label
        """
        super().__init__(len(vertices[0]), vertices, label)
        if len(vertices) != 2:
            raise NDGeometryError("Edge must have exactly 2 vertices")
    
    @property
    def direction_vector(self) -> np.ndarray:
        """Get normalized direction vector of the edge"""
        v1, v2 = self.vertices
        direction = v2 - v1
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            raise NDGeometryError("Edge has zero length")
        return direction / norm
    
    def length(self) -> float:
        """Compute edge length"""
        v1, v2 = self.vertices
        return np.linalg.norm(v2 - v1)


class NDFace(NDPrimitive):
    """2-dimensional primitive (polygon in nD space)"""
    
    def __init__(self, vertices: List[np.ndarray], label: str = None):
        """
        Create a face (2D polygon) in n-dimensional space.
        
        Parameters:
            vertices: List of at least 3 vertices
            label: Optional label
        """
        super().__init__(len(vertices[0]), vertices, label)
        if len(vertices) < 3:
            raise NDGeometryError("Face must have at least 3 vertices")
    
    def normal_vector(self) -> np.ndarray:
        """Compute normal vector to the 2D face in nD space"""
        if self.dimension <= 2:
            # For 2D space, normal is perpendicular in 2D
            if self.dimension == 2:
                v0, v1 = self.vertices[0], self.vertices[1]
                edge = v1 - v0
                # 2D perpendicular: rotate 90 degrees
                normal = np.array([-edge[1], edge[0]])
                return normal / np.linalg.norm(normal)
            raise NDGeometryError("Normal vector requires dimension >= 2")
        
        # Use first 3 points to define the plane
        v0, v1, v2 = self.vertices[:3]
        u = v1 - v0
        w = v2 - v0
        
        # For nD (n > 2), compute a vector orthogonal to both u and w via null space
        A = np.vstack([u, w])
        null_vectors = null_space(A)
        
        if null_vectors.size == 0:
            raise NDGeometryError("Could not compute normal vector - points may be collinear")
        
        n = null_vectors[:, 0]
        return n / np.linalg.norm(n)


class NDHyperface(NDPrimitive):
    """(n-1)-dimensional primitive (hyperplane in nD space)"""
    
    def __init__(self, vertices: List[np.ndarray], label: str = None):
        """
        Create a hyperface ((n-1)-dimensional face) in n-dimensional space.
        
        Parameters:
            vertices: List of at least n vertices
            label: Optional label
        """
        super().__init__(len(vertices[0]), vertices, label)
        if len(vertices) < self.dimension:
            raise NDGeometryError("Hyperface must have at least n vertices")
    
    def normal_vector(self) -> np.ndarray:
        """Compute normal vector to hyperface in nD space"""
        vertices = self.vertex_coordinates
        basis_points = vertices[:self.dimension]
        v0 = basis_points[0]
        
        # Create matrix of direction vectors
        A = np.array([v - v0 for v in basis_points[1:]]).T
        
        # Normal vector is in the null space of A
        normal = null_space(A)
        if normal.size == 0:
            raise NDGeometryError("Could not compute hyperface normal")
        
        n = normal[:, 0]
        return n / np.linalg.norm(n)
    
    def hyperplane_equation(self) -> Tuple[np.ndarray, float]:
        """
        Return hyperplane equation: n·x = d
        
        Returns:
            Tuple of (normal_vector, offset)
        """
        n = self.normal_vector()
        d = np.dot(n, self.vertices[0])
        return n, d


class NDPolytope:
    """n-dimensional polytope"""
    
    def __init__(self, dimension: int, vertices: List[np.ndarray],
                 faces: List[NDPrimitive], label: str = None):
        """
        Create an n-dimensional polytope.
        
        Parameters:
            dimension: Dimension of the polytope
            vertices: List of vertex coordinates
            faces: List of face primitives
            label: Optional label
        """
        self.dimension = dimension
        self.vertices = vertices
        self.faces = faces
        self.label = label
        self._adjacency_cache = {}
    
    def validate_topology(self):
        """Validate the polytope's topological consistency"""
        # Check all faces have correct dimension
        for face in self.faces:
            if face.dimension > self.dimension:
                raise NDGeometryError(
                    f"Face dimension {face.dimension} > polytope dimension {self.dimension}"
                )
        
        # Check Euler-Poincaré characteristic for simple low-dimensional cases
        if self.dimension <= 3:
            self._check_euler_characteristic()
    
    def _check_euler_characteristic(self):
        """Check Euler characteristic for low-dimensional polytopes"""
        if self.dimension == 2:
            # For polygons: V - E = 0
            edges = [f for f in self.faces if isinstance(f, NDEdge)]
            if len(self.vertices) - len(edges) != 0:
                raise NDGeometryError("2D polytope violates Euler characteristic")
        elif self.dimension == 3:
            # For polyhedra: V - E + F = 2
            edges = [f for f in self.faces if isinstance(f, NDEdge)]
            faces_2d = [f for f in self.faces if isinstance(f, NDFace)]
            euler = len(self.vertices) - len(edges) + len(faces_2d)
            if abs(euler - 2) > 1e-10:
                # Warning instead of error for flexibility
                print(f"Warning: 3D polytope Euler characteristic = {euler}, expected 2")
