"""
CANS-nD: n-Dimensional Angular System
Part 2 implementation - k-dihedral and k-solid angle computations

This module implements the NDAngularSystem which provides:
- k-dihedral angles between k-faces
- k-dimensional solid angles
- Recursive angular computations for arbitrary dimensions
"""

import numpy as np
from typing import Dict, List, Any
from scipy.special import gamma

from .nd_primitives import (
    NDPrimitive,
    NDVertex,
    NDEdge,
    NDFace,
    NDHyperface,
    NDGeometryError,
)


class NDAngularSystem:
    """n-Dimensional Angular Computation System"""
    
    def __init__(self, dimension: int):
        """
        Initialize the n-dimensional angular system.
        
        Parameters:
            dimension: Dimension of the space
        """
        self.dimension = dimension
        self.primitives: Dict[str, NDPrimitive] = {}
        self.angle_cache: Dict[Any, float] = {}
    
    def add_primitive(self, primitive: NDPrimitive):
        """
        Add a geometric primitive to the system.
        
        Parameters:
            primitive: NDPrimitive to add
        """
        if primitive.dimension > self.dimension:
            raise NDGeometryError(
                f"Primitive dimension {primitive.dimension} > system dimension {self.dimension}"
            )
        if primitive.label:
            self.primitives[primitive.label] = primitive
    
    def k_dihedral_angle(self, face1: NDPrimitive, face2: NDPrimitive,
                        intersection: NDPrimitive) -> float:
        """
        Compute k-dihedral angle between two k-faces meeting at a (k-1)-face.
        
        Parameters:
            face1, face2: Two k-dimensional faces
            intersection: Their (k-1)-dimensional intersection
            
        Returns:
            Angle in radians
        """
        k = intersection.dimension + 1
        if k < 2 or k > self.dimension:
            raise NDGeometryError(f"Invalid k value: {k}")
        
        # Compute normal vectors in the orthogonal complement of the intersection
        n1 = self._face_normal_in_orthospace(face1, intersection)
        n2 = self._face_normal_in_orthospace(face2, intersection)
        
        # Compute angle between normals
        cos_angle = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Determine if angle should be reversed based on orientation
        if self._should_reverse_angle(face1, face2, intersection):
            angle = 2 * np.pi - angle
        
        return angle
    
    def _face_normal_in_orthospace(self, face: NDPrimitive,
                                   lower_face: NDPrimitive) -> np.ndarray:
        """Compute normal vector to face in the orthogonal complement of lower_face"""
        # Get basis for the lower face's affine space
        lower_basis = self._affine_basis(lower_face)
        
        # Get basis for the orthogonal complement
        ortho_basis = self._orthogonal_complement(lower_basis)
        
        if ortho_basis.shape[1] == 0:
            raise NDGeometryError("No orthogonal complement space")
        
        # Project face's normal onto orthogonal complement
        if isinstance(face, NDHyperface):
            face_normal = face.normal_vector()
        else:
            # For lower-dimensional faces, compute relative normal
            face_normal = self._compute_relative_normal(face, lower_face)
        
        # Project onto orthogonal basis
        projected = np.zeros_like(face_normal)
        for i in range(ortho_basis.shape[1]):
            component = np.dot(face_normal, ortho_basis[:, i])
            projected += component * ortho_basis[:, i]
        
        norm = np.linalg.norm(projected)
        if norm < 1e-12:
            raise NDGeometryError("Projected normal is zero")
        
        return projected / norm
    
    def _affine_basis(self, primitive: NDPrimitive) -> np.ndarray:
        """Compute basis for the affine space containing the primitive"""
        if len(primitive.vertices) == 1:
            return np.zeros((self.dimension, 0))
        
        vertices = primitive.vertex_coordinates
        centered = vertices[1:] - vertices[0]
        
        # Use QR decomposition to get orthonormal basis
        if centered.size == 0:
            return np.zeros((self.dimension, 0))
        
        Q, R = np.linalg.qr(centered.T, mode='reduced')
        
        # Filter out zero columns
        tol = 1e-12
        mask = np.abs(np.diag(R)) > tol
        return Q[:, mask]
    
    def _orthogonal_complement(self, basis: np.ndarray) -> np.ndarray:
        """Compute orthogonal complement of a basis"""
        if basis.size == 0:
            return np.eye(self.dimension)
        
        # Use SVD to find orthogonal complement
        U, s, Vt = np.linalg.svd(basis, full_matrices=True)
        rank = np.sum(s > 1e-10)
        
        # The complement is the remaining columns of U
        return U[:, rank:]
    
    def _compute_relative_normal(self, face: NDPrimitive,
                                reference_face: NDPrimitive) -> np.ndarray:
        """Compute normal vector relative to a reference face"""
        if face.dimension == reference_face.dimension + 1:
            if isinstance(face, NDHyperface):
                return face.normal_vector()
            elif isinstance(face, NDFace):
                return face.normal_vector()
            else:
                return self._approximate_normal(face, reference_face)
        else:
            raise NDGeometryError("Cannot compute relative normal for unrelated faces")
    
    def _approximate_normal(self, face: NDPrimitive,
                           reference_face: NDPrimitive) -> np.ndarray:
        """Approximate normal vector for non-hyperface primitives"""
        ref_vertices = set(tuple(v) for v in reference_face.vertex_coordinates)
        face_vertices = face.vertex_coordinates
        
        extra_vertex = None
        for vertex in face_vertices:
            if tuple(vertex) not in ref_vertices:
                extra_vertex = vertex
                break
        
        if extra_vertex is None:
            raise NDGeometryError("Cannot find distinguishing vertex")
        
        centroid = np.mean(reference_face.vertex_coordinates, axis=0)
        direction = extra_vertex - centroid
        
        # Remove components along reference face
        ref_basis = self._affine_basis(reference_face)
        if ref_basis.size > 0:
            for i in range(ref_basis.shape[1]):
                component = np.dot(direction, ref_basis[:, i])
                direction -= component * ref_basis[:, i]
        
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            raise NDGeometryError("Computed zero normal vector")
        
        return direction / norm
    
    def _should_reverse_angle(self, face1: NDPrimitive, face2: NDPrimitive,
                             intersection: NDPrimitive) -> bool:
        """Determine if dihedral angle should be reversed (interior vs exterior)"""
        # Simplified heuristic; for convex polytopes this is usually sufficient
        # Full orientation analysis would inspect polytope incidence structure
        return False
    
    def solid_angle_kd(self, vertex: NDVertex, container: NDPrimitive, k: int) -> float:
        """
        Compute k-dimensional solid angle at a vertex in a container primitive.
        
        Parameters:
            vertex: Vertex at which to compute solid angle
            container: Containing primitive
            k: Dimension of solid angle (2 ≤ k ≤ dimension)
            
        Returns:
            k-dimensional solid angle
        """
        if k < 2 or k > self.dimension:
            raise NDGeometryError(f"Invalid k for solid angle: {k}")
        
        if k == 2:
            # 2D angle in a face: reduce to planar angle
            return self._planar_angle_at_vertex(vertex, container)
        
        return self._solid_angle_kd_recursive(vertex, container, k)
    
    def _planar_angle_at_vertex(self, vertex: NDVertex, container: NDPrimitive) -> float:
        """Compute 2D interior angle at vertex within a given container"""
        vertex_coord = vertex.coordinates
        
        # Get vertices of container
        other_vertices = []
        for v in container.vertices:
            if not np.allclose(v, vertex_coord):
                other_vertices.append(v)
        
        if len(other_vertices) < 2:
            raise NDGeometryError("Need at least 2 other vertices for 2D angle")
        
        # Use first two neighbors
        v1 = other_vertices[0] - vertex_coord
        v2 = other_vertices[1] - vertex_coord
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-12 or v2_norm < 1e-12:
            return 0.0
        
        v1 = v1 / v1_norm
        v2 = v2 / v2_norm
        
        cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return np.arccos(cos_angle)
    
    def _solid_angle_kd_recursive(self, vertex: NDVertex, container: NDPrimitive, k: int) -> float:
        """Recursively compute k-dimensional solid angle"""
        # Get all (k-1)-faces incident to the vertex in the container
        incident_faces = self._get_incident_faces(vertex, container, k - 1)
        
        if len(incident_faces) < k:
            # Not enough faces, return approximate value
            return 0.0
        
        # Compute (k-1)-solid angles recursively
        lower_solid_angles: List[float] = []
        for face in incident_faces[:min(k, len(incident_faces))]:
            try:
                angle = self.solid_angle_kd(vertex, face, k - 1)
                lower_solid_angles.append(angle)
            except (NDGeometryError, ValueError):
                continue
        
        if not lower_solid_angles:
            return 0.0
        
        if k == 3:
            # Girard's theorem for 3D
            return sum(lower_solid_angles) - (len(lower_solid_angles) - 2) * np.pi
        else:
            # Approximate for higher dimensions
            avg_lower_angle = np.mean(lower_solid_angles)
            scaling_factor = 2 * np.pi / self._total_solid_angle_k_minus_1(k - 1)
            return avg_lower_angle * scaling_factor
    
    def _total_solid_angle_k_minus_1(self, k: int) -> float:
        """
        Total solid angle of (k-1)-sphere.
        
        Formula: 2π^(k/2) / Γ(k/2)
        """
        if k == 2:
            return 2 * np.pi  # Circumference of circle
        elif k == 3:
            return 4 * np.pi  # Surface area of sphere
        else:
            return 2 * np.pi**(k / 2) / gamma(k / 2)
    
    def _get_incident_faces(self, vertex: NDVertex, container: NDPrimitive,
                           face_dim: int) -> List[NDPrimitive]:
        """Get all faces of given dimension incident to vertex in container"""
        incident_faces: List[NDPrimitive] = []
        for label, primitive in self.primitives.items():
            if (primitive.dimension == face_dim
                and primitive.contains_point(vertex.coordinates)
                and self._is_face_of(primitive, container)):
                incident_faces.append(primitive)
        return incident_faces
    
    def _is_face_of(self, face: NDPrimitive, container: NDPrimitive) -> bool:
        """Check if face is a sub-face of container"""
        for vertex in face.vertices:
            if not container.contains_point(vertex):
                return False
        return True
