"""
CANS-3D: Polyhedral Angle System
Part 1 implementation of the Comprehensive Angular Naming System

This module implements the reference 3D implementation of CANS, providing:
- Planar angles (2D interior angles on faces)
- Dihedral angles (3D angles along edges)
- Solid angles (3D conical spread, extrinsic)
- Vertex defects (2D intrinsic curvature)
"""

import numpy as np
from typing import List, Dict, Tuple, Set


class PolyhedralAngleSystem:
    """
    Reference implementation of CANS-3D for polyhedral geometry.
    
    Computes and manages angular properties at vertices, edges, and faces
    of a 3D polyhedron, including the explicit de-conflation of solid angle
    Ω(V) and vertex defect δ(V).
    """
    
    def __init__(self, vertices: List, faces: List[List[int]]):
        """
        Initialize the polyhedral angle system.
        
        Parameters:
            vertices: List of 3D coordinates, e.g. [(x1, y1, z1), (x2, y2, z2), ...]
            faces: List of faces, each face is a list of vertex indices
        """
        self.vertices = {f"V_{i}": np.array(v, dtype=float) for i, v in enumerate(vertices)}
        self.faces = {f"F_{i}": face for i, face in enumerate(faces)}
        self.edges = self._extract_edges()
        
    def _extract_edges(self) -> Dict[str, Tuple[int, int]]:
        """
        Extract all edges from face definitions.
        
        Returns:
            Dictionary mapping edge labels to (vertex_idx1, vertex_idx2) tuples
        """
        edges = set()
        for face in self.faces.values():
            n = len(face)
            for i in range(n):
                v1, v2 = face[i], face[(i + 1) % n]
                # Store edge as sorted tuple for consistency
                edge = tuple(sorted([v1, v2]))
                edges.add(edge)
        
        return {f"E_{i}": edge for i, edge in enumerate(edges)}
    
    def planar_angle(self, v_prev: int, v_curr: int, v_next: int, face: List[int]) -> float:
        """
        Compute planar angle at vertex v_curr on a face.
        
        Parameters:
            v_prev: Previous vertex index in the face
            v_curr: Current vertex index (where angle is measured)
            v_next: Next vertex index in the face
            face: The face containing these vertices
            
        Returns:
            Angle in radians (0, 2π)
        """
        # Get coordinates
        curr = self.vertices[f"V_{v_curr}"]
        prev = self.vertices[f"V_{v_prev}"]
        next_v = self.vertices[f"V_{v_next}"]
        
        # Compute edge vectors
        u = prev - curr
        v = next_v - curr
        
        # Normalize
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
        
        if u_norm < 1e-12 or v_norm < 1e-12:
            return 0.0
        
        u = u / u_norm
        v = v / v_norm
        
        # Compute angle using arccos - this gives interior angle
        cos_theta = np.clip(np.dot(u, v), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # Note: For a convex polytope, this is the interior angle
        # For non-convex cases, we may need to check orientation
        # but for standard meshes, arccos gives the correct interior angle
        return theta
    
    def _is_reentrant(self, v_prev: int, v_curr: int, v_next: int, face: List[int]) -> bool:
        """
        Check if the angle at v_curr is re-entrant (non-convex).
        
        Uses the face normal to determine if the angle is interior or exterior.
        """
        # Get face normal
        face_normal = self._compute_face_normal(face)
        
        # Get edge vectors
        curr = self.vertices[f"V_{v_curr}"]
        prev = self.vertices[f"V_{v_prev}"]
        next_v = self.vertices[f"V_{v_next}"]
        
        u = prev - curr
        v = next_v - curr
        
        # Cross product gives a vector perpendicular to both edges
        cross = np.cross(u, v)
        
        # If cross product points opposite to face normal, angle is re-entrant
        return np.dot(cross, face_normal) < 0
    
    def _compute_face_normal(self, face: List[int]) -> np.ndarray:
        """
        Compute outward-facing normal vector for a face using Newell's method.
        """
        normal = np.zeros(3)
        n = len(face)
        
        for i in range(n):
            v_curr = self.vertices[f"V_{face[i]}"]
            v_next = self.vertices[f"V_{face[(i + 1) % n]}"]
            
            normal[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
            normal[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
            normal[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])
        
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            raise ValueError(f"Face {face} is degenerate (zero normal)")
        
        return normal / norm
    
    def vertex_defect(self, vertex: str) -> float:
        """
        Compute vertex defect δ(V) = 2π - Σ A_p(V, F).
        
        The vertex defect is the 2D intrinsic curvature measure at a vertex,
        measured in radians.
        
        Parameters:
            vertex: Vertex label (e.g., "V_0")
            
        Returns:
            Vertex defect in radians
        """
        vertex_idx = int(vertex.split('_')[1])
        total_planar_angles = 0.0
        
        # Find all faces incident to this vertex
        for face_label, face in self.faces.items():
            if vertex_idx in face:
                # Find position of vertex in face
                idx = face.index(vertex_idx)
                n = len(face)
                v_prev = face[(idx - 1) % n]
                v_next = face[(idx + 1) % n]
                
                angle = self.planar_angle(v_prev, vertex_idx, v_next, face)
                total_planar_angles += angle
        
        return 2 * np.pi - total_planar_angles
    
    def solid_angle(self, vertex: str) -> float:
        """
        Compute solid angle Ω(V) at a vertex using Girard's theorem.
        
        The solid angle is the 3D extrinsic measure of the conical spread
        at a vertex, measured in steradians.
        
        Parameters:
            vertex: Vertex label (e.g., "V_0")
            
        Returns:
            Solid angle in steradians
        """
        vertex_idx = int(vertex.split('_')[1])
        vertex_pos = self.vertices[vertex]
        
        # Get all faces incident to this vertex
        incident_faces = []
        for face in self.faces.values():
            if vertex_idx in face:
                incident_faces.append(face)
        
        if len(incident_faces) < 3:
            return 0.0
        
        # Project incident edges onto unit sphere centered at vertex
        # and compute spherical polygon angles
        spherical_angles = []
        
        for face in incident_faces:
            idx = face.index(vertex_idx)
            n = len(face)
            v_prev = face[(idx - 1) % n]
            v_next = face[(idx + 1) % n]
            
            # Get unit vectors from vertex to neighbors
            prev_vec = self.vertices[f"V_{v_prev}"] - vertex_pos
            next_vec = self.vertices[f"V_{v_next}"] - vertex_pos
            
            prev_vec = prev_vec / np.linalg.norm(prev_vec)
            next_vec = next_vec / np.linalg.norm(next_vec)
            
            # Spherical angle
            cos_angle = np.clip(np.dot(prev_vec, next_vec), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            spherical_angles.append(angle)
        
        # Apply Girard's theorem: Ω = Σα_k - (n - 2)π
        n_faces = len(incident_faces)
        solid_angle_value = sum(spherical_angles) - (n_faces - 2) * np.pi
        
        return max(0.0, solid_angle_value)  # Ensure non-negative
    
    def dihedral_angle(self, edge: str) -> float:
        """
        Compute dihedral angle A_d(E) between two faces meeting at an edge.
        
        Uses arctan2 formulation for robustness and signed angle computation.
        
        Parameters:
            edge: Edge label (e.g., "E_0")
            
        Returns:
            Dihedral angle in radians (-π, π]
        """
        if edge not in self.edges:
            raise ValueError(f"Edge {edge} not found")
        
        v1_idx, v2_idx = self.edges[edge]
        
        # Find the two faces sharing this edge
        adjacent_faces = []
        for face_label, face in self.faces.items():
            if v1_idx in face and v2_idx in face:
                adjacent_faces.append((face_label, face))
        
        if len(adjacent_faces) != 2:
            raise ValueError(f"Edge {edge} does not have exactly 2 adjacent faces")
        
        # Compute face normals
        n1 = self._compute_face_normal(adjacent_faces[0][1])
        n2 = self._compute_face_normal(adjacent_faces[1][1])
        
        # Get edge direction vector
        edge_vec = self.vertices[f"V_{v2_idx}"] - self.vertices[f"V_{v1_idx}"]
        edge_vec = edge_vec / np.linalg.norm(edge_vec)
        
        # Compute dihedral angle using arctan2 for robustness
        cos_angle = np.dot(n1, n2)
        sin_angle_vec = np.cross(n1, n2)
        sin_angle = np.dot(sin_angle_vec, edge_vec)
        
        # arctan2 gives signed angle in (-π, π]
        angle = np.arctan2(sin_angle, cos_angle)
        
        return angle
    
    def vector_angle(self, v1_start: int, v1_end: int, v2_start: int, v2_end: int) -> float:
        """
        Compute angle between two directed vector segments.
        
        Parameters:
            v1_start, v1_end: Start and end vertices of first vector
            v2_start, v2_end: Start and end vertices of second vector
            
        Returns:
            Angle in radians [0, π]
        """
        vec1 = self.vertices[f"V_{v1_end}"] - self.vertices[f"V_{v1_start}"]
        vec2 = self.vertices[f"V_{v2_end}"] - self.vertices[f"V_{v2_start}"]
        
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)
        
        if vec1_norm < 1e-12 or vec2_norm < 1e-12:
            return 0.0
        
        vec1 = vec1 / vec1_norm
        vec2 = vec2 / vec2_norm
        
        cos_angle = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
        return np.arccos(cos_angle)
    
    def torsion_angle(self, v1: int, v2: int, v3: int, v4: int) -> float:
        """
        Compute torsion angle (4-point dihedral) along a chain of 4 vertices.
        
        This is the molecular torsion angle A_t used in protein structure analysis.
        
        Parameters:
            v1, v2, v3, v4: Four consecutive vertex indices
            
        Returns:
            Torsion angle in radians (-π, π]
        """
        # Get coordinates
        p1 = self.vertices[f"V_{v1}"]
        p2 = self.vertices[f"V_{v2}"]
        p3 = self.vertices[f"V_{v3}"]
        p4 = self.vertices[f"V_{v4}"]
        
        # Compute bond vectors
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        
        # Compute normals to planes
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        # Normalize
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        
        if n1_norm < 1e-12 or n2_norm < 1e-12:
            return 0.0
        
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        
        # Compute torsion angle
        b2_unit = b2 / np.linalg.norm(b2)
        
        cos_angle = np.dot(n1, n2)
        sin_angle = np.dot(np.cross(n1, n2), b2_unit)
        
        return np.arctan2(sin_angle, cos_angle)
    
    def verify_gauss_bonnet(self) -> Tuple[float, float, bool]:
        """
        Verify the discrete Gauss-Bonnet theorem: Σδ(V_i) = 2πχ
        
        Returns:
            Tuple of (total_defect, expected_value, is_valid)
        """
        # Compute total vertex defect
        total_defect = sum(self.vertex_defect(v) for v in self.vertices.keys())
        
        # Compute Euler characteristic χ = V - E + F
        V = len(self.vertices)
        E = len(self.edges)
        F = len(self.faces)
        chi = V - E + F
        
        expected = 2 * np.pi * chi
        is_valid = np.isclose(total_defect, expected, rtol=1e-6)
        
        return total_defect, expected, is_valid
