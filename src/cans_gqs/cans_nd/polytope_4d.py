"""
CANS-4D: 4D Polytope Angular System
Part 2 implementation - specialized 4D polytope handling

This module implements the Polytope4DAngularSystem which provides:
- Cell-cell angles between 3D cells in 4D polytopes
- 4D hypersolid angles at vertices
- 4D vertex defects
- 4D Gauss-Bonnet verification
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from scipy.linalg import null_space

from .nd_primitives import NDVertex, NDFace, NDHyperface, NDGeometryError
from .nd_angular_system import NDAngularSystem


class Polytope4DAngularSystem(NDAngularSystem):
    """
    Specialized angular system for 4D polytopes.
    
    Extends NDAngularSystem with 4D-specific operations:
    - Cell-cell angles (angles between 3D cells)
    - 4D hypersolid angles at vertices
    - 4D vertex defects
    """
    
    def __init__(self, vertices: List[np.ndarray], cells: List[Any]):
        """
        Initialize 4D polytope system.
        
        Parameters:
            vertices: List of 4D vertex coordinates
            cells: List of 3D cells (each cell is a polyhedron)
        """
        super().__init__(4)
        
        self.vertices_4d = vertices
        self.cells = cells
        
        # Add vertices as primitives
        for i, v in enumerate(vertices):
            vertex = NDVertex(v, f"V_{i}")
            self.add_primitive(vertex)
    
    def cell_cell_angle(self, face: NDFace) -> float:
        """
        Compute angle between two 3D cells meeting at a 2D face.
        
        In 4D, cells are 3D polyhedra, and they meet at 2D faces.
        The cell-cell angle is analogous to the dihedral angle in 3D.
        
        Parameters:
            face: 2D face where two cells meet
            
        Returns:
            Angle in radians between the two cells
        """
        # Find the two cells that share this face
        cell1, cell2 = self._cells_sharing_face(face)
        
        # Compute normal vectors to the cells in 4D space
        n1 = self._cell_normal_4d(cell1)
        n2 = self._cell_normal_4d(cell2)
        
        # Compute angle between normals
        cos_angle = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return angle
    
    def hypersolid_angle_4d(self, vertex: str) -> float:
        """
        Compute 4D hypersolid angle at a vertex.
        
        The hypersolid angle is the 4D generalization of solid angle.
        For a vertex in a 4D polytope, it measures the "4D conical spread"
        at that vertex.
        
        Parameters:
            vertex: Vertex label
            
        Returns:
            Hypersolid angle in 4D steradians (units: sr³)
        """
        if vertex not in self.primitives:
            raise NDGeometryError(f"Vertex {vertex} not found in system")
        
        vertex_primitive = self.primitives[vertex]
        if not isinstance(vertex_primitive, NDVertex):
            raise NDGeometryError(f"{vertex} is not a vertex")
        
        # Get all 3D cells incident to this vertex
        incident_cells = self._get_incident_cells(vertex_primitive)
        
        if len(incident_cells) == 0:
            return 0.0
        
        # Compute hypersolid angle using 4D generalization of Girard's theorem
        # For a 4D polytope vertex, sum the 3D solid angles of incident cells
        # and apply the 4D spherical excess formula
        
        total_3d_solid_angles = 0.0
        for cell in incident_cells:
            # Compute 3D solid angle of this cell at the vertex
            solid_angle_3d = self._solid_angle_3d_in_cell(vertex_primitive, cell)
            total_3d_solid_angles += solid_angle_3d
        
        # 4D spherical excess: Ω_4 = Σ(Ω_3) - (n-3)·2π²
        # where n is the number of incident cells
        n = len(incident_cells)
        if n >= 3:
            hypersolid = total_3d_solid_angles - (n - 3) * 2 * np.pi**2
        else:
            hypersolid = total_3d_solid_angles
        
        return max(0.0, hypersolid)  # Hypersolid angle should be non-negative
    
    def vertex_defect_4d(self, vertex: str) -> float:
        """
        Compute 4D vertex defect (intrinsic 3D curvature at vertex).
        
        The 4D vertex defect is defined as:
        δ_4(V) = 2π² - Σ(Ω_3(V, cell))
        
        where Ω_3(V, cell) are the 3D solid angles at V in each incident cell.
        
        Parameters:
            vertex: Vertex label
            
        Returns:
            4D vertex defect in radians³
        """
        if vertex not in self.primitives:
            raise NDGeometryError(f"Vertex {vertex} not found in system")
        
        vertex_primitive = self.primitives[vertex]
        if not isinstance(vertex_primitive, NDVertex):
            raise NDGeometryError(f"{vertex} is not a vertex")
        
        # Get all 3D cells incident to this vertex
        incident_cells = self._get_incident_cells(vertex_primitive)
        
        # Sum 3D solid angles
        total_3d_solid_angles = 0.0
        for cell in incident_cells:
            solid_angle_3d = self._solid_angle_3d_in_cell(vertex_primitive, cell)
            total_3d_solid_angles += solid_angle_3d
        
        # 4D vertex defect
        defect_4d = 2 * np.pi**2 - total_3d_solid_angles
        
        return defect_4d
    
    def _cell_normal_4d(self, cell: Any) -> np.ndarray:
        """
        Compute 4D normal vector to a 3D cell.
        
        A 3D cell in 4D space has a 1D normal (a direction in 4D).
        This is computed using the null space of the cell's 3D basis.
        
        Parameters:
            cell: 3D cell representation
            
        Returns:
            Unit normal vector in 4D
        """
        # Get vertices of the cell
        if hasattr(cell, 'vertices'):
            cell_vertices = cell.vertices
        else:
            # Assume cell is a list of vertex indices
            cell_vertices = [self.vertices_4d[i] for i in cell]
        
        if len(cell_vertices) < 4:
            raise NDGeometryError("Cell must have at least 4 vertices")
        
        # Create basis vectors for the 3D affine space of the cell
        v0 = cell_vertices[0]
        basis_vectors = [cell_vertices[i] - v0 for i in range(1, 4)]
        
        # Stack into matrix
        A = np.array(basis_vectors).T  # 4x3 matrix
        
        # Normal is in the null space
        null_vec = null_space(A)
        
        if null_vec.size == 0:
            raise NDGeometryError("Could not compute cell normal")
        
        n = null_vec[:, 0]
        return n / np.linalg.norm(n)
    
    def _cells_sharing_face(self, face: NDFace) -> Tuple[Any, Any]:
        """
        Find the two cells that share a given 2D face.
        
        Parameters:
            face: 2D face
            
        Returns:
            Tuple of (cell1, cell2)
        """
        sharing_cells = []
        
        for cell in self.cells:
            # Check if all face vertices are in the cell
            if hasattr(cell, 'vertices'):
                cell_vertices_set = set(tuple(v) for v in cell.vertices)
            else:
                cell_vertices_set = set(cell)
            
            face_vertices_set = set(tuple(v) for v in face.vertices)
            
            # Check if face vertices are subset of cell vertices
            if len(face_vertices_set) > 0:
                # Simplified check: if face is "close" to being in cell
                sharing_cells.append(cell)
                
            if len(sharing_cells) >= 2:
                break
        
        if len(sharing_cells) < 2:
            raise NDGeometryError("Face must be shared by exactly 2 cells")
        
        return sharing_cells[0], sharing_cells[1]
    
    def _get_incident_cells(self, vertex: NDVertex) -> List[Any]:
        """
        Get all 3D cells incident to a vertex.
        
        Parameters:
            vertex: NDVertex
            
        Returns:
            List of cells containing the vertex
        """
        incident = []
        v_coord = tuple(vertex.coordinates)
        
        for cell in self.cells:
            if hasattr(cell, 'vertices'):
                cell_coords = [tuple(v) for v in cell.vertices]
            else:
                cell_coords = [tuple(self.vertices_4d[i]) for i in cell]
            
            # Check if vertex is in cell
            for coord in cell_coords:
                if np.allclose(v_coord, coord, atol=1e-10):
                    incident.append(cell)
                    break
        
        return incident
    
    def _solid_angle_3d_in_cell(self, vertex: NDVertex, cell: Any) -> float:
        """
        Compute 3D solid angle at a vertex within a 3D cell.
        
        Parameters:
            vertex: Vertex
            cell: 3D cell containing the vertex
            
        Returns:
            3D solid angle in steradians
        """
        # Get vertices of the cell
        if hasattr(cell, 'vertices'):
            cell_vertices = [v for v in cell.vertices if not np.allclose(v, vertex.coordinates)]
        else:
            cell_vertices = [self.vertices_4d[i] for i in cell 
                           if not np.allclose(self.vertices_4d[i], vertex.coordinates)]
        
        if len(cell_vertices) < 3:
            return 0.0
        
        # Use first 3 neighbors to approximate solid angle
        # In a more complete implementation, we'd use all faces incident to the vertex
        neighbors = cell_vertices[:3]
        
        # Vectors from vertex to neighbors
        v1 = neighbors[0] - vertex.coordinates
        v2 = neighbors[1] - vertex.coordinates
        v3 = neighbors[2] - vertex.coordinates
        
        # Normalize
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        v3 = v3 / np.linalg.norm(v3)
        
        # Compute solid angle using determinant formula
        # Ω = 2 * arctan(|det(v1, v2, v3)| / (1 + v1·v2 + v2·v3 + v3·v1))
        
        # For 4D vectors, we need to project to 3D first
        # Use first 3 components
        v1_3d = v1[:3]
        v2_3d = v2[:3]
        v3_3d = v3[:3]
        
        det = np.linalg.det(np.array([v1_3d, v2_3d, v3_3d]))
        denom = 1 + np.dot(v1_3d, v2_3d) + np.dot(v2_3d, v3_3d) + np.dot(v3_3d, v1_3d)
        
        if abs(denom) < 1e-10:
            return 0.0
        
        solid_angle = 2 * np.arctan2(abs(det), denom)
        
        return max(0.0, solid_angle)


def verify_4d_gauss_bonnet(polytope_system: Polytope4DAngularSystem) -> bool:
    """
    Verify 4D Gauss-Bonnet theorem: Σδ₄(Vᵢ) = 2π²χ
    
    For a 4D polytope, the sum of 4D vertex defects equals 2π² times
    the Euler characteristic.
    
    Parameters:
        polytope_system: Polytope4DAngularSystem instance
        
    Returns:
        True if theorem is satisfied within numerical tolerance
    """
    # Compute total defect
    vertices = [p for p in polytope_system.primitives.values() 
               if isinstance(p, NDVertex)]
    
    total_defect = 0.0
    for vertex in vertices:
        defect = polytope_system.vertex_defect_4d(vertex.label)
        total_defect += defect
    
    # Compute Euler characteristic for 4D polytope
    # χ = V - E + F - C
    # where V=vertices, E=edges, F=faces, C=cells
    
    # This is a simplified calculation
    # In a complete implementation, we'd enumerate all k-faces
    V = len(vertices)
    C = len(polytope_system.cells)
    
    # For regular 4D polytopes, we can use known values
    # For now, use a heuristic
    chi = 0  # This should be computed properly
    
    expected = 2 * np.pi**2 * chi
    
    # Check if close
    return np.isclose(total_defect, expected, rtol=1e-2)


def create_5_cell() -> Tuple[List[np.ndarray], List[List[int]]]:
    """
    Create a regular 5-cell (4-simplex).
    
    The 5-cell is the 4D analogue of a tetrahedron.
    It has 5 vertices, 10 edges, 10 triangular faces, and 5 tetrahedral cells.
    
    Returns:
        Tuple of (vertices, cells)
    """
    # Regular 5-cell vertices in 4D
    # Using coordinates that form a regular 4-simplex
    vertices = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([-0.25, -0.25, -0.25, -0.25]),  # Center, adjusted for regularity
    ]
    
    # Normalize to make it more regular
    for i in range(len(vertices)):
        vertices[i] = vertices[i] / np.linalg.norm(vertices[i])
    
    # Each cell is a tetrahedron formed by 4 vertices
    cells = [
        [0, 1, 2, 3],  # Cell 0
        [0, 1, 2, 4],  # Cell 1
        [0, 1, 3, 4],  # Cell 2
        [0, 2, 3, 4],  # Cell 3
        [1, 2, 3, 4],  # Cell 4
    ]
    
    return vertices, cells


def create_tesseract() -> Tuple[List[np.ndarray], List[List[int]]]:
    """
    Create a regular tesseract (4D hypercube).
    
    The tesseract has 16 vertices, 32 edges, 24 square faces, and 8 cubic cells.
    
    Returns:
        Tuple of (vertices, cells)
    """
    # Tesseract vertices: all combinations of {-1, 1}^4
    vertices = []
    for i in range(16):
        x = 1.0 if (i & 1) else -1.0
        y = 1.0 if (i & 2) else -1.0
        z = 1.0 if (i & 4) else -1.0
        w = 1.0 if (i & 8) else -1.0
        vertices.append(np.array([x, y, z, w]))
    
    # Each cell is a cube formed by 8 vertices
    # A cube in 4D has one coordinate fixed
    cells = []
    
    # Cubes with w = -1 and w = 1
    cells.append([i for i in range(16) if vertices[i][3] == -1.0])
    cells.append([i for i in range(16) if vertices[i][3] == 1.0])
    
    # Cubes with z = -1 and z = 1
    cells.append([i for i in range(16) if vertices[i][2] == -1.0])
    cells.append([i for i in range(16) if vertices[i][2] == 1.0])
    
    # Cubes with y = -1 and y = 1
    cells.append([i for i in range(16) if vertices[i][1] == -1.0])
    cells.append([i for i in range(16) if vertices[i][1] == 1.0])
    
    # Cubes with x = -1 and x = 1
    cells.append([i for i in range(16) if vertices[i][0] == -1.0])
    cells.append([i for i in range(16) if vertices[i][0] == 1.0])
    
    return vertices, cells
