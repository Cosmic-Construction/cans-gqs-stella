"""
CANS-GQS: Geometric Algebra Integration
Part 3 implementation - Clifford algebra interoperability

This module provides integration with geometric algebra libraries
for cross-validation and alternative computational approaches.
"""

import numpy as np
from typing import Dict, Any

from ..cans_nd.nd_primitives import NDPrimitive, NDVertex, NDGeometryError
from ..cans_nd.nd_angular_system import NDAngularSystem

try:
    import clifford
    from clifford import Cl
    CLIFFORD_AVAILABLE = True
except ImportError:
    CLIFFORD_AVAILABLE = False
    clifford = None


class GeometricAlgebraIntegration:
    """Integration with geometric algebra libraries"""
    
    def __init__(self, dimension: int):
        """
        Initialize geometric algebra integration.
        
        Parameters:
            dimension: Dimension of the space
        """
        self.dimension = dimension
        self.ga_layout = None
        self.ga_blades = None
        
        if not CLIFFORD_AVAILABLE:
            raise ImportError(
                "Clifford package required for geometric algebra integration. "
                "Install with: pip install clifford"
            )
        
        self._initialize_geometric_algebra()
    
    def _initialize_geometric_algebra(self):
        """Initialize geometric algebra for given dimension"""
        try:
            self.ga_layout, self.ga_blades = Cl(self.dimension)
        except Exception as e:
            raise NDGeometryError(f"Failed to initialize geometric algebra: {str(e)}")
    
    def vector_to_multivector(self, vector: np.ndarray):
        """
        Convert numpy vector to geometric algebra multivector.
        
        Parameters:
            vector: Numpy array
            
        Returns:
            Geometric algebra multivector
        """
        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} != GA dimension {self.dimension}"
            )
        
        # Dynamically create the multivector from basis blades
        mv = sum(
            component * self.ga_blades[f"e{i+1}"]
            for i, component in enumerate(vector)
        )
        return mv
    
    def primitive_to_ga_representation(self, primitive: NDPrimitive):
        """
        Convert geometric primitive to GA representation (a blade).
        
        Parameters:
            primitive: NDPrimitive to convert
            
        Returns:
            Geometric algebra blade
        """
        if isinstance(primitive, NDVertex):
            return self.vector_to_multivector(primitive.coordinates)
        
        # Represent k-primitives as k-blades
        vertices = [self.vector_to_multivector(v) for v in primitive.vertices]
        
        if len(vertices) < 2:
            return vertices[0]
        
        v0 = vertices[0]
        blade = vertices[1] - v0
        
        # Find k linearly independent vectors
        dim = primitive.affine_dimension()
        if dim == 0:
            return v0
        
        vec_count = 1
        for i in range(2, len(vertices)):
            if vec_count >= dim:
                break
            blade = blade ^ (vertices[i] - v0)
            vec_count += 1
        
        return blade
    
    def ga_dihedral_angle(self, face1: NDPrimitive, face2: NDPrimitive) -> float:
        """
        Compute dihedral angle using geometric algebra.
        
        Parameters:
            face1, face2: Two faces
            
        Returns:
            Angle in radians
        """
        # Convert to GA representations
        blade1 = self.primitive_to_ga_representation(face1)
        blade2 = self.primitive_to_ga_representation(face2)
        
        try:
            # Normalize blades
            blade1_norm = blade1 / abs(blade1)
            blade2_norm = blade2 / abs(blade2)
            
            # Use pseudo-scalar dual to obtain normals
            I = self.ga_layout.pseudoScalar
            normal1 = (blade1_norm * I).normal()
            normal2 = (blade2_norm * I).normal()
            
            # Inner product of normals → cos(angle)
            cos_angle = float((normal1 | normal2))  # scalar part
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle = np.arccos(cos_angle)
            
            return angle
        
        except Exception as e:
            raise NDGeometryError(f"GA dihedral computation failed: {str(e)}")
    
    def compare_approaches(self, la_system: NDAngularSystem,
                          face1: NDPrimitive, face2: NDPrimitive,
                          intersection: NDPrimitive) -> Dict[str, Any]:
        """
        Compare linear algebra vs geometric algebra approaches.
        
        Parameters:
            la_system: NDAngularSystem for LA approach
            face1, face2: Two faces
            intersection: Their intersection
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        # Linear algebra approach (CANS-nD)
        try:
            la_angle = la_system.k_dihedral_angle(face1, face2, intersection)
            results["linear_algebra"] = {"angle": la_angle, "success": True}
        except Exception as e:
            results["linear_algebra"] = {"error": str(e), "success": False}
        
        # Geometric Algebra approach
        try:
            ga_angle = self.ga_dihedral_angle(face1, face2)
            results["geometric_algebra"] = {"angle": ga_angle, "success": True}
        except Exception as e:
            results["geometric_algebra"] = {"error": str(e), "success": False}
        
        # Compare results when both succeed
        if (
            results["linear_algebra"]["success"]
            and results["geometric_algebra"]["success"]
        ):
            diff = abs(results["linear_algebra"]["angle"]
                      - results["geometric_algebra"]["angle"])
            results["comparison"] = {
                "absolute_difference": diff,
                "agreement": diff < 1e-10,
                "relative_error": diff / max(abs(results["linear_algebra"]["angle"]), 1e-10),
            }
        
        return results
    
    def rotor_from_vectors(self, v1: np.ndarray, v2: np.ndarray):
        """
        Create a rotor that rotates v1 to v2.
        
        Parameters:
            v1, v2: Vectors
            
        Returns:
            Rotor (geometric algebra element)
        """
        mv1 = self.vector_to_multivector(v1)
        mv2 = self.vector_to_multivector(v2)
        
        # Normalize
        mv1 = mv1 / abs(mv1)
        mv2 = mv2 / abs(mv2)
        
        # Rotor: R = sqrt((1 + v2·v1) + v2∧v1)
        # Simplified: R = 1 + v2*v1, then normalize
        rotor = 1 + mv2 * mv1
        rotor = rotor / abs(rotor)
        
        return rotor
    
    def apply_rotor(self, rotor, vector: np.ndarray) -> np.ndarray:
        """
        Apply a rotor to a vector.
        
        Parameters:
            rotor: Geometric algebra rotor
            vector: Vector to rotate
            
        Returns:
            Rotated vector
        """
        mv = self.vector_to_multivector(vector)
        
        # Apply rotor: v' = R v R†
        rotor_conj = ~rotor
        result_mv = rotor * mv * rotor_conj
        
        # Extract vector components
        result = np.zeros(self.dimension)
        for i in range(self.dimension):
            blade_name = f"e{i+1}"
            result[i] = float(result_mv[self.ga_blades[blade_name]])
        
        return result


class GAFallback:
    """
    Fallback implementation when Clifford is not available.
    Provides basic functionality without geometric algebra.
    """
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        print("Warning: Clifford package not available. GA integration disabled.")
    
    def vector_to_multivector(self, vector: np.ndarray):
        raise NotImplementedError("Geometric algebra not available")
    
    def primitive_to_ga_representation(self, primitive: NDPrimitive):
        raise NotImplementedError("Geometric algebra not available")
    
    def ga_dihedral_angle(self, face1: NDPrimitive, face2: NDPrimitive) -> float:
        raise NotImplementedError("Geometric algebra not available")
    
    def compare_approaches(self, la_system: NDAngularSystem,
                          face1: NDPrimitive, face2: NDPrimitive,
                          intersection: NDPrimitive) -> Dict[str, Any]:
        return {
            "linear_algebra": {"error": "GA not available", "success": False},
            "geometric_algebra": {"error": "Clifford package not installed", "success": False},
        }


# Use fallback if Clifford is not available
if not CLIFFORD_AVAILABLE:
    GeometricAlgebraIntegration = GAFallback
