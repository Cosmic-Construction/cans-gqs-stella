"""Utilities for CANS/GQS framework"""

from .numba_kernels import (
    numba_planar_angle,
    numba_vector_angle,
    numba_dihedral_angle,
    numba_orthogonal_complement,
    numba_total_solid_angle,
    numba_spherical_excess,
    numba_cross_product_3d,
    numba_vertex_defect,
    numba_normalize_vector,
    NUMBA_AVAILABLE,
)
from .geometric_algebra import GeometricAlgebraIntegration, CLIFFORD_AVAILABLE

__all__ = [
    "numba_planar_angle",
    "numba_vector_angle",
    "numba_dihedral_angle",
    "numba_orthogonal_complement",
    "numba_total_solid_angle",
    "numba_spherical_excess",
    "numba_cross_product_3d",
    "numba_vertex_defect",
    "numba_normalize_vector",
    "NUMBA_AVAILABLE",
    "GeometricAlgebraIntegration",
    "CLIFFORD_AVAILABLE",
]
