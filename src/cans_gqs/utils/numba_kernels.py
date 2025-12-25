"""
CANS-GQS: Numba-Optimized Computational Kernels
Part 3 implementation - high-performance numerical routines

This module provides Numba JIT-compiled versions of core angular operations
for performance-critical applications.
"""

import numpy as np
from typing import Tuple

try:
    import numba
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Provide dummy decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True)
def numba_planar_angle(v_prev: np.ndarray, v_curr: np.ndarray,
                      v_next: np.ndarray) -> float:
    """
    Numba-optimized planar angle computation.
    
    Computes the angle at v_curr between edges to v_prev and v_next.
    
    Parameters:
        v_prev: Previous vertex coordinates
        v_curr: Current vertex coordinates
        v_next: Next vertex coordinates
        
    Returns:
        Angle in radians
    """
    u = v_prev - v_curr
    v = v_next - v_curr
    
    # Dynamic shape handling
    n = u.shape[0]
    
    # Compute norms with robust handling
    norm_u = 0.0
    norm_v = 0.0
    dot_product = 0.0
    
    for i in range(n):
        norm_u += u[i] * u[i]
        norm_v += v[i] * v[i]
        dot_product += u[i] * v[i]
    
    norm_u = np.sqrt(norm_u)
    norm_v = np.sqrt(norm_v)
    
    # Avoid division by zero
    if norm_u < 1e-12 or norm_v < 1e-12:
        return 0.0
    
    # Compute cosine with clamping
    cos_theta = dot_product / (norm_u * norm_v)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    
    return np.arccos(cos_theta)


@jit(nopython=True, cache=True)
def numba_vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute angle between two vectors with robust numerical handling.
    
    Parameters:
        v1: First vector
        v2: Second vector
        
    Returns:
        Angle in radians
    """
    n = v1.shape[0]
    
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    
    for i in range(n):
        dot += v1[i] * v2[i]
        norm1 += v1[i] * v1[i]
        norm2 += v2[i] * v2[i]
    
    norm1 = np.sqrt(norm1)
    norm2 = np.sqrt(norm2)
    
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    
    cos_angle = dot / (norm1 * norm2)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    
    return np.arccos(cos_angle)


@jit(nopython=True, cache=True)
def numba_dihedral_angle(v1: np.ndarray, v2: np.ndarray, 
                        v3: np.ndarray, v4: np.ndarray) -> float:
    """
    Compute dihedral angle defined by four points.
    
    The dihedral angle is the angle between two planes:
    - Plane 1 contains v1, v2, v3
    - Plane 2 contains v2, v3, v4
    
    Parameters:
        v1, v2, v3, v4: Four points defining the dihedral angle
        
    Returns:
        Angle in radians
    """
    # Vectors
    b1 = v2 - v1
    b2 = v3 - v2
    b3 = v4 - v3
    
    # Normalize b2
    b2_norm = np.sqrt(np.sum(b2 * b2))
    if b2_norm < 1e-12:
        return 0.0
    b2_unit = b2 / b2_norm
    
    # Compute normals to the two planes
    # n1 = b1 × b2
    n1_x = b1[1] * b2[2] - b1[2] * b2[1]
    n1_y = b1[2] * b2[0] - b1[0] * b2[2]
    n1_z = b1[0] * b2[1] - b1[1] * b2[0]
    n1 = np.array([n1_x, n1_y, n1_z])
    
    # n2 = b2 × b3
    n2_x = b2[1] * b3[2] - b2[2] * b3[1]
    n2_y = b2[2] * b3[0] - b2[0] * b3[2]
    n2_z = b2[0] * b3[1] - b2[1] * b3[0]
    n2 = np.array([n2_x, n2_y, n2_z])
    
    # Normalize normals
    n1_norm = np.sqrt(np.sum(n1 * n1))
    n2_norm = np.sqrt(np.sum(n2 * n2))
    
    if n1_norm < 1e-12 or n2_norm < 1e-12:
        return 0.0
    
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    
    # Compute angle using atan2 for proper quadrant
    cos_angle = np.sum(n1 * n2)
    
    # Compute signed angle
    # m1 = n1 × b2_unit
    m1_x = n1[1] * b2_unit[2] - n1[2] * b2_unit[1]
    m1_y = n1[2] * b2_unit[0] - n1[0] * b2_unit[2]
    m1_z = n1[0] * b2_unit[1] - n1[1] * b2_unit[0]
    m1 = np.array([m1_x, m1_y, m1_z])
    
    sin_angle = np.sum(m1 * n2)
    
    angle = np.arctan2(sin_angle, cos_angle)
    
    # Return positive angle
    if angle < 0:
        angle += 2 * np.pi
    
    return angle


@jit(nopython=True, cache=True)
def numba_orthogonal_complement(basis: np.ndarray, dimension: int) -> np.ndarray:
    """
    Compute orthogonal complement with dynamic shape handling.
    
    Parameters:
        basis: Basis vectors as columns (dimension x basis_size)
        dimension: Dimension of the space
        
    Returns:
        Orthogonal complement basis (dimension x complement_size)
    """
    if basis.size == 0:
        # Return identity matrix if no basis
        result = np.eye(dimension)
        return result
    
    basis_size = basis.shape[1]
    
    # Use Gram-Schmidt to find orthogonal complement
    complement_vecs = []
    identity = np.eye(dimension)
    
    for i in range(dimension):
        vec = identity[:, i].copy()
        
        # Remove components along basis vectors
        for j in range(basis_size):
            proj = 0.0
            for k in range(dimension):
                proj += vec[k] * basis[k, j]
            for k in range(dimension):
                vec[k] -= proj * basis[k, j]
        
        # Normalize
        norm = 0.0
        for k in range(dimension):
            norm += vec[k] * vec[k]
        norm = np.sqrt(norm)
        
        if norm > 1e-12:
            vec = vec / norm
            complement_vecs.append(vec)
    
    if len(complement_vecs) == 0:
        return np.zeros((dimension, 0))
    
    # Stack vectors as columns
    result = np.zeros((dimension, len(complement_vecs)))
    for i, vec in enumerate(complement_vecs):
        result[:, i] = vec
    
    return result


@jit(nopython=True, cache=True)
def numba_total_solid_angle(k: int) -> float:
    """
    Numba-compatible total solid angle of (k-1)-sphere.
    
    Formula: Ω_{k-1} = 2π^(k/2) / Γ(k/2)
    
    Parameters:
        k: Dimension
        
    Returns:
        Total solid angle
    """
    if k == 2:
        return 2 * np.pi
    elif k == 3:
        return 4 * np.pi
    elif k == 4:
        return 2 * np.pi**2
    
    # Manual gamma approximation for integer / half-integer k/2
    k_half = k / 2.0
    
    if k % 2 == 0:  # integer argument
        # Γ(n) = (n-1)!
        g = 1.0
        n_int = int(k_half)
        for i in range(1, n_int):
            g *= float(i)
    else:  # half-integer argument
        # Γ(n + 1/2) = sqrt(π) * (2n-1)!! / 2^n
        g = np.sqrt(np.pi)
        n = int(k_half - 0.5)
        for i in range(n):
            g *= (0.5 + float(i))
    
    result = 2 * np.pi**(k / 2) / g
    return result


@jit(nopython=True, cache=True)
def numba_spherical_excess(angles: np.ndarray) -> float:
    """
    Compute spherical excess (solid angle) from spherical polygon angles.
    
    Uses Girard's theorem: E = Σαᵢ - (n-2)π
    
    Parameters:
        angles: Array of interior angles of spherical polygon
        
    Returns:
        Spherical excess (solid angle in steradians)
    """
    n = angles.shape[0]
    if n < 3:
        return 0.0
    
    angle_sum = 0.0
    for i in range(n):
        angle_sum += angles[i]
    
    excess = angle_sum - (n - 2) * np.pi
    
    return excess


@jit(nopython=True, cache=True)
def numba_cross_product_3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Fast 3D cross product.
    
    Parameters:
        a, b: 3D vectors
        
    Returns:
        Cross product a × b
    """
    result = np.zeros(3)
    result[0] = a[1] * b[2] - a[2] * b[1]
    result[1] = a[2] * b[0] - a[0] * b[2]
    result[2] = a[0] * b[1] - a[1] * b[0]
    return result


@jit(nopython=True, cache=True)
def numba_vertex_defect(planar_angles: np.ndarray) -> float:
    """
    Compute vertex defect from planar angles.
    
    δ(V) = 2π - Σθᵢ
    
    Parameters:
        planar_angles: Array of planar angles at the vertex
        
    Returns:
        Vertex defect in radians
    """
    angle_sum = 0.0
    for i in range(planar_angles.shape[0]):
        angle_sum += planar_angles[i]
    
    defect = 2 * np.pi - angle_sum
    return defect


@jit(nopython=True, cache=True)
def numba_normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector (make it unit length).
    
    Parameters:
        v: Input vector
        
    Returns:
        Normalized vector
    """
    n = v.shape[0]
    norm_sq = 0.0
    
    for i in range(n):
        norm_sq += v[i] * v[i]
    
    norm = np.sqrt(norm_sq)
    
    if norm < 1e-12:
        return v.copy()
    
    result = np.zeros_like(v)
    for i in range(n):
        result[i] = v[i] / norm
    
    return result


# Provide fallback implementations if Numba is not available
if not NUMBA_AVAILABLE:
    def planar_angle_fallback(v_prev: np.ndarray, v_curr: np.ndarray,
                             v_next: np.ndarray) -> float:
        """Fallback implementation without Numba"""
        u = v_prev - v_curr
        v = v_next - v_curr
        
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        
        if norm_u < 1e-12 or norm_v < 1e-12:
            return 0.0
        
        cos_theta = np.dot(u, v) / (norm_u * norm_v)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        return np.arccos(cos_theta)
    
    # Replace the decorated versions with fallbacks
    numba_planar_angle = planar_angle_fallback
