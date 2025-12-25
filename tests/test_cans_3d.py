"""
Basic tests for CANS-3D implementation
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cans_gqs import PolyhedralAngleSystem


def test_cube_planar_angles():
    """Test that cube planar angles are all 90 degrees"""
    # Create unit cube
    vertices = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
    ]
    faces = [
        [0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
        [2, 6, 7, 3], [0, 3, 7, 4], [1, 5, 6, 2]
    ]
    
    cube = PolyhedralAngleSystem(vertices, faces)
    
    # Check planar angles at vertex 0
    for face_label, face in cube.faces.items():
        if 0 in face:
            idx = face.index(0)
            n = len(face)
            v_prev = face[(idx - 1) % n]
            v_next = face[(idx + 1) % n]
            angle = cube.planar_angle(v_prev, 0, v_next, face)
            
            # All angles should be π/2 (90 degrees)
            assert np.isclose(angle, np.pi / 2, rtol=1e-10), \
                f"Expected π/2, got {angle}"
    
    print("✓ test_cube_planar_angles passed")


def test_cube_vertex_defect():
    """Test that cube vertex defect is π/2"""
    vertices = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
    ]
    faces = [
        [0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
        [2, 6, 7, 3], [0, 3, 7, 4], [1, 5, 6, 2]
    ]
    
    cube = PolyhedralAngleSystem(vertices, faces)
    
    # Check vertex defect at each corner (should all be π/2)
    for i in range(8):
        defect = cube.vertex_defect(f"V_{i}")
        assert np.isclose(defect, np.pi / 2, rtol=1e-10), \
            f"Vertex {i}: Expected π/2, got {defect}"
    
    print("✓ test_cube_vertex_defect passed")


def test_cube_gauss_bonnet():
    """Test Gauss-Bonnet theorem on cube"""
    vertices = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
    ]
    faces = [
        [0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
        [2, 6, 7, 3], [0, 3, 7, 4], [1, 5, 6, 2]
    ]
    
    cube = PolyhedralAngleSystem(vertices, faces)
    
    total_defect, expected, is_valid = cube.verify_gauss_bonnet()
    
    assert is_valid, \
        f"Gauss-Bonnet verification failed: {total_defect} != {expected}"
    
    print("✓ test_cube_gauss_bonnet passed")


def test_cube_solid_angle():
    """Test that cube solid angle at corner is π/2 sr"""
    vertices = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
    ]
    faces = [
        [0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
        [2, 6, 7, 3], [0, 3, 7, 4], [1, 5, 6, 2]
    ]
    
    cube = PolyhedralAngleSystem(vertices, faces)
    
    # Check solid angle at each corner
    for i in range(8):
        solid_angle = cube.solid_angle(f"V_{i}")
        # Solid angle at cube corner is π/2 steradians
        assert np.isclose(solid_angle, np.pi / 2, rtol=1e-2), \
            f"Vertex {i}: Expected π/2, got {solid_angle}"
    
    print("✓ test_cube_solid_angle passed")


if __name__ == "__main__":
    print("Running CANS-3D tests...")
    print()
    
    test_cube_planar_angles()
    test_cube_vertex_defect()
    test_cube_gauss_bonnet()
    test_cube_solid_angle()
    
    print()
    print("All tests passed! ✓")
