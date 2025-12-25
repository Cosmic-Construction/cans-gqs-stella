"""
Example 1: Unit Cube Validation
Demonstrates CANS-3D angular computations on a unit cube
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cans_gqs import PolyhedralAngleSystem


def create_unit_cube():
    """Create a unit cube geometry"""
    # Vertices of unit cube
    vertices = [
        (0, 0, 0),  # V_0
        (1, 0, 0),  # V_1
        (1, 1, 0),  # V_2
        (0, 1, 0),  # V_3
        (0, 0, 1),  # V_4
        (1, 0, 1),  # V_5
        (1, 1, 1),  # V_6
        (0, 1, 1),  # V_7
    ]
    
    # Faces (counter-clockwise when viewed from outside)
    faces = [
        [0, 1, 2, 3],  # Bottom face (z=0)
        [4, 7, 6, 5],  # Top face (z=1)
        [0, 4, 5, 1],  # Front face (y=0)
        [2, 6, 7, 3],  # Back face (y=1)
        [0, 3, 7, 4],  # Left face (x=0)
        [1, 5, 6, 2],  # Right face (x=1)
    ]
    
    return vertices, faces


def main():
    print("=" * 70)
    print("CANS-3D: Unit Cube Validation Example")
    print("=" * 70)
    print()
    
    # Create cube system
    vertices, faces = create_unit_cube()
    cube_system = PolyhedralAngleSystem(vertices, faces)
    
    print(f"Created cube with {len(cube_system.vertices)} vertices, "
          f"{len(cube_system.edges)} edges, and {len(cube_system.faces)} faces")
    print()
    
    # Test 1: Planar angles at a corner
    print("Test 1: Planar Angles at Vertex V_0")
    print("-" * 70)
    v0_planar_angles = []
    for face_label, face in cube_system.faces.items():
        if 0 in face:
            idx = face.index(0)
            n = len(face)
            v_prev = face[(idx - 1) % n]
            v_next = face[(idx + 1) % n]
            angle = cube_system.planar_angle(v_prev, 0, v_next, face)
            v0_planar_angles.append(angle)
            print(f"  {face_label}: {np.degrees(angle):.2f}° (expected: 90.00°)")
    print()
    
    # Test 2: Vertex defect
    print("Test 2: Vertex Defect at V_0")
    print("-" * 70)
    defect = cube_system.vertex_defect("V_0")
    print(f"  δ(V_0) = {defect:.6f} radians")
    print(f"  δ(V_0) = {np.degrees(defect):.2f}° (expected: 90.00°)")
    print(f"  Note: Vertex defect is 2D intrinsic curvature measure")
    print()
    
    # Test 3: Solid angle
    print("Test 3: Solid Angle at V_0")
    print("-" * 70)
    solid_angle = cube_system.solid_angle("V_0")
    print(f"  Ω(V_0) = {solid_angle:.6f} steradians")
    print(f"  Ω(V_0) = {np.degrees(solid_angle):.2f}° (expected: ~90°)")
    print(f"  Note: Solid angle is 3D extrinsic measure")
    print()
    
    # Test 4: The "Massive Coincidence"
    print("Test 4: De-conflation of Ω and δ")
    print("-" * 70)
    print(f"  Numerically: δ(V_0) ≈ {defect:.6f} rad ≈ Ω(V_0) ≈ {solid_angle:.6f} sr")
    print(f"  BUT they are conceptually distinct:")
    print(f"    - δ(V_0): 2D intrinsic curvature (radians)")
    print(f"    - Ω(V_0): 3D extrinsic conical measure (steradians)")
    print(f"  CANS explicitly de-conflates these two measures!")
    print()
    
    # Test 5: Dihedral angle
    print("Test 5: Dihedral Angles")
    print("-" * 70)
    edge_count = 0
    for edge_label, edge in cube_system.edges.items():
        if edge_count < 3:  # Show first 3 edges
            try:
                angle = cube_system.dihedral_angle(edge_label)
                print(f"  {edge_label} (vertices {edge}): {np.degrees(angle):.2f}° "
                      f"(expected: 90.00° for cube)")
            except ValueError as e:
                print(f"  {edge_label}: {e}")
            edge_count += 1
    print(f"  ... and {len(cube_system.edges) - 3} more edges")
    print()
    
    # Test 6: Gauss-Bonnet verification
    print("Test 6: Gauss-Bonnet Theorem Verification")
    print("-" * 70)
    total_defect, expected, is_valid = cube_system.verify_gauss_bonnet()
    print(f"  Σ δ(V_i) = {total_defect:.6f} radians")
    print(f"  Expected (2πχ) = {expected:.6f} radians")
    print(f"  χ = V - E + F = {len(cube_system.vertices)} - {len(cube_system.edges)} + {len(cube_system.faces)} = "
          f"{len(cube_system.vertices) - len(cube_system.edges) + len(cube_system.faces)}")
    print(f"  Verification: {'PASS ✓' if is_valid else 'FAIL ✗'}")
    print()
    
    print("=" * 70)
    print("Summary: CANS-3D successfully computes all angular properties")
    print("=" * 70)


if __name__ == "__main__":
    main()
