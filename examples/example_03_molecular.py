"""
Example 3: Molecular Torsion Angles
Demonstrates CANS-3D torsion angle computation for molecular geometry
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cans_gqs import PolyhedralAngleSystem


def create_simple_chain():
    """Create a simple 4-atom chain for torsion angle demonstration"""
    # Create a simple molecular chain with 4 atoms
    # This could represent a backbone fragment: C-N-C-C
    vertices = [
        (0.0, 0.0, 0.0),    # V_0: First carbon
        (1.5, 0.0, 0.0),    # V_1: Nitrogen
        (2.5, 1.2, 0.0),    # V_2: Second carbon
        (4.0, 1.2, 0.5),    # V_3: Third carbon
    ]
    
    # For torsion angle computation, we don't need faces
    # but PolyhedralAngleSystem requires them
    faces = []
    
    return vertices, faces


def main():
    print("=" * 70)
    print("CANS-3D: Molecular Torsion Angle Example")
    print("=" * 70)
    print()
    
    # Create molecular chain
    vertices, faces = create_simple_chain()
    system = PolyhedralAngleSystem(vertices, faces)
    
    print("Molecular Chain Geometry:")
    print("-" * 70)
    for i, (label, coord) in enumerate(system.vertices.items()):
        print(f"  {label}: ({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})")
    print()
    
    # Test 1: Torsion angle along the chain
    print("Test 1: Torsion Angle (Dihedral Angle along 4-atom chain)")
    print("-" * 70)
    print("  Computing torsion angle for atoms V_0 - V_1 - V_2 - V_3")
    print("  This is the angle between planes (V_0, V_1, V_2) and (V_1, V_2, V_3)")
    print()
    
    torsion = system.torsion_angle(0, 1, 2, 3)
    print(f"  Torsion angle: {torsion:.6f} radians")
    print(f"  Torsion angle: {np.degrees(torsion):.2f}°")
    print()
    
    # Test 2: Vector angles
    print("Test 2: Vector Angles")
    print("-" * 70)
    
    # Angle between bond V_0->V_1 and bond V_1->V_2
    angle_012 = system.vector_angle(0, 1, 1, 2)
    print(f"  Angle V_0-V_1-V_2: {np.degrees(angle_012):.2f}°")
    
    # Angle between bond V_1->V_2 and bond V_2->V_3
    angle_123 = system.vector_angle(1, 2, 2, 3)
    print(f"  Angle V_1-V_2-V_3: {np.degrees(angle_123):.2f}°")
    print()
    
    # Test 3: Different configurations
    print("Test 3: Torsion Angles for Different Configurations")
    print("-" * 70)
    
    # Eclipsed configuration (torsion ~ 0°)
    vertices_eclipsed = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 1.0),
    ]
    system_eclipsed = PolyhedralAngleSystem(vertices_eclipsed, [])
    torsion_eclipsed = system_eclipsed.torsion_angle(0, 1, 2, 3)
    print(f"  Eclipsed: {np.degrees(torsion_eclipsed):.2f}° (expected: ~0°)")
    
    # Staggered configuration (torsion ~ 180°)
    vertices_staggered = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 1.0, 0.0),
        (3.0, 1.0, 1.0),
    ]
    system_staggered = PolyhedralAngleSystem(vertices_staggered, [])
    torsion_staggered = system_staggered.torsion_angle(0, 1, 2, 3)
    print(f"  Staggered: {np.degrees(torsion_staggered):.2f}° (expected: ~0°-180°)")
    
    # 90° configuration
    vertices_90 = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 1.0, 0.0),
        (2.0, 1.0, 1.0),
    ]
    system_90 = PolyhedralAngleSystem(vertices_90, [])
    torsion_90 = system_90.torsion_angle(0, 1, 2, 3)
    print(f"  90° twist: {np.degrees(torsion_90):.2f}° (expected: ~90°)")
    print()
    
    # Application context
    print("Application Context: Protein Backbone Angles")
    print("-" * 70)
    print("  In molecular dynamics, torsion angles are crucial:")
    print("  - φ (phi): N-Cα torsion angle")
    print("  - ψ (psi): Cα-C torsion angle")
    print("  - ω (omega): peptide bond torsion (~180° for trans)")
    print()
    print("  CANS provides a unified, unambiguous notation:")
    print("    A_t(E_φ) for phi angle")
    print("    A_t(E_ψ) for psi angle")
    print("    A_t(E_ω) for omega angle")
    print()
    
    print("=" * 70)
    print("Summary: CANS-3D torsion angles work for molecular geometry")
    print("=" * 70)


if __name__ == "__main__":
    main()
