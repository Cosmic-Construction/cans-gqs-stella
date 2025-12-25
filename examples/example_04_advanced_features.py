"""
Example 4: Advanced Features Demonstration
Demonstrates new CANS-GQS features including:
- 4D polytopes (5-cell and tesseract)
- Quantum 4D angular systems
- Numba optimization
- Geometric algebra integration (if available)
- Strategic positioning and query language
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cans_gqs.cans_nd.polytope_4d import (
    Polytope4DAngularSystem,
    create_5_cell,
    create_tesseract,
    verify_4d_gauss_bonnet,
)
from cans_gqs.applications.quantum_data_apps import Quantum4DAngularSystem, Data4DAngularAnalysis
from cans_gqs.applications.strategic_positioning import (
    GQSQueryLanguageDemo,
    GQS3DPositioning,
    AcademicCredibilityFramework,
)
from cans_gqs.utils.numba_kernels import NUMBA_AVAILABLE, numba_vector_angle, numba_planar_angle
from cans_gqs.utils.geometric_algebra import CLIFFORD_AVAILABLE, GeometricAlgebraIntegration


def demo_4d_polytopes():
    """Demonstrate 4D polytope systems"""
    print("=" * 70)
    print("4D Polytope Systems")
    print("=" * 70)
    print()
    
    # Test 5-cell (4-simplex)
    print("1. Regular 5-Cell (4-Simplex)")
    print("-" * 70)
    vertices_5cell, cells_5cell = create_5_cell()
    system_5cell = Polytope4DAngularSystem(vertices_5cell, cells_5cell)
    
    print(f"Created 5-cell with {len(vertices_5cell)} vertices and {len(cells_5cell)} cells")
    print(f"Expected: 5 vertices, 5 tetrahedral cells")
    print()
    
    # Test tesseract (4D hypercube)
    print("2. Tesseract (4D Hypercube)")
    print("-" * 70)
    vertices_tesseract, cells_tesseract = create_tesseract()
    system_tesseract = Polytope4DAngularSystem(vertices_tesseract, cells_tesseract)
    
    print(f"Created tesseract with {len(vertices_tesseract)} vertices and {len(cells_tesseract)} cells")
    print(f"Expected: 16 vertices, 8 cubic cells")
    print()
    
    # Test 4D Gauss-Bonnet
    print("3. 4D Gauss-Bonnet Verification")
    print("-" * 70)
    try:
        is_valid = verify_4d_gauss_bonnet(system_tesseract)
        print(f"4D Gauss-Bonnet for tesseract: {'PASS ✓' if is_valid else 'FAIL ✗'}")
    except Exception as e:
        print(f"4D Gauss-Bonnet verification: {e}")
    print()


def demo_quantum_4d():
    """Demonstrate quantum 4D angular system"""
    print("=" * 70)
    print("Quantum 4D Angular System")
    print("=" * 70)
    print()
    
    quantum_system = Quantum4DAngularSystem()
    
    # Create a quantum state
    print("1. Bell States (Maximally Entangled)")
    print("-" * 70)
    bell_states = quantum_system.bell_states_4d()
    
    for name, state in bell_states.items():
        ent_angle = quantum_system.entanglement_angle_4d(state)
        print(f"{name:12s}: entanglement angle = {np.degrees(ent_angle):.2f}° (π/2 = 90°)")
    print()
    
    # Test custom state
    print("2. Custom Quantum State")
    print("-" * 70)
    angles = (np.pi/4, 0, np.pi/3, 0, np.pi/6)
    custom_state = quantum_system.quantum_state_4d(angles, label="custom")
    ent_angle = quantum_system.entanglement_angle_4d(custom_state)
    print(f"Custom state angles: θ₁={np.degrees(angles[0]):.1f}°, φ₁={np.degrees(angles[1]):.1f}°, "
          f"θ₂={np.degrees(angles[2]):.1f}°, φ₂={np.degrees(angles[3]):.1f}°, ψ={np.degrees(angles[4]):.1f}°")
    print(f"Entanglement angle: {np.degrees(ent_angle):.2f}°")
    print()


def demo_data_analysis_4d():
    """Demonstrate 4D data analysis"""
    print("=" * 70)
    print("4D Data Angular Analysis")
    print("=" * 70)
    print()
    
    # Generate random 4D data
    np.random.seed(42)
    data_4d = np.random.randn(50, 4)
    
    print("Analyzing 50 random 4D data points...")
    print()
    
    try:
        analyzer = Data4DAngularAnalysis(data_4d)
        results = analyzer.analyze_data_angles()
        
        print(f"Convex hull vertices: {results['num_vertices']}")
        print(f"Convex hull cells: {results['num_cells']}")
        print(f"Effective dimensionality: {results['dimensionality_metrics']:.2f}")
        print()
    except Exception as e:
        print(f"Analysis failed: {e}")
        print()


def demo_numba_performance():
    """Demonstrate Numba-optimized kernels"""
    print("=" * 70)
    print("Numba Performance Optimization")
    print("=" * 70)
    print()
    
    print(f"Numba available: {NUMBA_AVAILABLE}")
    
    if NUMBA_AVAILABLE:
        # Test numba functions
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])
        
        # Vector angle
        angle = numba_vector_angle(v1, v2)
        print(f"Angle between [1,0,0] and [0,1,0]: {np.degrees(angle):.2f}° (expected: 90.00°)")
        
        # Planar angle
        angle = numba_planar_angle(v1, np.zeros(3), v2)
        print(f"Planar angle at origin: {np.degrees(angle):.2f}° (expected: 90.00°)")
    else:
        print("Install numba for performance optimizations: pip install numba")
    
    print()


def demo_geometric_algebra():
    """Demonstrate geometric algebra integration"""
    print("=" * 70)
    print("Geometric Algebra Integration")
    print("=" * 70)
    print()
    
    print(f"Clifford (GA) library available: {CLIFFORD_AVAILABLE}")
    
    if CLIFFORD_AVAILABLE:
        try:
            ga = GeometricAlgebraIntegration(3)
            
            # Test vector conversion
            v = np.array([1.0, 2.0, 3.0])
            mv = ga.vector_to_multivector(v)
            print(f"Converted vector {v} to multivector")
            print(f"Multivector: {mv}")
        except Exception as e:
            print(f"GA integration test failed: {e}")
    else:
        print("Install clifford for geometric algebra integration: pip install clifford")
    
    print()


def demo_query_language():
    """Demonstrate query language"""
    print("=" * 70)
    print("GQS Query Language")
    print("=" * 70)
    print()
    
    demo = GQSQueryLanguageDemo()
    
    print("1. Molecular Dynamics Queries")
    print("-" * 70)
    md_queries = demo.molecular_geometry_queries()
    for name, query in list(md_queries.items())[:2]:
        print(f"{name}:")
        print(f"  {query}")
        print()
    
    print("2. FEA Mesh Quality Queries")
    print("-" * 70)
    fea_queries = demo.fea_mesh_quality_queries()
    for name, query in list(fea_queries.items())[:2]:
        print(f"{name}:")
        print(f"  {query}")
        print()


def demo_strategic_positioning():
    """Demonstrate strategic positioning"""
    print("=" * 70)
    print("Strategic Positioning")
    print("=" * 70)
    print()
    
    positioning = GQS3DPositioning()
    
    print("1. Molecular Rosetta Stone")
    print("-" * 70)
    md_pos = positioning.molecular_rosetta_stone()
    print(f"Problem: {md_pos['problem']}")
    print(f"Solution: {md_pos['gqs_solution']}")
    print(f"Value: {md_pos['value']}")
    print()
    
    print("2. Academic Credibility")
    print("-" * 70)
    framework = AcademicCredibilityFramework()
    statements = framework.peer_review_ready_statements()
    print("Complexity claims:")
    for claim in statements['complexity_claims'][:2]:
        print(f"  • {claim}")
    print()


def main():
    """Run all demonstrations"""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 13 + "CANS-GQS Advanced Features Demo" + " " * 24 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    try:
        demo_4d_polytopes()
    except Exception as e:
        print(f"4D polytopes demo failed: {e}\n")
    
    try:
        demo_quantum_4d()
    except Exception as e:
        print(f"Quantum 4D demo failed: {e}\n")
    
    try:
        demo_data_analysis_4d()
    except Exception as e:
        print(f"Data analysis demo failed: {e}\n")
    
    try:
        demo_numba_performance()
    except Exception as e:
        print(f"Numba demo failed: {e}\n")
    
    try:
        demo_geometric_algebra()
    except Exception as e:
        print(f"Geometric algebra demo failed: {e}\n")
    
    try:
        demo_query_language()
    except Exception as e:
        print(f"Query language demo failed: {e}\n")
    
    try:
        demo_strategic_positioning()
    except Exception as e:
        print(f"Strategic positioning demo failed: {e}\n")
    
    print("=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
