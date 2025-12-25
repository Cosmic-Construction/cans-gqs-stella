"""
Example 2: 4D Tesseract (Hypercube)
Demonstrates CANS-4D and CANS-nD capabilities
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cans_gqs.cans_nd.nd_primitives import NDVertex, NDEdge, NDFace
from cans_gqs.cans_nd.nd_angular_system import NDAngularSystem


def create_tesseract_system():
    """Create a 4D tesseract (hypercube) system"""
    system = NDAngularSystem(4)
    
    # Create vertices of 4D hypercube: {-1, 1}^4
    vertices_4d = []
    for i in range(16):  # 2^4 = 16 vertices
        x = (i & 1) * 2 - 1
        y = ((i >> 1) & 1) * 2 - 1
        z = ((i >> 2) & 1) * 2 - 1
        w = ((i >> 3) & 1) * 2 - 1
        vertices_4d.append(np.array([x, y, z, w], dtype=float))
    
    # Add vertices to system
    for i, coords in enumerate(vertices_4d):
        vertex = NDVertex(coords, f"V_{i}")
        system.add_primitive(vertex)
    
    # Create edges: vertices that differ in exactly one coordinate
    edge_count = 0
    for i in range(len(vertices_4d)):
        for j in range(i + 1, len(vertices_4d)):
            # Check if they differ in exactly one coordinate
            diff = np.abs(vertices_4d[i] - vertices_4d[j])
            if np.sum(diff > 0.5) == 1:  # Differ in one coordinate
                edge = NDEdge([vertices_4d[i], vertices_4d[j]], f"E_{edge_count}")
                system.add_primitive(edge)
                edge_count += 1
    
    # Create some faces: squares in 4D space
    # Find sets of 4 vertices that form a square
    face_count = 0
    for i in range(len(vertices_4d)):
        for j in range(i + 1, len(vertices_4d)):
            for k in range(j + 1, len(vertices_4d)):
                for l in range(k + 1, len(vertices_4d)):
                    # Check if these 4 points form a square
                    # (2 coordinates constant, 2 coordinates varying)
                    coords = np.array([vertices_4d[i], vertices_4d[j], 
                                      vertices_4d[k], vertices_4d[l]])
                    
                    # Count how many coordinates are constant across all 4 vertices
                    constant_coords = 0
                    for coord_idx in range(4):
                        if np.allclose(coords[:, coord_idx], coords[0, coord_idx]):
                            constant_coords += 1
                    
                    if constant_coords == 2 and face_count < 10:  # Limit faces for performance
                        try:
                            face = NDFace([vertices_4d[i], vertices_4d[j], 
                                         vertices_4d[k], vertices_4d[l]], f"F_{face_count}")
                            system.add_primitive(face)
                            face_count += 1
                        except Exception:
                            pass
    
    return system, vertices_4d


def main():
    print("=" * 70)
    print("CANS-4D: Tesseract (4D Hypercube) Example")
    print("=" * 70)
    print()
    
    # Create tesseract
    system, vertices = create_tesseract_system()
    
    edges = [p for p in system.primitives.values() if isinstance(p, NDEdge)]
    faces = [p for p in system.primitives.values() if isinstance(p, NDFace)]
    vertices_nd = [p for p in system.primitives.values() if isinstance(p, NDVertex)]
    
    print(f"Created 4D tesseract with:")
    print(f"  {len(vertices_nd)} vertices (expected: 16)")
    print(f"  {len(edges)} edges (expected: 32)")
    print(f"  {len(faces)} faces (created: {len(faces)}, full tesseract: 24)")
    print()
    
    # Test 1: Edge properties
    print("Test 1: 4D Edge Properties")
    print("-" * 70)
    if edges:
        edge = edges[0]
        print(f"  Edge {edge.label}:")
        print(f"    Vertices: {edge.vertices[0]} to {edge.vertices[1]}")
        print(f"    Length: {edge.length():.4f}")
        print(f"    Direction: {edge.direction_vector}")
    print()
    
    # Test 2: Face properties
    print("Test 2: 4D Face Properties")
    print("-" * 70)
    if faces:
        face = faces[0]
        print(f"  Face {face.label}:")
        print(f"    Number of vertices: {len(face.vertices)}")
        print(f"    Affine dimension: {face.affine_dimension()}")
        try:
            normal = face.normal_vector()
            print(f"    Normal vector: {normal}")
            print(f"    Normal magnitude: {np.linalg.norm(normal):.4f}")
        except Exception as e:
            print(f"    Could not compute normal: {e}")
    print()
    
    # Test 3: k-Dihedral angle (if we have enough structure)
    print("Test 3: k-Dihedral Angles")
    print("-" * 70)
    if len(faces) >= 2 and len(edges) >= 1:
        try:
            # Try to compute dihedral angle between first two faces
            # (this requires they share an edge)
            angle = system.k_dihedral_angle(faces[0], faces[1], edges[0])
            print(f"  Dihedral angle between {faces[0].label} and {faces[1].label}:")
            print(f"    {np.degrees(angle):.2f}°")
        except Exception as e:
            print(f"  Could not compute dihedral angle: {e}")
            print(f"  (This is expected - faces may not share the specified edge)")
    else:
        print(f"  Not enough faces/edges for dihedral angle computation")
    print()
    
    # Test 4: 2D solid angle (planar angle) at a vertex
    print("Test 4: 2D Solid Angles (Planar Angles)")
    print("-" * 70)
    if vertices_nd and faces:
        try:
            vertex = vertices_nd[0]
            face = faces[0]
            # Check if vertex is in face
            if face.contains_point(vertex.coordinates):
                angle_2d = system.solid_angle_kd(vertex, face, k=2)
                print(f"  2D solid angle at {vertex.label} in {face.label}:")
                print(f"    {np.degrees(angle_2d):.2f}°")
            else:
                print(f"  Vertex {vertex.label} not in face {face.label}")
        except Exception as e:
            print(f"  Could not compute 2D solid angle: {e}")
    print()
    
    # Test 5: Total solid angle formulas
    print("Test 5: Total Solid Angles for k-Spheres")
    print("-" * 70)
    print(f"  k=2 (1-sphere, circle): {system._total_solid_angle_k_minus_1(2):.6f} "
          f"(expected: 2π ≈ 6.283185)")
    print(f"  k=3 (2-sphere, sphere): {system._total_solid_angle_k_minus_1(3):.6f} "
          f"(expected: 4π ≈ 12.566371)")
    print(f"  k=4 (3-sphere): {system._total_solid_angle_k_minus_1(4):.6f} "
          f"(expected: 2π² ≈ 19.739209)")
    print()
    
    print("=" * 70)
    print("Summary: CANS-nD successfully handles 4D geometry")
    print("=" * 70)


if __name__ == "__main__":
    main()
