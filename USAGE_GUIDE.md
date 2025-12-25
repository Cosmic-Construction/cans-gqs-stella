# CANS/GQS Usage Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Basic Concepts](#basic-concepts)
4. [CANS-3D Tutorial](#cans-3d-tutorial)
5. [CANS-nD Tutorial](#cans-nd-tutorial)
6. [GQS Tutorial](#gqs-tutorial)
7. [Advanced Topics](#advanced-topics)
8. [API Reference](#api-reference)

## Introduction

The Comprehensive Angular Naming System (CANS) and Geodesic Query System (GQS) provide a unified framework for computing and reasoning about geometric angles in 3D and higher dimensions.

### Why CANS/GQS?

- **Unambiguous Notation**: Clear, formal notation for all angular quantities
- **De-conflation**: Explicit separation of solid angle Ω(V) and vertex defect δ(V)
- **Scalable**: Works in 3D, 4D, and arbitrary n-dimensional spaces
- **Validated**: Implements classical theorems (Gauss-Bonnet, Girard's theorem)

## Installation

```bash
# Clone repository
git clone https://github.com/Cosmic-Construction/The-Comprehensive-Angular-Naming-System-CANS-and-the-Geodesic-Query-System-GQS-.git

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Basic Concepts

### Angular Quantities in CANS-3D

1. **Planar Angle** `A_p(V_i, F_j; V_a, V_e)`: 2D angle at vertex on a face
2. **Dihedral Angle** `A_d(E_k)`: 3D angle between faces at an edge
3. **Solid Angle** `Ω(V_i)`: 3D conical spread at a vertex (steradians)
4. **Vertex Defect** `δ(V_i)`: 2D intrinsic curvature at a vertex (radians)

### The "Massive Coincidence"

For many polyhedra (like cubes), Ω(V) and δ(V) have the same numerical value:
- Ω(V) = π/2 **steradians** (3D extrinsic measure)
- δ(V) = π/2 **radians** (2D intrinsic measure)

CANS explicitly treats these as distinct concepts!

## CANS-3D Tutorial

### Example 1: Computing Angles on a Cube

```python
from cans_gqs import PolyhedralAngleSystem
import numpy as np

# Define cube vertices
vertices = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
]

# Define faces (counterclockwise from outside)
faces = [
    [0, 1, 2, 3],  # Bottom
    [4, 7, 6, 5],  # Top
    [0, 4, 5, 1],  # Front
    [2, 6, 7, 3],  # Back
    [0, 3, 7, 4],  # Left
    [1, 5, 6, 2],  # Right
]

# Create CANS-3D system
cube = PolyhedralAngleSystem(vertices, faces)

# Compute vertex defect (2D intrinsic curvature)
defect = cube.vertex_defect("V_0")
print(f"Vertex defect: {np.degrees(defect):.2f}°")  # 90.00°

# Compute solid angle (3D extrinsic measure)
solid_angle = cube.solid_angle("V_0")
print(f"Solid angle: {solid_angle:.4f} sr")  # π/2 ≈ 1.5708 sr

# Compute dihedral angle
dihedral = cube.dihedral_angle("E_0")
print(f"Dihedral angle: {np.degrees(dihedral):.2f}°")  # ±90°

# Verify Gauss-Bonnet theorem
total_defect, expected, is_valid = cube.verify_gauss_bonnet()
print(f"Gauss-Bonnet: {is_valid}")  # True
```

### Example 2: Molecular Geometry (Torsion Angles)

```python
# Create a simple molecular chain
vertices = [
    (0.0, 0.0, 0.0),  # Atom 1
    (1.5, 0.0, 0.0),  # Atom 2
    (2.5, 1.2, 0.0),  # Atom 3
    (4.0, 1.2, 0.5),  # Atom 4
]

system = PolyhedralAngleSystem(vertices, [])

# Compute torsion angle (φ, ψ, or ω in proteins)
torsion = system.torsion_angle(0, 1, 2, 3)
print(f"Torsion angle: {np.degrees(torsion):.2f}°")

# Compute bond angle
angle = system.vector_angle(0, 1, 1, 2)
print(f"Bond angle: {np.degrees(angle):.2f}°")
```

## CANS-nD Tutorial

### Example: 4D Tesseract

```python
from cans_gqs import NDVertex, NDEdge, NDFace, NDAngularSystem
import numpy as np

# Create 4D angular system
system = NDAngularSystem(4)

# Add 4D vertices
v1 = NDVertex(np.array([1, 0, 0, 0]), "V_1")
v2 = NDVertex(np.array([0, 1, 0, 0]), "V_2")
system.add_primitive(v1)
system.add_primitive(v2)

# Create 4D edge
edge = NDEdge([v1.coordinates, v2.coordinates], "E_1")
system.add_primitive(edge)

# Compute edge properties
print(f"Edge length: {edge.length():.4f}")
print(f"Direction: {edge.direction_vector}")

# Create 4D face (square in 4D)
vertices_4d = [
    np.array([0, 0, 0, 0]),
    np.array([1, 0, 0, 0]),
    np.array([1, 1, 0, 0]),
    np.array([0, 1, 0, 0]),
]
face = NDFace(vertices_4d, "F_1")
system.add_primitive(face)

# Compute face normal in 4D
normal = face.normal_vector()
print(f"4D normal vector: {normal}")
```

### Total Solid Angles for k-Spheres

```python
# Total solid angle formulas: 2π^(k/2) / Γ(k/2)
print(f"Circle (1-sphere): {system._total_solid_angle_k_minus_1(2):.6f}")
# Output: 6.283185 (2π)

print(f"Sphere (2-sphere): {system._total_solid_angle_k_minus_1(3):.6f}")
# Output: 12.566371 (4π)

print(f"3-sphere: {system._total_solid_angle_k_minus_1(4):.6f}")
# Output: 19.739209 (2π²)
```

## GQS Tutorial

### Basic GQS Setup

```python
from cans_gqs import GeodesicQuerySystem

# Create GQS in 3D
gqs = GeodesicQuerySystem(dimension=3)

# Add entities (particles, rigid bodies, etc.)
class Particle:
    def __init__(self, position, velocity, mass):
        self.position = position
        self.velocity = velocity
        self.mass = mass

particle = Particle(
    position=np.array([0, 0, 0]),
    velocity=np.array([1, 0, 0]),
    mass=1.0
)

gqs.add_entity("particle_1", particle)

# Add forces
def gravity_force(entity, entities):
    return np.array([0, 0, -9.81 * entity.mass])

gqs.add_force("gravity", gravity_force)

# Run simulation
gqs.timestep = 0.01
new_entities = gqs.simulation_step()
```

## Advanced Topics

### Custom Mesh Analysis

```python
# Load mesh from file (e.g., OBJ format)
import numpy as np

def load_obj_mesh(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                coords = [float(x) for x in line.split()[1:]]
                vertices.append(tuple(coords))
            elif line.startswith('f '):
                face = [int(x.split('/')[0]) - 1 for x in line.split()[1:]]
                faces.append(face)
    return vertices, faces

# Analyze mesh
vertices, faces = load_obj_mesh('mesh.obj')
mesh_system = PolyhedralAngleSystem(vertices, faces)

# Compute all vertex defects
defects = [mesh_system.vertex_defect(f"V_{i}") 
           for i in range(len(vertices))]

# Find high-curvature vertices
high_curvature = [(i, d) for i, d in enumerate(defects) 
                  if abs(d) > np.pi/4]
print(f"High curvature vertices: {high_curvature}")
```

### Mesh Quality Metrics

```python
# Check for degenerate angles
for edge_label in cube.edges.keys():
    angle = cube.dihedral_angle(edge_label)
    if abs(np.degrees(angle)) < 15:  # Too acute
        print(f"Warning: {edge_label} has acute dihedral angle {np.degrees(angle):.2f}°")
```

## API Reference

### PolyhedralAngleSystem

**Constructor:**
```python
PolyhedralAngleSystem(vertices: List[Tuple], faces: List[List[int]])
```

**Methods:**
- `planar_angle(v_prev, v_curr, v_next, face) -> float`
- `vertex_defect(vertex: str) -> float`
- `solid_angle(vertex: str) -> float`
- `dihedral_angle(edge: str) -> float`
- `vector_angle(v1_start, v1_end, v2_start, v2_end) -> float`
- `torsion_angle(v1, v2, v3, v4) -> float`
- `verify_gauss_bonnet() -> Tuple[float, float, bool]`

### NDAngularSystem

**Constructor:**
```python
NDAngularSystem(dimension: int)
```

**Methods:**
- `add_primitive(primitive: NDPrimitive)`
- `k_dihedral_angle(face1, face2, intersection) -> float`
- `solid_angle_kd(vertex, container, k) -> float`

### NDPrimitive Classes

- `NDVertex(coordinates, label=None)`
- `NDEdge(vertices, label=None)`
- `NDFace(vertices, label=None)`
- `NDHyperface(vertices, label=None)`
- `NDPolytope(dimension, vertices, faces, label=None)`

### GeodesicQuerySystem

**Constructor:**
```python
GeodesicQuerySystem(dimension: int)
```

**Methods:**
- `add_entity(label, entity)`
- `add_force(label, force_function)`
- `add_constraint(label, constraint)`
- `simulation_step() -> Dict`

## Examples Gallery

See the `examples/` directory for complete working examples:
- `example_01_cube.py`: Unit cube validation
- `example_02_tesseract.py`: 4D hypercube
- `example_03_molecular.py`: Molecular torsion angles

## Further Reading

For theoretical background, see the comprehensive paper in the `all in one` file which covers:
- Mathematical foundations
- Implementation details
- Application domains
- Performance considerations
