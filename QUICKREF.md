# CANS/GQS Quick Reference

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## CANS-3D Quick Start

```python
from cans_gqs import PolyhedralAngleSystem

# Create system
system = PolyhedralAngleSystem(vertices, faces)

# Angular computations
defect = system.vertex_defect("V_0")           # Vertex defect (radians)
solid = system.solid_angle("V_0")              # Solid angle (steradians)
dihedral = system.dihedral_angle("E_0")        # Dihedral angle
planar = system.planar_angle(v_p, v_c, v_n, f) # Planar angle
vector = system.vector_angle(v1s, v1e, v2s, v2e) # Vector angle
torsion = system.torsion_angle(v1, v2, v3, v4)  # Torsion (4-point)

# Validation
total, expected, valid = system.verify_gauss_bonnet()
```

## CANS-nD Quick Start

```python
from cans_gqs import NDVertex, NDEdge, NDFace, NDAngularSystem
import numpy as np

# Create n-dimensional system
system = NDAngularSystem(dimension=4)

# Create and add primitives
vertex = NDVertex(np.array([1, 0, 0, 0]), "V_1")
edge = NDEdge([coords1, coords2], "E_1")
face = NDFace([c1, c2, c3, c4], "F_1")

system.add_primitive(vertex)
system.add_primitive(edge)
system.add_primitive(face)

# Compute k-dihedral angle
angle = system.k_dihedral_angle(face1, face2, shared_edge)

# Compute k-dimensional solid angle
solid_angle = system.solid_angle_kd(vertex, container, k=2)
```

## GQS Quick Start

```python
from cans_gqs import GeodesicQuerySystem

# Create GQS
gqs = GeodesicQuerySystem(dimension=3)

# Add entities
gqs.add_entity("particle_1", particle_object)

# Add forces
def force_func(entity, all_entities):
    return force_vector

gqs.add_force("gravity", force_func)

# Add constraints
gqs.add_constraint("angle_constraint", constraint_spec)

# Simulate
gqs.timestep = 0.01
new_entities = gqs.simulation_step()
```

## Key Concepts

### Angular Quantities

| Symbol | Name | Dimension | Units | Description |
|--------|------|-----------|-------|-------------|
| A_p | Planar angle | 2D | radians | Interior angle on face |
| A_d | Dihedral angle | 3D | radians | Angle between faces at edge |
| Ω | Solid angle | 3D | steradians | Conical spread at vertex (extrinsic) |
| δ | Vertex defect | 2D | radians | Intrinsic curvature at vertex |
| A_v | Vector angle | 3D | radians | Angle between vectors |
| A_t | Torsion angle | 3D | radians | 4-point dihedral |

### Gauss-Bonnet Theorem

**3D:** Σ δ(V_i) = 2πχ where χ = V - E + F

**4D:** Σ δ_4(V_i) = 2π²χ

### Total Solid Angles

| Dimension | Manifold | Formula | Value |
|-----------|----------|---------|-------|
| k=2 | Circle (1-sphere) | 2π | 6.283 |
| k=3 | Sphere (2-sphere) | 4π | 12.566 |
| k=4 | 3-sphere | 2π² | 19.739 |
| k (general) | (k-1)-sphere | 2π^(k/2) / Γ(k/2) | - |

## Typical Use Cases

### Mesh Quality (FEA)
```python
for edge in system.edges:
    angle = system.dihedral_angle(edge)
    if abs(np.degrees(angle)) < 15:
        print(f"Poor quality: {edge}")
```

### Molecular Dynamics
```python
# Ramachandran plot angles
phi = system.torsion_angle(N, Ca, C, N_next)
psi = system.torsion_angle(Ca, C, N_next, Ca_next)
omega = system.torsion_angle(Ca, C, N_next, Ca_next)
```

### Curvature Analysis
```python
defects = [system.vertex_defect(f"V_{i}") 
           for i in range(len(vertices))]
mean_curvature = np.mean(defects)
gaussian_curvature = np.sum(defects) / (2 * np.pi)
```

## Common Patterns

### Cube (Reference)
- Planar angles: 90°
- Vertex defects: 90° (π/2 rad)
- Solid angles: π/2 sr
- Dihedral angles: ±90°

### Tetrahedron
- Planar angles: 60°
- Vertex defects: π rad
- Solid angles: (varies by regularity)
- χ = 2

### Tesseract (4D)
- 16 vertices
- 32 edges
- 24 faces (2D)
- 8 cells (3D)

## Running Examples

```bash
python examples/example_01_cube.py        # Cube validation
python examples/example_02_tesseract.py   # 4D tesseract
python examples/example_03_molecular.py   # Molecular torsion

python tests/test_cans_3d.py              # Run unit tests
```

## Troubleshooting

**ImportError**: Ensure you've run `pip install -r requirements.txt`

**Dimension mismatch**: Check that all vertices have consistent dimensionality

**Gauss-Bonnet fails**: Verify face orientations are consistent (counterclockwise from outside)

**Negative defects**: Normal for hyperbolic/saddle surfaces; not an error

## Resources

- `README.md`: Overview and installation
- `USAGE_GUIDE.md`: Detailed tutorials
- `all in one`: Complete mathematical theory
- `examples/`: Working code examples
- `tests/`: Unit tests and validation

## Citation

```
ContributorX Ltd. (2024)
The Comprehensive Angular Naming System (CANS) and 
the Geodesic Query System (GQS)
```
