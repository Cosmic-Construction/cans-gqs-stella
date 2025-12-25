# CANS-GQS Complete Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [CANS-3D: Angular Geometry](#cans-3d-angular-geometry)
4. [CANS-nD: Higher Dimensions](#cans-nd-higher-dimensions)
5. [GQS: Physics Simulation](#gqs-physics-simulation)
6. [Applications](#applications)
7. [Performance Optimization](#performance-optimization)
8. [Examples](#examples)

## Installation

### Basic Installation
```bash
git clone https://github.com/Cosmic-Construction/CANS-GQS.git
cd CANS-GQS
pip install -e .
```

### Optional Dependencies
```bash
# For 10-100x performance boost
pip install numba

# For geometric algebra integration
pip install clifford

# For visualization
pip install matplotlib scikit-learn

# For GPU acceleration (optional)
pip install cupy-cuda11x  # or appropriate CUDA version
```

## Quick Start

### 3D Polyhedral Angles
```python
from cans_gqs import PolyhedralAngleSystem
import numpy as np

# Define a cube
vertices = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
]
faces = [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], 
         [2, 6, 7, 3], [0, 3, 7, 4], [1, 5, 6, 2]]

# Create system
cube = PolyhedralAngleSystem(vertices, faces)

# Compute angles
vertex_defect = cube.vertex_defect("V_0")  # 2D intrinsic curvature
solid_angle = cube.solid_angle("V_0")      # 3D extrinsic measure
dihedral_angle = cube.dihedral_angle("E_0")  # Angle between faces

print(f"Vertex defect: {np.degrees(vertex_defect):.2f}°")
print(f"Solid angle: {solid_angle:.4f} sr")
print(f"Dihedral angle: {np.degrees(dihedral_angle):.2f}°")

# Verify Gauss-Bonnet theorem
is_valid = cube.verify_gauss_bonnet()[2]
print(f"Gauss-Bonnet: {'PASS' if is_valid else 'FAIL'}")
```

### Physics Simulation
```python
from cans_gqs.gqs import (
    GeodesicQuerySystem,
    Particle,
    GravityForceField,
    SpringForceField,
)
import numpy as np

# Create GQS system with Verlet integrator
gqs = GeodesicQuerySystem(dimension=3, integrator="verlet")
gqs.timestep = 0.001

# Add particles
p1 = Particle(label="p1", position=np.array([0.0, 0.0, 0.0]), mass=1.0)
p2 = Particle(label="p2", position=np.array([1.5, 0.0, 0.0]), mass=1.0)
gqs.add_entity(p1)
gqs.add_entity(p2)

# Add forces
gqs.add_force_field(GravityForceField(gravity=np.array([0.0, 0.0, -9.81])))
gqs.add_force_field(SpringForceField("p1", "p2", stiffness=10.0, rest_length=1.0))

# Simulate
for step in range(1000):
    gqs.simulation_step()

print(f"Final positions:")
print(f"  p1: {gqs.entities['p1'].position}")
print(f"  p2: {gqs.entities['p2'].position}")
```

## CANS-3D: Angular Geometry

### Planar Angles
Planar angles are 2D angles at vertices on faces:
```python
# A_p(V, F; V_prev, V_next)
angle = system.planar_angle("V_0", "F_0")
```

### Dihedral Angles
Dihedral angles are 3D angles between faces:
```python
# A_d(E) - angle between two faces sharing edge E
angle = system.dihedral_angle("E_0")
```

### Solid Angles
Solid angles measure 3D conical spread at vertices:
```python
# Ω(V) in steradians
solid_angle = system.solid_angle("V_0")
```

### Vertex Defects
Vertex defects measure 2D intrinsic curvature:
```python
# δ(V) in radians
vertex_defect = system.vertex_defect("V_0")

# Gauss-Bonnet: Σ δ(V_i) = 2πχ
total_defect, expected, is_valid = system.verify_gauss_bonnet()
```

## CANS-nD: Higher Dimensions

### 4D Polytopes
```python
from cans_gqs.cans_nd import Polytope4DAngularSystem, create_tesseract

# Create a 4D tesseract (hypercube)
vertices, cells = create_tesseract()
system = Polytope4DAngularSystem(vertices, cells)

# Compute 4D angles
cell_angle = system.cell_cell_angle(face)  # Angle between 3D cells
hyper_angle = system.hypersolid_angle_4d("V_0")  # 4D solid angle
vertex_defect_4d = system.vertex_defect_4d("V_0")  # 4D intrinsic curvature
```

### N-Dimensional Systems
```python
from cans_gqs import NDAngularSystem, NDVertex, NDEdge
import numpy as np

# Create 5D system
system = NDAngularSystem(5)

# Add primitives
v1 = NDVertex(np.random.randn(5), "V_1")
v2 = NDVertex(np.random.randn(5), "V_2")
system.add_primitive(v1)
system.add_primitive(v2)

edge = NDEdge([v1.coordinates, v2.coordinates], "E_1")
system.add_primitive(edge)

# Compute properties
length = edge.length()
print(f"5D edge length: {length:.4f}")
```

## GQS: Physics Simulation

### Entity Systems

#### Particles
```python
from cans_gqs.gqs import Particle
import numpy as np

particle = Particle(
    label="p1",
    position=np.array([0.0, 0.0, 0.0]),
    mass=1.0,
    velocity=np.array([1.0, 0.0, 0.0])
)

# Kinetic energy
ke = particle.kinetic_energy()
```

#### Rigid Bodies
```python
from cans_gqs.gqs import RigidBody
import numpy as np

body = RigidBody(
    label="body1",
    position=np.array([0.0, 0.0, 0.0]),
    mass=10.0,
    orientation=np.eye(3),  # Rotation matrix
    angular_velocity=np.array([0.0, 0.0, 1.0])
)

# Total kinetic energy (translational + rotational)
total_ke = body.kinetic_energy()
```

### Integrators

#### Available Integrators
```python
from cans_gqs.gqs import GeodesicQuerySystem

# Create system with chosen integrator
integrators = ["euler", "rk4", "verlet", "implicit_euler"]

for integrator_name in integrators:
    gqs = GeodesicQuerySystem(dimension=3, integrator=integrator_name)
    # ... add entities and forces ...
    gqs.simulation_step()
```

**Integrator Characteristics:**
- **Euler**: Fastest, first-order accuracy, explicit
- **RK4**: Fourth-order accuracy, explicit, good for smooth systems
- **Verlet**: Symplectic, energy-conserving, best for conservative systems
- **Implicit Euler**: Unconditionally stable, best for stiff systems

#### Changing Integrator at Runtime
```python
gqs = GeodesicQuerySystem(dimension=3, integrator="euler")
# ... simulate ...

# Switch to more accurate integrator
gqs.set_integrator("rk4")
# ... continue simulation ...
```

### Force Fields

#### Gravity
```python
from cans_gqs.gqs import GravityForceField
import numpy as np

# Earth gravity (pointing down in z)
gravity = GravityForceField(gravity=np.array([0.0, 0.0, -9.81]))
gqs.add_force_field(gravity)
```

#### Springs
```python
from cans_gqs.gqs import SpringForceField

# Hooke's law spring between two entities
spring = SpringForceField(
    entity1_label="p1",
    entity2_label="p2",
    stiffness=10.0,      # Spring constant (N/m)
    rest_length=1.0      # Equilibrium length (m)
)
gqs.add_force_field(spring)
```

#### Damping
```python
from cans_gqs.gqs import DampingForceField

# Viscous damping (energy dissipation)
damping = DampingForceField(damping_coefficient=0.1)
gqs.add_force_field(damping)
```

### Constraints

#### Distance Constraints
```python
from cans_gqs.gqs import DistanceConstraint

# Rigid distance constraint (like SHAKE algorithm)
constraint = DistanceConstraint(
    entity1_label="p1",
    entity2_label="p2",
    distance=1.0  # Fixed distance in meters
)
gqs.add_constraint(constraint)
```

#### Angular Constraints (Framework)
```python
from cans_gqs.gqs import AngularConstraint

# CANS-based angular constraint
constraint = AngularConstraint(
    entity_labels=["p1", "p2", "p3"],
    target_angle=np.pi/2,  # 90 degrees
    tolerance=0.01
)
gqs.add_constraint(constraint)
```

## Applications

### Molecular Dynamics

#### Protein Backbone Analysis
```python
# Compute torsion angles φ, ψ, ω
def compute_torsion_angle(p1, p2, p3, p4):
    """CANS notation: A_t(E) where E connects p2-p3"""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    
    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    if np.dot(np.cross(n1, n2), b2) < 0:
        angle = -angle
    
    return np.degrees(angle)

# φ (phi): C(i-1) - N(i) - CA(i) - C(i)
phi = compute_torsion_angle(c_prev, n_curr, ca_curr, c_curr)

# ψ (psi): N(i) - CA(i) - C(i) - N(i+1)
psi = compute_torsion_angle(n_curr, ca_curr, c_curr, n_next)
```

#### CANS as Rosetta Stone
CANS provides unambiguous notation across MD packages:
- **GROMACS**: dihedral angles
- **AMBER**: PHI/PSI angles
- **CHARMM**: DIHE parameters
- **CANS**: A_t(E_φ) = -1.1065 rad (unambiguous)

### FEA Mesh Quality

#### Element Quality Assessment
```python
from cans_gqs import PolyhedralAngleSystem
import numpy as np

def analyze_element_quality(vertices, faces):
    """Assess FEA element quality using CANS angles"""
    system = PolyhedralAngleSystem(vertices, faces)
    
    angles = []
    for v_idx in range(len(vertices)):
        for f_idx in range(len(faces)):
            if v_idx in faces[f_idx]:
                try:
                    angle = system.planar_angle(f"V_{v_idx}", f"F_{f_idx}")
                    angles.append(np.degrees(angle))
                except:
                    pass
    
    min_angle = min(angles) if angles else 0
    max_angle = max(angles) if angles else 180
    
    # Quality criteria
    is_degenerate = min_angle < 15.0 or max_angle > 165.0
    is_poor = min_angle < 30.0 or max_angle > 150.0
    is_good = min_angle >= 30.0 and max_angle <= 150.0
    
    return {
        'min_angle': min_angle,
        'max_angle': max_angle,
        'is_degenerate': is_degenerate,
        'is_poor': is_poor,
        'is_good': is_good
    }
```

#### CANS Query Language
```
FIND elements WHERE A_p(V, F) < 15° OR A_p(V, F) > 165°
FIND boundary_faces WHERE A_d(E) > 170°
COMPUTE aspect_ratio FOR ALL elements
```

## Performance Optimization

### Numba Acceleration
```python
from cans_gqs.utils.numba_kernels import (
    numba_vector_angle,
    numba_planar_angle,
    NUMBA_AVAILABLE
)

if NUMBA_AVAILABLE:
    # 15-22x speedup for tight loops
    angle = numba_vector_angle(v1, v2)
else:
    # Automatic fallback to NumPy
    angle = np.arccos(np.dot(v1, v2))
```

### GPU Acceleration
```python
from cans_gqs.gqs import GPUCapableGQS

# Automatically uses CuPy if available, otherwise CPU
gqs_gpu = GPUCapableGQS(dimension=3, integrator="verlet")

if gqs_gpu.gpu_available:
    # GPU-accelerated simulation
    gqs_gpu.gpu_simulation_step()
else:
    # CPU fallback
    gqs_gpu.simulation_step()
```

### Performance Tips

1. **Use Verlet for long simulations**: Energy-conserving, stable
2. **Enable Numba**: `pip install numba` for 15-22x speedup
3. **Batch operations**: Add all entities before simulating
4. **Optimize timestep**: Smaller = more accurate but slower
5. **Profile your code**: Use the benchmarking suite

## Examples

All examples are in the `examples/` directory:

### Example 1: Cube (CANS-3D basics)
```bash
python examples/example_01_cube.py
```
Demonstrates: Planar angles, dihedral angles, solid angles, vertex defects, Gauss-Bonnet theorem

### Example 2: Tesseract (4D geometry)
```bash
python examples/example_02_tesseract.py
```
Demonstrates: 4D polytopes, hypersolid angles, 4D Gauss-Bonnet

### Example 3: Molecular (CANS for molecules)
```bash
python examples/example_03_molecular.py
```
Demonstrates: Molecular applications of CANS

### Example 4: Advanced Features
```bash
python examples/example_04_advanced_features.py
```
Demonstrates: 4D systems, quantum applications, query language, strategic positioning

### Example 5: Physics Simulation
```bash
python examples/example_05_physics_simulation.py
```
Demonstrates: Integrators, force fields, constraints, energy conservation

### Example 6: Molecular Dynamics
```bash
python examples/example_06_molecular_dynamics.py
```
Demonstrates: Protein torsion angles, cross-package compatibility, constrained MD

### Example 7: FEA Mesh Quality
```bash
python examples/example_07_fea_mesh_quality.py
```
Demonstrates: Automated mesh quality assessment, degenerate element detection

### Benchmark Performance
```bash
python examples/benchmark_performance.py
```
Comprehensive performance benchmarking across all operations

## Troubleshooting

### Import Errors
```python
# If you get import errors, ensure package is installed:
pip install -e .
```

### Numba Warnings
```python
# Numba JIT compilation causes first-call overhead
# Subsequent calls are 15-22x faster
```

### GPU Not Available
```python
# CuPy is optional - system automatically falls back to CPU
# To enable GPU: pip install cupy-cuda11x
```

### Performance Issues
```python
# Install optional dependencies for better performance:
pip install numba  # 15-22x speedup
pip install cupy-cuda11x  # GPU acceleration
```

## Citation

If you use CANS-GQS in your research, please cite:

```
ContributorX Ltd. (2024)
The Comprehensive Angular Naming System (CANS) and the Geodesic Query System (GQS):
A Formal n-Dimensional Framework for Computational Geometry and Dynamic Simulation
```

## License

MIT License - see LICENSE file for details

## Support

For questions or issues:
1. Check this usage guide
2. Run example scripts
3. Review the "all in one" comprehensive document
4. Open an issue on GitHub
