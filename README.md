# The Comprehensive Angular Naming System (CANS) and the Geodesic Query System (GQS)

**A Formal n-Dimensional Framework for Computational Geometry and Dynamic Simulation**

Author: ContributorX Ltd.

## Overview

The Comprehensive Angular Naming System (CANS) and its associated Geodesic Query System (GQS) provide a unified, computable framework for describing and simulating geometric structure in three and higher dimensions.

### Key Features

- **CANS-3D**: Complete 3D angular language with planar angles, dihedral angles, solid angles, and vertex defects
- **CANS-4D & CANS-nD**: Generalization to 4D polytopes and arbitrary n-dimensional spaces
- **GQS**: Dynamic simulation engine integrating CANS angular relationships with physics, constraints, and solvers
- **Explicit De-conflation**: Clear separation of solid angle Ω(V) (3D extrinsic) and vertex defect δ(V) (2D intrinsic)

## Installation

```bash
# Clone the repository
git clone https://github.com/Cosmic-Construction/The-Comprehensive-Angular-Naming-System-CANS-and-the-Geodesic-Query-System-GQS-.git
cd The-Comprehensive-Angular-Naming-System-CANS-and-the-Geodesic-Query-System-GQS-

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Example 1: 3D Cube Analysis

```python
from cans_gqs import PolyhedralAngleSystem
import numpy as np

# Define unit cube
vertices = [
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
]

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

# Compute angular properties
vertex_defect = cube.vertex_defect("V_0")
solid_angle = cube.solid_angle("V_0")
dihedral_angle = cube.dihedral_angle("E_0")

print(f"Vertex defect: {np.degrees(vertex_defect):.2f}°")
print(f"Solid angle: {solid_angle:.4f} sr")
print(f"Dihedral angle: {np.degrees(dihedral_angle):.2f}°")

# Verify Gauss-Bonnet theorem
total_defect, expected, is_valid = cube.verify_gauss_bonnet()
print(f"Gauss-Bonnet verification: {'PASS' if is_valid else 'FAIL'}")
```

### Example 2: n-Dimensional System

```python
from cans_gqs import NDVertex, NDEdge, NDAngularSystem
import numpy as np

# Create 4D system
system = NDAngularSystem(4)

# Add 4D vertices
v1 = NDVertex(np.array([1, 0, 0, 0]), "V_1")
v2 = NDVertex(np.array([0, 1, 0, 0]), "V_2")
system.add_primitive(v1)
system.add_primitive(v2)

# Add 4D edge
edge = NDEdge([v1.coordinates, v2.coordinates], "E_1")
system.add_primitive(edge)

# Compute properties
length = edge.length()
print(f"4D edge length: {length:.4f}")
```

## Running Examples

The repository includes comprehensive examples:

```bash
# 3D cube validation
python examples/example_01_cube.py

# 4D tesseract demonstration
python examples/example_02_tesseract.py

# Advanced features (4D polytopes, quantum systems, optimization)
python examples/example_04_advanced_features.py

# Physics simulation with integrators
python examples/example_05_physics_simulation.py

# Molecular dynamics with CANS
python examples/example_06_molecular_dynamics.py

# FEA mesh quality analysis
python examples/example_07_fea_mesh_quality.py

# Performance benchmarks
python examples/benchmark_performance.py
```

## New Features (December 2024 - Complete Implementation)

This repository now includes **all** features from the comprehensive "all in one" document:

### Part 2: CANS-nD Extensions
- ✅ **4D Polytope System**: Complete `Polytope4DAngularSystem` with cell-cell angles, hypersolid angles, and 4D vertex defects
- ✅ **N-D Visualization**: PCA-based projection and plotting for systems in any dimension
- ✅ **4D Primitives**: Helper functions for 5-cell and tesseract creation

### Part 3: Physics & Performance ⭐ NEW
- ✅ **Complete Physics Engine**: Entity systems, force fields, and constraint solvers
- ✅ **Multiple Integrators**: Euler, RK4, Verlet, Implicit Euler with automatic selection
- ✅ **Force Fields**: Gravity, Spring, Damping with extensible architecture
- ✅ **Constraints**: Distance and angular constraints with SHAKE-like algorithms
- ✅ **Numba Optimization**: JIT-compiled kernels for 15-22x performance boost (optional)
- ✅ **Geometric Algebra**: Cross-validation with Clifford algebra (optional)
- ✅ **Performance Benchmarks**: Comprehensive benchmarking suite with results

### Part 4: Applications & Examples ⭐ NEW
- ✅ **Molecular Dynamics**: CANS as "Rosetta Stone" for protein torsion angles
- ✅ **FEA Mesh Quality**: Automated angular-based quality assessment
- ✅ **Quantum 4D System**: Angular analysis of 2-qubit quantum states and entanglement
- ✅ **Data Analysis**: 4D data cluster angular characterization
- ✅ **Query Language**: Domain-specific queries for MD, FEA, and CAD
- ✅ **Strategic Positioning**: Academic credibility and market positioning frameworks

### Quick Start: Physics Simulation

```python
from cans_gqs.gqs import (
    GeodesicQuerySystem,
    Particle,
    GravityForceField,
    SpringForceField,
    DistanceConstraint,
)
import numpy as np

# Create GQS with Verlet integrator (energy-conserving)
gqs = GeodesicQuerySystem(dimension=3, integrator="verlet")
gqs.timestep = 0.001

# Add particles
p1 = Particle(label="p1", position=np.array([0.0, 0.0, 0.0]), mass=1.0)
p2 = Particle(label="p2", position=np.array([1.5, 0.0, 0.0]), mass=1.0)

gqs.add_entity(p1)
gqs.add_entity(p2)

# Add forces and constraints
gqs.add_force_field(GravityForceField())
gqs.add_force_field(SpringForceField("p1", "p2", stiffness=10.0, rest_length=1.0))
gqs.add_constraint(DistanceConstraint("p1", "p2", distance=1.0))

# Simulate
for step in range(1000):
    gqs.simulation_step()
    
print(f"Final energy: {p1.kinetic_energy() + p2.kinetic_energy():.4f} J")
```

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for detailed documentation of all features.

## Framework Architecture

### CANS-3D Components

1. **Planar Angles** A_p(V_i, F_j; V_a, V_e): 2D angles at vertices on faces
2. **Dihedral Angles** A_d(E_k): 3D angles between faces along edges
3. **Solid Angles** Ω(V_i): 3D conical spread at vertices (steradians)
4. **Vertex Defects** δ(V_i): 2D intrinsic curvature at vertices (radians)

### CANS-nD Extensions

- **k-Dihedral Angles**: Angles between k-faces along (k-1)-faces
- **k-Solid Angles**: k-dimensional solid angles at vertices
- **Recursive Computation**: Consistent angular measures across dimensions

### GQS (Geodesic Query System)

```
GQS = CANS-nD (angular system) + Physics + Constraints + Solvers
```

The GQS provides a complete simulation framework:

**Entity Systems:**
- `Particle`: Point mass with position, velocity, force
- `RigidBody`: Extended body with orientation, angular velocity, torque

**Integrators:**
- `EulerIntegrator`: Forward Euler (explicit, first-order)
- `RK4Integrator`: Runge-Kutta 4th order (explicit, fourth-order accuracy)
- `VerletIntegrator`: Velocity Verlet (symplectic, energy-conserving)
- `ImplicitEulerIntegrator`: Implicit Euler (unconditionally stable for stiff systems)

**Force Fields:**
- `GravityForceField`: Uniform gravitational field
- `SpringForceField`: Hooke's law springs between entities
- `DampingForceField`: Viscous damping (energy dissipation)

**Constraints:**
- `DistanceConstraint`: Rigid distance constraints (SHAKE-like)
- `AngularConstraint`: CANS-based angular constraints

**Performance:**
- Numba acceleration: 15-22x speedup on tight loops
- 10,000+ simulation steps/second (10 particles, Euler)
- Optional GPU support via CuPy

## Applications

### Molecular Dynamics
- Unified torsion angle notation (φ, ψ, ω)
- Protein backbone geometry analysis
- Cross-platform validation

### FEA Mesh Quality
- Automated mesh quality checks
- Angular-based quality metrics
- Dynamic remeshing triggers

### CAD/CAE Integration
- Geometric design intent preservation
- Cross-platform interoperability
- Manufacturing constraint verification

### Quantum Computing
- 2-qubit state geometry (4D)
- Entanglement angle measures
- High-dimensional manifold analysis

## Mathematical Foundations

### Gauss-Bonnet Theorem (3D)
```
Σ δ(V_i) = 2πχ
```
where χ is the Euler characteristic (V - E + F).

### 4D Gauss-Bonnet
```
Σ δ_4(V_i) = 2π²χ
```

### Total Solid Angle of (k-1)-Sphere
```
Ω_{k-1} = 2π^(k/2) / Γ(k/2)
```

## Documentation

For detailed documentation, see the comprehensive paper in `all in one` which covers:
- Part 1: CANS-3D formalization and implementation
- Part 2: CANS-4D and CANS-nD extensions
- Part 3: GQS dynamic simulation engine
- Part 4: Applications and query language

## Citation

If you use CANS/GQS in your research, please cite:

```
ContributorX Ltd. (2024)
The Comprehensive Angular Naming System (CANS) and the Geodesic Query System (GQS):
A Formal n-Dimensional Framework for Computational Geometry and Dynamic Simulation
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Contact

For questions and support, please open an issue on GitHub
