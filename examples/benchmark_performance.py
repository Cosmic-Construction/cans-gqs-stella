"""
Performance Benchmarking Suite for CANS-GQS

Benchmarks:
1. Angular operations across dimensions
2. Integrator performance comparison
3. Scaling with system size
4. Numba acceleration measurements
"""

import numpy as np
import time
from typing import Dict, List, Tuple
import sys

from cans_gqs import PolyhedralAngleSystem, NDAngularSystem
from cans_gqs.gqs import (
    GeodesicQuerySystem,
    Particle,
    SpringForceField,
    DampingForceField,
)
from cans_gqs.utils.numba_kernels import NUMBA_AVAILABLE

# Try to import optional dependencies
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class PerformanceBenchmark:
    """Performance benchmarking suite"""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_angular_operations(self, dimensions: List[int] = None,
                                    num_iterations: int = 1000) -> Dict:
        """Benchmark angular operations across dimensions"""
        if dimensions is None:
            dimensions = [2, 3, 4, 5, 6]
        
        print("=" * 70)
        print("Benchmark 1: Angular Operations Across Dimensions")
        print("=" * 70)
        print(f"Iterations per dimension: {num_iterations}")
        print()
        
        results = {}
        
        for dim in dimensions:
            # Create system
            system = NDAngularSystem(dim)
            
            # Generate random vectors
            v1 = np.random.randn(dim)
            v2 = np.random.randn(dim)
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)
            
            # Benchmark vector angle computation
            start = time.perf_counter()
            for _ in range(num_iterations):
                angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            end = time.perf_counter()
            
            time_per_op = (end - start) / num_iterations * 1e6  # microseconds
            results[dim] = time_per_op
            
            print(f"  Dimension {dim}: {time_per_op:.2f} µs/op")
        
        print()
        self.results['angular_operations'] = results
        return results
    
    def benchmark_integrators(self, num_particles: int = 10,
                             num_steps: int = 1000) -> Dict:
        """Benchmark different integrators"""
        print("=" * 70)
        print("Benchmark 2: Integrator Performance")
        print("=" * 70)
        print(f"System size: {num_particles} particles")
        print(f"Simulation steps: {num_steps}")
        print()
        
        integrators = ["euler", "rk4", "verlet", "implicit_euler"]
        results = {}
        
        for integrator_name in integrators:
            # Create system
            gqs = GeodesicQuerySystem(dimension=3, integrator=integrator_name)
            gqs.timestep = 0.001
            
            # Add particles
            for i in range(num_particles):
                particle = Particle(
                    label=f"p{i}",
                    position=np.random.randn(3),
                    mass=1.0,
                    velocity=np.random.randn(3) * 0.1
                )
                gqs.add_entity(particle)
            
            # Add forces
            gqs.add_force_field(DampingForceField(damping_coefficient=0.1))
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(num_steps):
                gqs.simulation_step()
            end = time.perf_counter()
            
            total_time = end - start
            time_per_step = total_time / num_steps * 1000  # milliseconds
            
            results[integrator_name] = {
                'total_time': total_time,
                'time_per_step': time_per_step,
                'steps_per_second': num_steps / total_time
            }
            
            print(f"  {integrator_name.upper():15s}: "
                  f"{time_per_step:.3f} ms/step "
                  f"({results[integrator_name]['steps_per_second']:.0f} steps/s)")
        
        print()
        self.results['integrators'] = results
        return results
    
    def benchmark_scaling(self, particle_counts: List[int] = None,
                         num_steps: int = 100) -> Dict:
        """Benchmark scaling with system size"""
        if particle_counts is None:
            particle_counts = [10, 50, 100, 200, 500]
        
        print("=" * 70)
        print("Benchmark 3: Scaling with System Size")
        print("=" * 70)
        print(f"Simulation steps: {num_steps}")
        print()
        
        results = {}
        
        for n_particles in particle_counts:
            # Create system with Verlet (efficient)
            gqs = GeodesicQuerySystem(dimension=3, integrator="verlet")
            gqs.timestep = 0.001
            
            # Add particles
            for i in range(n_particles):
                particle = Particle(
                    label=f"p{i}",
                    position=np.random.randn(3),
                    mass=1.0,
                    velocity=np.random.randn(3) * 0.1
                )
                gqs.add_entity(particle)
            
            # Add spring connections (sparse - only nearest neighbors)
            if n_particles > 1:
                for i in range(min(n_particles - 1, 5)):
                    spring = SpringForceField(
                        f"p{i}", f"p{i+1}",
                        stiffness=10.0, rest_length=1.0
                    )
                    gqs.add_force_field(spring)
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(num_steps):
                gqs.simulation_step()
            end = time.perf_counter()
            
            total_time = end - start
            time_per_step = total_time / num_steps * 1000
            
            results[n_particles] = {
                'total_time': total_time,
                'time_per_step': time_per_step,
            }
            
            print(f"  {n_particles:4d} particles: {time_per_step:.3f} ms/step")
        
        print()
        self.results['scaling'] = results
        return results
    
    def benchmark_numba_acceleration(self, num_iterations: int = 10000) -> Dict:
        """Benchmark Numba acceleration"""
        print("=" * 70)
        print("Benchmark 4: Numba Acceleration")
        print("=" * 70)
        print(f"Numba available: {NUMBA_AVAILABLE}")
        print(f"Iterations: {num_iterations}")
        print()
        
        if not NUMBA_AVAILABLE:
            print("  Numba not available - skipping benchmark")
            print()
            return {}
        
        from cans_gqs.utils.numba_kernels import (
            numba_vector_angle,
            numba_planar_angle,
        )
        
        # Generate test data
        v1 = np.random.randn(3)
        v2 = np.random.randn(3)
        v3 = np.random.randn(3)
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        v3 /= np.linalg.norm(v3)
        
        # Pure NumPy version
        def numpy_vector_angle(v1, v2):
            cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
            return np.arccos(cos_angle)
        
        # Warm up JIT
        _ = numba_vector_angle(v1, v2)
        _ = numba_planar_angle(v1, v2, v3)
        
        results = {}
        
        # Benchmark vector angle
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = numpy_vector_angle(v1, v2)
        numpy_time = time.perf_counter() - start
        
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = numba_vector_angle(v1, v2)
        numba_time = time.perf_counter() - start
        
        speedup = numpy_time / numba_time
        results['vector_angle'] = {
            'numpy_time': numpy_time,
            'numba_time': numba_time,
            'speedup': speedup
        }
        
        print(f"  Vector angle:")
        print(f"    NumPy: {numpy_time*1e6/num_iterations:.2f} µs/op")
        print(f"    Numba: {numba_time*1e6/num_iterations:.2f} µs/op")
        print(f"    Speedup: {speedup:.2f}x")
        print()
        
        # Benchmark planar angle
        start = time.perf_counter()
        for _ in range(num_iterations):
            u = v1 - v2
            v = v3 - v2
            cos_angle = np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1.0, 1.0)
            _ = np.arccos(cos_angle)
        numpy_time = time.perf_counter() - start
        
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = numba_planar_angle(v1, v2, v3)
        numba_time = time.perf_counter() - start
        
        speedup = numpy_time / numba_time
        results['planar_angle'] = {
            'numpy_time': numpy_time,
            'numba_time': numba_time,
            'speedup': speedup
        }
        
        print(f"  Planar angle:")
        print(f"    NumPy: {numpy_time*1e6/num_iterations:.2f} µs/op")
        print(f"    Numba: {numba_time*1e6/num_iterations:.2f} µs/op")
        print(f"    Speedup: {speedup:.2f}x")
        print()
        
        self.results['numba'] = results
        return results
    
    def benchmark_polyhedral_angles(self, mesh_sizes: List[int] = None) -> Dict:
        """Benchmark polyhedral angle computations"""
        if mesh_sizes is None:
            mesh_sizes = [8, 27, 64, 125]  # Cubic meshes
        
        print("=" * 70)
        print("Benchmark 5: Polyhedral Angle Computations")
        print("=" * 70)
        print()
        
        results = {}
        
        for n in mesh_sizes:
            # Create a cubic mesh (n = k^3 vertices)
            k = int(np.cbrt(n))
            vertices = []
            for i in range(k):
                for j in range(k):
                    for l in range(k):
                        vertices.append((i, j, l))
            
            # Create faces (simplified - just outer faces)
            faces = [
                [0, 1, 2, 3],  # Bottom
                [4, 7, 6, 5],  # Top (for k=2)
            ]
            
            # Benchmark
            start = time.perf_counter()
            try:
                system = PolyhedralAngleSystem(vertices, faces)
                
                # Compute some angles
                if len(vertices) > 0:
                    system.vertex_defect("V_0")
                    if len(system.edges) > 0:
                        system.dihedral_angle("E_0")
            except Exception as e:
                print(f"  {n:4d} vertices: ERROR - {str(e)[:50]}")
                continue
            
            end = time.perf_counter()
            total_time = end - start
            
            results[n] = {
                'total_time': total_time,
                'vertices': len(vertices),
                'faces': len(faces),
            }
            
            print(f"  {n:4d} vertices: {total_time*1000:.2f} ms")
        
        print()
        self.results['polyhedral_angles'] = results
        return results
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        print("=" * 70)
        print("Benchmark Summary Report")
        print("=" * 70)
        print()
        
        # System information
        print("System Information:")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  NumPy: {np.__version__}")
        print(f"  Numba: {'Available' if NUMBA_AVAILABLE else 'Not available'}")
        print()
        
        # Angular operations
        if 'angular_operations' in self.results:
            print("Angular Operations:")
            for dim, time_us in self.results['angular_operations'].items():
                print(f"  {dim}D: {time_us:.2f} µs/op")
            print()
        
        # Integrators
        if 'integrators' in self.results:
            print("Integrators (10 particles, 1000 steps):")
            for name, data in self.results['integrators'].items():
                print(f"  {name.upper():15s}: {data['time_per_step']:.3f} ms/step")
            print()
        
        # Scaling
        if 'scaling' in self.results:
            print("Scaling (Verlet integrator, 100 steps):")
            for n, data in self.results['scaling'].items():
                print(f"  {n:4d} particles: {data['time_per_step']:.3f} ms/step")
            print()
        
        # Numba
        if 'numba' in self.results:
            print("Numba Acceleration:")
            for op, data in self.results['numba'].items():
                print(f"  {op}: {data['speedup']:.2f}x speedup")
            print()
        
        print("=" * 70)


def main():
    """Run all benchmarks"""
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "CANS-GQS Performance Benchmark" + " " * 18 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    benchmark = PerformanceBenchmark()
    
    # Run benchmarks
    benchmark.benchmark_angular_operations(dimensions=[2, 3, 4, 5, 6], num_iterations=10000)
    benchmark.benchmark_integrators(num_particles=10, num_steps=1000)
    benchmark.benchmark_scaling(particle_counts=[10, 50, 100, 200], num_steps=100)
    
    if NUMBA_AVAILABLE:
        benchmark.benchmark_numba_acceleration(num_iterations=10000)
    
    benchmark.benchmark_polyhedral_angles(mesh_sizes=[8, 27, 64])
    
    # Generate report
    benchmark.generate_report()
    
    print("Benchmarking complete!")


if __name__ == "__main__":
    main()
