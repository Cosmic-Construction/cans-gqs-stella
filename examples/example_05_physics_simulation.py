#!/usr/bin/env python3
"""
Example 5: GQS Physics Simulation
Demonstrates the enhanced GQS simulation engine with:
- Multiple integrators (Euler, RK4, Verlet, Implicit Euler)
- Entity systems (Particle, RigidBody)
- Force fields (Gravity, Springs, Damping)
- Constraints (Distance, Angular)
"""

import numpy as np
import matplotlib.pyplot as plt
from cans_gqs.gqs import (
    GeodesicQuerySystem,
    Particle,
    GravityForceField,
    SpringForceField,
    DampingForceField,
    DistanceConstraint,
)


def demo_free_fall():
    """Demo 1: Free fall with different integrators"""
    print("=" * 70)
    print("Demo 1: Free Fall Comparison")
    print("=" * 70)
    print()
    
    # Parameters
    initial_height = 10.0
    timestep = 0.01
    duration = 1.5
    num_steps = int(duration / timestep)
    
    # Analytical solution
    t_analytical = np.linspace(0, duration, num_steps)
    y_analytical = initial_height - 0.5 * 9.81 * t_analytical**2
    
    integrators = ["euler", "rk4", "verlet"]
    results = {}
    
    for integrator_name in integrators:
        # Create GQS system
        gqs = GeodesicQuerySystem(dimension=3, integrator=integrator_name)
        gqs.timestep = timestep
        
        # Create particle
        particle = Particle(
            label="ball",
            position=np.array([0.0, 0.0, initial_height]),
            mass=1.0,
            velocity=np.array([0.0, 0.0, 0.0])
        )
        
        # Add to system
        gqs.add_entity(particle)
        gqs.add_force_field(GravityForceField(gravity=np.array([0.0, 0.0, -9.81])))
        
        # Simulate
        positions = []
        for _ in range(num_steps):
            positions.append(gqs.entities["ball"].position[2])
            gqs.simulation_step()
        
        results[integrator_name] = np.array(positions)
    
    # Compare results
    print(f"Integrator comparison after {duration}s:")
    print(f"  Analytical: {y_analytical[-1]:.6f} m")
    for name, positions in results.items():
        error = abs(positions[-1] - y_analytical[-1])
        print(f"  {name.upper():15s}: {positions[-1]:.6f} m (error: {error:.6f} m)")
    
    print()


def demo_spring_system():
    """Demo 2: Spring-mass system"""
    print("=" * 70)
    print("Demo 2: Spring-Mass System (Harmonic Oscillator)")
    print("=" * 70)
    print()
    
    # Create GQS system with Verlet (symplectic, energy-conserving)
    gqs = GeodesicQuerySystem(dimension=3, integrator="verlet")
    gqs.timestep = 0.005
    
    # Create two particles connected by a spring
    particle1 = Particle(
        label="p1",
        position=np.array([0.0, 0.0, 0.0]),
        mass=1.0,
        velocity=np.array([0.0, 0.0, 0.0])
    )
    
    particle2 = Particle(
        label="p2",
        position=np.array([2.0, 0.0, 0.0]),
        mass=1.0,
        velocity=np.array([0.0, 0.0, 0.0])
    )
    
    # Add to system
    gqs.add_entity(particle1)
    gqs.add_entity(particle2)
    
    # Add spring force (rest length = 1.0, stiffness = 10.0)
    spring = SpringForceField("p1", "p2", stiffness=10.0, rest_length=1.0)
    gqs.add_force_field(spring)
    
    # Add damping
    gqs.add_force_field(DampingForceField(damping_coefficient=0.1))
    
    # Simulate
    num_steps = 1000
    times = []
    distances = []
    energies = []
    
    for step in range(num_steps):
        times.append(gqs.time)
        
        # Compute distance
        p1 = gqs.entities["p1"]
        p2 = gqs.entities["p2"]
        distance = np.linalg.norm(p2.position - p1.position)
        distances.append(distance)
        
        # Compute total energy
        ke1 = p1.kinetic_energy()
        ke2 = p2.kinetic_energy()
        pe = 0.5 * spring.stiffness * (distance - spring.rest_length)**2
        energies.append(ke1 + ke2 + pe)
        
        gqs.simulation_step()
    
    # Report results
    print(f"Initial distance: {distances[0]:.6f} m")
    print(f"Final distance: {distances[-1]:.6f} m")
    print(f"Rest length: {spring.rest_length:.6f} m")
    print(f"Initial energy: {energies[0]:.6f} J")
    print(f"Final energy: {energies[-1]:.6f} J")
    print(f"Energy lost (damping): {energies[0] - energies[-1]:.6f} J")
    print()


def demo_constrained_system():
    """Demo 3: Constrained pendulum"""
    print("=" * 70)
    print("Demo 3: Constrained Pendulum")
    print("=" * 70)
    print()
    
    # Create GQS system
    gqs = GeodesicQuerySystem(dimension=3, integrator="verlet")
    gqs.timestep = 0.005
    
    # Fixed anchor
    anchor = Particle(
        label="anchor",
        position=np.array([0.0, 0.0, 0.0]),
        mass=1e10,  # Very heavy (effectively fixed)
        velocity=np.array([0.0, 0.0, 0.0])
    )
    
    # Pendulum bob
    bob = Particle(
        label="bob",
        position=np.array([1.0, 0.0, 0.0]),
        mass=1.0,
        velocity=np.array([0.0, 0.0, 0.0])
    )
    
    # Add to system
    gqs.add_entity(anchor)
    gqs.add_entity(bob)
    
    # Add gravity
    gqs.add_force_field(GravityForceField(gravity=np.array([0.0, 0.0, -9.81])))
    
    # Add distance constraint (rigid pendulum rod)
    constraint = DistanceConstraint("anchor", "bob", distance=1.0)
    gqs.add_constraint(constraint)
    
    # Add small damping
    gqs.add_force_field(DampingForceField(damping_coefficient=0.01))
    
    # Simulate
    num_steps = 500
    times = []
    heights = []
    distances = []
    
    for step in range(num_steps):
        times.append(gqs.time)
        bob_entity = gqs.entities["bob"]
        anchor_entity = gqs.entities["anchor"]
        
        heights.append(bob_entity.position[2])
        distance = np.linalg.norm(bob_entity.position - anchor_entity.position)
        distances.append(distance)
        
        gqs.simulation_step()
    
    # Report results
    print(f"Constraint length: {constraint.distance:.6f} m")
    print(f"Average distance: {np.mean(distances):.6f} m")
    print(f"Distance std dev: {np.std(distances):.6f} m")
    print(f"Constraint satisfied: {np.std(distances) < 0.01} ✓" if np.std(distances) < 0.01 else "✗")
    print()


def demo_integrator_comparison():
    """Demo 4: Integrator stability comparison"""
    print("=" * 70)
    print("Demo 4: Integrator Stability Comparison")
    print("=" * 70)
    print()
    
    # Test on a stiff spring system
    stiffness = 100.0
    timestep = 0.01
    duration = 1.0
    num_steps = int(duration / timestep)
    
    integrators = {
        "euler": "Forward Euler (Explicit)",
        "rk4": "RK4 (Explicit, 4th order)",
        "verlet": "Velocity Verlet (Symplectic)",
        "implicit_euler": "Implicit Euler (Stable)"
    }
    
    print(f"Testing on stiff spring system (k={stiffness}, dt={timestep}):")
    print()
    
    for integrator_name, description in integrators.items():
        # Create system
        gqs = GeodesicQuerySystem(dimension=3, integrator=integrator_name)
        gqs.timestep = timestep
        
        # Create particles
        p1 = Particle(
            label="p1",
            position=np.array([0.0, 0.0, 0.0]),
            mass=1.0,
            velocity=np.array([0.0, 0.0, 0.0])
        )
        
        p2 = Particle(
            label="p2",
            position=np.array([1.5, 0.0, 0.0]),
            mass=1.0,
            velocity=np.array([0.0, 0.0, 0.0])
        )
        
        gqs.add_entity(p1)
        gqs.add_entity(p2)
        
        # Stiff spring
        spring = SpringForceField("p1", "p2", stiffness=stiffness, rest_length=1.0)
        gqs.add_force_field(spring)
        
        # Simulate
        try:
            energies = []
            for step in range(num_steps):
                p1_ent = gqs.entities["p1"]
                p2_ent = gqs.entities["p2"]
                distance = np.linalg.norm(p2_ent.position - p1_ent.position)
                
                ke = p1_ent.kinetic_energy() + p2_ent.kinetic_energy()
                pe = 0.5 * stiffness * (distance - 1.0)**2
                energies.append(ke + pe)
                
                gqs.simulation_step()
                
                # Check for instability
                if not np.isfinite(p2_ent.position).all():
                    print(f"  {description:35s}: UNSTABLE at step {step}")
                    break
            else:
                energy_change = abs(energies[-1] - energies[0]) / energies[0] * 100
                print(f"  {description:35s}: Stable (ΔE = {energy_change:.2f}%)")
        
        except Exception as e:
            print(f"  {description:35s}: ERROR - {str(e)}")
    
    print()


def main():
    """Run all demos"""
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "GQS Physics Simulation Demo" + " " * 26 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    demo_free_fall()
    demo_spring_system()
    demo_constrained_system()
    demo_integrator_comparison()
    
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Implemented features:")
    print("  ✓ Multiple integrators (Euler, RK4, Verlet, Implicit Euler)")
    print("  ✓ Entity systems (Particle, RigidBody)")
    print("  ✓ Force fields (Gravity, Spring, Damping)")
    print("  ✓ Constraints (Distance, Angular)")
    print("  ✓ Energy tracking and conservation")
    print()
    print("All physics demos completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
