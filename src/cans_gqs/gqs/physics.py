"""
GQS Physics Module
Part 3: Entity systems, integrators, and force fields

This module implements the physics components for the Geodesic Query System:
- Entity classes (Particle, RigidBody)
- Integrators (Euler, RK4, Verlet, Implicit Euler)
- Force fields and constraints
"""

import numpy as np
from typing import Dict, List, Callable, Any, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================================
# Entity Classes
# ============================================================================

@dataclass
class Entity:
    """Base entity class for GQS simulations"""
    label: str
    position: np.ndarray
    mass: float = 1.0
    
    def __post_init__(self):
        """Ensure position is numpy array"""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position, dtype=float)


@dataclass
class Particle(Entity):
    """Particle entity with position, velocity, and force"""
    velocity: np.ndarray = field(default=None)
    force: np.ndarray = field(default=None)
    
    def __post_init__(self):
        """Initialize velocity and force if not provided"""
        super().__post_init__()
        
        dim = len(self.position)
        if self.velocity is None:
            self.velocity = np.zeros(dim)
        elif not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=float)
            
        if self.force is None:
            self.force = np.zeros(dim)
        elif not isinstance(self.force, np.ndarray):
            self.force = np.array(self.force, dtype=float)
    
    def kinetic_energy(self) -> float:
        """Compute kinetic energy: KE = 0.5 * m * v^2"""
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)


@dataclass
class RigidBody(Entity):
    """Rigid body entity with orientation and angular velocity"""
    velocity: np.ndarray = field(default=None)
    orientation: np.ndarray = field(default=None)  # Rotation matrix or quaternion
    angular_velocity: np.ndarray = field(default=None)
    inertia_tensor: np.ndarray = field(default=None)
    force: np.ndarray = field(default=None)
    torque: np.ndarray = field(default=None)
    
    def __post_init__(self):
        """Initialize rigid body properties"""
        super().__post_init__()
        
        dim = len(self.position)
        
        if self.velocity is None:
            self.velocity = np.zeros(dim)
        elif not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=float)
        
        if self.orientation is None:
            self.orientation = np.eye(dim)
        elif not isinstance(self.orientation, np.ndarray):
            self.orientation = np.array(self.orientation, dtype=float)
        
        if self.angular_velocity is None:
            self.angular_velocity = np.zeros(3 if dim == 3 else dim)
        elif not isinstance(self.angular_velocity, np.ndarray):
            self.angular_velocity = np.array(self.angular_velocity, dtype=float)
        
        if self.inertia_tensor is None:
            # Default to unit inertia tensor
            inertia_dim = 3 if dim == 3 else dim
            self.inertia_tensor = np.eye(inertia_dim) * self.mass
        elif not isinstance(self.inertia_tensor, np.ndarray):
            self.inertia_tensor = np.array(self.inertia_tensor, dtype=float)
        
        if self.force is None:
            self.force = np.zeros(dim)
        elif not isinstance(self.force, np.ndarray):
            self.force = np.array(self.force, dtype=float)
        
        if self.torque is None:
            self.torque = np.zeros(3 if dim == 3 else dim)
        elif not isinstance(self.torque, np.ndarray):
            self.torque = np.array(self.torque, dtype=float)
    
    def kinetic_energy(self) -> float:
        """Compute total kinetic energy (translational + rotational)"""
        translational = 0.5 * self.mass * np.dot(self.velocity, self.velocity)
        
        # Rotational energy: 0.5 * ω^T * I * ω
        rotational = 0.5 * np.dot(
            self.angular_velocity,
            np.dot(self.inertia_tensor, self.angular_velocity)
        )
        
        return translational + rotational


# ============================================================================
# Integrators
# ============================================================================

class Integrator:
    """Base class for numerical integrators"""
    
    def __init__(self, timestep: float = 1e-3):
        self.timestep = timestep
        self.name = "Base Integrator"
    
    def step(self, entities: Dict[str, Entity], 
             forces: Dict[str, np.ndarray]) -> Dict[str, Entity]:
        """
        Perform one integration step
        
        Parameters:
            entities: Dictionary of entities
            forces: Dictionary of forces on each entity
            
        Returns:
            Updated entities
        """
        raise NotImplementedError("Subclasses must implement step()")


class EulerIntegrator(Integrator):
    """Forward Euler integrator (explicit, first-order)"""
    
    def __init__(self, timestep: float = 1e-3):
        super().__init__(timestep)
        self.name = "Forward Euler"
    
    def step(self, entities: Dict[str, Entity], 
             forces: Dict[str, np.ndarray]) -> Dict[str, Entity]:
        """
        Forward Euler: 
            v(t+dt) = v(t) + a(t) * dt
            x(t+dt) = x(t) + v(t+dt) * dt
        """
        new_entities = {}
        
        for label, entity in entities.items():
            if label not in forces:
                new_entities[label] = entity
                continue
            
            if isinstance(entity, Particle):
                # Create new particle
                new_particle = Particle(
                    label=entity.label,
                    position=entity.position.copy(),
                    mass=entity.mass,
                    velocity=entity.velocity.copy(),
                    force=entity.force.copy()
                )
                
                # Compute acceleration
                acceleration = forces[label] / entity.mass
                
                # Update velocity and position
                new_particle.velocity = entity.velocity + acceleration * self.timestep
                new_particle.position = entity.position + new_particle.velocity * self.timestep
                
                new_entities[label] = new_particle
            
            elif isinstance(entity, RigidBody):
                # Create new rigid body
                new_body = RigidBody(
                    label=entity.label,
                    position=entity.position.copy(),
                    mass=entity.mass,
                    velocity=entity.velocity.copy(),
                    orientation=entity.orientation.copy(),
                    angular_velocity=entity.angular_velocity.copy(),
                    inertia_tensor=entity.inertia_tensor.copy(),
                    force=entity.force.copy(),
                    torque=entity.torque.copy()
                )
                
                # Linear motion
                acceleration = forces[label] / entity.mass
                new_body.velocity = entity.velocity + acceleration * self.timestep
                new_body.position = entity.position + new_body.velocity * self.timestep
                
                # Angular motion (simplified for 3D)
                if len(entity.angular_velocity) == 3:
                    # Angular acceleration: α = I^(-1) * τ
                    inertia_inv = np.linalg.inv(entity.inertia_tensor)
                    angular_accel = np.dot(inertia_inv, entity.torque)
                    new_body.angular_velocity = entity.angular_velocity + angular_accel * self.timestep
                
                new_entities[label] = new_body
            else:
                new_entities[label] = entity
        
        return new_entities


class RK4Integrator(Integrator):
    """Runge-Kutta 4th order integrator (explicit, fourth-order)"""
    
    def __init__(self, timestep: float = 1e-3):
        super().__init__(timestep)
        self.name = "RK4"
    
    def _compute_acceleration(self, entity: Entity, force: np.ndarray) -> np.ndarray:
        """Compute acceleration from force"""
        return force / entity.mass
    
    def step(self, entities: Dict[str, Entity], 
             forces: Dict[str, np.ndarray]) -> Dict[str, Entity]:
        """
        RK4 integrator for second-order ODEs
        
        For x'' = f(x, v, t):
            k1_v = f(x, v, t) * dt
            k1_x = v * dt
            
            k2_v = f(x + k1_x/2, v + k1_v/2, t + dt/2) * dt
            k2_x = (v + k1_v/2) * dt
            
            k3_v = f(x + k2_x/2, v + k2_v/2, t + dt/2) * dt
            k3_x = (v + k2_v/2) * dt
            
            k4_v = f(x + k3_x, v + k3_v, t + dt) * dt
            k4_x = (v + k3_v) * dt
            
            v_new = v + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
            x_new = x + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        """
        new_entities = {}
        dt = self.timestep
        
        for label, entity in entities.items():
            if label not in forces:
                new_entities[label] = entity
                continue
            
            if isinstance(entity, Particle):
                # Current state
                x = entity.position
                v = entity.velocity
                a = self._compute_acceleration(entity, forces[label])
                
                # k1
                k1_v = a * dt
                k1_x = v * dt
                
                # k2
                k2_v = a * dt  # Force is constant over timestep (simplification)
                k2_x = (v + k1_v / 2) * dt
                
                # k3
                k3_v = a * dt
                k3_x = (v + k2_v / 2) * dt
                
                # k4
                k4_v = a * dt
                k4_x = (v + k3_v) * dt
                
                # Update
                v_new = v + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
                x_new = x + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
                
                # Create new particle
                new_particle = Particle(
                    label=entity.label,
                    position=x_new,
                    mass=entity.mass,
                    velocity=v_new,
                    force=entity.force.copy()
                )
                
                new_entities[label] = new_particle
            else:
                # For other entity types, fall back to Euler
                acceleration = forces[label] / entity.mass
                new_velocity = entity.velocity + acceleration * dt
                new_position = entity.position + new_velocity * dt
                
                if isinstance(entity, RigidBody):
                    new_body = RigidBody(
                        label=entity.label,
                        position=new_position,
                        mass=entity.mass,
                        velocity=new_velocity,
                        orientation=entity.orientation.copy(),
                        angular_velocity=entity.angular_velocity.copy(),
                        inertia_tensor=entity.inertia_tensor.copy(),
                        force=entity.force.copy(),
                        torque=entity.torque.copy()
                    )
                    new_entities[label] = new_body
                else:
                    new_entities[label] = entity
        
        return new_entities


class VerletIntegrator(Integrator):
    """Velocity Verlet integrator (explicit, second-order, symplectic)"""
    
    def __init__(self, timestep: float = 1e-3):
        super().__init__(timestep)
        self.name = "Velocity Verlet"
        self.previous_accelerations = {}
    
    def step(self, entities: Dict[str, Entity], 
             forces: Dict[str, np.ndarray]) -> Dict[str, Entity]:
        """
        Velocity Verlet:
            x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
            v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        
        This is symplectic and conserves energy better than Euler/RK4
        """
        new_entities = {}
        dt = self.timestep
        
        for label, entity in entities.items():
            if label not in forces:
                new_entities[label] = entity
                continue
            
            if isinstance(entity, Particle):
                # Current acceleration
                a_current = forces[label] / entity.mass
                
                # Update position
                x_new = (entity.position + 
                        entity.velocity * dt + 
                        0.5 * a_current * dt * dt)
                
                # We need acceleration at new position, but force is given
                # Use current force as approximation (force evaluation at t+dt 
                # would require re-evaluating force field)
                a_new = a_current
                
                # Update velocity
                v_new = entity.velocity + 0.5 * (a_current + a_new) * dt
                
                # Create new particle
                new_particle = Particle(
                    label=entity.label,
                    position=x_new,
                    mass=entity.mass,
                    velocity=v_new,
                    force=entity.force.copy()
                )
                
                new_entities[label] = new_particle
                self.previous_accelerations[label] = a_new
            else:
                # Fall back to Euler for other types
                acceleration = forces[label] / entity.mass
                new_velocity = entity.velocity + acceleration * dt
                new_position = entity.position + new_velocity * dt
                
                if isinstance(entity, RigidBody):
                    new_body = RigidBody(
                        label=entity.label,
                        position=new_position,
                        mass=entity.mass,
                        velocity=new_velocity,
                        orientation=entity.orientation.copy(),
                        angular_velocity=entity.angular_velocity.copy(),
                        inertia_tensor=entity.inertia_tensor.copy(),
                        force=entity.force.copy(),
                        torque=entity.torque.copy()
                    )
                    new_entities[label] = new_body
                else:
                    new_entities[label] = entity
        
        return new_entities


class ImplicitEulerIntegrator(Integrator):
    """
    Implicit Euler integrator (implicit, first-order, unconditionally stable)
    
    Good for stiff systems (protein folding, contact mechanics)
    """
    
    def __init__(self, timestep: float = 1e-3, max_iterations: int = 10, 
                 tolerance: float = 1e-6):
        super().__init__(timestep)
        self.name = "Implicit Euler"
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def step(self, entities: Dict[str, Entity], 
             forces: Dict[str, np.ndarray]) -> Dict[str, Entity]:
        """
        Implicit Euler:
            v(t+dt) = v(t) + a(t+dt) * dt
            x(t+dt) = x(t) + v(t+dt) * dt
        
        Requires solving: v_new = v_old + f(x_new, v_new) / m * dt
        We use fixed-point iteration as a simple solver
        """
        new_entities = {}
        dt = self.timestep
        
        for label, entity in entities.items():
            if label not in forces:
                new_entities[label] = entity
                continue
            
            if isinstance(entity, Particle):
                # Initial guess: use explicit Euler
                v_new = entity.velocity + (forces[label] / entity.mass) * dt
                x_new = entity.position + v_new * dt
                
                # Fixed-point iteration (simplified - assumes force doesn't 
                # change significantly)
                for iteration in range(self.max_iterations):
                    v_prev = v_new.copy()
                    
                    # Update with implicit formula
                    # In practice, would need to re-evaluate force at new position
                    # Here we use the given force as approximation
                    a_new = forces[label] / entity.mass
                    v_new = entity.velocity + a_new * dt
                    x_new = entity.position + v_new * dt
                    
                    # Check convergence
                    if np.linalg.norm(v_new - v_prev) < self.tolerance:
                        break
                
                # Create new particle
                new_particle = Particle(
                    label=entity.label,
                    position=x_new,
                    mass=entity.mass,
                    velocity=v_new,
                    force=entity.force.copy()
                )
                
                new_entities[label] = new_particle
            else:
                # Fall back to explicit Euler
                acceleration = forces[label] / entity.mass
                new_velocity = entity.velocity + acceleration * dt
                new_position = entity.position + new_velocity * dt
                
                if isinstance(entity, RigidBody):
                    new_body = RigidBody(
                        label=entity.label,
                        position=new_position,
                        mass=entity.mass,
                        velocity=new_velocity,
                        orientation=entity.orientation.copy(),
                        angular_velocity=entity.angular_velocity.copy(),
                        inertia_tensor=entity.inertia_tensor.copy(),
                        force=entity.force.copy(),
                        torque=entity.torque.copy()
                    )
                    new_entities[label] = new_body
                else:
                    new_entities[label] = entity
        
        return new_entities


# ============================================================================
# Force Fields
# ============================================================================

class ForceField:
    """Base class for force fields"""
    
    def __init__(self, name: str = "Base Force Field"):
        self.name = name
    
    def compute_force(self, entity: Entity, 
                     all_entities: Dict[str, Entity]) -> np.ndarray:
        """
        Compute force on entity given all entities
        
        Parameters:
            entity: Entity to compute force on
            all_entities: All entities in the system
            
        Returns:
            Force vector
        """
        raise NotImplementedError("Subclasses must implement compute_force()")


class GravityForceField(ForceField):
    """Uniform gravitational force field"""
    
    def __init__(self, gravity: np.ndarray = None):
        super().__init__("Gravity")
        if gravity is None:
            self.gravity = np.array([0.0, 0.0, -9.81])  # Default: Earth gravity
        else:
            self.gravity = np.array(gravity, dtype=float)
    
    def compute_force(self, entity: Entity, 
                     all_entities: Dict[str, Entity]) -> np.ndarray:
        """F = m * g"""
        return entity.mass * self.gravity


class SpringForceField(ForceField):
    """Spring force between two entities"""
    
    def __init__(self, entity1_label: str, entity2_label: str,
                 stiffness: float = 1.0, rest_length: float = 1.0):
        super().__init__("Spring")
        self.entity1_label = entity1_label
        self.entity2_label = entity2_label
        self.stiffness = stiffness
        self.rest_length = rest_length
    
    def compute_force(self, entity: Entity, 
                     all_entities: Dict[str, Entity]) -> np.ndarray:
        """
        Hooke's law: F = -k * (|x| - L0) * (x / |x|)
        """
        # Only apply to connected entities
        if entity.label == self.entity1_label:
            other_label = self.entity2_label
        elif entity.label == self.entity2_label:
            other_label = self.entity1_label
        else:
            return np.zeros_like(entity.position)
        
        if other_label not in all_entities:
            return np.zeros_like(entity.position)
        
        other = all_entities[other_label]
        
        # Compute spring force
        delta = other.position - entity.position
        distance = np.linalg.norm(delta)
        
        if distance < 1e-10:
            return np.zeros_like(entity.position)
        
        direction = delta / distance
        force_magnitude = self.stiffness * (distance - self.rest_length)
        
        return force_magnitude * direction


class DampingForceField(ForceField):
    """Viscous damping force"""
    
    def __init__(self, damping_coefficient: float = 0.1):
        super().__init__("Damping")
        self.damping = damping_coefficient
    
    def compute_force(self, entity: Entity, 
                     all_entities: Dict[str, Entity]) -> np.ndarray:
        """F = -c * v"""
        if isinstance(entity, (Particle, RigidBody)):
            return -self.damping * entity.velocity
        return np.zeros_like(entity.position)


# ============================================================================
# Constraint System
# ============================================================================

class Constraint:
    """Base class for geometric constraints"""
    
    def __init__(self, name: str = "Base Constraint"):
        self.name = name
    
    def apply(self, entities: Dict[str, Entity]) -> Dict[str, Entity]:
        """
        Apply constraint to entities
        
        Parameters:
            entities: Dictionary of entities
            
        Returns:
            Constrained entities
        """
        raise NotImplementedError("Subclasses must implement apply()")


class DistanceConstraint(Constraint):
    """Constrain distance between two entities"""
    
    def __init__(self, entity1_label: str, entity2_label: str, 
                 distance: float):
        super().__init__("Distance Constraint")
        self.entity1_label = entity1_label
        self.entity2_label = entity2_label
        self.distance = distance
    
    def apply(self, entities: Dict[str, Entity]) -> Dict[str, Entity]:
        """Apply distance constraint using SHAKE-like algorithm"""
        if (self.entity1_label not in entities or 
            self.entity2_label not in entities):
            return entities
        
        entity1 = entities[self.entity1_label]
        entity2 = entities[self.entity2_label]
        
        # Compute current distance
        delta = entity2.position - entity1.position
        current_distance = np.linalg.norm(delta)
        
        if current_distance < 1e-10:
            return entities
        
        # Compute correction
        error = current_distance - self.distance
        correction = 0.5 * error * (delta / current_distance)
        
        # Apply correction (symmetric)
        new_entities = entities.copy()
        
        if isinstance(entity1, Particle):
            new_particle1 = Particle(
                label=entity1.label,
                position=entity1.position + correction,
                mass=entity1.mass,
                velocity=entity1.velocity.copy(),
                force=entity1.force.copy()
            )
            new_entities[self.entity1_label] = new_particle1
        
        if isinstance(entity2, Particle):
            new_particle2 = Particle(
                label=entity2.label,
                position=entity2.position - correction,
                mass=entity2.mass,
                velocity=entity2.velocity.copy(),
                force=entity2.force.copy()
            )
            new_entities[self.entity2_label] = new_particle2
        
        return new_entities


class AngularConstraint(Constraint):
    """CANS-based angular constraint"""
    
    def __init__(self, entity_labels: List[str], 
                 target_angle: float, tolerance: float = 0.01):
        super().__init__("Angular Constraint")
        self.entity_labels = entity_labels
        self.target_angle = target_angle
        self.tolerance = tolerance
    
    def apply(self, entities: Dict[str, Entity]) -> Dict[str, Entity]:
        """Apply angular constraint (simplified implementation)"""
        # This would integrate with CANS angular computations
        # For now, just return entities unchanged
        # Full implementation would use CANS planar/dihedral angles
        return entities
