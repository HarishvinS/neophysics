"""
Physics Simulation Engine
A robust physics reasoning system that uses actual physics simulation for predictions
rather than hardcoded patterns. This addresses the limitations of the template-based approach.
"""

import pybullet as p
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import copy

from dynamic_scene_representation import DynamicPhysicsScene, DynamicPhysicsObject
from scene_representation import ObjectType, MaterialType, Vector3
from multi_step_physics_reasoning import PhysicsStep, PhysicsChain, InteractionType


class SimulationEvent(Enum):
    """Types of events that can occur in physics simulation."""
    OBJECT_COLLISION = "object_collision"
    OBJECT_FALLS = "object_falls"
    OBJECT_STOPS = "object_stops"
    CONSTRAINT_BREAKS = "constraint_breaks"
    EQUILIBRIUM_REACHED = "equilibrium_reached"


@dataclass
class SimulationState:
    """Represents the state of objects at a point in time."""
    timestamp: float
    object_states: Dict[str, Dict[str, Any]]  # object_id -> {position, velocity, etc.}
    events: List[SimulationEvent]
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'object_states': self.object_states,
            'events': [event.value for event in self.events]
        }


class PhysicsSimulationEngine:
    """
    A physics reasoning engine that uses actual physics simulation to predict outcomes.
    This replaces the template-based approach with dynamic, physics-based reasoning.
    """
    
    def __init__(self):
        """Initialize physics simulation engine."""
        self.simulation_client = None
        self.object_mapping = {}  # scene_object_id -> pybullet_body_id
        self.simulation_timestep = 1/240  # 240 Hz simulation
        self.prediction_horizon = 5.0  # Predict 5 seconds into future
        
    def _initialize_simulation(self) -> int:
        """Initialize a headless PyBullet simulation."""
        client_id = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=client_id)
        p.setTimeStep(self.simulation_timestep, physicsClientId=client_id)
        return client_id
    
    def _cleanup_simulation(self, client_id: int):
        """Clean up simulation resources."""
        p.disconnect(physicsClientId=client_id)
    
    def _create_physics_object(self, obj: DynamicPhysicsObject, client_id: int) -> int:
        """Create a PyBullet object from scene object."""
        # Create collision shape based on object type
        if obj.object_type == ObjectType.SPHERE:
            collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=obj.scale.x,
                physicsClientId=client_id
            )
        elif obj.object_type == ObjectType.BOX:
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[obj.scale.x/2, obj.scale.y/2, obj.scale.z/2],
                physicsClientId=client_id
            )
        elif obj.object_type == ObjectType.CYLINDER:
            collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=obj.scale.x,
                height=obj.scale.z,
                physicsClientId=client_id
            )
        else:
            # Default to box
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[obj.scale.x/2, obj.scale.y/2, obj.scale.z/2],
                physicsClientId=client_id
            )
        
        # Create visual shape (optional, for debugging)
        if obj.object_type == ObjectType.SPHERE:
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE,
                radius=obj.scale.x,
                rgbaColor=[0.5, 0.5, 0.5, 1.0],
                physicsClientId=client_id
            )
        elif obj.object_type == ObjectType.CYLINDER:
            visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=obj.scale.x,
                length=obj.scale.z,
                rgbaColor=[0.5, 0.5, 0.5, 1.0],
                physicsClientId=client_id
            )
        else:
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[obj.scale.x/2, obj.scale.y/2, obj.scale.z/2],
                rgbaColor=[0.5, 0.5, 0.5, 1.0],
                physicsClientId=client_id
            )
        
        # Create multi-body
        body_id = p.createMultiBody(
            baseMass=obj.mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=obj.position.to_list(),
            baseOrientation=p.getQuaternionFromEuler(obj.rotation.to_list()),
            physicsClientId=client_id
        )
        
        # Apply material properties
        self._apply_material_properties(body_id, obj.material, client_id)
        
        return body_id
    
    def _apply_material_properties(self, body_id: int, material: MaterialType, client_id: int):
        """Apply material properties to PyBullet object."""
        material_props = {
            MaterialType.WOOD: {'friction': 0.7, 'restitution': 0.3},
            MaterialType.METAL: {'friction': 0.5, 'restitution': 0.1},
            MaterialType.RUBBER: {'friction': 0.9, 'restitution': 0.8},
            MaterialType.PLASTIC: {'friction': 0.6, 'restitution': 0.4},
            MaterialType.GLASS: {'friction': 0.3, 'restitution': 0.1},
            MaterialType.STONE: {'friction': 0.8, 'restitution': 0.2}
        }
        
        props = material_props.get(material, {'friction': 0.5, 'restitution': 0.3})
        
        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=props['friction'],
            restitution=props['restitution'],
            physicsClientId=client_id
        )
    
    def predict_physics_chain(self, scene: DynamicPhysicsScene, 
                            initial_conditions: Dict[str, Any] = None) -> PhysicsChain:
        """
        Predict a physics chain by running actual physics simulation.
        This is the core method that replaces template-based reasoning.
        """
        client_id = self._initialize_simulation()
        
        try:
            # Create ground plane
            p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client_id)
            p.createMultiBody(0, 0, basePosition=[0, 0, 0], physicsClientId=client_id)
            
            # Create all objects in simulation
            object_mapping = {}
            for obj_id, obj in scene.objects.items():
                body_id = self._create_physics_object(obj, client_id)
                object_mapping[obj_id] = body_id
            
            # Apply initial conditions if provided
            if initial_conditions:
                self._apply_initial_conditions(initial_conditions, object_mapping, client_id)
            
            # Run simulation and track events
            steps = self._simulate_and_track_events(scene, object_mapping, client_id)
            
            # Create physics chain from simulation results
            chain = PhysicsChain(
                chain_id=f"simulation_chain_{int(time.time())}",
                description="Physics chain predicted by simulation",
                steps=steps,
                total_duration=self.prediction_horizon,
                overall_confidence=0.95,  # High confidence since based on actual physics
                trigger_conditions=initial_conditions or {}
            )
            
            return chain
            
        finally:
            self._cleanup_simulation(client_id)
    
    def _apply_initial_conditions(self, conditions: Dict[str, Any], 
                                object_mapping: Dict[str, int], client_id: int):
        """Apply initial conditions to simulation objects."""
        for obj_id, condition in conditions.items():
            if obj_id in object_mapping:
                body_id = object_mapping[obj_id]
                
                if 'velocity' in condition:
                    velocity = condition['velocity']
                    p.resetBaseVelocity(
                        body_id,
                        linearVelocity=velocity[:3] if len(velocity) >= 3 else [0, 0, 0],
                        angularVelocity=velocity[3:6] if len(velocity) >= 6 else [0, 0, 0],
                        physicsClientId=client_id
                    )
                
                if 'position' in condition:
                    position = condition['position']
                    p.resetBasePositionAndOrientation(
                        body_id,
                        posObj=position,
                        ornObj=[0, 0, 0, 1],
                        physicsClientId=client_id
                    )
    
    def _simulate_and_track_events(self, scene: DynamicPhysicsScene, 
                                 object_mapping: Dict[str, int], client_id: int) -> List[PhysicsStep]:
        """Run simulation and track significant physics events."""
        steps = []
        previous_states = {}
        collision_pairs = set()
        
        # Initialize previous states
        for obj_id, body_id in object_mapping.items():
            pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=client_id)
            vel, ang_vel = p.getBaseVelocity(body_id, physicsClientId=client_id)
            previous_states[obj_id] = {
                'position': pos,
                'velocity': vel,
                'angular_velocity': ang_vel
            }
        
        # Run simulation
        simulation_time = 0.0
        step_count = 0
        
        while simulation_time < self.prediction_horizon:
            p.stepSimulation(physicsClientId=client_id)
            simulation_time += self.simulation_timestep
            step_count += 1
            
            # Check for events every 10 simulation steps (reduce overhead)
            if step_count % 10 == 0:
                events = self._detect_simulation_events(
                    scene, object_mapping, previous_states, collision_pairs, client_id
                )
                
                for event in events:
                    steps.append(event)
                
                # Update previous states
                for obj_id, body_id in object_mapping.items():
                    pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=client_id)
                    vel, ang_vel = p.getBaseVelocity(body_id, physicsClientId=client_id)
                    previous_states[obj_id] = {
                        'position': pos,
                        'velocity': vel,
                        'angular_velocity': ang_vel
                    }
        
        return steps
    
    def _detect_simulation_events(self, scene: DynamicPhysicsScene, 
                                object_mapping: Dict[str, int],
                                previous_states: Dict[str, Dict[str, Any]],
                                collision_pairs: set, client_id: int) -> List[PhysicsStep]:
        """Detect significant events during simulation."""
        events = []
        
        # Check for collisions
        contact_points = p.getContactPoints(physicsClientId=client_id)
        
        for contact in contact_points:
            body_a, body_b = contact[1], contact[2]
            
            # Find object IDs
            obj_a_id = None
            obj_b_id = None
            for obj_id, body_id in object_mapping.items():
                if body_id == body_a:
                    obj_a_id = obj_id
                elif body_id == body_b:
                    obj_b_id = obj_id
            
            if obj_a_id and obj_b_id:
                collision_key = tuple(sorted([obj_a_id, obj_b_id]))
                
                # Only record new collisions
                if collision_key not in collision_pairs:
                    collision_pairs.add(collision_key)
                    
                    # Get current states
                    pos_a, _ = p.getBasePositionAndOrientation(body_a, physicsClientId=client_id)
                    vel_a, _ = p.getBaseVelocity(body_a, physicsClientId=client_id)
                    pos_b, _ = p.getBasePositionAndOrientation(body_b, physicsClientId=client_id)
                    vel_b, _ = p.getBaseVelocity(body_b, physicsClientId=client_id)
                    
                    collision_step = PhysicsStep(
                        step_id=f"collision_{obj_a_id}_{obj_b_id}_{int(time.time() * 1000)}",
                        timestamp=time.time(),
                        interaction_type=InteractionType.COLLISION,
                        primary_object=obj_a_id,
                        affected_objects=[obj_b_id],
                        initial_state={
                            obj_a_id: {'position': list(pos_a), 'velocity': list(vel_a)},
                            obj_b_id: {'position': list(pos_b), 'velocity': list(vel_b)}
                        },
                        predicted_state={
                            obj_a_id: {'velocity': list(vel_a)},
                            obj_b_id: {'velocity': list(vel_b)}
                        },
                        confidence=0.95,  # High confidence from simulation
                        prerequisites=[]
                    )
                    
                    events.append(collision_step)
        
        # Check for objects coming to rest
        for obj_id, body_id in object_mapping.items():
            vel, ang_vel = p.getBaseVelocity(body_id, physicsClientId=client_id)
            speed = np.linalg.norm(vel)
            
            if speed < 0.01 and obj_id in previous_states:  # Nearly stopped
                prev_speed = np.linalg.norm(previous_states[obj_id]['velocity'])
                
                if prev_speed > 0.1:  # Was moving, now stopped
                    pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=client_id)
                    
                    stop_step = PhysicsStep(
                        step_id=f"stop_{obj_id}_{int(time.time() * 1000)}",
                        timestamp=time.time(),
                        interaction_type=InteractionType.FRICTION_DRAG,
                        primary_object=obj_id,
                        affected_objects=[],
                        initial_state={obj_id: {'velocity': list(previous_states[obj_id]['velocity'])}},
                        predicted_state={obj_id: {'velocity': [0, 0, 0], 'position': list(pos)}},
                        confidence=0.9,
                        prerequisites=[]
                    )
                    
                    events.append(stop_step)
        
        return events
    
    def analyze_scene_stability(self, scene: DynamicPhysicsScene) -> Dict[str, Any]:
        """Analyze if a scene is in stable equilibrium or will change."""
        client_id = self._initialize_simulation()
        
        try:
            # Create objects in simulation
            object_mapping = {}
            for obj_id, obj in scene.objects.items():
                body_id = self._create_physics_object(obj, client_id)
                object_mapping[obj_id] = body_id
            
            # Run brief simulation to check stability
            initial_positions = {}
            for obj_id, body_id in object_mapping.items():
                pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=client_id)
                initial_positions[obj_id] = pos
            
            # Simulate for 2 seconds
            for _ in range(480):  # 2 seconds at 240 Hz
                p.stepSimulation(physicsClientId=client_id)
            
            # Check final positions
            final_positions = {}
            total_movement = 0.0
            
            for obj_id, body_id in object_mapping.items():
                pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=client_id)
                final_positions[obj_id] = pos
                
                # Calculate movement
                initial_pos = np.array(initial_positions[obj_id])
                final_pos = np.array(pos)
                movement = np.linalg.norm(final_pos - initial_pos)
                total_movement += movement
            
            stability_analysis = {
                'is_stable': total_movement < 0.1,  # Less than 10cm total movement
                'total_movement': total_movement,
                'object_movements': {},
                'prediction': "Scene is stable" if total_movement < 0.1 else "Scene will change"
            }
            
            for obj_id in object_mapping.keys():
                initial_pos = np.array(initial_positions[obj_id])
                final_pos = np.array(final_positions[obj_id])
                movement = np.linalg.norm(final_pos - initial_pos)
                stability_analysis['object_movements'][obj_id] = movement
            
            return stability_analysis
            
        finally:
            self._cleanup_simulation(client_id)


def test_physics_simulation_engine():
    """Test the physics simulation engine."""
    print("Testing Physics Simulation Engine...")
    
    # Create test scene
    scene = DynamicPhysicsScene("simulation_test")
    
    # Ball above ground
    ball = DynamicPhysicsObject(
        object_id="test_ball",
        object_type=ObjectType.SPHERE,
        position=Vector3(0, 0, 2),
        rotation=Vector3(0, 0, 0),
        scale=Vector3(0.3, 0.3, 0.3),
        mass=1.0,
        material=MaterialType.RUBBER
    )
    scene.add_object(ball)
    
    # Box on ground
    box = DynamicPhysicsObject(
        object_id="test_box",
        object_type=ObjectType.BOX,
        position=Vector3(1, 0, 0.5),
        rotation=Vector3(0, 0, 0),
        scale=Vector3(0.5, 0.5, 0.5),
        mass=2.0,
        material=MaterialType.WOOD
    )
    scene.add_object(box)
    
    print(f"✅ Created test scene with {scene.get_object_count()} objects")
    
    # Test physics simulation engine
    engine = PhysicsSimulationEngine()
    
    # Test stability analysis
    stability = engine.analyze_scene_stability(scene)
    print(f"✅ Stability analysis: {stability['prediction']}")
    print(f"   Total movement: {stability['total_movement']:.3f}m")
    
    # Test physics chain prediction
    initial_conditions = {
        "test_ball": {"velocity": [1.0, 0, 0]}  # Give ball horizontal velocity
    }
    
    chain = engine.predict_physics_chain(scene, initial_conditions)
    print(f"✅ Predicted physics chain: {len(chain.steps)} events")
    print(f"   Chain confidence: {chain.overall_confidence:.2f}")
    
    for i, step in enumerate(chain.steps, 1):
        print(f"   Event {i}: {step.interaction_type.value}")
        print(f"     Objects: {step.primary_object} → {step.affected_objects}")
        print(f"     Confidence: {step.confidence:.2f}")
    
    print("✅ Physics simulation engine test completed!")


if __name__ == "__main__":
    test_physics_simulation_engine()
