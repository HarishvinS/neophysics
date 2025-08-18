"""
Advanced Object Types
Expands beyond basic shapes to include complex objects like hinges, springs, chains, and compound objects.
Enables sophisticated physics scenarios with realistic mechanical components.
"""

import pybullet as p
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math

from scene_representation import ObjectType, MaterialType, Vector3, PhysicsObject
from dynamic_scene_representation import DynamicPhysicsObject, DynamicPhysicsScene


class AdvancedObjectType(Enum):
    """Advanced object types beyond basic shapes."""
    # Mechanical components
    HINGE = "hinge"
    SPRING = "spring"
    CHAIN = "chain"
    GEAR = "gear"
    PULLEY = "pulley"
    LEVER = "lever"
    
    # Compound objects
    PENDULUM = "pendulum"
    SEESAW = "seesaw"
    DOMINO = "domino"
    BRIDGE = "bridge"
    TOWER = "tower"
    
    # Flexible objects
    ROPE = "rope"
    CLOTH = "cloth"
    LIQUID = "liquid"
    
    # Vehicles
    CAR = "car"
    CART = "cart"
    WHEEL = "wheel"


@dataclass
class ConstraintDefinition:
    """Defines a physics constraint between objects."""
    constraint_type: str  # "hinge", "spring", "fixed", "slider"
    object_a_id: str
    object_b_id: str
    anchor_a: Vector3
    anchor_b: Vector3
    axis: Vector3
    parameters: Dict[str, Any]


class AdvancedObjectBuilder:
    """Builds complex physics objects with constraints and compound structures."""
    
    def __init__(self, physics_client_id: int = 0):
        """Initialize advanced object builder."""
        self.physics_client = physics_client_id
        self.created_objects = {}
        self.created_constraints = {}
        self.object_counter = 0
    
    def create_advanced_object(self, object_type: AdvancedObjectType, 
                             position: Vector3, 
                             scale: Vector3 = Vector3(1, 1, 1),
                             material: MaterialType = MaterialType.WOOD,
                             **kwargs) -> Dict[str, Any]:
        """Create an advanced physics object."""
        
        if object_type == AdvancedObjectType.HINGE:
            return self._create_hinge(position, scale, material, **kwargs)
        elif object_type == AdvancedObjectType.SPRING:
            return self._create_spring(position, scale, material, **kwargs)
        elif object_type == AdvancedObjectType.CHAIN:
            return self._create_chain(position, scale, material, **kwargs)
        elif object_type == AdvancedObjectType.PENDULUM:
            return self._create_pendulum(position, scale, material, **kwargs)
        elif object_type == AdvancedObjectType.SEESAW:
            return self._create_seesaw(position, scale, material, **kwargs)
        elif object_type == AdvancedObjectType.DOMINO:
            return self._create_domino(position, scale, material, **kwargs)
        elif object_type == AdvancedObjectType.BRIDGE:
            return self._create_bridge(position, scale, material, **kwargs)
        elif object_type == AdvancedObjectType.CAR:
            return self._create_car(position, scale, material, **kwargs)
        elif object_type == AdvancedObjectType.ROPE:
            return self._create_rope(position, scale, material, **kwargs)
        else:
            raise ValueError(f"Unsupported advanced object type: {object_type}")
    
    def _create_hinge(self, position: Vector3, scale: Vector3, material: MaterialType, **kwargs) -> Dict[str, Any]:
        """Create a hinged door or gate."""
        # Create door frame (static)
        frame_id = self._create_box(
            position=Vector3(position.x - scale.x/2, position.y, position.z),
            size=Vector3(0.1, 0.1, scale.z),
            mass=0,  # Static
            material=material
        )
        
        # Create door (dynamic)
        door_id = self._create_box(
            position=Vector3(position.x, position.y, position.z),
            size=Vector3(scale.x, 0.1, scale.z),
            mass=scale.x * scale.z * 2,  # Proportional mass
            material=material
        )
        
        # Create hinge constraint
        hinge_constraint = p.createConstraint(
            parentBodyUniqueId=frame_id,
            parentLinkIndex=-1,
            childBodyUniqueId=door_id,
            childLinkIndex=-1,
            jointType=p.JOINT_REVOLUTE,
            jointAxis=[0, 0, 1],  # Rotate around Z axis
            parentFramePosition=[scale.x/2, 0, 0],
            childFramePosition=[-scale.x/2, 0, 0],
            physicsClientId=self.physics_client
        )
        
        # Set hinge limits
        p.changeConstraint(hinge_constraint, lowerLimit=-math.pi/2, upperLimit=math.pi/2, physicsClientId=self.physics_client)
        
        object_info = {
            'type': 'hinge',
            'components': {'frame': frame_id, 'door': door_id},
            'constraints': [hinge_constraint],
            'position': position.to_list(),
            'scale': scale.to_list()
        }
        
        return object_info
    
    def _create_spring(self, position: Vector3, scale: Vector3, material: MaterialType, **kwargs) -> Dict[str, Any]:
        """Create a spring mechanism."""
        spring_strength = kwargs.get('spring_strength', 100.0)
        damping = kwargs.get('damping', 10.0)
        
        # Create base (static)
        base_id = self._create_box(
            position=Vector3(position.x, position.y, position.z - scale.z/2),
            size=Vector3(scale.x, scale.y, 0.2),
            mass=0,
            material=material
        )
        
        # Create moving platform
        platform_id = self._create_box(
            position=Vector3(position.x, position.y, position.z + scale.z/2),
            size=Vector3(scale.x * 0.8, scale.y * 0.8, 0.1),
            mass=1.0,
            material=material
        )
        
        # Create spring constraint
        spring_constraint = p.createConstraint(
            parentBodyUniqueId=base_id,
            parentLinkIndex=-1,
            childBodyUniqueId=platform_id,
            childLinkIndex=-1,
            jointType=p.JOINT_PRISMATIC,
            jointAxis=[0, 0, 1],  # Move along Z axis
            parentFramePosition=[0, 0, 0.1],
            childFramePosition=[0, 0, -0.05],
            physicsClientId=self.physics_client
        )
        
        # Set spring properties
        p.changeConstraint(
            spring_constraint,
            lowerLimit=-scale.z/2,
            upperLimit=scale.z/2,
            force=spring_strength,
            physicsClientId=self.physics_client
        )
        
        object_info = {
            'type': 'spring',
            'components': {'base': base_id, 'platform': platform_id},
            'constraints': [spring_constraint],
            'position': position.to_list(),
            'scale': scale.to_list(),
            'spring_strength': spring_strength
        }
        
        return object_info
    
    def _create_chain(self, position: Vector3, scale: Vector3, material: MaterialType, **kwargs) -> Dict[str, Any]:
        """Create a chain of connected links."""
        num_links = kwargs.get('num_links', int(scale.z * 5))  # 5 links per unit length
        link_size = scale.z / num_links
        
        links = []
        constraints = []
        
        for i in range(num_links):
            # Create chain link
            link_pos = Vector3(
                position.x,
                position.y,
                position.z + (i - num_links/2) * link_size
            )
            
            link_id = self._create_box(
                position=link_pos,
                size=Vector3(scale.x * 0.1, scale.y * 0.1, link_size * 0.8),
                mass=0.1,
                material=material
            )
            
            links.append(link_id)
            
            # Connect to previous link
            if i > 0:
                constraint = p.createConstraint(
                    parentBodyUniqueId=links[i-1],
                    parentLinkIndex=-1,
                    childBodyUniqueId=link_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_POINT2POINT,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, link_size * 0.4],
                    childFramePosition=[0, 0, -link_size * 0.4],
                    physicsClientId=self.physics_client
                )
                constraints.append(constraint)
        
        object_info = {
            'type': 'chain',
            'components': {'links': links},
            'constraints': constraints,
            'position': position.to_list(),
            'scale': scale.to_list(),
            'num_links': num_links
        }
        
        return object_info
    
    def _create_pendulum(self, position: Vector3, scale: Vector3, material: MaterialType, **kwargs) -> Dict[str, Any]:
        """Create a pendulum."""
        # Create anchor point (static)
        anchor_id = self._create_sphere(
            position=Vector3(position.x, position.y, position.z + scale.z),
            radius=0.05,
            mass=0,
            material=material
        )
        
        # Create pendulum bob
        bob_id = self._create_sphere(
            position=Vector3(position.x, position.y, position.z),
            radius=scale.x * 0.5,
            mass=scale.x * 2,
            material=material
        )
        
        # Create pendulum constraint
        pendulum_constraint = p.createConstraint(
            parentBodyUniqueId=anchor_id,
            parentLinkIndex=-1,
            childBodyUniqueId=bob_id,
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, scale.z],
            physicsClientId=self.physics_client
        )
        
        object_info = {
            'type': 'pendulum',
            'components': {'anchor': anchor_id, 'bob': bob_id},
            'constraints': [pendulum_constraint],
            'position': position.to_list(),
            'scale': scale.to_list()
        }
        
        return object_info
    
    def _create_seesaw(self, position: Vector3, scale: Vector3, material: MaterialType, **kwargs) -> Dict[str, Any]:
        """Create a seesaw."""
        # Create fulcrum (static)
        fulcrum_id = self._create_box(
            position=Vector3(position.x, position.y, position.z - scale.z/4),
            size=Vector3(0.2, scale.y, scale.z/2),
            mass=0,
            material=material
        )
        
        # Create seesaw plank
        plank_id = self._create_box(
            position=position,
            size=Vector3(scale.x, scale.y * 0.8, 0.1),
            mass=scale.x * 0.5,
            material=material
        )
        
        # Create pivot constraint
        pivot_constraint = p.createConstraint(
            parentBodyUniqueId=fulcrum_id,
            parentLinkIndex=-1,
            childBodyUniqueId=plank_id,
            childLinkIndex=-1,
            jointType=p.JOINT_REVOLUTE,
            jointAxis=[0, 1, 0],  # Rotate around Y axis
            parentFramePosition=[0, 0, scale.z/4],
            childFramePosition=[0, 0, 0],
            physicsClientId=self.physics_client
        )
        
        object_info = {
            'type': 'seesaw',
            'components': {'fulcrum': fulcrum_id, 'plank': plank_id},
            'constraints': [pivot_constraint],
            'position': position.to_list(),
            'scale': scale.to_list()
        }
        
        return object_info
    
    def _create_domino(self, position: Vector3, scale: Vector3, material: MaterialType, **kwargs) -> Dict[str, Any]:
        """Create a domino piece."""
        domino_id = self._create_box(
            position=position,
            size=Vector3(scale.x * 0.1, scale.y, scale.z),
            mass=scale.z * 0.5,
            material=material
        )
        
        # Set domino to be slightly unstable for easy tipping
        p.changeDynamics(
            domino_id,
            -1,
            linearDamping=0.1,
            angularDamping=0.1,
            physicsClientId=self.physics_client
        )
        
        object_info = {
            'type': 'domino',
            'components': {'domino': domino_id},
            'constraints': [],
            'position': position.to_list(),
            'scale': scale.to_list()
        }
        
        return object_info
    
    def _create_bridge(self, position: Vector3, scale: Vector3, material: MaterialType, **kwargs) -> Dict[str, Any]:
        """Create a flexible bridge."""
        num_planks = kwargs.get('num_planks', int(scale.x * 3))  # 3 planks per unit length
        plank_width = scale.x / num_planks
        
        planks = []
        constraints = []
        
        for i in range(num_planks):
            # Create bridge plank
            plank_pos = Vector3(
                position.x + (i - num_planks/2) * plank_width,
                position.y,
                position.z
            )
            
            plank_id = self._create_box(
                position=plank_pos,
                size=Vector3(plank_width * 0.9, scale.y, scale.z * 0.1),
                mass=0.2,
                material=material
            )
            
            planks.append(plank_id)
            
            # Connect to previous plank with hinge
            if i > 0:
                constraint = p.createConstraint(
                    parentBodyUniqueId=planks[i-1],
                    parentLinkIndex=-1,
                    childBodyUniqueId=plank_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_REVOLUTE,
                    jointAxis=[0, 1, 0],  # Rotate around Y axis
                    parentFramePosition=[plank_width * 0.45, 0, 0],
                    childFramePosition=[-plank_width * 0.45, 0, 0],
                    physicsClientId=self.physics_client
                )
                constraints.append(constraint)
        
        object_info = {
            'type': 'bridge',
            'components': {'planks': planks},
            'constraints': constraints,
            'position': position.to_list(),
            'scale': scale.to_list(),
            'num_planks': num_planks
        }
        
        return object_info
    
    def _create_car(self, position: Vector3, scale: Vector3, material: MaterialType, **kwargs) -> Dict[str, Any]:
        """Create a simple car with wheels."""
        # Create car body
        body_id = self._create_box(
            position=Vector3(position.x, position.y, position.z + scale.z * 0.3),
            size=Vector3(scale.x, scale.y * 0.6, scale.z * 0.3),
            mass=scale.x * scale.y * 2,
            material=material
        )
        
        # Create wheels
        wheel_radius = scale.z * 0.3
        wheel_positions = [
            Vector3(position.x - scale.x * 0.3, position.y - scale.y * 0.4, position.z),
            Vector3(position.x + scale.x * 0.3, position.y - scale.y * 0.4, position.z),
            Vector3(position.x - scale.x * 0.3, position.y + scale.y * 0.4, position.z),
            Vector3(position.x + scale.x * 0.3, position.y + scale.y * 0.4, position.z)
        ]
        
        wheels = []
        wheel_constraints = []
        
        for i, wheel_pos in enumerate(wheel_positions):
            wheel_id = self._create_cylinder(
                position=wheel_pos,
                radius=wheel_radius,
                height=0.1,
                mass=0.5,
                material=MaterialType.RUBBER
            )
            wheels.append(wheel_id)
            
            # Create wheel constraint
            constraint = p.createConstraint(
                parentBodyUniqueId=body_id,
                parentLinkIndex=-1,
                childBodyUniqueId=wheel_id,
                childLinkIndex=-1,
                jointType=p.JOINT_REVOLUTE,
                jointAxis=[0, 1, 0],  # Rotate around Y axis
                parentFramePosition=[
                    wheel_pos.x - position.x,
                    wheel_pos.y - position.y,
                    wheel_pos.z - (position.z + scale.z * 0.3)
                ],
                childFramePosition=[0, 0, 0],
                physicsClientId=self.physics_client
            )
            wheel_constraints.append(constraint)
        
        object_info = {
            'type': 'car',
            'components': {'body': body_id, 'wheels': wheels},
            'constraints': wheel_constraints,
            'position': position.to_list(),
            'scale': scale.to_list()
        }
        
        return object_info
    
    def _create_rope(self, position: Vector3, scale: Vector3, material: MaterialType, **kwargs) -> Dict[str, Any]:
        """Create a flexible rope."""
        num_segments = kwargs.get('num_segments', int(scale.z * 10))  # 10 segments per unit length
        segment_length = scale.z / num_segments
        
        segments = []
        constraints = []
        
        for i in range(num_segments):
            # Create rope segment
            segment_pos = Vector3(
                position.x,
                position.y,
                position.z + (i - num_segments/2) * segment_length
            )
            
            segment_id = self._create_cylinder(
                position=segment_pos,
                radius=scale.x * 0.05,
                height=segment_length * 0.8,
                mass=0.01,
                material=material
            )
            
            segments.append(segment_id)
            
            # Connect to previous segment
            if i > 0:
                constraint = p.createConstraint(
                    parentBodyUniqueId=segments[i-1],
                    parentLinkIndex=-1,
                    childBodyUniqueId=segment_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_POINT2POINT,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, segment_length * 0.4],
                    childFramePosition=[0, 0, -segment_length * 0.4],
                    physicsClientId=self.physics_client
                )
                constraints.append(constraint)
        
        object_info = {
            'type': 'rope',
            'components': {'segments': segments},
            'constraints': constraints,
            'position': position.to_list(),
            'scale': scale.to_list(),
            'num_segments': num_segments
        }
        
        return object_info
    
    def _create_box(self, position: Vector3, size: Vector3, mass: float, material: MaterialType) -> int:
        """Helper to create a box."""
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[size.x/2, size.y/2, size.z/2],
            physicsClientId=self.physics_client
        )
        
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[size.x/2, size.y/2, size.z/2],
            rgbaColor=self._get_material_color(material),
            physicsClientId=self.physics_client
        )
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position.to_list(),
            physicsClientId=self.physics_client
        )
        
        self._apply_material_properties(body_id, material)
        return body_id
    
    def _create_sphere(self, position: Vector3, radius: float, mass: float, material: MaterialType) -> int:
        """Helper to create a sphere."""
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius,
            physicsClientId=self.physics_client
        )
        
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=self._get_material_color(material),
            physicsClientId=self.physics_client
        )
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position.to_list(),
            physicsClientId=self.physics_client
        )
        
        self._apply_material_properties(body_id, material)
        return body_id
    
    def _create_cylinder(self, position: Vector3, radius: float, height: float, mass: float, material: MaterialType) -> int:
        """Helper to create a cylinder."""
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=radius,
            height=height,
            physicsClientId=self.physics_client
        )
        
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=self._get_material_color(material),
            physicsClientId=self.physics_client
        )
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position.to_list(),
            physicsClientId=self.physics_client
        )
        
        self._apply_material_properties(body_id, material)
        return body_id
    
    def _get_material_color(self, material: MaterialType) -> List[float]:
        """Get color for material type."""
        colors = {
            MaterialType.WOOD: [0.6, 0.4, 0.2, 1.0],
            MaterialType.METAL: [0.7, 0.7, 0.8, 1.0],
            MaterialType.RUBBER: [0.2, 0.2, 0.2, 1.0],
            MaterialType.PLASTIC: [0.8, 0.2, 0.2, 1.0],
            MaterialType.GLASS: [0.9, 0.9, 0.9, 0.7],
            MaterialType.STONE: [0.5, 0.5, 0.5, 1.0]
        }
        return colors.get(material, [0.5, 0.5, 0.5, 1.0])
    
    def _apply_material_properties(self, body_id: int, material: MaterialType):
        """Apply material properties to object."""
        properties = {
            MaterialType.WOOD: {'friction': 0.7, 'restitution': 0.3},
            MaterialType.METAL: {'friction': 0.5, 'restitution': 0.1},
            MaterialType.RUBBER: {'friction': 0.9, 'restitution': 0.8},
            MaterialType.PLASTIC: {'friction': 0.6, 'restitution': 0.4},
            MaterialType.GLASS: {'friction': 0.3, 'restitution': 0.1},
            MaterialType.STONE: {'friction': 0.8, 'restitution': 0.2}
        }
        
        props = properties.get(material, {'friction': 0.5, 'restitution': 0.3})
        
        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=props['friction'],
            restitution=props['restitution'],
            physicsClientId=self.physics_client
        )


def test_advanced_object_types():
    """Test the advanced object types system."""
    print("Testing Advanced Object Types...")
    
    # Initialize PyBullet
    physics_client = p.connect(p.DIRECT)  # Use DIRECT for testing
    p.setGravity(0, 0, -9.81, physicsClientId=physics_client)
    
    # Create ground
    p.createCollisionShape(p.GEOM_PLANE, physicsClientId=physics_client)
    p.createMultiBody(0, 0, basePosition=[0, 0, 0], physicsClientId=physics_client)
    
    # Create advanced object builder
    builder = AdvancedObjectBuilder(physics_client)
    
    # Test different advanced object types
    test_objects = [
        (AdvancedObjectType.HINGE, Vector3(0, 0, 1), Vector3(2, 1, 2)),
        (AdvancedObjectType.SPRING, Vector3(3, 0, 1), Vector3(1, 1, 2)),
        (AdvancedObjectType.PENDULUM, Vector3(6, 0, 1), Vector3(0.5, 0.5, 2)),
        (AdvancedObjectType.CHAIN, Vector3(9, 0, 3), Vector3(0.2, 0.2, 3)),
        (AdvancedObjectType.SEESAW, Vector3(12, 0, 1), Vector3(4, 1, 1))
    ]
    
    created_objects = []
    
    for obj_type, position, scale in test_objects:
        try:
            obj_info = builder.create_advanced_object(
                obj_type, position, scale, MaterialType.WOOD
            )
            created_objects.append(obj_info)
            print(f"✅ Created {obj_type.value}: {len(obj_info['components'])} components, {len(obj_info['constraints'])} constraints")
        
        except Exception as e:
            print(f"❌ Failed to create {obj_type.value}: {e}")
    
    # Run brief simulation
    print(f"\nRunning simulation with {len(created_objects)} advanced objects...")
    
    for step in range(100):
        p.stepSimulation(physicsClientId=physics_client)
        time.sleep(0.01)
    
    print(f"✅ Simulation completed successfully")
    
    # Cleanup
    p.disconnect(physicsClientId=physics_client)
    
    print("✅ Advanced object types test completed!")


if __name__ == "__main__":
    test_advanced_object_types()
