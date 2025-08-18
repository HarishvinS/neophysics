"""
Dynamic Scene Representation
Variable-size scene representation that can handle arbitrary numbers of objects
and complex spatial relationships without fixed-size constraints.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

from scene_representation import PhysicsScene, PhysicsObject, ObjectType, MaterialType, Vector3


class RelationType(Enum):
    """Types of spatial relationships between objects."""
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    BETWEEN = "between"
    NEAR = "near"
    FAR_FROM = "far_from"
    TOUCHING = "touching"
    INSIDE = "inside"
    ON_TOP_OF = "on_top_of"
    SUPPORTING = "supporting"


@dataclass
class SpatialRelation:
    """Represents a spatial relationship between objects."""
    relation_type: RelationType
    subject_id: str  # The object being described
    target_id: str   # The reference object
    confidence: float = 1.0
    parameters: Dict[str, Any] = None  # Additional parameters (e.g., distance for "near")
    
    def to_dict(self):
        return {
            'relation_type': self.relation_type.value,
            'subject_id': self.subject_id,
            'target_id': self.target_id,
            'confidence': self.confidence,
            'parameters': self.parameters or {}
        }


@dataclass
class DynamicPhysicsObject:
    """Enhanced physics object with relationship awareness."""
    object_id: str
    object_type: ObjectType
    position: Vector3
    rotation: Vector3
    scale: Vector3
    mass: float
    material: MaterialType
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    
    # Dynamic properties
    relationships: List[SpatialRelation] = None
    properties: Dict[str, Any] = None
    constraints: List[str] = None  # Physics constraints (e.g., "fixed", "kinematic")
    
    def __post_init__(self):
        if self.relationships is None:
            self.relationships = []
        if self.properties is None:
            self.properties = {}
        if self.constraints is None:
            self.constraints = []
    
    def add_relationship(self, relation: SpatialRelation):
        """Add a spatial relationship."""
        self.relationships.append(relation)
    
    def get_relationships_of_type(self, relation_type: RelationType) -> List[SpatialRelation]:
        """Get all relationships of a specific type."""
        return [r for r in self.relationships if r.relation_type == relation_type]
    
    def to_dict(self):
        return {
            'object_id': self.object_id,
            'object_type': self.object_type.value,
            'position': self.position.to_list(),
            'rotation': self.rotation.to_list(),
            'scale': self.scale.to_list(),
            'mass': self.mass,
            'material': self.material.value,
            'color': list(self.color),
            'relationships': [r.to_dict() for r in self.relationships],
            'properties': self.properties,
            'constraints': self.constraints
        }


class DynamicPhysicsScene:
    """Variable-size physics scene with relationship modeling."""
    
    def __init__(self, scene_id: str = None):
        """Initialize dynamic scene."""
        self.scene_id = scene_id or f"scene_{np.random.randint(10000, 99999)}"
        self.objects: Dict[str, DynamicPhysicsObject] = {}
        self.global_relationships: List[SpatialRelation] = []
        self.scene_properties: Dict[str, Any] = {
            'gravity': [0, 0, -9.81],
            'time_step': 1/240,
            'environment': 'default'
        }
        self.metadata: Dict[str, Any] = {}
    
    def add_object(self, obj: DynamicPhysicsObject):
        """Add an object to the scene."""
        self.objects[obj.object_id] = obj
    
    def remove_object(self, object_id: str):
        """Remove an object and all its relationships."""
        if object_id in self.objects:
            # Remove object
            del self.objects[object_id]
            
            # Remove relationships involving this object
            self.global_relationships = [
                r for r in self.global_relationships 
                if r.subject_id != object_id and r.target_id != object_id
            ]
            
            # Remove relationships from other objects
            for obj in self.objects.values():
                obj.relationships = [
                    r for r in obj.relationships 
                    if r.subject_id != object_id and r.target_id != object_id
                ]
    
    def add_relationship(self, relation: SpatialRelation):
        """Add a global relationship between objects."""
        self.global_relationships.append(relation)
        
        # Also add to subject object if it exists
        if relation.subject_id in self.objects:
            self.objects[relation.subject_id].add_relationship(relation)
    
    def get_object_count(self) -> int:
        """Get the number of objects in the scene."""
        return len(self.objects)
    
    def get_objects_by_type(self, object_type: ObjectType) -> List[DynamicPhysicsObject]:
        """Get all objects of a specific type."""
        return [obj for obj in self.objects.values() if obj.object_type == object_type]
    
    def get_relationships_involving(self, object_id: str) -> List[SpatialRelation]:
        """Get all relationships involving a specific object."""
        return [
            r for r in self.global_relationships 
            if r.subject_id == object_id or r.target_id == object_id
        ]
    
    def validate_relationships(self) -> List[str]:
        """Validate that all relationships are physically plausible."""
        errors = []
        
        for relation in self.global_relationships:
            # Check that referenced objects exist
            if relation.subject_id not in self.objects:
                errors.append(f"Subject object {relation.subject_id} not found")
            if relation.target_id not in self.objects:
                errors.append(f"Target object {relation.target_id} not found")
                continue
            
            # Validate specific relationship types
            if relation.relation_type == RelationType.ABOVE:
                subject = self.objects[relation.subject_id]
                target = self.objects[relation.target_id]
                if subject.position.z <= target.position.z:
                    errors.append(f"Object {relation.subject_id} is not above {relation.target_id}")
        
        return errors
    
    def to_dict(self):
        """Convert scene to dictionary."""
        return {
            'scene_id': self.scene_id,
            'objects': {obj_id: obj.to_dict() for obj_id, obj in self.objects.items()},
            'global_relationships': [r.to_dict() for r in self.global_relationships],
            'scene_properties': self.scene_properties,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create scene from dictionary."""
        scene = cls(data['scene_id'])
        scene.scene_properties = data.get('scene_properties', {})
        scene.metadata = data.get('metadata', {})
        
        # Load objects
        for obj_id, obj_data in data.get('objects', {}).items():
            obj = DynamicPhysicsObject(
                object_id=obj_data['object_id'],
                object_type=ObjectType(obj_data['object_type']),
                position=Vector3(*obj_data['position']),
                rotation=Vector3(*obj_data['rotation']),
                scale=Vector3(*obj_data['scale']),
                mass=obj_data['mass'],
                material=MaterialType(obj_data['material']),
                color=tuple(obj_data['color']),
                properties=obj_data.get('properties', {}),
                constraints=obj_data.get('constraints', [])
            )
            
            # Load object relationships
            for rel_data in obj_data.get('relationships', []):
                relation = SpatialRelation(
                    relation_type=RelationType(rel_data['relation_type']),
                    subject_id=rel_data['subject_id'],
                    target_id=rel_data['target_id'],
                    confidence=rel_data.get('confidence', 1.0),
                    parameters=rel_data.get('parameters', {})
                )
                obj.add_relationship(relation)
            
            scene.add_object(obj)
        
        # Load global relationships
        for rel_data in data.get('global_relationships', []):
            relation = SpatialRelation(
                relation_type=RelationType(rel_data['relation_type']),
                subject_id=rel_data['subject_id'],
                target_id=rel_data['target_id'],
                confidence=rel_data.get('confidence', 1.0),
                parameters=rel_data.get('parameters', {})
            )
            scene.global_relationships.append(relation)
        
        return scene


class SpatialRelationshipAnalyzer:
    """Analyzes and infers spatial relationships from object positions."""
    
    def __init__(self, tolerance: float = 0.1):
        """
        Initialize analyzer.
        
        Args:
            tolerance: Distance tolerance for relationship detection
        """
        self.tolerance = tolerance
    
    def analyze_scene(self, scene: DynamicPhysicsScene) -> List[SpatialRelation]:
        """Analyze a scene and infer spatial relationships."""
        relationships = []
        objects = list(scene.objects.values())
        
        # Analyze pairwise relationships
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j:
                    relations = self.analyze_pair(obj1, obj2)
                    relationships.extend(relations)
        
        return relationships
    
    def analyze_pair(self, obj1: DynamicPhysicsObject, obj2: DynamicPhysicsObject) -> List[SpatialRelation]:
        """Analyze relationship between two objects."""
        relationships = []
        
        pos1 = obj1.position
        pos2 = obj2.position
        
        # Calculate relative positions
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        dz = pos1.z - pos2.z
        
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Vertical relationships
        if abs(dz) > self.tolerance:
            if dz > 0:
                relationships.append(SpatialRelation(
                    RelationType.ABOVE, obj1.object_id, obj2.object_id,
                    confidence=min(1.0, abs(dz) / 2.0)
                ))
            else:
                relationships.append(SpatialRelation(
                    RelationType.BELOW, obj1.object_id, obj2.object_id,
                    confidence=min(1.0, abs(dz) / 2.0)
                ))
        
        # Horizontal relationships
        if abs(dx) > abs(dy) and abs(dx) > self.tolerance:
            if dx > 0:
                relationships.append(SpatialRelation(
                    RelationType.RIGHT_OF, obj1.object_id, obj2.object_id,
                    confidence=min(1.0, abs(dx) / 2.0)
                ))
            else:
                relationships.append(SpatialRelation(
                    RelationType.LEFT_OF, obj1.object_id, obj2.object_id,
                    confidence=min(1.0, abs(dx) / 2.0)
                ))
        
        if abs(dy) > abs(dx) and abs(dy) > self.tolerance:
            if dy > 0:
                relationships.append(SpatialRelation(
                    RelationType.IN_FRONT_OF, obj1.object_id, obj2.object_id,
                    confidence=min(1.0, abs(dy) / 2.0)
                ))
            else:
                relationships.append(SpatialRelation(
                    RelationType.BEHIND, obj1.object_id, obj2.object_id,
                    confidence=min(1.0, abs(dy) / 2.0)
                ))
        
        # Distance-based relationships
        if distance < 1.0:
            relationships.append(SpatialRelation(
                RelationType.NEAR, obj1.object_id, obj2.object_id,
                confidence=1.0 - distance,
                parameters={'distance': distance}
            ))
        
        if distance < 0.2:  # Very close
            relationships.append(SpatialRelation(
                RelationType.TOUCHING, obj1.object_id, obj2.object_id,
                confidence=1.0 - (distance / 0.2),
                parameters={'distance': distance}
            ))
        
        return relationships


def test_dynamic_scene_representation():
    """Test the dynamic scene representation."""
    print("Testing Dynamic Scene Representation...")
    
    # Create a dynamic scene
    scene = DynamicPhysicsScene("test_scene")
    
    # Add objects
    ball = DynamicPhysicsObject(
        object_id="ball_1",
        object_type=ObjectType.SPHERE,
        position=Vector3(0, 0, 2),
        rotation=Vector3(0, 0, 0),
        scale=Vector3(0.5, 0.5, 0.5),
        mass=1.0,
        material=MaterialType.RUBBER
    )
    
    ramp = DynamicPhysicsObject(
        object_id="ramp_1",
        object_type=ObjectType.RAMP,
        position=Vector3(0, 0, 0),
        rotation=Vector3(0, 0.3, 0),
        scale=Vector3(2, 0.1, 1),
        mass=0,  # Static
        material=MaterialType.WOOD,
        constraints=["fixed"]
    )
    
    scene.add_object(ball)
    scene.add_object(ramp)
    
    # Add relationships
    above_relation = SpatialRelation(
        RelationType.ABOVE, "ball_1", "ramp_1", confidence=0.9
    )
    scene.add_relationship(above_relation)
    
    print(f"✅ Created scene with {scene.get_object_count()} objects")
    
    # Test relationship analysis
    analyzer = SpatialRelationshipAnalyzer()
    inferred_relations = analyzer.analyze_scene(scene)
    
    print(f"✅ Inferred {len(inferred_relations)} spatial relationships")
    for relation in inferred_relations[:3]:  # Show first 3
        print(f"   {relation.subject_id} {relation.relation_type.value} {relation.target_id} (confidence: {relation.confidence:.2f})")
    
    # Test validation
    errors = scene.validate_relationships()
    print(f"✅ Validation: {len(errors)} errors found")
    
    # Test serialization
    scene_dict = scene.to_dict()
    reconstructed_scene = DynamicPhysicsScene.from_dict(scene_dict)
    
    print(f"✅ Serialization: {reconstructed_scene.get_object_count()} objects reconstructed")
    
    print("✅ Dynamic scene representation test completed!")


if __name__ == "__main__":
    test_dynamic_scene_representation()
