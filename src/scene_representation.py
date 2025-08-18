"""
Scene Representation System
Defines data structures for representing physics scenes and converting them to/from various formats.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
import json
import numpy as np
from enum import Enum


class ObjectType(Enum):
    """Enumeration of supported object types."""
    SPHERE = "sphere"
    BOX = "box"
    CYLINDER = "cylinder"
    RAMP = "ramp"
    PLANE = "plane"
    CONE = "cone"


class MaterialType(Enum):
    """Enumeration of material types with different physical properties."""
    RUBBER = "rubber"
    METAL = "metal"
    WOOD = "wood"
    ICE = "ice"
    BOUNCY = "bouncy"
    PLASTIC = "plastic"
    GLASS = "glass"
    STONE = "stone"


@dataclass
class Vector3:
    """3D vector representation."""
    x: float
    y: float
    z: float
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple format."""
        return (self.x, self.y, self.z)
    
    def to_list(self) -> List[float]:
        """Convert to list format."""
        return [self.x, self.y, self.z]
    
    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> 'Vector3':
        """Create from tuple."""
        return cls(t[0], t[1], t[2])
    
    @classmethod
    def zero(cls) -> 'Vector3':
        """Create zero vector."""
        return cls(0.0, 0.0, 0.0)


@dataclass
class MaterialProperties:
    """Physical material properties."""
    friction: float = 0.5
    restitution: float = 0.3  # Bounciness
    density: float = 1000.0   # kg/m³
    
    # Class-level material database to avoid recreation
    _MATERIAL_DB = {
        MaterialType.RUBBER: lambda: MaterialProperties(friction=0.8, restitution=0.9, density=1200),
        MaterialType.METAL: lambda: MaterialProperties(friction=0.6, restitution=0.3, density=7800),
        MaterialType.WOOD: lambda: MaterialProperties(friction=0.7, restitution=0.5, density=600),
        MaterialType.ICE: lambda: MaterialProperties(friction=0.05, restitution=0.1, density=900),
        MaterialType.BOUNCY: lambda: MaterialProperties(friction=0.3, restitution=0.95, density=500),
        MaterialType.PLASTIC: lambda: MaterialProperties(friction=0.4, restitution=0.6, density=1000),
        MaterialType.GLASS: lambda: MaterialProperties(friction=0.2, restitution=0.1, density=2500),
        MaterialType.STONE: lambda: MaterialProperties(friction=0.8, restitution=0.2, density=2700),
    }
    
    @classmethod
    def from_material_type(cls, material: MaterialType) -> 'MaterialProperties':
        """Create material properties from material type."""
        factory = cls._MATERIAL_DB.get(material)
        return factory() if factory else cls()


@dataclass
class PhysicsObject:
    """Represents a single physics object in the scene."""
    object_id: str
    object_type: ObjectType
    position: Vector3
    rotation: Vector3  # Euler angles in radians
    scale: Vector3
    mass: float
    material: MaterialType
    material_properties: MaterialProperties
    initial_velocity: Vector3 = None
    initial_angular_velocity: Vector3 = None
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    
    def __post_init__(self):
        """Initialize default values after creation."""
        if self.initial_velocity is None:
            self.initial_velocity = Vector3.zero()
        if self.initial_angular_velocity is None:
            self.initial_angular_velocity = Vector3.zero()
        if self.material_properties is None:
            self.material_properties = MaterialProperties.from_material_type(self.material)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'object_id': self.object_id,
            'object_type': self.object_type.value,
            'position': self.position.to_list(),
            'rotation': self.rotation.to_list(),
            'scale': self.scale.to_list(),
            'mass': self.mass,
            'material': self.material.value,
            'material_properties': asdict(self.material_properties),
            'initial_velocity': self.initial_velocity.to_list(),
            'initial_angular_velocity': self.initial_angular_velocity.to_list(),
            'color': list(self.color)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhysicsObject':
        """Create from dictionary."""
        return cls(
            object_id=data['object_id'],
            object_type=ObjectType(data['object_type']),
            position=Vector3(*data['position']),
            rotation=Vector3(*data['rotation']),
            scale=Vector3(*data['scale']),
            mass=data['mass'],
            material=MaterialType(data['material']),
            material_properties=MaterialProperties(**data['material_properties']),
            initial_velocity=Vector3(*data['initial_velocity']),
            initial_angular_velocity=Vector3(*data['initial_angular_velocity']),
            color=tuple(data['color'])
        )


@dataclass
class SceneEnvironment:
    """Global scene environment settings."""
    gravity: Vector3 = None
    air_resistance: float = 0.0
    scene_bounds: Tuple[float, float, float, float, float, float] = (-10, 10, -10, 10, -5, 10)  # x_min, x_max, y_min, y_max, z_min, z_max
    time_step: float = 1.0/240.0

    def __post_init__(self):
        """Initialize default values."""
        if self.gravity is None:
            self.gravity = Vector3(0, 0, -9.81)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'gravity': self.gravity.to_list(),
            'air_resistance': self.air_resistance,
            'scene_bounds': list(self.scene_bounds),
            'time_step': self.time_step
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SceneEnvironment':
        """Create from dictionary."""
        return cls(
            gravity=Vector3(*data['gravity']),
            air_resistance=data['air_resistance'],
            scene_bounds=tuple(data['scene_bounds']),
            time_step=data['time_step']
        )


@dataclass
class PhysicsScene:
    """Complete physics scene representation."""
    scene_id: str
    objects: List[PhysicsObject]
    environment: SceneEnvironment
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.tags is None:
            self.tags = []
    
    def add_object(self, obj: PhysicsObject):
        """Add an object to the scene."""
        self.objects.append(obj)
    
    def get_object_by_id(self, object_id: str) -> Optional[PhysicsObject]:
        """Get object by ID."""
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None
    
    def get_objects_by_type(self, object_type: ObjectType) -> List[PhysicsObject]:
        """Get all objects of a specific type."""
        return [obj for obj in self.objects if obj.object_type == object_type]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'scene_id': self.scene_id,
            'objects': [obj.to_dict() for obj in self.objects],
            'environment': self.environment.to_dict(),
            'description': self.description,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhysicsScene':
        """Create from dictionary."""
        return cls(
            scene_id=data['scene_id'],
            objects=[PhysicsObject.from_dict(obj_data) for obj_data in data['objects']],
            environment=SceneEnvironment.from_dict(data['environment']),
            description=data.get('description', ''),
            tags=data.get('tags', [])
        )
    
    def save_to_file(self, filepath: str):
        """Save scene to JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except (IOError, PermissionError, OSError) as e:
            print(f"Error saving scene to {filepath}: {e}")
            raise
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'PhysicsScene':
        """Load scene from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (IOError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading scene from {filepath}: {e}")
            raise


@dataclass
class TrainingExample:
    """A single training example pairing text with a physics scene."""
    example_id: str
    text_description: str
    scene: PhysicsScene
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'example_id': self.example_id,
            'text_description': self.text_description,
            'scene': self.scene.to_dict(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingExample':
        """Create from dictionary."""
        return cls(
            example_id=data['example_id'],
            text_description=data['text_description'],
            scene=PhysicsScene.from_dict(data['scene']),
            metadata=data.get('metadata', {})
        )


class SceneBuilder:
    """Helper class for building physics scenes programmatically."""
    
    def __init__(self, scene_id: str):
        """Initialize scene builder."""
        self.scene = PhysicsScene(
            scene_id=scene_id,
            objects=[],
            environment=SceneEnvironment()
        )
        self.object_counter = 0
    
    def add_ground_plane(self) -> PhysicsObject:
        """Add a ground plane to the scene."""
        ground = PhysicsObject(
            object_id=f"ground_{self.object_counter}",
            object_type=ObjectType.PLANE,
            position=Vector3(0, 0, 0),
            rotation=Vector3(0, 0, 0),
            scale=Vector3(10, 10, 1),
            mass=0,  # Static
            material=MaterialType.STONE,
            material_properties=MaterialProperties.from_material_type(MaterialType.STONE),
            color=(0.5, 0.5, 0.5)
        )
        self.scene.add_object(ground)
        self.object_counter += 1
        return ground
    
    def add_sphere(self, position: Tuple[float, float, float], radius: float = 0.1, 
                   mass: float = 1.0, material: MaterialType = MaterialType.RUBBER) -> PhysicsObject:
        """Add a sphere to the scene."""
        sphere = PhysicsObject(
            object_id=f"sphere_{self.object_counter}",
            object_type=ObjectType.SPHERE,
            position=Vector3.from_tuple(position),
            rotation=Vector3.zero(),
            scale=Vector3(radius, radius, radius),
            mass=mass,
            material=material,
            material_properties=MaterialProperties.from_material_type(material),
            color=(1, 0, 0)  # Default red
        )
        self.scene.add_object(sphere)
        self.object_counter += 1
        return sphere
    
    def add_ramp(self, position: Tuple[float, float, float], angle: float = 0.3,
                 size: Tuple[float, float, float] = (2, 0.2, 1)) -> PhysicsObject:
        """Add a ramp to the scene."""
        ramp = PhysicsObject(
            object_id=f"ramp_{self.object_counter}",
            object_type=ObjectType.RAMP,
            position=Vector3.from_tuple(position),
            rotation=Vector3(0, angle, 0),
            scale=Vector3.from_tuple(size),
            mass=0,  # Static
            material=MaterialType.WOOD,
            material_properties=MaterialProperties.from_material_type(MaterialType.WOOD),
            color=(0.6, 0.4, 0.2)  # Brown
        )
        self.scene.add_object(ramp)
        self.object_counter += 1
        return ramp
    
    def get_scene(self) -> PhysicsScene:
        """Get the built scene."""
        return self.scene


# Test function
def test_scene_representation():
    """Test the scene representation system."""
    print("Testing scene representation system...")
    
    # Create a scene using the builder
    builder = SceneBuilder("test_scene_001")
    
    # Add ground
    ground = builder.add_ground_plane()
    
    # Add a ramp
    ramp = builder.add_ramp(position=(0, 0, 0), angle=0.4)
    
    # Add a ball
    ball = builder.add_sphere(position=(-1, 0, 1.5), radius=0.1, mass=2.0, material=MaterialType.RUBBER)
    
    # Get the scene
    scene = builder.get_scene()
    scene.description = "A rubber ball rolling down a wooden ramp"
    scene.tags = ["ramp", "ball", "rolling", "gravity"]
    
    # Test serialization
    scene_dict = scene.to_dict()
    print(f"Scene serialized successfully: {len(scene_dict)} keys")
    
    # Test deserialization
    scene_restored = PhysicsScene.from_dict(scene_dict)
    print(f"Scene restored: {len(scene_restored.objects)} objects")
    
    # Test file I/O
    scene.save_to_file("data/test_scene.json")
    scene_loaded = PhysicsScene.load_from_file("data/test_scene.json")
    print(f"Scene loaded from file: {scene_loaded.scene_id}")
    
    # Create training example
    example = TrainingExample(
        example_id="example_001",
        text_description="Create a rubber ball and place it on a wooden ramp so it rolls down",
        scene=scene,
        metadata={"difficulty": "easy", "concepts": ["gravity", "friction", "rolling"]}
    )
    
    print(f"Training example created: {example.example_id}")
    print(f"Text: {example.text_description}")
    
    print("✅ Scene representation system working correctly!")


if __name__ == "__main__":
    test_scene_representation()
