"""
Data structures for representing a dynamic 3D physics scene.
Used for data generation, ML-physics bridge, and manual creation tools.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional


class ObjectType(Enum):
    SPHERE = "sphere"
    BOX = "box"
    RAMP = "ramp"

class MaterialType(Enum):
    WOOD = "wood"
    METAL = "metal"
    RUBBER = "rubber"

@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

@dataclass
class DynamicPhysicsObject:
    object_id: str
    object_type: ObjectType
    position: Vector3
    rotation: Vector3 = field(default_factory=Vector3)
    scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))
    mass: float = 1.0
    material: MaterialType = MaterialType.WOOD
    color: Tuple[float, float, float] = (1, 1, 1)
    initial_velocity: Vector3 = field(default_factory=Vector3)

class DynamicPhysicsScene:
    def __init__(self, scene_id: str):
        self.scene_id = scene_id
        self.objects: Dict[str, DynamicPhysicsObject] = {}

    def add_object(self, obj: DynamicPhysicsObject):
        self.objects[obj.object_id] = obj