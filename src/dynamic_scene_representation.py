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
    ICE = "ice"
    BOUNCY = "bouncy"
    PLASTIC = "plastic"
    GLASS = "glass"
    STONE = "stone"

class RelationType(Enum):
    """Enumerates the types of spatial relationships between objects."""
    ON = "on"
    NEXT_TO = "next_to"
    ABOVE = "above"
    ON_TOP_OF = "on_top_of"
    BELOW = "below"
    BETWEEN = "between"
    INSIDE = "inside"
    OUTSIDE = "outside"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    UNDER = "under"
    OVER = "over"
    NEAR = "near"
    FAR_FROM = "far_from"
    TO_THE_LEFT_OF = "to_the_left_of"
    TO_THE_RIGHT_OF = "to_the_right_of"
    TO_THE_SIDE_OF = "to_the_side_of"
    ON_THE_BACK_OF = "on_the_back_of"
    ON_THE_FRONT_OF = "on_the_front_of"
    CLOSE_TO = "close_to"
    FAR_AWAY_FROM = "far_away_from" 

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

@dataclass
class SpatialRelation:
    """Represents a spatial relationship between two objects."""
    subject_id: str
    relation_type: RelationType
    target_id: str

class DynamicPhysicsScene:
    def __init__(self, scene_id: str):
        self.scene_id = scene_id
        self.objects: Dict[str, DynamicPhysicsObject] = {}

    def add_object(self, obj: DynamicPhysicsObject):
        self.objects[obj.object_id] = obj

    def get_object(self, object_id: str) -> Optional[DynamicPhysicsObject]:
        """Retrieves an object by its ID."""
        return self.objects.get(object_id)

    def get_object_count(self) -> int:
        """Returns the number of objects in the scene."""
        return len(self.objects)