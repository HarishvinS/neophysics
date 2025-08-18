"""
Relational Understanding System
Implements spatial relationship parsing that understands concepts like 'between', 'above', 'next to' dynamically.
Goes beyond fixed templates to understand spatial relationships contextually.
"""

import torch
import torch.nn as nn
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
# import spacy  # Optional dependency
# from transformers import AutoTokenizer, AutoModel  # Optional dependency

from dynamic_scene_representation import DynamicPhysicsScene, DynamicPhysicsObject, SpatialRelation, RelationType
from scene_representation import ObjectType, MaterialType, Vector3


@dataclass
class SpatialConcept:
    """Represents a spatial concept extracted from text."""
    concept_type: str  # 'position', 'relation', 'arrangement'
    primary_object: str
    reference_objects: List[str]
    spatial_relation: RelationType
    parameters: Dict[str, Any]
    confidence: float
    text_span: str


class SpatialLanguageParser:
    """Parses natural language to extract spatial relationships."""
    
    def __init__(self):
        """Initialize spatial language parser."""
        # Load spaCy model for NLP (optional)
        self.nlp = None  # Simplified version without spaCy
        
        # Spatial relationship patterns
        self.spatial_patterns = {
            RelationType.ABOVE: [
                r"(\w+)\s+(?:is\s+)?(?:placed\s+)?(?:on\s+top\s+of|above|over)\s+(?:the\s+)?(\w+)",
                r"put\s+(?:the\s+)?(\w+)\s+(?:on\s+top\s+of|above|over)\s+(?:the\s+)?(\w+)",
                r"place\s+(?:a\s+|the\s+)?(\w+)\s+(?:on\s+top\s+of|above|over)\s+(?:the\s+)?(\w+)"
            ],
            RelationType.BELOW: [
                r"(\w+)\s+(?:is\s+)?(?:placed\s+)?(?:below|under|beneath)\s+(?:the\s+)?(\w+)",
                r"put\s+(?:the\s+)?(\w+)\s+(?:below|under|beneath)\s+(?:the\s+)?(\w+)"
            ],
            RelationType.BETWEEN: [
                r"(\w+)\s+(?:is\s+)?(?:placed\s+)?between\s+(?:the\s+)?(\w+)\s+and\s+(?:the\s+)?(\w+)",
                r"put\s+(?:the\s+)?(\w+)\s+between\s+(?:the\s+)?(\w+)\s+and\s+(?:the\s+)?(\w+)",
                r"place\s+(?:a\s+|the\s+)?(\w+)\s+between\s+(?:the\s+)?(\w+)\s+and\s+(?:the\s+)?(\w+)"
            ],
            RelationType.NEAR: [
                r"(\w+)\s+(?:is\s+)?(?:placed\s+)?(?:near|close\s+to|next\s+to)\s+(?:the\s+)?(\w+)",
                r"put\s+(?:the\s+)?(\w+)\s+(?:near|close\s+to|next\s+to)\s+(?:the\s+)?(\w+)"
            ],
            RelationType.LEFT_OF: [
                r"(\w+)\s+(?:is\s+)?(?:placed\s+)?(?:to\s+the\s+left\s+of|left\s+of)\s+(?:the\s+)?(\w+)",
                r"put\s+(?:the\s+)?(\w+)\s+(?:to\s+the\s+left\s+of|left\s+of)\s+(?:the\s+)?(\w+)"
            ],
            RelationType.RIGHT_OF: [
                r"(\w+)\s+(?:is\s+)?(?:placed\s+)?(?:to\s+the\s+right\s+of|right\s+of)\s+(?:the\s+)?(\w+)",
                r"put\s+(?:the\s+)?(\w+)\s+(?:to\s+the\s+right\s+of|right\s+of)\s+(?:the\s+)?(\w+)"
            ],
            RelationType.IN_FRONT_OF: [
                r"(\w+)\s+(?:is\s+)?(?:placed\s+)?(?:in\s+front\s+of|before)\s+(?:the\s+)?(\w+)",
                r"put\s+(?:the\s+)?(\w+)\s+(?:in\s+front\s+of|before)\s+(?:the\s+)?(\w+)"
            ],
            RelationType.BEHIND: [
                r"(\w+)\s+(?:is\s+)?(?:placed\s+)?(?:behind|after)\s+(?:the\s+)?(\w+)",
                r"put\s+(?:the\s+)?(\w+)\s+(?:behind|after)\s+(?:the\s+)?(\w+)"
            ]
        }
        
        # Object type mappings
        self.object_synonyms = {
            'ball': ['ball', 'sphere', 'orb'],
            'box': ['box', 'cube', 'block', 'container'],
            'ramp': ['ramp', 'incline', 'slope', 'wedge'],
            'cylinder': ['cylinder', 'tube', 'pipe'],
            'plane': ['plane', 'ground', 'floor', 'surface']
        }
    
    def parse_spatial_text(self, text: str) -> List[SpatialConcept]:
        """Parse text to extract spatial concepts."""
        concepts = []
        text_lower = text.lower()
        
        # Extract spatial relationships using patterns
        for relation_type, patterns in self.spatial_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    concept = self._create_spatial_concept(relation_type, match, text)
                    if concept:
                        concepts.append(concept)
        
        # If no explicit relationships found, try to infer from context
        if not concepts:
            concepts.extend(self._infer_implicit_relationships(text))
        
        return concepts
    
    def _create_spatial_concept(self, relation_type: RelationType, match: re.Match, original_text: str) -> Optional[SpatialConcept]:
        """Create a spatial concept from a regex match."""
        groups = match.groups()
        
        if relation_type == RelationType.BETWEEN and len(groups) >= 3:
            # Special case for "between" which has 3 objects
            primary_obj = self._normalize_object_name(groups[0])
            ref_obj1 = self._normalize_object_name(groups[1])
            ref_obj2 = self._normalize_object_name(groups[2])
            
            return SpatialConcept(
                concept_type='relation',
                primary_object=primary_obj,
                reference_objects=[ref_obj1, ref_obj2],
                spatial_relation=relation_type,
                parameters={'distance_tolerance': 1.0},
                confidence=0.9,
                text_span=match.group(0)
            )
        
        elif len(groups) >= 2:
            # Standard two-object relationship
            primary_obj = self._normalize_object_name(groups[0])
            ref_obj = self._normalize_object_name(groups[1])
            
            return SpatialConcept(
                concept_type='relation',
                primary_object=primary_obj,
                reference_objects=[ref_obj],
                spatial_relation=relation_type,
                parameters=self._get_relation_parameters(relation_type),
                confidence=0.8,
                text_span=match.group(0)
            )
        
        return None
    
    def _normalize_object_name(self, obj_name: str) -> str:
        """Normalize object name to standard form."""
        obj_name = obj_name.lower().strip()
        
        # Check synonyms
        for standard_name, synonyms in self.object_synonyms.items():
            if obj_name in synonyms:
                return standard_name
        
        return obj_name
    
    def _get_relation_parameters(self, relation_type: RelationType) -> Dict[str, Any]:
        """Get default parameters for a relation type."""
        params = {
            RelationType.ABOVE: {'vertical_offset': 1.0, 'tolerance': 0.5},
            RelationType.BELOW: {'vertical_offset': -1.0, 'tolerance': 0.5},
            RelationType.NEAR: {'distance': 1.0, 'tolerance': 0.3},
            RelationType.LEFT_OF: {'horizontal_offset': -1.0, 'tolerance': 0.5},
            RelationType.RIGHT_OF: {'horizontal_offset': 1.0, 'tolerance': 0.5},
            RelationType.IN_FRONT_OF: {'depth_offset': 1.0, 'tolerance': 0.5},
            RelationType.BEHIND: {'depth_offset': -1.0, 'tolerance': 0.5},
            RelationType.BETWEEN: {'distance_tolerance': 1.0}
        }
        
        return params.get(relation_type, {})
    
    def _infer_implicit_relationships(self, text: str) -> List[SpatialConcept]:
        """Infer spatial relationships from context when not explicitly stated."""
        concepts = []
        text_lower = text.lower()
        
        # Look for implicit "on" relationships
        on_patterns = [
            r"(\w+)\s+on\s+(?:a\s+|the\s+)?(\w+)",
            r"place\s+(?:a\s+|the\s+)?(\w+)\s+on\s+(?:a\s+|the\s+)?(\w+)"
        ]
        
        for pattern in on_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    primary_obj = self._normalize_object_name(groups[0])
                    ref_obj = self._normalize_object_name(groups[1])
                    
                    concepts.append(SpatialConcept(
                        concept_type='relation',
                        primary_object=primary_obj,
                        reference_objects=[ref_obj],
                        spatial_relation=RelationType.ON_TOP_OF,
                        parameters={'vertical_offset': 0.5, 'contact': True},
                        confidence=0.7,
                        text_span=match.group(0)
                    ))
        
        return concepts


class SpatialRelationshipResolver:
    """Resolves spatial relationships into 3D coordinates."""
    
    def __init__(self):
        """Initialize spatial relationship resolver."""
        self.default_spacing = 2.0  # Default spacing between objects
        self.height_offset = 1.0    # Default height for "above" relationships
    
    def resolve_spatial_layout(self, concepts: List[SpatialConcept], 
                             existing_objects: Dict[str, DynamicPhysicsObject] = None) -> Dict[str, Vector3]:
        """Resolve spatial concepts into 3D positions."""
        if existing_objects is None:
            existing_objects = {}
        
        positions = {}
        
        # Start with existing object positions
        for obj_id, obj in existing_objects.items():
            positions[obj_id] = obj.position
        
        # Process spatial concepts
        for concept in concepts:
            new_positions = self._resolve_concept(concept, positions)
            positions.update(new_positions)
        
        return positions
    
    def _resolve_concept(self, concept: SpatialConcept, current_positions: Dict[str, Vector3]) -> Dict[str, Vector3]:
        """Resolve a single spatial concept."""
        new_positions = {}
        
        primary_obj = concept.primary_object
        reference_objects = concept.reference_objects
        relation = concept.spatial_relation
        
        # Get reference positions
        ref_positions = []
        for ref_obj in reference_objects:
            if ref_obj in current_positions:
                ref_positions.append(current_positions[ref_obj])
            else:
                # Create default position for unknown reference object
                ref_pos = Vector3(0, 0, 0)
                current_positions[ref_obj] = ref_pos
                ref_positions.append(ref_pos)
        
        if not ref_positions:
            # No reference, place at origin
            new_positions[primary_obj] = Vector3(0, 0, 1)
            return new_positions
        
        # Calculate position based on relationship
        if relation == RelationType.ABOVE:
            ref_pos = ref_positions[0]
            new_positions[primary_obj] = Vector3(
                ref_pos.x, 
                ref_pos.y, 
                ref_pos.z + self.height_offset
            )
        
        elif relation == RelationType.BELOW:
            ref_pos = ref_positions[0]
            new_positions[primary_obj] = Vector3(
                ref_pos.x, 
                ref_pos.y, 
                max(0.1, ref_pos.z - self.height_offset)
            )
        
        elif relation == RelationType.LEFT_OF:
            ref_pos = ref_positions[0]
            new_positions[primary_obj] = Vector3(
                ref_pos.x - self.default_spacing, 
                ref_pos.y, 
                ref_pos.z
            )
        
        elif relation == RelationType.RIGHT_OF:
            ref_pos = ref_positions[0]
            new_positions[primary_obj] = Vector3(
                ref_pos.x + self.default_spacing, 
                ref_pos.y, 
                ref_pos.z
            )
        
        elif relation == RelationType.IN_FRONT_OF:
            ref_pos = ref_positions[0]
            new_positions[primary_obj] = Vector3(
                ref_pos.x, 
                ref_pos.y + self.default_spacing, 
                ref_pos.z
            )
        
        elif relation == RelationType.BEHIND:
            ref_pos = ref_positions[0]
            new_positions[primary_obj] = Vector3(
                ref_pos.x, 
                ref_pos.y - self.default_spacing, 
                ref_pos.z
            )
        
        elif relation == RelationType.BETWEEN and len(ref_positions) >= 2:
            # Position between two reference objects
            pos1, pos2 = ref_positions[0], ref_positions[1]
            midpoint = Vector3(
                (pos1.x + pos2.x) / 2,
                (pos1.y + pos2.y) / 2,
                (pos1.z + pos2.z) / 2
            )
            new_positions[primary_obj] = midpoint
        
        elif relation == RelationType.NEAR:
            ref_pos = ref_positions[0]
            # Place nearby with small random offset
            offset = np.random.uniform(-0.5, 0.5, 3)
            new_positions[primary_obj] = Vector3(
                ref_pos.x + offset[0],
                ref_pos.y + offset[1],
                ref_pos.z + abs(offset[2])  # Keep above ground
            )
        
        elif relation == RelationType.ON_TOP_OF:
            ref_pos = ref_positions[0]
            new_positions[primary_obj] = Vector3(
                ref_pos.x, 
                ref_pos.y, 
                ref_pos.z + 0.5  # Small offset for contact
            )
        
        else:
            # Default positioning
            ref_pos = ref_positions[0]
            new_positions[primary_obj] = Vector3(
                ref_pos.x + 1.0, 
                ref_pos.y, 
                ref_pos.z
            )
        
        return new_positions


class RelationalSceneBuilder:
    """Builds scenes based on relational understanding."""
    
    def __init__(self):
        """Initialize relational scene builder."""
        self.parser = SpatialLanguageParser()
        self.resolver = SpatialRelationshipResolver()
        
        # Default object properties
        self.default_objects = {
            'ball': {'type': ObjectType.SPHERE, 'scale': Vector3(0.5, 0.5, 0.5), 'mass': 1.0},
            'box': {'type': ObjectType.BOX, 'scale': Vector3(0.5, 0.5, 0.5), 'mass': 2.0},
            'ramp': {'type': ObjectType.RAMP, 'scale': Vector3(2.0, 0.2, 1.0), 'mass': 0.0},
            'cylinder': {'type': ObjectType.CYLINDER, 'scale': Vector3(0.5, 0.5, 1.0), 'mass': 1.5},
            'sphere': {'type': ObjectType.SPHERE, 'scale': Vector3(0.5, 0.5, 0.5), 'mass': 1.0}
        }
    
    def build_scene_from_text(self, text: str) -> DynamicPhysicsScene:
        """Build a physics scene from natural language text."""
        scene = DynamicPhysicsScene(f"relational_scene_{int(np.random.random() * 10000)}")
        
        # Parse spatial concepts
        concepts = self.parser.parse_spatial_text(text)
        
        # Collect all mentioned objects
        all_objects = set()
        for concept in concepts:
            all_objects.add(concept.primary_object)
            all_objects.update(concept.reference_objects)
        
        # If no relationships found, extract objects from simple text
        if not concepts:
            all_objects = self._extract_objects_from_text(text)
        
        # Create object instances
        objects_dict = {}
        for obj_name in all_objects:
            if obj_name in self.default_objects:
                obj_props = self.default_objects[obj_name]
                
                obj = DynamicPhysicsObject(
                    object_id=f"{obj_name}_{len(objects_dict) + 1}",
                    object_type=obj_props['type'],
                    position=Vector3(0, 0, 1),  # Default position
                    rotation=Vector3(0, 0, 0),
                    scale=obj_props['scale'],
                    mass=obj_props['mass'],
                    material=MaterialType.WOOD  # Default material
                )
                
                objects_dict[obj_name] = obj
                scene.add_object(obj)
        
        # Resolve spatial layout
        if concepts:
            positions = self.resolver.resolve_spatial_layout(concepts, objects_dict)
            
            # Update object positions
            for obj_name, position in positions.items():
                if obj_name in objects_dict:
                    objects_dict[obj_name].position = position
            
            # Add spatial relationships to scene
            for concept in concepts:
                relation = SpatialRelation(
                    relation_type=concept.spatial_relation,
                    subject_id=objects_dict[concept.primary_object].object_id if concept.primary_object in objects_dict else concept.primary_object,
                    target_id=objects_dict[concept.reference_objects[0]].object_id if concept.reference_objects and concept.reference_objects[0] in objects_dict else "unknown",
                    confidence=concept.confidence,
                    parameters=concept.parameters
                )
                scene.add_relationship(relation)
        
        return scene
    
    def _extract_objects_from_text(self, text: str) -> Set[str]:
        """Extract object names from text when no relationships are found."""
        objects = set()
        text_lower = text.lower()
        
        # Look for object creation patterns
        creation_patterns = [
            r"create\s+(?:a\s+|an\s+|the\s+)?(\w+)",
            r"add\s+(?:a\s+|an\s+|the\s+)?(\w+)",
            r"place\s+(?:a\s+|an\s+|the\s+)?(\w+)",
            r"put\s+(?:a\s+|an\s+|the\s+)?(\w+)",
            r"make\s+(?:a\s+|an\s+|the\s+)?(\w+)"
        ]
        
        for pattern in creation_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                obj_name = self.parser._normalize_object_name(match.group(1))
                if obj_name in self.default_objects:
                    objects.add(obj_name)
        
        return objects


def test_relational_understanding():
    """Test the relational understanding system."""
    print("Testing Relational Understanding System...")
    
    # Test spatial language parsing
    parser = SpatialLanguageParser()
    
    test_texts = [
        "place a ball above the box",
        "put the sphere between the two cubes",
        "create a ball on the ramp",
        "add a box to the left of the sphere",
        "place the cylinder behind the ramp"
    ]
    
    print("Testing spatial language parsing:")
    for text in test_texts:
        concepts = parser.parse_spatial_text(text)
        print(f"  '{text}' -> {len(concepts)} concepts")
        for concept in concepts:
            print(f"    {concept.primary_object} {concept.spatial_relation.value} {concept.reference_objects}")
    
    # Test scene building
    builder = RelationalSceneBuilder()
    
    print("\nTesting scene building:")
    for text in test_texts[:3]:  # Test first 3
        scene = builder.build_scene_from_text(text)
        print(f"  '{text}' -> {scene.get_object_count()} objects, {len(scene.global_relationships)} relationships")
        
        # Show object positions
        for obj in scene.objects.values():
            print(f"    {obj.object_id}: {obj.position.to_list()}")
    
    print("✅ Relational understanding test completed!")


if __name__ == "__main__":
    test_relational_understanding()
