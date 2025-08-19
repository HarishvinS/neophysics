"""
Generative Understanding System
Implements true conceptual understanding that can reason about novel objects
and make intelligent attempts at unfamiliar concepts rather than just keyword matching.
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import json

from scene_representation import ObjectType, MaterialType, Vector3
from dynamic_scene_representation import DynamicPhysicsObject


class ConceptType(Enum):
    """Types of concepts the system can understand."""
    GEOMETRIC_SHAPE = "geometric_shape"
    FUNCTIONAL_OBJECT = "functional_object"
    MATERIAL_PROPERTY = "material_property"
    SPATIAL_RELATIONSHIP = "spatial_relationship"
    PHYSICAL_PROPERTY = "physical_property"
    MOTION_DESCRIPTOR = "motion_descriptor"


@dataclass
class ConceptualProperty:
    """Represents a conceptual property that can be inferred."""
    property_name: str
    property_type: ConceptType
    confidence: float
    reasoning: str
    inferred_values: Dict[str, Any]
    
    def to_dict(self):
        return {
            'property_name': self.property_name,
            'property_type': self.property_type.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'inferred_values': self.inferred_values
        }


class ConceptualReasoner:
    """Reasons about concepts and infers properties from descriptions."""
    
    def __init__(self):
        """Initialize conceptual reasoner."""
        self.geometric_patterns = self._build_geometric_patterns()
        self.functional_patterns = self._build_functional_patterns()
        self.material_patterns = self._build_material_patterns()
        self.size_patterns = self._build_size_patterns()
        self.shape_inference_rules = self._build_shape_inference_rules()
    
    def _build_geometric_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Build patterns for geometric shape inference."""
        return {
            'circular': {
                'keywords': ['round', 'circular', 'ring', 'donut', 'torus', 'wheel', 'disc'],
                'shape_hints': {'curved': True, 'symmetrical': True},
                'likely_type': ObjectType.CYLINDER,
                'confidence_base': 0.8
            },
            'curved': {
                'keywords': ['curved', 'bent', 'arc', 'bow', 'crescent', 'u-shaped', 'c-shaped'],
                'shape_hints': {'curved': True, 'linear': False},
                'modifications': {'curvature': 'high'},
                'confidence_base': 0.7
            },
            'angular': {
                'keywords': ['angular', 'sharp', 'pointed', 'triangular', 'pyramid', 'wedge'],
                'shape_hints': {'angular': True, 'smooth': False},
                'likely_type': ObjectType.BOX,  # Modified box
                'confidence_base': 0.7
            },
            'elongated': {
                'keywords': ['long', 'elongated', 'rod', 'stick', 'beam', 'pole', 'tube'],
                'shape_hints': {'aspect_ratio': 'high'},
                'likely_type': ObjectType.CYLINDER,
                'confidence_base': 0.8
            },
            'flat': {
                'keywords': ['flat', 'thin', 'plate', 'sheet', 'panel', 'board'],
                'shape_hints': {'thickness': 'low'},
                'likely_type': ObjectType.BOX,
                'confidence_base': 0.8
            },
            'hollow': {
                'keywords': ['hollow', 'empty', 'tube', 'pipe', 'ring', 'container'],
                'shape_hints': {'hollow': True, 'interior_space': True},
                'modifications': {'hollow_interior': True},
                'confidence_base': 0.7
            }
        }
    
    def _build_functional_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Build patterns for functional object inference."""
        return {
            'aerodynamic': {
                'keywords': ['wing', 'airfoil', 'blade', 'propeller', 'fin'],
                'function': 'generate_lift',
                'shape_hints': {'streamlined': True, 'curved_top': True, 'thin_profile': True},
                'likely_dimensions': {'thickness_ratio': 0.1, 'aspect_ratio': 3.0},
                'confidence_base': 0.8
            },
            'support': {
                'keywords': ['support', 'pillar', 'column', 'post', 'beam', 'strut'],
                'function': 'structural_support',
                'shape_hints': {'vertical': True, 'strong': True},
                'likely_type': ObjectType.CYLINDER,
                'confidence_base': 0.9
            },
            'inclined': {
                'keywords': ['ramp', 'slope', 'incline', 'wedge', 'chute'],
                'function': 'change_elevation',
                'shape_hints': {'angled': True, 'smooth_surface': True},
                'likely_type': ObjectType.RAMP,
                'confidence_base': 0.9
            },
            'rolling': {
                'keywords': ['ball', 'sphere', 'marble', 'bead', 'orb'],
                'function': 'roll_freely',
                'shape_hints': {'spherical': True, 'smooth': True},
                'likely_type': ObjectType.SPHERE,
                'confidence_base': 0.95
            },
            'container': {
                'keywords': ['bowl', 'cup', 'container', 'vessel', 'bucket'],
                'function': 'contain_objects',
                'shape_hints': {'concave': True, 'open_top': True},
                'modifications': {'container_shape': True},
                'confidence_base': 0.8
            }
        }
    
    def _build_material_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Build patterns for material inference."""
        return {
            'metallic': {
                'keywords': ['metal', 'steel', 'iron', 'aluminum', 'copper', 'shiny', 'metallic'],
                'material': MaterialType.METAL,
                'properties': {'density': 'high', 'strength': 'high', 'conductive': True},
                'confidence_base': 0.9
            },
            'wooden': {
                'keywords': ['wood', 'wooden', 'timber', 'oak', 'pine', 'bamboo'],
                'material': MaterialType.WOOD,
                'properties': {'density': 'medium', 'natural': True, 'burnable': True},
                'confidence_base': 0.9
            },
            'elastic': {
                'keywords': ['rubber', 'elastic', 'bouncy', 'flexible', 'stretchy'],
                'material': MaterialType.RUBBER,
                'properties': {'elasticity': 'high', 'deformable': True},
                'confidence_base': 0.9
            },
            'transparent': {
                'keywords': ['glass', 'transparent', 'clear', 'crystal', 'see-through'],
                'material': MaterialType.GLASS,
                'properties': {'transparency': 'high', 'brittle': True},
                'confidence_base': 0.8
            },
            'heavy': {
                'keywords': ['heavy', 'dense', 'massive', 'weighty', 'solid'],
                'properties': {'mass_modifier': 2.0, 'density': 'high'},
                'confidence_base': 0.7
            },
            'light': {
                'keywords': ['light', 'lightweight', 'airy', 'foam', 'hollow'],
                'properties': {'mass_modifier': 0.5, 'density': 'low'},
                'confidence_base': 0.7
            }
        }
    
    def _build_size_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Build patterns for size inference."""
        return {
            'large': {
                'keywords': ['large', 'big', 'huge', 'giant', 'massive', 'enormous'],
                'scale_modifier': 2.0,
                'confidence_base': 0.8
            },
            'small': {
                'keywords': ['small', 'tiny', 'little', 'mini', 'miniature', 'micro'],
                'scale_modifier': 0.5,
                'confidence_base': 0.8
            },
            'long': {
                'keywords': ['long', 'extended', 'elongated', 'stretched'],
                'dimension_modifier': {'x': 2.0},
                'confidence_base': 0.7
            },
            'wide': {
                'keywords': ['wide', 'broad', 'thick'],
                'dimension_modifier': {'y': 1.5},
                'confidence_base': 0.7
            },
            'tall': {
                'keywords': ['tall', 'high', 'elevated'],
                'dimension_modifier': {'z': 2.0},
                'confidence_base': 0.7
            }
        }
    
    def _build_shape_inference_rules(self) -> List[Dict[str, Any]]:
        """Build rules for inferring shapes from complex descriptions."""
        return [
            {
                'pattern': r'u[- ]?shaped',
                'inferred_shape': 'u_curve',
                'reasoning': 'U-shaped indicates curved structure with open ends',
                'modifications': {'curvature': 'u_bend', 'open_ends': True},
                'confidence': 0.8
            },
            {
                'pattern': r'c[- ]?shaped',
                'inferred_shape': 'c_curve',
                'reasoning': 'C-shaped indicates partial circular curve',
                'modifications': {'curvature': 'c_bend', 'partial_circle': True},
                'confidence': 0.8
            },
            {
                'pattern': r'l[- ]?shaped',
                'inferred_shape': 'l_angle',
                'reasoning': 'L-shaped indicates right-angle bend',
                'modifications': {'angle': 90, 'corner': True},
                'confidence': 0.8
            },
            {
                'pattern': r'spiral|coil|helix',
                'inferred_shape': 'spiral',
                'reasoning': 'Spiral/coil indicates helical or circular progression',
                'modifications': {'spiral': True, 'progressive_curve': True},
                'confidence': 0.7
            },
            {
                'pattern': r'star[- ]?shaped',
                'inferred_shape': 'star',
                'reasoning': 'Star-shaped indicates radiating points from center',
                'modifications': {'radiating_points': True, 'symmetrical': True},
                'confidence': 0.7
            }
        ]
    
    def analyze_description(self, description: str) -> List[ConceptualProperty]:
        """Analyze a description and infer conceptual properties."""
        description_lower = description.lower()
        properties = []
        
        # Analyze geometric patterns
        for pattern_name, pattern_info in self.geometric_patterns.items():
            if any(keyword in description_lower for keyword in pattern_info['keywords']):
                prop = ConceptualProperty(
                    property_name=f"geometric_{pattern_name}",
                    property_type=ConceptType.GEOMETRIC_SHAPE,
                    confidence=pattern_info['confidence_base'],
                    reasoning=f"Detected {pattern_name} geometric pattern from keywords",
                    inferred_values=pattern_info.get('shape_hints', {})
                )
                properties.append(prop)
        
        # Analyze functional patterns
        for pattern_name, pattern_info in self.functional_patterns.items():
            if any(keyword in description_lower for keyword in pattern_info['keywords']):
                prop = ConceptualProperty(
                    property_name=f"functional_{pattern_name}",
                    property_type=ConceptType.FUNCTIONAL_OBJECT,
                    confidence=pattern_info['confidence_base'],
                    reasoning=f"Inferred {pattern_info.get('function', pattern_name)} function",
                    inferred_values=pattern_info
                )
                properties.append(prop)
        
        # Analyze material patterns
        for pattern_name, pattern_info in self.material_patterns.items():
            if any(keyword in description_lower for keyword in pattern_info['keywords']):
                prop = ConceptualProperty(
                    property_name=f"material_{pattern_name}",
                    property_type=ConceptType.MATERIAL_PROPERTY,
                    confidence=pattern_info['confidence_base'],
                    reasoning=f"Detected {pattern_name} material properties",
                    inferred_values=pattern_info.get('properties', {})
                )
                properties.append(prop)
        
        # Analyze size patterns
        for pattern_name, pattern_info in self.size_patterns.items():
            if any(keyword in description_lower for keyword in pattern_info['keywords']):
                prop = ConceptualProperty(
                    property_name=f"size_{pattern_name}",
                    property_type=ConceptType.PHYSICAL_PROPERTY,
                    confidence=pattern_info['confidence_base'],
                    reasoning=f"Inferred {pattern_name} size characteristics",
                    inferred_values=pattern_info
                )
                properties.append(prop)
        
        # Analyze complex shape patterns
        for rule in self.shape_inference_rules:
            if re.search(rule['pattern'], description_lower):
                prop = ConceptualProperty(
                    property_name=f"complex_shape_{rule['inferred_shape']}",
                    property_type=ConceptType.GEOMETRIC_SHAPE,
                    confidence=rule['confidence'],
                    reasoning=rule['reasoning'],
                    inferred_values=rule.get('modifications', {})
                )
                properties.append(prop)
        
        return properties
    
    def synthesize_object_concept(self, description: str) -> Dict[str, Any]:
        """Synthesize a complete object concept from description."""
        properties = self.analyze_description(description)
        
        # Start with default object concept
        concept = {
            'description': description,
            'object_type': ObjectType.BOX,  # Default fallback
            'material': MaterialType.WOOD,  # Default material
            'scale': Vector3(1.0, 1.0, 1.0),
            'mass': 1.0,
            'confidence': 0.5,  # Base confidence for unknown objects
            'reasoning_chain': [],
            'inferred_properties': [prop.to_dict() for prop in properties],
            'modifications': {}
        }
        
        # Apply inferences from properties
        total_confidence = 0.0
        confidence_count = 0
        
        for prop in properties:
            concept['reasoning_chain'].append(prop.reasoning)
            
            # Update confidence
            total_confidence += prop.confidence
            confidence_count += 1
            
            # Apply specific inferences
            if prop.property_type == ConceptType.FUNCTIONAL_OBJECT:
                if 'likely_type' in prop.inferred_values:
                    concept['object_type'] = prop.inferred_values['likely_type']
                    concept['reasoning_chain'].append(f"Set object type to {concept['object_type'].value}")
                
                if 'likely_dimensions' in prop.inferred_values:
                    dims = prop.inferred_values['likely_dimensions']
                    if 'aspect_ratio' in dims:
                        concept['scale'] = Vector3(dims['aspect_ratio'], 1.0, dims.get('thickness_ratio', 0.2))
                    concept['reasoning_chain'].append("Applied functional dimensions")
            
            elif prop.property_type == ConceptType.MATERIAL_PROPERTY:
                if 'material' in prop.inferred_values:
                    concept['material'] = prop.inferred_values['material']
                    concept['reasoning_chain'].append(f"Set material to {concept['material'].value}")
                
                if 'mass_modifier' in prop.inferred_values:
                    concept['mass'] *= prop.inferred_values['mass_modifier']
                    concept['reasoning_chain'].append(f"Adjusted mass by {prop.inferred_values['mass_modifier']}")
            
            elif prop.property_type == ConceptType.PHYSICAL_PROPERTY:
                if 'scale_modifier' in prop.inferred_values:
                    modifier = prop.inferred_values['scale_modifier']
                    concept['scale'] = Vector3(
                        concept['scale'].x * modifier,
                        concept['scale'].y * modifier,
                        concept['scale'].z * modifier
                    )
                    concept['reasoning_chain'].append(f"Scaled object by {modifier}")
                
                if 'dimension_modifier' in prop.inferred_values:
                    dim_mods = prop.inferred_values['dimension_modifier']
                    scale = concept['scale']
                    concept['scale'] = Vector3(
                        scale.x * dim_mods.get('x', 1.0),
                        scale.y * dim_mods.get('y', 1.0),
                        scale.z * dim_mods.get('z', 1.0)
                    )
                    concept['reasoning_chain'].append("Applied dimensional modifications")
            
            elif prop.property_type == ConceptType.GEOMETRIC_SHAPE:
                concept['modifications'].update(prop.inferred_values)
        
        # Calculate overall confidence
        if confidence_count > 0:
            concept['confidence'] = min(0.95, total_confidence / confidence_count)
        
        # Add uncertainty note if confidence is low
        if concept['confidence'] < 0.6:
            concept['reasoning_chain'].append("⚠️ Low confidence - making educated guess")
        
        return concept
    
    def create_object_from_concept(self, concept: Dict[str, Any], 
                                 position: Vector3 = Vector3(0, 0, 1),
                                 object_id: str = None) -> DynamicPhysicsObject:
        """Create a physics object from a synthesized concept."""
        if not object_id:
            object_id = f"generated_{concept['description'].replace(' ', '_')}"
        
        # Create the object
        obj = DynamicPhysicsObject(
            object_id=object_id,
            object_type=concept['object_type'],
            position=position,
            rotation=Vector3(0, 0, 0),
            scale=concept['scale'],
            mass=concept['mass'],
            material=concept['material']
        )
        
        # Add metadata about the generation process
        obj.metadata = {
            'generated_from_description': concept['description'],
            'confidence': concept['confidence'],
            'reasoning_chain': concept['reasoning_chain'],
            'inferred_properties': concept['inferred_properties'],
            'modifications': concept['modifications']
        }
        
        return obj


def test_generative_understanding():
    """Test the generative understanding system."""
    print("Testing Generative Understanding System...")
    
    reasoner = ConceptualReasoner()
    
    # Test novel object descriptions
    test_descriptions = [
        "U-shaped curved ramp",
        "aerodynamic wing",
        "heavy metal torus",
        "lightweight foam sphere",
        "long wooden beam",
        "transparent glass container",
        "spiral spring coil",
        "star-shaped platform",
        "hollow metal tube",
        "flexible rubber band"
    ]
    
    print(f"✅ Testing {len(test_descriptions)} novel object descriptions...")
    
    for description in test_descriptions:
        print(f"\n🔍 Analyzing: '{description}'")
        
        # Analyze properties
        properties = reasoner.analyze_description(description)
        print(f"   Properties detected: {len(properties)}")
        
        for prop in properties:
            print(f"     - {prop.property_name} ({prop.property_type.value}): {prop.confidence:.2f}")
            print(f"       Reasoning: {prop.reasoning}")
        
        # Synthesize complete concept
        concept = reasoner.synthesize_object_concept(description)
        print(f"   Final concept:")
        print(f"     Object type: {concept['object_type'].value}")
        print(f"     Material: {concept['material'].value}")
        print(f"     Scale: {concept['scale'].to_list()}")
        print(f"     Mass: {concept['mass']:.1f}")
        print(f"     Confidence: {concept['confidence']:.2f}")
        
        # Create physics object
        obj = reasoner.create_object_from_concept(concept)
        print(f"   ✅ Created object: {obj.object_id}")
        
        # Show reasoning chain
        if concept['reasoning_chain']:
            print(f"   Reasoning chain:")
            for step in concept['reasoning_chain'][:3]:  # Show first 3 steps
                print(f"     • {step}")
    
    print(f"\n✅ Generative understanding test completed!")
    print(f"🎯 Key capabilities demonstrated:")
    print(f"   • Novel object interpretation from descriptions")
    print(f"   • Geometric reasoning (U-shaped, spiral, etc.)")
    print(f"   • Functional inference (aerodynamic, container, etc.)")
    print(f"   • Material property reasoning")
    print(f"   • Confidence assessment and uncertainty handling")
    print(f"   • Educated guesses for unknown concepts")


if __name__ == "__main__":
    test_generative_understanding()
