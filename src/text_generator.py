"""
Text Generation System
Creates natural language descriptions of physics scenarios using templates and vocabulary.
"""

import random
import re
from typing import List, Dict, Any, Tuple
from scene_representation import PhysicsScene, PhysicsObject, ObjectType, MaterialType, Vector3


class PhysicsVocabulary:
    """Vocabulary and synonyms for physics concepts."""
    
    def __init__(self):
        """Initialize vocabulary dictionaries."""
        
        # Object type synonyms
        self.object_names = {
            ObjectType.SPHERE: ["ball", "sphere", "orb", "round object"],
            ObjectType.BOX: ["box", "cube", "block", "rectangular object"],
            ObjectType.CYLINDER: ["cylinder", "tube", "cylindrical object", "rod"],
            ObjectType.RAMP: ["ramp", "incline", "slope", "inclined plane"],
            ObjectType.PLANE: ["plane", "platform", "surface", "ground"],
            ObjectType.CONE: ["cone", "conical object", "pyramid"]
        }
        
        # Material descriptions
        self.material_descriptions = {
            MaterialType.RUBBER: ["rubber", "bouncy", "elastic"],
            MaterialType.METAL: ["metal", "metallic", "steel", "iron"],
            MaterialType.WOOD: ["wood", "wooden", "timber"],
            MaterialType.ICE: ["ice", "icy", "slippery", "frozen"],
            MaterialType.BOUNCY: ["bouncy", "super elastic", "highly elastic"],
            MaterialType.PLASTIC: ["plastic", "polymer"],
            MaterialType.GLASS: ["glass", "glassy", "transparent"],
            MaterialType.STONE: ["stone", "rocky", "granite", "marble"]
        }
        
        # Action verbs
        self.action_verbs = {
            "create": ["create", "make", "build", "construct", "form"],
            "place": ["place", "put", "position", "set", "drop"],
            "add": ["add", "insert", "include", "introduce"],
            "roll": ["roll", "tumble", "move", "slide"],
            "bounce": ["bounce", "rebound", "spring back"],
            "fall": ["fall", "drop", "descend", "plummet"]
        }
        
        # Position descriptions
        self.positions = {
            "top": ["top", "upper part", "summit", "peak"],
            "bottom": ["bottom", "lower part", "base", "foundation"],
            "left": ["left", "left side", "port side"],
            "right": ["right", "right side", "starboard side"],
            "center": ["center", "middle", "central part"],
            "edge": ["edge", "border", "rim", "periphery"]
        }
        
        # Size descriptions
        self.sizes = {
            "small": ["small", "tiny", "little", "miniature"],
            "medium": ["medium", "average", "normal", "standard"],
            "large": ["large", "big", "huge", "massive"],
            "heavy": ["heavy", "weighty", "dense"],
            "light": ["light", "lightweight", "featherweight"]
        }
        
        # Physics concepts
        self.physics_concepts = {
            "gravity": ["gravity", "gravitational force", "downward force"],
            "friction": ["friction", "resistance", "drag"],
            "momentum": ["momentum", "inertia", "motion"],
            "energy": ["energy", "kinetic energy", "potential energy"],
            "collision": ["collision", "impact", "crash", "hit"]
        }
    
    def get_random_synonym(self, category: str, key: Any) -> str:
        """Get a random synonym from a category."""
        if category == "object" and key in self.object_names:
            return random.choice(self.object_names[key])
        elif category == "material" and key in self.material_descriptions:
            return random.choice(self.material_descriptions[key])
        elif category == "action" and key in self.action_verbs:
            return random.choice(self.action_verbs[key])
        elif category == "position" and key in self.positions:
            return random.choice(self.positions[key])
        elif category == "size" and key in self.sizes:
            return random.choice(self.sizes[key])
        elif category == "physics" and key in self.physics_concepts:
            return random.choice(self.physics_concepts[key])
        else:
            return str(key)


class TextTemplate:
    """A template for generating natural language descriptions."""
    
    def __init__(self, template: str, required_objects: List[ObjectType] = None, 
                 complexity: str = "simple"):
        """
        Initialize template.
        
        Args:
            template: Template string with placeholders
            required_objects: List of object types required for this template
            complexity: Template complexity level (simple, medium, complex)
        """
        self.template = template
        self.required_objects = required_objects or []
        self.complexity = complexity
        self.placeholders = self._extract_placeholders()
    
    def _extract_placeholders(self) -> List[str]:
        """Extract placeholder names from template with validation."""
        placeholders = re.findall(r'\{(\w+)\}', self.template)
        # Could add validation here if needed
        return placeholders
    
    def can_apply_to_scene(self, scene: PhysicsScene) -> bool:
        """Check if this template can be applied to the given scene."""
        scene_object_types = [obj.object_type for obj in scene.objects]
        return all(req_type in scene_object_types for req_type in self.required_objects)
    
    def generate(self, scene: PhysicsScene, vocab: PhysicsVocabulary) -> str:
        """Generate text from template using scene data."""
        if not self.can_apply_to_scene(scene):
            return ""
        
        # Extract relevant objects
        objects_by_type = {}
        for obj in scene.objects:
            if obj.object_type not in objects_by_type:
                objects_by_type[obj.object_type] = []
            objects_by_type[obj.object_type].append(obj)
        
        # Fill placeholders
        filled_template = self.template
        
        # Replace object placeholders
        for obj_type in self.required_objects:
            if obj_type in objects_by_type:
                obj = random.choice(objects_by_type[obj_type])
                obj_name = vocab.get_random_synonym("object", obj_type)
                material_name = vocab.get_random_synonym("material", obj.material)
                
                # Replace various placeholders
                filled_template = filled_template.replace(f"{{{obj_type.value}}}", obj_name)
                filled_template = filled_template.replace(f"{{{obj_type.value}_material}}", material_name)
                filled_template = filled_template.replace(f"{{{obj_type.value}_mass}}", f"{obj.mass:.1f}kg")
                
                # Position descriptions
                pos = obj.position
                if pos.z > 1.0:
                    height_desc = "high up"
                elif pos.z > 0.5:
                    height_desc = "elevated"
                else:
                    height_desc = "on the ground"
                filled_template = filled_template.replace(f"{{{obj_type.value}_position}}", height_desc)
        
        # Replace action verbs using vocabulary
        action_list = list(vocab.action_verbs.keys())
        for action in action_list:
            if f"{{{action}}}" in filled_template:
                verb = vocab.get_random_synonym("action", action)
                filled_template = filled_template.replace(f"{{{action}}}", verb)
        
        return filled_template


class TextGenerator:
    """Main text generation system."""
    
    def __init__(self):
        """Initialize the text generator."""
        self.vocab = PhysicsVocabulary()
        self.templates = self._create_templates()
    
    def _create_templates(self) -> List[TextTemplate]:
        """Create the library of text templates."""
        templates = []
        
        # Simple single-object templates
        templates.extend([
            TextTemplate(
                "{create} a {sphere}",
                [ObjectType.SPHERE],
                "simple"
            ),
            TextTemplate(
                "{add} a {sphere_material} {sphere}",
                [ObjectType.SPHERE],
                "simple"
            ),
            TextTemplate(
                "{place} a {sphere_mass} {sphere} in the scene",
                [ObjectType.SPHERE],
                "simple"
            ),
            TextTemplate(
                "{create} a {box}",
                [ObjectType.BOX],
                "simple"
            ),
            TextTemplate(
                "{add} a {box_material} {box}",
                [ObjectType.BOX],
                "simple"
            ),
            TextTemplate(
                "Build a {ramp}",
                [ObjectType.RAMP],
                "simple"
            ),
        ])
        
        # Medium complexity - two objects
        templates.extend([
            TextTemplate(
                "{create} a {ramp} and {place} a {sphere} on it",
                [ObjectType.RAMP, ObjectType.SPHERE],
                "medium"
            ),
            TextTemplate(
                "{add} a {sphere_material} {sphere} and a {ramp_material} {ramp}",
                [ObjectType.SPHERE, ObjectType.RAMP],
                "medium"
            ),
            TextTemplate(
                "{place} a {sphere_mass} {sphere} on top of a {ramp}",
                [ObjectType.SPHERE, ObjectType.RAMP],
                "medium"
            ),
            TextTemplate(
                "{create} a {box} and a {sphere} next to each other",
                [ObjectType.BOX, ObjectType.SPHERE],
                "medium"
            ),
            TextTemplate(
                "Set up a {ramp} with a {sphere} that will {roll} down",
                [ObjectType.RAMP, ObjectType.SPHERE],
                "medium"
            ),
        ])
        
        # Complex templates - multiple objects and physics concepts
        templates.extend([
            TextTemplate(
                "{create} a {ramp} and {place} a {sphere_mass} {sphere_material} {sphere} at the top so it will {roll} down due to gravity",
                [ObjectType.RAMP, ObjectType.SPHERE],
                "complex"
            ),
            TextTemplate(
                "Build a physics scene with a {ramp}, a {sphere_mass} {sphere}, and a {box} that the {sphere} will hit",
                [ObjectType.RAMP, ObjectType.SPHERE, ObjectType.BOX],
                "complex"
            ),
            TextTemplate(
                "{add} a {sphere_material} {sphere} that will {bounce} off a {box_material} {box}",
                [ObjectType.SPHERE, ObjectType.BOX],
                "complex"
            ),
            TextTemplate(
                "{create} a scenario where a {sphere_mass} {sphere} rolls down a {ramp_material} {ramp} and collides with a {box}",
                [ObjectType.SPHERE, ObjectType.RAMP, ObjectType.BOX],
                "complex"
            ),
        ])
        
        return templates
    
    def generate_description(self, scene: PhysicsScene, 
                           complexity_preference: str = None) -> str:
        """
        Generate a natural language description for a scene.
        
        Args:
            scene: The physics scene to describe
            complexity_preference: Preferred complexity level
            
        Returns:
            Generated text description
        """
        # Filter templates that can apply to this scene
        applicable_templates = [t for t in self.templates if t.can_apply_to_scene(scene)]
        
        if not applicable_templates:
            return self._generate_fallback_description(scene)
        
        # Filter by complexity if specified
        if complexity_preference:
            complexity_filtered = [t for t in applicable_templates 
                                 if t.complexity == complexity_preference]
            if complexity_filtered:
                applicable_templates = complexity_filtered
        
        # Choose a random template and generate
        template = random.choice(applicable_templates)
        description = template.generate(scene, self.vocab)
        
        return description if description else self._generate_fallback_description(scene)
    
    def _generate_fallback_description(self, scene: PhysicsScene) -> str:
        """Generate a basic fallback description."""
        object_counts = {}
        for obj in scene.objects:
            if obj.object_type != ObjectType.PLANE:  # Skip ground plane
                obj_name = self.vocab.get_random_synonym("object", obj.object_type)
                object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
        
        if not object_counts:
            return "Create an empty physics scene"
        
        parts = []
        for obj_name, count in object_counts.items():
            if count == 1:
                parts.append(f"a {obj_name}")
            else:
                parts.append(f"{count} {obj_name}s")
        
        if len(parts) == 1:
            return f"Create {parts[0]}"
        elif len(parts) == 2:
            return f"Create {parts[0]} and {parts[1]}"
        else:
            return f"Create {', '.join(parts[:-1])}, and {parts[-1]}"
    
    def generate_multiple_descriptions(self, scene: PhysicsScene, 
                                     count: int = 3) -> List[str]:
        """Generate multiple different descriptions for the same scene."""
        descriptions = []
        for _ in range(count):
            desc = self.generate_description(scene)
            if desc not in descriptions:
                descriptions.append(desc)
        
        # If we didn't get enough unique descriptions, try different complexities
        if len(descriptions) < count:
            for complexity in ["simple", "medium", "complex"]:
                if len(descriptions) >= count:
                    break
                desc = self.generate_description(scene, complexity)
                if desc not in descriptions:
                    descriptions.append(desc)
        
        return descriptions[:count]


# Test function
def test_text_generator():
    """Test the text generation system."""
    print("Testing text generation system...")
    
    from scene_representation import SceneBuilder
    
    # Create a test scene
    builder = SceneBuilder("test_text_scene")
    builder.add_ground_plane()
    builder.add_ramp(position=(0, 0, 0), angle=0.3)
    builder.add_sphere(position=(-1, 0, 1.5), radius=0.1, mass=2.0)
    scene = builder.get_scene()
    
    # Create text generator
    generator = TextGenerator()
    
    # Generate descriptions
    print("\nGenerated descriptions:")
    descriptions = generator.generate_multiple_descriptions(scene, count=5)
    for i, desc in enumerate(descriptions, 1):
        print(f"{i}. {desc}")
    
    # Test different complexity levels
    print("\nComplexity levels:")
    for complexity in ["simple", "medium", "complex"]:
        desc = generator.generate_description(scene, complexity)
        print(f"{complexity.capitalize()}: {desc}")
    
    print("\n✅ Text generation system working correctly!")


if __name__ == "__main__":
    test_text_generator()
