"""
Data Generation System
Creates diverse, realistic physics scenarios for training ML models.
"""

import random
import time
import uuid
from typing import List, Dict, Any, Tuple
import numpy as np

from scene_representation import (
    PhysicsScene, PhysicsObject, ObjectType, MaterialType, Vector3,
    SceneEnvironment, SceneBuilder, TrainingExample
)
from text_generator import TextGenerator


class ScenarioGenerator:
    """Generates different types of physics scenarios."""
    
    # Class-level constants to avoid recreation
    SCENARIO_TYPES = {
        "simple_drop": 0.2,      # Just objects falling
        "ramp_rolling": 0.3,     # Objects rolling down ramps
        "collision": 0.2,        # Objects colliding
        "bouncing": 0.15,        # Bouncy objects
        "multi_object": 0.15     # Complex multi-object scenes
    }
    
    MATERIAL_WEIGHTS = {
        MaterialType.RUBBER: 0.25,
        MaterialType.METAL: 0.15,
        MaterialType.WOOD: 0.20,
        MaterialType.PLASTIC: 0.15,
        MaterialType.STONE: 0.10,
        MaterialType.ICE: 0.05,
        MaterialType.BOUNCY: 0.05,
        MaterialType.GLASS: 0.05
    }
    
    OBJECT_WEIGHTS = {
        ObjectType.SPHERE: 0.4,
        ObjectType.BOX: 0.3,
        ObjectType.CYLINDER: 0.15,
        ObjectType.CONE: 0.15
    }
    
    def __init__(self):
        """Initialize the scenario generator."""
        self.text_generator = TextGenerator()
    
    def generate_scenario(self, scenario_type: str = None) -> PhysicsScene:
        """Generate a physics scenario of the specified type."""
        if scenario_type is None:
            scenario_type = self._choose_random_scenario_type()
        
        scene_id = f"scene_{uuid.uuid4().hex[:8]}"
        
        if scenario_type == "simple_drop":
            return self._generate_simple_drop(scene_id)
        elif scenario_type == "ramp_rolling":
            return self._generate_ramp_rolling(scene_id)
        elif scenario_type == "collision":
            return self._generate_collision(scene_id)
        elif scenario_type == "bouncing":
            return self._generate_bouncing(scene_id)
        elif scenario_type == "multi_object":
            return self._generate_multi_object(scene_id)
        else:
            return self._generate_simple_drop(scene_id)
    
    def _choose_random_scenario_type(self) -> str:
        """Choose a random scenario type based on probabilities."""
        return random.choices(
            list(self.SCENARIO_TYPES.keys()),
            weights=list(self.SCENARIO_TYPES.values())
        )[0]
    
    def _choose_random_material(self) -> MaterialType:
        """Choose a random material based on weights."""
        return random.choices(
            list(self.MATERIAL_WEIGHTS.keys()),
            weights=list(self.MATERIAL_WEIGHTS.values())
        )[0]
    
    def _choose_random_object_type(self) -> ObjectType:
        """Choose a random object type based on weights."""
        return random.choices(
            list(self.OBJECT_WEIGHTS.keys()),
            weights=list(self.OBJECT_WEIGHTS.values())
        )[0]
    
    def _create_random_object(self, obj_id_prefix: str, index: int,
                              pos_range: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
                              mass_range: Tuple[float, float] = (0.5, 5.0),
                              initial_velocity: Vector3 = None) -> PhysicsObject:
        """Helper to create a single random object."""
        obj_type = self._choose_random_object_type()
        material = self._choose_random_material()

        x = random.uniform(*pos_range[0])
        y = random.uniform(*pos_range[1])
        z = random.uniform(*pos_range[2])

        mass = random.uniform(*mass_range)

        if obj_type == ObjectType.SPHERE:
            radius = random.uniform(0.05, 0.2)
            scale = Vector3(radius, radius, radius)
        else:
            size = random.uniform(0.1, 0.3)
            scale = Vector3(size, size, size)

        return PhysicsObject(
            object_id=f"{obj_id_prefix}_{index}",
            object_type=obj_type,
            position=Vector3(x, y, z),
            rotation=Vector3.zero(),
            scale=scale,
            mass=mass,
            material=material,
            initial_velocity=initial_velocity,
            color=self._random_color()
        )

    def _generate_simple_drop(self, scene_id: str) -> PhysicsScene:
        """Generate a simple dropping scenario."""
        builder = SceneBuilder(scene_id)
        builder.add_ground_plane()

        num_objects = random.randint(1, 3)
        pos_range = ((-2, 2), (-2, 2), (1, 4))

        for i in range(num_objects):
            obj = self._create_random_object("obj", i, pos_range)
            builder.scene.add_object(obj)

        scene = builder.get_scene()
        scene.tags = ["drop", "gravity", "falling"]
        return scene

    def _generate_ramp_rolling(self, scene_id: str) -> PhysicsScene:
        """Generate a ramp rolling scenario."""
        builder = SceneBuilder(scene_id)
        builder.add_ground_plane()
        
        # Add ramp
        ramp_angle = random.uniform(0.2, 0.6)  # 11-34 degrees
        ramp_length = random.uniform(2, 4)
        ramp_width = random.uniform(0.8, 1.5)
        
        ramp = builder.add_ramp(
            position=(0, 0, 0),
            angle=ramp_angle,
            size=(ramp_length, 0.2, ramp_width)
        )
        
        # Add 1-2 objects on the ramp
        num_objects = random.randint(1, 2)
        
        for i in range(num_objects):
            # Position on the high end of the ramp
            x = random.uniform(-ramp_length/2 + 0.2, -ramp_length/4)
            y = random.uniform(-ramp_width/3, ramp_width/3)
            z = random.uniform(1.5, 2.5)
            
            obj = self._create_random_object("rolling_obj", i, ((x, x), (y, y), (z, z)), (0.5, 3.0))
            # Prefer spheres for rolling
            if random.random() < 0.7:
                obj.object_type = ObjectType.SPHERE
            builder.scene.add_object(obj)
        
        scene = builder.get_scene()
        scene.tags = ["ramp", "rolling", "incline", "gravity"]
        return scene
    
    def _generate_collision(self, scene_id: str) -> PhysicsScene:
        """Generate a collision scenario."""
        builder = SceneBuilder(scene_id)
        builder.add_ground_plane()
        
        # Add target object (stationary)
        target_x = random.uniform(-1, 1)
        target_y = random.uniform(-1, 1)
        target_pos_range = ((target_x, target_x), (target_y, target_y), (0.5, 0.5))
        target_obj = self._create_random_object("target", 0, target_pos_range, (1.0, 3.0))
        builder.scene.add_object(target_obj)
        
        # Add moving object (will collide)
        # Position it higher and to the side
        moving_x = target_x + random.uniform(-3, -2)  # To the left
        moving_y = target_y + random.uniform(-0.5, 0.5)
        moving_pos_range = ((moving_x, moving_x), (moving_y, moving_y), (2.0, 2.0))
        initial_velocity = Vector3(random.uniform(2, 5), 0, 0)
        moving_obj = self._create_random_object("projectile", 0, moving_pos_range, (0.5, 2.0), initial_velocity)
        moving_obj.object_type = ObjectType.SPHERE # Spheres are better projectiles
        builder.scene.add_object(moving_obj)
        
        scene = builder.get_scene()
        scene.tags = ["collision", "impact", "momentum"]
        return scene

    def _generate_bouncing(self, scene_id: str) -> PhysicsScene:
        """Generate a bouncing scenario."""
        builder = SceneBuilder(scene_id)
        builder.add_ground_plane()
        
        # Add bouncy objects
        num_objects = random.randint(1, 3)
        
        for i in range(num_objects):
            pos_range = ((-2, 2), (-2, 2), (2, 5)) # Higher drop
            obj = self._create_random_object("bouncy_obj", i, pos_range, (0.3, 1.5))
            obj.object_type = ObjectType.SPHERE # Spheres bounce well
            # Prefer bouncy materials
            if random.random() < 0.4:
                obj.material = MaterialType.BOUNCY
            builder.scene.add_object(obj)
        
        scene = builder.get_scene()
        scene.tags = ["bouncing", "elastic", "restitution"]
        return scene
    
    def _generate_multi_object(self, scene_id: str) -> PhysicsScene:
        """Generate a complex multi-object scenario."""
        builder = SceneBuilder(scene_id)
        builder.add_ground_plane()
        
        # Add a ramp
        ramp_angle = random.uniform(0.3, 0.5)
        builder.add_ramp(position=(0, 0, 0), angle=ramp_angle)
        
        # Add multiple objects with different properties
        num_objects = random.randint(3, 6)
        
        for i in range(num_objects):
            # Spread objects around the scene
            if i < 2:  # First two on the ramp
                x = random.uniform(-1.5, -0.5)
                y = random.uniform(-0.5, 0.5)
                z = random.uniform(1.5, 2.5)
                pos_range = ((x, x), (y, y), (z, z))
            else:  # Others scattered around
                pos_range = ((-3, 3), (-3, 3), (0.5, 3))
            
            obj = self._create_random_object("multi_obj", i, pos_range, (0.5, 4.0))
            builder.scene.add_object(obj)
        
        scene = builder.get_scene()
        scene.tags = ["complex", "multi-object", "interaction"]
        return scene
    
    def _random_color(self) -> Tuple[float, float, float]:
        """Generate a random color."""
        return (random.random(), random.random(), random.random())


class DataGenerator:
    """Main data generation system."""
    
    def __init__(self):
        """Initialize the data generator."""
        self.scenario_generator = ScenarioGenerator()
        self.text_generator = TextGenerator()
    
    def generate_training_example(self) -> TrainingExample:
        """Generate a single training example."""
        # Generate a physics scene
        scene = self.scenario_generator.generate_scenario()
        
        # Generate action sequence from the scene
        action_sequence = self._scene_to_action_sequence(scene)
        
        # Generate text description
        text_description = self.text_generator.generate_description(scene)
        
        # Create training example
        example = TrainingExample(
            example_id=f"example_{uuid.uuid4().hex[:8]}",
            text_description=text_description,
            scene=scene,
            action_sequence=action_sequence,
            metadata={
                "generated_at": time.time(),
                "scenario_type": scene.tags[0] if scene.tags else "unknown",
                "num_objects": sum(1 for obj in scene.objects if obj.object_type != ObjectType.PLANE),
                "complexity": "simple" if len(scene.objects) <= 3 else "complex"
            }
        )
        
        return example
    
    def generate_dataset(self, num_examples: int, 
                        save_path: str = None) -> List[TrainingExample]:
        """
        Generate a complete dataset.
        
        Args:
            num_examples: Number of examples to generate
            save_path: Optional path to save the dataset
            
        Returns:
            List of training examples
        """
        print(f"Generating {num_examples} training examples...")
        
        dataset = []
        
        for i in range(num_examples):
            # Less frequent progress reporting for large datasets
            progress_interval = max(100, num_examples // 20)
            if i % progress_interval == 0 and i > 0:
                print(f"Progress: {i}/{num_examples} ({i/num_examples*100:.1f}%)")
            
            try:
                example = self.generate_training_example()
                dataset.append(example)
            except Exception as e:
                print(f"Error generating example {i}: {e}")
                continue
        
        print(f"Generated {len(dataset)} examples successfully")
        
        # Save if path provided
        if save_path:
            self._save_dataset(dataset, save_path)
        
        return dataset
    
    def _save_dataset(self, dataset: List[TrainingExample], save_path: str):
        """Save dataset to file."""
        import json
        
        dataset_dict = {
            "metadata": {
                "num_examples": len(dataset),
                "generated_at": time.time(),
                "generator_version": "1.0"
            },
            "examples": [example.to_dict() for example in dataset]
        }
        
        with open(save_path, 'w') as f:
            json.dump(dataset_dict, f, indent=2)
        
        print(f"Dataset saved to {save_path}")

    def _infer_relationships(self, objects: List[PhysicsObject]) -> List[Dict[str, str]]:
        """Infers spatial relationships between objects."""
        relationships = []
        if len(objects) < 2:
            return relationships

        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                # Check for ON_TOP_OF relationship
                # A is on top of B if A's center is above B's top surface
                # and their horizontal projections overlap.
                z1_bottom = obj1.position.z - obj1.scale.z / 2
                z2_top = obj2.position.z + obj2.scale.z / 2
                
                # Check for horizontal overlap
                x_overlap = abs(obj1.position.x - obj2.position.x) < (obj1.scale.x + obj2.scale.x) / 2
                y_overlap = abs(obj1.position.y - obj2.position.y) < (obj1.scale.y + obj2.scale.y) / 2

                # Check for vertical proximity (obj1 is slightly above obj2)
                vertical_proximity = 0 < (z1_bottom - z2_top) < 0.2 

                if x_overlap and y_overlap and vertical_proximity:
                    relationships.append({
                        'subject': obj1.object_id,
                        'type': 'ON_TOP_OF',
                        'target': obj2.object_id
                    })
        return relationships

    def _scene_to_action_sequence(self, scene: PhysicsScene) -> str:
        """Converts a PhysicsScene object to a structured action sequence string."""
        actions = []
        
        sorted_objects = sorted(
            [obj for obj in scene.objects if obj.object_type != ObjectType.PLANE],
            key=lambda o: (o.object_type.value, o.position.x, o.position.y)
        )

        for obj in sorted_objects:
            props = [
                f"type={obj.object_type.value}",
                f"id={obj.object_id}",
                f"pos=({obj.position.x:.2f},{obj.position.y:.2f},{obj.position.z:.2f})",
                f"rot=({obj.rotation.x:.2f},{obj.rotation.y:.2f},{obj.rotation.z:.2f})",
                f"scale=({obj.scale.x:.2f},{obj.scale.y:.2f},{obj.scale.z:.2f})",
                f"mass={obj.mass:.2f}",
                f"material={obj.material.value}",
                f"color=({obj.color[0]:.2f},{obj.color[1]:.2f},{obj.color[2]:.2f})"
            ]
            if obj.initial_velocity and (obj.initial_velocity.x != 0 or obj.initial_velocity.y != 0 or obj.initial_velocity.z != 0):
                props.append(f"vel=({obj.initial_velocity.x:.2f},{obj.initial_velocity.y:.2f},{obj.initial_velocity.z:.2f})")
            
            actions.append(f"CREATE {' '.join(props)};")

        # Infer relationships from the final scene state
        relationships = self._infer_relationships(sorted_objects)
        for rel in relationships:
            actions.append(f"RELATE subject_id={rel['subject']} type={rel['type']} target_id={rel['target']};")
        
        return " ".join(actions)


# Test function
def test_data_generator():
    """Test the data generation system."""
    print("Testing data generation system...")
    
    generator = DataGenerator()
    
    # Generate a few examples
    print("\nGenerating sample examples:")
    for i in range(3):
        example = generator.generate_training_example()
        print(f"\nExample {i+1}:")
        print(f"ID: {example.example_id}")
        print(f"Text: {example.text_description}")
        print(f"Scene: {len(example.scene.objects)} objects, tags: {example.scene.tags}")
        print(f"Metadata: {example.metadata}")
    
    print("\n✅ Data generation system working correctly!")


if __name__ == "__main__":
    test_data_generator()
