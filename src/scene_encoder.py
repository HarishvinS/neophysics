"""
Scene Encoder/Decoder System
Converts between PhysicsScene objects and neural network tensor representations.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple
import json

from scene_representation import PhysicsScene, PhysicsObject, ObjectType, MaterialType, Vector3, SceneEnvironment


class SceneEncoder:
    """Encodes PhysicsScene objects into tensor representations for training."""
    
    def __init__(self, max_objects: int = 8):
        """
        Initialize scene encoder.
        
        Args:
            max_objects: Maximum number of objects to encode
        """
        self.max_objects = max_objects
        
        # Create mappings for categorical variables
        self.object_type_to_idx = {obj_type: i for i, obj_type in enumerate(ObjectType)}
        self.idx_to_object_type = {i: obj_type for obj_type, i in self.object_type_to_idx.items()}
        
        self.material_to_idx = {material: i for i, material in enumerate(MaterialType)}
        self.idx_to_material = {i: material for material, i in self.material_to_idx.items()}
    
    def encode_scene(self, scene: PhysicsScene) -> Dict[str, torch.Tensor]:
        """
        Encode a physics scene into tensor format.
        
        Args:
            scene: PhysicsScene object to encode
            
        Returns:
            Dictionary of tensors representing the scene
        """
        # Filter out ground plane for training
        objects = [obj for obj in scene.objects if obj.object_type != ObjectType.PLANE]
        
        # Object count (as one-hot)
        object_count = min(len(objects), self.max_objects)
        object_count_tensor = torch.zeros(self.max_objects + 1)
        object_count_tensor[object_count] = 1.0
        
        # Initialize object tensors
        object_types = torch.zeros(self.max_objects, len(ObjectType))
        object_positions = torch.zeros(self.max_objects, 3)
        object_rotations = torch.zeros(self.max_objects, 3)
        object_scales = torch.zeros(self.max_objects, 3)
        object_masses = torch.zeros(self.max_objects, 1)
        object_materials = torch.zeros(self.max_objects, len(MaterialType))
        
        # Fill in object data
        for i, obj in enumerate(objects[:self.max_objects]):
            # Object type (one-hot)
            type_idx = self.object_type_to_idx[obj.object_type]
            object_types[i, type_idx] = 1.0
            
            # Position
            object_positions[i] = torch.tensor([obj.position.x, obj.position.y, obj.position.z])
            
            # Rotation
            object_rotations[i] = torch.tensor([obj.rotation.x, obj.rotation.y, obj.rotation.z])
            
            # Scale
            object_scales[i] = torch.tensor([obj.scale.x, obj.scale.y, obj.scale.z])
            
            # Mass
            object_masses[i] = torch.tensor([obj.mass])
            
            # Material (one-hot)
            material_idx = self.material_to_idx[obj.material]
            object_materials[i, material_idx] = 1.0
        
        # Environment
        gravity = torch.tensor([
            scene.environment.gravity.x,
            scene.environment.gravity.y,
            scene.environment.gravity.z
        ])
        
        return {
            'object_count': object_count_tensor,
            'object_types': object_types,
            'object_positions': object_positions,
            'object_rotations': object_rotations,
            'object_scales': object_scales,
            'object_masses': object_masses,
            'object_materials': object_materials,
            'gravity': gravity
        }
    
    def encode_batch(self, scenes: List[PhysicsScene]) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of scenes.
        
        Args:
            scenes: List of PhysicsScene objects
            
        Returns:
            Dictionary of batched tensors
        """
        batch_size = len(scenes)
        
        # Initialize batch tensors
        batch_data = {
            'object_count': torch.zeros(batch_size, self.max_objects + 1),
            'object_types': torch.zeros(batch_size, self.max_objects, len(ObjectType)),
            'object_positions': torch.zeros(batch_size, self.max_objects, 3),
            'object_rotations': torch.zeros(batch_size, self.max_objects, 3),
            'object_scales': torch.zeros(batch_size, self.max_objects, 3),
            'object_masses': torch.zeros(batch_size, self.max_objects, 1),
            'object_materials': torch.zeros(batch_size, self.max_objects, len(MaterialType)),
            'gravity': torch.zeros(batch_size, 3)
        }
        
        # Encode each scene
        for i, scene in enumerate(scenes):
            encoded = self.encode_scene(scene)
            for key, tensor in encoded.items():
                batch_data[key][i] = tensor
        
        return batch_data


class SceneDecoder:
    """Decodes tensor representations back into PhysicsScene objects."""
    
    def __init__(self, max_objects: int = 8):
        """
        Initialize scene decoder.
        
        Args:
            max_objects: Maximum number of objects to decode
        """
        self.max_objects = max_objects
        
        # Create mappings for categorical variables
        self.idx_to_object_type = {i: obj_type for i, obj_type in enumerate(ObjectType)}
        self.idx_to_material = {i: material for i, material in enumerate(MaterialType)}
    
    def decode_scene(self, predictions: Dict[str, torch.Tensor], 
                    scene_id: str = "decoded_scene",
                    threshold: float = 0.5) -> PhysicsScene:
        """
        Decode tensor predictions into a PhysicsScene.
        
        Args:
            predictions: Dictionary of predicted tensors
            scene_id: ID for the created scene
            threshold: Threshold for object existence
            
        Returns:
            Decoded PhysicsScene object
        """
        # Determine number of objects
        object_count_probs = predictions['object_count_probs']
        if object_count_probs.dim() > 1:
            object_count_probs = object_count_probs[0]  # Take first in batch
        predicted_count = torch.argmax(object_count_probs).item()
        
        # Create environment
        gravity_tensor = predictions['gravity']
        if gravity_tensor.dim() > 1:
            gravity_tensor = gravity_tensor[0]  # Take first in batch
        
        environment = SceneEnvironment(
            gravity=Vector3(
                gravity_tensor[0].item(),
                gravity_tensor[1].item(),
                gravity_tensor[2].item()
            )
        )
        
        # Create scene
        scene = PhysicsScene(
            scene_id=scene_id,
            objects=[],
            environment=environment
        )
        
        # Add ground plane
        ground = PhysicsObject(
            object_id="ground",
            object_type=ObjectType.PLANE,
            position=Vector3(0, 0, 0),
            rotation=Vector3(0, 0, 0),
            scale=Vector3(10, 10, 1),
            mass=0,
            material=MaterialType.STONE,
            material_properties=None,
            color=(0.5, 0.5, 0.5)
        )
        scene.add_object(ground)
        
        # Decode objects
        for i in range(min(predicted_count, self.max_objects)):
            # Get object type
            type_probs = predictions['object_type_probs']
            if type_probs.dim() > 2:
                type_probs = type_probs[0]  # Take first in batch
            
            type_idx = torch.argmax(type_probs[i]).item()
            object_type = self.idx_to_object_type[type_idx]
            
            # Get position
            positions = predictions['object_positions']
            if positions.dim() > 2:
                positions = positions[0]  # Take first in batch
            position = Vector3(
                positions[i, 0].item(),
                positions[i, 1].item(),
                positions[i, 2].item()
            )
            
            # Get rotation
            rotations = predictions['object_rotations']
            if rotations.dim() > 2:
                rotations = rotations[0]  # Take first in batch
            rotation = Vector3(
                rotations[i, 0].item(),
                rotations[i, 1].item(),
                rotations[i, 2].item()
            )
            
            # Get scale
            scales = predictions['object_scales']
            if scales.dim() > 2:
                scales = scales[0]  # Take first in batch
            scale = Vector3(
                max(0.01, scales[i, 0].item()),  # Ensure positive
                max(0.01, scales[i, 1].item()),
                max(0.01, scales[i, 2].item())
            )
            
            # Get mass
            masses = predictions['object_masses']
            if masses.dim() > 2:
                masses = masses[0]  # Take first in batch
            mass = max(0.1, masses[i, 0].item())  # Ensure positive
            
            # Get material
            material_probs = predictions['object_material_probs']
            if material_probs.dim() > 2:
                material_probs = material_probs[0]  # Take first in batch
            
            material_idx = torch.argmax(material_probs[i]).item()
            material = self.idx_to_material[material_idx]
            
            # Create object
            obj = PhysicsObject(
                object_id=f"decoded_obj_{i}",
                object_type=object_type,
                position=position,
                rotation=rotation,
                scale=scale,
                mass=mass,
                material=material,
                material_properties=None,
                color=(0.7, 0.3, 0.3)  # Red for decoded objects
            )
            
            scene.add_object(obj)
        
        return scene


class PhysicsDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and processing physics scene examples.
    It loads examples from a JSON file and encodes them on-the-fly.
    """
    def __init__(self, dataset_path: str, encoder: SceneEncoder = None):
        """
        Args:
            dataset_path: Path to the dataset JSON file.
            encoder: An instance of SceneEncoder.
        """
        self.encoder = encoder
        self.examples = self._load_examples(dataset_path)
        self.texts = [ex['text_description'] for ex in self.examples]

    def _load_examples(self, filepath: str) -> List[Dict[str, any]]:
        """Load dataset from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            examples = data.get('examples', data)
            if not examples:
                raise ValueError(f"Dataset file {filepath} is empty or has no 'examples'.")

            # Check the first few examples for the action_sequence key to prevent zero-loss training
            if not all(ex.get('action_sequence') for ex in examples[:min(5, len(examples))]):
                print("\n" + "="*60)
                print("⚠️ WARNING: Dataset examples are missing the 'action_sequence' field.")
                print("This will cause the model's training loss to be zero, resulting in an ineffective model.")
                print("Please regenerate the dataset using 'python src/generate_dataset.py'")
                print("="*60 + "\n")
            return examples
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise IOError(f"Failed to load or parse dataset file at {filepath}: {e}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """
        Retrieves an item from the dataset.

        Args:
            idx: The index of the item.

        Returns:
            A tuple containing the text description and the target action sequence string.
        """
        example = self.examples[idx]
        text = example['text_description']
        action_sequence = example.get('action_sequence', '')
        return text, action_sequence


def test_scene_encoder_decoder():
    """Test the scene encoder/decoder system."""
    print("Testing scene encoder/decoder...")
    
    # Create test scene
    from scene_representation import SceneBuilder
    
    builder = SceneBuilder("test_encode_scene")
    builder.add_ground_plane()
    builder.add_ramp(position=(0, 0, 0), angle=0.3)
    builder.add_sphere(position=(-1, 0, 1.5), radius=0.1, mass=2.0)
    original_scene = builder.get_scene()
    
    print(f"Original scene: {len(original_scene.objects)} objects")
    
    # Test encoder
    encoder = SceneEncoder(max_objects=8)
    encoded = encoder.encode_scene(original_scene)
    
    print("Encoded tensors:")
    for key, tensor in encoded.items():
        print(f"  {key}: {tensor.shape}")
    
    # Test decoder with dummy predictions (simulating model output)
    decoder = SceneDecoder(max_objects=8)
    
    # Create dummy predictions in the format the model would output
    dummy_predictions = {
        'object_count_probs': torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),  # 2 objects
        'object_type_probs': torch.zeros(1, 8, len(ObjectType)),
        'object_positions': torch.randn(1, 8, 3),
        'object_rotations': torch.randn(1, 8, 3),
        'object_scales': torch.sigmoid(torch.randn(1, 8, 3)),
        'object_masses': torch.sigmoid(torch.randn(1, 8, 1)) * 5,
        'object_material_probs': torch.softmax(torch.randn(1, 8, len(MaterialType)), dim=-1),
        'gravity': torch.tensor([[0.0, 0.0, -9.81]])
    }
    
    # Set specific object types
    ramp_idx = list(ObjectType).index(ObjectType.RAMP)
    sphere_idx = list(ObjectType).index(ObjectType.SPHERE)
    dummy_predictions['object_type_probs'][0, 0, ramp_idx] = 1.0  # First object is ramp
    dummy_predictions['object_type_probs'][0, 1, sphere_idx] = 1.0  # Second object is sphere
    
    decoded_scene = decoder.decode_scene(dummy_predictions)
    
    print(f"\nDecoded scene: {len(decoded_scene.objects)} objects")
    for i, obj in enumerate(decoded_scene.objects):
        print(f"  Object {i}: {obj.object_type.value} at {obj.position.to_list()}")
    
    # Test batch processing
    print("\nTesting batch processing...")
    scenes = [original_scene] * 3
    batch_encoded = encoder.encode_batch(scenes)
    
    print("Batch encoded shapes:")
    for key, tensor in batch_encoded.items():
        print(f"  {key}: {tensor.shape}")
    
    print("\n✅ Scene encoder/decoder test completed!")


if __name__ == "__main__":
    test_scene_encoder_decoder()
