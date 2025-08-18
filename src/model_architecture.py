"""
Neural Network Architecture for Text-to-Scene Translation
Implements encoder-decoder architecture to convert natural language to physics scenes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Tuple, Optional

from scene_representation import PhysicsScene, PhysicsObject, ObjectType, MaterialType, Vector3


class TextEncoder(nn.Module):
    """Encodes natural language text into a fixed-size representation."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", hidden_size: int = 256, dropout_rate: float = 0.1):
        """
        Initialize text encoder.
        
        Args:
            model_name: Pre-trained transformer model name
            hidden_size: Size of the output hidden representation
        """
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze transformer weights initially (can be unfrozen later)
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        # Project transformer output to our hidden size
        transformer_hidden_size = self.transformer.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(transformer_hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        self.hidden_size = hidden_size
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts into fixed-size representations.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tensor of shape (batch_size, hidden_size)
        """
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Move inputs to model device
        device = next(self.transformer.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Get transformer outputs
        with torch.no_grad():
            transformer_outputs = self.transformer(**inputs)
        
        # Use CLS token representation
        cls_embeddings = transformer_outputs.last_hidden_state[:, 0, :]
        
        # Project to our hidden size
        encoded = self.projection(cls_embeddings)
        
        return encoded


class SceneDecoder(nn.Module):
    """Decodes hidden representations into physics scene parameters."""
    
    def __init__(self, hidden_size: int = 256, max_objects: int = 8, dropout_rate: float = 0.1):
        """
        Initialize scene decoder.
        
        Args:
            hidden_size: Size of input hidden representation
            max_objects: Maximum number of objects to predict
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_objects = max_objects
        
        # Object count predictor
        self.object_count_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, max_objects + 1),  # 0 to max_objects
            nn.Softmax(dim=-1)
        )
        
        # Object type predictor (for each potential object)
        self.object_type_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, len(ObjectType) * max_objects),
            nn.Sigmoid()
        )
        
        # Object properties predictor
        # Each object has: position (3), rotation (3), scale (3), mass (1), material (len(MaterialType))
        props_per_object = 3 + 3 + 3 + 1 + len(MaterialType)
        self.object_props_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, props_per_object * max_objects)
        )
        
        # Scene environment predictor
        self.environment_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # gravity (x, y, z)
        )
    
    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode hidden representation into scene parameters.
        
        Args:
            hidden: Hidden representation of shape (batch_size, hidden_size)
            
        Returns:
            Dictionary containing predicted scene parameters
        """
        batch_size = hidden.shape[0]
        
        # Predict object count
        object_count_probs = self.object_count_head(hidden)
        
        # Predict object types (return logits for BCEWithLogitsLoss)
        object_type_logits = self.object_type_head(hidden)
        object_type_logits = object_type_logits.view(batch_size, self.max_objects, len(ObjectType))
        
        # Predict object properties
        object_props = self.object_props_head(hidden)
        props_per_object = 3 + 3 + 3 + 1 + len(MaterialType)
        object_props = object_props.view(batch_size, self.max_objects, props_per_object)
        
        # Split properties
        position = object_props[:, :, 0:3]  # x, y, z
        rotation = object_props[:, :, 3:6]  # rx, ry, rz
        scale = torch.sigmoid(object_props[:, :, 6:9]) * 2.0  # sx, sy, sz (0-2 range)
        mass = torch.sigmoid(object_props[:, :, 9:10]) * 10.0  # mass (0-10 kg)
        material_logits = object_props[:, :, 10:]  # material probabilities
        material_probs = torch.softmax(material_logits, dim=-1)
        
        # Predict environment
        gravity = self.environment_head(hidden)
        
        return {
            'object_count_probs': object_count_probs,
            'object_type_logits': object_type_logits,
            'object_positions': position,
            'object_rotations': rotation,
            'object_scales': scale,
            'object_masses': mass,
            'object_material_probs': material_probs,
            'gravity': gravity
        }


class TextToSceneModel(nn.Module):
    """Complete text-to-scene translation model."""
    
    def __init__(self, hidden_size: int = 256, max_objects: int = 8, config: 'ModelConfig' = None):
        """
        Initialize the complete model.
        
        Args:
            hidden_size: Size of hidden representations
            max_objects: Maximum number of objects to predict
            config: Model configuration for dropout and other settings
        """
        super().__init__()
        
        dropout_rate = config.dropout_rate if config else 0.1
        self.encoder = TextEncoder(hidden_size=hidden_size, dropout_rate=dropout_rate)
        self.decoder = SceneDecoder(hidden_size=hidden_size, max_objects=max_objects, dropout_rate=dropout_rate)
        
        self.hidden_size = hidden_size
        self.max_objects = max_objects
    
    def forward(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Translate texts to scene parameters.
        
        Args:
            texts: List of text descriptions
            
        Returns:
            Dictionary containing predicted scene parameters
        """
        # Encode texts
        hidden = self.encoder(texts)
        
        # Decode to scene parameters
        scene_params = self.decoder(hidden)
        
        return scene_params
    
    def predict_scene(self, text: str) -> PhysicsScene:
        """
        Predict a complete physics scene from text.
        
        Args:
            text: Text description
            threshold: Threshold for object existence
            
        Returns:
            Predicted PhysicsScene object
        """
        self.eval()
        with torch.inference_mode():
            # Get predictions
            predictions = self.forward([text])
            
            # Extract most likely object count
            object_count_probs = predictions['object_count_probs'][0]
            predicted_count = torch.argmax(object_count_probs).item()
            
            # Create scene
            scene = PhysicsScene(
                scene_id=f"predicted_scene",
                objects=[],
                environment=None
            )
            
            # Extract all tensors at once for efficiency
            positions = predictions['object_positions'][0].cpu().numpy()
            rotations = predictions['object_rotations'][0].cpu().numpy()
            scales = predictions['object_scales'][0].cpu().numpy()
            masses = predictions['object_masses'][0].cpu().numpy()
            
            # Add predicted objects
            for i in range(min(predicted_count, self.max_objects)):
                # Get object type (apply sigmoid to logits)
                type_logits = predictions['object_type_logits'][0, i]
                type_probs = torch.sigmoid(type_logits)
                object_type_idx = torch.argmax(type_probs).item()
                object_type = list(ObjectType)[object_type_idx]
                
                # Get object properties
                position = positions[i]
                rotation = rotations[i]
                scale = scales[i]
                mass = masses[i, 0]
                
                # Get material
                material_probs = predictions['object_material_probs'][0, i]
                material_idx = torch.argmax(material_probs).item()
                material = list(MaterialType)[material_idx]
                
                # Create object
                obj = PhysicsObject(
                    object_id=f"predicted_obj_{i}",
                    object_type=object_type,
                    position=Vector3(position[0], position[1], position[2]),
                    rotation=Vector3(rotation[0], rotation[1], rotation[2]),
                    scale=Vector3(scale[0], scale[1], scale[2]),
                    mass=mass,
                    material=material,
                    material_properties=None,
                    color=(0.5, 0.5, 0.5)
                )
                
                scene.add_object(obj)
            
            return scene


class ModelConfig:
    """Configuration class for model hyperparameters."""
    
    def __init__(self):
        # Model architecture
        self.hidden_size = 256
        self.max_objects = 8
        self.transformer_model = "distilbert-base-uncased"
        
        # Training parameters
        self.learning_rate = 1e-3
        self.batch_size = 16
        self.num_epochs = 50
        self.weight_decay = 1e-4
        
        # Loss weights
        self.object_count_weight = 1.0
        self.object_type_weight = 1.0
        self.object_props_weight = 1.0
        self.environment_weight = 0.5
        
        # Regularization
        self.dropout_rate = 0.1
        self.gradient_clip_norm = 1.0
        
        # Paths
        self.checkpoint_dir = 'models/checkpoints'
        
        # Evaluation
        self.eval_frequency = 5  # epochs
        self.save_frequency = 10  # epochs


def test_model_architecture():
    """Test the model architecture with dummy data."""
    print("Testing model architecture...")
    
    # Create model
    config = ModelConfig()
    model = TextToSceneModel(
        hidden_size=config.hidden_size,
        max_objects=config.max_objects
    )
    
    # Test with dummy texts
    test_texts = [
        "create a ball on a ramp",
        "add two spheres and a box",
        "place a bouncy ball"
    ]
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    with torch.no_grad():
        predictions = model(test_texts)
    
    print("\nPrediction shapes:")
    for key, tensor in predictions.items():
        print(f"  {key}: {tensor.shape}")
    
    # Test scene prediction
    predicted_scene = model.predict_scene("create a ball on a ramp")
    print(f"\nPredicted scene:")
    print(f"  Objects: {len(predicted_scene.objects)}")
    for i, obj in enumerate(predicted_scene.objects):
        print(f"  Object {i}: {obj.object_type.value} at {obj.position.to_list()}")
    
    print("\n✅ Model architecture test completed!")


if __name__ == "__main__":
    test_model_architecture()
