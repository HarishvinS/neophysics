"""
Neural Network Architecture for Text-to-Scene Translation
Implements encoder-decoder architecture to convert natural language to physics scenes.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, List

from scene_representation import PhysicsScene, PhysicsObject, ObjectType, MaterialType, Vector3


class TextToSceneModel(nn.Module):
    """Seq2Seq model for translating text to an action sequence."""
    
    def __init__(self, model_name: str = "t5-small", max_objects: int = 8, config: 'ModelConfig' = None):
        """
        Initialize the complete model.
        
        Args:
            model_name: Name of the pre-trained seq2seq model from Hugging Face.
            max_objects: Maximum number of objects to predict
            config: Model configuration for dropout and other settings
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Keep for compatibility with existing config and scripts
        self.max_objects = max_objects
        self.hidden_size = self.model.config.hidden_size
        
        # Load pre-trained weights if specified
        if config and config.model_path:
            self.load_checkpoint(config.model_path)
    
    def forward(self, texts: List[str], targets: List[str] = None):
        """
        Process a batch of texts and optional targets.
        
        Args:
            texts: List of text descriptions
            targets: List of target action sequences (for training)
            
        Returns:
            Hugging Face model output, which includes loss if targets are provided.
        """
        # Tokenize inputs
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        device = self.model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if targets:
            # Tokenize targets for loss calculation
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(targets, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs["labels"] = labels["input_ids"].to(device)
            return self.model(**inputs)
        else:
            # For inference, just return the inputs for generation
            return self.model.generate(**inputs)

    def predict_action_sequence(self, text: str, max_length: int = 256) -> str:
        """
        Predict a structured action sequence from a text command.
        
        Args:
            text: Text description
            max_length: Maximum length of the generated sequence.
            
        Returns:
            The predicted action sequence as a string.
        """
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=128)
            device = self.model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            output_ids = self.model.generate(**inputs, max_length=max_length)
            
            action_sequence = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return action_sequence
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)

            # Check for architecture mismatch by comparing keys
            model_keys = set(self.state_dict().keys())
            checkpoint_keys = set(state_dict.keys())

            if not model_keys.issubset(checkpoint_keys):
                missing_keys = model_keys - checkpoint_keys
                error_msg = (
                    "Failed to load checkpoint due to an architecture mismatch. "
                    "The saved model file is from an older, incompatible version of the code.\n"
                    f"  - The current model expects keys like: {list(missing_keys)[0]}, ...\n"
                    "  - Please retrain the model using 'python src/train_model.py' to generate a new, compatible checkpoint."
                )
                raise RuntimeError(error_msg)

            self.load_state_dict(state_dict)
            print(f"✅ Loaded model weights from {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint {checkpoint_path}: {e}")
    
    def save_checkpoint(self, checkpoint_path: str, epoch: int = None, optimizer_state: dict = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'hidden_size': self.hidden_size,
            'max_objects': self.max_objects
        }
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Saved model checkpoint to {checkpoint_path}")


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
        self.model_path = None  # Path to pre-trained model
        
        # Evaluation
        self.eval_frequency = 5  # epochs
        self.save_frequency = 10  # epochs


def test_model_architecture():
    """Test the model architecture with dummy data."""
    print("Testing model architecture...")
    
    # Create model
    config = ModelConfig()
    model = TextToSceneModel(config=config)
    
    # Test with dummy texts
    test_texts = [
        "create a ball on a ramp",
        "add two spheres and a box",
        "place a bouncy ball"
    ]
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test action sequence prediction
    print("\nTesting action sequence prediction...")
    for text in test_texts:
        action_sequence = model.predict_action_sequence(text)
        print(f"  Input: '{text}'")
        print(f"  > Predicted Action Sequence: {action_sequence}")
    
    print("\n✅ Model architecture test completed!")


if __name__ == "__main__":
    test_model_architecture()
