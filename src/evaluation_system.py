"""
Evaluation System for Text-to-Scene Model
Provides metrics and evaluation tools to measure model performance.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

from model_architecture import TextToSceneModel
from scene_representation import PhysicsScene, ObjectType, MaterialType
from scene_encoder import SceneEncoder, SceneDecoder
from physics_engine import PhysicsEngine


class SceneMetrics:
    """Metrics for evaluating scene prediction quality."""
    
    def __init__(self, max_objects: int = 8):
        """Initialize scene metrics."""
        self.max_objects = max_objects
        self.encoder = SceneEncoder(max_objects=max_objects)
        self.decoder = SceneDecoder(max_objects=max_objects)
    
    def object_count_accuracy(self, predictions: Dict[str, torch.Tensor], 
                             targets: Dict[str, torch.Tensor]) -> float:
        """Calculate object count prediction accuracy."""
        pred_counts = torch.argmax(predictions['object_count_probs'], dim=1)
        true_counts = torch.argmax(targets['object_count'], dim=1)
        
        accuracy = (pred_counts == true_counts).float().mean().item()
        return accuracy
    
    def object_type_accuracy(self, predictions: Dict[str, torch.Tensor], 
                           targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate object type prediction accuracy using argmax for consistency."""
        # Get predicted and true type indices
        pred_type_indices = torch.argmax(predictions['object_type_probs'], dim=2)
        true_type_indices = torch.argmax(targets['object_types'], dim=2)

        # Create a mask to only evaluate existing objects based on ground truth count
        true_counts = torch.argmax(targets['object_count'], dim=1)
        max_len = pred_type_indices.shape[1]
        mask = torch.arange(max_len, device=true_counts.device).unsqueeze(0) < true_counts.unsqueeze(1)

        if mask.sum() == 0:
            return {'overall': 1.0, 'per_type': {}} # Perfect score if no objects to evaluate

        # Calculate accuracy only for existing objects
        correct_predictions = (pred_type_indices[mask] == true_type_indices[mask])
        overall_accuracy = correct_predictions.float().mean().item()

        # Per-type accuracy
        type_accuracies = {}
        for i, obj_type in enumerate(ObjectType):
            # Mask for where this type was the ground truth among existing objects
            type_mask = (true_type_indices == i) & mask
            if type_mask.sum() > 0:
                # Accuracy for this specific type
                type_acc = (pred_type_indices[type_mask] == true_type_indices[type_mask]).float().mean().item()
                type_accuracies[obj_type.value] = type_acc
        
        return {
            'overall': overall_accuracy,
            'per_type': type_accuracies
        }
    
    def position_error(self, predictions: Dict[str, torch.Tensor], 
                      targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate position prediction errors."""
        pred_pos = predictions['object_positions']
        true_pos = targets['object_positions']
        
        # Calculate L2 distance error
        l2_error = torch.norm(pred_pos - true_pos, dim=2)
        
        # Calculate L1 (Manhattan) distance error
        l1_error = torch.abs(pred_pos - true_pos).sum(dim=2)
        
        return {
            'l2_mean': l2_error.mean().item(),
            'l2_std': l2_error.std().item(),
            'l1_mean': l1_error.mean().item(),
            'l1_std': l1_error.std().item()
        }
    
    def scale_error(self, predictions: Dict[str, torch.Tensor], 
                   targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate scale prediction errors."""
        pred_scale = predictions['object_scales']
        true_scale = targets['object_scales']
        
        # Relative error (to handle different scales)
        epsilon = torch.finfo(true_scale.dtype).eps * 10
        relative_error = torch.abs(pred_scale - true_scale) / (true_scale + epsilon)
        
        return {
            'relative_mean': relative_error.mean().item(),
            'relative_std': relative_error.std().item()
        }
    
    def mass_error(self, predictions: Dict[str, torch.Tensor], 
                  targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate mass prediction errors."""
        pred_mass = predictions['object_masses']
        true_mass = targets['object_masses']
        
        # Relative error
        epsilon = torch.finfo(true_mass.dtype).eps * 10
        relative_error = torch.abs(pred_mass - true_mass) / (true_mass + epsilon)
        
        return {
            'relative_mean': relative_error.mean().item(),
            'relative_std': relative_error.std().item()
        }
    
    def material_accuracy(self, predictions: Dict[str, torch.Tensor], 
                         targets: Dict[str, torch.Tensor]) -> float:
        """Calculate material prediction accuracy."""
        pred_materials = torch.argmax(predictions['object_material_probs'], dim=2)
        true_materials = torch.argmax(targets['object_materials'], dim=2)
        
        accuracy = (pred_materials == true_materials).float().mean().item()
        return accuracy
    
    def compute_all_metrics(self, predictions: Dict[str, torch.Tensor], 
                           targets: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """Compute all evaluation metrics."""
        metrics = {}
        
        # Object count
        metrics['object_count_accuracy'] = self.object_count_accuracy(predictions, targets)
        
        # Object types
        metrics['object_type_accuracy'] = self.object_type_accuracy(predictions, targets)
        
        # Position errors
        metrics['position_error'] = self.position_error(predictions, targets)
        
        # Scale errors
        metrics['scale_error'] = self.scale_error(predictions, targets)
        
        # Mass errors
        metrics['mass_error'] = self.mass_error(predictions, targets)
        
        # Material accuracy
        metrics['material_accuracy'] = self.material_accuracy(predictions, targets)
        
        return metrics


class PhysicsEvaluator:
    """Evaluates predicted scenes using physics simulation."""
    
    def __init__(self):
        """Initialize physics evaluator."""
        self.physics_engine = None
    
    def evaluate_physics_plausibility(self, scene: PhysicsScene) -> Dict[str, float]:
        """
        Evaluate how physically plausible a scene is.
        
        Args:
            scene: PhysicsScene to evaluate
            
        Returns:
            Dictionary of physics plausibility scores
        """
        scores = {}
        
        # Check for basic physics violations
        scores['objects_in_bounds'] = self._check_objects_in_bounds(scene)
        scores['no_overlaps'] = self._check_no_overlaps(scene)
        scores['stable_configuration'] = self._check_stability(scene)
        scores['reasonable_masses'] = self._check_reasonable_masses(scene)
        scores['reasonable_scales'] = self._check_reasonable_scales(scene)
        
        # Overall plausibility score
        scores['overall'] = np.mean(list(scores.values()))
        
        return scores
    
    def _check_objects_in_bounds(self, scene: PhysicsScene) -> float:
        """Check if objects are within reasonable bounds."""
        violations = 0
        total_objects = 0
        
        for obj in scene.objects:
            if obj.object_type.value == 'plane':  # Skip ground plane
                continue
            
            total_objects += 1
            
            # Check position bounds
            if abs(obj.position.x) > 10 or abs(obj.position.y) > 10:
                violations += 1
            if obj.position.z < -1 or obj.position.z > 10:
                violations += 1
        
        if total_objects == 0:
            return 1.0
        
        return 1.0 - (violations / total_objects)
    
    def _check_no_overlaps(self, scene: PhysicsScene) -> float:
        """Check for object overlaps."""
        objects = [obj for obj in scene.objects if obj.object_type.value != 'plane']
        overlaps = 0
        total_pairs = 0
        
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                total_pairs += 1
                
                # Calculate distance between objects
                distance = np.sqrt(
                    (obj1.position.x - obj2.position.x)**2 +
                    (obj1.position.y - obj2.position.y)**2 +
                    (obj1.position.z - obj2.position.z)**2
                )
                
                # Estimate minimum safe distance
                size1 = max(obj1.scale.x, obj1.scale.y, obj1.scale.z)
                size2 = max(obj2.scale.x, obj2.scale.y, obj2.scale.z)
                min_distance = (size1 + size2) * 0.6
                
                if distance < min_distance:
                    overlaps += 1
        
        if total_pairs == 0:
            return 1.0
        
        return 1.0 - (overlaps / total_pairs)
    
    def _check_stability(self, scene: PhysicsScene) -> float:
        """Check for basic stability (objects not floating unreasonably)."""
        violations = 0
        total_objects = 0
        
        for obj in scene.objects:
            if obj.object_type.value == 'plane' or obj.mass == 0:  # Skip static objects
                continue
            
            total_objects += 1
            
            # Check if object is floating too high without support
            if obj.position.z > 3.0:
                violations += 1
        
        if total_objects == 0:
            return 1.0
        
        return 1.0 - (violations / total_objects)
    
    def _check_reasonable_masses(self, scene: PhysicsScene) -> float:
        """Check if masses are reasonable."""
        violations = 0
        total_objects = 0
        
        for obj in scene.objects:
            if obj.mass == 0:  # Skip static objects
                continue
            
            total_objects += 1
            
            # Check mass bounds
            if obj.mass < 0.01 or obj.mass > 100:
                violations += 1
        
        if total_objects == 0:
            return 1.0
        
        return 1.0 - (violations / total_objects)
    
    def _check_reasonable_scales(self, scene: PhysicsScene) -> float:
        """Check if object scales are reasonable."""
        violations = 0
        total_objects = 0
        
        for obj in scene.objects:
            if obj.object_type.value == 'plane':  # Skip ground plane
                continue
            
            total_objects += 1
            
            # Check scale bounds
            min_scale = min(obj.scale.x, obj.scale.y, obj.scale.z)
            max_scale = max(obj.scale.x, obj.scale.y, obj.scale.z)
            
            if min_scale < 0.01 or max_scale > 5.0:
                violations += 1
        
        if total_objects == 0:
            return 1.0
        
        return 1.0 - (violations / total_objects)


class ModelEvaluator:
    """Complete model evaluation system."""
    
    def __init__(self, model: TextToSceneModel, max_objects: int = 8):
        """Initialize model evaluator."""
        self.model = model
        self.scene_metrics = SceneMetrics(max_objects=max_objects)
        self.physics_evaluator = PhysicsEvaluator()
        self.decoder = SceneDecoder(max_objects=max_objects)
    
    def evaluate_on_dataset(self, test_loader: DataLoader) -> Dict[str, any]:
        """
        Evaluate model on a dataset.
        
        Args:
            test_loader: DataLoader for the test set.
            
        Returns:
            Comprehensive evaluation results
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        num_examples = 0
        
        with torch.inference_mode():
            for texts, targets_batch in test_loader:
                # Get model predictions for the batch
                predictions_batch = self.model(texts)
                
                # Store predictions and targets for metric calculation
                all_predictions.append(predictions_batch)
                all_targets.append(targets_batch)
                num_examples += len(texts)

        # Collate all batch results
        predictions = {key: torch.cat([p[key] for p in all_predictions], dim=0) for key in all_predictions[0]}
        
        # The targets are a list of dicts, need to re-format
        target_keys = all_targets[0][0].keys()
        targets = {key: torch.stack([t[key] for batch in all_targets for t in batch]) for key in target_keys}

        # Compute scene metrics
        scene_metrics = self.scene_metrics.compute_all_metrics(predictions, targets)
        
        # Evaluate physics plausibility
        physics_scores = []
        for i in range(num_examples):
            single_pred = {key: tensor[i:i+1] for key, tensor in predictions.items()}
            predicted_scene = self.decoder.decode_scene(single_pred, f"eval_scene_{i}")
            physics_score = self.physics_evaluator.evaluate_physics_plausibility(predicted_scene)
            physics_scores.append(physics_score)
        
            # Aggregate physics scores
            physics_metrics = {}
            if physics_scores:
                for key in physics_scores[0].keys():
                    physics_metrics[key] = np.mean([score[key] for score in physics_scores])
        
        return {
            'scene_metrics': scene_metrics,
            'physics_metrics': physics_metrics,
            'num_examples': num_examples
        }
    
    def evaluate_text_examples(self, test_texts: List[str]) -> List[Dict[str, any]]:
        """
        Evaluate model on specific text examples.
        
        Args:
            test_texts: List of text descriptions to test
            
        Returns:
            List of evaluation results for each text
        """
        results = []
        
        self.model.eval()
        with torch.inference_mode():
            for i, text in enumerate(test_texts):
                # Get prediction
                predictions = self.model([text])
                
                # Decode scene
                predicted_scene = self.decoder.decode_scene(predictions, f"test_scene_{i}")
                
                # Evaluate physics
                physics_score = self.physics_evaluator.evaluate_physics_plausibility(predicted_scene)
                
                results.append({
                    'text': text,
                    'predicted_scene': predicted_scene,
                    'physics_score': physics_score,
                    'num_objects': sum(1 for obj in predicted_scene.objects if obj.object_type.value != 'plane')
                })
        
        return results


def test_evaluation_system():
    """Test the evaluation system."""
    print("Testing evaluation system...")
    
    # Create a simple model for testing
    from model_architecture import ModelConfig
    config = ModelConfig()
    model = TextToSceneModel(hidden_size=config.hidden_size, max_objects=config.max_objects)
    
    # Create evaluator
    evaluator = ModelEvaluator(model, max_objects=config.max_objects)
    
    # Test with sample texts
    test_texts = [
        "create a ball on a ramp",
        "add two spheres",
        "place a box next to a cylinder"
    ]
    
    print(f"\nEvaluating {len(test_texts)} text examples...")
    results = evaluator.evaluate_text_examples(test_texts)
    
    for i, result in enumerate(results):
        print(f"\nExample {i+1}: '{result['text']}'")
        print(f"  Predicted objects: {result['num_objects']}")
        print(f"  Physics score: {result['physics_score']['overall']:.3f}")
        print(f"  Plausibility breakdown:")
        for key, score in result['physics_score'].items():
            if key != 'overall':
                print(f"    {key}: {score:.3f}")
    
    print("\n✅ Evaluation system test completed!")


if __name__ == "__main__":
    test_evaluation_system()
