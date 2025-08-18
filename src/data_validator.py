"""
Data Validation System
Ensures generated scenes are physically plausible and text descriptions are accurate.
"""

import re
from typing import List, Dict, Any, Tuple
import numpy as np

from scene_representation import PhysicsScene, PhysicsObject, ObjectType, MaterialType, Vector3, TrainingExample


class PhysicsValidator:
    """Validates physics scenes for plausibility."""
    
    def __init__(self):
        """Initialize the physics validator."""
        self.validation_rules = {
            "object_bounds": self._check_object_bounds,
            "mass_constraints": self._check_mass_constraints,
            "size_constraints": self._check_size_constraints,
            "material_consistency": self._check_material_consistency,
            "spatial_overlap": self._check_spatial_overlap,
            "stability": self._check_stability
        }
    
    def validate_scene(self, scene: PhysicsScene) -> Dict[str, Any]:
        """
        Validate a physics scene.
        
        Args:
            scene: The scene to validate
            
        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "score": 1.0,
            "issues": [],
            "warnings": [],
            "rule_results": {}
        }
        
        # Run all validation rules
        for rule_name, rule_func in self.validation_rules.items():
            try:
                rule_result = rule_func(scene)
                results["rule_results"][rule_name] = rule_result
                
                if not rule_result["passed"]:
                    results["valid"] = False
                    results["issues"].extend(rule_result.get("issues", []))
                
                if rule_result.get("warnings"):
                    results["warnings"].extend(rule_result["warnings"])
                
                # Update score (multiply by rule score)
                results["score"] *= rule_result.get("score", 1.0)
                
            except Exception as e:
                results["issues"].append(f"Error in rule {rule_name}: {str(e)}")
                results["valid"] = False
                results["score"] *= 0.5
        
        return results
    
    def _check_object_bounds(self, scene: PhysicsScene) -> Dict[str, Any]:
        """Check if objects are within scene bounds."""
        bounds = scene.environment.scene_bounds
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        
        issues = []
        warnings = []
        
        for obj in scene.objects:
            pos = obj.position
            
            # Check if object is within bounds
            if not (x_min <= pos.x <= x_max):
                issues.append(f"Object {obj.object_id} x-position {pos.x} outside bounds [{x_min}, {x_max}]")
            
            if not (y_min <= pos.y <= y_max):
                issues.append(f"Object {obj.object_id} y-position {pos.y} outside bounds [{y_min}, {y_max}]")
            
            if not (z_min <= pos.z <= z_max):
                issues.append(f"Object {obj.object_id} z-position {pos.z} outside bounds [{z_min}, {z_max}]")
            
            # Warning for objects very close to bounds
            margin = 0.5
            if pos.z < margin and obj.object_type != ObjectType.PLANE:
                warnings.append(f"Object {obj.object_id} very close to ground (z={pos.z:.2f})")
        
        return {
            "passed": not issues,
            "score": 1.0 if not issues else max(0.1, 1.0 - len(issues) * 0.2),
            "issues": issues,
            "warnings": warnings
        }
    
    def _check_mass_constraints(self, scene: PhysicsScene) -> Dict[str, Any]:
        """Check if object masses are reasonable."""
        issues = []
        warnings = []
        
        for obj in scene.objects:
            if obj.mass < 0:
                issues.append(f"Object {obj.object_id} has negative mass: {obj.mass}")
            
            # Check for unreasonably large masses
            if obj.mass > 1000:
                warnings.append(f"Object {obj.object_id} has very large mass: {obj.mass}kg")
            
            # Check for unreasonably small masses (except static objects)
            if obj.mass > 0 and obj.mass < 0.01:
                warnings.append(f"Object {obj.object_id} has very small mass: {obj.mass}kg")
            
            # Static objects should have mass 0
            if obj.object_type in [ObjectType.PLANE, ObjectType.RAMP] and obj.mass > 0:
                warnings.append(f"Static object {obj.object_id} has non-zero mass: {obj.mass}")
        
        return {
            "passed": not issues,
            "score": 1.0 if not issues else 0.5,
            "issues": issues,
            "warnings": warnings
        }
    
    def _check_size_constraints(self, scene: PhysicsScene) -> Dict[str, Any]:
        """Check if object sizes are reasonable."""
        issues = []
        warnings = []
        
        for obj in scene.objects:
            scale = obj.scale
            
            # Check for negative or zero dimensions
            if scale.x <= 0 or scale.y <= 0 or scale.z <= 0:
                issues.append(f"Object {obj.object_id} has invalid dimensions: {scale.to_list()}")
            
            # Check for unreasonably large objects
            max_dim = max(scale.x, scale.y, scale.z)
            if max_dim > 10:
                warnings.append(f"Object {obj.object_id} is very large: max dimension {max_dim}")
            
            # Check for unreasonably small objects
            min_dim = min(scale.x, scale.y, scale.z)
            if min_dim < 0.01:
                warnings.append(f"Object {obj.object_id} is very small: min dimension {min_dim}")
        
        return {
            "passed": not issues,
            "score": 1.0 if not issues else 0.3,
            "issues": issues,
            "warnings": warnings
        }
    
    def _check_material_consistency(self, scene: PhysicsScene) -> Dict[str, Any]:
        """Check if material properties are consistent."""
        issues = []
        warnings = []
        
        for obj in scene.objects:
            if obj.material_properties is None:
                continue
            
            props = obj.material_properties
            
            # Check friction bounds
            if props.friction < 0 or props.friction > 2:
                issues.append(f"Object {obj.object_id} has invalid friction: {props.friction}")
            
            # Check restitution bounds
            if props.restitution < 0 or props.restitution > 1:
                issues.append(f"Object {obj.object_id} has invalid restitution: {props.restitution}")
            
            # Check density bounds
            if props.density <= 0 or props.density > 50000:
                issues.append(f"Object {obj.object_id} has invalid density: {props.density}")
            
            # Material-specific checks
            if obj.material == MaterialType.ICE and props.friction > 0.3:
                warnings.append(f"Ice object {obj.object_id} has high friction: {props.friction}")
            
            if obj.material == MaterialType.BOUNCY and props.restitution < 0.8:
                warnings.append(f"Bouncy object {obj.object_id} has low restitution: {props.restitution}")
        
        return {
            "passed": not issues,
            "score": 1.0 if not issues else 0.4,
            "issues": issues,
            "warnings": warnings
        }
    
    def _check_spatial_overlap(self, scene: PhysicsScene) -> Dict[str, Any]:
        """Check for unreasonable object overlaps."""
        issues = []
        warnings = []
        
        objects = [obj for obj in scene.objects if obj.object_type != ObjectType.PLANE]
        
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                distance = np.sqrt(
                    (obj1.position.x - obj2.position.x)**2 +
                    (obj1.position.y - obj2.position.y)**2 +
                    (obj1.position.z - obj2.position.z)**2
                )
                
                # Estimate object sizes
                size1 = max(obj1.scale.x, obj1.scale.y, obj1.scale.z)
                size2 = max(obj2.scale.x, obj2.scale.y, obj2.scale.z)
                min_distance = (size1 + size2) * 0.5
                
                if distance < min_distance * 0.5:  # Significant overlap
                    issues.append(f"Objects {obj1.object_id} and {obj2.object_id} overlap significantly")
                elif distance < min_distance:  # Minor overlap
                    warnings.append(f"Objects {obj1.object_id} and {obj2.object_id} are very close")
        
        return {
            "passed": not issues,
            "score": 1.0 if not issues else max(0.2, 1.0 - len(issues) * 0.3),
            "issues": issues,
            "warnings": warnings
        }
    
    def _check_stability(self, scene: PhysicsScene) -> Dict[str, Any]:
        """Check for basic stability issues."""
        issues = []
        warnings = []
        
        # Check for objects floating in air without support
        for obj in scene.objects:
            if obj.object_type == ObjectType.PLANE or obj.mass == 0:
                continue
            
            if obj.position.z > 5:
                warnings.append(f"Object {obj.object_id} is very high (z={obj.position.z:.2f})")
            
            # Check for objects with extreme initial velocities
            if obj.initial_velocity:
                speed = np.sqrt(obj.initial_velocity.x**2 + obj.initial_velocity.y**2 + obj.initial_velocity.z**2)
                if speed > 20:
                    warnings.append(f"Object {obj.object_id} has very high initial velocity: {speed:.2f} m/s")
        
        return {
            "passed": True,  # Stability issues are usually warnings
            "score": 1.0 if not warnings else max(0.7, 1.0 - len(warnings) * 0.1),
            "issues": issues,
            "warnings": warnings
        }


class TextValidator:
    """Validates text descriptions for quality and accuracy."""
    
    def __init__(self):
        """Initialize the text validator."""
        self.min_length = 5
        self.max_length = 200
        
        # Common physics terms that should appear in descriptions
        self.physics_terms = {
            "objects": ["ball", "sphere", "box", "cube", "ramp", "cylinder", "cone"],
            "materials": ["rubber", "metal", "wood", "ice", "plastic", "bouncy", "glass", "stone"],
            "actions": ["create", "place", "add", "roll", "bounce", "fall", "drop", "build"],
            "properties": ["heavy", "light", "kg", "mass", "velocity", "speed"]
        }
    
    def validate_text(self, text: str, scene: PhysicsScene) -> Dict[str, Any]:
        """
        Validate a text description.
        
        Args:
            text: The text description to validate
            scene: The corresponding scene
            
        Returns:
            Validation results dictionary
        """
        results = {
            "valid": True,
            "score": 1.0,
            "issues": [],
            "warnings": []
        }
        
        # Basic text quality checks
        length_result = self._check_text_length(text)
        results = self._merge_results(results, length_result)
        
        # Grammar and structure checks
        grammar_result = self._check_grammar(text)
        results = self._merge_results(results, grammar_result)
        
        # Content accuracy checks
        accuracy_result = self._check_content_accuracy(text, scene)
        results = self._merge_results(results, accuracy_result)
        
        # Vocabulary richness
        vocab_result = self._check_vocabulary(text)
        results = self._merge_results(results, vocab_result)
        
        return results
    
    def _check_text_length(self, text: str) -> Dict[str, Any]:
        """Check if text length is appropriate."""
        length = len(text.strip())
        
        if length < self.min_length:
            return {
                "valid": False,
                "score": 0.1,
                "issues": [f"Text too short: {length} characters (minimum {self.min_length})"],
                "warnings": []
            }
        
        if length > self.max_length:
            return {
                "valid": False,
                "score": 0.3,
                "issues": [f"Text too long: {length} characters (maximum {self.max_length})"],
                "warnings": []
            }
        
        return {"valid": True, "score": 1.0, "issues": [], "warnings": []}
    
    def _check_grammar(self, text: str) -> Dict[str, Any]:
        """Basic grammar and structure checks."""
        issues = []
        warnings = []
        
        # Check for basic sentence structure
        if not text.strip():
            issues.append("Empty text")
        
        # Check for proper capitalization
        if text and not text[0].isupper():
            warnings.append("Text should start with capital letter")
        
        # Check for repeated words
        words = text.lower().split()
        if len(words) != len(set(words)):
            warnings.append("Text contains repeated words")
        
        # Check for very short sentences
        if len(words) < 3:
            warnings.append("Text is very short")
        
        return {
            "valid": not issues,
            "score": 1.0 if not issues else 0.5,
            "issues": issues,
            "warnings": warnings
        }
    
    def _check_content_accuracy(self, text: str, scene: PhysicsScene) -> Dict[str, Any]:
        """Check if text accurately describes the scene."""
        issues = []
        warnings = []
        
        text_lower = text.lower()
        
        # Count objects in scene (excluding ground plane)
        scene_objects = [obj for obj in scene.objects if obj.object_type != ObjectType.PLANE]
        
        # Check if text mentions appropriate number of objects
        object_mentions = 0
        for term in self.physics_terms["objects"]:
            if term in text_lower:
                object_mentions += text_lower.count(term)
        
        if len(scene_objects) > 1 and object_mentions < 2:
            warnings.append("Text doesn't mention multiple objects present in scene")
        
        # Check for material mentions when materials are diverse
        materials_in_scene = set(obj.material for obj in scene_objects)
        material_mentions = sum(1 for term in self.physics_terms["materials"] if term in text_lower)
        
        if len(materials_in_scene) > 1 and material_mentions == 0:
            warnings.append("Scene has diverse materials but text doesn't mention any")
        
        # Check for action words
        action_mentions = sum(1 for term in self.physics_terms["actions"] if term in text_lower)
        if action_mentions == 0:
            warnings.append("Text doesn't contain action words")
        
        return {
            "valid": not issues,
            "score": max(0.3, 1.0 - len(warnings) * 0.2),
            "issues": issues,
            "warnings": warnings
        }
    
    def _check_vocabulary(self, text: str) -> Dict[str, Any]:
        """Check vocabulary richness and appropriateness."""
        warnings = []
        
        words = text.lower().split()
        unique_words = set(words)
        
        # Check vocabulary diversity
        if len(words) > 5 and len(unique_words) / len(words) < 0.7:
            warnings.append("Low vocabulary diversity")
        
        # Check for physics-related terms
        physics_term_count = 0
        for category in self.physics_terms.values():
            physics_term_count += sum(1 for term in category if term in text.lower())
        
        if physics_term_count == 0:
            warnings.append("No physics-related terms found")
        
        return {
            "valid": True,
            "score": max(0.5, 1.0 - len(warnings) * 0.2),
            "issues": [],
            "warnings": warnings
        }
    
    def _merge_results(self, results1: Dict[str, Any], results2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two validation results."""
        return {
            "valid": results1["valid"] and results2["valid"],
            "score": results1["score"] * results2["score"],
            "issues": results1["issues"] + results2["issues"],
            "warnings": results1["warnings"] + results2["warnings"]
        }


class DatasetValidator:
    """Validates complete training datasets."""
    
    def __init__(self):
        """Initialize the dataset validator."""
        self.physics_validator = PhysicsValidator()
        self.text_validator = TextValidator()
    
    def validate_example(self, example: TrainingExample) -> Dict[str, Any]:
        """Validate a single training example."""
        # Validate scene
        scene_results = self.physics_validator.validate_scene(example.scene)
        
        # Validate text
        text_results = self.text_validator.validate_text(example.text_description, example.scene)
        
        # Combine results
        overall_valid = scene_results["valid"] and text_results["valid"]
        overall_score = scene_results["score"] * text_results["score"]
        
        return {
            "valid": overall_valid,
            "score": overall_score,
            "scene_validation": scene_results,
            "text_validation": text_results,
            "issues": scene_results["issues"] + text_results["issues"],
            "warnings": scene_results["warnings"] + text_results["warnings"]
        }
    
    def validate_dataset(self, examples: List[TrainingExample]) -> Dict[str, Any]:
        """Validate an entire dataset."""
        print(f"Validating dataset with {len(examples)} examples...")
        
        valid_count = 0
        total_score = 0
        all_issues = []
        all_warnings = []
        
        example_results = []
        
        for i, example in enumerate(examples):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(examples)} ({i/len(examples)*100:.1f}%)")
            
            result = self.validate_example(example)
            example_results.append(result)
            
            if result["valid"]:
                valid_count += 1
            
            total_score += result["score"]
            all_issues.extend(result["issues"])
            all_warnings.extend(result["warnings"])
        
        # Calculate statistics
        validity_rate = valid_count / len(examples)
        average_score = total_score / len(examples)
        
        return {
            "dataset_valid": validity_rate > 0.8,  # 80% validity threshold
            "validity_rate": validity_rate,
            "average_score": average_score,
            "total_examples": len(examples),
            "valid_examples": valid_count,
            "total_issues": len(all_issues),
            "total_warnings": len(all_warnings),
            "example_results": example_results
        }


# Test function
def test_data_validator():
    """Test the data validation system."""
    print("Testing data validation system...")
    
    from data_generator import DataGenerator
    
    # Generate some test examples
    generator = DataGenerator()
    examples = [generator.generate_training_example() for _ in range(5)]
    
    # Test individual validation
    validator = DatasetValidator()
    
    print("\nValidating individual examples:")
    for i, example in enumerate(examples[:3]):
        result = validator.validate_example(example)
        print(f"\nExample {i+1}:")
        print(f"Valid: {result['valid']}, Score: {result['score']:.3f}")
        print(f"Text: {example.text_description}")
        if result['issues']:
            print(f"Issues: {result['issues']}")
        if result['warnings']:
            print(f"Warnings: {result['warnings'][:3]}")  # Show first 3 warnings
    
    # Test dataset validation
    print(f"\nValidating complete dataset...")
    dataset_result = validator.validate_dataset(examples)
    print(f"Dataset valid: {dataset_result['dataset_valid']}")
    print(f"Validity rate: {dataset_result['validity_rate']:.1%}")
    print(f"Average score: {dataset_result['average_score']:.3f}")
    print(f"Total issues: {dataset_result['total_issues']}")
    print(f"Total warnings: {dataset_result['total_warnings']}")
    
    print("\n✅ Data validation system working correctly!")


if __name__ == "__main__":
    test_data_validator()
