"""
Physics Validation System
Validates ML predictions against actual physics simulation outcomes.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

from ml_physics_bridge import MLPhysicsBridge
from realtime_simulator import RealTimeSimulator
from scene_representation import PhysicsScene, ObjectType
from model_architecture import TextToSceneModel


@dataclass
class ValidationResult:
    """Result of physics validation."""
    text_input: str
    prediction_valid: bool
    physics_plausible: bool
    simulation_successful: bool
    validation_score: float
    details: Dict
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'text_input': self.text_input,
            'prediction_valid': self.prediction_valid,
            'physics_plausible': self.physics_plausible,
            'simulation_successful': self.simulation_successful,
            'validation_score': self.validation_score,
            'details': self.details
        }


class PhysicsValidator:
    """Validates ML predictions through physics simulation."""
    
    def __init__(self, bridge: MLPhysicsBridge, simulator: RealTimeSimulator):
        """
        Initialize physics validator.
        
        Args:
            bridge: ML-Physics bridge
            simulator: Real-time simulator
        """
        self.bridge = bridge
        self.simulator = simulator
        
        # Validation thresholds
        self.min_objects = 1
        self.max_objects = 6
        self.min_simulation_time = 1.0
        self.max_simulation_time = 10.0
        
        # Physics plausibility thresholds
        self.max_reasonable_speed = 20.0  # m/s
        self.max_reasonable_displacement = 15.0  # m
        self.min_stability_time = 0.5  # s
    
    def validate_prediction(self, text: str, simulation_duration: float = 3.0) -> ValidationResult:
        """
        Validate an ML prediction through physics simulation.
        
        Args:
            text: Input text description
            simulation_duration: How long to simulate
            
        Returns:
            ValidationResult with comprehensive validation data
        """
        print(f"🔍 Validating: '{text}'")
        
        details = {
            'prediction_time': 0,
            'simulation_time': 0,
            'objects_created': 0,
            'objects_moved': 0,
            'max_speed': 0,
            'max_displacement': 0,
            'stability_achieved': False,
            'errors': []
        }
        
        try:
            # Step 1: Get ML prediction and create physics scene
            start_time = time.time()
            result = self.bridge.predict_and_simulate(text)
            details['prediction_time'] = time.time() - start_time
            details['objects_created'] = result['total_objects']
            
            # Check if prediction is valid
            prediction_valid = self._validate_prediction_structure(result, details)
            
            if not prediction_valid:
                return ValidationResult(
                    text_input=text,
                    prediction_valid=False,
                    physics_plausible=False,
                    simulation_successful=False,
                    validation_score=0.0,
                    details=details
                )
            
            # Step 2: Run physics simulation
            start_time = time.time()
            self.simulator.start_simulation(duration=simulation_duration, record=True)
            
            # Wait for simulation to complete
            while self.simulator.running:
                time.sleep(0.1)
            
            details['simulation_time'] = time.time() - start_time
            simulation_successful = details['simulation_time'] > 0
            
            # Step 3: Analyze simulation results
            motion_analysis = self.simulator.analyze_motion()
            physics_plausible = self._validate_physics_plausibility(motion_analysis, details)
            
            # Step 4: Calculate overall validation score
            validation_score = self._calculate_validation_score(
                prediction_valid, physics_plausible, simulation_successful, details
            )
            
            return ValidationResult(
                text_input=text,
                prediction_valid=prediction_valid,
                physics_plausible=physics_plausible,
                simulation_successful=simulation_successful,
                validation_score=validation_score,
                details=details
            )
            
        except Exception as e:
            details['errors'].append(str(e))
            print(f"❌ Validation error: {e}")
            
            return ValidationResult(
                text_input=text,
                prediction_valid=False,
                physics_plausible=False,
                simulation_successful=False,
                validation_score=0.0,
                details=details
            )
    
    def _validate_prediction_structure(self, result: Dict, details: Dict) -> bool:
        """Validate the structure of ML prediction."""
        try:
            # Check if objects were created
            if result['total_objects'] < self.min_objects:
                details['errors'].append(f"Too few objects: {result['total_objects']}")
                return False
            
            if result['total_objects'] > self.max_objects:
                details['errors'].append(f"Too many objects: {result['total_objects']}")
                return False
            
            # Check prediction time
            if result['prediction_time'] > 5.0:
                details['errors'].append(f"Prediction too slow: {result['prediction_time']:.2f}s")
                return False
            
            # Check if scene exists
            if 'predicted_scene' not in result or result['predicted_scene'] is None:
                details['errors'].append("No predicted scene")
                return False
            
            return True
            
        except Exception as e:
            details['errors'].append(f"Structure validation error: {e}")
            return False
    
    def _validate_physics_plausibility(self, motion_analysis: Dict, details: Dict) -> bool:
        """Validate physics plausibility from simulation results."""
        try:
            if 'error' in motion_analysis:
                details['errors'].append(f"Motion analysis error: {motion_analysis['error']}")
                return False
            
            objects = motion_analysis.get('objects', {})
            if not objects:
                details['errors'].append("No objects to analyze")
                return False
            
            plausible = True
            max_speed = 0
            max_displacement = 0
            objects_moved = 0
            
            for obj_id, obj_data in objects.items():
                # Check speeds
                speed = obj_data.get('max_speed', 0)
                max_speed = max(max_speed, speed)
                
                if speed > self.max_reasonable_speed:
                    details['errors'].append(f"Object {obj_id} too fast: {speed:.2f} m/s")
                    plausible = False
                
                # Check displacement
                displacement = obj_data.get('total_displacement', 0)
                max_displacement = max(max_displacement, displacement)
                
                if displacement > self.max_reasonable_displacement:
                    details['errors'].append(f"Object {obj_id} moved too far: {displacement:.2f} m")
                    plausible = False
                
                # Count objects that moved significantly
                if displacement > 0.1:
                    objects_moved += 1
                
                # Check for NaN or infinite values
                final_pos = obj_data.get('final_position', [0, 0, 0])
                if any(not np.isfinite(x) for x in final_pos):
                    details['errors'].append(f"Object {obj_id} has invalid final position")
                    plausible = False
            
            # Update details
            details['max_speed'] = max_speed
            details['max_displacement'] = max_displacement
            details['objects_moved'] = objects_moved
            
            # Check if simulation makes sense (some objects should move for most scenarios)
            if objects_moved == 0 and len(objects) > 1:
                details['errors'].append("No objects moved in simulation")
                plausible = False
            
            return plausible
            
        except Exception as e:
            details['errors'].append(f"Physics validation error: {e}")
            return False
    
    def _calculate_validation_score(self, prediction_valid: bool, physics_plausible: bool, 
                                  simulation_successful: bool, details: Dict) -> float:
        """Calculate overall validation score (0-1)."""
        score = 0.0
        
        # Base scores
        if prediction_valid:
            score += 0.3
        
        if simulation_successful:
            score += 0.3
        
        if physics_plausible:
            score += 0.4
        
        # Bonus for good performance
        if details['prediction_time'] < 0.1:
            score += 0.05  # Fast prediction
        
        if details['objects_moved'] > 0:
            score += 0.05  # Objects actually moved
        
        # Penalty for errors
        error_penalty = min(0.2, len(details['errors']) * 0.05)
        score -= error_penalty
        
        return max(0.0, min(1.0, score))
    
    def validate_batch(self, texts: List[str], simulation_duration: float = 3.0) -> List[ValidationResult]:
        """
        Validate a batch of text inputs.
        
        Args:
            texts: List of text descriptions
            simulation_duration: Simulation duration for each
            
        Returns:
            List of validation results
        """
        print(f"🔍 Validating batch of {len(texts)} texts...")
        
        results = []
        
        for i, text in enumerate(texts):
            print(f"\nValidating {i+1}/{len(texts)}: '{text}'")
            
            try:
                result = self.validate_prediction(text, simulation_duration)
                results.append(result)
                
                # Brief summary
                print(f"  Score: {result.validation_score:.3f}, "
                      f"Valid: {result.prediction_valid}, "
                      f"Plausible: {result.physics_plausible}")
                
            except Exception as e:
                print(f"  Error: {e}")
                # Create error result
                error_result = ValidationResult(
                    text_input=text,
                    prediction_valid=False,
                    physics_plausible=False,
                    simulation_successful=False,
                    validation_score=0.0,
                    details={'errors': [str(e)]}
                )
                results.append(error_result)
        
        return results
    
    def generate_validation_report(self, results: List[ValidationResult]) -> Dict:
        """Generate a comprehensive validation report."""
        if not results:
            return {'error': 'No results to analyze'}
        
        # Calculate statistics
        total_results = len(results)
        valid_predictions = sum(1 for r in results if r.prediction_valid)
        plausible_physics = sum(1 for r in results if r.physics_plausible)
        successful_simulations = sum(1 for r in results if r.simulation_successful)
        
        scores = [r.validation_score for r in results]
        avg_score = np.mean(scores)
        
        # Collect common errors
        all_errors = []
        for result in results:
            all_errors.extend(result.details.get('errors', []))
        
        error_counts = {}
        for error in all_errors:
            error_counts[error] = error_counts.get(error, 0) + 1
        
        # Performance metrics
        prediction_times = [r.details.get('prediction_time', 0) for r in results]
        simulation_times = [r.details.get('simulation_time', 0) for r in results]
        
        report = {
            'summary': {
                'total_tests': total_results,
                'valid_predictions': valid_predictions,
                'plausible_physics': plausible_physics,
                'successful_simulations': successful_simulations,
                'average_score': avg_score,
                'success_rate': valid_predictions / total_results if total_results > 0 else 0
            },
            'performance': {
                'avg_prediction_time': np.mean(prediction_times),
                'avg_simulation_time': np.mean(simulation_times),
                'max_prediction_time': np.max(prediction_times),
                'max_simulation_time': np.max(simulation_times)
            },
            'common_errors': dict(sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'score_distribution': {
                'min': np.min(scores),
                'max': np.max(scores),
                'std': np.std(scores),
                'median': np.median(scores)
            }
        }
        
        return report


def test_physics_validator():
    """Test the physics validation system."""
    print("Testing Physics Validator...")
    
    # Load model
    model_path = "models/trained_model/final_model.pth"
    
    if os.path.exists(model_path):
        print("Loading trained model...")
        from model_architecture import ModelConfig
        
        config = ModelConfig()
        model = TextToSceneModel(hidden_size=config.hidden_size, max_objects=config.max_objects)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    else:
        print("Using untrained model for testing...")
        from model_architecture import ModelConfig
        
        config = ModelConfig()
        model = TextToSceneModel(hidden_size=config.hidden_size, max_objects=config.max_objects)
    
    # Create components
    bridge = MLPhysicsBridge(model, use_gui=False)  # No GUI for testing
    simulator = RealTimeSimulator(bridge, fps=60)
    validator = PhysicsValidator(bridge, simulator)
    
    # Test examples
    test_texts = [
        "create a ball",
        "add a sphere on a ramp",
        "place two boxes"
    ]
    
    try:
        # Validate batch
        results = validator.validate_batch(test_texts, simulation_duration=2.0)
        
        # Generate report
        report = validator.generate_validation_report(results)
        
        print(f"\n📊 Validation Report:")
        print(f"  Success rate: {report['summary']['success_rate']:.1%}")
        print(f"  Average score: {report['summary']['average_score']:.3f}")
        print(f"  Valid predictions: {report['summary']['valid_predictions']}/{report['summary']['total_tests']}")
        print(f"  Plausible physics: {report['summary']['plausible_physics']}/{report['summary']['total_tests']}")
        
        if report['common_errors']:
            print(f"  Common errors:")
            for error, count in list(report['common_errors'].items())[:3]:
                print(f"    {error}: {count}")
    
    finally:
        bridge.disconnect()
    
    print("\n✅ Physics validator test completed!")


if __name__ == "__main__":
    import os
    import torch
    test_physics_validator()
