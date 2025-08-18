"""
Generalization Tester
Validates that the system can handle novel scenarios not seen in training data.
Tests the limits of the current approach and identifies areas for improvement.
"""

import torch
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from model_architecture import TextToSceneModel, ModelConfig
from ml_physics_bridge import MLPhysicsBridge
from physics_validator import PhysicsValidator
from realtime_simulator import RealTimeSimulator
from relational_understanding import RelationalSceneBuilder
from dynamic_scene_representation import DynamicPhysicsScene


@dataclass
class GeneralizationTest:
    """Represents a generalization test case."""
    test_id: str
    category: str
    description: str
    input_text: str
    expected_behavior: str
    difficulty_level: int  # 1-5, where 5 is most challenging
    novel_aspects: List[str]  # What makes this test novel


class GeneralizationTester:
    """Tests system generalization capabilities."""
    
    def __init__(self, model: TextToSceneModel):
        """Initialize generalization tester."""
        self.model = model
        self.bridge = MLPhysicsBridge(model, use_gui=False)
        self.simulator = RealTimeSimulator(self.bridge, fps=60)
        self.validator = PhysicsValidator(self.bridge, self.simulator)
        self.relational_builder = RelationalSceneBuilder()
        
        # Test categories
        self.test_categories = {
            'novel_objects': 'Objects not in training data',
            'complex_relationships': 'Spatial relationships beyond training',
            'unusual_grammar': 'Non-standard sentence structures',
            'ambiguous_language': 'Ambiguous or unclear instructions',
            'multi_step_commands': 'Commands requiring multiple actions',
            'physics_edge_cases': 'Unusual physics scenarios',
            'scale_variations': 'Different numbers of objects than trained',
            'material_combinations': 'Novel material combinations'
        }
    
    def create_test_suite(self) -> List[GeneralizationTest]:
        """Create comprehensive generalization test suite."""
        tests = []
        
        # Novel objects (not in training)
        tests.extend([
            GeneralizationTest(
                test_id="novel_obj_1",
                category="novel_objects",
                description="Objects not in training vocabulary",
                input_text="create a pyramid on a platform",
                expected_behavior="Should handle unknown objects gracefully",
                difficulty_level=3,
                novel_aspects=["pyramid", "platform"]
            ),
            GeneralizationTest(
                test_id="novel_obj_2",
                category="novel_objects",
                description="Scientific equipment",
                input_text="place a beaker next to a microscope",
                expected_behavior="Should create reasonable substitutes",
                difficulty_level=4,
                novel_aspects=["beaker", "microscope"]
            )
        ])
        
        # Complex spatial relationships
        tests.extend([
            GeneralizationTest(
                test_id="complex_rel_1",
                category="complex_relationships",
                description="Three-way spatial relationship",
                input_text="put the ball between the box and the ramp, but closer to the box",
                expected_behavior="Should understand relative positioning",
                difficulty_level=4,
                novel_aspects=["relative distance", "three-way relationship"]
            ),
            GeneralizationTest(
                test_id="complex_rel_2",
                category="complex_relationships",
                description="Nested spatial relationships",
                input_text="place a small ball inside a large box that is on top of a ramp",
                expected_behavior="Should handle nested containment",
                difficulty_level=5,
                novel_aspects=["containment", "nested relationships", "size modifiers"]
            )
        ])
        
        # Unusual grammar
        tests.extend([
            GeneralizationTest(
                test_id="grammar_1",
                category="unusual_grammar",
                description="Passive voice",
                input_text="a ball should be placed above the box by the system",
                expected_behavior="Should understand passive construction",
                difficulty_level=3,
                novel_aspects=["passive voice", "formal language"]
            ),
            GeneralizationTest(
                test_id="grammar_2",
                category="unusual_grammar",
                description="Question format",
                input_text="can you put a sphere where it would roll down the ramp?",
                expected_behavior="Should interpret question as command",
                difficulty_level=4,
                novel_aspects=["question format", "implied positioning"]
            )
        ])
        
        # Ambiguous language
        tests.extend([
            GeneralizationTest(
                test_id="ambiguous_1",
                category="ambiguous_language",
                description="Unclear reference",
                input_text="put it next to the thing",
                expected_behavior="Should handle ambiguous references",
                difficulty_level=5,
                novel_aspects=["pronoun reference", "vague nouns"]
            ),
            GeneralizationTest(
                test_id="ambiguous_2",
                category="ambiguous_language",
                description="Multiple interpretations",
                input_text="place the ball by the box",
                expected_behavior="Should choose reasonable interpretation",
                difficulty_level=2,
                novel_aspects=["spatial ambiguity"]
            )
        ])
        
        # Multi-step commands
        tests.extend([
            GeneralizationTest(
                test_id="multi_step_1",
                category="multi_step_commands",
                description="Sequential actions",
                input_text="first create a ramp, then place a ball on top, and finally add a box below",
                expected_behavior="Should handle sequence of actions",
                difficulty_level=4,
                novel_aspects=["temporal sequence", "multiple objects", "ordering"]
            ),
            GeneralizationTest(
                test_id="multi_step_2",
                category="multi_step_commands",
                description="Conditional actions",
                input_text="if there's space, add another ball next to the first one",
                expected_behavior="Should handle conditional logic",
                difficulty_level=5,
                novel_aspects=["conditional logic", "spatial reasoning"]
            )
        ])
        
        # Physics edge cases
        tests.extend([
            GeneralizationTest(
                test_id="physics_1",
                category="physics_edge_cases",
                description="Impossible physics",
                input_text="make a ball float in mid-air without support",
                expected_behavior="Should handle impossible requests gracefully",
                difficulty_level=3,
                novel_aspects=["impossible physics", "constraint violation"]
            ),
            GeneralizationTest(
                test_id="physics_2",
                category="physics_edge_cases",
                description="Extreme scales",
                input_text="create a tiny ball on a massive ramp",
                expected_behavior="Should handle scale variations",
                difficulty_level=3,
                novel_aspects=["scale modifiers", "relative sizing"]
            )
        ])
        
        # Scale variations
        tests.extend([
            GeneralizationTest(
                test_id="scale_1",
                category="scale_variations",
                description="Many objects",
                input_text="create ten balls in a line",
                expected_behavior="Should handle large numbers",
                difficulty_level=4,
                novel_aspects=["large quantity", "arrangement pattern"]
            ),
            GeneralizationTest(
                test_id="scale_2",
                category="scale_variations",
                description="Complex arrangement",
                input_text="make a pyramid of boxes with spheres on each level",
                expected_behavior="Should handle complex structures",
                difficulty_level=5,
                novel_aspects=["hierarchical structure", "pattern recognition"]
            )
        ])
        
        return tests
    
    def run_generalization_test(self, test: GeneralizationTest) -> Dict[str, Any]:
        """Run a single generalization test."""
        print(f"🧪 Running test {test.test_id}: {test.description}")
        
        result = {
            'test_id': test.test_id,
            'category': test.category,
            'input_text': test.input_text,
            'success': False,
            'ml_prediction_success': False,
            'relational_parsing_success': False,
            'physics_simulation_success': False,
            'validation_score': 0.0,
            'errors': [],
            'novel_aspects_handled': [],
            'execution_time': 0.0
        }
        
        start_time = time.time()
        
        try:
            # Test 1: ML Model Prediction
            try:
                ml_result = self.bridge.predict_and_simulate(test.input_text)
                result['ml_prediction_success'] = ml_result['total_objects'] > 0
                if result['ml_prediction_success']:
                    result['novel_aspects_handled'].append('ml_prediction')
            except Exception as e:
                result['errors'].append(f"ML prediction failed: {str(e)}")
            
            # Test 2: Relational Understanding
            try:
                relational_scene = self.relational_builder.build_scene_from_text(test.input_text)
                result['relational_parsing_success'] = relational_scene.get_object_count() > 0
                if result['relational_parsing_success']:
                    result['novel_aspects_handled'].append('relational_parsing')
            except Exception as e:
                result['errors'].append(f"Relational parsing failed: {str(e)}")
            
            # Test 3: Physics Simulation (if ML prediction succeeded)
            if result['ml_prediction_success']:
                try:
                    sim_result = self.bridge.run_simulation(duration=2.0, real_time=False)
                    result['physics_simulation_success'] = 'error' not in sim_result
                    if result['physics_simulation_success']:
                        result['novel_aspects_handled'].append('physics_simulation')
                except Exception as e:
                    result['errors'].append(f"Physics simulation failed: {str(e)}")
            
            # Test 4: Validation (if simulation succeeded)
            if result['physics_simulation_success']:
                try:
                    validation_result = self.validator.validate_prediction(test.input_text, simulation_duration=1.0)
                    result['validation_score'] = validation_result.validation_score
                    if result['validation_score'] > 0.5:
                        result['novel_aspects_handled'].append('validation')
                except Exception as e:
                    result['errors'].append(f"Validation failed: {str(e)}")
            
            # Overall success criteria
            result['success'] = (
                result['ml_prediction_success'] or result['relational_parsing_success']
            ) and len(result['errors']) == 0
            
        except Exception as e:
            result['errors'].append(f"Test execution failed: {str(e)}")
        
        result['execution_time'] = time.time() - start_time
        
        # Log result
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"   {status} - {len(result['novel_aspects_handled'])}/{len(test.novel_aspects)} aspects handled")
        
        return result
    
    def run_test_suite(self, test_suite: List[GeneralizationTest] = None) -> Dict[str, Any]:
        """Run the complete generalization test suite."""
        if test_suite is None:
            test_suite = self.create_test_suite()
        
        print(f"🚀 Running generalization test suite with {len(test_suite)} tests...")
        
        results = {
            'total_tests': len(test_suite),
            'tests_passed': 0,
            'tests_failed': 0,
            'category_results': {},
            'difficulty_analysis': {},
            'novel_aspects_analysis': {},
            'individual_results': [],
            'overall_score': 0.0,
            'execution_time': 0.0
        }
        
        start_time = time.time()
        
        # Run each test
        for test in test_suite:
            test_result = self.run_generalization_test(test)
            results['individual_results'].append(test_result)
            
            # Update counters
            if test_result['success']:
                results['tests_passed'] += 1
            else:
                results['tests_failed'] += 1
            
            # Update category analysis
            category = test.category
            if category not in results['category_results']:
                results['category_results'][category] = {'passed': 0, 'total': 0}
            
            results['category_results'][category]['total'] += 1
            if test_result['success']:
                results['category_results'][category]['passed'] += 1
            
            # Update difficulty analysis
            difficulty = test.difficulty_level
            if difficulty not in results['difficulty_analysis']:
                results['difficulty_analysis'][difficulty] = {'passed': 0, 'total': 0}
            
            results['difficulty_analysis'][difficulty]['total'] += 1
            if test_result['success']:
                results['difficulty_analysis'][difficulty]['passed'] += 1
            
            # Update novel aspects analysis
            for aspect in test.novel_aspects:
                if aspect not in results['novel_aspects_analysis']:
                    results['novel_aspects_analysis'][aspect] = {'handled': 0, 'total': 0}
                
                results['novel_aspects_analysis'][aspect]['total'] += 1
                if aspect in test_result.get('novel_aspects_handled', []):
                    results['novel_aspects_analysis'][aspect]['handled'] += 1
        
        results['execution_time'] = time.time() - start_time
        results['overall_score'] = results['tests_passed'] / results['total_tests'] if results['total_tests'] > 0 else 0
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive generalization report."""
        report = []
        report.append("🧪 GENERALIZATION TEST REPORT")
        report.append("=" * 50)
        
        # Overall results
        report.append(f"\nOverall Results:")
        report.append(f"  Tests Passed: {results['tests_passed']}/{results['total_tests']} ({results['overall_score']:.1%})")
        report.append(f"  Execution Time: {results['execution_time']:.2f}s")
        
        # Category breakdown
        report.append(f"\nResults by Category:")
        for category, stats in results['category_results'].items():
            success_rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
            report.append(f"  {category}: {stats['passed']}/{stats['total']} ({success_rate:.1%})")
        
        # Difficulty analysis
        report.append(f"\nResults by Difficulty Level:")
        for difficulty in sorted(results['difficulty_analysis'].keys()):
            stats = results['difficulty_analysis'][difficulty]
            success_rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
            report.append(f"  Level {difficulty}: {stats['passed']}/{stats['total']} ({success_rate:.1%})")
        
        # Novel aspects analysis
        report.append(f"\nNovel Aspects Handling:")
        for aspect, stats in results['novel_aspects_analysis'].items():
            success_rate = stats['handled'] / stats['total'] if stats['total'] > 0 else 0
            report.append(f"  {aspect}: {stats['handled']}/{stats['total']} ({success_rate:.1%})")
        
        # Recommendations
        report.append(f"\nRecommendations:")
        
        # Find weakest categories
        weak_categories = []
        for category, stats in results['category_results'].items():
            success_rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
            if success_rate < 0.5:
                weak_categories.append(category)
        
        if weak_categories:
            report.append(f"  Priority areas for improvement:")
            for category in weak_categories:
                report.append(f"    - {category}")
        else:
            report.append(f"  ✅ Strong performance across all categories!")
        
        return "\n".join(report)
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.simulator:
                self.simulator.stop_simulation()
            if self.bridge:
                self.bridge.disconnect()
        except:
            pass


def test_generalization_capabilities():
    """Test the generalization testing system."""
    print("Testing Generalization Capabilities...")
    
    # Load model
    model_path = "models/trained_model/final_model.pth"
    
    if os.path.exists(model_path):
        print("Loading trained model...")
        config = ModelConfig()
        model = TextToSceneModel(hidden_size=config.hidden_size, max_objects=config.max_objects)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    else:
        print("Using untrained model for testing...")
        config = ModelConfig()
        model = TextToSceneModel(hidden_size=config.hidden_size, max_objects=config.max_objects)
    
    # Create tester
    tester = GeneralizationTester(model)
    
    try:
        # Create a small test suite for demonstration
        demo_tests = [
            GeneralizationTest(
                test_id="demo_1",
                category="novel_objects",
                description="Novel object test",
                input_text="create a pyramid",
                expected_behavior="Should handle gracefully",
                difficulty_level=3,
                novel_aspects=["pyramid"]
            ),
            GeneralizationTest(
                test_id="demo_2",
                category="complex_relationships",
                description="Complex spatial relationship",
                input_text="put the ball between the two boxes",
                expected_behavior="Should understand three-way relationship",
                difficulty_level=4,
                novel_aspects=["three-way relationship"]
            )
        ]
        
        # Run tests
        results = tester.run_test_suite(demo_tests)
        
        # Generate and print report
        report = tester.generate_report(results)
        print(f"\n{report}")
        
        # Save results
        with open('data/generalization_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📁 Detailed results saved to: data/generalization_test_results.json")
    
    finally:
        tester.cleanup()
    
    print("✅ Generalization testing completed!")


if __name__ == "__main__":
    import os
    import torch
    test_generalization_capabilities()
