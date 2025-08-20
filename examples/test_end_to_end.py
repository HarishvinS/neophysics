"""
End-to-End Pipeline Test
Validates the complete text → ML → physics → visualization pipeline.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import time
import json
from typing import List, Dict

from model_architecture import TextToSceneModel, ModelConfig
from ml_physics_bridge import MLPhysicsBridge
from realtime_simulator import RealTimeSimulator
from physics_validator import PhysicsValidator


class EndToEndTester:
    """Comprehensive end-to-end pipeline tester."""
    
    def __init__(self):
        """Initialize the tester."""
        self.model = None
        self.bridge = None
        self.simulator = None
        self.validator = None
        
        self.test_results = []
        self.overall_success = True
    
    def setup_system(self) -> bool:
        """Setup the complete system."""
        print("🔧 Setting up end-to-end system...")
        
        try:
            # Load model
            model_path = "models/trained_model/final_model.pth"
            
            if os.path.exists(model_path):
                print("📦 Loading trained model...")
                
                config = ModelConfig()
                config.model_path = model_path
                self.model = TextToSceneModel(config=config)
                
                print("✅ Trained model loaded")
            else:
                print("⚠️ No trained model found, using untrained model")
                
                config = ModelConfig()
                self.model = TextToSceneModel(config=config)
            
            # Initialize components
            print("🔗 Initializing ML-Physics bridge...")
            self.bridge = MLPhysicsBridge(self.model, use_gui=False)  # No GUI for testing
            
            print("⚡ Setting up real-time simulator...")
            self.simulator = RealTimeSimulator(self.bridge, fps=60)
            
            print("🔍 Initializing physics validator...")
            self.validator = PhysicsValidator(self.bridge, self.simulator)
            
            print("✅ System setup complete!")
            return True
            
        except Exception as e:
            print(f"❌ System setup failed: {e}")
            return False
    
    def test_basic_pipeline(self) -> Dict:
        """Test basic text-to-physics pipeline."""
        print("\n🧪 Testing Basic Pipeline...")
        
        test_commands = [
            "create a ball",
            "add a sphere on a ramp",
            "place two boxes"
        ]
        
        results = {
            'test_name': 'Basic Pipeline',
            'commands_tested': len(test_commands),
            'successful_predictions': 0,
            'successful_simulations': 0,
            'total_time': 0,
            'details': []
        }
        
        start_time = time.time()
        
        for i, command in enumerate(test_commands):
            print(f"  Testing {i+1}/{len(test_commands)}: '{command}'")
            
            try:
                # Test prediction and physics creation
                result = self.bridge.predict_and_simulate(command)
                
                if result['total_objects'] > 0:
                    results['successful_predictions'] += 1
                    
                    # Test simulation
                    sim_result = self.bridge.run_simulation(duration=2.0, real_time=False)
                    
                    if 'error' not in sim_result:
                        results['successful_simulations'] += 1
                
                results['details'].append({
                    'command': command,
                    'objects_created': result['total_objects'],
                    'prediction_time': result['prediction_time'],
                    'success': result['total_objects'] > 0
                })
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                results['details'].append({
                    'command': command,
                    'error': str(e),
                    'success': False
                })
        
        results['total_time'] = time.time() - start_time
        
        print(f"  Results: {results['successful_predictions']}/{results['commands_tested']} predictions successful")
        print(f"  Simulations: {results['successful_simulations']}/{results['commands_tested']} successful")
        
        return results
    
    def test_validation_system(self) -> Dict:
        """Test the physics validation system."""
        print("\n🔍 Testing Validation System...")
        
        test_commands = [
            "create a ball",
            "add a ramp and sphere",
            "place three boxes"
        ]
        
        results = {
            'test_name': 'Validation System',
            'commands_tested': len(test_commands),
            'validations_completed': 0,
            'average_score': 0,
            'total_time': 0,
            'details': []
        }
        
        start_time = time.time()
        scores = []
        
        for i, command in enumerate(test_commands):
            print(f"  Validating {i+1}/{len(test_commands)}: '{command}'")
            
            try:
                validation_result = self.validator.validate_prediction(command, simulation_duration=2.0)
                
                results['validations_completed'] += 1
                scores.append(validation_result.validation_score)
                
                results['details'].append({
                    'command': command,
                    'score': validation_result.validation_score,
                    'prediction_valid': validation_result.prediction_valid,
                    'physics_plausible': validation_result.physics_plausible,
                    'simulation_successful': validation_result.simulation_successful
                })
                
                print(f"    Score: {validation_result.validation_score:.3f}")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                results['details'].append({
                    'command': command,
                    'error': str(e)
                })
        
        results['total_time'] = time.time() - start_time
        results['average_score'] = sum(scores) / len(scores) if scores else 0
        
        print(f"  Results: {results['validations_completed']}/{results['commands_tested']} validations completed")
        print(f"  Average score: {results['average_score']:.3f}")
        
        return results
    
    def test_real_time_simulation(self) -> Dict:
        """Test real-time simulation capabilities."""
        print("\n⚡ Testing Real-time Simulation...")
        
        results = {
            'test_name': 'Real-time Simulation',
            'simulation_started': False,
            'simulation_completed': False,
            'frames_recorded': 0,
            'motion_analysis_successful': False,
            'total_time': 0,
            'details': {}
        }
        
        start_time = time.time()
        
        try:
            # Create a scene
            print("  Creating physics scene...")
            result = self.bridge.predict_and_simulate("create a ball on a ramp")
            
            if result['total_objects'] > 0:
                print("  Starting real-time simulation...")
                
                # Add event callback to track completion
                simulation_completed = False
                
                def on_event(event_type, data):
                    nonlocal simulation_completed
                    if event_type == "simulation_stopped":
                        simulation_completed = True
                
                self.simulator.add_event_callback(on_event)
                
                # Start simulation
                self.simulator.start_simulation(duration=3.0, record=True)
                results['simulation_started'] = True
                
                # Wait for completion
                timeout = 10.0
                wait_start = time.time()
                
                while self.simulator.running and (time.time() - wait_start) < timeout:
                    time.sleep(0.1)
                
                results['simulation_completed'] = simulation_completed
                
                # Analyze results
                if simulation_completed:
                    print("  Analyzing motion...")
                    
                    stats = self.simulator.get_simulation_stats()
                    results['frames_recorded'] = stats.get('frames_recorded', 0)
                    
                    motion_analysis = self.simulator.analyze_motion()
                    if 'objects' in motion_analysis:
                        results['motion_analysis_successful'] = True
                        results['details']['objects_analyzed'] = len(motion_analysis['objects'])
                    
                    print(f"    Frames recorded: {results['frames_recorded']}")
                    print(f"    Motion analysis: {'✅' if results['motion_analysis_successful'] else '❌'}")
            
            else:
                print("  ❌ No objects created, skipping simulation")
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results['details']['error'] = str(e)
        
        results['total_time'] = time.time() - start_time
        
        return results
    
    def test_performance_metrics(self) -> Dict:
        """Test performance and timing metrics."""
        print("\n📊 Testing Performance Metrics...")
        
        results = {
            'test_name': 'Performance Metrics',
            'prediction_times': [],
            'simulation_times': [],
            'total_objects_created': 0,
            'average_prediction_time': 0,
            'average_simulation_time': 0,
            'throughput': 0  # commands per second
        }
        
        test_commands = [
            "create a ball",
            "add a box",
            "place a sphere",
            "create two boxes",
            "add a ramp"
        ]
        
        start_time = time.time()
        
        for command in test_commands:
            try:
                # Measure prediction time
                pred_start = time.time()
                result = self.bridge.predict_and_simulate(command)
                pred_time = time.time() - pred_start
                
                results['prediction_times'].append(pred_time)
                results['total_objects_created'] += result['total_objects']
                
                # Measure simulation time
                sim_start = time.time()
                self.bridge.run_simulation(duration=1.0, real_time=False)
                sim_time = time.time() - sim_start
                
                results['simulation_times'].append(sim_time)
                
            except Exception as e:
                print(f"  ⚠️ Error with '{command}': {e}")
        
        total_time = time.time() - start_time
        
        # Calculate averages
        if results['prediction_times']:
            results['average_prediction_time'] = sum(results['prediction_times']) / len(results['prediction_times'])
        
        if results['simulation_times']:
            results['average_simulation_time'] = sum(results['simulation_times']) / len(results['simulation_times'])
        
        results['throughput'] = len(test_commands) / total_time if total_time > 0 else 0
        
        print(f"  Average prediction time: {results['average_prediction_time']:.3f}s")
        print(f"  Average simulation time: {results['average_simulation_time']:.3f}s")
        print(f"  Throughput: {results['throughput']:.2f} commands/second")
        print(f"  Total objects created: {results['total_objects_created']}")
        
        return results
    
    def run_comprehensive_test(self) -> Dict:
        """Run all tests and generate comprehensive report."""
        print("🚀 Starting Comprehensive End-to-End Test")
        print("=" * 60)
        
        if not self.setup_system():
            return {'error': 'System setup failed'}
        
        # Run all tests
        test_results = []
        
        test_results.append(self.test_basic_pipeline())
        test_results.append(self.test_validation_system())
        test_results.append(self.test_real_time_simulation())
        test_results.append(self.test_performance_metrics())
        
        # Generate overall report
        report = {
            'timestamp': time.time(),
            'system_info': {
                'model_loaded': self.model is not None,
                'bridge_initialized': self.bridge is not None,
                'simulator_ready': self.simulator is not None,
                'validator_ready': self.validator is not None
            },
            'test_results': test_results,
            'summary': self._generate_summary(test_results)
        }
        
        return report
    
    def _generate_summary(self, test_results: List[Dict]) -> Dict:
        """Generate summary of all test results."""
        summary = {
            'total_tests': len(test_results),
            'tests_passed': 0,
            'overall_success': True,
            'key_metrics': {}
        }
        
        for result in test_results:
            test_name = result['test_name']
            
            # Determine if test passed
            if test_name == 'Basic Pipeline':
                passed = result['successful_predictions'] > 0 and result['successful_simulations'] > 0
            elif test_name == 'Validation System':
                passed = result['validations_completed'] > 0 and result['average_score'] > 0.3
            elif test_name == 'Real-time Simulation':
                passed = result['simulation_started'] and result['simulation_completed']
            elif test_name == 'Performance Metrics':
                passed = result['average_prediction_time'] < 1.0 and result['throughput'] > 0.5
            else:
                passed = True
            
            if passed:
                summary['tests_passed'] += 1
            else:
                summary['overall_success'] = False
            
            # Extract key metrics
            if test_name == 'Performance Metrics':
                summary['key_metrics']['prediction_time'] = result['average_prediction_time']
                summary['key_metrics']['throughput'] = result['throughput']
            elif test_name == 'Validation System':
                summary['key_metrics']['validation_score'] = result['average_score']
        
        return summary
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            if self.simulator:
                self.simulator.stop_simulation()
            
            if self.bridge:
                self.bridge.disconnect()
        except:
            pass


def main():
    """Main test function."""
    tester = EndToEndTester()
    
    try:
        # Run comprehensive test
        report = tester.run_comprehensive_test()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        
        if 'error' in report:
            print(f"❌ Test failed: {report['error']}")
            return
        
        summary = report['summary']
        
        print(f"Overall Success: {'✅ PASS' if summary['overall_success'] else '❌ FAIL'}")
        print(f"Tests Passed: {summary['tests_passed']}/{summary['total_tests']}")
        
        if 'key_metrics' in summary:
            metrics = summary['key_metrics']
            print(f"\nKey Performance Metrics:")
            if 'prediction_time' in metrics:
                print(f"  Average Prediction Time: {metrics['prediction_time']:.3f}s")
            if 'throughput' in metrics:
                print(f"  Throughput: {metrics['throughput']:.2f} commands/second")
            if 'validation_score' in metrics:
                print(f"  Validation Score: {metrics['validation_score']:.3f}")
        
        print(f"\nSystem Status:")
        sys_info = report['system_info']
        print(f"  Model Loaded: {'✅' if sys_info['model_loaded'] else '❌'}")
        print(f"  Bridge Ready: {'✅' if sys_info['bridge_initialized'] else '❌'}")
        print(f"  Simulator Ready: {'✅' if sys_info['simulator_ready'] else '❌'}")
        print(f"  Validator Ready: {'✅' if sys_info['validator_ready'] else '❌'}")
        
        # Save detailed report
        with open('data/end_to_end_test_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📁 Detailed report saved to: data/end_to_end_test_report.json")
        
        if summary['overall_success']:
            print("\n🎉 All tests passed! The end-to-end pipeline is working correctly.")
        else:
            print("\n⚠️ Some tests failed. Check the detailed report for more information.")
    
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
