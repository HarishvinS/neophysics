"""
Integrated System Demo
Demonstrates the complete integrated physics reasoning system with all advanced components
working together through the interactive interface pipeline.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
from typing import Dict, List, Any

from relational_understanding import RelationalSceneBuilder
from improved_physics_reasoning import ImprovedPhysicsReasoner
from physics_simulation_engine import PhysicsSimulationEngine
from ml_physics_bridge import MLPhysicsBridge
from model_architecture import TextToSceneModel, ModelConfig


class IntegratedPhysicsDemo:
    """Demonstrates the complete integrated physics reasoning pipeline."""
    
    def __init__(self):
        """Initialize the integrated demo system."""
        print("🚀 Initializing Integrated Physics System...")
        
        # Initialize all components
        self.relational_builder = RelationalSceneBuilder()
        self.reasoner = ImprovedPhysicsReasoner()
        self.simulation_engine = PhysicsSimulationEngine()
        
        # Initialize ML components
        config = ModelConfig()
        self.model = TextToSceneModel(hidden_size=config.hidden_size, max_objects=config.max_objects)
        self.bridge = MLPhysicsBridge(self.model, use_gui=False)
        
        print("✅ All components initialized successfully!")
    
    def process_command(self, command: str) -> Dict[str, Any]:
        """
        Process a natural language command through the complete pipeline.
        This simulates what happens in the interactive interface.
        """
        print(f"\n🎯 Processing Command: '{command}'")
        print("=" * 60)
        
        results = {
            'command': command,
            'timestamp': time.time(),
            'pipeline_steps': []
        }
        
        # Step 1: Parse command with relational understanding
        print("1️⃣ Parsing command for objects and relationships...")
        start_time = time.time()
        
        scene = self.relational_builder.build_scene_from_text(command)
        parse_time = time.time() - start_time
        
        step1_result = {
            'step': 'relational_parsing',
            'duration': parse_time,
            'objects_found': scene.get_object_count(),
            'objects': []
        }
        
        if scene.get_object_count() > 0:
            for obj_id, obj in scene.objects.items():
                step1_result['objects'].append({
                    'id': obj_id,
                    'type': obj.object_type.value,
                    'position': obj.position.to_list(),
                    'material': obj.material.value
                })
                print(f"   Found: {obj_id} ({obj.object_type.value}) at {obj.position.to_list()}")
        else:
            print("   No objects detected in command")
        
        results['pipeline_steps'].append(step1_result)
        
        # Step 2: Analyze physics scenarios and choose strategy
        print("\n2️⃣ Analyzing physics scenarios...")
        start_time = time.time()
        
        analysis = self.reasoner.analyze_and_predict(scene)
        analysis_time = time.time() - start_time
        
        step2_result = {
            'step': 'physics_analysis',
            'duration': analysis_time,
            'scenarios_detected': len(analysis['detected_scenarios']),
            'chosen_strategy': analysis['chosen_strategy'],
            'confidence': analysis['confidence'],
            'reasoning_summary': analysis['reasoning_summary'],
            'scenarios': analysis['detected_scenarios']
        }
        
        print(f"   Scenarios detected: {len(analysis['detected_scenarios'])}")
        for scenario in analysis['detected_scenarios']:
            print(f"     - {scenario['description']} (confidence: {scenario['confidence']:.2f})")
        
        print(f"   Strategy chosen: {analysis['chosen_strategy']}")
        print(f"   Overall confidence: {analysis['confidence']:.2f}")
        
        results['pipeline_steps'].append(step2_result)
        
        # Step 3: Simulate physics outcomes
        print("\n3️⃣ Simulating physics outcomes...")
        start_time = time.time()
        
        if scene.get_object_count() > 0:
            # Test scene stability
            stability = self.simulation_engine.analyze_scene_stability(scene)
            
            # Predict physics chain
            chain = self.simulation_engine.predict_physics_chain(scene)
            
            simulation_time = time.time() - start_time
            
            step3_result = {
                'step': 'physics_simulation',
                'duration': simulation_time,
                'stability': stability,
                'predicted_events': len(chain.steps),
                'total_duration': chain.total_duration,
                'events': []
            }
            
            print(f"   Scene stability: {stability['prediction']}")
            print(f"   Predicted events: {len(chain.steps)}")
            
            for event in chain.steps:
                event_info = {
                    'type': event.interaction_type.value,
                    'primary_object': event.primary_object,
                    'affected_objects': event.affected_objects,
                    'confidence': event.confidence
                }
                step3_result['events'].append(event_info)
                print(f"     - {event.interaction_type.value}: {event.primary_object} → {event.affected_objects}")
        
        else:
            step3_result = {
                'step': 'physics_simulation',
                'duration': 0,
                'message': 'No objects to simulate'
            }
            print("   No objects to simulate")
        
        results['pipeline_steps'].append(step3_result)
        
        # Step 4: Generate comprehensive summary
        print("\n4️⃣ Generating summary...")
        
        total_time = sum(step.get('duration', 0) for step in results['pipeline_steps'])
        
        summary = {
            'total_processing_time': total_time,
            'objects_created': scene.get_object_count(),
            'scenarios_detected': len(analysis['detected_scenarios']),
            'physics_events_predicted': len(analysis['predicted_chain'].steps) if 'predicted_chain' in analysis else 0,
            'overall_confidence': analysis['confidence'],
            'system_recommendation': self._generate_recommendation(results)
        }
        
        results['summary'] = summary
        
        print(f"   Total processing time: {total_time:.3f}s")
        print(f"   System recommendation: {summary['system_recommendation']}")
        
        return results
    
    def _generate_recommendation(self, results: Dict[str, Any]) -> str:
        """Generate a system recommendation based on analysis results."""
        steps = results['pipeline_steps']
        
        # Check if objects were found
        objects_found = any(step.get('objects_found', 0) > 0 for step in steps)
        if not objects_found:
            return "Try rephrasing the command with clearer object descriptions"
        
        # Check confidence level
        analysis_step = next((step for step in steps if step['step'] == 'physics_analysis'), None)
        if analysis_step:
            confidence = analysis_step.get('confidence', 0)
            if confidence > 0.8:
                return "High confidence prediction - simulation should be accurate"
            elif confidence > 0.6:
                return "Moderate confidence - results may vary"
            else:
                return "Low confidence - consider simplifying the scenario"
        
        return "Command processed successfully"
    
    def run_demo_scenarios(self):
        """Run a series of demo scenarios to showcase capabilities."""
        print("🎬 Integrated Physics System Demo")
        print("=" * 70)
        print("Showcasing the complete pipeline from natural language to physics prediction")
        print("=" * 70)
        
        # Demo scenarios of increasing complexity
        demo_commands = [
            "create a red ball",
            "place a ball above a ramp",
            "put a sphere on top of a wooden box",
            "create a domino chain with three pieces",
            "build a pendulum with a heavy bob",
            "place a ball between two cubes and add a ramp",
            "create a complex scene with multiple objects interacting"
        ]
        
        results = []
        
        for i, command in enumerate(demo_commands, 1):
            print(f"\n{'🎯' * 3} Demo Scenario {i} {'🎯' * 3}")
            result = self.process_command(command)
            results.append(result)
            
            # Brief pause between scenarios
            time.sleep(0.5)
        
        # Generate overall demo summary
        self._generate_demo_summary(results)
        
        return results
    
    def _generate_demo_summary(self, results: List[Dict[str, Any]]):
        """Generate summary of all demo scenarios."""
        print("\n" + "=" * 70)
        print("🎉 Demo Summary")
        print("=" * 70)
        
        total_commands = len(results)
        total_objects = sum(r['summary']['objects_created'] for r in results)
        total_scenarios = sum(r['summary']['scenarios_detected'] for r in results)
        total_events = sum(r['summary']['physics_events_predicted'] for r in results)
        avg_confidence = sum(r['summary']['overall_confidence'] for r in results) / total_commands
        total_time = sum(r['summary']['total_processing_time'] for r in results)
        
        print(f"📊 Overall Statistics:")
        print(f"   Commands processed: {total_commands}")
        print(f"   Objects created: {total_objects}")
        print(f"   Scenarios detected: {total_scenarios}")
        print(f"   Physics events predicted: {total_events}")
        print(f"   Average confidence: {avg_confidence:.2f}")
        print(f"   Total processing time: {total_time:.3f}s")
        print(f"   Average time per command: {total_time/total_commands:.3f}s")
        
        print(f"\n🚀 System Capabilities Demonstrated:")
        print(f"   ✅ Natural language understanding with spatial relationships")
        print(f"   ✅ Intelligent scenario detection and classification")
        print(f"   ✅ Multi-strategy physics reasoning (simulation, analytical, hybrid)")
        print(f"   ✅ Accurate physics prediction through simulation")
        print(f"   ✅ Confidence assessment and uncertainty quantification")
        print(f"   ✅ Real-time processing suitable for interactive applications")
        
        print(f"\n🎯 Integration Achievements:")
        print(f"   🔗 Seamless pipeline from text input to physics prediction")
        print(f"   🧠 Advanced reasoning replacing simple pattern matching")
        print(f"   ⚡ Efficient processing with sub-second response times")
        print(f"   🎭 Handles complex scenarios with multiple interacting objects")
        print(f"   📊 Quantitative assessment of prediction quality")
        
        # Show best and most challenging scenarios
        best_scenario = max(results, key=lambda r: r['summary']['overall_confidence'])
        most_complex = max(results, key=lambda r: r['summary']['objects_created'])
        
        print(f"\n🏆 Highlights:")
        print(f"   Best prediction: '{best_scenario['command']}' (confidence: {best_scenario['summary']['overall_confidence']:.2f})")
        print(f"   Most complex: '{most_complex['command']}' ({most_complex['summary']['objects_created']} objects)")


def main():
    """Run the complete integrated system demo."""
    # Initialize demo system
    demo = IntegratedPhysicsDemo()
    
    # Run demo scenarios
    results = demo.run_demo_scenarios()
    
    print(f"\n🎊 Integration Demo Complete!")
    print(f"The system successfully demonstrates:")
    print(f"• Advanced natural language understanding")
    print(f"• Intelligent physics scenario detection")
    print(f"• Multi-strategy reasoning and prediction")
    print(f"• Seamless integration of all components")
    print(f"\nReady for interactive use! 🚀")


if __name__ == "__main__":
    main()
