"""
Continuous Learning Tester
Validates that the system actually improves over time through real usage.
Tests the complete learning loop with simulated user interactions.
"""

import torch
import numpy as np
import time
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt

from ml_physics_bridge import MLPhysicsBridge
from physics_validator import PhysicsValidator
from model_architecture import TextToSceneModel, ModelConfig
from self_improvement_loop import SelfImprovementLoop, ImprovementPlan
from user_feedback_system import UserFeedback, FeedbackType
from dynamic_scene_representation import DynamicPhysicsScene
from performance_tracking import PerformanceTracker, MetricType


@dataclass
class LearningExperiment:
    """Represents a continuous learning experiment."""
    experiment_id: str
    description: str
    duration_hours: float
    user_interactions: int
    baseline_performance: Dict[str, float]
    final_performance: Dict[str, float]
    improvement_actions: List[str]
    learning_curve: List[Tuple[float, float]]  # (time, performance)
    
    def to_dict(self):
        return {
            'experiment_id': self.experiment_id,
            'description': self.description,
            'duration_hours': self.duration_hours,
            'user_interactions': self.user_interactions,
            'baseline_performance': self.baseline_performance,
            'final_performance': self.final_performance,
            'improvement_actions': self.improvement_actions,
            'learning_curve': self.learning_curve
        }


class SimulatedUser:
    """Simulates user interactions for testing continuous learning."""
    
    def __init__(self, user_id: str, behavior_profile: str = "mixed"):
        """
        Initialize simulated user.
        
        Args:
            user_id: Unique user identifier
            behavior_profile: "positive", "negative", "mixed", "expert", "novice"
        """
        self.user_id = user_id
        self.behavior_profile = behavior_profile
        
        # Define behavior patterns
        self.behavior_patterns = {
            "positive": {"thumbs_up_rate": 0.8, "avg_rating": 4.2, "feedback_rate": 0.3},
            "negative": {"thumbs_up_rate": 0.2, "avg_rating": 2.1, "feedback_rate": 0.7},
            "mixed": {"thumbs_up_rate": 0.6, "avg_rating": 3.5, "feedback_rate": 0.4},
            "expert": {"thumbs_up_rate": 0.7, "avg_rating": 3.8, "feedback_rate": 0.8},
            "novice": {"thumbs_up_rate": 0.5, "avg_rating": 3.2, "feedback_rate": 0.2}
        }
        
        self.pattern = self.behavior_patterns.get(behavior_profile, self.behavior_patterns["mixed"])
        
        # Test commands this user might try
        self.test_commands = [
            "create a ball",
            "place sphere above box",
            "add a ramp",
            "put ball on ramp",
            "create two boxes",
            "place cylinder next to sphere",
            "add metal ball",
            "create wooden ramp",
            "place rubber sphere",
            "add three objects"
        ]
    
    def generate_command(self) -> str:
        """Generate a command this user might try."""
        return random.choice(self.test_commands)
    
    def provide_feedback(self, prediction_id: str, text_input: str, 
                        predicted_scene: DynamicPhysicsScene, 
                        actual_performance: Dict[str, float]) -> UserFeedback:
        """Provide feedback based on user behavior profile."""
        
        # Determine feedback type based on performance and user profile
        accuracy = actual_performance.get('accuracy', 0.5)
        plausibility = actual_performance.get('plausibility', 0.5)
        overall_quality = (accuracy + plausibility) / 2
        
        # Adjust quality perception based on user profile
        if self.behavior_profile == "expert":
            # Experts are more critical
            perceived_quality = overall_quality * 0.8
        elif self.behavior_profile == "novice":
            # Novices are more forgiving
            perceived_quality = min(1.0, overall_quality * 1.2)
        else:
            perceived_quality = overall_quality
        
        # Determine if user will provide feedback
        will_provide_feedback = random.random() < self.pattern["feedback_rate"]
        
        if not will_provide_feedback:
            return None
        
        # Generate feedback based on perceived quality
        if perceived_quality > 0.7:
            feedback_type = FeedbackType.THUMBS_UP if random.random() < self.pattern["thumbs_up_rate"] else FeedbackType.RATING
            rating = min(5, max(1, int(np.random.normal(self.pattern["avg_rating"] + 0.5, 0.5))))
            text_feedback = random.choice([
                "Good result!",
                "Works well",
                "Nice physics",
                None
            ]) if random.random() < 0.3 else None
        
        elif perceived_quality > 0.4:
            feedback_type = FeedbackType.RATING
            rating = min(5, max(1, int(np.random.normal(self.pattern["avg_rating"], 0.7))))
            text_feedback = random.choice([
                "Could be better",
                "Okay but not great",
                "Needs improvement",
                None
            ]) if random.random() < 0.4 else None
        
        else:
            feedback_type = FeedbackType.THUMBS_DOWN if random.random() < 0.7 else FeedbackType.CORRECTION
            rating = min(5, max(1, int(np.random.normal(self.pattern["avg_rating"] - 1, 0.5))))
            text_feedback = random.choice([
                "Wrong placement",
                "Physics doesn't work",
                "Not what I expected",
                "Poor quality"
            ]) if random.random() < 0.6 else None
        
        return UserFeedback(
            feedback_id=f"feedback_{self.user_id}_{int(time.time() * 1000)}",
            prediction_id=prediction_id,
            user_id=self.user_id,
            feedback_type=feedback_type,
            timestamp=time.time(),
            rating=rating,
            text_feedback=text_feedback,
            original_text=text_input
        )


class ContinuousLearningTester:
    """Tests continuous learning capabilities."""
    
    def __init__(self):
        """Initialize continuous learning tester."""
        self.experiments = []
        self.simulated_users = []
        
        # Create diverse simulated users
        user_profiles = ["positive", "negative", "mixed", "expert", "novice"]
        for i, profile in enumerate(user_profiles):
            for j in range(2):  # 2 users per profile
                user = SimulatedUser(f"user_{profile}_{j}", profile)
                self.simulated_users.append(user)
    
    def run_learning_experiment(self, duration_minutes: int = 30, 
                               interactions_per_minute: int = 2) -> LearningExperiment:
        """Run a continuous learning experiment."""
        experiment_id = f"experiment_{int(time.time())}"
        print(f"🧪 Starting learning experiment: {experiment_id}")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Interaction rate: {interactions_per_minute} per minute")
        
        # Setup system
        config = ModelConfig()
        model = TextToSceneModel(hidden_size=config.hidden_size, max_objects=config.max_objects)
        
        validator = self._setup_validator(model)
        improvement_loop = SelfImprovementLoop(model)
        
        # Track improvements
        improvement_actions = []
        
        def improvement_callback(plan: ImprovementPlan):
            improvement_actions.extend([action.value for action in plan.actions])
            print(f"   🔧 Improvement: {plan.trigger_reason}")
        
        improvement_loop.add_improvement_callback(improvement_callback)
        
        # Start improvement loop
        improvement_loop.start_improvement_loop()
        
        # Measure baseline performance
        baseline_performance = self._measure_performance(validator, num_tests=10)
        print(f"   📊 Baseline performance: {baseline_performance}")
        
        # Run simulation
        learning_curve = [(0, baseline_performance.get('overall', 0.5))]
        total_interactions = duration_minutes * interactions_per_minute
        
        try:
            for interaction in range(total_interactions):
                # Select random user
                user = random.choice(self.simulated_users)
                
                # Generate command
                command = user.generate_command()
                
                # Simulate prediction and get a REAL performance score
                scene = DynamicPhysicsScene(f"scene_{interaction}")
                validation_result = validator.validate_prediction(command, simulation_duration=1.0)
                actual_performance = { 'score': validation_result.validation_score }

                # The user feedback simulation needs accuracy and plausibility. Let's use the validator's output.
                # This makes the feedback realistic.
                accuracy = 1.0 if validation_result.prediction_valid else 0.0
                plausibility = 1.0 if validation_result.physics_plausible else 0.0

                # Record outcome
                outcome = improvement_loop.record_prediction_outcome(
                    command, scene, {'response_time': random.uniform(0.1, 1.0)}
                )
                outcome.prediction_accuracy = accuracy
                outcome.physics_plausibility = plausibility
                
                # Get user feedback
                feedback = user.provide_feedback(
                    outcome.prediction_id, command, scene, {'accuracy': accuracy, 'plausibility': plausibility}
                )
                
                if feedback:
                    improvement_loop.record_user_feedback(feedback)
                
                # Record learning curve point every 10% of progress
                if interaction % (total_interactions // 10) == 0:
                    current_performance = self._measure_performance(validator, num_tests=5)
                    elapsed_time = (interaction / total_interactions) * duration_minutes
                    learning_curve.append((elapsed_time, current_performance.get('overall', 0.5)))
                
                # Brief pause to simulate real-time
                if duration_minutes <= 5:  # Only for short tests
                    time.sleep(0.1)
            
            # Measure final performance
            final_performance = self._measure_performance(validator, num_tests=10)
            print(f"   📈 Final performance: {final_performance}")
            
            # Create experiment record
            experiment = LearningExperiment(
                experiment_id=experiment_id,
                description=f"Continuous learning test with {len(self.simulated_users)} users",
                duration_hours=duration_minutes / 60,
                user_interactions=total_interactions,
                baseline_performance=baseline_performance,
                final_performance=final_performance,
                improvement_actions=improvement_actions,
                learning_curve=learning_curve
            )
            
            self.experiments.append(experiment)
            
            return experiment
        
        finally:
            validator.bridge.disconnect()
            improvement_loop.stop_improvement_loop()
    
    def _measure_performance(self, validator: PhysicsValidator, num_tests: int = 10) -> Dict[str, float]:
        """Measure current system performance."""
        validation_texts = [
            "create a ball",
            "place sphere above box",
            "add a ramp",
            "put ball on ramp",
            "create two boxes"
        ]

        scores = []
        
        for _ in range(num_tests):
            command = random.choice(validation_texts)
            try:
                # This runs the actual validation
                validation_result = validator.validate_prediction(command, simulation_duration=1.0)
                scores.append(validation_result.validation_score)
            except Exception:
                scores.append(0.0) # Penalize errors

        return {
            'accuracy': np.mean(scores), # Use score as a proxy for accuracy
            'plausibility': np.mean(scores), # and plausibility for this test
            'overall': np.mean(scores)
        }
    
    def analyze_learning_effectiveness(self, experiment: LearningExperiment) -> Dict[str, Any]:
        """Analyze the effectiveness of continuous learning."""
        baseline = experiment.baseline_performance
        final = experiment.final_performance
        
        analysis = {
            'experiment_id': experiment.experiment_id,
            'learning_occurred': False,
            'improvements': {},
            'degradations': {},
            'overall_improvement': 0.0,
            'learning_rate': 0.0,
            'effectiveness_score': 0.0
        }
        
        # Calculate improvements
        for metric in ['accuracy', 'plausibility', 'overall']:
            if metric in baseline and metric in final:
                improvement = final[metric] - baseline[metric]
                if improvement > 0.05:  # Significant improvement threshold
                    analysis['improvements'][metric] = improvement
                    analysis['learning_occurred'] = True
                elif improvement < -0.05:  # Significant degradation
                    analysis['degradations'][metric] = improvement
        
        # Calculate overall improvement
        if 'overall' in baseline and 'overall' in final:
            analysis['overall_improvement'] = final['overall'] - baseline['overall']
        
        # Calculate learning rate (improvement per hour)
        if experiment.duration_hours > 0:
            analysis['learning_rate'] = analysis['overall_improvement'] / experiment.duration_hours
        
        # Calculate effectiveness score (0-1)
        if analysis['learning_occurred']:
            improvement_score = min(1.0, max(0.0, analysis['overall_improvement'] * 2))  # Scale to 0-1
            action_score = min(1.0, len(experiment.improvement_actions) / 10)  # More actions = more learning
            analysis['effectiveness_score'] = (improvement_score + action_score) / 2
        
        return analysis
    
    def generate_learning_report(self, experiments: List[LearningExperiment] = None) -> str:
        """Generate a comprehensive learning report."""
        if experiments is None:
            experiments = self.experiments
        
        if not experiments:
            return "No experiments to analyze."
        
        report = []
        report.append("🧠 CONTINUOUS LEARNING ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Overall statistics
        total_experiments = len(experiments)
        successful_learning = 0
        total_improvement = 0.0
        total_actions = 0
        
        for exp in experiments:
            analysis = self.analyze_learning_effectiveness(exp)
            if analysis['learning_occurred']:
                successful_learning += 1
            total_improvement += analysis['overall_improvement']
            total_actions += len(exp.improvement_actions)
        
        report.append(f"\nOverall Results:")
        report.append(f"  Total Experiments: {total_experiments}")
        report.append(f"  Successful Learning: {successful_learning}/{total_experiments} ({successful_learning/total_experiments:.1%})")
        report.append(f"  Average Improvement: {total_improvement/total_experiments:.3f}")
        report.append(f"  Total Improvement Actions: {total_actions}")
        
        # Individual experiment analysis
        report.append(f"\nIndividual Experiment Results:")
        for exp in experiments:
            analysis = self.analyze_learning_effectiveness(exp)
            
            report.append(f"\n  Experiment {exp.experiment_id}:")
            report.append(f"    Duration: {exp.duration_hours:.1f} hours")
            report.append(f"    Interactions: {exp.user_interactions}")
            report.append(f"    Learning Occurred: {'✅' if analysis['learning_occurred'] else '❌'}")
            report.append(f"    Overall Improvement: {analysis['overall_improvement']:.3f}")
            report.append(f"    Effectiveness Score: {analysis['effectiveness_score']:.3f}")
            report.append(f"    Improvement Actions: {len(exp.improvement_actions)}")
        
        # Recommendations
        report.append(f"\nRecommendations:")
        if successful_learning / total_experiments > 0.7:
            report.append(f"  ✅ Continuous learning is working effectively!")
        elif successful_learning / total_experiments > 0.4:
            report.append(f"  ⚠️ Learning is partially effective - consider tuning parameters")
        else:
            report.append(f"  ❌ Learning effectiveness is low - review learning algorithms")
        
        if total_actions / total_experiments < 2:
            report.append(f"  📈 Consider more aggressive improvement triggers")
        
        return "\n".join(report)


def test_continuous_learning():
    """Test the continuous learning system."""
    print("Testing Continuous Learning System...")
    
    # Create tester
    tester = ContinuousLearningTester()
    
    print(f"✅ Created tester with {len(tester.simulated_users)} simulated users")
    
    # Run short learning experiment
    experiment = tester.run_learning_experiment(duration_minutes=2, interactions_per_minute=5)
    
    print(f"✅ Completed experiment: {experiment.experiment_id}")
    
    # Analyze results
    analysis = tester.analyze_learning_effectiveness(experiment)
    
    print(f"✅ Learning analysis:")
    print(f"  Learning occurred: {analysis['learning_occurred']}")
    print(f"  Overall improvement: {analysis['overall_improvement']:.3f}")
    print(f"  Effectiveness score: {analysis['effectiveness_score']:.3f}")
    print(f"  Improvement actions: {len(experiment.improvement_actions)}")
    
    # Generate report
    report = tester.generate_learning_report([experiment])
    print(f"\n{report}")
    
    # Save experiment data
    with open('data/learning_experiment.json', 'w') as f:
        json.dump(experiment.to_dict(), f, indent=2)
    
    print(f"\n📁 Experiment data saved to: data/learning_experiment.json")
    print("✅ Continuous learning test completed!")


if __name__ == "__main__":
    test_continuous_learning()
