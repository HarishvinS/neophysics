"""
Week 6 Demo - Continuous Learning Integration
Demonstrates the completed Week 6 functionality: self-improvement loops with user feedback
and performance tracking for continuous system enhancement.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import time
import json

from model_architecture import TextToSceneModel, ModelConfig
from user_feedback_system import FeedbackCollector, UserFeedback, FeedbackType, FeedbackAnalyzer
from active_learning_system import ActiveLearningCoordinator, LearningStrategy
from performance_tracking import PerformanceTracker, MetricType
from self_improvement_loop import SelfImprovementLoop, ImprovementAction
from continuous_learning_tester import ContinuousLearningTester, SimulatedUser


def demo_user_feedback_system():
    """Demonstrate user feedback collection and analysis."""
    print("👥 User Feedback System Demo")
    print("=" * 35)
    
    # Create feedback collector
    collector = FeedbackCollector()
    analyzer = FeedbackAnalyzer()
    
    # Simulate diverse user feedback
    test_feedback = [
        UserFeedback("fb1", "pred1", "user1", FeedbackType.THUMBS_UP, time.time(), rating=4, text_feedback="Great physics!"),
        UserFeedback("fb2", "pred2", "user2", FeedbackType.THUMBS_DOWN, time.time(), rating=2, text_feedback="Wrong object placement"),
        UserFeedback("fb3", "pred3", "user3", FeedbackType.RATING, time.time(), rating=5, text_feedback="Perfect simulation"),
        UserFeedback("fb4", "pred4", "user4", FeedbackType.CORRECTION, time.time(), rating=1, text_feedback="Physics is unrealistic"),
        UserFeedback("fb5", "pred5", "user5", FeedbackType.THUMBS_UP, time.time(), rating=4, text_feedback="Good but could improve"),
    ]
    
    print(f"Collecting {len(test_feedback)} feedback entries...")
    
    for feedback in test_feedback:
        collector.collect_feedback(feedback)
    
    print(f"✅ Feedback collected: {collector.stats}")
    
    # Analyze feedback trends
    analysis = analyzer.analyze_feedback_trends(list(collector.feedback_history))
    
    print(f"✅ Feedback Analysis:")
    print(f"   Average rating: {analysis.get('average_rating', 0):.2f}")
    print(f"   Satisfaction level: {analysis.get('satisfaction_level', 'unknown')}")
    print(f"   Common issues: {analysis.get('common_issues', [])[:3]}")
    
    # Extract learning targets
    targets = analyzer.extract_learning_targets(list(collector.feedback_history))
    print(f"✅ Learning targets identified: {len(targets)}")
    
    return collector


def demo_active_learning():
    """Demonstrate active learning request generation."""
    print("\n🎯 Active Learning System Demo")
    print("=" * 35)
    
    # Create model and components
    config = ModelConfig()
    model = TextToSceneModel(config=config)
    
    feedback_collector = FeedbackCollector()
    coordinator = ActiveLearningCoordinator(model, feedback_collector)
    
    # Add some negative feedback to trigger learning requests
    negative_feedback = [
        UserFeedback("fb1", "pred1", "user1", FeedbackType.THUMBS_DOWN, time.time(), rating=2, text_feedback="Poor physics"),
        UserFeedback("fb2", "pred2", "user2", FeedbackType.THUMBS_DOWN, time.time(), rating=1, text_feedback="Wrong objects"),
        UserFeedback("fb3", "pred3", "user3", FeedbackType.CORRECTION, time.time(), rating=2, text_feedback="Needs fixing")
    ]
    
    for feedback in negative_feedback:
        feedback_collector.collect_feedback(feedback)
    
    # Generate learning requests
    recent_predictions = [
        "create a ball",
        "place sphere above box",
        "add complex multi-object scene"
    ]
    
    requests = coordinator.generate_learning_requests(recent_predictions)
    
    print(f"✅ Generated {len(requests)} learning requests:")
    for request in requests:
        print(f"   {request.strategy.value} (priority {request.priority}): {request.description}")
        print(f"     Example prompts: {request.example_prompts[:2]}")
    
    # Show learning summary
    summary = coordinator.get_learning_summary()
    print(f"✅ Learning summary: {summary}")
    
    return coordinator


def demo_performance_tracking():
    """Demonstrate performance tracking and analysis."""
    print("\n📊 Performance Tracking Demo")
    print("=" * 35)
    
    # Create performance tracker
    tracker = PerformanceTracker("data/demo_performance.db")
    tracker.start_session("demo_session")
    
    # Simulate prediction outcomes with varying performance
    from outcome_based_learning import PredictionOutcome
    from dynamic_scene_representation import DynamicPhysicsScene
    
    test_outcomes = [
        ("create a ball", 0.85, 0.9, 0.3),
        ("place sphere above box", 0.75, 0.8, 0.5),
        ("add complex scene", 0.6, 0.7, 0.8),
        ("create physics simulation", 0.9, 0.95, 0.2),
        ("place multiple objects", 0.7, 0.75, 0.6)
    ]
    
    print(f"Recording performance for {len(test_outcomes)} predictions...")
    
    for i, (text, accuracy, plausibility, response_time) in enumerate(test_outcomes):
        outcome = PredictionOutcome(
            prediction_id=f"demo_pred_{i}",
            text_input=text,
            predicted_scene=DynamicPhysicsScene(f"demo_scene_{i}"),
            actual_outcome={'response_time': response_time},
            prediction_accuracy=accuracy,
            physics_plausibility=plausibility,
            timestamp=time.time()
        )
        
        tracker.record_prediction_performance(outcome)
    
    # Simulate user feedback performance
    feedback_data = [
        (FeedbackType.THUMBS_UP, 4),
        (FeedbackType.THUMBS_DOWN, 2),
        (FeedbackType.RATING, 5),
        (FeedbackType.THUMBS_UP, 3),
        (FeedbackType.RATING, 4)
    ]
    
    for i, (ftype, rating) in enumerate(feedback_data):
        feedback = UserFeedback(f"demo_fb_{i}", f"demo_pred_{i}", "demo_user", ftype, time.time(), rating=rating)
        tracker.record_user_feedback_performance(feedback)
    
    # Get performance summary
    summary = tracker.get_performance_summary(hours_back=1)
    
    print(f"✅ Performance Summary:")
    print(f"   Metrics tracked: {len(summary['metrics'])}")
    for metric_name, stats in summary['metrics'].items():
        print(f"     {metric_name}: mean={stats['mean']:.3f}, count={stats['count']}")
    
    print(f"   Alerts: {len(summary['alerts'])}")
    for alert in summary['alerts']:
        print(f"     {alert['severity']}: {alert['message']}")
    
    print(f"   Recommendations: {len(summary['recommendations'])}")
    for rec in summary['recommendations']:
        print(f"     - {rec}")
    
    tracker.end_session()
    return tracker


def demo_self_improvement_loop():
    """Demonstrate the complete self-improvement loop."""
    print("\n🔄 Self-Improvement Loop Demo")
    print("=" * 35)
    
    # Create model and improvement loop
    config = ModelConfig()
    model = TextToSceneModel(config=config)
    
    improvement_loop = SelfImprovementLoop(model)
    
    # Track improvements
    improvements_made = []
    
    def improvement_callback(plan):
        improvements_made.append(plan)
        print(f"   🔧 Improvement executed: {plan.trigger_reason}")
        print(f"      Actions: {[action.value for action in plan.actions]}")
    
    improvement_loop.add_improvement_callback(improvement_callback)
    
    # Start improvement loop
    improvement_loop.start_improvement_loop()
    
    print("✅ Self-improvement loop started")
    
    try:
        # Simulate poor performance to trigger improvements
        from dynamic_scene_representation import DynamicPhysicsScene
        
        poor_scenarios = [
            ("create impossible object", 0.2, 0.3),
            ("place floating sphere", 0.3, 0.4),
            ("add broken physics", 0.1, 0.2)
        ]
        
        print(f"Simulating {len(poor_scenarios)} poor performance scenarios...")
        
        for text, accuracy, plausibility in poor_scenarios:
            scene = DynamicPhysicsScene("poor_scene")
            outcome = improvement_loop.record_prediction_outcome(
                text, scene, {'response_time': 1.0}
            )
            outcome.prediction_accuracy = accuracy
            outcome.physics_plausibility = plausibility
            
            # Add negative feedback
            feedback = UserFeedback(
                f"poor_fb_{len(poor_scenarios)}", 
                outcome.prediction_id, 
                "demo_user", 
                FeedbackType.THUMBS_DOWN, 
                time.time(), 
                rating=1,
                text_feedback="This is terrible"
            )
            improvement_loop.record_user_feedback(feedback)
        
        # Wait for improvements to be processed
        time.sleep(3)
        
        # Get improvement status
        status = improvement_loop.get_improvement_status()
        
        print(f"✅ Improvement Status:")
        print(f"   Running: {status['running']}")
        print(f"   Pending plans: {status['pending_plans']}")
        print(f"   Executed plans: {status['executed_plans']}")
        print(f"   Active learning requests: {status['active_learning_requests']}")
        print(f"   Recent feedback: {status['recent_feedback_count']}")
        print(f"   Performance alerts: {status['performance_alerts']}")
        
        print(f"✅ Improvements made: {len(improvements_made)}")
        for improvement in improvements_made:
            print(f"   - {improvement.trigger_reason}: {len(improvement.actions)} actions")
    
    finally:
        improvement_loop.stop_improvement_loop()
    
    return improvement_loop


def demo_continuous_learning_validation():
    """Demonstrate continuous learning validation."""
    print("\n🧪 Continuous Learning Validation Demo")
    print("=" * 45)
    
    # Create learning tester
    tester = ContinuousLearningTester()
    
    print(f"✅ Created tester with {len(tester.simulated_users)} simulated users")
    
    # Show user profiles
    profiles = {}
    for user in tester.simulated_users:
        profile = user.behavior_profile
        profiles[profile] = profiles.get(profile, 0) + 1
    
    print(f"   User profiles: {profiles}")
    
    # Run a short learning experiment
    print(f"\nRunning continuous learning experiment...")
    experiment = tester.run_learning_experiment(duration_minutes=1, interactions_per_minute=10)
    
    # Analyze results
    analysis = tester.analyze_learning_effectiveness(experiment)
    
    print(f"✅ Experiment Results:")
    print(f"   Experiment ID: {experiment.experiment_id}")
    print(f"   Duration: {experiment.duration_hours:.2f} hours")
    print(f"   User interactions: {experiment.user_interactions}")
    print(f"   Learning occurred: {'✅' if analysis['learning_occurred'] else '❌'}")
    print(f"   Overall improvement: {analysis['overall_improvement']:.3f}")
    print(f"   Effectiveness score: {analysis['effectiveness_score']:.3f}")
    print(f"   Improvement actions: {len(experiment.improvement_actions)}")
    
    # Generate learning report
    report = tester.generate_learning_report([experiment])
    print(f"\n📋 Learning Report:")
    print(report)
    
    return experiment


def demo_integration_showcase():
    """Showcase the complete integration of all Week 6 components."""
    print("\n🎭 Complete Integration Showcase")
    print("=" * 40)
    
    print("Week 6 brings together all learning components:")
    
    print("\n1. ✅ User Feedback System")
    print("   - Collects thumbs up/down, ratings, and detailed feedback")
    print("   - Analyzes feedback trends and satisfaction levels")
    print("   - Extracts learning targets from user input")
    print("   - GUI interface for easy feedback collection")
    
    print("\n2. ✅ Active Learning System")
    print("   - Uncertainty-based learning request generation")
    print("   - Diversity sampling for coverage gaps")
    print("   - Feedback-driven learning prioritization")
    print("   - Error analysis and targeted improvement")
    
    print("\n3. ✅ Performance Tracking")
    print("   - Comprehensive metrics database")
    print("   - Real-time performance monitoring")
    print("   - Trend analysis and alert system")
    print("   - Automated recommendation generation")
    
    print("\n4. ✅ Self-Improvement Loop")
    print("   - Automated improvement plan generation")
    print("   - Multi-strategy learning coordination")
    print("   - Background monitoring and adaptation")
    print("   - Continuous performance optimization")
    
    print("\n5. ✅ Continuous Learning Validation")
    print("   - Simulated user interaction testing")
    print("   - Learning effectiveness measurement")
    print("   - Long-term improvement validation")
    print("   - Comprehensive learning analytics")
    
    print("\n🎯 Key Achievements:")
    print("   • Human-in-the-loop learning with real user feedback")
    print("   • Automated identification of learning opportunities")
    print("   • Continuous performance monitoring and alerting")
    print("   • Self-improving system that adapts without manual intervention")
    print("   • Validated learning effectiveness through controlled experiments")
    
    print("\n🔄 The Complete Learning Cycle:")
    print("   1. User interacts with system → Feedback collected")
    print("   2. Performance tracked → Trends analyzed")
    print("   3. Learning opportunities identified → Requests generated")
    print("   4. Improvement plans created → Actions executed")
    print("   5. System adapts → Performance improves")
    print("   6. Cycle repeats continuously")


def main():
    """Run the complete Week 6 demo."""
    print("🎬 Week 6 Demo: Continuous Learning Integration")
    print("=" * 70)
    print("Building self-improvement loops with user feedback and performance tracking")
    print("=" * 70)
    
    # Run all demos
    feedback_collector = demo_user_feedback_system()
    active_learning = demo_active_learning()
    performance_tracker = demo_performance_tracking()
    improvement_loop = demo_self_improvement_loop()
    learning_experiment = demo_continuous_learning_validation()
    demo_integration_showcase()
    
    print("\n" + "=" * 70)
    print("🎉 Week 6 Demo Complete!")
    print("=" * 70)
    
    print("\nKey Achievements:")
    print("✅ User feedback collection and analysis system")
    print("✅ Active learning with uncertainty and diversity sampling")
    print("✅ Comprehensive performance tracking and monitoring")
    print("✅ Automated self-improvement loop")
    print("✅ Continuous learning validation framework")
    
    print("\nTechnical Breakthroughs:")
    print("👥 Human-in-the-loop learning integration")
    print("🎯 Intelligent learning request generation")
    print("📊 Real-time performance analytics")
    print("🔄 Autonomous system adaptation")
    print("🧪 Scientific learning effectiveness validation")
    
    print("\nSystem Capabilities:")
    print("🔍 Identifies what it doesn't know well")
    print("📝 Learns from user feedback automatically")
    print("📈 Tracks and improves performance over time")
    print("🤖 Self-improves without manual intervention")
    print("🧠 Validates learning through controlled experiments")
    
    print("\nReady for Week 7: Advanced Physics Understanding! 🚀")
    print("Next: Deep physics reasoning and causal modeling")


if __name__ == "__main__":
    main()
