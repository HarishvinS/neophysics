"""
Self-Improvement Loop
Automated system that learns from outcomes and adapts without manual intervention.
Integrates all learning components into a continuous improvement cycle.
"""

import torch
import numpy as np
import time
import json
import threading
from threading import Lock
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
# import schedule  # Optional dependency
from collections import deque

from model_architecture import TextToSceneModel
from user_feedback_system import FeedbackCollector, UserFeedback, FeedbackAnalyzer
from active_learning_system import ActiveLearningCoordinator, LearningRequest
from performance_tracking import PerformanceTracker, MetricType
from outcome_based_learning import ContinuousLearningSystem, PredictionOutcome
from physics_reasoning_engine import CausalRuleLearner, PhysicsEventDetector


class ImprovementAction(Enum):
    """Types of improvement actions the system can take."""
    RETRAIN_MODEL = "retrain_model"
    UPDATE_RULES = "update_rules"
    REQUEST_DATA = "request_data"
    ADJUST_THRESHOLDS = "adjust_thresholds"
    OPTIMIZE_PERFORMANCE = "optimize_performance"
    GATHER_FEEDBACK = "gather_feedback"


@dataclass
class ImprovementPlan:
    """Represents a plan for system improvement."""
    plan_id: str
    trigger_reason: str
    actions: List[ImprovementAction]
    priority: int
    estimated_impact: float
    resource_cost: int
    timestamp: float
    
    def to_dict(self):
        return {
            'plan_id': self.plan_id,
            'trigger_reason': self.trigger_reason,
            'actions': [action.value for action in self.actions],
            'priority': self.priority,
            'estimated_impact': self.estimated_impact,
            'resource_cost': self.resource_cost,
            'timestamp': self.timestamp
        }


class SelfImprovementLoop:
    """Main self-improvement system that coordinates all learning components."""
    
    def __init__(self, model: TextToSceneModel):
        """Initialize self-improvement loop."""
        self.model = model
        
        # Core components
        self.feedback_collector = FeedbackCollector()
        self.feedback_analyzer = FeedbackAnalyzer()
        self.active_learning = ActiveLearningCoordinator(model, self.feedback_collector)
        self.performance_tracker = PerformanceTracker()
        self.continuous_learning = ContinuousLearningSystem(model)
        self.rule_learner = CausalRuleLearner()
        
        # Improvement state
        self.improvement_plans = deque(maxlen=100)
        self.executed_plans = deque(maxlen=500)
        self.running = False
        self.plan_lock = Lock()
        self.improvement_thread = None
        
        # Configuration
        self.improvement_interval = 3600  # 1 hour
        self.performance_check_interval = 300  # 5 minutes
        self.min_feedback_for_action = 5
        self.performance_decline_threshold = 0.1
        
        # Callbacks
        self.improvement_callbacks: List[Callable[[ImprovementPlan], None]] = []
        
        # Setup automatic scheduling
        self._setup_scheduled_tasks()
    
    def add_improvement_callback(self, callback: Callable[[ImprovementPlan], None]):
        """Add callback to be called when improvements are made."""
        self.improvement_callbacks.append(callback)
    
    def start_improvement_loop(self):
        """Start the continuous improvement loop."""
        if self.running:
            print("⚠️ Improvement loop already running")
            return
        
        self.running = True
        self.performance_tracker.start_session("continuous_improvement")
        
        # Start background thread for continuous monitoring
        self.improvement_thread = threading.Thread(
            target=self._improvement_loop,
            daemon=True
        )
        self.improvement_thread.start()
        
        print("🔄 Self-improvement loop started")
    
    def stop_improvement_loop(self):
        """Stop the continuous improvement loop."""
        self.running = False
        
        if self.improvement_thread and self.improvement_thread.is_alive():
            self.improvement_thread.join(timeout=5.0)
        
        self.performance_tracker.end_session()
        print("🛑 Self-improvement loop stopped")
    
    def _setup_scheduled_tasks(self):
        """Setup scheduled tasks for improvement."""
        # Simplified scheduling without external dependency
        self.last_performance_check = time.time()
        self.last_improvement_plan = time.time()
        self.last_learning_request = time.time()
    
    def _improvement_loop(self):
        """Main improvement loop (runs in background thread)."""
        while self.running:
            try:
                current_time = time.time()

                # Check if it's time for performance check (every 5 minutes)
                if current_time - self.last_performance_check > 300:
                    self._check_performance()
                    self.last_performance_check = current_time

                # Check if it's time for improvement planning (every hour)
                if current_time - self.last_improvement_plan > 3600:
                    self._plan_improvements()
                    self.last_improvement_plan = current_time

                # Check if it's time for learning requests (every 30 minutes)
                if current_time - self.last_learning_request > 1800:
                    self._generate_learning_requests()
                    self.last_learning_request = current_time

                # Process any pending improvement plans
                self._execute_pending_plans()

                # Brief sleep to prevent excessive CPU usage
                time.sleep(10)

            except Exception as e:
                print(f"❌ Error in improvement loop: {e}")
                time.sleep(60)  # Wait longer on error
    
    def record_prediction_outcome(self, text_input: str, predicted_scene, simulation_result: Dict[str, Any]) -> PredictionOutcome:
        """Record a prediction outcome and trigger learning if needed."""
        # Record in continuous learning system
        outcome = self.continuous_learning.record_outcome(text_input, predicted_scene, simulation_result)
        
        # Record performance metrics
        self.performance_tracker.record_prediction_performance(outcome)
        
        # Check if immediate improvement is needed
        if outcome.prediction_accuracy < 0.5 or outcome.physics_plausibility < 0.6:
            self._trigger_immediate_improvement("poor_prediction_quality")
        
        return outcome
    
    def record_user_feedback(self, feedback: UserFeedback):
        """Record user feedback and trigger learning if needed."""
        # Collect feedback
        self.feedback_collector.collect_feedback(feedback)
        
        # Record performance metrics
        self.performance_tracker.record_user_feedback_performance(feedback)
        
        # Check if immediate improvement is needed
        if feedback.feedback_type.value == 'thumbs_down' or (feedback.rating and feedback.rating <= 2):
            recent_negative = len([f for f in self.feedback_collector.get_recent_feedback(1) 
                                 if f.feedback_type.value == 'thumbs_down'])
            
            if recent_negative >= 3:  # 3 negative feedback in 1 hour
                self._trigger_immediate_improvement("multiple_negative_feedback")
    
    def _check_performance(self):
        """Check current performance and trigger improvements if needed."""
        if not self.running:
            return
        
        try:
            summary = self.performance_tracker.get_performance_summary(hours_back=1)
            
            # Check for performance alerts
            if summary['alerts']:
                high_severity_alerts = [a for a in summary['alerts'] if a['severity'] == 'high']
                if high_severity_alerts:
                    self._trigger_immediate_improvement("performance_alerts")
            
            # Check for declining trends
            declining_metrics = [
                metric for metric, trend in summary.get('trends', {}).items()
                if trend.get('trend') == 'declining' and trend.get('change_percent', 0) < -10
            ]
            
            if declining_metrics:
                self._trigger_immediate_improvement(f"declining_performance_{declining_metrics[0]}")
            
        except Exception as e:
            print(f"⚠️ Error checking performance: {e}")
    
    def _plan_improvements(self):
        """Plan improvements based on current state."""
        if not self.running:
            return
        
        try:
            # Analyze recent feedback
            recent_feedback = self.feedback_collector.get_recent_feedback(24)
            feedback_analysis = self.feedback_analyzer.analyze_feedback_trends(recent_feedback)
            
            # Get performance summary
            performance_summary = self.performance_tracker.get_performance_summary(24)
            
            # Generate improvement plan
            plan = self._create_improvement_plan(feedback_analysis, performance_summary)
            
            if plan:
                self.improvement_plans.append(plan)
                print(f"📋 Created improvement plan: {plan.plan_id}")
            
        except Exception as e:
            print(f"⚠️ Error planning improvements: {e}")
    
    def _generate_learning_requests(self):
        """Generate active learning requests."""
        if not self.running:
            return
        
        try:
            # Get recent prediction texts for analysis
            recent_outcomes = list(self.continuous_learning.outcome_history)[-20:]
            recent_texts = [o.text_input for o in recent_outcomes if o.text_input]
            
            # Generate learning requests
            requests = self.active_learning.generate_learning_requests(recent_texts)
            
            if requests:
                print(f"🎯 Generated {len(requests)} active learning requests")
            
        except Exception as e:
            print(f"⚠️ Error generating learning requests: {e}")
    
    def _trigger_immediate_improvement(self, reason: str):
        """Trigger immediate improvement action."""
        plan = ImprovementPlan(
            plan_id=f"immediate_{int(time.time())}",
            trigger_reason=reason,
            actions=[ImprovementAction.GATHER_FEEDBACK, ImprovementAction.UPDATE_RULES],
            priority=5,  # Highest priority
            estimated_impact=0.8,
            resource_cost=2,
            timestamp=time.time()
        )
        with self.plan_lock:
            self.improvement_plans.appendleft(plan)  # Add to front for immediate execution
        print(f"🚨 Triggered immediate improvement: {reason}")
    
    def _create_improvement_plan(self, feedback_analysis: Dict[str, Any], 
                               performance_summary: Dict[str, Any]) -> Optional[ImprovementPlan]:
        """Create an improvement plan based on analysis."""
        actions = []
        priority = 1
        estimated_impact = 0.1
        
        # Analyze feedback
        if feedback_analysis.get('total_feedback', 0) >= self.min_feedback_for_action:
            avg_rating = feedback_analysis.get('average_rating', 5.0)
            
            if avg_rating < 3.0:
                actions.extend([ImprovementAction.RETRAIN_MODEL, ImprovementAction.REQUEST_DATA])
                priority = 4
                estimated_impact = 0.6
            elif avg_rating < 4.0:
                actions.append(ImprovementAction.UPDATE_RULES)
                priority = 3
                estimated_impact = 0.3
        
        # Analyze performance metrics
        for metric_name, stats in performance_summary.get('metrics', {}).items():
            if metric_name == 'accuracy' and stats['mean'] < 0.7:
                actions.append(ImprovementAction.RETRAIN_MODEL)
                priority = max(priority, 4)
                estimated_impact = max(estimated_impact, 0.5)
            elif metric_name == 'response_time' and stats['mean'] > 2.0:
                actions.append(ImprovementAction.OPTIMIZE_PERFORMANCE)
                priority = max(priority, 2)
                estimated_impact = max(estimated_impact, 0.2)
        
        # Check for recommendations
        recommendations = performance_summary.get('recommendations', [])
        if recommendations:
            if any('training' in rec.lower() for rec in recommendations):
                actions.append(ImprovementAction.REQUEST_DATA)
            if any('feedback' in rec.lower() for rec in recommendations):
                actions.append(ImprovementAction.GATHER_FEEDBACK)
        
        if not actions:
            return None
        
        # Remove duplicates
        actions = list(set(actions))
        
        return ImprovementPlan(
            plan_id=f"plan_{int(time.time())}",
            trigger_reason="scheduled_analysis",
            actions=actions,
            priority=priority,
            estimated_impact=estimated_impact,
            resource_cost=len(actions),
            timestamp=time.time()
        )
    
    def _execute_pending_plans(self):
        """Execute pending improvement plans."""
        with self.plan_lock:
            if not self.improvement_plans:
                return
            
            # Sort by priority (highest first)
            # This operation is now safe within the lock
            sorted_plans = sorted(list(self.improvement_plans), key=lambda p: p.priority, reverse=True)
            plan = sorted_plans[0]
            self.improvement_plans.remove(plan)
        
        try:
            self._execute_improvement_plan(plan)
            self.executed_plans.append(plan)
            
            # Trigger callbacks
            for callback in self.improvement_callbacks:
                try:
                    callback(plan)
                except Exception as e:
                    print(f"⚠️ Improvement callback error: {e}")
            
        except Exception as e:
            print(f"❌ Error executing improvement plan {plan.plan_id}: {e}")
    
    def _execute_improvement_plan(self, plan: ImprovementPlan):
        """Execute a specific improvement plan."""
        print(f"🔧 Executing improvement plan: {plan.plan_id}")
        
        for action in plan.actions:
            try:
                if action == ImprovementAction.UPDATE_RULES:
                    self._update_physics_rules()
                elif action == ImprovementAction.RETRAIN_MODEL:
                    self._trigger_model_retraining()
                elif action == ImprovementAction.REQUEST_DATA:
                    self._request_additional_data()
                elif action == ImprovementAction.GATHER_FEEDBACK:
                    self._gather_more_feedback()
                elif action == ImprovementAction.OPTIMIZE_PERFORMANCE:
                    self._optimize_performance()
                elif action == ImprovementAction.ADJUST_THRESHOLDS:
                    self._adjust_thresholds()
                
                print(f"  ✅ Completed action: {action.value}")
                
            except Exception as e:
                print(f"  ❌ Failed action {action.value}: {e}")
    
    def _update_physics_rules(self):
        """Update physics rules based on recent learning."""
        # Trigger continuous learning cycle
        results = self.continuous_learning.trigger_learning_cycle()
        print(f"    Updated physics rules: {results.get('rules_learned', 0)} new rules")
    
    def _trigger_model_retraining(self):
        """Trigger model retraining with recent data."""
        # Get recent poor outcomes for retraining
        poor_outcomes = [
            o for o in self.continuous_learning.outcome_history
            if o.prediction_accuracy < 0.7 or o.physics_plausibility < 0.7
        ]
        
        if poor_outcomes:
            # This would trigger actual retraining in a real system
            print(f"    Triggered retraining with {len(poor_outcomes)} poor outcomes")
        else:
            print(f"    No poor outcomes found for retraining")
    
    def _request_additional_data(self):
        """Request additional training data."""
        requests = self.active_learning.get_active_requests()
        print(f"    Active learning requests: {len(requests)}")
    
    def _gather_more_feedback(self):
        """Gather more user feedback."""
        # This would trigger feedback collection in a real system
        print(f"    Gathering more user feedback")
    
    def _optimize_performance(self):
        """Optimize system performance."""
        # This would trigger performance optimizations
        print(f"    Optimizing system performance")
    
    def _adjust_thresholds(self):
        """Adjust performance thresholds."""
        # This would adjust various system thresholds
        print(f"    Adjusting performance thresholds")
    
    def get_improvement_status(self) -> Dict[str, Any]:
        """Get current improvement status."""
        return {
            'running': self.running,
            'pending_plans': len(self.improvement_plans),
            'executed_plans': len(self.executed_plans),
            'active_learning_requests': len(self.active_learning.get_active_requests()),
            'recent_feedback_count': len(self.feedback_collector.get_recent_feedback(24)),
            'performance_alerts': len(self.performance_tracker.get_performance_summary(1).get('alerts', [])),
            'last_improvement': self.executed_plans[-1].timestamp if self.executed_plans else None
        }


def test_self_improvement_loop():
    """Test the self-improvement loop."""
    print("Testing Self-Improvement Loop...")
    
    # Create test model
    from model_architecture import ModelConfig
    config = ModelConfig()
    model = TextToSceneModel(hidden_size=config.hidden_size, max_objects=config.max_objects)
    
    # Create improvement loop
    improvement_loop = SelfImprovementLoop(model)
    
    # Add callback to monitor improvements
    def improvement_callback(plan: ImprovementPlan):
        print(f"🔧 Improvement executed: {plan.plan_id} - {plan.trigger_reason}")
    
    improvement_loop.add_improvement_callback(improvement_callback)
    
    # Start improvement loop
    improvement_loop.start_improvement_loop()
    
    # Simulate some prediction outcomes
    from dynamic_scene_representation import DynamicPhysicsScene
    
    test_outcomes = [
        ("create a ball", 0.6, 0.7),  # Mediocre performance
        ("place sphere above box", 0.4, 0.5),  # Poor performance
        ("add ramp", 0.8, 0.9)  # Good performance
    ]
    
    for text, accuracy, plausibility in test_outcomes:
        scene = DynamicPhysicsScene("test")
        outcome = improvement_loop.record_prediction_outcome(
            text, scene, {'response_time': 0.5}
        )
        outcome.prediction_accuracy = accuracy
        outcome.physics_plausibility = plausibility
    
    print(f"✅ Recorded {len(test_outcomes)} prediction outcomes")
    
    # Simulate user feedback
    from user_feedback_system import UserFeedback, FeedbackType
    
    test_feedback = [
        UserFeedback("fb1", "pred1", "user1", FeedbackType.THUMBS_DOWN, time.time(), rating=2),
        UserFeedback("fb2", "pred2", "user1", FeedbackType.THUMBS_DOWN, time.time(), rating=1),
        UserFeedback("fb3", "pred3", "user1", FeedbackType.THUMBS_UP, time.time(), rating=4)
    ]
    
    for feedback in test_feedback:
        improvement_loop.record_user_feedback(feedback)
    
    print(f"✅ Recorded {len(test_feedback)} user feedback entries")
    
    # Wait a bit for processing
    time.sleep(2)
    
    # Get improvement status
    status = improvement_loop.get_improvement_status()
    print(f"✅ Improvement status: {status}")
    
    # Stop improvement loop
    improvement_loop.stop_improvement_loop()
    
    print("✅ Self-improvement loop test completed!")


if __name__ == "__main__":
    test_self_improvement_loop()
