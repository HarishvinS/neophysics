"""
Performance Tracking System
Comprehensive metrics tracking and trend analysis for continuous improvement.
Monitors model performance over time and identifies improvement opportunities.
"""

import torch
import numpy as np
import time
import json
import sqlite3
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from user_feedback_system import UserFeedback, FeedbackCollector
from outcome_based_learning import PredictionOutcome


class MetricType(Enum):
    """Types of performance metrics."""
    ACCURACY = "accuracy"
    PHYSICS_PLAUSIBILITY = "physics_plausibility"
    USER_SATISFACTION = "user_satisfaction"
    RESPONSE_TIME = "response_time"
    SUCCESS_RATE = "success_rate"
    ERROR_RATE = "error_rate"
    LEARNING_RATE = "learning_rate"
    GENERALIZATION = "generalization"


@dataclass
class PerformanceMetric:
    """Represents a single performance metric measurement."""
    metric_id: str
    metric_type: MetricType
    value: float
    timestamp: float
    context: Dict[str, Any]
    session_id: Optional[str] = None
    
    def to_dict(self):
        return {
            'metric_id': self.metric_id,
            'metric_type': self.metric_type.value,
            'value': self.value,
            'timestamp': self.timestamp,
            'context': self.context,
            'session_id': self.session_id
        }


class MetricsDatabase:
    """SQLite database for storing performance metrics."""
    
    def __init__(self, db_path: str = "data/performance_metrics.db"):
        """Initialize metrics database."""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_id TEXT UNIQUE,
                metric_type TEXT,
                value REAL,
                timestamp REAL,
                context TEXT,
                session_id TEXT
            )
        ''')
        
        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time REAL,
                end_time REAL,
                total_predictions INTEGER,
                session_type TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_metric(self, metric: PerformanceMetric):
        """Store a performance metric."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO metrics 
            (metric_id, metric_type, value, timestamp, context, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            metric.metric_id,
            metric.metric_type.value,
            metric.value,
            metric.timestamp,
            json.dumps(metric.context),
            metric.session_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_metrics(self, metric_type: MetricType = None, 
                   hours_back: int = 24) -> List[PerformanceMetric]:
        """Retrieve metrics from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = time.time() - (hours_back * 3600)
        
        if metric_type:
            cursor.execute('''
                SELECT metric_id, metric_type, value, timestamp, context, session_id
                FROM metrics 
                WHERE metric_type = ? AND timestamp > ?
                ORDER BY timestamp DESC
            ''', (metric_type.value, cutoff_time))
        else:
            cursor.execute('''
                SELECT metric_id, metric_type, value, timestamp, context, session_id
                FROM metrics 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', (cutoff_time,))
        
        rows = cursor.fetchall()
        conn.close()
        
        metrics = []
        for row in rows:
            metric = PerformanceMetric(
                metric_id=row[0],
                metric_type=MetricType(row[1]),
                value=row[2],
                timestamp=row[3],
                context=json.loads(row[4]) if row[4] else {},
                session_id=row[5]
            )
            metrics.append(metric)
        
        return metrics


class PerformanceTracker:
    """Tracks and analyzes performance metrics."""
    
    def __init__(self, db_path: str = "data/performance_metrics.db"):
        """Initialize performance tracker."""
        self.db = MetricsDatabase(db_path)
        self.current_session_id = None
        self.session_start_time = None
        
        # In-memory cache for recent metrics
        self.recent_metrics = deque(maxlen=1000)
        
        # Metric thresholds for alerts
        self.thresholds = {
            MetricType.ACCURACY: {'min': 0.7, 'target': 0.85},
            MetricType.PHYSICS_PLAUSIBILITY: {'min': 0.75, 'target': 0.9},
            MetricType.USER_SATISFACTION: {'min': 3.0, 'target': 4.0},
            MetricType.RESPONSE_TIME: {'max': 2.0, 'target': 0.5},
            MetricType.SUCCESS_RATE: {'min': 0.8, 'target': 0.95}
        }
    
    def start_session(self, session_type: str = "interactive"):
        """Start a new performance tracking session."""
        self.current_session_id = f"session_{int(time.time())}"
        self.session_start_time = time.time()
        
        print(f"📊 Started performance tracking session: {self.current_session_id}")
    
    def end_session(self):
        """End the current performance tracking session."""
        if self.current_session_id:
            # Store session summary in database
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO sessions (session_id, start_time, end_time, total_predictions, session_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                self.current_session_id,
                self.session_start_time,
                time.time(),
                len([m for m in self.recent_metrics if m.session_id == self.current_session_id]),
                "interactive"
            ))
            
            conn.commit()
            conn.close()
            
            print(f"📊 Ended performance tracking session: {self.current_session_id}")
            self.current_session_id = None
    
    def record_prediction_performance(self, outcome: PredictionOutcome):
        """Record performance metrics from a prediction outcome."""
        timestamp = time.time()
        
        # Accuracy metric
        accuracy_metric = PerformanceMetric(
            metric_id=f"accuracy_{outcome.prediction_id}",
            metric_type=MetricType.ACCURACY,
            value=outcome.prediction_accuracy,
            timestamp=timestamp,
            context={'prediction_id': outcome.prediction_id, 'text': outcome.text_input},
            session_id=self.current_session_id
        )
        self._store_metric(accuracy_metric)
        
        # Physics plausibility metric
        physics_metric = PerformanceMetric(
            metric_id=f"physics_{outcome.prediction_id}",
            metric_type=MetricType.PHYSICS_PLAUSIBILITY,
            value=outcome.physics_plausibility,
            timestamp=timestamp,
            context={'prediction_id': outcome.prediction_id, 'text': outcome.text_input},
            session_id=self.current_session_id
        )
        self._store_metric(physics_metric)
        
        # Response time (if available)
        if 'response_time' in outcome.actual_outcome:
            response_metric = PerformanceMetric(
                metric_id=f"response_{outcome.prediction_id}",
                metric_type=MetricType.RESPONSE_TIME,
                value=outcome.actual_outcome['response_time'],
                timestamp=timestamp,
                context={'prediction_id': outcome.prediction_id},
                session_id=self.current_session_id
            )
            self._store_metric(response_metric)
    
    def record_user_feedback_performance(self, feedback: UserFeedback):
        """Record performance metrics from user feedback."""
        timestamp = time.time()
        
        # User satisfaction from rating
        if feedback.rating:
            satisfaction_metric = PerformanceMetric(
                metric_id=f"satisfaction_{feedback.feedback_id}",
                metric_type=MetricType.USER_SATISFACTION,
                value=feedback.rating,
                timestamp=timestamp,
                context={
                    'feedback_id': feedback.feedback_id,
                    'prediction_id': feedback.prediction_id,
                    'feedback_type': feedback.feedback_type.value
                },
                session_id=self.current_session_id
            )
            self._store_metric(satisfaction_metric)
        
        # Success rate (binary from thumbs up/down)
        if feedback.feedback_type.value in ['thumbs_up', 'thumbs_down']:
            success_value = 1.0 if feedback.feedback_type.value == 'thumbs_up' else 0.0
            success_metric = PerformanceMetric(
                metric_id=f"success_{feedback.feedback_id}",
                metric_type=MetricType.SUCCESS_RATE,
                value=success_value,
                timestamp=timestamp,
                context={'feedback_id': feedback.feedback_id, 'prediction_id': feedback.prediction_id},
                session_id=self.current_session_id
            )
            self._store_metric(success_metric)
    
    def _store_metric(self, metric: PerformanceMetric):
        """Store metric in database and cache."""
        self.db.store_metric(metric)
        self.recent_metrics.append(metric)
    
    def get_performance_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'time_period': f"Last {hours_back} hours",
            'timestamp': time.time(),
            'metrics': {},
            'trends': {},
            'alerts': [],
            'recommendations': []
        }
        
        # Get metrics for each type
        for metric_type in MetricType:
            metrics = self.db.get_metrics(metric_type, hours_back)
            
            if metrics:
                values = [m.value for m in metrics]
                summary['metrics'][metric_type.value] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'latest': values[0] if values else None
                }
                
                # Analyze trends
                if len(values) >= 5:
                    trend = self._analyze_trend(values)
                    summary['trends'][metric_type.value] = trend
                
                # Check for alerts
                alerts = self._check_alerts(metric_type, values)
                summary['alerts'].extend(alerts)
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations(summary)
        
        return summary
    
    def _analyze_trend(self, values: List[float]) -> Dict[str, Any]:
        """Analyze trend in metric values."""
        if len(values) < 5:
            return {'trend': 'insufficient_data'}
        
        # Simple linear trend analysis
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        trend_direction = 'improving' if slope > 0.01 else 'declining' if slope < -0.01 else 'stable'
        
        # Recent vs older comparison
        recent_avg = np.mean(values[:len(values)//3])  # Most recent third
        older_avg = np.mean(values[len(values)//3:])   # Older two thirds
        
        return {
            'trend': trend_direction,
            'slope': slope,
            'recent_avg': recent_avg,
            'older_avg': older_avg,
            'change_percent': ((recent_avg - older_avg) / older_avg * 100) if older_avg != 0 else 0
        }
    
    def _check_alerts(self, metric_type: MetricType, values: List[float]) -> List[Dict[str, Any]]:
        """Check for performance alerts."""
        alerts = []
        
        if metric_type not in self.thresholds:
            return alerts
        
        thresholds = self.thresholds[metric_type]
        latest_value = values[0] if values else None
        
        if latest_value is None:
            return alerts
        
        # Check minimum threshold
        if 'min' in thresholds and latest_value < thresholds['min']:
            alerts.append({
                'type': 'performance_alert',
                'metric': metric_type.value,
                'severity': 'high',
                'message': f"{metric_type.value} below minimum threshold: {latest_value:.3f} < {thresholds['min']}"
            })
        
        # Check maximum threshold (for metrics like response time)
        if 'max' in thresholds and latest_value > thresholds['max']:
            alerts.append({
                'type': 'performance_alert',
                'metric': metric_type.value,
                'severity': 'high',
                'message': f"{metric_type.value} above maximum threshold: {latest_value:.3f} > {thresholds['max']}"
            })
        
        return alerts
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        # Check accuracy
        if 'accuracy' in summary['metrics']:
            acc_stats = summary['metrics']['accuracy']
            if acc_stats['mean'] < 0.8:
                recommendations.append("Consider additional training data to improve accuracy")
        
        # Check user satisfaction
        if 'user_satisfaction' in summary['metrics']:
            sat_stats = summary['metrics']['user_satisfaction']
            if sat_stats['mean'] < 3.5:
                recommendations.append("User satisfaction is low - review recent feedback for improvement areas")
        
        # Check response time
        if 'response_time' in summary['metrics']:
            time_stats = summary['metrics']['response_time']
            if time_stats['mean'] > 1.0:
                recommendations.append("Response time is high - consider model optimization")
        
        # Check trends
        for metric, trend_data in summary.get('trends', {}).items():
            if trend_data.get('trend') == 'declining':
                recommendations.append(f"{metric} is declining - investigate recent changes")
        
        return recommendations
    
    def export_metrics(self, filepath: str, hours_back: int = 168):  # 1 week default
        """Export metrics to JSON file."""
        all_metrics = []
        
        for metric_type in MetricType:
            metrics = self.db.get_metrics(metric_type, hours_back)
            all_metrics.extend([m.to_dict() for m in metrics])
        
        export_data = {
            'export_timestamp': time.time(),
            'time_period_hours': hours_back,
            'total_metrics': len(all_metrics),
            'metrics': all_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📁 Exported {len(all_metrics)} metrics to {filepath}")


def test_performance_tracking():
    """Test the performance tracking system."""
    print("Testing Performance Tracking System...")
    
    # Create performance tracker
    tracker = PerformanceTracker("data/test_performance.db")
    
    # Start a session
    tracker.start_session("test_session")
    
    # Create test prediction outcomes
    from outcome_based_learning import PredictionOutcome
    from dynamic_scene_representation import DynamicPhysicsScene
    
    test_outcomes = [
        PredictionOutcome(
            prediction_id="test_pred_1",
            text_input="create a ball",
            predicted_scene=DynamicPhysicsScene("test_scene_1"),
            actual_outcome={'response_time': 0.5},
            prediction_accuracy=0.85,
            physics_plausibility=0.9,
            timestamp=time.time()
        ),
        PredictionOutcome(
            prediction_id="test_pred_2",
            text_input="place sphere above box",
            predicted_scene=DynamicPhysicsScene("test_scene_2"),
            actual_outcome={'response_time': 0.3},
            prediction_accuracy=0.75,
            physics_plausibility=0.8,
            timestamp=time.time()
        )
    ]
    
    # Record prediction performance
    for outcome in test_outcomes:
        tracker.record_prediction_performance(outcome)
    
    print(f"✅ Recorded performance for {len(test_outcomes)} predictions")
    
    # Create test user feedback
    from user_feedback_system import UserFeedback, FeedbackType
    
    test_feedback = [
        UserFeedback(
            feedback_id="test_fb_1",
            prediction_id="test_pred_1",
            user_id="test_user",
            feedback_type=FeedbackType.THUMBS_UP,
            timestamp=time.time(),
            rating=4
        ),
        UserFeedback(
            feedback_id="test_fb_2",
            prediction_id="test_pred_2",
            user_id="test_user",
            feedback_type=FeedbackType.THUMBS_DOWN,
            timestamp=time.time(),
            rating=2
        )
    ]
    
    # Record feedback performance
    for feedback in test_feedback:
        tracker.record_user_feedback_performance(feedback)
    
    print(f"✅ Recorded performance for {len(test_feedback)} feedback entries")
    
    # Get performance summary
    summary = tracker.get_performance_summary(hours_back=1)
    
    print(f"✅ Performance Summary:")
    print(f"  Metrics tracked: {len(summary['metrics'])}")
    for metric_name, stats in summary['metrics'].items():
        print(f"    {metric_name}: mean={stats['mean']:.3f}, count={stats['count']}")
    
    print(f"  Alerts: {len(summary['alerts'])}")
    for alert in summary['alerts']:
        print(f"    {alert['severity']}: {alert['message']}")
    
    print(f"  Recommendations: {len(summary['recommendations'])}")
    for rec in summary['recommendations']:
        print(f"    - {rec}")
    
    # End session
    tracker.end_session()
    
    print("✅ Performance tracking test completed!")


if __name__ == "__main__":
    test_performance_tracking()
