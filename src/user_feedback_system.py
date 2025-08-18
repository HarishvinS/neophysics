"""
User Feedback System
Creates interface for users to provide feedback on predictions and guide learning.
Enables human-in-the-loop learning for continuous improvement.
"""

import torch
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from collections import deque

from dynamic_scene_representation import DynamicPhysicsScene
from outcome_based_learning import PredictionOutcome


class FeedbackType(Enum):
    """Types of feedback users can provide."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    CORRECTION = "correction"
    SUGGESTION = "suggestion"
    RATING = "rating"
    DETAILED = "detailed"


@dataclass
class UserFeedback:
    """Represents user feedback on a prediction."""
    feedback_id: str
    prediction_id: str
    user_id: str
    feedback_type: FeedbackType
    timestamp: float
    
    # Feedback content
    rating: Optional[int] = None  # 1-5 scale
    text_feedback: Optional[str] = None
    corrections: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    
    # Context
    original_text: Optional[str] = None
    predicted_scene: Optional[Dict] = None
    
    def to_dict(self):
        return {
            'feedback_id': self.feedback_id,
            'prediction_id': self.prediction_id,
            'user_id': self.user_id,
            'feedback_type': self.feedback_type.value,
            'timestamp': self.timestamp,
            'rating': self.rating,
            'text_feedback': self.text_feedback,
            'corrections': self.corrections,
            'suggestions': self.suggestions,
            'original_text': self.original_text,
            'predicted_scene': self.predicted_scene
        }


class FeedbackCollector:
    """Collects and manages user feedback."""
    
    def __init__(self, max_feedback_history: int = 10000):
        """Initialize feedback collector."""
        self.max_feedback_history = max_feedback_history
        self.feedback_history = deque(maxlen=max_feedback_history)
        self.feedback_callbacks: List[Callable[[UserFeedback], None]] = []
        
        # Feedback statistics
        self.stats = {
            'total_feedback': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'corrections_provided': 0,
            'average_rating': 0.0
        }
    
    def add_feedback_callback(self, callback: Callable[[UserFeedback], None]):
        """Add callback to be called when feedback is received."""
        self.feedback_callbacks.append(callback)
    
    def collect_feedback(self, feedback: UserFeedback):
        """Collect user feedback."""
        # Store feedback
        self.feedback_history.append(feedback)
        
        # Update statistics
        self._update_stats(feedback)
        
        # Trigger callbacks
        for callback in self.feedback_callbacks:
            try:
                callback(feedback)
            except Exception as e:
                print(f"⚠️ Feedback callback error: {e}")
        
        print(f"📝 Collected {feedback.feedback_type.value} feedback for prediction {feedback.prediction_id}")
    
    def _update_stats(self, feedback: UserFeedback):
        """Update feedback statistics."""
        self.stats['total_feedback'] += 1
        
        if feedback.feedback_type == FeedbackType.THUMBS_UP:
            self.stats['positive_feedback'] += 1
        elif feedback.feedback_type == FeedbackType.THUMBS_DOWN:
            self.stats['negative_feedback'] += 1
        
        if feedback.corrections:
            self.stats['corrections_provided'] += 1
        
        if feedback.rating:
            # Update average rating
            total_ratings = sum(1 for f in self.feedback_history if f.rating)
            sum_ratings = sum(f.rating for f in self.feedback_history if f.rating)
            self.stats['average_rating'] = sum_ratings / total_ratings if total_ratings > 0 else 0.0
    
    def get_feedback_for_prediction(self, prediction_id: str) -> List[UserFeedback]:
        """Get all feedback for a specific prediction."""
        return [f for f in self.feedback_history if f.prediction_id == prediction_id]
    
    def get_recent_feedback(self, hours: int = 24) -> List[UserFeedback]:
        """Get feedback from the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [f for f in self.feedback_history if f.timestamp > cutoff_time]
    
    def get_negative_feedback(self) -> List[UserFeedback]:
        """Get all negative feedback for learning."""
        return [f for f in self.feedback_history 
                if f.feedback_type in [FeedbackType.THUMBS_DOWN, FeedbackType.CORRECTION]
                or (f.rating and f.rating <= 2)]
    
    def export_feedback(self, filepath: str):
        """Export feedback to JSON file."""
        feedback_data = {
            'metadata': {
                'total_feedback': len(self.feedback_history),
                'exported_at': time.time(),
                'stats': self.stats
            },
            'feedback': [f.to_dict() for f in self.feedback_history]
        }
        
        with open(filepath, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        print(f"📁 Exported {len(self.feedback_history)} feedback entries to {filepath}")


class FeedbackGUI:
    """GUI for collecting user feedback."""
    
    def __init__(self, feedback_collector: FeedbackCollector):
        """Initialize feedback GUI."""
        self.feedback_collector = feedback_collector
        self.current_prediction = None
        self.user_id = "user_001"  # Could be configurable
        
        # Create GUI
        self.root = tk.Toplevel()
        self.root.title("🔄 Physics Engine Feedback")
        self.root.geometry("500x600")
        self.root.withdraw()  # Hide initially
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the feedback GUI."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🔄 Provide Feedback", font=('Arial', 14, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Prediction display
        self.prediction_frame = ttk.LabelFrame(main_frame, text="Current Prediction", padding="10")
        self.prediction_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        self.prediction_frame.columnconfigure(0, weight=1)
        
        self.prediction_text = tk.Text(self.prediction_frame, height=4, wrap=tk.WORD, state=tk.DISABLED)
        self.prediction_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Quick feedback buttons
        quick_frame = ttk.LabelFrame(main_frame, text="Quick Feedback", padding="10")
        quick_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        
        button_frame = ttk.Frame(quick_frame)
        button_frame.grid(row=0, column=0)
        
        ttk.Button(button_frame, text="👍 Good", command=self.thumbs_up).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="👎 Bad", command=self.thumbs_down).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="🔧 Needs Fix", command=self.needs_correction).grid(row=0, column=2, padx=5)
        
        # Rating
        rating_frame = ttk.LabelFrame(main_frame, text="Rating (1-5)", padding="10")
        rating_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.rating_var = tk.IntVar(value=3)
        rating_buttons = ttk.Frame(rating_frame)
        rating_buttons.grid(row=0, column=0)
        
        for i in range(1, 6):
            ttk.Radiobutton(rating_buttons, text=str(i), variable=self.rating_var, value=i).grid(row=0, column=i-1, padx=5)
        
        # Text feedback
        text_frame = ttk.LabelFrame(main_frame, text="Additional Comments", padding="10")
        text_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 20))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        self.feedback_text = tk.Text(text_frame, height=6, wrap=tk.WORD)
        self.feedback_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Submit button
        ttk.Button(main_frame, text="Submit Feedback", command=self.submit_feedback).grid(row=5, column=0, pady=10)
        
        # Close button
        ttk.Button(main_frame, text="Close", command=self.hide).grid(row=6, column=0)
    
    def show_for_prediction(self, prediction_id: str, text_input: str, predicted_scene: DynamicPhysicsScene):
        """Show feedback GUI for a specific prediction."""
        self.current_prediction = {
            'id': prediction_id,
            'text': text_input,
            'scene': predicted_scene.to_dict() if predicted_scene else None
        }
        
        # Update prediction display
        self.prediction_text.config(state=tk.NORMAL)
        self.prediction_text.delete(1.0, tk.END)
        self.prediction_text.insert(1.0, f"Input: {text_input}\n\n")
        if predicted_scene:
            self.prediction_text.insert(tk.END, f"Objects created: {predicted_scene.get_object_count()}\n")
            for obj in list(predicted_scene.objects.values())[:3]:  # Show first 3 objects
                self.prediction_text.insert(tk.END, f"  {obj.object_id}: {obj.object_type.value} at {obj.position.to_list()}\n")
        self.prediction_text.config(state=tk.DISABLED)
        
        # Reset form
        self.rating_var.set(3)
        self.feedback_text.delete(1.0, tk.END)
        
        # Show window
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
    
    def thumbs_up(self):
        """Handle thumbs up feedback."""
        self._submit_quick_feedback(FeedbackType.THUMBS_UP)
    
    def thumbs_down(self):
        """Handle thumbs down feedback."""
        self._submit_quick_feedback(FeedbackType.THUMBS_DOWN)
    
    def needs_correction(self):
        """Handle correction needed feedback."""
        self._submit_quick_feedback(FeedbackType.CORRECTION)
    
    def _submit_quick_feedback(self, feedback_type: FeedbackType):
        """Submit quick feedback and close."""
        if not self.current_prediction:
            return
        
        feedback = UserFeedback(
            feedback_id=f"feedback_{int(time.time() * 1000)}",
            prediction_id=self.current_prediction['id'],
            user_id=self.user_id,
            feedback_type=feedback_type,
            timestamp=time.time(),
            original_text=self.current_prediction['text'],
            predicted_scene=self.current_prediction['scene']
        )
        
        self.feedback_collector.collect_feedback(feedback)
        self.hide()
    
    def submit_feedback(self):
        """Submit detailed feedback."""
        if not self.current_prediction:
            return
        
        text_feedback = self.feedback_text.get(1.0, tk.END).strip()
        rating = self.rating_var.get()
        
        feedback = UserFeedback(
            feedback_id=f"feedback_{int(time.time() * 1000)}",
            prediction_id=self.current_prediction['id'],
            user_id=self.user_id,
            feedback_type=FeedbackType.DETAILED,
            timestamp=time.time(),
            rating=rating,
            text_feedback=text_feedback if text_feedback else None,
            original_text=self.current_prediction['text'],
            predicted_scene=self.current_prediction['scene']
        )
        
        self.feedback_collector.collect_feedback(feedback)
        messagebox.showinfo("Feedback", "Thank you for your feedback!")
        self.hide()
    
    def hide(self):
        """Hide the feedback window."""
        self.root.withdraw()


class FeedbackAnalyzer:
    """Analyzes user feedback to extract learning insights."""
    
    def __init__(self):
        """Initialize feedback analyzer."""
        self.analysis_cache = {}
    
    def analyze_feedback_trends(self, feedback_list: List[UserFeedback]) -> Dict[str, Any]:
        """Analyze trends in user feedback."""
        if not feedback_list:
            return {'error': 'No feedback to analyze'}
        
        analysis = {
            'total_feedback': len(feedback_list),
            'feedback_types': {},
            'rating_distribution': {},
            'common_issues': [],
            'improvement_areas': [],
            'satisfaction_trend': 'stable'
        }
        
        # Analyze feedback types
        for feedback in feedback_list:
            ftype = feedback.feedback_type.value
            analysis['feedback_types'][ftype] = analysis['feedback_types'].get(ftype, 0) + 1
        
        # Analyze ratings
        ratings = [f.rating for f in feedback_list if f.rating]
        if ratings:
            for rating in range(1, 6):
                count = ratings.count(rating)
                analysis['rating_distribution'][rating] = count
            
            analysis['average_rating'] = sum(ratings) / len(ratings)
            analysis['satisfaction_level'] = 'high' if analysis['average_rating'] >= 4 else 'medium' if analysis['average_rating'] >= 3 else 'low'
        
        # Extract common issues from text feedback
        text_feedback = [f.text_feedback for f in feedback_list if f.text_feedback]
        if text_feedback:
            # Simple keyword analysis (could be enhanced with NLP)
            issue_keywords = ['wrong', 'incorrect', 'bad', 'error', 'fix', 'improve', 'better']
            common_words = {}
            
            for text in text_feedback:
                words = text.lower().split()
                for word in words:
                    if word in issue_keywords:
                        common_words[word] = common_words.get(word, 0) + 1
            
            analysis['common_issues'] = sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Determine improvement areas
        negative_feedback = [f for f in feedback_list if f.feedback_type in [FeedbackType.THUMBS_DOWN, FeedbackType.CORRECTION]]
        if negative_feedback:
            analysis['improvement_areas'] = [
                'object_positioning' if any('position' in (f.text_feedback or '') for f in negative_feedback) else None,
                'object_types' if any('object' in (f.text_feedback or '') for f in negative_feedback) else None,
                'physics_realism' if any('physics' in (f.text_feedback or '') for f in negative_feedback) else None
            ]
            analysis['improvement_areas'] = [area for area in analysis['improvement_areas'] if area]
        
        return analysis
    
    def extract_learning_targets(self, feedback_list: List[UserFeedback]) -> List[Dict[str, Any]]:
        """Extract specific learning targets from feedback."""
        learning_targets = []
        
        # Find feedback with corrections
        correction_feedback = [f for f in feedback_list if f.corrections or f.feedback_type == FeedbackType.CORRECTION]
        
        for feedback in correction_feedback:
            target = {
                'feedback_id': feedback.feedback_id,
                'original_text': feedback.original_text,
                'issue_type': 'correction_needed',
                'priority': 'high' if feedback.rating and feedback.rating <= 2 else 'medium',
                'suggested_improvements': []
            }
            
            if feedback.text_feedback:
                # Extract improvement suggestions from text
                if 'position' in feedback.text_feedback.lower():
                    target['suggested_improvements'].append('improve_object_positioning')
                if 'physics' in feedback.text_feedback.lower():
                    target['suggested_improvements'].append('improve_physics_realism')
                if 'object' in feedback.text_feedback.lower():
                    target['suggested_improvements'].append('improve_object_selection')
            
            learning_targets.append(target)
        
        return learning_targets


def test_user_feedback_system():
    """Test the user feedback system."""
    print("Testing User Feedback System...")
    
    # Create feedback collector
    collector = FeedbackCollector()
    
    # Add callback to monitor feedback
    def feedback_received(feedback: UserFeedback):
        print(f"📝 Received {feedback.feedback_type.value} feedback: {feedback.text_feedback}")
    
    collector.add_feedback_callback(feedback_received)
    
    # Create test feedback
    test_feedback = [
        UserFeedback(
            feedback_id="test_1",
            prediction_id="pred_1",
            user_id="test_user",
            feedback_type=FeedbackType.THUMBS_UP,
            timestamp=time.time(),
            rating=4,
            text_feedback="Good prediction!",
            original_text="create a ball"
        ),
        UserFeedback(
            feedback_id="test_2",
            prediction_id="pred_2",
            user_id="test_user",
            feedback_type=FeedbackType.THUMBS_DOWN,
            timestamp=time.time(),
            rating=2,
            text_feedback="Wrong position for the object",
            original_text="place ball above box"
        )
    ]
    
    # Collect feedback
    for feedback in test_feedback:
        collector.collect_feedback(feedback)
    
    print(f"✅ Collected {len(test_feedback)} feedback entries")
    print(f"Stats: {collector.stats}")
    
    # Test feedback analysis
    analyzer = FeedbackAnalyzer()
    analysis = analyzer.analyze_feedback_trends(list(collector.feedback_history))
    
    print(f"✅ Feedback analysis:")
    print(f"  Average rating: {analysis.get('average_rating', 0):.2f}")
    print(f"  Satisfaction level: {analysis.get('satisfaction_level', 'unknown')}")
    print(f"  Common issues: {analysis.get('common_issues', [])}")
    
    # Extract learning targets
    targets = analyzer.extract_learning_targets(list(collector.feedback_history))
    print(f"✅ Learning targets: {len(targets)} identified")
    
    print("✅ User feedback system test completed!")


if __name__ == "__main__":
    test_user_feedback_system()
