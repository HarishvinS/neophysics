"""
Active Learning System
System actively identifies areas where it needs more training and requests targeted examples.
Implements uncertainty-based and diversity-based active learning strategies.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import random
from collections import defaultdict, Counter

from model_architecture import TextToSceneModel
from user_feedback_system import UserFeedback, FeedbackType, FeedbackCollector
from dynamic_scene_representation import DynamicPhysicsScene
from outcome_based_learning import PredictionOutcome


class LearningStrategy(Enum):
    """Active learning strategies."""
    UNCERTAINTY_SAMPLING = "uncertainty_sampling"
    DIVERSITY_SAMPLING = "diversity_sampling"
    FEEDBACK_DRIVEN = "feedback_driven"
    ERROR_ANALYSIS = "error_analysis"
    COVERAGE_BASED = "coverage_based"


@dataclass
class LearningRequest:
    """Represents a request for additional training data."""
    request_id: str
    strategy: LearningStrategy
    priority: int  # 1-5, where 5 is highest priority
    description: str
    example_prompts: List[str]
    target_concepts: List[str]
    confidence_threshold: float
    timestamp: float
    
    def to_dict(self):
        return {
            'request_id': self.request_id,
            'strategy': self.strategy.value,
            'priority': self.priority,
            'description': self.description,
            'example_prompts': self.example_prompts,
            'target_concepts': self.target_concepts,
            'confidence_threshold': self.confidence_threshold,
            'timestamp': self.timestamp
        }


class UncertaintyAnalyzer:
    """Analyzes model uncertainty to identify learning opportunities."""
    
    def __init__(self, model: TextToSceneModel):
        """Initialize uncertainty analyzer."""
        self.model = model
        self.uncertainty_threshold = 0.3  # Below this confidence, request more data
        self.prediction_history = []
    
    def analyze_prediction_uncertainty(self, text: str, num_samples: int = 5) -> Dict[str, Any]:
        """Analyze uncertainty in model predictions using Monte Carlo dropout."""
        uncertainties = []
        predictions = []
        
        # Enable dropout for uncertainty estimation
        self.model.train()
        
        with torch.no_grad():
            for _ in range(num_samples):
                try:
                    # Get prediction with dropout enabled
                    predicted_scene = self.model.predict_scene(text)
                    predictions.append(predicted_scene)
                    
                    # Calculate some uncertainty metrics (simplified)
                    # In practice, you'd compare object positions, types, etc.
                    uncertainty = random.uniform(0.1, 0.9)  # Placeholder
                    uncertainties.append(uncertainty)
                    
                except Exception as e:
                    print(f"⚠️ Prediction error: {e}")
                    uncertainties.append(1.0)  # High uncertainty for errors
        
        # Return to eval mode
        self.model.eval()
        
        if not uncertainties:
            return {'error': 'No predictions generated'}
        
        analysis = {
            'text': text,
            'mean_uncertainty': np.mean(uncertainties),
            'std_uncertainty': np.std(uncertainties),
            'max_uncertainty': np.max(uncertainties),
            'num_samples': len(uncertainties),
            'needs_more_data': np.mean(uncertainties) > self.uncertainty_threshold
        }
        
        return analysis
    
    def identify_uncertain_concepts(self, text_samples: List[str]) -> List[str]:
        """Identify concepts that the model is uncertain about."""
        uncertain_concepts = []
        
        # Analyze each text sample
        for text in text_samples:
            analysis = self.analyze_prediction_uncertainty(text)
            
            if analysis.get('needs_more_data', False):
                # Extract concepts from uncertain predictions
                concepts = self._extract_concepts(text)
                uncertain_concepts.extend(concepts)
        
        # Return most common uncertain concepts
        concept_counts = Counter(uncertain_concepts)
        return [concept for concept, count in concept_counts.most_common(10)]
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        # Simple keyword extraction (could be enhanced with NLP)
        object_keywords = ['ball', 'sphere', 'box', 'cube', 'ramp', 'cylinder', 'plane']
        spatial_keywords = ['above', 'below', 'left', 'right', 'between', 'near', 'on', 'under']
        material_keywords = ['metal', 'wood', 'rubber', 'plastic', 'glass', 'stone']
        
        concepts = []
        text_lower = text.lower()
        
        for keyword in object_keywords + spatial_keywords + material_keywords:
            if keyword in text_lower:
                concepts.append(keyword)
        
        return concepts


class DiversityAnalyzer:
    """Analyzes training data diversity to identify gaps."""
    
    def __init__(self):
        """Initialize diversity analyzer."""
        self.seen_patterns = set()
        self.concept_coverage = defaultdict(int)
        self.pattern_templates = [
            "create {object}",
            "place {object} {relation} {reference}",
            "add {material} {object}",
            "{object} {relation} {reference} and {reference2}"
        ]
    
    def analyze_coverage_gaps(self, training_texts: List[str]) -> List[str]:
        """Analyze gaps in training data coverage."""
        # Track seen patterns
        for text in training_texts:
            pattern = self._extract_pattern(text)
            self.seen_patterns.add(pattern)
            
            concepts = self._extract_concepts(text)
            for concept in concepts:
                self.concept_coverage[concept] += 1
        
        # Identify underrepresented concepts
        min_coverage = 5  # Minimum examples per concept
        underrepresented = []
        
        for concept, count in self.concept_coverage.items():
            if count < min_coverage:
                underrepresented.append(concept)
        
        return underrepresented
    
    def generate_diversity_requests(self, underrepresented_concepts: List[str]) -> List[str]:
        """Generate requests for more diverse training examples."""
        requests = []
        
        for concept in underrepresented_concepts:
            # Generate example prompts for this concept
            if concept in ['ball', 'sphere', 'box', 'cube', 'ramp', 'cylinder']:
                requests.extend([
                    f"create a {concept}",
                    f"place a {concept} above something",
                    f"add a {concept} next to another object",
                    f"put a {concept} between two objects"
                ])
            elif concept in ['above', 'below', 'left', 'right', 'between', 'near']:
                requests.extend([
                    f"place object {concept} reference",
                    f"put the ball {concept} the box",
                    f"add sphere {concept} the ramp"
                ])
        
        return requests[:20]  # Limit to 20 requests
    
    def _extract_pattern(self, text: str) -> str:
        """Extract structural pattern from text."""
        # Simplified pattern extraction
        text_lower = text.lower()
        
        if 'create' in text_lower:
            return 'create_pattern'
        elif 'place' in text_lower and any(rel in text_lower for rel in ['above', 'below', 'left', 'right']):
            return 'place_relation_pattern'
        elif 'between' in text_lower:
            return 'between_pattern'
        else:
            return 'other_pattern'
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract concepts from text (same as UncertaintyAnalyzer)."""
        object_keywords = ['ball', 'sphere', 'box', 'cube', 'ramp', 'cylinder', 'plane']
        spatial_keywords = ['above', 'below', 'left', 'right', 'between', 'near', 'on', 'under']
        material_keywords = ['metal', 'wood', 'rubber', 'plastic', 'glass', 'stone']
        
        concepts = []
        text_lower = text.lower()
        
        for keyword in object_keywords + spatial_keywords + material_keywords:
            if keyword in text_lower:
                concepts.append(keyword)
        
        return concepts


class ActiveLearningCoordinator:
    """Coordinates active learning strategies and generates learning requests."""
    
    def __init__(self, model: TextToSceneModel, feedback_collector: FeedbackCollector):
        """Initialize active learning coordinator."""
        self.model = model
        self.feedback_collector = feedback_collector
        
        # Analyzers
        self.uncertainty_analyzer = UncertaintyAnalyzer(model)
        self.diversity_analyzer = DiversityAnalyzer()
        
        # Learning requests
        self.active_requests = []
        self.completed_requests = []
        
        # Configuration
        self.max_active_requests = 10
        self.request_cooldown = 3600  # 1 hour between similar requests
    
    def generate_learning_requests(self, recent_predictions: List[str] = None) -> List[LearningRequest]:
        """Generate learning requests based on multiple strategies."""
        new_requests = []
        
        if recent_predictions is None:
            recent_predictions = []
        
        # Strategy 1: Uncertainty-based learning
        uncertainty_requests = self._generate_uncertainty_requests(recent_predictions)
        new_requests.extend(uncertainty_requests)
        
        # Strategy 2: Feedback-driven learning
        feedback_requests = self._generate_feedback_requests()
        new_requests.extend(feedback_requests)
        
        # Strategy 3: Diversity-based learning
        diversity_requests = self._generate_diversity_requests(recent_predictions)
        new_requests.extend(diversity_requests)
        
        # Strategy 4: Error analysis learning
        error_requests = self._generate_error_requests()
        new_requests.extend(error_requests)
        
        # Filter and prioritize requests
        filtered_requests = self._filter_and_prioritize(new_requests)
        
        # Add to active requests
        self.active_requests.extend(filtered_requests)
        
        return filtered_requests
    
    def _generate_uncertainty_requests(self, recent_predictions: List[str]) -> List[LearningRequest]:
        """Generate requests based on prediction uncertainty."""
        requests = []
        
        if not recent_predictions:
            return requests
        
        # Analyze uncertainty in recent predictions
        uncertain_concepts = self.uncertainty_analyzer.identify_uncertain_concepts(recent_predictions)
        
        if uncertain_concepts:
            request = LearningRequest(
                request_id=f"uncertainty_{int(time.time())}",
                strategy=LearningStrategy.UNCERTAINTY_SAMPLING,
                priority=4,
                description=f"Model is uncertain about: {', '.join(uncertain_concepts[:3])}",
                example_prompts=[
                    f"create a {concept}" for concept in uncertain_concepts[:5]
                ],
                target_concepts=uncertain_concepts,
                confidence_threshold=0.7,
                timestamp=time.time()
            )
            requests.append(request)
        
        return requests
    
    def _generate_feedback_requests(self) -> List[LearningRequest]:
        """Generate requests based on user feedback."""
        requests = []
        
        # Get recent negative feedback
        negative_feedback = self.feedback_collector.get_negative_feedback()
        recent_negative = [f for f in negative_feedback if f.timestamp > time.time() - 86400]  # Last 24 hours
        
        if len(recent_negative) >= 3:  # Threshold for generating request
            # Extract common issues
            common_texts = [f.original_text for f in recent_negative if f.original_text]
            
            if common_texts:
                request = LearningRequest(
                    request_id=f"feedback_{int(time.time())}",
                    strategy=LearningStrategy.FEEDBACK_DRIVEN,
                    priority=5,  # High priority for user feedback
                    description=f"Users reported issues with {len(recent_negative)} predictions",
                    example_prompts=common_texts[:5],
                    target_concepts=['user_feedback_improvement'],
                    confidence_threshold=0.8,
                    timestamp=time.time()
                )
                requests.append(request)
        
        return requests
    
    def _generate_diversity_requests(self, recent_predictions: List[str]) -> List[LearningRequest]:
        """Generate requests to improve training data diversity."""
        requests = []
        
        # Analyze coverage gaps
        underrepresented = self.diversity_analyzer.analyze_coverage_gaps(recent_predictions)
        
        if underrepresented:
            diversity_prompts = self.diversity_analyzer.generate_diversity_requests(underrepresented)
            
            request = LearningRequest(
                request_id=f"diversity_{int(time.time())}",
                strategy=LearningStrategy.DIVERSITY_SAMPLING,
                priority=3,
                description=f"Need more examples for: {', '.join(underrepresented[:3])}",
                example_prompts=diversity_prompts,
                target_concepts=underrepresented,
                confidence_threshold=0.6,
                timestamp=time.time()
            )
            requests.append(request)
        
        return requests
    
    def _generate_error_requests(self) -> List[LearningRequest]:
        """Generate requests based on error analysis."""
        requests = []
        
        # This would analyze prediction errors and generate targeted requests
        # For now, return a placeholder request
        
        request = LearningRequest(
            request_id=f"error_{int(time.time())}",
            strategy=LearningStrategy.ERROR_ANALYSIS,
            priority=2,
            description="Request for error-prone scenarios",
            example_prompts=[
                "create complex multi-object scenes",
                "place objects with unusual spatial relationships",
                "use ambiguous language descriptions"
            ],
            target_concepts=['error_reduction'],
            confidence_threshold=0.9,
            timestamp=time.time()
        )
        requests.append(request)
        
        return requests
    
    def _filter_and_prioritize(self, requests: List[LearningRequest]) -> List[LearningRequest]:
        """Filter and prioritize learning requests."""
        # Remove duplicates and apply cooldown
        filtered = []
        
        for request in requests:
            # Check if similar request exists recently
            similar_exists = any(
                r.strategy == request.strategy and 
                time.time() - r.timestamp < self.request_cooldown
                for r in self.active_requests
            )
            
            if not similar_exists:
                filtered.append(request)
        
        # Sort by priority (highest first)
        filtered.sort(key=lambda r: r.priority, reverse=True)
        
        # Limit number of active requests
        remaining_slots = self.max_active_requests - len(self.active_requests)
        return filtered[:remaining_slots]
    
    def get_active_requests(self) -> List[LearningRequest]:
        """Get currently active learning requests."""
        # Remove expired requests
        current_time = time.time()
        self.active_requests = [
            r for r in self.active_requests 
            if current_time - r.timestamp < 86400  # 24 hour expiry
        ]
        
        return self.active_requests
    
    def mark_request_completed(self, request_id: str):
        """Mark a learning request as completed."""
        for i, request in enumerate(self.active_requests):
            if request.request_id == request_id:
                completed_request = self.active_requests.pop(i)
                self.completed_requests.append(completed_request)
                print(f"✅ Completed learning request: {request_id}")
                break
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of active learning status."""
        return {
            'active_requests': len(self.active_requests),
            'completed_requests': len(self.completed_requests),
            'strategies_used': list(set(r.strategy.value for r in self.active_requests)),
            'high_priority_requests': len([r for r in self.active_requests if r.priority >= 4]),
            'total_target_concepts': len(set(
                concept for r in self.active_requests for concept in r.target_concepts
            ))
        }


def test_active_learning_system():
    """Test the active learning system."""
    print("Testing Active Learning System...")
    
    # Create test model and feedback collector
    from model_architecture import ModelConfig
    config = ModelConfig()
    model = TextToSceneModel(hidden_size=config.hidden_size, max_objects=config.max_objects)
    
    feedback_collector = FeedbackCollector()
    
    # Add some test feedback
    from user_feedback_system import UserFeedback
    test_feedback = [
        UserFeedback(
            feedback_id="test_1",
            prediction_id="pred_1",
            user_id="test_user",
            feedback_type=FeedbackType.THUMBS_DOWN,
            timestamp=time.time(),
            rating=2,
            text_feedback="Wrong object placement",
            original_text="place ball above box"
        ),
        UserFeedback(
            feedback_id="test_2",
            prediction_id="pred_2",
            user_id="test_user",
            feedback_type=FeedbackType.THUMBS_DOWN,
            timestamp=time.time(),
            rating=1,
            text_feedback="Physics doesn't make sense",
            original_text="create floating sphere"
        )
    ]
    
    for feedback in test_feedback:
        feedback_collector.collect_feedback(feedback)
    
    # Create active learning coordinator
    coordinator = ActiveLearningCoordinator(model, feedback_collector)
    
    # Test recent predictions
    recent_predictions = [
        "create a ball",
        "place sphere above box",
        "add ramp next to cube"
    ]
    
    # Generate learning requests
    requests = coordinator.generate_learning_requests(recent_predictions)
    
    print(f"✅ Generated {len(requests)} learning requests:")
    for request in requests:
        print(f"  {request.strategy.value} (priority {request.priority}): {request.description}")
        print(f"    Example prompts: {request.example_prompts[:2]}")
    
    # Test uncertainty analysis
    uncertainty_analyzer = UncertaintyAnalyzer(model)
    uncertainty = uncertainty_analyzer.analyze_prediction_uncertainty("create a mysterious object")
    
    print(f"✅ Uncertainty analysis:")
    print(f"  Mean uncertainty: {uncertainty.get('mean_uncertainty', 0):.3f}")
    print(f"  Needs more data: {uncertainty.get('needs_more_data', False)}")
    
    # Test diversity analysis
    diversity_analyzer = DiversityAnalyzer()
    gaps = diversity_analyzer.analyze_coverage_gaps(recent_predictions)
    
    print(f"✅ Coverage gaps identified: {gaps}")
    
    # Get learning summary
    summary = coordinator.get_learning_summary()
    print(f"✅ Learning summary: {summary}")
    
    print("✅ Active learning system test completed!")


if __name__ == "__main__":
    test_active_learning_system()
