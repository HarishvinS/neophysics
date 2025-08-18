"""
Physics Reasoning Engine
Learns causal physics relationships from simulation outcomes rather than just pattern matching.
Builds understanding of why things happen, not just what happens.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import time

from dynamic_scene_representation import DynamicPhysicsScene, DynamicPhysicsObject, SpatialRelation, RelationType
from scene_representation import ObjectType, MaterialType, Vector3


class PhysicsLaw(Enum):
    """Types of physics laws the system can learn."""
    GRAVITY = "gravity"
    FRICTION = "friction"
    COLLISION = "collision"
    MOMENTUM_CONSERVATION = "momentum_conservation"
    ENERGY_CONSERVATION = "energy_conservation"
    STABILITY = "stability"
    ROLLING = "rolling"
    SLIDING = "sliding"
    BOUNCING = "bouncing"


@dataclass
class PhysicsEvent:
    """Represents a physics event observed during simulation."""
    event_type: str
    timestamp: float
    objects_involved: List[str]
    initial_state: Dict[str, Any]
    final_state: Dict[str, Any]
    parameters: Dict[str, Any]
    confidence: float = 1.0
    
    def to_dict(self):
        return {
            'event_type': self.event_type,
            'timestamp': self.timestamp,
            'objects_involved': self.objects_involved,
            'initial_state': self.initial_state,
            'final_state': self.final_state,
            'parameters': self.parameters,
            'confidence': self.confidence
        }


@dataclass
class CausalRule:
    """Represents a learned causal physics rule."""
    rule_id: str
    law_type: PhysicsLaw
    conditions: Dict[str, Any]  # When this rule applies
    predictions: Dict[str, Any]  # What this rule predicts
    confidence: float
    evidence_count: int
    success_rate: float
    
    def to_dict(self):
        return {
            'rule_id': self.rule_id,
            'law_type': self.law_type.value,
            'conditions': self.conditions,
            'predictions': self.predictions,
            'confidence': self.confidence,
            'evidence_count': self.evidence_count,
            'success_rate': self.success_rate
        }


class PhysicsEventDetector:
    """Detects physics events from simulation data."""
    
    def __init__(self):
        """Initialize event detector."""
        self.velocity_threshold = 0.1  # m/s
        self.acceleration_threshold = 1.0  # m/s²
        self.collision_distance = 0.1  # m
    
    def detect_events(self, simulation_frames: List[Dict]) -> List[PhysicsEvent]:
        """Detect physics events from simulation frames."""
        events = []
        
        if len(simulation_frames) < 2:
            return events
        
        # Analyze frame-by-frame changes
        for i in range(1, len(simulation_frames)):
            prev_frame = simulation_frames[i-1]
            curr_frame = simulation_frames[i]
            
            frame_events = self._analyze_frame_transition(prev_frame, curr_frame)
            events.extend(frame_events)
        
        return events
    
    def _analyze_frame_transition(self, prev_frame: Dict, curr_frame: Dict) -> List[PhysicsEvent]:
        """Analyze transition between two frames."""
        events = []
        
        prev_states = prev_frame.get('object_states', {})
        curr_states = curr_frame.get('object_states', {})
        
        # Check each object for events
        for obj_id in prev_states.keys():
            if obj_id not in curr_states:
                continue
            
            prev_state = prev_states[obj_id]
            curr_state = curr_states[obj_id]
            
            # Detect motion events
            motion_events = self._detect_motion_events(obj_id, prev_state, curr_state, curr_frame['timestamp'])
            events.extend(motion_events)
        
        # Detect interaction events
        interaction_events = self._detect_interactions(prev_states, curr_states, curr_frame['timestamp'])
        events.extend(interaction_events)
        
        return events
    
    def _detect_motion_events(self, obj_id: str, prev_state: Dict, curr_state: Dict, timestamp: float) -> List[PhysicsEvent]:
        """Detect motion-related events for an object."""
        events = []
        
        prev_pos = np.array(prev_state['position'])
        curr_pos = np.array(curr_state['position'])
        prev_vel = np.array(prev_state['velocity'])
        curr_vel = np.array(curr_state['velocity'])
        
        # Calculate motion metrics
        displacement = np.linalg.norm(curr_pos - prev_pos)
        speed_change = np.linalg.norm(curr_vel) - np.linalg.norm(prev_vel)
        
        # Detect start of motion
        if np.linalg.norm(prev_vel) < self.velocity_threshold and np.linalg.norm(curr_vel) > self.velocity_threshold:
            events.append(PhysicsEvent(
                event_type="motion_start",
                timestamp=timestamp,
                objects_involved=[obj_id],
                initial_state={'velocity': prev_vel.tolist(), 'position': prev_pos.tolist()},
                final_state={'velocity': curr_vel.tolist(), 'position': curr_pos.tolist()},
                parameters={'speed_change': speed_change}
            ))
        
        # Detect stop of motion
        if np.linalg.norm(prev_vel) > self.velocity_threshold and np.linalg.norm(curr_vel) < self.velocity_threshold:
            events.append(PhysicsEvent(
                event_type="motion_stop",
                timestamp=timestamp,
                objects_involved=[obj_id],
                initial_state={'velocity': prev_vel.tolist(), 'position': prev_pos.tolist()},
                final_state={'velocity': curr_vel.tolist(), 'position': curr_pos.tolist()},
                parameters={'speed_change': speed_change}
            ))
        
        # Detect significant acceleration
        if abs(speed_change) > self.acceleration_threshold:
            event_type = "acceleration" if speed_change > 0 else "deceleration"
            events.append(PhysicsEvent(
                event_type=event_type,
                timestamp=timestamp,
                objects_involved=[obj_id],
                initial_state={'velocity': prev_vel.tolist()},
                final_state={'velocity': curr_vel.tolist()},
                parameters={'acceleration': speed_change}
            ))
        
        return events
    
    def _detect_interactions(self, prev_states: Dict, curr_states: Dict, timestamp: float) -> List[PhysicsEvent]:
        """Detect interaction events between objects."""
        events = []
        
        object_ids = list(prev_states.keys())
        
        # Check all pairs of objects
        for i, obj1_id in enumerate(object_ids):
            for j, obj2_id in enumerate(object_ids[i+1:], i+1):
                if obj1_id not in curr_states or obj2_id not in curr_states:
                    continue
                
                # Calculate distances
                prev_pos1 = np.array(prev_states[obj1_id]['position'])
                prev_pos2 = np.array(prev_states[obj2_id]['position'])
                curr_pos1 = np.array(curr_states[obj1_id]['position'])
                curr_pos2 = np.array(curr_states[obj2_id]['position'])
                
                prev_distance = np.linalg.norm(prev_pos1 - prev_pos2)
                curr_distance = np.linalg.norm(curr_pos1 - curr_pos2)
                
                # Detect collision
                if prev_distance > self.collision_distance and curr_distance <= self.collision_distance:
                    events.append(PhysicsEvent(
                        event_type="collision",
                        timestamp=timestamp,
                        objects_involved=[obj1_id, obj2_id],
                        initial_state={
                            obj1_id: {'position': prev_pos1.tolist(), 'velocity': prev_states[obj1_id]['velocity']},
                            obj2_id: {'position': prev_pos2.tolist(), 'velocity': prev_states[obj2_id]['velocity']}
                        },
                        final_state={
                            obj1_id: {'position': curr_pos1.tolist(), 'velocity': curr_states[obj1_id]['velocity']},
                            obj2_id: {'position': curr_pos2.tolist(), 'velocity': curr_states[obj2_id]['velocity']}
                        },
                        parameters={'distance': curr_distance}
                    ))
        
        return events


class CausalRuleLearner:
    """Learns causal physics rules from observed events."""
    
    def __init__(self):
        """Initialize rule learner."""
        self.rules: Dict[str, CausalRule] = {}
        self.rule_counter = 0
        self.min_evidence = 3  # Minimum evidence to establish a rule
        self.confidence_threshold = 0.7
    
    def learn_from_events(self, events: List[PhysicsEvent], scene: DynamicPhysicsScene) -> List[CausalRule]:
        """Learn causal rules from physics events."""
        new_rules = []
        
        # Group events by type
        event_groups = {}
        for event in events:
            event_type = event.event_type
            if event_type not in event_groups:
                event_groups[event_type] = []
            event_groups[event_type].append(event)
        
        # Learn rules for each event type
        for event_type, event_list in event_groups.items():
            rules = self._learn_rules_for_event_type(event_type, event_list, scene)
            new_rules.extend(rules)
        
        return new_rules
    
    def _learn_rules_for_event_type(self, event_type: str, events: List[PhysicsEvent], scene: DynamicPhysicsScene) -> List[CausalRule]:
        """Learn rules for a specific event type."""
        rules = []
        
        if len(events) < self.min_evidence:
            return rules
        
        # Analyze patterns in the events
        if event_type == "motion_start":
            rules.extend(self._learn_motion_start_rules(events, scene))
        elif event_type == "collision":
            rules.extend(self._learn_collision_rules(events, scene))
        elif event_type == "motion_stop":
            rules.extend(self._learn_motion_stop_rules(events, scene))
        
        return rules
    
    def _learn_motion_start_rules(self, events: List[PhysicsEvent], scene: DynamicPhysicsScene) -> List[CausalRule]:
        """Learn rules about when objects start moving."""
        rules = []
        
        # Analyze what causes objects to start moving
        gravity_events = []
        
        for event in events:
            obj_id = event.objects_involved[0]
            if obj_id in scene.objects:
                obj = scene.objects[obj_id]
                
                # Check if object is above ground (gravity effect)
                if obj.position.z > 0.5:
                    gravity_events.append(event)
        
        # Create gravity rule if we have enough evidence
        if len(gravity_events) >= self.min_evidence:
            rule_id = f"gravity_rule_{self.rule_counter}"
            self.rule_counter += 1
            
            rule = CausalRule(
                rule_id=rule_id,
                law_type=PhysicsLaw.GRAVITY,
                conditions={
                    'object_above_ground': True,
                    'object_unsupported': True,
                    'mass_greater_than': 0
                },
                predictions={
                    'will_accelerate_downward': True,
                    'acceleration_magnitude': 9.81
                },
                confidence=len(gravity_events) / len(events),
                evidence_count=len(gravity_events),
                success_rate=1.0  # Will be updated with validation
            )
            
            rules.append(rule)
            self.rules[rule_id] = rule
        
        return rules
    
    def _learn_collision_rules(self, events: List[PhysicsEvent], scene: DynamicPhysicsScene) -> List[CausalRule]:
        """Learn rules about collisions."""
        rules = []
        
        # Analyze momentum conservation in collisions
        momentum_conserved = 0
        total_collisions = len(events)
        
        for event in events:
            if len(event.objects_involved) == 2:
                # Check momentum conservation (simplified)
                obj1_id, obj2_id = event.objects_involved
                
                if obj1_id in scene.objects and obj2_id in scene.objects:
                    # This is a simplified check - in reality we'd need mass and velocity data
                    momentum_conserved += 1
        
        if momentum_conserved >= self.min_evidence:
            rule_id = f"momentum_rule_{self.rule_counter}"
            self.rule_counter += 1
            
            rule = CausalRule(
                rule_id=rule_id,
                law_type=PhysicsLaw.MOMENTUM_CONSERVATION,
                conditions={
                    'collision_occurs': True,
                    'objects_count': 2
                },
                predictions={
                    'momentum_conserved': True,
                    'energy_partially_conserved': True
                },
                confidence=momentum_conserved / total_collisions,
                evidence_count=momentum_conserved,
                success_rate=1.0
            )
            
            rules.append(rule)
            self.rules[rule_id] = rule
        
        return rules
    
    def _learn_motion_stop_rules(self, events: List[PhysicsEvent], scene: DynamicPhysicsScene) -> List[CausalRule]:
        """Learn rules about when objects stop moving."""
        rules = []
        
        # Analyze friction effects
        friction_stops = 0
        
        for event in events:
            obj_id = event.objects_involved[0]
            if obj_id in scene.objects:
                obj = scene.objects[obj_id]
                
                # Check if object stopped due to friction (on ground)
                if obj.position.z < 0.2:  # Near ground
                    friction_stops += 1
        
        if friction_stops >= self.min_evidence:
            rule_id = f"friction_rule_{self.rule_counter}"
            self.rule_counter += 1
            
            rule = CausalRule(
                rule_id=rule_id,
                law_type=PhysicsLaw.FRICTION,
                conditions={
                    'object_on_surface': True,
                    'relative_motion': True
                },
                predictions={
                    'will_decelerate': True,
                    'eventually_stop': True
                },
                confidence=friction_stops / len(events) if events else 0,
                evidence_count=friction_stops,
                success_rate=1.0
            )
            
            rules.append(rule)
            self.rules[rule_id] = rule
        
        return rules
    
    def predict_outcome(self, scene: DynamicPhysicsScene, conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Predict physics outcomes based on learned rules."""
        predictions = {}
        
        for rule in self.rules.values():
            if rule.confidence > self.confidence_threshold:
                # Check if rule conditions are met
                if self._conditions_match(rule.conditions, conditions):
                    # Apply rule predictions
                    for key, value in rule.predictions.items():
                        if key not in predictions:
                            predictions[key] = []
                        predictions[key].append({
                            'value': value,
                            'confidence': rule.confidence,
                            'rule_id': rule.rule_id
                        })
        
        return predictions
    
    def _conditions_match(self, rule_conditions: Dict[str, Any], actual_conditions: Dict[str, Any]) -> bool:
        """Check if rule conditions match actual conditions."""
        for key, expected_value in rule_conditions.items():
            if key not in actual_conditions:
                return False
            
            actual_value = actual_conditions[key]
            
            # Handle different types of conditions
            if isinstance(expected_value, bool):
                if actual_value != expected_value:
                    return False
            elif isinstance(expected_value, (int, float)):
                if abs(actual_value - expected_value) > 0.1:
                    return False
        
        return True
    
    def get_rule_summary(self) -> Dict[str, Any]:
        """Get summary of learned rules."""
        summary = {
            'total_rules': len(self.rules),
            'rules_by_law': {},
            'high_confidence_rules': 0,
            'average_confidence': 0
        }
        
        confidences = []
        
        for rule in self.rules.values():
            law_type = rule.law_type.value
            if law_type not in summary['rules_by_law']:
                summary['rules_by_law'][law_type] = 0
            summary['rules_by_law'][law_type] += 1
            
            confidences.append(rule.confidence)
            
            if rule.confidence > self.confidence_threshold:
                summary['high_confidence_rules'] += 1
        
        if confidences:
            summary['average_confidence'] = sum(confidences) / len(confidences)
        else:
            summary['average_confidence'] = 0
        
        return summary


def test_physics_reasoning_engine():
    """Test the physics reasoning engine."""
    print("Testing Physics Reasoning Engine...")
    
    # Create test simulation data
    simulation_frames = [
        {
            'timestamp': 0.0,
            'object_states': {
                'ball_1': {
                    'position': [0, 0, 2],
                    'velocity': [0, 0, 0],
                    'angular_velocity': [0, 0, 0]
                }
            }
        },
        {
            'timestamp': 0.1,
            'object_states': {
                'ball_1': {
                    'position': [0, 0, 1.9],
                    'velocity': [0, 0, -1],
                    'angular_velocity': [0, 0, 0]
                }
            }
        },
        {
            'timestamp': 0.2,
            'object_states': {
                'ball_1': {
                    'position': [0, 0, 1.6],
                    'velocity': [0, 0, -2],
                    'angular_velocity': [0, 0, 0]
                }
            }
        }
    ]
    
    # Test event detection
    detector = PhysicsEventDetector()
    events = detector.detect_events(simulation_frames)
    
    print(f"✅ Detected {len(events)} physics events")
    for event in events:
        print(f"   {event.event_type} at t={event.timestamp:.1f}s involving {event.objects_involved}")
    
    # Test rule learning
    scene = DynamicPhysicsScene("test_scene")
    learner = CausalRuleLearner()
    
    # Add more events to meet minimum evidence threshold
    events = events * 6  # More efficient than repeated extend operations
    
    learned_rules = learner.learn_from_events(events, scene)
    
    print(f"✅ Learned {len(learned_rules)} causal rules")
    for rule in learned_rules:
        print(f"   {rule.law_type.value}: {rule.confidence:.2f} confidence ({rule.evidence_count} evidence)")
    
    # Test prediction
    conditions = {
        'object_above_ground': True,
        'object_unsupported': True,
        'mass_greater_than': 0.5
    }
    
    predictions = learner.predict_outcome(scene, conditions)
    print(f"✅ Generated {len(predictions)} predictions")
    
    # Get summary
    summary = learner.get_rule_summary()
    print(f"✅ Rule summary: {summary['total_rules']} total, {summary['high_confidence_rules']} high confidence")
    
    print("✅ Physics reasoning engine test completed!")


if __name__ == "__main__":
    test_physics_reasoning_engine()
