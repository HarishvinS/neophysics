"""
Multi-Step Physics Reasoning
This module acts as a high-level wrapper for the more advanced reasoning engines,
ensuring that complex, multi-step physical interactions are analyzed and predicted
using robust, simulation-based methods.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict

from dynamic_scene_representation import DynamicPhysicsScene, DynamicPhysicsObject, SpatialRelation
from physics_reasoning_engine import PhysicsEvent, CausalRule, PhysicsLaw
from advanced_physics_materials import AdvancedMaterialType, MaterialProperties
from scene_representation import ObjectType, MaterialType, Vector3


class InteractionType(Enum):
    """Types of physics interactions."""
    COLLISION = "collision"
    GRAVITY_PULL = "gravity_pull"
    FRICTION_DRAG = "friction_drag"
    SPRING_FORCE = "spring_force"
    MAGNETIC_ATTRACTION = "magnetic_attraction"
    THERMAL_TRANSFER = "thermal_transfer"
    PRESSURE_WAVE = "pressure_wave"
    CHAIN_REACTION = "chain_reaction"


@dataclass
class PhysicsStep:
    """Represents a single step in a multi-step physics process."""
    step_id: str
    timestamp: float
    interaction_type: InteractionType
    primary_object: str
    affected_objects: List[str]
    initial_state: Dict[str, Any]
    predicted_state: Dict[str, Any]
    confidence: float
    prerequisites: List[str]  # Previous steps that must occur first
    
    def to_dict(self):
        return {
            'step_id': self.step_id,
            'timestamp': self.timestamp,
            'interaction_type': self.interaction_type.value,
            'primary_object': self.primary_object,
            'affected_objects': self.affected_objects,
            'initial_state': self.initial_state,
            'predicted_state': self.predicted_state,
            'confidence': self.confidence,
            'prerequisites': self.prerequisites
        }


@dataclass
class PhysicsChain:
    """Represents a complete chain of physics interactions."""
    chain_id: str
    description: str
    steps: List[PhysicsStep]
    total_duration: float
    overall_confidence: float
    trigger_conditions: Dict[str, Any]
    
    def to_dict(self):
        return {
            'chain_id': self.chain_id,
            'description': self.description,
            'steps': [step.to_dict() for step in self.steps],
            'total_duration': self.total_duration,
            'overall_confidence': self.overall_confidence,
            'trigger_conditions': self.trigger_conditions
        }


class ChainReasoningEngine:
    """Reasons about multi-step physics chains."""
    
    def __init__(self):
        """Initialize chain reasoning engine."""
        # This engine now exclusively uses the robust, simulation-based reasoner.
        # The import is done here to avoid potential circular dependency issues.
        from improved_physics_reasoning import ImprovedPhysicsReasoner
        self.reasoner = ImprovedPhysicsReasoner()
        self.reasoning_cache = {}
    
    def analyze_scene_for_chains(self, scene: DynamicPhysicsScene) -> List[PhysicsChain]:
        """
        Analyze scene to identify potential physics chains.
        This method now exclusively uses the robust, simulation-based reasoning engine.
        """
        analysis = self.reasoner.analyze_and_predict(scene)
        # The new reasoner returns a comprehensive analysis. We extract the predicted chain.
        # The chain is returned as an object, so we can wrap it in a list.
        return [analysis['predicted_chain']]
    
    def predict_chain_outcome(self, chain: PhysicsChain) -> Dict[str, Any]:
        """Predict the final outcome of a physics chain."""
        final_states = {}
        
        # Process each step in sequence
        for step in chain.steps:
            # Update states based on predictions
            for obj_id, predicted_state in step.predicted_state.items():
                if obj_id not in final_states:
                    final_states[obj_id] = {}
                final_states[obj_id].update(predicted_state)
        
        outcome = {
            'chain_id': chain.chain_id,
            'final_states': final_states,
            'total_duration': chain.total_duration,
            'success_probability': chain.overall_confidence,
            'key_events': [step.interaction_type.value for step in chain.steps]
        }
        
        return outcome


def test_multi_step_physics_reasoning():
    """Test the multi-step physics reasoning system."""
    print("Testing Multi-Step Physics Reasoning...")
    
    # Create test scene
    scene = DynamicPhysicsScene("test_multi_step")
    
    # Add objects for testing
    from scene_representation import ObjectType, MaterialType, Vector3, SceneBuilder
    
    # Ball above ramp
    ball = DynamicPhysicsObject(
        object_id="ball_1",
        object_type=ObjectType.SPHERE,
        position=Vector3(0, 0, 3),
        rotation=Vector3(0, 0, 0),
        scale=Vector3(0.5, 0.5, 0.5),
        mass=1.0,
        material=MaterialType.RUBBER
    )
    scene.add_object(ball)
    
    # Ramp
    ramp = DynamicPhysicsObject(
        object_id="ramp_1",
        object_type=ObjectType.RAMP,
        position=Vector3(0, 0, 1),
        rotation=Vector3(0, 0.3, 0),
        scale=Vector3(2, 0.2, 1),
        mass=0,  # Static
        material=MaterialType.WOOD
    )
    scene.add_object(ramp)
    
    # Domino-like objects
    for i in range(3):
        domino = DynamicPhysicsObject(
            object_id=f"domino_{i}",
            object_type=ObjectType.BOX,
            position=Vector3(3 + i * 1.5, 0, 1),
            rotation=Vector3(0, 0, 0),
            scale=Vector3(0.1, 0.5, 1.0),
            mass=0.5,
            material=MaterialType.WOOD
        )
        scene.add_object(domino)
    
    print(f"✅ Created test scene with {scene.get_object_count()} objects")
    
    # Test chain reasoning
    reasoning_engine = ChainReasoningEngine()
    
    # Analyze scene for chains
    chains = reasoning_engine.analyze_scene_for_chains(scene)
    print(f"✅ Detected {len(chains)} physics chains:")
    
    for chain in chains:
        print(f"   {chain.description} ({len(chain.steps)} steps, confidence: {chain.overall_confidence:.2f})")
        
        # Predict chain outcome
        outcome = reasoning_engine.predict_chain_outcome(chain)
        print(f"     Predicted duration: {outcome['total_duration']:.1f}s")
        print(f"     Success probability: {outcome['success_probability']:.2f}")
        print(f"     Key events: {outcome['key_events']}")
    
    print("✅ Multi-step physics reasoning test completed!")


if __name__ == "__main__":
    test_multi_step_physics_reasoning()
