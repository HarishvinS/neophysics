"""
Multi-Step Physics Reasoning
Build system that can predict and reason about multi-step physics interactions and chain reactions.
Enables understanding of complex cause-and-effect chains in physics simulations.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import json
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


class PhysicsPredictor:
    """
    Predicts individual physics interactions.

    NOTE: The current implementation uses highly simplified, non-physical models.
# In improved_physics_reasoning.py, inside ScenarioAnalyzer

def _check_domino_chain(self, scene: DynamicPhysicsScene, 
                       pattern_info: Dict[str, Any]) -> Optional[PhysicsScenario]:
    # ... (existing code to find potential_dominos) ...

    if len(potential_dominos) < 3:
        return None
    
    # This could be enhanced with a clustering algorithm (like DBSCAN)
    # to find multiple, separate chains of dominos, even if they aren't perfectly linear.
    # For now, the linear check is a good starting point.
    
    # ... (rest of the function) ...
    A robust implementation should use a headless physics simulation (like PyBullet)
    to predict outcomes based on initial conditions, rather than relying on hardcoded formulas.
    """

    def __init__(self):
        """Initialize physics predictor."""
        self.interaction_models = self._build_interaction_models()
        self.prediction_cache = {}
        # TODO: Integrate with PyBullet for accurate physics predictions
    
    def _build_interaction_models(self) -> Dict[InteractionType, Dict[str, Any]]:
        """
        Build models for different interaction types.

        FLAW: These models are oversimplified and use "magic numbers".
        They do not accurately represent physical laws.
        """
        models = {}
        
        # Collision model
        models[InteractionType.COLLISION] = {
            'velocity_transfer_efficiency': 0.8,
            'energy_loss_factor': 0.1,
            'minimum_impact_velocity': 0.1,
            'restitution_factor': 0.7
        }
        
        # Gravity model
        models[InteractionType.GRAVITY_PULL] = {
            'acceleration': 9.81,
            'terminal_velocity': 50.0,
            'air_resistance_factor': 0.01
        }
        
        # Friction model
        models[InteractionType.FRICTION_DRAG] = {
            'static_threshold': 0.1,
            'kinetic_coefficient': 0.7,
            'velocity_decay_rate': 0.95
        }
        
        # Spring force model
        models[InteractionType.SPRING_FORCE] = {
            'spring_constant': 100.0,
            'damping_factor': 0.1,
            'maximum_compression': 0.5
        }
        
        return models
    
    def predict_collision(self, obj_a: DynamicPhysicsObject, obj_b: DynamicPhysicsObject,
                         velocity_a: np.ndarray, velocity_b: np.ndarray) -> PhysicsStep:
        """
        Predict collision outcome.

        FLAW: This is a simplified 1D collision model that ignores 3D vectors,
        angular velocity, and material properties like restitution. It is not
        a physically accurate prediction.
        """
        model = self.interaction_models[InteractionType.COLLISION]
        
        # Calculate relative velocity
        relative_velocity = np.linalg.norm(velocity_a - velocity_b)
        
        if relative_velocity < model['minimum_impact_velocity']:
            # No significant collision
            return None
        
        # Predict post-collision velocities (simplified)
        mass_a = obj_a.mass
        mass_b = obj_b.mass
        total_mass = mass_a + mass_b
        
        # Conservation of momentum with energy loss
        efficiency = model['velocity_transfer_efficiency']
        
        final_velocity_a = (velocity_a * (mass_a - mass_b) + 2 * mass_b * velocity_b) / total_mass * efficiency
        final_velocity_b = (velocity_b * (mass_b - mass_a) + 2 * mass_a * velocity_a) / total_mass * efficiency
        
        # Create physics step
        step = PhysicsStep(
            step_id=f"collision_{obj_a.object_id}_{obj_b.object_id}_{int(time.time() * 1000)}",
            timestamp=time.time(),
            interaction_type=InteractionType.COLLISION,
            primary_object=obj_a.object_id,
            affected_objects=[obj_b.object_id],
            initial_state={
                obj_a.object_id: {'velocity': velocity_a.tolist(), 'position': obj_a.position.to_list()},
                obj_b.object_id: {'velocity': velocity_b.tolist(), 'position': obj_b.position.to_list()}
            },
            predicted_state={
                obj_a.object_id: {'velocity': final_velocity_a.tolist()},
                obj_b.object_id: {'velocity': final_velocity_b.tolist()}
            },
            confidence=0.8 if relative_velocity > 1.0 else 0.6,
            prerequisites=[]
        )
        
        return step
    
    def predict_gravity_fall(self, obj: DynamicPhysicsObject, height: float) -> PhysicsStep:
        """Predict gravity-induced fall."""
        model = self.interaction_models[InteractionType.GRAVITY_PULL]
        
        # Calculate fall time and final velocity
        fall_time = np.sqrt(2 * height / model['acceleration'])
        final_velocity = model['acceleration'] * fall_time
        
        # Apply air resistance
        final_velocity *= (1 - model['air_resistance_factor'])
        
        step = PhysicsStep(
            step_id=f"gravity_{obj.object_id}_{int(time.time() * 1000)}",
            timestamp=time.time(),
            interaction_type=InteractionType.GRAVITY_PULL,
            primary_object=obj.object_id,
            affected_objects=[],
            initial_state={
                obj.object_id: {'position': obj.position.to_list(), 'velocity': [0, 0, 0]}
            },
            predicted_state={
                obj.object_id: {
                    'position': [obj.position.x, obj.position.y, obj.position.z - height],
                    'velocity': [0, 0, -final_velocity],
                    'fall_time': fall_time
                }
            },
            confidence=0.9,
            prerequisites=[]
        )
        
        return step
    
    def predict_domino_effect(self, dominos: List[DynamicPhysicsObject], spacing: float) -> List[PhysicsStep]:
        """
        Predict domino chain reaction.

        FLAW: This method hardcodes the behavior of a domino chain. The tip-over time
        is a guess, not a calculation based on physics (mass, height, center of gravity, etc.).
        A true reasoning engine would predict this by simulating a sequence of collisions.
        """
        steps = []
        
        for i in range(len(dominos) - 1):
            current_domino = dominos[i]
            next_domino = dominos[i + 1]
            
            # Calculate tip-over time based on spacing
            # FLAW: This is not based on physics. It's a hardcoded guess.
            tip_time = spacing / 2.0  # Simplified model
            
            step = PhysicsStep(
                step_id=f"domino_{current_domino.object_id}_tips_{next_domino.object_id}",
                timestamp=time.time() + i * tip_time,
                interaction_type=InteractionType.CHAIN_REACTION,
                primary_object=current_domino.object_id,
                affected_objects=[next_domino.object_id],
                initial_state={
                    current_domino.object_id: {'rotation': [0, 0, 0]},
                    next_domino.object_id: {'rotation': [0, 0, 0]}
                },
                predicted_state={
                    current_domino.object_id: {'rotation': [0, 1.57, 0]},  # 90 degrees
                    next_domino.object_id: {'rotation': [0, 1.57, 0]}
                },
                confidence=0.85,
                prerequisites=[steps[-1].step_id] if steps else []
            )
            
            steps.append(step)
        
        return steps


class ChainReasoningEngine:
    """Reasons about multi-step physics chains."""
    
    def __init__(self, use_improved_reasoner: bool = True):
        """Initialize chain reasoning engine."""
        self.use_improved_reasoner = use_improved_reasoner
        if self.use_improved_reasoner:
            # Import here to avoid potential circular dependency issues
            from improved_physics_reasoning import ImprovedPhysicsReasoner
            self.reasoner = ImprovedPhysicsReasoner()
            print("INFO: Using ImprovedPhysicsReasoner for chain analysis.")
        else:
            self.predictor = PhysicsPredictor()
            print("INFO: Using legacy PhysicsPredictor for chain analysis.")
        self.reasoning_cache = {}
    
    def analyze_scene_for_chains(self, scene: DynamicPhysicsScene) -> List[PhysicsChain]:
        """
        Analyze scene to identify potential physics chains.
        This method now acts as a wrapper for the new, more powerful reasoning systems.
        """
        if self.use_improved_reasoner and hasattr(self, 'reasoner'):
            analysis = self.reasoner.analyze_and_predict(scene)
            # The new reasoner returns a comprehensive analysis. We extract the predicted chain.
            # The chain is returned as an object, so we can wrap it in a list.
            return [analysis['predicted_chain']]
        else:
            # Fallback to the legacy template-based approach
            return self.analyze_scene_for_chains_legacy(scene)

    def analyze_scene_for_chains_legacy(self, scene: DynamicPhysicsScene) -> List[PhysicsChain]:
        """
        Analyze scene to identify potential physics chains.
        NOTE: This is the original template-based approach, kept for reference.
        It is fundamentally limited and not true "reasoning".
        """
        """Analyze scene to identify potential physics chains."""
        chains = []
        chains.extend(self._detect_ramp_scenarios(scene))
        chains.extend(self._detect_domino_scenarios(scene))
        chains.extend(self._detect_collision_scenarios(scene))
        chains.extend(self._detect_pendulum_scenarios(scene))
        
        return chains
    
    def _detect_ramp_scenarios(self, scene: DynamicPhysicsScene) -> List[PhysicsChain]:
        """Detect ball-on-ramp scenarios."""
        chains = []
        
        # Find ramps and balls
        ramps = scene.get_objects_by_type(ObjectType.RAMP)
        spheres = scene.get_objects_by_type(ObjectType.SPHERE)
        
        for ramp in ramps:
            for sphere in spheres:
                # A more robust check to see if the sphere is positioned to roll down the ramp.
                # This is still a heuristic, but it's an improvement over the original.
                is_on_ramp_surface = abs(sphere.position.y - ramp.position.y) < ramp.scale.z / 2
                is_at_top_of_ramp = sphere.position.x < ramp.position.x and sphere.position.z > ramp.position.z

                if is_on_ramp_surface and is_at_top_of_ramp:
                    chain = self._create_ramp_chain(sphere, ramp)
                    chains.append(chain)
        
        return chains
    
    def _detect_domino_scenarios(self, scene: DynamicPhysicsScene) -> List[PhysicsChain]:
        """Detect domino chain scenarios."""
        chains = []
        
        # Find objects that could be dominos (tall, thin objects)
        dominos = []
        for obj in scene.objects.values():
            if obj.scale.z > obj.scale.x * 2:  # Tall and thin
                dominos.append(obj)
        
        if len(dominos) >= 3:
            # Sort by position to find chains
            dominos.sort(key=lambda obj: obj.position.x)
            
            # Check for linear arrangement
            for i in range(len(dominos) - 2):
                if self._are_dominos_aligned(dominos[i:i+3]):
                    chain = self._create_domino_chain(dominos[i:i+3])
                    chains.append(chain)
        
        return chains
    
    def _detect_collision_scenarios(self, scene: DynamicPhysicsScene) -> List[PhysicsChain]:
        """Detect potential collision scenarios."""
        chains = []
        
        # Find objects that might collide
        moving_objects = [obj for obj in scene.objects.values() if obj.mass > 0]
        
        for i, obj_a in enumerate(moving_objects):
            for obj_b in moving_objects[i+1:]:
                # Check if objects are on collision course
                distance = np.sqrt(
                    (obj_a.position.x - obj_b.position.x)**2 +
                    (obj_a.position.y - obj_b.position.y)**2 +
                    (obj_a.position.z - obj_b.position.z)**2
                )
                
                if distance < 3.0:  # Close enough for potential collision
                    chain = self._create_collision_chain(obj_a, obj_b)
                    chains.append(chain)
        
        return chains
    
    def _detect_pendulum_scenarios(self, scene: DynamicPhysicsScene) -> List[PhysicsChain]:
        """Detect pendulum scenarios."""
        chains = []
        
        # Look for pendulum-like arrangements
        for obj in scene.objects.values():
            # Check if object has constraints that could make it a pendulum
            if obj.constraints and 'fixed' not in obj.constraints:
                chain = self._create_pendulum_chain(obj)
                chains.append(chain)
        
        return chains
    
    def _create_ramp_chain(self, sphere: DynamicPhysicsObject, ramp: DynamicPhysicsObject) -> PhysicsChain:
        """Create a ball-on-ramp physics chain."""
        steps = []
        
        # Step 1: Ball starts rolling due to gravity
        height_diff = sphere.position.z - ramp.position.z
        gravity_step = self.predictor.predict_gravity_fall(sphere, height_diff * 0.5)  # Partial fall
        if gravity_step:
            steps.append(gravity_step)
        
        # Step 2: Ball rolls down ramp
        roll_step = PhysicsStep(
            step_id=f"roll_{sphere.object_id}_down_{ramp.object_id}",
            timestamp=time.time() + 0.5,
            interaction_type=InteractionType.FRICTION_DRAG,
            primary_object=sphere.object_id,
            affected_objects=[ramp.object_id],
            initial_state={sphere.object_id: {'velocity': [0, 0, -1]}},
            predicted_state={sphere.object_id: {'velocity': [2, 0, -0.5]}},
            confidence=0.8,
            prerequisites=[gravity_step.step_id] if gravity_step else []
        )
        steps.append(roll_step)
        
        chain = PhysicsChain(
            chain_id=f"ramp_chain_{sphere.object_id}_{ramp.object_id}",
            description=f"Ball {sphere.object_id} rolls down ramp {ramp.object_id}",
            steps=steps,
            total_duration=2.5,
            overall_confidence=0.85,
            trigger_conditions={'sphere_above_ramp': True, 'gravity_enabled': True}
        )
        
        return chain
    
    def _create_domino_chain(self, dominos: List[DynamicPhysicsObject]) -> PhysicsChain:
        """Create a domino chain reaction."""
        steps = self.predictor.predict_domino_effect(dominos, spacing=1.0)
        
        chain = PhysicsChain(
            chain_id=f"domino_chain_{dominos[0].object_id}_to_{dominos[-1].object_id}",
            description=f"Domino chain reaction from {dominos[0].object_id} to {dominos[-1].object_id}",
            steps=steps,
            total_duration=len(dominos) * 0.2,
            overall_confidence=0.8,
            trigger_conditions={'dominos_aligned': True, 'first_domino_pushed': True}
        )
        
        return chain
    
    def _create_collision_chain(self, obj_a: DynamicPhysicsObject, obj_b: DynamicPhysicsObject) -> PhysicsChain:
        """Create a collision chain."""
        # Assume objects are moving towards each other
        velocity_a = np.array([1.0, 0, 0])  # Simplified
        velocity_b = np.array([-1.0, 0, 0])
        
        collision_step = self.predictor.predict_collision(obj_a, obj_b, velocity_a, velocity_b)
        
        if not collision_step:
            return None
        
        chain = PhysicsChain(
            chain_id=f"collision_chain_{obj_a.object_id}_{obj_b.object_id}",
            description=f"Collision between {obj_a.object_id} and {obj_b.object_id}",
            steps=[collision_step],
            total_duration=0.5,
            overall_confidence=0.7,
            trigger_conditions={'objects_moving': True, 'collision_course': True}
        )
        
        return chain
    
    def _create_pendulum_chain(self, obj: DynamicPhysicsObject) -> PhysicsChain:
        """Create a pendulum swing chain."""
        # Simplified pendulum model
        swing_step = PhysicsStep(
            step_id=f"pendulum_swing_{obj.object_id}",
            timestamp=time.time(),
            interaction_type=InteractionType.SPRING_FORCE,
            primary_object=obj.object_id,
            affected_objects=[],
            initial_state={obj.object_id: {'angle': 0.5}},  # Initial displacement
            predicted_state={obj.object_id: {'angle': -0.5}},  # Swing to other side
            confidence=0.8,
            prerequisites=[]
        )
        
        chain = PhysicsChain(
            chain_id=f"pendulum_chain_{obj.object_id}",
            description=f"Pendulum {obj.object_id} swings back and forth",
            steps=[swing_step],
            total_duration=2.0,
            overall_confidence=0.8,
            trigger_conditions={'pendulum_displaced': True}
        )
        
        return chain
    
    def _are_dominos_aligned(self, dominos: List[DynamicPhysicsObject]) -> bool:
        """Check if dominos are aligned for chain reaction."""
        if len(dominos) < 2:
            return False
        
        # Check if dominos are roughly in a line
        positions = [(obj.position.x, obj.position.y) for obj in dominos]
        
        # Simple linearity check
        for i in range(len(positions) - 1):
            distance = np.sqrt((positions[i+1][0] - positions[i][0])**2 + (positions[i+1][1] - positions[i][1])**2)
            if distance > 2.0:  # Too far apart
                return False
        
        return True
    
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
    from scene_representation import ObjectType, MaterialType, Vector3
    
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
    
    # Test physics predictor
    predictor = PhysicsPredictor()
    
    # Test gravity prediction
    gravity_step = predictor.predict_gravity_fall(ball, 2.0)
    print(f"✅ Gravity prediction: {gravity_step.interaction_type.value}, confidence: {gravity_step.confidence}")
    
    # Test collision prediction
    velocity_a = np.array([1.0, 0, 0])
    velocity_b = np.array([-0.5, 0, 0])
    collision_step = predictor.predict_collision(ball, ramp, velocity_a, velocity_b)
    if collision_step:
        print(f"✅ Collision prediction: confidence: {collision_step.confidence}")
    
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
    
    # Test domino prediction
    dominos = [scene.objects[f"domino_{i}"] for i in range(3)]
    domino_steps = predictor.predict_domino_effect(dominos, 1.5)
    print(f"✅ Domino chain prediction: {len(domino_steps)} steps")
    
    print("✅ Multi-step physics reasoning test completed!")


if __name__ == "__main__":
    test_multi_step_physics_reasoning()
