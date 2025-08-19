"""
Improved Physics Reasoning
A robust physics reasoning system that combines simulation-based prediction
with intelligent scenario analysis. This addresses the limitations of template-based reasoning.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from dynamic_scene_representation import DynamicPhysicsScene, DynamicPhysicsObject
from scene_representation import ObjectType, MaterialType, Vector3
from multi_step_physics_reasoning import PhysicsStep, PhysicsChain, InteractionType
from physics_simulation_engine import PhysicsSimulationEngine


class ReasoningStrategy(Enum):
    """Different reasoning strategies for physics prediction."""
    SIMULATION_BASED = "simulation_based"
    ANALYTICAL = "analytical"
    HYBRID = "hybrid"
    PATTERN_RECOGNITION = "pattern_recognition"


@dataclass
class PhysicsScenario:
    """Represents a recognized physics scenario with prediction strategy."""
    scenario_id: str
    scenario_type: str
    description: str
    objects_involved: List[str]
    confidence: float
    recommended_strategy: ReasoningStrategy
    key_parameters: Dict[str, Any]
    
    def to_dict(self):
        return {
            'scenario_id': self.scenario_id,
            'scenario_type': self.scenario_type,
            'description': self.description,
            'objects_involved': self.objects_involved,
            'confidence': self.confidence,
            'recommended_strategy': self.recommended_strategy.value,
            'key_parameters': self.key_parameters
        }


class ScenarioAnalyzer:
    """Analyzes scenes to identify physics scenarios and recommend reasoning strategies."""
    
    def __init__(self):
        """Initialize scenario analyzer."""
        self.scenario_patterns = self._build_scenario_patterns()
    
    def _build_scenario_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Build patterns for recognizing physics scenarios."""
        patterns = {}
        
        # Ball on inclined surface
        patterns['ball_on_ramp'] = {
            'description': 'Spherical object on inclined surface',
            'detection_criteria': {
                'has_sphere': True,
                'has_inclined_surface': True,
                'sphere_above_surface': True
            },
            'strategy': ReasoningStrategy.SIMULATION_BASED,
            'confidence_base': 0.9
        }
        
        # Domino chain
        patterns['domino_chain'] = {
            'description': 'Linear arrangement of tall, thin objects',
            'detection_criteria': {
                'has_tall_thin_objects': True,
                'objects_in_line': True,
                'appropriate_spacing': True
            },
            'strategy': ReasoningStrategy.HYBRID,
            'confidence_base': 0.8
        }
        
        # Pendulum system
        patterns['pendulum'] = {
            'description': 'Suspended object that can swing',
            'detection_criteria': {
                'has_suspended_object': True,
                'has_pivot_point': True,
                'can_swing_freely': True
            },
            'strategy': ReasoningStrategy.ANALYTICAL,
            'confidence_base': 0.85
        }
        
        # Collision course
        patterns['collision_course'] = {
            'description': 'Objects moving toward each other',
            'detection_criteria': {
                'has_moving_objects': True,
                'trajectories_intersect': True,
                'sufficient_velocity': True
            },
            'strategy': ReasoningStrategy.SIMULATION_BASED,
            'confidence_base': 0.9
        }
        
        # Unstable stack
        patterns['unstable_stack'] = {
            'description': 'Objects stacked in potentially unstable configuration',
            'detection_criteria': {
                'has_stacked_objects': True,
                'center_of_mass_offset': True,
                'insufficient_support': True
            },
            'strategy': ReasoningStrategy.SIMULATION_BASED,
            'confidence_base': 0.85
        }
        
        return patterns
    
    def analyze_scene(self, scene: DynamicPhysicsScene) -> List[PhysicsScenario]:
        """Analyze scene to identify physics scenarios."""
        scenarios = []
        
        for pattern_name, pattern_info in self.scenario_patterns.items():
            scenario = self._check_pattern(scene, pattern_name, pattern_info)
            if scenario:
                scenarios.append(scenario)
        
        return scenarios
    
    def _check_pattern(self, scene: DynamicPhysicsScene, pattern_name: str, 
                      pattern_info: Dict[str, Any]) -> Optional[PhysicsScenario]:
        """Check if a scene matches a specific pattern."""
        criteria = pattern_info['detection_criteria']
        
        if pattern_name == 'ball_on_ramp':
            return self._check_ball_on_ramp(scene, pattern_info)
        elif pattern_name == 'domino_chain':
            return self._check_domino_chain(scene, pattern_info)
        elif pattern_name == 'pendulum':
            return self._check_pendulum(scene, pattern_info)
        elif pattern_name == 'collision_course':
            return self._check_collision_course(scene, pattern_info)
        elif pattern_name == 'unstable_stack':
            return self._check_unstable_stack(scene, pattern_info)
        
        return None
    
    def _check_ball_on_ramp(self, scene: DynamicPhysicsScene, 
                           pattern_info: Dict[str, Any]) -> Optional[PhysicsScenario]:
        """Check for ball on ramp scenario."""
        spheres = [obj for obj in scene.objects.values() if obj.object_type == ObjectType.SPHERE]
        ramps = [obj for obj in scene.objects.values() if obj.object_type == ObjectType.RAMP]
        
        for sphere in spheres:
            for ramp in ramps:
                # Check if sphere is positioned to roll down ramp
                height_diff = sphere.position.z - ramp.position.z
                horizontal_dist = abs(sphere.position.x - ramp.position.x)
                
                if height_diff > 0.1 and horizontal_dist < ramp.scale.x:
                    return PhysicsScenario(
                        scenario_id=f"ball_ramp_{sphere.object_id}_{ramp.object_id}",
                        scenario_type="ball_on_ramp",
                        description=f"Ball {sphere.object_id} positioned to roll down ramp {ramp.object_id}",
                        objects_involved=[sphere.object_id, ramp.object_id],
                        confidence=pattern_info['confidence_base'],
                        recommended_strategy=pattern_info['strategy'],
                        key_parameters={
                            'height_difference': height_diff,
                            'ramp_angle': ramp.rotation.y,
                            'ball_mass': sphere.mass,
                            'surface_friction': 0.7  # Estimated
                        }
                    )
        
        return None
    
    def _check_domino_chain(self, scene: DynamicPhysicsScene, 
                           pattern_info: Dict[str, Any]) -> Optional[PhysicsScenario]:
        """Check for domino chain scenario."""
        # Find tall, thin objects
        potential_dominos = []
        for obj in scene.objects.values():
            aspect_ratio = obj.scale.z / max(obj.scale.x, obj.scale.y)
            if aspect_ratio > 2.0:  # Tall and thin
                potential_dominos.append(obj)
        
        if len(potential_dominos) < 3:
            return None
        
        # Check if they're arranged in a line
        positions = [(obj.position.x, obj.position.y) for obj in potential_dominos]
        
        # Simple linearity check
        if self._are_points_roughly_linear(positions):
            return PhysicsScenario(
                scenario_id=f"domino_chain_{len(potential_dominos)}_objects",
                scenario_type="domino_chain",
                description=f"Chain of {len(potential_dominos)} domino-like objects",
                objects_involved=[obj.object_id for obj in potential_dominos],
                confidence=pattern_info['confidence_base'],
                recommended_strategy=pattern_info['strategy'],
                key_parameters={
                    'num_dominos': len(potential_dominos),
                    'average_spacing': self._calculate_average_spacing(positions),
                    'domino_height': np.mean([obj.scale.z for obj in potential_dominos])
                }
            )
        
        return None
    
    def _check_pendulum(self, scene: DynamicPhysicsScene, 
                       pattern_info: Dict[str, Any]) -> Optional[PhysicsScenario]:
        """Check for pendulum scenario."""
        # Look for objects with constraints that could be pendulums
        for obj in scene.objects.values():
            if hasattr(obj, 'constraints') and obj.constraints:
                # Check if it's a point constraint (pendulum-like)
                if any('point' in str(constraint).lower() for constraint in obj.constraints):
                    return PhysicsScenario(
                        scenario_id=f"pendulum_{obj.object_id}",
                        scenario_type="pendulum",
                        description=f"Pendulum system with object {obj.object_id}",
                        objects_involved=[obj.object_id],
                        confidence=pattern_info['confidence_base'],
                        recommended_strategy=pattern_info['strategy'],
                        key_parameters={
                            'pendulum_length': obj.position.z,  # Simplified
                            'bob_mass': obj.mass,
                            'initial_angle': 0.1  # Small initial displacement
                        }
                    )
        
        return None
    
    def _check_collision_course(self, scene: DynamicPhysicsScene, 
                               pattern_info: Dict[str, Any]) -> Optional[PhysicsScenario]:
        """Check for collision course scenario."""
        moving_objects = [obj for obj in scene.objects.values() if obj.mass > 0]
        
        for i, obj_a in enumerate(moving_objects):
            for obj_b in moving_objects[i+1:]:
                distance = np.sqrt(
                    (obj_a.position.x - obj_b.position.x)**2 +
                    (obj_a.position.y - obj_b.position.y)**2 +
                    (obj_a.position.z - obj_b.position.z)**2
                )
                
                # If objects are close and could potentially collide
                if distance < 3.0:
                    return PhysicsScenario(
                        scenario_id=f"collision_{obj_a.object_id}_{obj_b.object_id}",
                        scenario_type="collision_course",
                        description=f"Potential collision between {obj_a.object_id} and {obj_b.object_id}",
                        objects_involved=[obj_a.object_id, obj_b.object_id],
                        confidence=pattern_info['confidence_base'] * (3.0 - distance) / 3.0,
                        recommended_strategy=pattern_info['strategy'],
                        key_parameters={
                            'distance': distance,
                            'mass_ratio': obj_a.mass / obj_b.mass,
                            'relative_size': (obj_a.scale.x + obj_a.scale.y + obj_a.scale.z) / 
                                           (obj_b.scale.x + obj_b.scale.y + obj_b.scale.z)
                        }
                    )
        
        return None
    
    def _check_unstable_stack(self, scene: DynamicPhysicsScene, 
                             pattern_info: Dict[str, Any]) -> Optional[PhysicsScenario]:
        """Check for unstable stack scenario."""
        # Find objects that might be stacked
        objects_by_height = sorted(scene.objects.values(), key=lambda obj: obj.position.z)
        
        for i, obj in enumerate(objects_by_height[1:], 1):  # Skip ground-level objects
            # Check if there's an object below this one
            for lower_obj in objects_by_height[:i]:
                height_diff = obj.position.z - lower_obj.position.z
                horizontal_dist = np.sqrt(
                    (obj.position.x - lower_obj.position.x)**2 +
                    (obj.position.y - lower_obj.position.y)**2
                )
                
                # If object is above another and close horizontally
                if 0.1 < height_diff < 2.0 and horizontal_dist < max(obj.scale.x, lower_obj.scale.x):
                    # Check for potential instability (simplified)
                    if horizontal_dist > 0.1:  # Some offset
                        return PhysicsScenario(
                            scenario_id=f"unstable_stack_{obj.object_id}_{lower_obj.object_id}",
                            scenario_type="unstable_stack",
                            description=f"Potentially unstable stack: {obj.object_id} on {lower_obj.object_id}",
                            objects_involved=[obj.object_id, lower_obj.object_id],
                            confidence=pattern_info['confidence_base'] * (horizontal_dist / max(obj.scale.x, lower_obj.scale.x)),
                            recommended_strategy=pattern_info['strategy'],
                            key_parameters={
                                'height_difference': height_diff,
                                'horizontal_offset': horizontal_dist,
                                'top_mass': obj.mass,
                                'bottom_mass': lower_obj.mass
                            }
                        )
        
        return None
    
    def _are_points_roughly_linear(self, points: List[Tuple[float, float]], tolerance: float = 0.5) -> bool:
        """Check if points are roughly in a line."""
        if len(points) < 3:
            return True
        
        # Use first and last points to define line
        p1 = np.array(points[0])
        p2 = np.array(points[-1])
        
        # Check if intermediate points are close to the line
        for point in points[1:-1]:
            p = np.array(point)
            
            # Distance from point to line
            line_vec = p2 - p1
            point_vec = p - p1
            
            if np.linalg.norm(line_vec) == 0:
                continue
            
            # Project point onto line and find distance
            projection = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec) * line_vec
            distance = np.linalg.norm(point_vec - projection)
            
            if distance > tolerance:
                return False
        
        return True
    
    def _calculate_average_spacing(self, points: List[Tuple[float, float]]) -> float:
        """Calculate average spacing between consecutive points."""
        if len(points) < 2:
            return 0.0
        
        distances = []
        for i in range(len(points) - 1):
            p1 = np.array(points[i])
            p2 = np.array(points[i + 1])
            distances.append(np.linalg.norm(p2 - p1))
        
        return np.mean(distances)


class ImprovedPhysicsReasoner:
    """
    Improved physics reasoning system that combines multiple strategies
    for robust and accurate physics prediction.
    """
    
    def __init__(self):
        """Initialize improved physics reasoner."""
        self.scenario_analyzer = ScenarioAnalyzer()
        self.simulation_engine = PhysicsSimulationEngine()
    
    def analyze_and_predict(self, scene: DynamicPhysicsScene, 
                          initial_conditions: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze scene and predict physics outcomes using the most appropriate strategy.
        """
        # Step 1: Analyze scene for recognizable scenarios
        scenarios = self.scenario_analyzer.analyze_scene(scene)
        
        # Step 2: Choose prediction strategy based on scenarios
        if not scenarios:
            # No specific scenarios detected, use simulation
            strategy = ReasoningStrategy.SIMULATION_BASED
            confidence_modifier = 0.8
        else:
            # Use the strategy recommended by the highest-confidence scenario
            best_scenario = max(scenarios, key=lambda s: s.confidence)
            strategy = best_scenario.recommended_strategy
            confidence_modifier = best_scenario.confidence
        
        # Step 3: Predict using chosen strategy
        if strategy == ReasoningStrategy.SIMULATION_BASED:
            chain = self.simulation_engine.predict_physics_chain(scene, initial_conditions)
            chain.overall_confidence *= confidence_modifier
        
        elif strategy == ReasoningStrategy.ANALYTICAL:
            # For analytical scenarios like pendulums, use mathematical models
            chain = self._predict_analytically(scenarios[0], scene)
        
        elif strategy == ReasoningStrategy.HYBRID:
            # Combine simulation with pattern-based insights
            chain = self._predict_hybrid(scenarios[0], scene, initial_conditions)
        
        else:
            # Fallback to simulation
            chain = self.simulation_engine.predict_physics_chain(scene, initial_conditions)
        
        # Step 4: Compile comprehensive analysis
        analysis = {
            'detected_scenarios': [scenario.to_dict() for scenario in scenarios],
            'chosen_strategy': strategy.value,
            'predicted_chain': chain,  # Return the object directly, not its dict representation
            'confidence': chain.overall_confidence,
            'reasoning_summary': self._generate_reasoning_summary(scenarios, strategy, chain)
        }
        
        return analysis
    
    def _predict_analytically(self, scenario: PhysicsScenario, 
                            scene: DynamicPhysicsScene) -> PhysicsChain:
        """Predict using analytical physics models."""
        # Simplified analytical prediction for pendulum
        if scenario.scenario_type == "pendulum":
            # Use pendulum period formula: T = 2π√(L/g)
            length = scenario.key_parameters.get('pendulum_length', 1.0)
            period = 2 * np.pi * np.sqrt(length / 9.81)
            
            swing_step = PhysicsStep(
                step_id=f"analytical_pendulum_{scenario.scenario_id}",
                timestamp=time.time(),
                interaction_type=InteractionType.SPRING_FORCE,
                primary_object=scenario.objects_involved[0],
                affected_objects=[],
                initial_state={scenario.objects_involved[0]: {'angle': 0.1}},
                predicted_state={scenario.objects_involved[0]: {'period': period}},
                confidence=0.95,  # High confidence for analytical solution
                prerequisites=[]
            )
            
            return PhysicsChain(
                chain_id=f"analytical_{scenario.scenario_id}",
                description=f"Analytical prediction for {scenario.description}",
                steps=[swing_step],
                total_duration=period,
                overall_confidence=0.95,
                trigger_conditions={'analytical_model': True}
            )
        
        # Fallback to simulation for other scenarios
        return self.simulation_engine.predict_physics_chain(scene)
    
    def _predict_hybrid(self, scenario: PhysicsScenario, scene: DynamicPhysicsScene,
                       initial_conditions: Dict[str, Any] = None) -> PhysicsChain:
        """Predict using hybrid approach combining simulation and pattern knowledge."""
        # For domino chains, use simulation but with pattern-informed initial conditions
        if scenario.scenario_type == "domino_chain":
            # Set up initial push for first domino
            first_domino = scenario.objects_involved[0]
            
            if not initial_conditions:
                initial_conditions = {}
            
            # Add small angular velocity to first domino to start chain
            initial_conditions[first_domino] = {
                'angular_velocity': [0, 0.5, 0]  # Small rotation to tip it
            }
            
            chain = self.simulation_engine.predict_physics_chain(scene, initial_conditions)
            chain.description = f"Hybrid prediction for {scenario.description}"
            
            return chain
        
        # Fallback to simulation
        return self.simulation_engine.predict_physics_chain(scene, initial_conditions)
    
    def _generate_reasoning_summary(self, scenarios: List[PhysicsScenario], 
                                  strategy: ReasoningStrategy, 
                                  chain: PhysicsChain) -> str:
        """Generate human-readable reasoning summary."""
        if not scenarios:
            return f"No specific scenarios detected. Used {strategy.value} approach with {len(chain.steps)} predicted events."
        
        scenario_descriptions = [s.description for s in scenarios]
        
        summary = f"Detected scenarios: {', '.join(scenario_descriptions)}. "
        summary += f"Used {strategy.value} strategy. "
        summary += f"Predicted {len(chain.steps)} physics events with {chain.overall_confidence:.2f} confidence."
        
        return summary


def test_improved_physics_reasoning():
    """Test the improved physics reasoning system."""
    print("Testing Improved Physics Reasoning...")
    
    # Create test scene with ball on ramp
    scene = DynamicPhysicsScene("improved_reasoning_test")
    
    ball = DynamicPhysicsObject(
        object_id="test_ball",
        object_type=ObjectType.SPHERE,
        position=Vector3(0, 0, 3),
        rotation=Vector3(0, 0, 0),
        scale=Vector3(0.3, 0.3, 0.3),
        mass=1.0,
        material=MaterialType.RUBBER
    )
    scene.add_object(ball)
    
    ramp = DynamicPhysicsObject(
        object_id="test_ramp",
        object_type=ObjectType.RAMP,
        position=Vector3(0, 0, 1),
        rotation=Vector3(0, 0.3, 0),
        scale=Vector3(3, 0.2, 1),
        mass=0,
        material=MaterialType.WOOD
    )
    scene.add_object(ramp)
    
    print(f"✅ Created test scene with {scene.get_object_count()} objects")
    
    # Test improved reasoning
    reasoner = ImprovedPhysicsReasoner()
    
    analysis = reasoner.analyze_and_predict(scene)
    
    print(f"✅ Physics Analysis Results:")
    print(f"   Detected scenarios: {len(analysis['detected_scenarios'])}")
    
    for scenario in analysis['detected_scenarios']:
        print(f"     - {scenario['description']} (confidence: {scenario['confidence']:.2f})")
        print(f"       Strategy: {scenario['recommended_strategy']}")
    
    print(f"   Chosen strategy: {analysis['chosen_strategy']}")
    print(f"   Predicted events: {len(analysis['predicted_chain']['steps'])}")
    print(f"   Overall confidence: {analysis['confidence']:.2f}")
    print(f"   Reasoning: {analysis['reasoning_summary']}")
    
    print("✅ Improved physics reasoning test completed!")


if __name__ == "__main__":
    test_improved_physics_reasoning()
