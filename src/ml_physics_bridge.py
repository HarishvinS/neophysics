"""
ML-Physics Bridge
Converts ML model predictions into PyBullet physics simulations.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

from model_architecture import TextToSceneModel
from scene_representation import PhysicsScene, PhysicsObject, ObjectType, MaterialType, Vector3
from physics_engine import PhysicsEngine
from scene_encoder import SceneDecoder


class MLPhysicsBridge:
    """Bridge between ML predictions and PyBullet physics simulation."""
    
    def __init__(self, model: TextToSceneModel, use_gui: bool = True):
        """
        Initialize the ML-Physics bridge.
        
        Args:
            model: Trained text-to-scene model
            use_gui: Whether to show PyBullet GUI
        """
        self.model = model
        self.use_gui = use_gui
        self.physics_engine = None
        self.decoder = SceneDecoder(max_objects=model.max_objects)
        
        # Object ID mapping for tracking
        self.ml_to_physics_mapping = {}
        self.physics_to_ml_mapping = {}
        
        # Simulation state
        self.current_scene = None
        self.simulation_running = False
    
    def initialize_physics(self):
        """Initialize the physics engine."""
        if self.physics_engine is None:
            self.physics_engine = PhysicsEngine(use_gui=self.use_gui)
            print("✅ Physics engine initialized")
    
    def predict_and_simulate(self, text: str) -> Dict:
        """
        Complete pipeline: text → ML prediction → physics simulation.
        
        Args:
            text: Natural language description
            
        Returns:
            Dictionary with prediction results and simulation data
        """
        print(f"🔮 Processing: '{text}'")
        
        # Initialize physics if needed
        self.initialize_physics()
        
        # Clear previous scene
        self.clear_scene()
        
        # Get ML prediction
        start_time = time.time()
        predicted_scene = self.model.predict_scene(text)
        prediction_time = time.time() - start_time
        
        print(f"⚡ ML prediction completed in {prediction_time:.3f}s")
        
        # Convert to physics simulation
        start_time = time.time()
        physics_objects = self.scene_to_physics(predicted_scene)
        conversion_time = time.time() - start_time
        
        print(f"🔧 Scene conversion completed in {conversion_time:.3f}s")
        print(f"📦 Created {len(physics_objects)} physics objects")
        
        # Store current scene
        self.current_scene = predicted_scene
        
        return {
            'text_input': text,
            'predicted_scene': predicted_scene,
            'physics_objects': physics_objects,
            'prediction_time': prediction_time,
            'conversion_time': conversion_time,
            'total_objects': len(physics_objects)
        }
    
    def scene_to_physics(self, scene: PhysicsScene) -> List[int]:
        """
        Convert a PhysicsScene to PyBullet objects.
        
        Args:
            scene: PhysicsScene object from ML prediction
            
        Returns:
            List of PyBullet object IDs
        """
        physics_objects = []
        self.ml_to_physics_mapping.clear()
        self.physics_to_ml_mapping.clear()
        
        # Process each object in the scene
        for obj in scene.objects:
            if obj.object_type == ObjectType.PLANE:
                # Ground plane is already created by physics engine
                continue
            
            try:
                physics_id = self.create_physics_object(obj)
                if physics_id is not None:
                    physics_objects.append(physics_id)
                    
                    # Store mapping
                    self.ml_to_physics_mapping[obj.object_id] = physics_id
                    self.physics_to_ml_mapping[physics_id] = obj.object_id
                    
            except Exception as e:
                print(f"⚠️ Error creating object {obj.object_id}: {e}")
                continue
        
        return physics_objects
    
    def create_physics_object(self, obj: PhysicsObject) -> Optional[int]:
        """
        Create a single physics object from ML prediction.
        
        Args:
            obj: PhysicsObject from ML prediction
            
        Returns:
            PyBullet object ID or None if creation failed
        """
        # Validate and clamp object properties
        position = self.validate_position(obj.position)
        mass = max(0.1, min(obj.mass, 50.0))  # Clamp mass between 0.1 and 50 kg
        
        # Get material properties
        material_props = self.get_material_properties(obj.material)
        
        # Create object based on type
        if obj.object_type == ObjectType.SPHERE:
            radius = max(0.02, min(obj.scale.x, 1.0))  # Clamp radius
            physics_id = self.physics_engine.create_sphere(
                position=position.to_tuple(),
                radius=radius,
                mass=mass,
                color=obj.color
            )
            
        elif obj.object_type == ObjectType.BOX:
            # Use scale as half-extents, clamped to reasonable values
            half_extents = (
                max(0.02, min(obj.scale.x, 2.0)),
                max(0.02, min(obj.scale.y, 2.0)),
                max(0.02, min(obj.scale.z, 2.0))
            )
            physics_id = self.physics_engine.create_box(
                position=position.to_tuple(),
                half_extents=half_extents,
                mass=mass,
                color=obj.color
            )
            
        elif obj.object_type == ObjectType.RAMP:
            # Create ramp with rotation
            angle = obj.rotation.y  # Use Y rotation for ramp angle
            angle = max(-1.0, min(angle, 1.0))  # Clamp angle
            
            size = (
                max(0.5, min(obj.scale.x, 4.0)),  # Length
                max(0.1, min(obj.scale.y, 0.5)),  # Thickness
                max(0.5, min(obj.scale.z, 2.0))   # Width
            )
            
            physics_id = self.physics_engine.create_ramp(
                position=position.to_tuple(),
                angle=angle,
                size=size,
                mass=0,  # Ramps are typically static
                color=obj.color
            )
            
        else:
            # For other object types, default to sphere
            radius = max(0.02, min(obj.scale.x, 1.0))
            physics_id = self.physics_engine.create_sphere(
                position=position.to_tuple(),
                radius=radius,
                mass=mass,
                color=obj.color
            )
        
        # Apply material properties
        if physics_id is not None and material_props:
            self.physics_engine.set_object_properties(
                physics_id,
                lateral_friction=material_props['friction'],
                restitution=material_props['restitution']
            )
        
        return physics_id
    
    def validate_position(self, position: Vector3) -> Vector3:
        """Validate and clamp object position to reasonable bounds."""
        return Vector3(
            max(-5.0, min(position.x, 5.0)),   # X bounds
            max(-5.0, min(position.y, 5.0)),   # Y bounds
            max(0.1, min(position.z, 10.0))    # Z bounds (above ground)
        )
    
    def get_material_properties(self, material: MaterialType) -> Dict[str, float]:
        """Get physics properties for a material type."""
        material_db = {
            MaterialType.RUBBER: {'friction': 0.8, 'restitution': 0.9},
            MaterialType.METAL: {'friction': 0.6, 'restitution': 0.3},
            MaterialType.WOOD: {'friction': 0.7, 'restitution': 0.5},
            MaterialType.ICE: {'friction': 0.05, 'restitution': 0.1},
            MaterialType.BOUNCY: {'friction': 0.3, 'restitution': 0.95},
            MaterialType.PLASTIC: {'friction': 0.4, 'restitution': 0.6},
            MaterialType.GLASS: {'friction': 0.2, 'restitution': 0.1},
            MaterialType.STONE: {'friction': 0.8, 'restitution': 0.2},
        }
        return material_db.get(material, {'friction': 0.5, 'restitution': 0.3})
    
    def run_simulation(self, duration: float = 3.0, real_time: bool = True) -> Dict:
        """
        Run physics simulation and collect results.
        
        Args:
            duration: Simulation duration in seconds
            real_time: Whether to run in real-time
            
        Returns:
            Simulation results
        """
        if self.physics_engine is None:
            raise RuntimeError("Physics engine not initialized")
        
        if not self.current_scene:
            raise RuntimeError("No scene loaded for simulation")
        
        print(f"🏃 Running simulation for {duration:.1f}s...")
        
        # Initialize simulation data collection
        simulation_data = {
            'duration': duration,
            'frames': [],
            'object_trajectories': {},
            'events': []
        }
        
        # Calculate simulation parameters
        timestep = 1.0 / 240.0  # 240 Hz
        total_steps = int(duration / timestep)
        
        # Run simulation
        for step in range(total_steps):
            # Step physics
            self.physics_engine.step_simulation()
            
            # Collect data every 10 steps (24 Hz data collection)
            if step % 10 == 0:
                frame_data = self._collect_frame_data(step * timestep)
                simulation_data['frames'].append(frame_data)
            
            # Real-time delay if requested
            if real_time:
                import time
                time.sleep(timestep)
        
        # Process trajectories
        self._process_trajectories(simulation_data)
        
        print(f"✅ Simulation completed: {len(simulation_data['frames'])} frames collected")
        
        return simulation_data
    
    def _collect_frame_data(self, timestamp: float) -> Dict:
        """Collect physics data for a single frame."""
        frame_data = {
            'timestamp': timestamp,
            'object_states': {}
        }
        
        # Collect state for each physics object
        for ml_id, physics_id in self.ml_to_physics_mapping.items():
            try:
                state = self.physics_engine.get_object_state(physics_id)
                if state:
                    frame_data['object_states'][ml_id] = {
                        'position': state['position'].tolist(),
                        'velocity': state['velocity'].tolist(),
                        'angular_velocity': state['angular_velocity'].tolist()
                    }
            except Exception as e:
                print(f"⚠️ Error collecting state for {ml_id}: {e}")
        
        return frame_data
    
    def _process_trajectories(self, simulation_data: Dict):
        """Process frame data into object trajectories."""
        trajectories = {}
        
        for frame in simulation_data['frames']:
            for obj_id, state in frame['object_states'].items():
                if obj_id not in trajectories:
                    trajectories[obj_id] = {
                        'positions': [],
                        'velocities': [],
                        'timestamps': []
                    }
                
                trajectories[obj_id]['positions'].append(state['position'])
                trajectories[obj_id]['velocities'].append(state['velocity'])
                trajectories[obj_id]['timestamps'].append(frame['timestamp'])
        
        simulation_data['object_trajectories'] = trajectories
    
    def clear_scene(self):
        """Clear the current physics scene."""
        if self.physics_engine:
            self.physics_engine.clear_objects(keep_ground=True)
        
        self.ml_to_physics_mapping.clear()
        self.physics_to_ml_mapping.clear()
        self.current_scene = None
        self.simulation_running = Falseinitialized")
        
        print(f"🏃 Running simulation for {duration}s...")
        
        # Record initial states
        initial_states = {}
        for physics_id in self.physics_to_ml_mapping.keys():
            initial_states[physics_id] = self.physics_engine.get_object_state(physics_id)
        
        # Run simulation
        start_time = time.time()
        self.simulation_running = True
        
        try:
            self.physics_engine.run_simulation(duration=duration, real_time=real_time)
            simulation_time = time.time() - start_time
            
            # Record final states
            final_states = {}
            for physics_id in self.physics_to_ml_mapping.keys():
                final_states[physics_id] = self.physics_engine.get_object_state(physics_id)
            
            print(f"✅ Simulation completed in {simulation_time:.3f}s")
            
            return {
                'duration': duration,
                'simulation_time': simulation_time,
                'initial_states': initial_states,
                'final_states': final_states,
                'object_count': len(initial_states)
            }
            
        except Exception as e:
            print(f"❌ Simulation error: {e}")
            return {'error': str(e)}
        
        finally:
            self.simulation_running = False
    
    def get_simulation_summary(self, sim_results: Dict) -> Dict:
        """Generate a summary of simulation results."""
        if 'error' in sim_results:
            return {'error': sim_results['error']}
        
        summary = {
            'total_objects': sim_results['object_count'],
            'simulation_duration': sim_results['duration'],
            'objects_moved': 0,
            'average_displacement': 0.0,
            'max_displacement': 0.0
        }
        
        displacements = []
        
        for physics_id in sim_results['initial_states'].keys():
            initial_pos = sim_results['initial_states'][physics_id]['position']
            final_pos = sim_results['final_states'][physics_id]['position']
            
            # Calculate displacement
            displacement = np.linalg.norm(final_pos - initial_pos)
            displacements.append(displacement)
            
            if displacement > 0.1:  # Threshold for "moved"
                summary['objects_moved'] += 1
        
        if displacements:
            summary['average_displacement'] = np.mean(displacements)
            summary['max_displacement'] = np.max(displacements)
        
        return summary
    
    def clear_scene(self):
        """Clear the current physics scene."""
        if self.physics_engine is not None:
            self.physics_engine.clear_objects(keep_ground=True)
        
        self.ml_to_physics_mapping.clear()
        self.physics_to_ml_mapping.clear()
        self.current_scene = None
        print("🧹 Scene cleared")
    
    def disconnect(self):
        """Disconnect from physics engine."""
        if self.physics_engine is not None:
            self.physics_engine.disconnect()
            self.physics_engine = None
        print("🔌 Physics engine disconnected")


def test_ml_physics_bridge():
    """Test the ML-Physics bridge."""
    print("Testing ML-Physics Bridge...")
    
    # Load trained model if available
    model_path = "models/trained_model/final_model.pth"
    
    if os.path.exists(model_path):
        print("Loading trained model...")
        from model_architecture import ModelConfig
        
        config = ModelConfig()
        model = TextToSceneModel(hidden_size=config.hidden_size, max_objects=config.max_objects)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✅ Trained model loaded")
    else:
        print("No trained model found, using untrained model for testing...")
        from model_architecture import ModelConfig
        
        config = ModelConfig()
        model = TextToSceneModel(hidden_size=config.hidden_size, max_objects=config.max_objects)
    
    # Create bridge
    bridge = MLPhysicsBridge(model, use_gui=True)
    
    # Test examples
    test_texts = [
        "create a ball",
        "add a sphere on a ramp",
        "place two boxes"
    ]
    
    try:
        for i, text in enumerate(test_texts):
            print(f"\n--- Test {i+1}: '{text}' ---")
            
            # Predict and create physics scene
            result = bridge.predict_and_simulate(text)
            
            # Run simulation
            sim_results = bridge.run_simulation(duration=2.0, real_time=True)
            
            # Get summary
            summary = bridge.get_simulation_summary(sim_results)
            
            print(f"Results:")
            print(f"  Objects created: {result['total_objects']}")
            print(f"  Objects moved: {summary.get('objects_moved', 0)}")
            print(f"  Max displacement: {summary.get('max_displacement', 0):.3f}m")
            
            time.sleep(1)  # Brief pause between tests
    
    finally:
        bridge.disconnect()
    
    print("\n✅ ML-Physics bridge test completed!")


if __name__ == "__main__":
    import os
    test_ml_physics_bridge()
