"""
Real-time Physics Simulator
Provides real-time simulation capabilities with data capture and analysis.
"""

import torch
import numpy as np
import time
import threading
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import json

from ml_physics_bridge import MLPhysicsBridge
from model_architecture import TextToSceneModel
from physics_engine import PhysicsEngine


@dataclass
class SimulationFrame:
    """Data for a single simulation frame."""
    timestamp: float
    step: int
    object_states: Dict[int, Dict]  # object_id -> state
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'step': self.step,
            'object_states': {str(k): v for k, v in self.object_states.items()}
        }


class RealTimeSimulator:
    """Real-time physics simulator with data capture."""
    
    def __init__(self, bridge: MLPhysicsBridge, fps: int = 60):
        """
        Initialize real-time simulator.
        
        Args:
            bridge: ML-Physics bridge
            fps: Target frames per second
        """
        self.bridge = bridge
        self.fps = fps
        self.dt = 1.0 / fps
        
        # Simulation state
        self.running = False
        self.paused = False
        self.current_step = 0
        self.start_time = 0
        
        # Data capture
        self.recording = False
        self.frames = []
        self.max_frames = 1000  # Limit memory usage
        
        # Threading
        self.sim_thread = None
        self.stop_event = threading.Event()
        
        # Callbacks
        self.frame_callbacks = []
        self.event_callbacks = []
    
    def add_frame_callback(self, callback: Callable[[SimulationFrame], None]):
        """Add callback to be called on each frame."""
        self.frame_callbacks.append(callback)
    
    def add_event_callback(self, callback: Callable[[str, Dict], None]):
        """Add callback for simulation events."""
        self.event_callbacks.append(callback)
    
    def start_simulation(self, duration: Optional[float] = None, record: bool = True):
        """
        Start real-time simulation.
        
        Args:
            duration: Maximum simulation duration (None for infinite)
            record: Whether to record simulation data
        """
        if self.running:
            print("⚠️ Simulation already running")
            return
        
        print(f"🚀 Starting real-time simulation (FPS: {self.fps})")
        
        self.running = True
        self.paused = False
        self.current_step = 0
        self.start_time = time.time()
        self.recording = record
        
        if record:
            self.frames.clear()
        
        # Start simulation thread
        self.stop_event.clear()
        self.sim_thread = threading.Thread(
            target=self._simulation_loop,
            args=(duration,),
            daemon=True
        )
        self.sim_thread.start()
        
        self._trigger_event("simulation_started", {
            'fps': self.fps,
            'duration': duration,
            'recording': record
        })
    
    def pause_simulation(self):
        """Pause the simulation."""
        if not self.running:
            return
        
        self.paused = not self.paused
        status = "paused" if self.paused else "resumed"
        print(f"⏸️ Simulation {status}")
        
        self._trigger_event("simulation_paused", {'paused': self.paused})
    
    def stop_simulation(self):
        """Stop the simulation."""
        if not self.running:
            return
        
        print("🛑 Stopping simulation...")
        
        self.running = False
        self.stop_event.set()
        
        if self.sim_thread and self.sim_thread.is_alive():
            self.sim_thread.join(timeout=2.0)
        
        elapsed_time = time.time() - self.start_time
        
        print(f"✅ Simulation stopped after {elapsed_time:.2f}s ({self.current_step} steps)")
        
        self._trigger_event("simulation_stopped", {
            'elapsed_time': elapsed_time,
            'total_steps': self.current_step,
            'frames_recorded': len(self.frames) if self.recording else 0
        })
    
    def _simulation_loop(self, duration: Optional[float]):
        """Main simulation loop (runs in separate thread)."""
        try:
            while self.running and not self.stop_event.is_set():
                frame_start = time.time()
                
                # Check duration limit
                if duration and (time.time() - self.start_time) >= duration:
                    break
                
                # Skip if paused
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # Step physics simulation
                if self.bridge.physics_engine:
                    self.bridge.physics_engine.step_simulation(steps=1)
                
                # Capture frame data
                if self.recording:
                    frame = self._capture_frame()
                    if frame:
                        self._process_frame(frame)
                
                self.current_step += 1
                
                # Frame rate control
                frame_time = time.time() - frame_start
                sleep_time = max(0, self.dt - frame_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except Exception as e:
            print(f"❌ Simulation error: {e}")
            self._trigger_event("simulation_error", {'error': str(e)})
        
        finally:
            self.running = False
    
    def _capture_frame(self) -> Optional[SimulationFrame]:
        """Capture current simulation frame."""
        if not self.bridge.physics_engine or not self.bridge.physics_to_ml_mapping:
            return None
        
        # Get states of all tracked objects
        object_states = {}
        
        for physics_id in self.bridge.physics_to_ml_mapping.keys():
            try:
                state = self.bridge.physics_engine.get_object_state(physics_id)
                object_states[physics_id] = {
                    'position': state['position'].tolist(),
                    'orientation': state['orientation'].tolist(),
                    'velocity': state['velocity'].tolist(),
                    'angular_velocity': state['angular_velocity'].tolist()
                }
            except Exception as e:
                print(f"⚠️ Error getting state for object {physics_id}: {e}")
                continue
        
        if not object_states:
            return None
        
        return SimulationFrame(
            timestamp=time.time() - self.start_time,
            step=self.current_step,
            object_states=object_states
        )
    
    def _process_frame(self, frame: SimulationFrame):
        """Process a captured frame."""
        # Store frame (with memory limit)
        if len(self.frames) >= self.max_frames:
            self.frames.pop(0)  # Remove oldest frame
        
        self.frames.append(frame)
        
        # Call frame callbacks
        for callback in self.frame_callbacks:
            try:
                callback(frame)
            except Exception as e:
                print(f"⚠️ Frame callback error: {e}")
    
    def _trigger_event(self, event_type: str, data: Dict):
        """Trigger event callbacks."""
        for callback in self.event_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                print(f"⚠️ Event callback error: {e}")
    
    def get_simulation_stats(self) -> Dict:
        """Get current simulation statistics."""
        if not self.running:
            return {'status': 'stopped'}
        
        elapsed_time = time.time() - self.start_time
        actual_fps = self.current_step / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'status': 'paused' if self.paused else 'running',
            'elapsed_time': elapsed_time,
            'current_step': self.current_step,
            'target_fps': self.fps,
            'actual_fps': actual_fps,
            'frames_recorded': len(self.frames) if self.recording else 0,
            'objects_tracked': len(self.bridge.physics_to_ml_mapping)
        }
    
    def export_recording(self, filepath: str):
        """Export recorded frames to JSON file."""
        if not self.frames:
            print("⚠️ No frames to export")
            return
        
        data = {
            'metadata': {
                'fps': self.fps,
                'total_frames': len(self.frames),
                'duration': self.frames[-1].timestamp if self.frames else 0,
                'exported_at': time.time()
            },
            'frames': [frame.to_dict() for frame in self.frames]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📁 Recording exported to {filepath}")
    
    def analyze_motion(self) -> Dict:
        """Analyze object motion from recorded frames."""
        if len(self.frames) < 2:
            return {'error': 'Insufficient frames for analysis'}
        
        analysis = {
            'total_frames': len(self.frames),
            'duration': self.frames[-1].timestamp,
            'objects': {}
        }
        
        # Analyze each object
        for physics_id in self.bridge.physics_to_ml_mapping.keys():
            object_analysis = self._analyze_object_motion(physics_id)
            if object_analysis:
                ml_id = self.bridge.physics_to_ml_mapping[physics_id]
                analysis['objects'][ml_id] = object_analysis
        
        return analysis
    
    def _analyze_object_motion(self, physics_id: int) -> Optional[Dict]:
        """Analyze motion for a specific object."""
        positions = []
        velocities = []
        
        for frame in self.frames:
            if physics_id in frame.object_states:
                state = frame.object_states[physics_id]
                positions.append(np.array(state['position']))
                velocities.append(np.array(state['velocity']))
        
        if len(positions) < 2:
            return None
        
        # Calculate motion metrics
        positions = np.array(positions)
        velocities = np.array(velocities)
        
        # Total displacement
        total_displacement = np.linalg.norm(positions[-1] - positions[0])
        
        # Path length
        path_length = 0
        for i in range(1, len(positions)):
            path_length += np.linalg.norm(positions[i] - positions[i-1])
        
        # Speed statistics
        speeds = np.linalg.norm(velocities, axis=1)
        
        return {
            'total_displacement': float(total_displacement),
            'path_length': float(path_length),
            'max_speed': float(np.max(speeds)),
            'avg_speed': float(np.mean(speeds)),
            'final_position': positions[-1].tolist(),
            'final_velocity': velocities[-1].tolist()
        }


def test_realtime_simulator():
    """Test the real-time simulator."""
    print("Testing Real-time Simulator...")
    
    # Load model
    model_path = "models/trained_model/final_model.pth"
    
    if os.path.exists(model_path):
        print("Loading trained model...")
        from model_architecture import ModelConfig
        
        config = ModelConfig()
        model = TextToSceneModel(hidden_size=config.hidden_size, max_objects=config.max_objects)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    else:
        print("Using untrained model for testing...")
        from model_architecture import ModelConfig
        
        config = ModelConfig()
        model = TextToSceneModel(hidden_size=config.hidden_size, max_objects=config.max_objects)
    
    # Create bridge and simulator
    bridge = MLPhysicsBridge(model, use_gui=True)
    simulator = RealTimeSimulator(bridge, fps=30)  # Lower FPS for testing
    
    # Add callbacks
    def frame_callback(frame):
        if frame.step % 30 == 0:  # Print every second
            print(f"Frame {frame.step}: {len(frame.object_states)} objects tracked")
    
    def event_callback(event_type, data):
        print(f"Event: {event_type} - {data}")
    
    simulator.add_frame_callback(frame_callback)
    simulator.add_event_callback(event_callback)
    
    try:
        # Create a scene
        print("\nCreating physics scene...")
        result = bridge.predict_and_simulate("create a ball on a ramp")
        
        if result['total_objects'] > 0:
            # Start real-time simulation
            print("\nStarting real-time simulation...")
            simulator.start_simulation(duration=5.0, record=True)
            
            # Monitor simulation
            while simulator.running:
                time.sleep(1)
                stats = simulator.get_simulation_stats()
                if 'current_step' in stats:
                    print(f"Stats: {stats['current_step']} steps, {stats['actual_fps']:.1f} FPS")
            
            # Analyze results
            print("\nAnalyzing motion...")
            analysis = simulator.analyze_motion()
            print(f"Motion analysis: {len(analysis.get('objects', {}))} objects analyzed")
            
            # Export recording
            simulator.export_recording("data/simulation_recording.json")
        
        else:
            print("No objects created, skipping simulation")
    
    finally:
        simulator.stop_simulation()
        bridge.disconnect()
    
    print("\n✅ Real-time simulator test completed!")


if __name__ == "__main__":
    import os
    test_realtime_simulator()
