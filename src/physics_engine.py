"""
Basic Physics Engine using PyBullet
Handles 3D physics simulation with gravity, collisions, and basic objects.
"""

import pybullet as p
import time
import numpy as np
from typing import List, Tuple, Dict, Optional


class PhysicsEngine:
    """Main physics engine class using PyBullet for simulation."""
    
    def __init__(self, use_gui=True):
        """
        Initialize the physics engine.
        
        Args:
            use_gui (bool): Whether to show PyBullet's 3D viewer
        """
        self.use_gui = use_gui
        self.physics_client = None
        self.objects = {}  # Track created objects
        self.object_counter = 0
        
        self.setup_physics_world()
    
    def setup_physics_world(self):
        """Initialize the physics world with basic settings."""
        # Connect to PyBullet
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set gravity (Earth-like)
        p.setGravity(0, 0, -9.81)
        
        # Set time step for simulation
        p.setTimeStep(1./240.)  # 240 Hz
        
        # Create ground plane
        self.create_ground_plane()
        
        print("Physics world initialized successfully!")
    
    def create_ground_plane(self):
        """Create a static ground plane for objects to land on."""
        ground_shape = p.createCollisionShape(p.GEOM_PLANE)
        ground_body = p.createMultiBody(
            baseMass=0,  # Static object
            baseCollisionShapeIndex=ground_shape,
            basePosition=[0, 0, 0]
        )
        
        # Set ground material properties
        p.changeDynamics(
            ground_body, -1,
            lateralFriction=0.7,
            restitution=0.1
        )
        
        self.objects['ground'] = ground_body
        return ground_body
    
    def create_sphere(self, position=(0, 0, 1), radius=0.1, mass=1.0, color=(1, 0, 0),
                      restitution=0.8, lateral_friction=0.5):
        """
        Create a sphere object.
        
        Args:
            position: (x, y, z) position
            radius: Sphere radius
            mass: Object mass (0 for static)
            color: RGB color tuple
            restitution: Bounciness of the object
            lateral_friction: Friction of the object
            
        Returns:
            int: Object ID
        """
        # Create collision shape
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius
        )
        
        # Create visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=list(color) + [1.0]
        )
        
        # Create multi-body
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        # Set material properties
        p.changeDynamics(
            body_id, -1,
            lateralFriction=lateral_friction,
            restitution=restitution
        )
        
        # Track object
        object_name = f"sphere_{self.object_counter}"
        self.objects[object_name] = body_id
        self.object_counter += 1
        
        return body_id
    
    def create_box(self, position=(0, 0, 1), half_extents=(0.5, 0.5, 0.5), 
                   mass=1.0, color=(0, 1, 0), restitution=0.3, lateral_friction=0.6):
        """
        Create a box object.
        
        Args:
            position: (x, y, z) position
            half_extents: (x, y, z) half-extents of the box
            mass: Object mass (0 for static)
            color: RGB color tuple
            restitution: Bounciness of the object
            lateral_friction: Friction of the object
            
        Returns:
            int: Object ID
        """
        # Create collision shape
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents
        )
        
        # Create visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=list(color) + [1.0]
        )
        
        # Create multi-body
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        # Set material properties
        p.changeDynamics(
            body_id, -1,
            lateralFriction=lateral_friction,
            restitution=restitution
        )
        
        # Track object
        object_name = f"box_{self.object_counter}"
        self.objects[object_name] = body_id
        self.object_counter += 1
        
        return body_id
    
    def create_ramp(self, position=(0, 0, 0), angle=0.3, size=(2, 0.2, 1), 
                    mass=0, color=(0, 0, 1), restitution=0.2, lateral_friction=0.6):
        """
        Create a ramp (inclined plane).
        
        Args:
            position: (x, y, z) position
            angle: Rotation angle in radians
            size: (length, thickness, width) of the ramp
            mass: Object mass (0 for static)
            color: RGB color tuple
            restitution: Bounciness of the object
            lateral_friction: Friction of the object
            
        Returns:
            int: Object ID
        """
        # Create collision shape
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[size[0]/2, size[1]/2, size[2]/2]
        )
        
        # Create visual shape
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[size[0]/2, size[1]/2, size[2]/2],
            rgbaColor=list(color) + [1.0]
        )
        
        # Calculate orientation (rotation around Y-axis)
        orientation = p.getQuaternionFromEuler([0, angle, 0])
        
        # Create multi-body
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        
        # Set material properties
        p.changeDynamics(
            body_id, -1,
            lateralFriction=lateral_friction,
            restitution=restitution
        )
        
        # Track object
        object_name = f"ramp_{self.object_counter}"
        self.objects[object_name] = body_id
        self.object_counter += 1
        
        return body_id

    def set_object_properties(self, object_id, restitution=None, lateral_friction=None):
        """
        Set physics properties for an object after creation.
        
        Args:
            object_id: PyBullet object ID
            restitution: New bounciness value
            lateral_friction: New friction value
        """
        p.changeDynamics(object_id, -1, restitution=restitution, lateralFriction=lateral_friction)
    
    def step_simulation(self, steps=1):
        """
        Step the physics simulation forward.
        
        Args:
            steps: Number of simulation steps
        """
        for _ in range(steps):
            p.stepSimulation()
    
    def run_simulation(self, duration=5.0, real_time=True):
        """
        Run simulation for a specified duration.
        
        Args:
            duration: Simulation duration in seconds
            real_time: Whether to run in real-time
        """
        steps = int(duration * 240)  # 240 Hz
        
        for i in range(steps):
            p.stepSimulation()
            
            if real_time:
                time.sleep(1./240.)
    
    def get_object_state(self, object_id):
        """
        Get the current state of an object.
        
        Args:
            object_id: PyBullet object ID
            
        Returns:
            dict: Object state information
        """
        try:
            pos, orn = p.getBasePositionAndOrientation(object_id)
            vel, ang_vel = p.getBaseVelocity(object_id)
            
            return {
                'position': np.array(pos),
                'orientation': np.array(orn),
                'velocity': np.array(vel),
                'angular_velocity': np.array(ang_vel)
            }
        except Exception as e:
            print(f"Error getting object state: {e}")
            return None
    
    def clear_objects(self, keep_ground=True):
        """
        Remove all objects from the simulation.
        
        Args:
            keep_ground: Whether to keep the ground plane
        """
        for name, object_id in list(self.objects.items()):
            if keep_ground and name == 'ground':
                continue
            
            p.removeBody(object_id)
            del self.objects[name]
        
        # Don't reset counter to maintain unique naming
        # self.object_counter = 0
    
    def disconnect(self):
        """Disconnect from PyBullet."""
        if self.physics_client is not None:
            p.disconnect(physicsClientId=self.physics_client)
            self.physics_client = None


# Test function
def test_physics_engine():
    """Test the basic physics engine functionality."""
    print("Testing Physics Engine...")
    
    # Create physics engine
    engine = PhysicsEngine(use_gui=True)
    
    # Create a ramp
    ramp_id = engine.create_ramp(
        position=[0, 0, 0],
        angle=0.3,  # ~17 degrees
        color=(0.5, 0.5, 1.0)
    )
    
    # Create a ball on the ramp
    ball_id = engine.create_sphere(
        position=[1, 0, 2],
        radius=0.1,
        mass=2.0,
        color=(1, 0, 0)
    )
    
    # Run simulation
    print("Running simulation for 5 seconds...")
    engine.run_simulation(duration=5.0, real_time=True)
    
    # Get final state
    ball_state = engine.get_object_state(ball_id)
    print(f"Final ball position: {ball_state['position']}")
    print(f"Final ball velocity: {ball_state['velocity']}")
    
    # Keep window open (optional for automated testing)
    try:
        input("Press Enter to close...")
    except (EOFError, KeyboardInterrupt):
        print("\nTest completed.")
    
    # Cleanup
    engine.disconnect()


if __name__ == "__main__":
    test_physics_engine()
