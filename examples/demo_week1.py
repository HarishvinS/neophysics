"""
Week 1 Demo - Basic Physics Engine
Demonstrates the completed Week 1 functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from physics_engine import PhysicsEngine
import time


def demo_basic_physics():
    """Demonstrate basic physics with visual feedback."""
    print("🎬 Week 1 Demo: Basic Physics Engine")
    print("=" * 50)
    
    # Create physics engine with GUI
    print("Initializing physics engine with 3D viewer...")
    engine = PhysicsEngine(use_gui=True)
    
    print("\n🏗️ Creating scene...")
    
    # Create a ramp
    print("Adding ramp...")
    ramp_id = engine.create_ramp(
        position=[0, 0, 0],
        angle=0.4,  # ~23 degrees
        size=[3, 0.2, 1],
        color=(0.2, 0.2, 0.8)  # Blue
    )
    
    # Create multiple balls
    print("Adding balls...")
    balls = []
    
    # Red ball - heavy
    ball1 = engine.create_sphere(
        position=[-1.2, -0.3, 1.5],
        radius=0.12,
        mass=3.0,
        color=(1, 0, 0)
    )
    balls.append(("Heavy red ball", ball1))
    
    # Green ball - light
    ball2 = engine.create_sphere(
        position=[-1.0, 0, 1.5],
        radius=0.08,
        mass=0.5,
        color=(0, 1, 0)
    )
    balls.append(("Light green ball", ball2))
    
    # Blue ball - medium
    ball3 = engine.create_sphere(
        position=[-0.8, 0.3, 1.5],
        radius=0.10,
        mass=1.5,
        color=(0, 0, 1)
    )
    balls.append(("Medium blue ball", ball3))
    
    # Add some boxes too
    print("Adding boxes...")
    box1 = engine.create_box(
        position=[2, 1, 0.5],
        half_extents=[0.2, 0.2, 0.2],
        mass=1.0,
        color=(1, 1, 0)  # Yellow
    )
    
    box2 = engine.create_box(
        position=[2, -1, 0.5],
        half_extents=[0.15, 0.15, 0.3],
        mass=0.8,
        color=(1, 0, 1)  # Magenta
    )
    
    print("\n🎬 Starting simulation...")
    print("Watch the PyBullet 3D viewer to see the physics in action!")
    print("The balls will roll down the ramp. Observe how their different properties affect their motion.")
    
    # Run simulation with status updates
    duration = 8.0
    display_fps = 60
    # PyBullet simulation runs at 240Hz, so we do 4 sim steps per display frame for real-time
    sim_steps_per_display_frame = 240 // display_fps
    total_display_frames = int(duration * display_fps)
    
    for i in range(total_display_frames):
        engine.step_simulation(steps=sim_steps_per_display_frame)
        
        # Print status every 2 seconds
        if i > 0 and i % (display_fps * 2) == 0:
            elapsed = i / display_fps
            print(f"⏱️ Simulation time: {elapsed:.1f}s / {duration}s")
            
            # Show ball positions
            for name, ball_id in balls:
                state = engine.get_object_state(ball_id)
                pos = state['position']
                speed = (state['velocity'][0]**2 + state['velocity'][1]**2 + state['velocity'][2]**2)**0.5
                print(f"   {name}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), speed={speed:.2f}m/s")
        
        time.sleep(1.0 / display_fps)
    
    print("\n✅ Simulation complete!")
    
    # Final positions
    print("\n📊 Final Results:")
    for name, ball_id in balls:
        state = engine.get_object_state(ball_id)
        pos = state['position']
        print(f"   {name}: final position = ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    
    print("\n🎉 Week 1 Demo Complete!")
    print("Key achievements:")
    print("✅ PyBullet physics engine integration")
    print("✅ 3D object creation (spheres, boxes, ramps)")
    print("✅ Realistic physics simulation (gravity, collisions, friction)")
    print("✅ Material properties and mass effects")
    print("✅ Real-time 3D visualization")
    
    input("\nPress Enter to close the demo...")
    engine.disconnect()


if __name__ == "__main__":
    demo_basic_physics()
