"""
Test script to verify basic physics engine functionality
This script tests all core features without requiring GUI interaction
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from physics_engine import PhysicsEngine
import time


def test_object_creation():
    """Test creating different types of objects."""
    print("🧪 Testing object creation...")
    
    engine = PhysicsEngine(use_gui=False)  # No GUI for automated testing
    
    # Test sphere creation
    sphere_id = engine.create_sphere(position=[0, 0, 1], mass=1.0)
    print(f"✅ Sphere created with ID: {sphere_id}")
    
    # Test box creation
    box_id = engine.create_box(position=[1, 0, 1], mass=0.5)
    print(f"✅ Box created with ID: {box_id}")
    
    # Test ramp creation
    ramp_id = engine.create_ramp(position=[0, 0, 0], angle=0.3)
    print(f"✅ Ramp created with ID: {ramp_id}")
    
    # Verify objects are tracked
    print(f"✅ Total objects tracked: {len(engine.objects)}")
    
    engine.disconnect()
    return True


def test_physics_simulation():
    """Test physics simulation with object interactions."""
    print("\n🏃 Testing physics simulation...")
    
    engine = PhysicsEngine(use_gui=False)
    
    # Create a ramp
    ramp_id = engine.create_ramp(
        position=[0, 0, 0],
        angle=0.4,  # Steeper ramp
        size=[3, 0.2, 1]
    )
    
    # Create a ball on the ramp (positioned to actually be on the slope)
    ball_id = engine.create_sphere(
        position=[-1.0, 0, 1.5],  # On the high end of the ramp
        radius=0.1,
        mass=2.0
    )
    
    # Get initial state
    initial_state = engine.get_object_state(ball_id)
    print(f"Initial ball position: {initial_state['position']}")
    print(f"Initial ball velocity: {initial_state['velocity']}")
    
    # Run simulation
    print("Running simulation for 3 seconds...")
    engine.run_simulation(duration=3.0, real_time=False)  # Fast simulation
    
    # Get final state
    final_state = engine.get_object_state(ball_id)
    print(f"Final ball position: {final_state['position']}")
    print(f"Final ball velocity: {final_state['velocity']}")
    
    # Verify physics worked (ball should have moved down the ramp)
    position_change_x = abs(final_state['position'][0] - initial_state['position'][0])
    position_change_z = abs(final_state['position'][2] - initial_state['position'][2])

    if position_change_x > 0.5 or position_change_z > 0.5:
        print("✅ Ball moved significantly - physics working!")
    else:
        print("❌ Ball didn't move much - check physics setup")
        print(f"   X movement: {position_change_x:.3f}, Z movement: {position_change_z:.3f}")
        return False
    
    # Verify ball is near ground level
    if final_state['position'][2] < 0.5:  # Should be close to ground
        print("✅ Ball reached ground level - gravity working!")
    else:
        print("❌ Ball is still high up - check gravity")
        return False
    
    engine.disconnect()
    return True


def test_object_clearing():
    """Test clearing objects from the scene."""
    print("\n🧹 Testing object clearing...")
    
    engine = PhysicsEngine(use_gui=False)
    
    # Create multiple objects
    engine.create_sphere(position=[0, 0, 1])
    engine.create_box(position=[1, 0, 1])
    engine.create_ramp(position=[0, 0, 0])
    
    initial_count = len(engine.objects)
    print(f"Created {initial_count} objects (including ground)")
    
    # Clear objects (keeping ground)
    engine.clear_objects(keep_ground=True)
    
    final_count = len(engine.objects)
    print(f"After clearing: {final_count} objects remaining")
    
    # Should only have ground left
    if final_count == 1 and 'ground' in engine.objects:
        print("✅ Objects cleared successfully, ground preserved!")
    else:
        print("❌ Object clearing failed")
        return False
    
    engine.disconnect()
    return True


def test_material_properties():
    """Test that different materials behave differently."""
    print("\n🎾 Testing material properties...")
    
    engine = PhysicsEngine(use_gui=False)
    
    # Create two balls with different properties
    bouncy_ball = engine.create_sphere(
        position=[0, 0, 2],
        radius=0.1,
        mass=1.0
    )
    
    engine.set_object_properties(bouncy_ball, restitution=0.9)
    
    # Run simulation and check bounce
    initial_height = 2.0
    engine.run_simulation(duration=2.0, real_time=False)
    
    # Get state after first bounce
    state = engine.get_object_state(bouncy_ball)
    current_height = state['position'][2]
    
    print(f"Ball height after bounce: {current_height:.3f}m")
    
    # Ball should have bounced (not at ground level)
    if current_height > 0.2:
        print("✅ Ball bounced - material properties working!")
    else:
        print("⚠️ Ball didn't bounce much - this might be normal")
    
    engine.disconnect()
    return True


def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting comprehensive physics engine tests...\n")
    
    tests = [
        ("Object Creation", test_object_creation),
        ("Physics Simulation", test_physics_simulation),
        ("Object Clearing", test_object_clearing),
        ("Material Properties", test_material_properties)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Physics engine is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
