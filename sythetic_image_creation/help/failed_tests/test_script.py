# Basic test script for Isaac Sim
# Save this as test_script.py

import os
import time
import datetime

# Start with clear indicator that our script is running
print("=" * 50)
print("TEST SCRIPT STARTING - BEFORE IMPORTS")
print("=" * 50)

try:
    # Try to import the SimulationApp
    from omni.isaac.kit import SimulationApp

    print("Successfully imported SimulationApp")

    # Initialize the simulation app
    print("Initializing SimulationApp...")
    simulation_app = SimulationApp({"headless": False})
    print("SimulationApp initialized")

    # Try to import other modules
    print("Importing additional modules...")
    import omni.replicator.core as rep

    print("Successfully imported omni.replicator.core")

    # Create a simple scene
    print("Creating a test scene...")
    rep.create.scene()
    print("Scene created")

    # Create a sphere
    print("Creating a sphere...")
    sphere = rep.create.sphere(position=(0, 0, 0), radius=1.0)
    print("Sphere created")

    # Run for a few steps
    print("Running simulation for 10 steps...")
    for i in range(10):
        simulation_app.update()
        print(f"Step {i + 1}/10 completed")
        time.sleep(0.5)  # Add delay to make it visible

    print("Test completed successfully")

except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback

    print(traceback.format_exc())
finally:
    print("Test script finishing...")
    try:
        simulation_app.close()
        print("SimulationApp closed")
    except:
        print("Could not close SimulationApp (might not be initialized)")

    print("TEST SCRIPT COMPLETE")