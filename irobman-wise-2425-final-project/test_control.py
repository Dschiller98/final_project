# This file contains the test for the Inverse Kinematics solver.

import yaml

import numpy as np

from typing import Dict, Any

from scipy.spatial.transform import Rotation as R

from src.simulation import Simulation

from src.control import Controller

def test_ik(config: Dict[str, Any]):
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    sim = Simulation(config)
    controller = Controller(sim)

    passed = 0
    total = 0
    for tstep in range(100):
        total += 1
        sim.reset("YcbTennisBall")
        print("---------------------------------")
        print((f"Timestep: {tstep}"))

        # Sample random goal position making sure it is above the table
        goal_position = np.array([np.random.uniform(0.0, 0.5), np.random.uniform(-0.7, -0.4), 
                                  np.random.uniform(1.4, 1.6)])
        
        # Sample random goal orientation
        goal_orientation = np.array([np.random.uniform(0.0, 0.1), np.random.uniform(0.9, 1.0), 
                                  np.random.uniform(0.0, 0.1), np.random.uniform(0.0, 0.1)])

        print(f"Testing IK Solver...")
        print(f"Goal Position: {goal_position}")
        print(f"Goal Orientation: {goal_orientation}")

        # Move the robot to the goal position using the IK solver
        controller.move_without_obstacles(goal_position, goal_orientation)

        # Get the final end-effector position
        final_position, final_orientation = sim.robot.get_ee_pose()

        # Check if the final position is within the threshold
        error_pos = np.linalg.norm(goal_position - final_position)
        print(f"Final Position: {final_position}, Error: {error_pos}")
        # Check if the final orientation is within the threshold
        error_ori = np.linalg.norm(goal_orientation - final_orientation)
        print(f"Final Orientation: {final_orientation}, Error: {error_ori}")

        if error_pos < 1e-2 and error_ori < 1e-1:
            print("Test Passed: Robot reached the goal position.")
            passed += 1
        else:
            print("Test Failed: Robot did not reach the goal position.")
    sim.close()
    print(f"Total tests Passed: {passed} out of {total}")

with open("irobman-wise-2425-final-project/configs/test_config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
        print(config)
    except yaml.YAMLError as exc:
        print(exc)
test_ik(config)