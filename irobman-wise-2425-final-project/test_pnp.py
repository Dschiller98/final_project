# This file contains the test for moving the object to the goal.
import os
import glob
import yaml

import numpy as np

import pybullet as p

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

import matplotlib.pyplot as plt

from src.simulation import Simulation

from src.perception import PoseEstimator

from src.grasping import PickAndPlace, GraspPlanner

from src.control import Controller

def test_pnp(config: Dict[str, Any]):
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [file.split('/')[-1] for file in files]
    sim = Simulation(config)
    controller = Controller(sim)
    pose_estimator = PoseEstimator(sim, controller)
    pnp = PickAndPlace(sim, controller)
    grasp_planner = GraspPlanner()

    passed = 0
    total = 0
    # Dictionary to store pass counts for each object
    object_pass_counts = {obj_name: 0 for obj_name in obj_names}
    print(f"Testing Pick and Place...")
    for obj_name in obj_names:
        for tstep in range(10):
            total += 1
            sim.reset(obj_name)
            print("---------------------------------")
            print((f"Timestep: {tstep}"))
            print(f"Object: {obj_name}, Position: {sim.get_ground_tuth_position_object}")

            # wait for object to fall onto the table before starting the program
            for i in range(150):
                sim.step()

            obj_id = sim.object.id

            _, pcd = pose_estimator.estimate_object_pose_all_cameras(obj_id)

            grasp_planner.set_pcd(pcd)

            grasp_pos, grasp_ori = grasp_planner.find_best_grasp()

            pnp.pick_and_place(grasp_pos, grasp_ori, [0.5, 0.4, 1.4])

            if sim.check_goal():
                print("Test Passed: Object placed successfully.")
                passed += 1
                object_pass_counts[obj_name] += 1
            else:
                print("Test Failed: Object not placed successfully.")
    sim.close()
    print(f"Total tests Passed: {passed} out of {total}")

    # Plot the results as a bar chart
    objects = list(object_pass_counts.keys())
    pass_counts = list(object_pass_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(objects, pass_counts, color='skyblue')
    plt.xlabel('Object Names')
    plt.ylabel('Number of Successful Runs')
    plt.title('Pick and Place Test Results')
    plt.xticks(rotation=45, ha='right')  # Rotate object names for better readability
    plt.tight_layout()
    plt.show()

with open("irobman-wise-2425-final-project/configs/test_config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
        print(config)
    except yaml.YAMLError as exc:
        print(exc)
test_pnp(config)