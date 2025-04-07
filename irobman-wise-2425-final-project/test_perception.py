# This file contains the test for the Pose Estimator.
import os
import glob
import yaml

import numpy as np

import pybullet as p

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from scipy.spatial.transform import Rotation as R

from src.simulation import Simulation

from src.perception import PoseEstimator

from src.control import Controller

def test_pose_estimator(config: Dict[str, Any]):
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [file.split('/')[-1] for file in files]
    sim = Simulation(config)
    controller = Controller(sim)
    pose_estimator = PoseEstimator(sim, controller)

    passed = 0
    total = 0
    for obj_name in obj_names:
        for tstep in range(10):
            total += 1
            sim.reset(obj_name)
            print("---------------------------------")
            print((f"Timestep: {tstep}"))

            print(f"Testing Pose Estimator...")
            print(f"Object: {obj_name}, Position: {sim.get_ground_tuth_position_object}")

            # wait for object to fall onto the table before starting the program
            for i in range(150):
                sim.step()
                
            obj_id = sim.object.id

            pos, pcd = pose_estimator.estimate_object_pose_all_cameras(obj_id)

            print(f"Estimated Position: {pos}")
            p.addUserDebugPoints(pcd.tolist(), [[0, 1, 0] for _ in range(len(pcd))], pointSize=2)

            error = np.linalg.norm(sim.get_ground_tuth_position_object[:2] - pos[:2])

            print(f"Error: {error}")

            if error < 5e-2:
                print("Test Passed: Estimated correct object position.")
                passed += 1
            else:
                print("Test Failed: Estimated false object position.")
    sim.close()
    print(f"Total tests Passed: {passed} out of {total}")

with open("irobman-wise-2425-final-project/configs/test_config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
        print(config)
    except yaml.YAMLError as exc:
        print(exc)
test_pose_estimator(config)