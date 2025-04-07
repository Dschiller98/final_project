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

from src.tracking import ObstacleTracker

from src.control import Controller

def test_tracking(config: Dict[str, Any]):
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [file.split('/')[-1] for file in files]
    sim = Simulation(config)
    controller = Controller(sim)
    pose_estimator = PoseEstimator(sim, controller)
    tracker = ObstacleTracker(pose_estimator)

    passed = 0
    total = 0
    print(f"Testing Obstacle tracking...")
    for tstep in range(3):
        sim.reset("YcbTennisBall")
        print("---------------------------------")
        print((f"Timestep: {tstep}"))

        for i in range(500):
            total += 1
            sim.step()
            states = tracker.track_obstacles()
            for state in states.values():
                p.addUserDebugPoints([state["position"]], [[0, 1, 0]], pointSize=6)
            obs_position_guess = np.array([states[0]["position"], states[1]["position"]])

            print((f"[{i}] Obstacle Position-Diff: "
                        f"{sim.check_obstacle_position(obs_position_guess)}"))

            if sim.check_obstacle_position(obs_position_guess) < 2e-2:
                print("Test Passed: Obstcale tracked successfully.")
                passed += 1
            else:
                print("Test Failed: Obstacle not tracked successfully.")
    sim.close()
    print(f"Total tests Passed: {passed} out of {total}")

with open("irobman-wise-2425-final-project/configs/test_config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
        config["world_settings"]["turn_on_obstacles"] = True
        print(config)
    except yaml.YAMLError as exc:
        print(exc)
test_tracking(config)