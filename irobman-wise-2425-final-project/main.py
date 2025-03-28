import os
import glob
import yaml

import numpy as np

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation

from src.pose_estimator import PoseEstimator

from src.ik import move_to_goal


def run_exp(config: Dict[str, Any]):
    # Example Experiment Runner File
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [file.split('/')[-1] for file in files]
    sim = Simulation(config)
    pose_estimator = PoseEstimator(sim)
    for obj_name in obj_names:
        for tstep in range(10):
            sim.reset("YcbPowerDrill")
            print((f"Object: {obj_name}, Timestep: {tstep},"
                   f" pose: {sim.get_ground_tuth_position_object}"))
            pos, ori = sim.robot.pos, sim.robot.ori
            print(f"Robot inital pos: {pos} orientation: {ori}")
            l_lim, u_lim = sim.robot.lower_limits, sim.robot.upper_limits
            print(f"Robot Joint Range {l_lim} -> {u_lim}")
            sim.robot.print_joint_infos()
            jpos = sim.robot.get_joint_positions()
            print(f"Robot current Joint Positions: {jpos}")
            jvel = sim.robot.get_joint_velocites()
            print(f"Robot current Joint Velocites: {jvel}")
            ee_pos, ee_ori = sim.robot.get_ee_pose()
            print(f"Robot End Effector Position: {ee_pos}")
            print(f"Robot End Effector Orientation: {ee_ori}")

            # wait for object to fall onto the table before starting the program
            for i in range(400):
                sim.step()
                
            obj_id = sim.object.id

            try:
                obj_position = pose_estimator.estimate_position_from_static(obj_id)
                p.addUserDebugPoints([obj_position], [[1, 0, 0]], pointSize=10)

            except Exception as e:
                print(f"Error in estimating object position from static camera image: {e}")
                #obj_position = pose_estimator.scan_table(obj_id)
            
            move_to_goal(sim.robot, np.array([0.0, -0.65, 1.40]), [0, 1, 0, 0])
            p.addUserDebugPoints([sim.robot.get_ee_pose()[0]], [[0, 1, 0]], pointSize=10)
            for i in range(10000):
                sim.step()
                # for getting renders
                #static_rgb, static_depth, static_seg = sim.get_static_renders()
                #ee_rgb, ee_depth, ee_seg = sim.get_ee_renders()

                
                print(f"Robot End Effector Position: {sim.robot.get_ee_pose()[0]}")
                print(f"Robot End Effector Orientation: {sim.robot.get_ee_pose()[1]}")
                obs_position_guess = np.zeros((2, 3))
                print((f"[{i}] Obstacle Position-Diff: "
                       f"{sim.check_obstacle_position(obs_position_guess)}"))
                goal_guess = np.zeros((7,))
                print((f"[{i}] Goal Obj Pos-Diff: "
                       f"{sim.check_goal_obj_pos(goal_guess)}"))
                print(f"[{i}] Goal Satisfied: {sim.check_goal()}")
    sim.close()


if __name__ == "__main__":
    import pybullet as p
    with open("irobman-wise-2425-final-project/configs/test_config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)
    run_exp(config)
