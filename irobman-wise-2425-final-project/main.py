import os
import glob
import yaml

import numpy as np

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation

from src.pose_estimator import PoseEstimator

from src.ik import move_to_goal

from src.grasping import PickAndPlace, GraspPlanner

from src.tracking import ObstacleTracker


def run_exp(config: Dict[str, Any]):
    # Example Experiment Runner File
    print("Simulation Start:")
    print(config['world_settings'], config['robot_settings'])
    object_root_path = ycb_objects.getDataPath()
    files = glob.glob(os.path.join(object_root_path, "Ycb*"))
    obj_names = [file.split('/')[-1] for file in files]
    sim = Simulation(config)
    pose_estimator = PoseEstimator(sim)
    tracker = ObstacleTracker(pose_estimator)
    for obj_name in obj_names:
        for tstep in range(1):
            sim.reset(obj_name)
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

            #p.addUserDebugPoints([[0.1, -0.5, 2]], [[1, 0, 0]], pointSize=10)


            #pcd = pose_estimator.estimate_object_pcd(obj_id)
            #p.addUserDebugPoints([obj_position], [[1, 0, 0]], pointSize=10)

            
            #pnp = PickAndPlace(sim.robot, sim)
            #grasp_planner = GraspPlanner(pcd)
            #p.addUserDebugPoints(pcd.tolist(), [[1, 0, 0] for _ in range(len(pcd))], pointSize=2)

            #grasp = grasp_planner.find_best_grasp()

            #pnp.pick_and_place(grasp[0], grasp[1], goal_position=[0.5, 0.4, 1.4])

            

            for i in range(2000):
                sim.step()
                # for getting renders
                #static_rgb, static_depth, static_seg = sim.get_static_renders()
                #ee_rgb, ee_depth, ee_seg = sim.get_ee_renders()

                if i % 5 == 0:
                    states = tracker.track_obstacles()
                    _, radii = tracker.estimate_obstacle_parameters()
                    for state, radius in zip(states.values(), radii):
                        tracker.draw_bounding_boxes(state["position"], radius)
                
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
