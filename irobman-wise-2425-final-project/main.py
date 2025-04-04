import os
import glob
import yaml

import numpy as np

from typing import Dict, Any

from pybullet_object_models import ycb_objects  # type:ignore

from src.simulation import Simulation

from src.pose_estimator import PoseEstimator

from src.ik import move_to_goal, get_joint_config, forward_kinematics

from src.grasping import PickAndPlace, GraspPlanner

from src.tracking import ObstacleTracker

from src.planning import LocalPlanner


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
        for tstep in range(10):
            sim.reset("YcbPear")
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


            """pcd = pose_estimator.estimate_object_pcd(obj_id)
            table_center = pose_estimator.estimate_position_from_ee(2)
            #p.addUserDebugPoints([obj_position], [[1, 0, 0]], pointSize=10)

            
            pnp = PickAndPlace(sim.robot, sim, table_center)
            grasp_planner = GraspPlanner(pcd, sim.robot)
            #p.addUserDebugPoints(pcd.tolist(), [[1, 0, 0] for _ in range(len(pcd))], pointSize=2)

            grasp = grasp_planner.find_best_grasp()

            pnp.pick_and_place(grasp[0], grasp[1], goal_position=[0.5, 0.4, 1.4])

            print(f"[{i}] Goal Satisfied: {sim.check_goal()}")"""

            start_config = sim.robot.get_joint_positions()
            start_ee = sim.robot.get_ee_pose()[0]
            estimate = forward_kinematics(start_config)

            goal_joint_angles = get_joint_config(sim.robot, [0.5, 0.4, 1.4], [0, 1, 0, 0])
            
            for i in range(2000):
                sim.step()
                # for getting renders
                #static_rgb, static_depth, static_seg = sim.get_static_renders()
                #ee_rgb, ee_depth, ee_seg = sim.get_ee_renders()

                # TODO: plot line from first to last prediction and bounding box for last prediction

                if i % 1 == 0:
                    states = tracker.track_obstacles()
                    _, radii = tracker.estimate_obstacle_parameters()
                    for state, radius in zip(states.values(), radii):
                        tracker.draw_bounding_boxes(state["position"], radius)
                        p.addUserDebugPoints([state["position"]], [[1, 0, 0]], pointSize=8)
                obs_position_guess = np.array([states[0]["position"], states[1]["position"]])

                loc_planner = LocalPlanner(sim, goal_joint_angles, obs_position_guess, repulsive_threshold=np.max(radii))
                out = loc_planner.move_toward_goal()
                print((f"[{i}] Obstacle Position-Diff: "
                       f"{sim.check_obstacle_position(obs_position_guess)}"))
                """
                obs_traj = tracker.predict_trajectory()
                for j in range(50):
                    sim.step()
                    for state, radius in zip(obs_traj.values(), radii):
                        tracker.draw_bounding_boxes(state[j], radius)
                        
                """
                
                print(f"Robot End Effector Position: {sim.robot.get_ee_pose()[0]}")
                print(f"Robot End Effector Orientation: {sim.robot.get_ee_pose()[1]}")
                
                
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
