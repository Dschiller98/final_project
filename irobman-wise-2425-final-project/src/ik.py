'''
Implement an IK-solver for the Franka-robot. 
You can use the pseudo-inverse or the transpose based solution. 
Use Your IK-solver to move the robot to a certain goal position. 
This Controller gets used throughout the project (e.g. executing the grasp - moving the object to the goal).
'''

import numpy as np
import pybullet as p
from .robot import Robot

def ik_solver(robot: Robot, goal_position: np.ndarray, goal_orientation: np.ndarray, max_iters: int = 5000, threshold: float = 7e-3):
        """
        Iterative inverse Kinematics solver using Jacobian pseudo-inverse.

        Args:
            goal_position: Desired end-effector position (x, y, z)
            max_iters: Maximum number of iterations
            threshold: Convergence threshold for position error
        """
        for i in range(max_iters):
            # Get current end-effector pose
            current_position, current_orientation = robot.get_ee_pose()

            # Compute position error
            position_error = goal_position - current_position

             # Compute orientation error
            orientation_error = p.getDifferenceQuaternion(current_orientation, goal_orientation)
            orientation_error = np.array(orientation_error)
            # Make the scale part of the quaternion positive
            if orientation_error[3] < 0:
                orientation_error[3] = -orientation_error[3]
            # Convert the orientation error to axis-angle representation
            axis, angle = p.getAxisAngleFromQuaternion(orientation_error)
            orientation_error = np.array(axis) * angle
            
            # Combine position and orientation error
            error = np.concatenate((position_error, orientation_error))

            #print(f"error: {np.linalg.norm(error)}")

            # Check if the error is within the threshold
            if np.linalg.norm(error) < threshold:
                print(f"Converged in {i} iterations")
                break

            # Compute the Jacobian matrix
            jacobian_lin, jacobian_rot = p.calculateJacobian(
                robot.id,
                robot.ee_idx,
                localPosition=[0.0, 0.0, 0.0],
                objPositions=robot.get_joint_positions().tolist() + robot.get_gripper_positions().tolist(),
                objVelocities=[0.0] * (len(robot.arm_idx)+len(robot.gripper_idx)),
                objAccelerations=[0.0] * (len(robot.arm_idx)+len(robot.gripper_idx)),
            )
            jacobian_lin = np.array(jacobian_lin, order='F')
            jacobian_rot = np.array(jacobian_rot, order='F')
            
            jacobian = np.vstack((jacobian_lin, jacobian_rot))

            # Compute the damped pseudo-inverse of the Jacobian
            identity = np.eye(jacobian.shape[0])
            jacobian_damped_pseudo_inverse = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + 0.01**2 * identity)

            joint_angles = np.dot(jacobian_damped_pseudo_inverse, error)

            # Update joint positions
            current_positions = np.concatenate((robot.get_joint_positions(),robot.get_gripper_positions()))
            new_positions = current_positions + joint_angles * 0.05 # Scale step size
            # Clamp joint positions to within limits
            new_positions = np.clip(new_positions[robot.arm_idx], robot.lower_limits, robot.upper_limits)
            robot.position_control(new_positions[robot.arm_idx])

            # TODO make step with sim 
            p.stepSimulation()

def move_to_goal(robot, goal_position: np.ndarray, goal_rotation: np.ndarray = [0, -1, 0, 0]):
    """
    Moves the robot to the specified goal position using the IK solver.
    default orientation looks down on the table.

    Args:
        goal_position: Desired end-effector position (x, y, z)
        goal_rotation: Desired end-effector orientation (quaternion)
    """

    ik_solver(robot, goal_position, goal_rotation)

    # planning mit informed rrt star, artificial potential field für local planning
    
