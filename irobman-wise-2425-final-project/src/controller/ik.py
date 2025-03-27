'''
Implement an IK-solver for the Franka-robot. 
You can use the pseudo-inverse or the transpose based solution. 
Use Your IK-solver to move the robot to a certain goal position. 
This Controller gets used throughout the project (e.g. executing the grasp - moving the object to the goal).
'''

import numpy as np
import pybullet as p
from robot import Robot

def ik_solver(robot: Robot, goal_position: np.ndarray, max_iters: int = 100, threshold: float = 1e-3):
        """
        Iterative inverse Kinematics solver using Jacobian pseudo-inverse.

        Args:
            goal_position: Desired end-effector position (x, y, z)
            max_iters: Maximum number of iterations
            threshold: Convergence threshold for position error
        """
        for _ in range(max_iters):
            # Get current end-effector pose
            current_position, _ = robot.get_ee_pose()

            # Compute position error
            error = goal_position - current_position

            # Check if the error is within the threshold
            if np.linalg.norm(error) < threshold:
                break

            # Compute the Jacobian matrix
            jacobian = p.calculateJacobian(
                robot.id,
                robot.ee_idx,
                localPosition=[0, 0, 0],
                objPositions=robot.get_joint_positions().tolist(),
                objVelocities=[0.0] * len(robot.arm_idx),
                objAccelerations=[0.0] * len(robot.arm_idx),
            )[0]

            # Compute joint velocity using the pseudo-inverse of the Jacobian
            jacobian_pseudo_inverse = np.linalg.pinv(jacobian)
            joint_velocities = np.dot(jacobian_pseudo_inverse, error)

            # Update joint positions
            current_positions = robot.get_joint_positions()
            new_positions = current_positions + joint_velocities * 0.1  # Scale step size
            robot.position_control(new_positions)

def move_to_goal(robot, goal_position: np.ndarray):
    """
    Moves the robot to the specified goal position using the IK solver.

    Args:
        goal_position: Desired end-effector position (x, y, z)
    """
    robot.ik_solver(goal_position)
    
