'''
Implement an IK-solver for the Franka-robot. 
You can use the pseudo-inverse or the transpose based solution. 
Use Your IK-solver to move the robot to a certain goal position. 
This Controller gets used throughout the project (e.g. executing the grasp - moving the object to the goal).
'''

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from .robot import Robot

def ik_solver(robot: Robot, goal_position: np.ndarray, goal_orientation: np.ndarray, max_iters: int = 5000, threshold: float = 7e-3):
        """
        Iterative inverse Kinematics solver using Jacobian pseudo-inverse.

        Args:
            goal_position: Desired end-effector position (x, y, z)
            max_iters: Maximum number of iterations
            threshold: Convergence threshold for position error
        """
        previous_positions = np.zeros(len(robot.arm_idx) + len(robot.gripper_idx))
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
            # Terminate if too little movement
            if np.linalg.norm(previous_positions - current_positions) < 5e-5:
                print("IK solver failed to converge.")
                break
            robot.position_control(new_positions[robot.arm_idx])

            previous_positions = current_positions

            # TODO make step with sim 
            p.stepSimulation()

def get_joint_config(robot: Robot, goal_position: np.ndarray, goal_orientation: np.ndarray, max_iters: int = 5000, threshold: float = 7e-3):
        """
        Iterative inverse Kinematics solver using Jacobian pseudo-inverse.

        Args:
            goal_position: Desired end-effector position (x, y, z)
            max_iters: Maximum number of iterations
            threshold: Convergence threshold for position error
        """
        # Get current joint positions
        current_pose = robot.get_joint_positions().tolist() + robot.get_gripper_positions().tolist()
        
        for i in range(max_iters):
            # Get current end-effector pose
            current_position, current_orientation = forward_kinematics(current_pose[:len(robot.arm_idx)])

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
                return current_pose[:len(robot.arm_idx)]

            # Compute the Jacobian matrix
            jacobian_lin, jacobian_rot = p.calculateJacobian(
                robot.id,
                robot.ee_idx,
                localPosition=[0.0, 0.0, 0.0],
                objPositions=current_pose,
                objVelocities=[0.0] * (len(current_pose)),
                objAccelerations=[0.0] * (len(current_pose)),
            )
            jacobian_lin = np.array(jacobian_lin, order='F')
            jacobian_rot = np.array(jacobian_rot, order='F')
            
            jacobian = np.vstack((jacobian_lin, jacobian_rot))

            # Compute the damped pseudo-inverse of the Jacobian
            identity = np.eye(jacobian.shape[0])
            jacobian_damped_pseudo_inverse = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + 0.01**2 * identity)

            joint_angles = np.dot(jacobian_damped_pseudo_inverse, error)

            # Update joint positions
            current_pose += joint_angles * 0.05 # Scale step size
            # Clamp joint positions to within limits
            current_pose = np.concatenate((np.clip(current_pose[robot.arm_idx], robot.lower_limits, robot.upper_limits), current_pose[-2:])).tolist()

        print("IK solver failed to converge in the maximum number of iterations.")
        return current_pose[:len(robot.arm_idx)]

def forward_kinematics(joint_angles: np.ndarray):
    #TODO dh parameter und dh matrix
    """
    Compute the forward kinematics for the Franka Panda robot.

    Args:
        joint_angles: A list of 7 joint angles (in radians).

    Returns:
        A tuple (position, orientation) representing the end-effector pose.
    """
    # DH parameters for the Franka Panda robot
    dh_params = [
        (0.0, 0.0, 0.333, joint_angles[0]),
        (0.0, -np.pi/2, 0.0, joint_angles[1]),
        (0.0, np.pi/2, 0.316, joint_angles[2]),
        (0.0825, np.pi/2, 0.0, joint_angles[3]),
        (-0.0825, -np.pi/2, 0.384, joint_angles[4]),
        (0.0, np.pi/2, 0.0, joint_angles[5]),
        (0.088, np.pi/2, 0.0, joint_angles[6]),
        (0.0, 0.0, 0.107, 0.0),
        (0.0, 0.0, 0.0, -np.pi/4),
        (0.0, 0.0, 0.1034, 0.0)
    ]


    # Start with the identity matrix
    T = np.eye(4)

    # Compute the cumulative transformation matrix
    for a, alpha, d, theta in dh_params:
        T = T @ dh_transform(a, alpha, d, theta)

    # Extract position and orientation
    position = T[:3, 3] + np.array([0, 0, 1.24])  # Add offset of the robot base
    orientation = R.from_matrix(T[:3, :3]).as_quat()

    return position, orientation

def dh_transform(a, alpha, d, theta):
    """
    Compute the transformation matrix using DH parameters.

    Args:
        a: Link length.
        alpha: Link twist.
        d: Link offset.
        theta: Joint angle.

    Returns:
        A 4x4 transformation matrix.
    """
    return np.array([[np.cos(theta), -np.sin(theta), 0, a],
                     [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                     [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                     [0, 0, 0, 1]])


def move_to_goal(robot, goal_position: np.ndarray, goal_rotation: np.ndarray = [0, 1, 0, 0]):
    """
    Moves the robot to the specified goal position using the IK solver.
    default orientation looks down on the table.

    Args:
        goal_position: Desired end-effector position (x, y, z)
        goal_rotation: Desired end-effector orientation (quaternion)
    """

    ik_solver(robot, goal_position, goal_rotation)

    # planning mit informed rrt star, artificial potential field für local planning
    
