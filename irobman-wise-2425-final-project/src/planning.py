
import pybullet as p
import numpy as np

class LocalPlanner:
    """
    A class to perform local motion planning using the Potential Field Method for obstacle avoidance.
    """

    def __init__(self, sim, attractive_gain=10.0, repulsive_gain=1.0, repulsive_threshold=0.5):
        """
        Initialize the LocalPlanner.

        Args:
            sim: The simulation object.
            goal_joint_angles: The goal joint angles as a numpy array (n,).
            obstacles: List of obstacle positions as [x, y, z].
            attractive_gain: Gain for the attractive force toward the goal.
            repulsive_gain: Gain for the repulsive force away from obstacles.
            repulsive_threshold: Distance threshold for repulsive forces to take effect.
        """
        self.sim = sim
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.repulsive_threshold = repulsive_threshold

    def compute_attractive_force(self, current_joint_angles, goal_joint_angles):
        """
        Compute the attractive force in joint space toward the goal.

        Args:
            current_joint_angles: The current joint angles as a numpy array (n,).

        Returns:
            The attractive force as a numpy array (n,).
        """
        force = self.attractive_gain * (goal_joint_angles - current_joint_angles)
        return force

    def compute_repulsive_force(self, current_joint_angles, obstacles):
        """
        Compute the repulsive force in joint space away from obstacles.

        Args:
            current_joint_angles: The current joint angles as a numpy array (n,).

        Returns:
            The total repulsive force as a numpy array (n,).
        """
        total_force = np.zeros_like(current_joint_angles)

        gripper_positions = self.sim.robot.get_gripper_positions()

        # Iterate over all links of the robot
        for link_idx in range(7):
            # Get the Cartesian position of the link
            link_state = p.getLinkState(self.sim.robot.id, link_idx)
            link_position = np.array(link_state[0])  # Extract the position (x, y, z)

            # Compute repulsive forces for each obstacle
            for obstacle_position in obstacles:
                obstacle_position = np.array(obstacle_position)
                distance_vector = link_position - obstacle_position
                distance = np.linalg.norm(distance_vector)

                if distance < self.repulsive_threshold and distance > 0:
                    # Compute the repulsive force
                    repulsive_force = self.repulsive_gain * (1 / distance - 1 / self.repulsive_threshold) / (distance**2)
                    repulsive_force_vector = repulsive_force * (distance_vector / distance)

                    # Map the Cartesian repulsive force to joint space using the Jacobian
                    jacobian = p.calculateJacobian(
                        self.sim.robot.id, 
                        link_idx, 
                        [0, 0, 0], 
                        current_joint_angles.tolist() + gripper_positions.tolist(), 
                        [0]*len(current_joint_angles.tolist() + gripper_positions.tolist()), 
                        [0]*len(current_joint_angles.tolist() + gripper_positions.tolist()))[0]
                    jacobian = np.array(jacobian)
                    joint_repulsive_force = jacobian.T @ repulsive_force_vector
                    total_force += joint_repulsive_force[:7]

        return total_force

    def compute_total_force(self, current_joint_angles, goal_joint_angles, obstacles):
        """
        Compute the total force acting on the robot in joint space.

        Args:
            current_joint_angles: The current joint angles as a numpy array (n,).

        Returns:
            The total force as a numpy array (n,).
        """
        attractive_force = self.compute_attractive_force(current_joint_angles, goal_joint_angles)
        repulsive_force = self.compute_repulsive_force(current_joint_angles, obstacles)
        total_force = attractive_force + repulsive_force
        return total_force