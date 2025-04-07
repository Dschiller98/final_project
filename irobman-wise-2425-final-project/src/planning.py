
import pybullet as p
import numpy as np
from pybullet_planning import plan_joint_motion, set_joint_positions, get_collision_fn, get_joint_positions

class GlobalPlanner:
    """
    A class to perform global motion planning using BiRRT from the pybullet_planning library.
    """

    def __init__(self, robot, start, goal, obstacles):
        """
        Initialize the GlobalPlanner.

        Args:
            robot: The robot object.
            start: List of joint positions representing the start configuration.
            goal: List of joint positions representing the goal configuration.
            obstacles: List of obstacle body IDs in the PyBullet simulation.
        """
        self.robot = robot
        self.start = start
        self.goal = goal
        self.obstacles = obstacles

    def plan(self):
        """
        Plan a collision-free path from start to goal using BiRRT.

        Returns:
            A list of joint configurations representing the planned path, or None if no path is found.
        """
        # Get the robot's movable joints
        movable_joints = list(range(p.getNumJoints(self.robot)))

        # Define the collision function
        collision_fn = get_collision_fn(
            self.robot,
            movable_joints,
            obstacles=self.obstacles,
            attachments=[],
            self_collisions=True
        )

        # Plan the motion using BiRRT
        path = plan_joint_motion(
            self.robot,
            movable_joints,
            self.goal,
            start_conf=self.start,
            custom_limits=None,
            collision_fn=collision_fn,
            algorithm='birrt'
        )

        return path

    def execute(self, path, time_step=0.01):
        """
        Execute the planned path in the PyBullet simulation.

        Args:
            path: A list of joint configurations representing the planned path.
            time_step: Time step for executing each configuration.
        """
        if path is None:
            print("No valid path found.")
            return

        for joint_positions in path:
            set_joint_positions(self.robot, list(range(p.getNumJoints(self.robot))), joint_positions)
            p.stepSimulation()
            p.sleep(time_step)


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