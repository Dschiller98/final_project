import numpy as np
import open3d as o3d
from .ik import move_to_goal

class PickAndPlace:
    """
    A class to perform pick-and-place operations using the robot.

    Attributes:
        robot: An instance of the Robot class.
    """

    def __init__(self, robot, sim):
        """
        Initialize the PickAndPlace class.

        Args:
            robot: An instance of the Robot class.
        """
        self.robot = robot
        self.sim = sim

    def pick(self, object_position: np.ndarray):
        """
        Perform a pick operation at the specified object position.

        Args:
            object_position: The position of the object to pick as a numpy array [x, y, z].
        """
        # Move above the object
        pre_grasp_position = object_position + np.array([0, 0, 0.1])  # Offset above the object
        move_to_goal(self.robot, pre_grasp_position, np.eye(3))

        # Move to the object
        move_to_goal(self.robot, object_position, np.eye(3))

        # Retrieve gripper limits
        lower_limits, _ = self.robot.get_gripper_limits()

        # Close the gripper to grasp the object
        self.robot.gripper_control(lower_limits)

        # Move back up with the object
        move_to_goal(self.robot, pre_grasp_position, np.eye(3))

    def place(self, goal_position: np.ndarray):
        """
        Perform a place operation at the specified goal position.

        Args:
            goal_position: The position to place the object as a numpy array [x, y, z].
        """
        # Move above the goal position
        pre_place_position = goal_position + np.array([0, 0, 0.1])  # Offset above the goal
        move_to_goal(self.robot, pre_place_position, np.eye(3))

        # Move to the goal position
        move_to_goal(self.robot, goal_position, np.eye(3))

        # Retrieve gripper limits
        _, upper_limits = self.robot.get_gripper_limits()

        # Open the gripper to release the object
        self.robot.gripper_control(upper_limits)

        # Move back up after placing the object
        move_to_goal(self.robot, pre_place_position, np.eye(3))

    def pick_and_place(self, object_position: np.ndarray, goal_position: np.ndarray):
        """
        Execute a pick-and-place operation.

        Args:
            object_position: The position of the object to pick as a numpy array [x, y, z].
            goal_position: The position to place the object as a numpy array [x, y, z].
        """
        print("Starting pick-and-place operation...")
        self.pick(object_position)
        print("Object picked successfully.")
        self.place(goal_position)
        print("Object placed successfully.")

