import numpy as np
import open3d as o3d
import pybullet as p
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
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

    def pick(self, object_position: np.ndarray, orientation: np.ndarray = None):
        """
        Perform a pick operation at the specified object position.

        Args:
            object_position: The position of the object to pick as a numpy array [x, y, z].
        """
        # Retrieve gripper limits
        lower_limits, upper_limits = self.robot.get_gripper_limits()

        # Open the gripper
        self.robot.gripper_control(upper_limits)

        # Estimate of grasp position tends to be too high
        grasp_position = object_position + np.array([0, 0, -0.01])
        #move_to_goal(self.robot, pre_grasp_position, orientation)

        # Move to the object
        move_to_goal(self.robot, grasp_position, orientation)

        # Close the gripper to grasp the object
        self.robot.gripper_control(lower_limits)

        # Move to a position above the object
        post_grasp_position = object_position + np.array([0, 0, 0.1])

        # Move back up with the object
        move_to_goal(self.robot, post_grasp_position, orientation)

    def place(self, goal_position: np.ndarray):
        """
        Perform a place operation at the specified goal position.

        Args:
            goal_position: The position to place the object as a numpy array [x, y, z].
        """
        # Move above the goal position
        pre_place_position = goal_position + np.array([0, 0, 0.2])  # Offset above the goal
        move_to_goal(self.robot, pre_place_position)

        # Move to the goal position
        #move_to_goal(self.robot, goal_position)

        # Retrieve gripper limits
        _, upper_limits = self.robot.get_gripper_limits()

        # Open the gripper to release the object
        self.robot.gripper_control(upper_limits)

        # Move back up after placing the object
        #move_to_goal(self.robot, pre_place_position)

    def pick_and_place(self, gripper_position: np.ndarray, grasp_orientation: np.ndarray, goal_position: np.ndarray):
        """
        Execute a pick-and-place operation.

        Args:
            object_position: The position of the object to pick as a numpy array [x, y, z].
            goal_position: The position to place the object as a numpy array [x, y, z].
        """
        print("Starting pick-and-place operation...")
        self.pick(gripper_position, grasp_orientation)
        print("Object picked successfully.")
        self.place(goal_position)
        print("Object placed successfully.")


class GraspPlanner:
    def __init__(self, pointcloud):
        # Convert the Open3D point cloud to a NumPy array
        self.pointcloud = pointcloud

    def compute_convex_hull(self):
        """ Computes the convex hull of the point cloud. """
        hull = ConvexHull(self.pointcloud)
        return hull

    def find_best_grasp(self):
        """ Finds the best grasp configuration for a two-fingered gripper.
        Returns the gripper center position and a rotation matrix.
        """
        hull = self.compute_convex_hull()
        points = self.pointcloud[hull.vertices]
        
        # Compute the centroid of the convex hull
        centroid = np.mean(points, axis=0)
        
        # Compute the covariance matrix and perform eigen decomposition
        cov_matrix = np.cov(points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # The principal axes are given by the eigenvectors corresponding to largest eigenvalues
        idx = np.argsort(eigenvalues)[::-1]  # Sort indices by eigenvalues
        axis1 = eigenvectors[:, idx[0]]  # First principal axis
        axis2 = eigenvectors[:, idx[1]]  # Second principal axis

        # Compute the gripper center position
        gripper_center = centroid  
        
        # Normalize axes
        y_axis = axis1 / np.linalg.norm(axis1)  
        x_axis = axis2 / np.linalg.norm(axis2)  
        z_axis = np.cross(x_axis, y_axis)  # Compute orthogonal z-axis
        z_axis /= np.linalg.norm(z_axis)
        if z_axis[2] > 0:
            z_axis = -z_axis
            x_axis = -x_axis

        rotation = R.from_matrix(np.column_stack((x_axis, y_axis, z_axis)))
        quaternion = rotation.as_quat()  # Convert to quaternion

        print(f"Quaternion: {quaternion}")

        return gripper_center, quaternion
