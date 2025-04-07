import numpy as np
import open3d as o3d
import pybullet as p
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

class PickAndPlace:
    """
    A class to perform pick-and-place operations using the robot.

    Attributes:
        robot: An instance of the Robot class.
    """

    def __init__(self, sim, controller):
        """
        Initialize the PickAndPlace class.

        Args:
            robot: An instance of the Robot class.
        """
        self.sim = sim
        self.controller = controller

    def pick(self, object_position: np.ndarray, orientation: np.ndarray = None, obstacles: bool = False):
        """
        Perform a pick operation at the specified object position.

        Args:
            object_position: The position of the object to pick as a numpy array [x, y, z].
        """
        # Retrieve gripper limits
        lower_limits, upper_limits = self.sim.robot.get_gripper_limits()

        # Open the gripper
        self.sim.robot.gripper_control(upper_limits)

        for _ in range(10):
            self.sim.step()

        # Move to a position above the object
        pre_grasp_position = object_position + np.array([0, 0, 0.1])

        if obstacles:
            self.controller.move_with_obstacles(pre_grasp_position, orientation)
        else:
            self.controller.move_without_obstacles(pre_grasp_position, orientation)

        # Estimate of grasp position tends to be too high
        # Lower grasp position but make sure to stay above the table
        grasp_position = object_position
        grasp_position[2] = np.max([1.24 + 0.01, grasp_position[2] - 0.04])

        # Move to the object
        if obstacles:
            self.controller.move_with_obstacles(grasp_position, orientation)
        else:
            self.controller.move_without_obstacles(grasp_position, orientation)

        # Close the gripper to grasp the object
        self.sim.robot.gripper_control(lower_limits)

        for _ in range(10):
            self.sim.step()

        # Move back up with the object
        if obstacles:
            self.controller.move_with_obstacles(pre_grasp_position, orientation)
        else:
            self.controller.move_without_obstacles(pre_grasp_position, orientation)

    def place(self, goal_position: np.ndarray, obstacles: bool = False):
        """
        Perform a place operation at the specified goal position.

        Args:
            goal_position: The position to place the object as a numpy array [x, y, z].
        """
        # Move above the goal position
        pre_place_position = goal_position + np.array([0, 0, 0.2])  # Offset above the goal
        if obstacles:
            self.controller.move_with_obstacles(pre_place_position)
        else:
            self.controller.move_without_obstacles(pre_place_position)

        # Retrieve gripper limits
        _, upper_limits = self.sim.robot.get_gripper_limits()

        # Open the gripper to release the object
        self.sim.robot.gripper_control(upper_limits)

        for _ in range(50):
            if obstacles:
                # Check for obstacles and adjust the robot's position if necessary
                current_position, current_orientation = self.sim.robot.get_ee_pose()
                self.controller.move_with_obstacles(current_position, current_orientation)
            self.sim.step()

    def pick_and_place(self, gripper_position: np.ndarray, grasp_orientation: np.ndarray, goal_position: np.ndarray, obstacles: bool = False):
        """
        Execute a pick-and-place operation.

        Args:
            gripper_position: The position of the gripper to pick the object as a numpy array [x, y, z].
            grasp_orientation: The orientation of the gripper as a quaternion [x, y, z, w].
            goal_position: The position to place the object as a numpy array [x, y, z].
            obstacles: Whether to consider obstacles during the operation.
        """
        print("Starting pick-and-place operation...")
        self.pick(gripper_position, grasp_orientation, obstacles)
        self.place(goal_position, obstacles)


class GraspPlanner:
    def __init__(self):
        """ Initializes the GraspPlanner. """
        self.pointcloud = None

    def set_pcd(self, pcd):
        """ Sets the point cloud. """
        self.pointcloud = pcd

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
        x_axis = axis1 / np.linalg.norm(axis1)  
        y_axis = axis2 / np.linalg.norm(axis2)  
        z_axis = np.cross(x_axis, y_axis)  # Compute orthogonal z-axis
        z_axis /= np.linalg.norm(z_axis)
        # Ensure the z-axis is pointing downwards
        if z_axis[2] > 0:
            z_axis = -z_axis
            x_axis = -x_axis

        # Ensure the gripper is oriented correctly
        if y_axis[0] > 0:
            x_axis = -x_axis
            y_axis = -y_axis

        # Check if the z_axis is within 20 degrees of the world z-axis
        world_z_axis = np.array([0, 0, -1])
        dot_product = np.dot(z_axis, world_z_axis)
        angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))  # Clip to handle numerical precision issues
        
        # If the angle is greater than 20 degrees, no valid grasp is found
        if angle > 20:
            print("No valid grasp found.")
            return gripper_center, [0, 1, 0, 0]  # Return a default quaternion
        
        # Draw the axes for visualization
        p.addUserDebugLine(gripper_center, gripper_center + x_axis*0.1, [1, 0, 0], 3)
        p.addUserDebugLine(gripper_center, gripper_center + y_axis*0.1, [0, 1, 0], 3)
        p.addUserDebugLine(gripper_center, gripper_center + z_axis*0.1, [0, 0, 1], 3)

        # Create a rotation matrix from the axes
        rotation = R.from_matrix(np.column_stack((x_axis, y_axis, z_axis)))
        quaternion = rotation.as_quat()

        return gripper_center, quaternion
