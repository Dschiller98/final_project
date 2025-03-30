import numpy as np
import pybullet as p
from filterpy.kalman import KalmanFilter
from collections import deque
from scipy.optimize import least_squares


class ObstacleTracker:
    """
    A class to track obstacles using a camera and a Kalman filter.

    Attributes:
        camera_intrinsics: Intrinsic parameters of the camera.
        kalman_filters: Dictionary of Kalman filters for each obstacle.
    """

    def __init__(self, pose_estimator):
        """
        Initialize the ObstacleTracker.

        Args:
            pose_estimator: An instance of the PoseEstimator class.
        """
        self.pose_estimator = pose_estimator
        self.kalman_filters = {}  # Dictionary to store Kalman filters for each obstacle
        self.debug_ids = deque()


    def initialize_kalman_filter(self, initial_position):
        """
        Initialize a Kalman filter for an obstacle.

        Args:
            initial_position: Initial 3D position of the obstacle.

        Returns:
            A KalmanFilter object.
        """
        # Initialize Kalman filter: 6 states (x, y, z, vx, vy, vz), 3 measurements (x, y, z)
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.x = np.array([*initial_position, 0, 0, 0])  # Initial state [x, y, z, vx, vy, vz]
        kf.F = np.eye(6)  # State transition matrix
        kf.F[:3, 3:] = np.eye(3) * 0.1  # Incorporate velocity into position
        kf.H = np.eye(3, 6)  # Measurement function
        kf.P *= 1000.0  # Initial uncertainty
        kf.R = np.eye(3) * 0.1  # Measurement noise
        kf.Q = np.eye(6) * 0.1  # Process noise
        return kf

    def update(self, obstacle_positions):
        """
        Update the tracker with new obstacle detections.

        Args:
            obstacle_positions: List of estimated obstacle positions in 3D space.
        """

        # Update Kalman filters for each obstacle
        for i, position in enumerate(obstacle_positions):
            if i not in self.kalman_filters:
                # Initialize a new Kalman filter for a new obstacle
                self.kalman_filters[i] = self.initialize_kalman_filter(position)
            else:
                # Update the Kalman filter with the new measurement
                self.kalman_filters[i].update(position)

        # Predict the next positions of all obstacles
        for kf in self.kalman_filters.values():
            kf.predict()
    
    def estimate_obstacle_parameters(self, object_ids=[6,7]):
        """
        Estimate the center and radius of all tracked obstacles.

        Returns:
            A list of obstacle radii.
        """
        def sphere_residuals(params, points):
            x0, y0, z0, r = params
            residuals = np.sum((points - np.array([x0, y0, z0]))**2, axis=1) - r**2
            return residuals
        
        radii = []
        centers =[]
        for i in range(len(object_ids)):
            pcd = self.pose_estimator.estimate_pcd_from_static(object_ids[i])
            x0_init = np.mean(pcd[:, 0])
            y0_init = np.mean(pcd[:, 1])
            z0_init = np.mean(pcd[:, 2])
            r_init = np.mean(np.linalg.norm(pcd - np.array([x0_init, y0_init, z0_init]), axis=1))
            initial_guess = [x0_init, y0_init, z0_init, r_init]
            # Use least squares to fit the sphere
            result = least_squares(sphere_residuals, initial_guess, args=(pcd,))
            x0, y0, z0, r = result.x
            radii.append(r)
            centers.append([x0, y0, z0])
        return centers, radii

    def get_obstacle_states(self):
        """
        Get the current states of all tracked obstacles.

        Returns:
            A dictionary of obstacle states, where each state contains:
                - position: The 3D position of the obstacle.
                - velocity: The 3D velocity of the obstacle.
        """
        states = {}
        for i, kf in self.kalman_filters.items():
            position = kf.x[:3]
            velocity = kf.x[3:]
            states[i] = {"position": position, "velocity": velocity}
        return states
    
    def draw_bounding_boxes(self, center, radius):

            # Compute the 8 corners of the bounding box
            offsets = np.array([
                [-radius, -radius, -radius],
                [-radius, -radius, radius],
                [-radius, radius, -radius],
                [-radius, radius, radius],
                [radius, -radius, -radius],
                [radius, -radius, radius],
                [radius, radius, -radius],
                [radius, radius, radius],
            ])
            corners = center + offsets

            # Define the edges of the bounding box (pairs of corner indices)
            edges = [
                (0, 1), (0, 2), (0, 4),  # Edges from corner 0
                (1, 3), (1, 5),          # Edges from corner 1
                (2, 3), (2, 6),          # Edges from corner 2
                (3, 7),                  # Edges from corner 3
                (4, 5), (4, 6),          # Edges from corner 4
                (5, 7),                  # Edges from corner 5
                (6, 7),                  # Edges from corner 6
            ]

            # Draw the edges of the bounding box
            for edge in edges:
                start, end = corners[edge[0]], corners[edge[1]]
                # check if debug_ids is empty:
                if len(self.debug_ids) < 24:
                    id = p.addUserDebugLine(start.tolist(), end.tolist(), [0,0,1], lineWidth=1)
                    self.debug_ids.append(id)
                else:
                    id = p.addUserDebugLine(start.tolist(), end.tolist(), [0,0,1], lineWidth=1, replaceItemUniqueId=self.debug_ids.popleft())
                    self.debug_ids.append(id)
    
    def track_obstacles(self, object_ids=[6,7]):
        positions, _ = self.estimate_obstacle_parameters(object_ids)
        self.update(positions)
        return self.get_obstacle_states()


