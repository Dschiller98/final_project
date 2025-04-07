
import numpy as np
import open3d as o3d
import pybullet as p


class PoseEstimator:
    def __init__(self, simulation, controller):
        """
        Initialize the Pose Estimator.

        Args:
            simulation: Instance of the Simulation class.
        """
        self.sim = simulation
        self.controller = controller

    def get_camera_data(self, camera_type="static"):
        """
        Capture RGB, depth, and segmentation images from the specified camera.

        Args:
            camera_type: Type of camera ("static" or "ee").

        Returns:
            Tuple of (RGB image, depth image, segmentation image).
        """
        if camera_type == "static":
            return self.sim.get_static_renders()
        elif camera_type == "ee":
            return self.sim.get_ee_renders()
        else:
            raise ValueError("Invalid camera type. Use 'static' or 'ee'.")

    def estimate_object_pose(self, object_id, camera_type="static"):
        """
        Estimate the position and pointcloud of the object using the specified camera.

        Args:
            object_id: ID of the target object.
            camera_type: Type of camera ("static" or "ee").

        Returns:
            Position and pointcloud of the object.
        """
        rgb, depth, seg = self.get_camera_data(camera_type)
        mask = seg == object_id

        # Check if the object is visible in the camera view
        if mask.sum() < 6:
            raise ValueError(f"No object found in the {camera_type} camera image.")

        # Extract 3D points from the depth image
        points = self.depth_to_point_cloud(depth, mask, camera_type)

        # Compute the position as the centroid of the points
        position = np.mean(points, axis=0)
        return position, points

    def depth_to_point_cloud(self, depth_image, mask, camera_type):
        """
        Convert a depth image to a 3D point cloud (in world coordinates).

        Args:
            depth_image: Depth image.
            mask: Binary mask of the segmented object.

        Returns:
            3D point cloud of the object.
        """

        # Mask depth image
        
        near = 0.01
        far = 5
        fov = 70
        depth_image = far * near / (far - (far - near) * depth_image)
        depth_image = np.where(mask, depth_image, 0)

        # Intrinsic camera parameters
        f = self.sim.height / (2 * np.tan(np.radians(fov / 2)))
        cx, cy = self.sim.width / 2, self.sim.height / 2  # Principal point

        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=self.sim.width, height=self.sim.height, fx=f, fy=f, cx=cx, cy=cy)

        # Extrinsic camera parameters

        Tc = np.array([[1,  0,  0,  0],
                   [0,  -1,  0,  0],
                   [0,  0,  -1,  0],
                   [0,  0,  0,  1]]).reshape(4,4)
        
        if camera_type == "static":
            view_matrix = self.sim.stat_viewMat
        elif camera_type == "ee":
            """Get end-effector camera viewmatrix. y-axis is up."""
            ee_pos, ee_rot = self.sim.robot.get_ee_pose()
            rot_matrix = p.getMatrixFromQuaternion(ee_rot)
            rot_matrix = np.array(rot_matrix).reshape(3, 3)
            init_camera_vector = (0, 0, 1)  # z-axis
            init_up_vector = (0, 1, 0)  # y-axis
            # Rotated vectors
            camera_vector = rot_matrix.dot(init_camera_vector)
            up_vector = rot_matrix.dot(init_up_vector)

            view_matrix = p.computeViewMatrix(ee_pos, ee_pos + 0.1 * camera_vector, up_vector)
        
        extrinsic = np.linalg.inv(np.array(view_matrix).reshape((4,4),order="F")) @ Tc

        # Convert depth image to point cloud

        pointcloud = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_image), intrinsic)

        points = np.asarray(pointcloud.points)
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
        points = np.dot(extrinsic, points.T).T # transform to world coordinates
        points = points[:, :3]

        return points
    
    def scan_table(self, object_id, obstacles=False):
        """
        Scan the table using the end-effector camera to estimate the object position and pointcloud.

        Args:
            object_id: ID of the target object.

        Returns:
            Position and pointcloud of the object.
        """
        camera_pos = [0.1, -0.5, 1.9]
        if obstacles:
            self.controller.move_with_obstacles(camera_pos)
        else:
            self.controller.move_without_obstacles(camera_pos)
        position, pcd = self.estimate_object_pose(object_id, "ee")
        return position, pcd
    
    def estimate_object_pose_all_cameras(self, object_id, obstacles=False):
        """
        Estimate the position and pointcloud of the object using both static and end-effector cameras.

        Args:
            object_id: ID of the target object.

        Returns:
            Position and pointcloud of the object.
        """
        # Try to estimate from static camera first
        # If the object is hidden by the robot, use end-effector camera
        try:
            position, pcd = self.estimate_object_pose(object_id, "static")
        except Exception as e:
            print(f"Error in estimating position from static camera: {e}")
            position, pcd = self.scan_table(object_id, obstacles)

        camera_pos = position + np.array([0, 0, 0.2]) # Offset above the object
        if obstacles:
            self.controller.move_with_obstacles(camera_pos)
        else:
            self.controller.move_without_obstacles(camera_pos)
        # Estimate from close up using end-effector camera
        position, pcd = self.estimate_object_pose(object_id, "ee")

        return position, pcd