
import numpy as np
import open3d as o3d
from .ik import move_to_goal
import pybullet as p


class PoseEstimator:
    def __init__(self, simulation):
        """
        Initialize the Pose Estimator.

        Args:
            simulation: Instance of the Simulation class.
        """
        self.sim = simulation

    def get_static_camera_data(self):
        """
        Capture RGB, depth, and segmentation images from the static camera.
        """
        rgb, depth, seg = self.sim.get_static_renders()
        return rgb, depth, seg

    def get_ee_camera_data(self):
        """
        Capture RGB, depth, and segmentation images from the end-effector camera.
        """
        rgb, depth, seg = self.sim.get_ee_renders()
        return rgb, depth, seg

    def segment_object(self, seg_image, object_id):
        """
        Segment the object using the segmentation image.

        Args:
            seg_image: Segmentation image from PyBullet.
            object_id: ID of the target object.

        Returns:
            Mask of the segmented object.
        """
        return seg_image == object_id

    def estimate_position_from_depth(self, depth_image, mask, camera_type):
        """
        Estimate the position of the object using the depth image from a camera.

        Args:
            depth_image: Depth image from the camera.
            mask: Binary mask of the segmented object.

        Returns:
            Position of the object center.
        """
        # Extract 3D points from the depth image
        points = self.depth_to_point_cloud(depth_image, mask, camera_type)

        # Compute the position as the centroid of the points
        position = np.mean(points, axis=0)
        return position

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

        # TODO hier noch richtig machen später
        #self.pcd = points
        return points

    
    def estimate_position_from_static(self, object_id):
        """
        Position estimation using the static camera.

        Args:
            object_id: ID of the target object.

        Returns:
            Position of the object center.
        """
        rgb_static, depth_static, seg_static = self.get_static_camera_data()
        mask_static = self.segment_object(seg_static, object_id)
        if mask_static.sum() < 6:
            raise ValueError("No object found in the static camera image.")
        position = self.estimate_position_from_depth(depth_static, mask_static, camera_type="static")

        return position
    

    def estimate_position_from_ee(self, object_id):
        """
        Position estimation using the end-effector camera.

        Args:
            object_id: ID of the target object.

        Returns:
            Position of the object center.
        """
        rgb_ee, depth_ee, seg_ee = self.get_ee_camera_data()
        mask_ee = self.segment_object(seg_ee, object_id)
        position = self.estimate_position_from_depth(depth_ee, mask_ee, camera_type="ee")

        return position
    
    def estimate_pcd_from_ee(self, object_id):
        """
        Estimate the point cloud of the object using the end-effector camera.

        Args:
            object_id: ID of the target object.

        Returns:
            Point cloud of the object.
        """
        rgb_ee, depth_ee, seg_ee = self.get_ee_camera_data()
        mask_ee = self.segment_object(seg_ee, object_id)
        pcd = self.depth_to_point_cloud(depth_ee, mask_ee, camera_type="ee")
        return pcd
    
    def scan_table(self, object_id):
        """
        Scan the table using the end-effector camera to estimate the object position.

        Args:
            object_id: ID of the target object.

        Returns:
            Position of the object center.
        """
        camera_pos = [0.1, -0.5, 1.9]
        move_to_goal(self.sim.robot, camera_pos)
        position = self.estimate_position_from_ee(object_id)
        return position
    
    def estimate_object_position(self, object_id):
        """
        Estimate the position of the object using both static and end-effector cameras.

        Args:
            object_id: ID of the target object.

        Returns:
            Position of the object center.
        """
        try:
            position = self.estimate_position_from_static(object_id)
        except Exception as e:
            print(f"Error in estimating position from static camera: {e}")
            position = self.scan_table(object_id)

        return position
    
    def estimate_object_pcd(self, object_id):

        position = self.estimate_object_position(object_id)
        camera_pos = position + np.array([0, 0, 0.2]) # Offset above the object
        move_to_goal(self.sim.robot, camera_pos)
        pcd = self.estimate_pcd_from_ee(object_id)

        return pcd