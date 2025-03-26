import numpy as np
import open3d as o3d
import pybullet as p
import os
from pybullet_object_models import ycb_objects
#from urdfpy import URDF
import os


def get_point_cloud(depth, view_matrix, proj_matrix):
    # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer

    height, width = depth.shape

    # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

    # create a grid with pixel coordinates and depth values
    y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)

    pixels = np.stack([x, y, z, h], axis=1)
    # filter out "infinite" depths
    #pixels = pixels[z < 0.99]
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]

    return points


def segment_point_cloud(point_cloud, seg, obj_id):
    """
    Segment the point cloud using the segmentation mask.

    Args:
        point_cloud: Point cloud as numpy array [N, 3]
        seg: Segmentation map as numpy array [height, width]
        obj_id: ID of the object to segment

    Returns:
        Segmented point cloud as numpy array [M, 3]
    """
    mask = seg.flatten() == obj_id

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[mask])
    return pcd


def preprocess_point_cloud(pcd, voxel_size):
    """
    Preprocess the point cloud by downsampling and estimating normals.
    compute FPFH features. nearest neighbour query in fpfh space.
    Args:
        pcd: Point cloud as open3d.geometry.PointCloud
        voxel_size: Voxel size for downsampling
    Returns:
        Downsampled point cloud and FPFH features
    """
    #pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, pcd_fpfh


def fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    """
    Execute global registration.
    Args:
        source_down: Source point cloud after downsampling
        target_down: Target point cloud after downsampling
        source_fpfh: Source FPFH features
        target_fpfh: Target FPFH features
        voxel_size: Voxel size for downsampling
    Returns:
        Registration result
    """
    distance_threshold = voxel_size * 0.5
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def load_ycb_object_point_cloud(obj_name, scaling=1.5):
    """
    Load the YCB object point cloud using PyBullet.

    Args:
        obj_name: Name of the YCB object
        scaling: Scaling factor for the object

    Returns:
        Point cloud as open3d.geometry.PointCloud
    """
    object_root_path = ycb_objects.getDataPath()
    obj_path = os.path.join(object_root_path, obj_name, "model.urdf")
    obj_id = p.loadURDF(obj_path, globalScaling=scaling)
    
    vertices = np.array(p.getMeshData(obj_id)[1])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    
    # Remove the object from the simulation
    p.removeBody(obj_id)
    
    return pcd

def run_icp(rgb, depth, seg, projection_matrix, view_matrix, obj_id, obj_name, voxel_size=0.05):
    """
    Run ICP to estimate the transformation between the source and target point clouds.
    Args:
        rgb: RGB image
        depth: Depth image
        seg: Segmentation map
        intrinsic_matrix: Camera intrinsic matrix
        target_pcd: Target point cloud as open3d.geometry.PointCloud
        voxel_size: Voxel size for downsampling"
    """
    source_pcd = get_point_cloud(depth, view_matrix, projection_matrix)
    source_obj_pcd = segment_point_cloud(source_pcd, seg, obj_id)
    source_pcd_down, source_pcd_fpfh = preprocess_point_cloud(source_obj_pcd, voxel_size)

    target_obj_pcd = load_ycb_object_point_cloud(obj_name)
    target_pcd_down, target_pcd_fpfh = preprocess_point_cloud(target_obj_pcd, voxel_size)

    result = fast_global_registration(source_pcd_down, target_pcd_down, source_pcd_fpfh, target_pcd_fpfh, voxel_size)

    return result.transformation