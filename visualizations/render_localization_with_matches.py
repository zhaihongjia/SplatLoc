import argparse
import os
import pickle
import glob 
from time import sleep, time

import natsort
import numpy as np
import open3d as o3d

import cv2

K_12scenes = np.array([[572, 0, 320],
                       [0, 572, 240],
                       [0,   0,   1]])

K_replica_nerf = np.array([[320, 0, 319.5],
                           [0, 320, 239.5],
                           [0,   0,     1]])

class Frustum:
    def __init__(self, line_set, view_dir=None, view_dir_behind=None, size=None):
        self.line_set = line_set
        self.view_dir = view_dir
        self.view_dir_behind = view_dir_behind
        self.size = size

    def update_pose(self, pose):
        points = np.asarray(self.line_set.points)
        points_hmg = np.hstack([points, np.ones((points.shape[0], 1))])
        points = (pose @ points_hmg.transpose())[0:3, :].transpose()

        base = np.array([[0.0, 0.0, 0.0]]) * self.size
        base_hmg = np.hstack([base, np.ones((base.shape[0], 1))])
        cameraeye = pose @ base_hmg.transpose()
        cameraeye = cameraeye[0:3, :].transpose()
        eye = cameraeye[0, :]

        base_behind = np.array([[0.0, -2.5, -30.0]]) * self.size
        base_behind_hmg = np.hstack([base_behind, np.ones((base_behind.shape[0], 1))])
        cameraeye_behind = pose @ base_behind_hmg.transpose()
        cameraeye_behind = cameraeye_behind[0:3, :].transpose()
        eye_behind = cameraeye_behind[0, :]

        center = np.mean(points[1:, :], axis=0)
        up = points[2] - points[4]

        self.view_dir = (center, eye, up, pose)
        self.view_dir_behind = (center, eye_behind, up, pose)

        self.center = center
        self.eye = eye
        self.up = up


def create_frustum(pose, frusutum_color=[0, 1, 0], size=0.02):
    points = (
        np.array(
            [
                [0.0, 0.0, 0],   # center
                [1.0, -0.5, 2],  # right-top
                [-1.0, -0.5, 2], # left-top
                [1.0, 0.5, 2],   # right-bottom
                [-1.0, 0.5, 2],  # left-bottom
            ]
        )
        * size
    )

    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
    colors = [frusutum_color for i in range(len(lines))]

    canonical_line_set = o3d.geometry.LineSet()
    canonical_line_set.points = o3d.utility.Vector3dVector(points)
    canonical_line_set.lines = o3d.utility.Vector2iVector(lines)
    canonical_line_set.colors = o3d.utility.Vector3dVector(colors)
    frustum = Frustum(canonical_line_set, size=size)
    frustum.update_pose(pose)
    return frustum


# Create a PinholeCameraParameters object to represent the camera
def create_camera_model(intrinsics, pose, image_size, c=[0,0,1], scale=1.0):
    """
    Create a camera model using Open3D
    :param intrinsics: Camera intrinsic matrix (3x3)
    :param pose: Camera pose as a 4x4 matrix
    :param image_size: Tuple of (width, height)
    :param scale: Scaling factor for the camera model
    :return: Open3D geometry representing the camera
    """
    # Create a frustum based on the camera intrinsics and image size
    width, height = image_size
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Define the frustum's near and far planes (scaled for visibility)
    near = 0.1 * scale
    far = 1.0 * scale

    # Calculate the frustum's corner points in the camera coordinate system
    corners = np.array([
        [(0 - cx) * far / fx, (0 - cy) * far / fy, far],
        [(width - cx) * far / fx, (0 - cy) * far / fy, far],
        [(width - cx) * far / fx, (height - cy) * far / fy, far],
        [(0 - cx) * far / fx, (height - cy) * far / fy, far],
        [0.0, 0.0, 0.0],
    ])

    # Transform the corners to the world coordinate system
    corners_world = (pose[:3, :3] @ corners.T).T + pose[:3, 3]

    # Create line pairs to represent the frustum
    cam_lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Far plane
        [0, 4], [1, 4], [2, 4], [3, 4]   # plane to center
    ]

    # create PC for visualization
    points = []
    for cam_line in cam_lines:
        begin_points, end_points = corners[cam_line[0]], corners[cam_line[1]]
        t_vals = np.linspace(0., 1., 100)
        point = begin_points[None, :] * (1. - t_vals)[:, None] + end_points[None, :] * (t_vals)[:, None]
        points.append(point)
    points = np.concatenate(points)
    points = points @ pose[:3,:3].transpose() + pose[:3, 3]

    camera_actor = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
    camera_actor.paint_uniform_color(c)

    return camera_actor


def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

CAM_POINTS = np.array(
    [
        [0, 0, 0],
        [-1, -1, 1.5],
        [1, -1, 1.5],
        [1, 1, 1.5],
        [-1, 1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5],
    ])

CAM_LINES = np.array([[1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]])

# visualize image on camera plane
def visualize_image_on_camera(image, intrinsics, pose, scale=1.0):
    # Convert the image to an Open3D Image object
    o3d_image = o3d.geometry.Image(image)

    # Create a simple 3D plane to display the image in 3D space
    near = 0.1 * scale
    far = 1.0 * scale
    height = image.shape[0]
    width = image.shape[1]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Define the vertices of the rectangle
    vertices = np.array([
            [-width*far/fx / 2, -height*far/fy / 2, 0],
            [ width*far/fx / 2, -height*far/fy / 2, 0],
            [ width*far/fx / 2,  height*far/fy / 2, 0],
            [-width*far/fx / 2,  height*far/fy / 2, 0]
            ], dtype=np.float32)

    faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            ])
    v_uv = np.array([[0, 1], [1, 1], [1, 0], [0, 1], [1, 0], [0, 0]])

    # Create the mesh from vertices and triangles
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    mesh.textures = [o3d_image]
    mesh.triangle_uvs = o3d.utility.Vector2dVector(v_uv)
    mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(faces))
    mesh.translate((0, 0, far))

    # Apply the camera pose to the plane
    mesh.transform(pose)
    return mesh

def project_2d_to_3d(image_points, K, c2w, depth=1.0, scale=1.0):
    """
    project 2D pixels to 3D spave
    :param image_points: 2d image pixels (N, 2)
    :param K: camera intrisic (3, 3)
    :param c2w: camera pose (4, 4)
    :param depth: depth value for image plane
    :return: 3D points (N, 3)
    """
    depth = depth * scale 
    K_inv = np.linalg.inv(K)
    
    ones = np.ones((image_points.shape[0], 1))
    image_points_hom = np.hstack((image_points, ones))
    
    points_3d_camera = (K_inv @ image_points_hom.T).T * depth
    
    points_3d_camera_hom = np.hstack((points_3d_camera, ones))
    points_3d_world = (c2w @ points_3d_camera_hom.T).T[:, :3]
    
    return points_3d_world

def visualize_match(kp_2d, pt_3d, K, pose, c=[0,1,0], scale=1.0):
    match_lines = []
    for i in range(kp_2d.shape[0]):
        match_lines.append([i, kp_2d.shape[0] + i])
    colors = [[0, 1, 0] for i in range(len(match_lines))]  

    # project image to D z=1 palne
    image_points_3d = project_2d_to_3d(kp_2d, K, pose, scale)
    all_points = np.vstack((pt_3d, image_points_3d))

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(all_points)
    lines.lines = o3d.utility.Vector2iVector(match_lines)
    lines.colors = o3d.utility.Vector3dVector(colors)

    # create PC for visualization
    points = []
    for match_line in match_lines:
        begin_points, end_points = all_points[match_line[0]], all_points[match_line[1]]
        t_vals = np.linspace(0., 1., 2)
        point = begin_points[None, :] * (1. - t_vals)[:, None] + end_points[None, :] * (t_vals)[:, None]
        points.append(point)
    points = np.concatenate(points)
    match_actor = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
    match_actor.paint_uniform_color(c)

    return match_actor, lines

def create_camera_actor(g, scale=0.05):
    """build open3d camera polydata"""
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES),
    )

    color = (g * 1.0, 0.5 * (1 - g), 0.9 * (1 - g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def get_trajectory_pc(poses, c=[1, 0, 0]):
    traj_actor = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(poses[:, :3, 3]))
    traj_actor.paint_uniform_color(c)
    return traj_actor


def write_video(imgs_path, save_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = 1080, 1920
    rgb_writer = cv2.VideoWriter(save_path, fourcc, 30, (w, h))
    rgbs = sorted(glob.glob(os.path.join(imgs_path, '*.png'))) # frame-000012.png

    for i in range(len(rgbs)):
        rgb = cv2.imread(rgbs[i])
        rgb_writer.write(rgb)

def filter_outlier(pre, gt):
    '''
    pre: [N, 4, 4]
    gts: [N, 4, 4]
    '''
    t1 = pre[:, :3, 3]
    t2 = gt[:, :3, 3]

    dist = np.linalg.norm((t1 - t2), axis=1)
    print('dist shape: ', dist.shape)
    print('max: ', np.max(dist))

    mask = dist < 0.1
    return mask

if __name__ == '__main__':
    '''
    12-scenes: given mesh, and 2D-3D matches, to render
    also need GT pose, RGB image
    '''
    render_type = 'color' # color normal
    src_root = '/home/hongjia/Documents/GS-loc/localization_process'
    save_root = '/home/hongjia/Documents/results'
    scenes = ['of2_5b'] # apt1_kitchen apt2_living of2_5b
    write_video_flag = False # True False

    for scene in scenes:
        # save path for render images
        save_path = os.path.join(save_root, scene, 'loc_{}'.format(render_type))
        os.makedirs(save_path, exist_ok=True)
        
        # file paths
        match_dir = os.path.join(src_root, scene, 'save_match')
        rgb_dir = os.path.join(src_root, scene, 'colors')
        pose_dir = os.path.join(src_root, scene, 'save_poses')
        ply_path = os.path.join(src_root, scene, 'mesh.ply')

        mesh = o3d.io.read_triangle_mesh(ply_path)
        mesh.compute_vertex_normals()

        matches = sorted(os.listdir(match_dir))
        pre_poses = np.load(os.path.join(pose_dir, 'match.npy'))
        gt_poses = np.load(os.path.join(pose_dir, 'gt.npy'))

        # filter outliers for smooth trajs
        mask = filter_outlier(pre_poses, gt_poses)
        pre_poses = pre_poses[mask]
        gt_poses = gt_poses[mask]
        print('masked pose: ', pre_poses.shape)

        matches = [d for d, m in zip(matches, mask) if m]
        print('masked matches: ', len(matches))
        num_match = len(matches)

        frame_count = 0
        camera_scale = 0.7 # 0.8
        fixed_viewpoint = None

        # write img to video
        if write_video_flag:
            imgs_path = os.path.join(src_root, scene, 'loc_color')
            save_path = os.path.join(save_root, scene, '{}_match_gt_traj.mp4'.format(scene))
            write_video(imgs_path, save_path)
            exit()

        def update_mesh_and_pose(vis):
            global frame_count, fixed_viewpoint
            vis.clear_geometries()

            # our pose
            pred_pose = pre_poses[frame_count]
            pred_traj = pre_poses[:frame_count]

            # gt pose
            gt_pose = gt_poses[frame_count]
            gt_traj = gt_poses[:frame_count]
            
            # see image from back face
            opt = vis.get_render_option()
            opt.mesh_show_back_face = True

            if render_type == 'normal':
                opt.mesh_color_option = o3d.visualization.MeshColorOption.Normal
            elif render_type == 'color':
                opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
            
            vis.add_geometry(mesh, reset_bounding_box=(frame_count == 0))

            # trajs
            if len(pred_traj) > 1:
                updated_trajectory_pre = get_trajectory_pc(pred_traj, [1,0,0]) # [1,0,0] is red
                vis.add_geometry(updated_trajectory_pre)

                updated_trajectory_gt = get_trajectory_pc(gt_traj, [0,0,0])
                vis.add_geometry(updated_trajectory_gt)

            pred_camera = create_camera_model(K_12scenes, pred_pose, (640, 480), [1,0,0], camera_scale) # red
            gt_camera = create_camera_model(K_12scenes, gt_pose, (640, 480), [0,0,0], camera_scale) # black
            vis.add_geometry(pred_camera)
            vis.add_geometry(gt_camera)

            # 2D-3D matches
            name = matches[frame_count].split('.')[0]
            match_info = np.load(os.path.join(match_dir, matches[frame_count]), allow_pickle=True).item()
            inlies = match_info['inliers']
            kp_2d = match_info['2d'][inlies] # [N, 2]
            pt_3d = match_info['3d'][inlies] # [N, 3]

            rgb_path = os.path.join(rgb_dir, name+'.color.jpg')
            rgb = cv2.imread(rgb_path)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  
            rgb = cv2.resize(rgb, (640, 480), cv2.INTER_AREA)
            rgb = cv2.flip(rgb, 0)
            # print('rgb: ', rgb.shape)
            
            image_plane = visualize_image_on_camera(rgb, K_12scenes, pred_pose, scale=camera_scale)
            match_actor, lines = visualize_match(kp_2d, pt_3d, K_12scenes, pred_pose, scale=camera_scale)

            vis.add_geometry(image_plane)
            vis.add_geometry(match_actor)
            vis.add_geometry(lines)

            # viewpoint in open3d
            if fixed_viewpoint is None:
                curr_frustum = create_frustum(pred_pose, size=1.0)
                viewpoint = curr_frustum.view_dir_behind
                view_control = vis.get_view_control()
                view_control.set_lookat(viewpoint[0])
                view_control.set_front(viewpoint[1])
                view_control.set_up(viewpoint[2])
                view_control.set_zoom(0.5)  
            else:
                view_ctl = vis.get_view_control()
                param = view_ctl.convert_to_pinhole_camera_parameters()
                param.extrinsic = fixed_viewpoint # [4, 4] np.array
                view_ctl.convert_from_pinhole_camera_parameters(param,  allow_arbitrary=True)

            vis.poll_events()
            vis.update_renderer()

            vis.capture_screen_image(os.path.join(save_path, '{}.png'.format(name)), False)
            frame_count += 1
            sleep(0.01)

    def update_mesh_and_pose_continuous(vis):
        global num_match
        print("start playing contiuously")
        for i in range(num_match):
            update_mesh_and_pose(vis)
        print("end playing contiuously")


    def set_fix_viewpoint(vis):
        global fixed_viewpoint
        view_ctl = vis.get_view_control()
        param = view_ctl.convert_to_pinhole_camera_parameters()
        fixed_viewpoint = np.array(param.extrinsic)
        print('Set fixed_viewpoint: ', fixed_viewpoint)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    vis.register_key_callback(65, update_mesh_and_pose) # a: update mesh and pose, once
    vis.register_key_callback(66, update_mesh_and_pose_continuous) # b: update mesh and pose automatically
    vis.register_key_callback(67, set_fix_viewpoint) # c: set current view as the fixed viewpoint

    vis.run()
    # vis.destroy_window()