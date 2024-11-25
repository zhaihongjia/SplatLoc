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
        # turn to pose 
        points = np.asarray(self.line_set.points)
        points_hmg = np.hstack([points, np.ones((points.shape[0], 1))])
        points = (pose @ points_hmg.transpose())[0:3, :].transpose()

        # turn camera orginal to pose
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

CAM_LINES = np.array(
    [[1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [
        0, 2], [3, 0], [0, 4], [5, 7], [7, 6]]
)

def get_trajectory_pc(poses, c=[1, 0, 0]):
    traj_actor = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(poses[:, :3, 3]))
    traj_actor.paint_uniform_color(c)
    return traj_actor

def write_video(imgs_path, save_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = 1080, 1920
    rgb_writer = cv2.VideoWriter(save_path, fourcc, 30, (w, h))
    rgbs = sorted(glob.glob(os.path.join(imgs_path, '*.png')), key=lambda x: int(os.path.basename(x).split('.')[0])) # basename: 1.png

    print(len(rgbs), ' images to video.')
    print('Save video to: ', save_path)

    for i in range(len(rgbs)):
        rgb = cv2.imread(rgbs[i])
        rgb_writer.write(rgb)

def filter_outlier(pre, gt, thre=0.1):
    '''
    pre: [N, 4, 4]
    gts: [N, 4, 4]
    '''
    t1 = pre[:, :3, 3]
    t2 = gt[:, :3, 3]

    dist = np.linalg.norm((t1 - t2), axis=1)
    print('dist shape: ', dist.shape)
    print('max: ', np.max(dist))
    print('min: ', np.min(dist))

    mask = dist < thre
    print('mask shape: ', pre[mask].shape)
    return pre[mask], gt[mask]


of2_5b_vp = np.array(
[[ 1.85602237e-02, -9.99827693e-01,  3.19038398e-04,  3.86510906e-01],
 [-7.08114658e-01, -1.33703021e-02, -7.05970867e-01,  9.23351131e-01],
 [ 7.05853490e-01,  1.28770614e-02, -7.08240801e-01,  5.49834995e+00],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

apt2_living_vp = np.array(
[[ 0.13354640, -0.98587043,  0.10111804, -0.02947762],
 [-0.98460175, -0.14359969, -0.09969212,  0.19621909],
 [ 0.11280403, -0.08624748, -0.98986697,  4.75530628],
 [ 0.,          0.,          0.,          1.,        ]])


if __name__ == '__main__':
    '''
    given mesh, pre_pose, gt_pose, render mesh, camera, trajs
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", type=str, default='/home/hongjia/Documents/GS-loc/localization_process')
    parser.add_argument("--save_path", type=str, default='/home/hongjia/Documents/results')
    parser.add_argument("--frame_rate", type=int, default=30)
    args = parser.parse_args()

    render_type = 'color' # color normal
    # ['apt1_kitchen', 'apt1_living', 'apt2_bed', 'apt2_kitchen', 'apt2_living', 'apt2_luke', 
    # 'of1_gates362', 'of1_gates381', 'of1_lounge', 'of1_manolis', 'of2_5a', 'of2_5b']
    scene = 'apt2_living' #  apt1_kitchen apt2_living of2_5b
    render_method = 'splatloc' # pnerfloc neuraloc splatloc
    save_video = False # False True

    if render_method == 'splatloc':
        save_path = os.path.join(args.save_path, scene, 'SplatLoc')
        video_save_path = os.path.join(args.save_path, scene, '{}_SplatLoc.mp4'.format(scene))

    os.makedirs(save_path, exist_ok=True)
    frame_rate = args.frame_rate

    # write img to video
    if save_video:
        write_video(save_path, video_save_path)
        exit()

    mesh = None
    frame_count = 0
    fixed_viewpoint = None
    camera_scale = 0.3

    # scene: Viewpoint mapping
    scene2viewpoint = {
                     'apt2_living': apt2_living_vp, 
                     'of2_5b': of2_5b_vp, 
                    }

    # load mesh
    mesh_path = '/home/hongjia/Documents/GS-loc/localization_process/{}/mesh.ply'.format(scene)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    # our poes
    c2ws = np.load('/home/hongjia/Documents/GS-loc/localization_process/{}/save_poses/match.npy'.format(scene))
    gts = np.load('/home/hongjia/Documents/GS-loc/localization_process/{}/save_poses/gt.npy'.format(scene))
    c2ws, gts = filter_outlier(c2ws, gts, 0.1)
    render_color = [1,0,0]

    max_frame = c2ws.shape[0] - 1

    def set_fix_viewpoint(vis):
        global fixed_viewpoint
        fixed_viewpoint = np.array(scene2viewpoint[scene])
        print('Set fixed_viewpoint: ', fixed_viewpoint)

    def set_current_as_fix_viewpoint(vis):
        global fixed_viewpoint
        view_ctl = vis.get_view_control()
        param = view_ctl.convert_to_pinhole_camera_parameters()
        fixed_viewpoint = np.array(param.extrinsic)
        print('Set fixed_viewpoint: ', fixed_viewpoint)

    def update_mesh_and_pose(vis):
        global frame_count, fixed_viewpoint
        vis.clear_geometries()

        pred_pose = c2ws[frame_count]
        pred_traj = c2ws[:frame_count]

        gt_pose = gts[frame_count]
        gt_traj = gts[:frame_count]
        
        opt = vis.get_render_option()
        if render_type == 'normal':
            opt.mesh_color_option = o3d.visualization.MeshColorOption.Normal
        elif render_type == 'color':
            opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
        
        vis.add_geometry(mesh, reset_bounding_box=(frame_count == 0))

        # trajs
        if len(pred_traj) > 1:
            updated_trajectory_pre = get_trajectory_pc(pred_traj, render_color) # color [1,0,0] is red
            vis.add_geometry(updated_trajectory_pre)

            updated_trajectory_gt = get_trajectory_pc(gt_traj, [0,0,0])
            vis.add_geometry(updated_trajectory_gt)

        pred_camera = create_camera_model(K_12scenes, pred_pose, (640, 480), render_color, camera_scale) # [1,0,0] red
        gt_camera = create_camera_model(K_12scenes, gt_pose, (640, 480), [0,0,0], camera_scale) # black
        vis.add_geometry(pred_camera)
        vis.add_geometry(gt_camera)

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

        vis.capture_screen_image(os.path.join(save_path, '{}.png'.format(frame_count)), False)
        frame_count += 1
        sleep(0.01)

    def update_mesh_and_pose_continuous(vis):
        global max_frame
        print("start playing contiuously")
        for i in range(max_frame):
            update_mesh_and_pose(vis)
        print("end playing contiuously")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    vis.register_key_callback(65, update_mesh_and_pose) # a
    vis.register_key_callback(66, update_mesh_and_pose_continuous) # b
    vis.register_key_callback(67, set_fix_viewpoint) # c
    vis.register_key_callback(68, set_current_as_fix_viewpoint) # d
    vis.run()
    # vis.destroy_window()