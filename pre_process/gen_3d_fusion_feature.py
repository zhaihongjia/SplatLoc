import sys
sys.path.append('..')

import utils.config_utils
from utils.datasets import load_dataset

import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
import torch
import argparse

from utils.fusion_utils import TSDFVolumeTorch, meshwrite

def run_tsdfusion(dataset):
    H, W = dataset.height, dataset.width
    fx, fy, cx, cy = dataset.K[0,0], dataset.K[1,1], dataset.K[0,2], dataset.K[1,2]
    K = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    voxel_length = 0.02
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel_length, sdf_trunc=0.04,
                                                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for i, frame in tqdm(enumerate(dataset)):
        if not frame['valid']:
            continue

        if i > 155 and i < 163:
            rgb, depth = frame["rgb"].cpu().numpy(), frame["depth"].cpu().numpy()
            c2w = frame["c2w"].cpu().numpy()
            rgb = rgb * 255
            rgb = rgb.astype(np.uint8)
            rgb = o3d.geometry.Image(rgb)
            depth = depth.astype(np.float32)
            depth = o3d.geometry.Image(depth)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth, depth_scale=1.0,
                                                                        depth_trunc=8.0,
                                                                        convert_rgb_to_intensity=False)
            w2c = np.linalg.inv(c2w)
            # w2c = c2w
            # requires w2c
            volume.integrate(rgbd, K, w2c)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh

def run_feature_fusion(dataset, bounds, data_path):
    H, W = dataset.height, dataset.width
    fx, fy, cx, cy = dataset.K[0,0], dataset.K[1,1], dataset.K[0,2], dataset.K[1,2]
    voxel_length = 0.02
    feat_dim = 256 # sam:256， SP： 256

    voxel_size = 0.02
    voxel_dim = (bounds[:,1] - bounds[:,0]) / voxel_size
    world_dims = (voxel_dim - 1) * voxel_size
    volume_origin = bounds[:,0] - (world_dims - bounds[:,1] + bounds[:,0]) / 2

    voxel_dim = torch.from_numpy(voxel_dim)
    volume_origin = torch.from_numpy(volume_origin)
    K = dataset.K

    print('bounds: ', bounds)
    print('voxel_dim: ', voxel_dim)
    print('world_dims: ', world_dims)
    print('volume_origin: ', volume_origin)
    
    # voxel_dim, origin, voxel_size, margin=3
    # _sdf_trunc = margin * voxel_size
    volume = TSDFVolumeTorch(voxel_dim=voxel_dim, origin=volume_origin, voxel_size=voxel_size, feat_dim=feat_dim, margin=2)
    
    # results save path
    mesh_path = os.path.join(data_path, 'train', 'mesh.ply') 
    volume_path = os.path.join(data_path, 'train', 'volume.pt') 
    feat_path = os.path.join(data_path, 'train', 'feat_cloud.npy') 
    
    for i in tqdm(range(len(dataset))):
        frame = dataset[i]
        if not frame['valid']:
            continue

        rgb, depth = frame["rgb"].cpu(), frame["depth"].cpu()
        c2w = frame["c2w"].cpu()
        rgb = rgb * 255
        feat = frame['sp_feature']
    
        # w2c = torch.from_numpy(np.linalg.inv(c2w.numpy()))
        # requires w2c
        volume.integrate(depth_im=depth, color_im=rgb, feat_im=feat, cam_intr=K, cam_pose=c2w)

    verts, faces, norms, colors, feats = volume.get_mesh()
    meshwrite(mesh_path, verts, faces, norms, colors)
    # volume.cpu().save(volume_path)
    np.save(feat_path, feats)

if __name__ == '__main__':
    '''
    python run_fusion.py --config ./configs/replica_nerf/room_0.yaml
    '''
    parser = argparse.ArgumentParser(description='Arguments for running feature TSDF Fusion')
    parser.add_argument('--config', type=str, help='Path to config.')    
    args = parser.parse_args()

    # load config
    cfg = config.load_config(args.config)
    scene_bound = np.array(cfg["scene"]["bound"])
    data_path = cfg['data']['datadir']
    print('Process: ', data_path)

    # load dataset
    dataset = load_dataset(config=cfg)

    ## run open3d tsdf-fusion to obtain scene bound
    # mesh = run_tsdfusion(dataset)
    # o3d.io.write_triangle_mesh("/mnt/nas_54/group/hongjia/test-replica-room0.ply", mesh)

    ## run our feature fusion
    # NOTE: you should make sure: load 2D feature map: 'sp_feature'
    dataset.set_feature_flag(True)
    run_feature_fusion(dataset, scene_bound, data_path)
