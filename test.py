from ctypes import pointer
from pathlib import Path
from numpy.core.fromnumeric import resize
import torch
import numpy as np
import cv2
import os
import yaml
import pycolmap
import time
from collections import defaultdict
from types import SimpleNamespace
import sys
import random
import datetime
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import argparse
import open3d as o3d
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.spatial import cKDTree
import imgviz

from munch import munchify

# HLoc import
sys.path.append('submodules/Hierarchical-Localization')
from hloc import extractors, extract_features
from hloc.utils.base_model import dynamic_load

# utils
from utils.match_utils import HungarianMatcher
from utils.config_utils import load_config
from utils.vis_match_utils import vis_matches
from utils.eval_utils import eval_rendering, save_gaussians, eval_pose, makePlyFile
from utils.camera_utils import Camera
from utils.dataset import load_dataset
from utils.selection import gaussian_selectition, random_down_sample

# gs rendering
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2

from models.decoders import FeatureDecoder

replica_intrinsic = {
    "model": 'OPENCV',
    "width": 640,
    "height": 480,
    # [fx, fy, cx, cy, k1, k2, p1, p2]
    "params": [640.0 / 2.0 / 0.9999999999999999, 640.0 / 2.0 / 0.9999999999999999, (640 - 1.0) / 2.0, (480 - 1.0) / 2.0, 0., 0., 0., 0.]
}

scene12_intrinsic = {
    "model": 'OPENCV',
    "width": 640,
    "height": 480,
    # [fx, fy, cx, cy, k1, k2, p1, p2]
    "params": [572, 572, 320, 240, 0., 0., 0., 0.]
}

def solve_pose(kp_2d, kp_3d, intrinsics):
    ransac_thresh = 12
    ret = pycolmap.absolute_pose_estimation(kp_2d, kp_3d, intrinsics)
    '''
        ret['qvec'] = q
        ret['tvec'] = t
        ret['success'] = True
        ret['num_inliers'] = tag['num_inliers']
        ret['inliers'] = tag['inliers']
    '''
    if not ret['success']:
        return None, None, ret

    q = ret['qvec'].tolist()
    q = [q[1], q[2], q[3], q[0]]
    rmatrix = R.from_quat(q).as_matrix()
    t = np.array(ret['tvec'].tolist())  
    rmatrix = np.transpose(rmatrix)
    t = - rmatrix @ t

    return rmatrix, t, ret

class LocalizeQuery:
    def __init__(self, config):
        self.device = 'cuda' 
        self.config = config

        self.pipeline_params = munchify(config["pipeline_params"])
        self.background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
        self.save_dir = config['save_dir']
        self.show_imgwise_error = config['Results']['show_imgwise_error']

        print('Eval from: ', self.save_dir)

        self.gt_poses = []
        self.ret_poses = []
        self.match_poses = []

        self.train_dataset = load_dataset(config=config)
        self.test_dataset = load_dataset(config=config, train=False)

        self.pre_setting(config)
        self.subset_xyz = None
        self.sp_kp_thre = 0.005

    def get_gt_pose(self, ):
        poses = []
        for it in range(len(self.train_dataset)):
            poses.append(self.train_dataset[it]['w2c'].squeeze().numpy())
        return np.array(poses)

    def pre_setting(self, config):
        self.dataset_name = config['Dataset']['type']
        self.scene_name = config['Dataset']['dataset_path'].split('/')[-1]

        # load feature decoder
        self.feat_decoder = FeatureDecoder(config).cuda()

        # load SuperPoint: indoor config
        print('=======> Load SuperPoint Model.')
        self.feature_conf = extract_features.confs['superpoint_inloc']
        self.model_sparse = dynamic_load(extractors, self.feature_conf['model']['name'])
        self.model_sparse = self.model_sparse(self.feature_conf['model']).eval().to(self.device)

        # load gaussians
        if self.dataset_name == 'replica':
            self.ply_path = os.path.join(self.save_dir, 'point_cloud/final/point_cloud.ply')
            self.feat_decoder.load_state_dict(torch.load(os.path.join(self.save_dir, 'train_feat/ckpt.pth')))
            self.ret_path = os.path.join(self.train_dataset.generated_folder , 'netvlad_retrieval.txt')
            self.intrinsics = replica_intrinsic

        elif self.dataset_name == '12scenes':
            self.pre_name = config['Dataset']['dataset_path'].split('/')[-2]
            self.ply_path = os.path.join(self.save_dir, 'point_cloud/final/point_cloud.ply')
            self.feat_decoder.load_state_dict(torch.load(os.path.join(self.save_dir, 'train_feat/ckpt.pth')))
            self.ret_path = os.path.join(self.train_dataset.generated_folder , 'netvlad_retrieval.txt')
            self.intrinsics = scene12_intrinsic

        print('=======> Load GS Map from: ', self.ply_path)
        self.gaussians = GaussianModel(config["model_params"]['sh_degree'], config=self.config)
        self.gaussians.load_ply(self.ply_path)

        print('=======> Load Retrieval file from: ', self.ret_path)
        self.load_retrieval_results(self.ret_path)

        self.save_match = self.config['Results']['save_match']
        if self.save_match:
            os.makedirs(os.path.join(self.config['save_dir'], 'save_match'), exist_ok=True)

    def query_feature_volume(self, points_3d, topk=1):
        '''
        input param:
        points_3d: [N, 3], cuda tensor
        topk: int
        '''
        # indices: [N, topk]
        distances, indices = self.feature_index.search(points_3d.cpu().numpy(), topk)

        features = self.pcd_features[indices] # [N, topk, feature_dim]
        features = features.mean(axis=1) # [N, feature_dim]

        return torch.from_numpy(features).cuda()

    def load_retrieval_results(self, file_path):      
        self.retrieval_results = {}

        with open(file_path, 'r') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            names = lines[i].replace('\n', '').split(' ')
            key = names[0]
            values = names[1:]

            self.retrieval_results[key] = values
        
    def preprocess_image(self, image):
        # refer to extract_features for more detail
        # Note: the image should be grayscale
        conf = self.feature_conf['preprocessing']
        if image is None:
            raise ValueError(f'Preprocess: No image input!')
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]
        w, h = size

        # may need change here for big picture
        if conf["resize_max"] and max(w, h) > conf["resize_max"]:
            scale = conf["resize_max"] / max(h, w)
            h_new, w_new = int(round(h * scale)), int(round(w * scale))
            image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        if conf["grayscale"]:
            image = image[None] ## zhj： from HxW to 1xHxW
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.

        data = {
            # 'name': [f"Query: {self.query_count}"],
            'image': torch.tensor([image]),
            'original_size': torch.tensor([np.array(size)]),
        }
        return data

    @torch.no_grad()
    def extract_feature(self, query_img_path):
        """
        query_img is a numpy array
        return feature as a dict:
        """
        query_image = cv2.imread(query_img_path, cv2.IMREAD_GRAYSCALE) # [H, W]
        query_image = cv2.resize(query_image, (640, 480), cv2.INTER_AREA)
        data = self.preprocess_image(query_image)

        pred = self.model_sparse({"image": data["image"].to(self.device, non_blocking=True)})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        pred['image_size'] = original_size = data['original_size'][0].numpy()
        if 'keypoints' in pred:
            size = np.array(data['image'].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5

        return pred
    
    def image_retrieval(self, query_image):
        """
        Image retrieval using NetVLAD
        """
        query_name = os.path.basename(query_image)
        retrieval_names = self.retrieval_results[query_name]
        return retrieval_names

    def eval_result(self, results, gt):
        pre_r = torch.from_numpy(results['r'][None, :3, :3])
        pre_t = torch.from_numpy(results['t'][None, :])

        gt_r = gt[None, :3, :3]
        gt_t = gt[None, :3, 3]

        rotation_error, trans_error = eval_pose(pre_r, pre_t, gt_r, gt_t, self.show_imgwise_error)
        return rotation_error.numpy(), trans_error.numpy()

    def get_frusm_pts(self, frame):
        w2c = frame['w2c'].numpy()
        h, w = self.train_dataset.height, self.train_dataset.width
        intrinsics_matrix = self.train_dataset.K

        if self.subset_xyz is not None:
            all_pts = self.subset_xyz
        else:
            all_pts = self.gaussians.get_xyz.detach().cpu().numpy()

        points_camera = np.dot(all_pts, w2c[:3, :3].T) + w2c[:3, 3]
        projected_points = np.dot(intrinsics_matrix, points_camera.T).T
        projected_points = projected_points[:, :2] / projected_points[:, 2][:, np.newaxis] # [w, h]
        
        mask = (points_camera[:, 2] > 0.05) & (0 <= projected_points[:, 0]) & (projected_points[:, 0] < w) \
                & (0 <= projected_points[:, 1]) & (projected_points[:, 1] < h)
        
        # select key gaussians as candidates
        if self.subset_xyz is None:
            score = self.gaussians.get_marker.squeeze().detach().cpu().numpy()
            mask = mask & (score > self.sp_kp_thre)

        # mask outlier
        ref_pts_3d = all_pts[mask]
        ref_pts_2d = projected_points[mask]

        if self.subset_xyz is None:
            ref_kp_3d = self.get_ref_keyponts_3d(frame) # [N, 3] numpy
            tree = cKDTree(ref_pts_3d)
            distances, indices = tree.query(ref_kp_3d, distance_upper_bound=0.1)        
            mask = distances < 0.1

            filtered_ref_kp = ref_kp_3d[mask]
            ref_pts_3d = ref_pts_3d[indices[mask]]
            ref_pts_2d = ref_pts_2d[indices[mask]]
            
        ref_feats_3d = self.feat_decoder(torch.from_numpy(ref_pts_3d)) # [N, 256]

        return ref_pts_3d, ref_feats_3d, ref_pts_2d
    
    def get_ref_keyponts_3d(self, frame):
        fx, fy, cx, cy = frame['K'][0, 0], frame['K'][1, 1], frame['K'][0, 2], frame['K'][1, 2] 
        mask = frame['sp_kp_mask'].numpy()  # [H, W]
        depth = frame['depth'].numpy()      # [H, W]
        pose = frame['c2w'].numpy()         # [4, 4]

        # project to 3d
        kps_2d = np.argwhere(mask == 1) # [H, W]
        kps_2d_depth = depth[mask == 1]
        xs = (kps_2d[:, 1] - cx) * kps_2d_depth / fx
        ys = (kps_2d[:, 0] - cy) * kps_2d_depth / fy
        zs = kps_2d_depth
        point_c = np.stack([xs, ys, zs], axis=-1)  # [N, 3]
        point_w = np.dot(pose[:3, :3], point_c.T).T + pose[:3, 3]

        return point_w

    @torch.no_grad()
    def match_feature(self, query_feature, retrieval_names, query_frame):
        # 1. parse query frame feature 
        query_kps_2d = query_feature['keypoints']       # [N, 2]   numpy(), (u, v): (width, height)
        query_feats_2d = query_feature['descriptors']   # [256, N] numpy()

        # 2. ret db frame feature
        name = retrieval_names[0]
        index = self.train_dataset.name_to_index(name)
        db_frame = self.train_dataset.get_frame(index)
        db_kps_3d, db_feats_3d, db_kps_2d = self.get_frusm_pts(db_frame)
        db_kps_2d = db_kps_2d[:, ::-1] # [w,h] -> [h,w]

        # unstable results
        if db_kps_3d.shape[0] < 5:
            retrieval_ret = {'r': db_frame['c2w'][:3, :3].cpu().numpy(),
                        't': db_frame['c2w'][:3, 3].cpu().numpy()}

            match_ret = {'r': retrieval_ret['r'],
                't': retrieval_ret['t'],
                'success': False}

            return retrieval_ret, match_ret

        # check projection
        # makePlyFile(db_kps_3d, db_kps_3d, '/mnt/nas_7/group/hongjia/tmp/point.ply')

        # 4. matcher
        macher = HungarianMatcher()
        data = {}
        data['query_descs'] = torch.from_numpy(query_feats_2d)
        data['train_descs'] = db_feats_3d.T
        match_results = macher(data)

        # 5. PnP
        match_query_kps_2d = query_kps_2d[match_results['matches'][0]]
        match_db_kps_3d = db_kps_3d[match_results['matches'][1]]
        match_db_kps_2d = db_kps_2d[match_results['matches'][1]]
        # print('query_kps_2d: ', query_kps_2d.shape)
        # print('db_kps_3d: ', db_kps_3d.shape)
        kp_3d_mask = match_db_kps_3d[:, 2] > - 10000
        r, t, ret = solve_pose(match_query_kps_2d[kp_3d_mask], match_db_kps_3d[kp_3d_mask], self.intrinsics)
        
        # vis debug
        # 2d: (h, w)
        # query_name = query_frame['img_path'].split('/')[-1].split('.')[0]
        # query_image = query_frame['rgb'].numpy() * 255

        # 2d kp, should be [h, w]
        # vis = self.vis_match_results(query_image, db_frame['rgb'].numpy()*255, match_query_kps_2d[:, ::-1], match_db_kps_2d, ret['inliers'])
        # save_path = os.path.join(self.save_dir, '{}_{}.png'.format(query_name, name))
        # cv2.imwrite(save_path, vis[:, :, ::-1])
        
        # save 2D-3D matches for visualization
        if self.save_match:
            match_info = {'success': ret['success'],
                        '2d': match_query_kps_2d, 
                        '3d': match_db_kps_3d,}
            
            if ret['success']:
                match_info['inliers'] = ret['inliers']
            
            query_name = query_frame['img_path'].split('/')[-1].split('.')[0]
            save_path = os.path.join(self.config['save_dir'], 'save_match', query_name+'.npy')
            np.save(save_path, match_info)
        
        retrieval_ret = {'r': db_frame['c2w'][:3, :3].cpu().numpy(),
                        't': db_frame['c2w'][:3, 3].cpu().numpy()}

        match_ret = {'r': r,
               't': t,
               'success': ret['success']}

        return retrieval_ret, match_ret

    def parse_feature(self, sp_feat):
        kps = sp_feat['keypoints']            # [N, 2]
        scores = sp_feat['scores']            # [N]
        descriptors = sp_feat['descriptors']  # [256, N]

        print('kps: ', kps.shape)
        print('kps: ', kps[:, 0].max(), kps[:, 1].max())
        assert kps[:, 0].max() < 640
        assert kps[:, 1].max() < 480

        print('scores: ', scores.shape)
        print('descriptors: ', descriptors.shape)

    def vis_match_results(self, img_a, img_b, kp1, kp2, mask):
        print('img_a: ', img_a.shape)
        print('img_b: ', img_b.shape)
        print('kp1: ', kp1.shape)
        print('kp2: ', kp2.shape)

        kp1 = kp1[mask]
        kp2 = kp2[mask]

        vis = vis_matches(img_a, img_b, kp1, kp2)

        return vis

    def localize_image(self, query_frame):
        query_img_path = query_frame['img_path']
        # 1. retrieval
        retrieval_names = self.image_retrieval(query_img_path)
        # print('retrieval_names: ', retrieval_names)

        # 2. extract superpoint kp & feature for query
        query_feature = self.extract_feature(query_img_path)
        # self.parse_feature(query_feature)

        # 3. perform pnp
        # query_image = cv2.resize(query_image, (640, 480), cv2.INTER_AREA)
        retrieval_results, match_result = self.match_feature(query_feature, retrieval_names, query_frame)
         
        return retrieval_results, match_result
    
    def record_pose(self, ret, match, gt):
        rt = np.eye(4)
        rt[:3, :3] = ret['r']
        rt[:3, 3] = ret['t']
        self.ret_poses.append(rt)

        rt = np.eye(4)
        rt[:3, :3] = match['r']
        rt[:3, 3] = match['t']
        self.match_poses.append(rt)
        
        rt = np.eye(4)
        rt[:3, :3] = gt[:3,:3]
        rt[:3, 3] = gt[:3, 3]
        self.gt_poses.append(rt)

    def save_poses(self,):
        os.makedirs(os.path.join(self.config['save_dir'], 'save_poses'), exist_ok=True)

        save_path = os.path.join(self.config['save_dir'], 'save_poses', 'retrieval.npy')
        np.save(save_path, self.ret_poses)

        save_path = os.path.join(self.config['save_dir'], 'save_poses', 'match.npy')
        np.save(save_path, self.match_poses)

        save_path = os.path.join(self.config['save_dir'], 'save_poses', 'gt.npy')
        np.save(save_path, self.gt_poses)

    def save_errors(self, retrieval, match):
        os.makedirs(os.path.join(self.config['save_dir'], 'pose_errors'), exist_ok=True)

        # save error results
        t_save_path = os.path.join(self.config['save_dir'], 'pose_errors', 'retrieval_t_errors.npy')
        r_save_path = os.path.join(self.config['save_dir'], 'pose_errors', 'retrieval_r_errors.npy')
        np.save(t_save_path, retrieval[0])
        np.save(r_save_path, retrieval[1])

        t_save_path = os.path.join(self.config['save_dir'], 'pose_errors', 'match_t_errors.npy')
        r_save_path = os.path.join(self.config['save_dir'], 'pose_errors', 'match_r_errors.npy')
        np.save(t_save_path, match[0])
        np.save(r_save_path, match[1])

    def eval_pose(self, file_name='eval_pose.txt', save_pose=False):
        retrieval_r_errors = []
        retrieval_t_errors = []

        match_r_errors = []
        match_t_errors = []

        progress_bar = tqdm(range(len(self.test_dataset)), position=0)
        progress_bar.set_description("Evaluate Pose")

        for i in progress_bar:
            img_path = self.test_dataset.color_paths[i]
            query_frame = self.test_dataset.get_frame(i)

            if not query_frame['valid']:
                continue

            query_frame['img_path'] = img_path
            # print('Eval: ', img_path)
            retrieval_results, match_result = self.localize_image(query_frame)

            if not match_result['success']:
                continue

            self.record_pose(retrieval_results, match_result, query_frame['c2w'])
            
            retrival_r_error, retrieval_t_error = self.eval_result(retrieval_results, query_frame['c2w'])
            match_r_error, match_t_error = self.eval_result(match_result, query_frame['c2w'])

            retrieval_r_errors.append(retrival_r_error)
            retrieval_t_errors.append(retrieval_t_error)
            
            match_r_errors.append(match_r_error)
            match_t_errors.append(match_t_error)

        # echo results
        print('Retrieval: rotation & translation:')
        print('Mean: ', np.mean(retrieval_r_errors), np.mean(retrieval_t_errors))
        print('Median: ', np.median(retrieval_r_errors), np.median(retrieval_t_errors))
        print('Match: rotation & translation:')
        print('Mean: ', np.mean(match_r_errors), np.mean(match_t_errors))
        print('Median: ', np.median(match_r_errors), np.median(match_t_errors))

        # pose error
        pose_error_file = os.path.join(save_dir, file_name)

        f = open(pose_error_file, 'w')
        f.write('Median Error: \n')
        f.write('Retrieval: Trans.(cm): {}. Rotation(deg): {}.\n'.format(np.median(retrieval_t_errors)*100, np.median(retrieval_r_errors)))
        f.write('Match    : Trans.(cm): {}. Rotation(deg): {}.\n'.format(np.median(match_t_errors)*100, np.median(match_r_errors)))
        f.close()

        if save_pose:
            self.save_poses()
            self.save_errors([retrieval_t_errors, retrieval_r_errors], [match_t_errors, match_r_errors])

    def eval_rendering(self,):
        projection_matrix = getProjectionMatrix2(
                    znear=0.01,
                    zfar=100.0,
                    fx=self.train_dataset.fx,
                    fy=self.train_dataset.fy,
                    cx=self.train_dataset.cx,
                    cy=self.train_dataset.cy,
                    W=self.train_dataset.width,
                    H=self.train_dataset.height,
                ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        
        views = []
        # train dataset: viewpoint
        for it in range(len(self.test_dataset)):
            viewpoint = Camera.init_from_dataset(self.test_dataset, it, projection_matrix)
            viewpoint.compute_grad_mask(self.config)
            views.append(viewpoint)

        # eval rendering
        rendering_result = eval_rendering(
            views,
            self.gaussians,
            self.test_dataset,
            self.save_dir,
            self.pipeline_params,
            self.background,
        )
        print('\n\n======> eval: \n')
        print('Mean PSNR:', rendering_result["mean_psnr"])
        print('Mean SSIM:', rendering_result["mean_ssim"])
        print('Mean LPIPS:', rendering_result["mean_lpips"])

    def eval_landmark_selection(self, landmark_num):
        points_w = self.gaussians.get_xyz.detach().cpu().numpy()
        kp_score = self.gaussians.get_marker.detach().squeeze().cpu().numpy()
        
        # select key gaussians as candidates
        points_w = points_w[kp_score > self.sp_kp_thre]
        print('Number of key gaussians: ', points_w.shape)
        w2cs = self.get_gt_pose()
        intrinsics = self.train_dataset.K
        depths = self.train_dataset.load_all_depth()

        self.subset_xyz = gaussian_selectition(points_w, w2cs, intrinsics, depths, num_gs=landmark_num)
        # self.subset_xyz = random_down_sample(points_w, num_gs=num_gs)
        print('Select {} 3D landmarks'.format(self.subset_xyz.shape[0]))

if __name__ == '__main__':
    print('Start running...')
    parser = argparse.ArgumentParser(description='Arguments for running the LocalizeQuery.')
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--landmark_num', type=int, default=5000, help='Path to config file.')
    parser.add_argument('--eval_pose', default=False, action='store_true')
    parser.add_argument('--eval_rendering', default=False, action='store_true')
    parser.add_argument('--eval_selection', default=False, action='store_true')
    
    args = parser.parse_args()
    config = load_config(args.config)
    path = config["Dataset"]["dataset_path"].split("/")

    if config['Dataset']['type'] == 'replica':
        save_dir = os.path.join(config["Results"]["save_dir"], path[-2], path[-1])
    elif config['Dataset']['type'] == '12scenes':
        save_dir = os.path.join(config["Results"]["save_dir"], path[-3], path[-2] + '_' + path[-1])
    else:
        print('Dataset type should be replica or 12scenes')
        exit()

    config['save_dir'] = save_dir
    loc_server = LocalizeQuery(config)

    # eval
    if args.eval_pose:
        print('Eval localization.')
        loc_server.eval_pose(save_pose=True)

    if args.eval_rendering:
        print('Eval rendering.')
        loc_server.eval_rendering()

    if args.eval_selection:
        print('Eval landmark selection.')
        loc_server.eval_landmark_selection(args.landmark_num)
        loc_server.eval_pose('eval_selection_{}.txt'.format(loc_server.subset_xyz.shape[0]))




