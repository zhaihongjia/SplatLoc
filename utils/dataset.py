import csv
import glob
import os

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from tqdm import tqdm

from gaussian_splatting.utils.graphics_utils import focal2fov

class ReplicaDataset(torch.utils.data.Dataset):
    def __init__(self, config, train=True):
        super().__init__()
        self.config = config
        self.device = "cuda"
        self.dtype = torch.float32
        self.input_folder = config["Dataset"]["dataset_path"]
        self.scene_name = self.input_folder.split('/')[-1]
        self.generated_folder = os.path.join(config["Dataset"]["generated_folder"], self.scene_name)
        self.train = train
        self.train_step = 5
        self.sp_score_thre = 0.005

        if self.train:        
            self.color_paths = sorted(glob.glob(os.path.join(self.input_folder, 'Sequence_1', 'rgb', '*.png')), key=lambda x: int(os.path.basename(x)[4:-4]))[::self.train_step]
            self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, 'Sequence_1', 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[6:-4]))[::self.train_step]
        else:
            self.color_paths = sorted(glob.glob(os.path.join(self.input_folder, 'Sequence_2', 'rgb', '*.png')), key=lambda x: int(os.path.basename(x)[4:-4]))
            self.depth_paths = sorted(glob.glob(os.path.join(self.input_folder, 'Sequence_2', 'depth', '*.png')), key=lambda x: int(os.path.basename(x)[6:-4]))

        self.n_img = len(self.color_paths)        
        self.num_imgs = self.n_img
        self.poses = self.load_gt_pose()
        
        self.load_sp_feat_flag = False   # load superpoint feature map for 3D feature volume reconstruction
        self.load_score_flag = True # load score map
        # only used for train set
        self.sp_feat_path = os.path.join(self.generated_folder, 'sp_feature')
        self.sp_score_path = os.path.join(self.generated_folder, 'score_map')
        self.sparse_ply = os.path.join(self.generated_folder, 'sp_inloc_pc.ply')
        self.sparse_feature = os.path.join(self.generated_folder, 'sp_inloc_feat.npy')

        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

    def set_feature_flag(self, value):
        self.load_sp_feat_flag = value
    
    def index_to_name(self, index):
        name = os.path.basename(self.color_paths[index])[:-4]    # rgb_xx.png
        return name
    
    def name_to_index(self, name):
        matching_indices = [i for i, item in enumerate(self.color_paths) if name in item]
        assert len(matching_indices) == 1, ' should be only one match.'
        return matching_indices[0]

    def load_sp_feat(self, index):
        '''
        only used for feature volume reconstruction
        '''
        name = self.index_to_name(index)
        feat = torch.load('{}/{}.pt'.format(self.sp_feat_path, name)) # [1, 256, H, W]
        return feat

    def load_kp_feature_score(self, index):
        name = self.index_to_name(index)
        mask = torch.from_numpy(np.load('{}/{}_score.npy'.format(self.sp_score_path, name))) # [H, W]
        return mask

    def load_depth(self, index):
        depth = cv2.imread(self.depth_paths[index], cv2.IMREAD_UNCHANGED)
        
        # from semantic nerf
        # uint16 mm depth, then turn depth from mm to meter
        depth = depth.astype(np.float32) / self.depth_scale
        return depth

    def load_image(self, index):
        rgb = cv2.imread(self.color_paths[index], -1)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb / 255.0

    def load_gt_pose(self):
        if self.train:
            gt_file = os.path.join(self.input_folder, 'Sequence_1', 'traj_w_c.txt')
        else:
            gt_file = os.path.join(self.input_folder, 'Sequence_2', 'traj_w_c.txt')

        gt_pose = np.loadtxt(gt_file, delimiter=" ").reshape(-1, 4, 4)
        if self.train:
            gt_pose = gt_pose[::self.train_step, :, :]
        return gt_pose
    
    def load_all_depth(self,):
        all_depths = []

        progress_bar = tqdm(range(self.num_imgs), position=0)
        progress_bar.set_description("Load frame depth")
        
        for index in progress_bar:
            depth = self.load_depth(index)
            pose = self.poses[index]
            c2w = torch.from_numpy(pose).float()

            if torch.isnan(c2w).any() or torch.isinf(c2w).any():
                continue

            all_depths.append(depth)
        return np.array(all_depths)

    def get_frame(self, index):
        rgb = torch.from_numpy(self.load_image(index)).float()
        depth = torch.from_numpy(self.load_depth(index)).float()

        pose = self.poses[index]
        c2w = torch.from_numpy(pose).float()
        w2c = torch.from_numpy(np.linalg.inv(pose)).float()

        ret = {
            "K": self.K,    # [3, 3]
            "c2w": c2w,     # [4, 4]
            "w2c": w2c,     # [4, 4]
            "rgb": rgb,     # [H, W, 3]
            "depth": depth, # [H, W]
            "valid": True,  # bool: replica 有一些视角已经被遮挡，需要跳过
        }
        
        if self.load_sp_feat_flag and self.train:
            sp_feat = self.load_sp_feat(index)
            sp_feat = sp_feat.squeeze().permute(1,2,0).contiguous() # [1, 256, h, w] -> [h, w, 256]
            ret['sp_feature'] = sp_feat      # [H, W, 256]

        if self.load_score_flag and self.train:
            kp_score = self.load_kp_feature_score(index)
            ret['sp_kp_score'] = kp_score      # [H, W]
            ret['sp_kp_mask'] = (kp_score > self.sp_score_thre).int() # [H, W]

        return ret

    def load_all_frame(self):
        self.frame_ids = []
        self.c2w_list = []
        self.rgb_list = []
        self.depth_list = []
        self.K_list = []
        self.scores_list = []

        self.sp_feature_list = []
        self.sp_kp_mask_list = []

        progress_bar = tqdm(range(self.num_imgs), position=0)
        progress_bar.set_description("Load frame")
        
        for i in progress_bar:
            ret = self.get_frame(i)
            c2w = ret['c2w']

            if torch.isnan(c2w).any() or torch.isinf(c2w).any():
                continue

            self.frame_ids.append(i)
            self.c2w_list.append(c2w)
            self.rgb_list.append(ret['rgb'])
            self.depth_list.append(ret['depth'])
            self.K_list.append(ret['K'])

            if self.load_sp_feat_flag:
                self.sp_feature_list.append(ret['sp_feature'])

            if self.load_score_flag:
                self.sp_kp_mask_list.append(ret['sp_kp_mask'])
  
        # convert to tensor
        self.c2w_list = torch.stack(self.c2w_list, dim=0)
        self.rgb_list = torch.stack(self.rgb_list, dim=0)
        self.depth_list = torch.stack(self.depth_list, dim=0)
        self.K_list = torch.stack(self.K_list, dim=0)

        if self.load_sp_feat_flag:
            self.sp_feature_list = torch.stack(self.sp_feature_list, dim=0)

        if self.load_score_flag:
            self.sp_kp_mask_list = torch.stack(self.sp_kp_mask_list, dim=0)
        
        ##################### 检查 shape #####################
        print('*'*20, ' CHECK SHAPE ', '*'*20)
        print('c2w_list: ', self.c2w_list.shape)
        print('rgb_list: ', self.rgb_list.shape)
        print('K_list: ', self.K_list.shape)

        if self.load_sp_feat_flag:
            print('sp_feature_list: ', self.sp_feature_list.shape)

        if self.load_score_flag:
            print('sp_kp_mask_list: ', self.sp_kp_mask_list.shape)
    
    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        ret = self.get_frame(idx)
        return ret


class Scenes12Dataset(torch.utils.data.Dataset):
    def __init__(self, config, train=True):
        super().__init__()
        self.config = config
        self.device = "cuda"
        self.dtype = torch.float32
        self.input_folder = config["Dataset"]["dataset_path"]
        self.scene_name = self.input_folder.split('/')[-2] + '_' + self.input_folder.split('/')[-1]
        self.generated_folder = os.path.join(config["Dataset"]["generated_folder"], self.scene_name.replace('office', 'of'))
        self.train = train
        self.train_step = 5
        self.sp_score_thre = 0.005

        # color name: frame-000000.color.jpg
        split, end = self.parse_split()
        print(self.scene_name, 'split info: test at: ', split, ' end at: ', end) 
        if self.train:
            self.color_paths = []
            for i in range(0, end+1, self.train_step):
                if i > split:
                    self.color_paths.append(os.path.join(self.input_folder, 'data', 'frame-{:0>6d}.color.jpg'.format(i)))
        else:
            self.color_paths = []
            for i in range(split+1):
                self.color_paths.append(os.path.join(self.input_folder, 'data', 'frame-{:0>6d}.color.jpg'.format(i)))

        self.n_img = len(self.color_paths)        
        self.num_imgs = self.n_img

        self.load_sp_feat_flag = False
        self.load_score_flag = True
        # only used for train set
        self.sp_feat_path = os.path.join(self.generated_folder, 'sp_feature')
        self.sp_score_path = os.path.join(self.generated_folder, 'score_map')
        self.sparse_ply = os.path.join(self.generated_folder, 'sp_inloc_pc.ply')
        self.sparse_feature = os.path.join(self.generated_folder, 'sp_inloc_feat.npy')

        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

    def parse_split(self,):
        with open(os.path.join(self.input_folder, 'split.txt')) as f:
            seqs = f.readlines()
            # print(seqs[0].split('='))
            # print(seqs[-1].split('='))
            split = int(seqs[0].replace("\n", "").split('=')[-1][:-1])
            end = int(seqs[-1].replace("\n", "").split('=')[-1][:-1])
            return split, end

    def set_feature_flag(self, value):
        self.load_sp_feat_flag = value
        
    def index_to_name(self, index):
        name = self.color_paths[index].split('/')[-1].split('.')[0] # frame-000000.color.jpg -> frame-000000
        return name
    
    def name_to_index(self, name):
        matching_indices = [i for i, item in enumerate(self.color_paths) if name == item.split('/')[-1]]
        assert len(matching_indices) == 1, ' should be only one match.'
        return matching_indices[0]

    def load_sp_feat(self, index):
        '''
        only used for feature volume reconstruction
        '''
        name = self.index_to_name(index)
        feat = torch.load('{}/{}.pt'.format(self.sp_feat_path, name)) # [1, 256, H, W]
        return feat

    def load_kp_feature_score(self, index):
        name = self.index_to_name(index)
        score = torch.from_numpy(np.load('{}/{}_score.npy'.format(self.sp_score_path, name))) # [H, W]
        return score

    def load_gt_pose(self, index):
        name = self.index_to_name(index)
        pose_path =  os.path.join(self.input_folder, 'data', '{}.pose.txt'.format(name))

        c2w = []
        with open(pose_path, 'r') as f:
            for line in f:
                if 'INF' in line:
                    return np.eye(4), False
                row = line.strip('\n').split()
                row = [float(c) for c in row]
                c2w.append(row)
        c2w = np.array(c2w).astype(np.float32)
        assert c2w.shape == (4,4)

        return c2w, True

    def load_depth(self, index):
        name = self.index_to_name(index)

        file_path =  os.path.join(self.input_folder, 'data', '{}.depth.png'.format(name))
        depth = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        
        # from semantic nerf
        # uint16 mm depth, then turn depth from mm to meter
        depth = depth.astype(np.float32) / self.depth_scale
        return depth

    def load_image(self, index):
        rgb = cv2.imread(self.color_paths[index], -1)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (640, 480), cv2.INTER_AREA)

        return rgb / 255.0
        
    def load_all_depth(self,):
        all_depths = []
        progress_bar = tqdm(range(self.num_imgs), position=0)
        progress_bar.set_description("Load frame depth")
        
        for index in progress_bar:
            depth = self.load_depth(index)
            pose, valid = self.load_gt_pose(index)

            if not valid:
                continue

            all_depths.append(depth)
        
        return np.array(all_depths)

    def get_frame(self, index):
        rgb = torch.from_numpy(self.load_image(index)).float()
        depth = torch.from_numpy(self.load_depth(index)).float()

        pose, valid = self.load_gt_pose(index)
        c2w = torch.from_numpy(pose).float()
        w2c = torch.from_numpy(np.linalg.inv(pose)).float()

        ret = {
            "K": self.K,    # [3, 3]
            "c2w": c2w,     # [4, 4]
            "w2c": w2c,     # [4, 4]
            "rgb": rgb,     # [H, W, 3]
            "depth": depth, # [H, W]
            "valid": valid,  # bool: 12 scenes 有一些frame的pose是nan，需要跳过
        }
                
        if self.load_sp_feat_flag and self.train:
            sp_feat = self.load_sp_feat(index)
            sp_feat = sp_feat.squeeze().permute(1,2,0).contiguous() # [1, 256, h, w] -> [h, w, 256]
            ret['sp_feature'] = sp_feat      # [H, W, 256]

        if self.load_score_flag and self.train:
            kp_score = self.load_kp_feature_score(index)
            ret['sp_kp_score'] = kp_score      # [H, W]
            ret['sp_kp_mask'] = (kp_score > self.sp_score_thre).int() # [H, W]

        return ret

    def load_all_frame(self):
        self.frame_ids = []
        self.c2w_list = []
        self.rgb_list = []
        self.depth_list = []
        self.K_list = []
        self.scores_list = []

        self.sp_feature_list = []
        self.sp_kp_mask_list = []

        progress_bar = tqdm(range(self.num_imgs), position=0)
        progress_bar.set_description("Load frame")
        
        for i in progress_bar:
            ret = self.get_frame(i)
            c2w = ret['c2w']

            if torch.isnan(c2w).any() or torch.isinf(c2w).any():
                continue

            self.frame_ids.append(i)
            self.c2w_list.append(c2w)
            self.rgb_list.append(ret['rgb'])
            self.depth_list.append(ret['depth'])
            self.K_list.append(ret['K'])

            if self.load_sp_feat_flag:
                self.sp_feature_list.append(ret['sp_feature'])

            if self.load_score_flag:
                self.sp_kp_mask_list.append(ret['sp_kp_mask'])
  
        # convert to tensor
        self.c2w_list = torch.stack(self.c2w_list, dim=0)
        self.rgb_list = torch.stack(self.rgb_list, dim=0)
        self.depth_list = torch.stack(self.depth_list, dim=0)
        self.K_list = torch.stack(self.K_list, dim=0)

        if self.load_sp_feat_flag:
            self.sp_feature_list = torch.stack(self.sp_feature_list, dim=0)

        if self.load_score_flag:
            self.sp_kp_mask_list = torch.stack(self.sp_kp_mask_list, dim=0)
        
        ##################### 检查 shape #####################
        print('*'*20, ' CHECK SHAPE ', '*'*20)
        print('c2w_list: ', self.c2w_list.shape)
        print('rgb_list: ', self.rgb_list.shape)
        print('K_list: ', self.K_list.shape)

        if self.load_sp_feat_flag:
            print('sp_feature_list: ', self.sp_feature_list.shape)

        if self.load_score_flag:
            print('sp_kp_mask_list: ', self.sp_kp_mask_list.shape)
    
    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        ret = self.get_frame(idx)

        return ret


def load_dataset(config, train=True):
    if config["Dataset"]["type"] == "replica":
        return ReplicaDataset(config, train)
    elif config["Dataset"]["type"] == "12scenes":
        return Scenes12Dataset(config, train)
    else:
        raise ValueError("Unknown dataset type")
