import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d

class Autoencoder_dataset(Dataset):
    def __init__(self, ply_dir, feat_dir):
        self.pcds = torch.from_numpy(np.asarray(o3d.io.read_point_cloud(ply_dir).points)) 
        self.features = torch.from_numpy(np.load(feat_dir)) # [N, 256]
        print('3D Feature shape: ', self.features.shape)

    def __getitem__(self, index):
        xyz = self.pcds[index]
        feat = self.features[index]
        return xyz, feat

    def __len__(self):
        return self.features.shape[0] 