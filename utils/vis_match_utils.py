# https://github.com/zju3dv/pats/blob/master/demo.py

import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch.utils.data.dataset import Dataset
import argparse
import os
import yaml
import random
import cv2
from torch.utils.data.dataloader import DataLoader


def _resize_img(depth, shape=np.array([640, 480])):
    w = depth.shape[1]
    h = depth.shape[0]
    w_new = shape[0]
    h_new = shape[1]
    if w/w_new < h/h_new:
        gap = int((h - w / w_new * h_new) / 2)
        crop_img = depth[gap:h - gap, :, :]
    else:
        gap = int((w - h / h_new * w_new) / 2)
        crop_img = depth[:, gap:w - gap, :]
    resize_img = cv2.resize(
        crop_img, tuple(shape))
    return resize_img



def Resize_img(left):
    size = 1600
    h_l, w_l = left.shape[:2]
    max_shape_l = max(h_l, w_l)
    size_l = size / max_shape_l
    left = _resize_img(left, np.array([int(w_l * size_l), int(h_l * size_l)]))
    h_l, w_l = left.shape[:2]
    h_l2 = h_l // 32 * 32 + (1 - int(h_l % 32 == 0)) * 32
    w_l2 = w_l // 32 * 32 + (1 - int(w_l % 32 == 0)) * 32

    max_width = w_l2
    max_height = h_l2
    left = cv2.copyMakeBorder(left, 0, max_height - h_l, 0, max_width - w_l, cv2.BORDER_CONSTANT, None, 0)
    return left




class VideoDataLoader(Dataset):
    # the folder should look like:
    # -- data_path
    #  -- 001.png
    #  -- 002.png
    def __init__(self, data_path):
        self.data_path = data_path
        images_list = sorted(os.listdir(data_path))
        self.source_path = f'{data_path}/{images_list[0]}'
        self.target_list = [f'{data_path}/{image}' for image in images_list[1:]]
 
    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, index):
        left = cv2.imread(self.source_path)
        right = cv2.imread(self.target_list[index])
        left = Resize_img(left)
        right = Resize_img(right)
        source_name = os.path.basename(self.source_path).split('.')[0]
        target_name = os.path.basename(self.target_list[index]).split('.')[0]
        data = {
            'image0': left,
            'image1': right,
            'pair_name': f'{source_name}_{target_name}'
        }

        return data


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image

def coord_trans(u, v):
    rad = np.sqrt(np.square(u) + np.square(v))
    u /= (rad+1e-3)
    v /= (rad+1e-3)
    return u, v

def kp_color(u, v, resolution):
    h, w = resolution
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    xx, yy = coord_trans(xx, yy)
    vis = flow_uv_to_colors(xx, yy)

    color = vis[v.astype(np.int32), u.astype(np.int32)]
    return color

def draw_kp(img, kps, colors):
    for i, kp in enumerate(kps):
        # img = cv2.circle(img, (int(kp[1]), int(kp[0])), radius=3, color=colors[i].tolist(), thickness=-1, lineType=cv2.LINE_AA)
        img = cv2.circle(img, (int(kp[1]), int(kp[0])), radius=3, color=(255,0,0), thickness=-1, lineType=cv2.FILLED) # red
    return img

def draw_matches(img, kps1, kps2):
    for i, kp in enumerate(kps1):
        cv2.line(img, (int(kps1[i][1]), int(kps1[i][0])), (int(kps2[i][1]), int(kps2[i][0])), (0,255,0), 1) 
    return img

def vis_matches(image0, image1, kp0, kp1):
    lh, lw = image0.shape[:2]
    rh, rw = image1.shape[:2]
    mask1 = np.logical_and.reduce(np.array((kp0[:,1]>=0, kp0[:,1]<lw, kp0[:,0]>=0, kp0[:,0]<lh)))
    mask2 = np.logical_and.reduce(np.array((kp1[:,1]>=0, kp1[:,1]<rw, kp1[:,0]>=0, kp1[:,0]<rh)))

    mask = np.logical_and.reduce(np.array((mask1, mask2)))
    kp0 = kp0[mask]
    kp1 = kp1[mask]

    color = kp_color(kp0[:,1], kp0[:,0], (lh, lw))
    color = kp_color(kp0[:,1], kp0[:,0], (lh, lw))

    image0 = draw_kp(image0, kp0, color)
    image1 = draw_kp(image1, kp1, color)

    pad_width = 5
    zero_image = np.zeros([lh, pad_width, 3])
    vis = np.concatenate([image0, zero_image, image1], axis=1)

    # 连线
    kp1[:,1] += lw + pad_width
    vis = draw_matches(vis, kp0, kp1)

    return vis

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)

    param = parser.parse_args()
    if param.config is not None:
        with open(param.config, 'r') as f:
            yaml_dict = yaml.safe_load(f)
            for k, v in yaml_dict.items():
                param.__dict__[k] = v

    # initialize random seed
    torch.manual_seed(param.seed)
    np.random.seed(param.seed)
    random.seed(param.seed)
    model = PATS(param)
    model.load_state_dict()
    model = model.cuda().eval()


    save_path = 'results'
    os.system(f'mkdir -p {save_path}')
    dataset = VideoDataLoader(param.data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    for data in tqdm(loader):
        data['image0'] = data['image0'].cuda()
        data['image1'] = data['image1'].cuda()
        pair_name = data['pair_name'][0]
        with torch.no_grad():
            result = model(data)

        kp0_numpy = result["matches_l"].cpu().numpy()
        kp1_numpy = result["matches_r"].cpu().numpy()
        image0_numpy = data["image0"][0].cpu().numpy()
        image1_numpy = data["image1"][0].cpu().numpy()

        vis_img = vis_matches(image0_numpy, image1_numpy, kp0_numpy, kp1_numpy)
        cv2.imwrite(f'{save_path}/{pair_name}.png', vis_img)