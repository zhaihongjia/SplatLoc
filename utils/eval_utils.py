import json
import os

import cv2
import evo
import numpy as np
import torch
from torch.nn import functional as F
import math

from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from utils.logging_utils import Log

import open3d as o3d
import matplotlib.pyplot as plt

def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
):
    # evaluation interval
    interval = 1
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")

    for idx in range(0, len(frames), interval):
        frame = frames[idx]
        ret = dataset[idx]
        valid = ret['valid']
        
        if not valid:
            continue

        gt_image, gt_depth, c2w = ret['rgb'], ret['depth'], ret['c2w']
        gt_image = gt_image.permute(2, 0, 1).cuda()

        render_results = render(frame, gaussians, pipe, background)
        rendering = render_results["render"]
        image = torch.clamp(rendering, 0.0, 1.0)
        mask = gt_image.cpu() > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    psnr_file = os.path.join(save_dir, "eval_rendering.txt")

    f = open(psnr_file, 'w')
    f.write('mean_psnr: {}\n'.format(output["mean_psnr"]))
    f.write('mean_ssim: {}\n'.format(output["mean_ssim"]))
    f.write('mean_lpips: {}'.format(output["mean_lpips"]))
    f.close()

    return output

# compute angle of two qunaternions
def compute_quaternion_dist(pred_quant, gt_quant, eps=1e-7):
    # (B, 4)
    assert len(pred_quant.shape) == 2
    assert len(gt_quant.shape) == 2
    assert pred_quant.shape[1] == gt_quant.shape[1]

    B, D = pred_quant.shape
    d = torch.abs(torch.bmm(pred_quant.view(B, 1, D), gt_quant.view(B, D, 1)))
    # https://github.com/pytorch/pytorch/issues/8069
    d = torch.where(d > 1.0 - eps, torch.full_like(d, 1.0 - eps), d)
    d = torch.where(d < -1.0 + eps, torch.full_like(d, -1.0 + eps), d)
    theta = 2 * torch.acos(d) * 180 / math.pi
    return theta

# convert SO3 to quaternion
def SO3_to_quat(SO3):
    # (B, 3, 3) -> (B, 4)
    # https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
    scale1 = 1.0 + SO3[:, 0, 0] - SO3[:, 1, 1] - SO3[:, 2, 2]
    quat1 = torch.stack([SO3[:, 1, 2] - SO3[:, 2, 1],
                         scale1,
                         SO3[:, 0, 1] + SO3[:, 1, 0],
                         SO3[:, 2, 0] + SO3[:, 0, 2]],
                        dim=-1) * 0.5 / torch.sqrt(scale1).unsqueeze(-1)

    scale2 = 1.0 - SO3[:, 0, 0] + SO3[:, 1, 1] - SO3[:, 2, 2]
    quat2 = torch.stack([SO3[:, 2, 0] - SO3[:, 0, 2],
                         SO3[:, 0, 1] + SO3[:, 1, 0],
                         scale2,
                         SO3[:, 1, 2] + SO3[:, 2, 1]],
                        dim=-1) * 0.5 / torch.sqrt(scale2).unsqueeze(-1)

    scale3 = 1.0 - SO3[:, 0, 0] - SO3[:, 1, 1] + SO3[:, 2, 2]
    quat3 = torch.stack([SO3[:, 0, 1] - SO3[:, 1, 0],
                         SO3[:, 2, 0] + SO3[:, 0, 2],
                         SO3[:, 1, 2] + SO3[:, 2, 1],
                         scale3],
                        dim=-1) * 0.5 / torch.sqrt(scale3).unsqueeze(-1)

    scale4 = 1.0 + SO3[:, 0, 0] + SO3[:, 1, 1] + SO3[:, 2, 2]
    quat4 = torch.stack([scale4,
                         SO3[:, 1, 2] - SO3[:, 2, 1],
                         SO3[:, 2, 0] - SO3[:, 0, 2],
                         SO3[:, 0, 1] - SO3[:, 1, 0]],
                        dim=-1) * 0.5 / torch.sqrt(scale4).unsqueeze(-1)

    cond1 = torch.logical_and(SO3[:, 2, 2] < 0, SO3[:, 0, 0] > SO3[:, 1, 1]).unsqueeze(-1).expand(-1, 4)
    cond2 = torch.logical_and(SO3[:, 2, 2] < 0, SO3[:, 0, 0] <= SO3[:, 1, 1]).unsqueeze(-1).expand(-1, 4)
    cond3 = torch.logical_and(SO3[:, 2, 2] >= 0, SO3[:, 0, 0] < -SO3[:, 1, 1]).unsqueeze(-1).expand(-1, 4)

    quat = quat4
    quat = torch.where(cond1, quat1, quat)
    quat = torch.where(cond2, quat2, quat)
    quat = torch.where(cond3, quat3, quat)

    quat = F.normalize(quat, dim=1)
    return quat

def eval_pose(eval_Rs, eval_ts, gt_Rs, gt_ts, show_results=False):
    '''
    input tensor : [1, 3, 3]
    '''
    gt_quats = SO3_to_quat(gt_Rs).float()
    eval_quats = SO3_to_quat(eval_Rs).float()
    thetas = compute_quaternion_dist(gt_quats, eval_quats)
    dists = (eval_ts - gt_ts).norm(dim=-1)
    
    if show_results:
        print('translation_error: ', dists)
        print('rotation_error: ', thetas)
    return thetas, dists

def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


def makePlyFile(xyzs, rgbs, fileName='makeply.ply'):
    '''Make a ply file for open3d.visualization.draw_geometries
    :param xyzs:    numpy array of point clouds 3D coordinate, shape (numpoints, 3).
    :param labels:  numpy array of point label, shape (numpoints, ).
    '''
    with open(fileName, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment PCL generated\n')
        f.write('element vertex {}\n'.format(len(xyzs)))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for i in range(len(xyzs)):
            r, g, b = rgbs[i]
            x, y, z = xyzs[i]
            f.write('{} {} {} {} {} {}\n'.format(x, y, z, int(r), int(g), int(b)))