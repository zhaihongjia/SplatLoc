import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
import time
from argparse import ArgumentParser
from datetime import datetime
import random
import yaml
from munch import munchify
import cv2
import open3d as o3d
import imgviz
import numpy as np
from tqdm import tqdm
import torch

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.eval_utils import save_gaussians

# for mapping
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.logging_utils import Log
from utils.utils import get_loss_mapping
from utils.camera_utils import Camera
from utils.eval_utils import save_gaussians
from OpenGL import GL as gl
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import matplotlib.pyplot as plt

def get_loss_marker(config, pred, gt):
    pred = torch.sigmoid(pred.view(-1))
    gt = gt.view(-1).float()
    loss = torch.nn.functional.binary_cross_entropy(pred, gt, reduction='mean')
    return loss

def get_loss_descriptor(config, pred, gt):
    pred = pred.view(-1, 3)
    gt = gt.view(-1, 3)
    sim = torch.cosine_similarity(pred, gt, dim=1)
    loss = 1 - sim.mean()
    return loss

class SplatLoc:
    def __init__(self, config, save_dir=None):
        self.config = config
        self.save_dir = save_dir
        self.device = "cuda"
        self.iteration_count = 0

        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (model_params, opt_params, pipeline_params,)
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        # init params
        self.viewpoints = {}
        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(opt_params)
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.scaling_slider = 1.0
        self.cameras_extent = 6.0

        self.dataset = load_dataset(config=config)

        # debug and eval flags
        self.save_debug = config['Results']['save_debug']
        self.kf_inter = self.config["Training"]["kf_interval"]
        self.primitive_reg = config["Training"]["primitive_reg"]

        self.set_hyperparams()

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (self.cameras_extent * self.config["Training"]["gaussian_extent"])
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]

    def debug(self, viewpoint):
        '''
        rendering: rgb, depth, opacity, elipsoid
        '''
        # name = self.dataset.index_to_name(viewpoint.uid)
        save_id = viewpoint.uid

        debug_save_path = os.path.join(self.save_dir, 'rendering')
        rgb_save_path = os.path.join(debug_save_path, 'rgb')
        depth_save_path = os.path.join(debug_save_path, 'depth')
        opacity_save_path = os.path.join(debug_save_path, 'opacity')
        # elipsoid_save_path = os.path.join(debug_save_path, 'elipsoid')

        os.makedirs(rgb_save_path, exist_ok=True)
        os.makedirs(depth_save_path, exist_ok=True)
        os.makedirs(opacity_save_path, exist_ok=True)

        rendering_data = render(
                viewpoint,
                self.gaussians,
                self.pipeline_params,
                self.background,
                self.scaling_slider,)

        # rgb
        rgb = ((torch.clamp(rendering_data["render"], min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        cv2.imwrite(os.path.join(rgb_save_path, 'rgb_{}.png'.format(save_id)), rgb[:,:,::-1])

        # depth
        depth = rendering_data["depth"]
        depth = depth[0, :, :].detach().cpu().numpy()
        max_depth = np.max(depth)
        depth = imgviz.depth2rgb(depth, min_value=0.1, max_value=max_depth, colormap="jet")
        depth = torch.from_numpy(depth)
        depth = torch.permute(depth, (2, 0, 1)).float()
        depth = (depth).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        cv2.imwrite(os.path.join(depth_save_path, 'depth_{}.png'.format(save_id)), depth[:,:,::-1])

        # opacity
        opacity = rendering_data["opacity"]
        opacity = opacity[0, :, :].detach().cpu().numpy()
        max_opacity = np.max(opacity)
        opacity = imgviz.depth2rgb(opacity, min_value=0.0, max_value=max_opacity, colormap="jet")
        opacity = torch.from_numpy(opacity)
        opacity = torch.permute(opacity, (2, 0, 1)).float()
        opacity = (opacity).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        cv2.imwrite(os.path.join(opacity_save_path, 'opacity_{}.png'.format(save_id)), opacity[:,:,::-1])

    def log(self, viewpoint):
        '''
        rgb, depth, elipsoid_chbox
        '''
        # 保存渲染和原始的 rgb， depth等信息，feature
        score_save_path = os.path.join(self.save_dir, 'score')
        depth_save_path = os.path.join(self.save_dir, 'depth')

        os.makedirs(score_save_path, exist_ok=True)
        os.makedirs(depth_save_path, exist_ok=True)

        rgb = viewpoint.original_image.permute(1,2,0).squeeze().cpu().numpy() * 255
        score = viewpoint.kp_score.squeeze().cpu().numpy()
        depth = viewpoint.depth
        name = self.dataset.index_to_name(viewpoint.uid)

        kp_mask = (score > 0.005).astype(float)
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        kp_mask = cv2.dilate(kp_mask, kernel)

        rgb[kp_mask > 0] = [0, 0, 255]
        cv2.imwrite(os.path.join(score_save_path, 'score_{}.png'.format(name.replace('rgb_', '').replace('.color', ''))), rgb[:,:,::-1])

        # depth
        ax = plt.subplot(111)
        ax.axis('off')
        ax.imshow(depth, cmap="plasma", vmin=0, vmax=np.max(depth))
        plt.tight_layout()
        plt.savefig(os.path.join(depth_save_path, 'depth_{}.png'.format(name.replace('rgb_', '').replace('.color', ''))), bbox_inches='tight', pad_inches=0.0)

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        '''
        添加关键帧，此处进行高斯的初始化
        '''
        self.gaussians.extend_from_pcd_seq(viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map)

    def map(self, iters=1):
        # all viewpoints        
        all_viewpoint_stack = []
        frames_to_optimize = self.window_size # self.config["Training"]["pose_window"]
        for cam_idx, viewpoint in self.viewpoints.items():
            all_viewpoint_stack.append(viewpoint)
        
        # mapping: random select views for optimization
        for _ in range(iters):
            self.iteration_count += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []

            for cam_idx in torch.randperm(len(all_viewpoint_stack))[:frames_to_optimize]:
                viewpoint = all_viewpoint_stack[cam_idx]
                render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
                (
                    image,
                    marker,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                ) = (
                    render_pkg["render"],
                    render_pkg["kp_prob"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                )
                loss_mapping += get_loss_mapping(self.config, image, depth, viewpoint, opacity)
                loss_mapping += get_loss_marker(self.config, marker, viewpoint.kp_score)
                
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
            
            # scaling loss
            scaling = self.gaussians.get_scaling
            score = self.gaussians.get_marker.detach() # [N, 1]
            mask = score.cpu().squeeze() > 0.005
            isotropic_loss = torch.abs(scaling.mean(dim=1).view(-1, 1)[mask] / (0.02 * (1 - score[mask])) - 1)
            if self.primitive_reg:
                loss_mapping += 0.01 * isotropic_loss.mean()
            loss_mapping.backward()

            # fix keyprimitive gs
            if self.primitive_reg:
                key_mask = self.gaussians.get_marker.detach().cpu().squeeze() > 0.005
                self.gaussians.get_xyz.grad[key_mask] = 0

            gaussian_split = False

            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(viewspace_point_tensor_acm[idx], visibility_filter_acm[idx])

                # update gaussian
                update_gaussian = (self.iteration_count % self.gaussian_update_every == self.gaussian_update_offset)
                if update_gaussian:
                    print('-------------> update_gaussian: ')
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (not update_gaussian):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True
                
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)

    def color_refinement(self):
        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(random.randint(0, len(viewpoint_idx_stack) - 1))
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(viewpoint_cam, self.gaussians, self.pipeline_params, self.background)
            
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (Ll1) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()

            # zhj: fix
            if self.primitive_reg:
                key_mask = self.gaussians.get_marker.detach().squeeze() > 0.005
                self.gaussians.get_xyz.grad[key_mask] = 0

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter],)
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)

    def load_depth(self, cur_frame_idx):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        viewpoint = self.viewpoints[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]

        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def do_recon(self):
        projection_matrix = getProjectionMatrix2(
                    znear=0.01,
                    zfar=100.0,
                    fx=self.dataset.fx,
                    fy=self.dataset.fy,
                    cx=self.dataset.cx,
                    cy=self.dataset.cy,
                    W=self.dataset.width,
                    H=self.dataset.height,
                ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)

        # for all kfs in dataset
        for cur_frame_idx in range(0, len(self.dataset), self.kf_inter):
            print('Recon: ', cur_frame_idx)
            viewpoint = Camera.init_from_dataset(self.dataset, cur_frame_idx, projection_matrix)
            viewpoint.compute_grad_mask(self.config)

            self.viewpoints[cur_frame_idx] = viewpoint

            depth_map = self.load_depth(cur_frame_idx)
            self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)

            frames_to_optimize = self.config["Training"]["window_size"]
            iter_per_kf = self.mapping_itr_num
            
            self.map(iters=iter_per_kf)

            if self.save_debug:
                self.debug(viewpoint)
                # self.log(viewpoint)

        self.color_refinement()

        # debug rendering results
        # for test_id in range(0, len(self.dataset), self.kf_inter):
        #     ret = self.dataset[test_id]
        #     valid = ret['valid']
        #     if valid:
        #         viewpoint = Camera.init_from_dataset(self.dataset, test_id, projection_matrix)
        #         viewpoint.compute_grad_mask(self.config)
        #         self.debug(viewpoint)

        # save ply file for test
        save_gaussians(self.gaussians, self.save_dir, "final", final=True)        


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    args = parser.parse_args(sys.argv[1:])

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])        
        path = config["Dataset"]["dataset_path"].split("/")
        if config['Dataset']['type'] == 'replica':
            save_dir = os.path.join(config["Results"]["save_dir"], path[-2], path[-1])
        elif config['Dataset']['type'] == '12scenes':
            save_dir = os.path.join(config["Results"]["save_dir"], path[-3], path[-2] + '_' + path[-1])
        else:
            print('Dataset type should be replica or 12scenes')
            exit()
        
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)

    splatloc = SplatLoc(config, save_dir=save_dir)
    splatloc.do_recon()
