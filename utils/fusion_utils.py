import numpy as np
from numba import njit, prange
from skimage import measure
import torch

# https://github.com/alibaba-damo-academy/former3d/blob/main/former3d/tsdf_fusion.py

def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud."""
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
    """Get corners of 3D camera view frustum of depth image"""
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array(
        [
            (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2])
            * np.array([0, max_depth, max_depth, max_depth, max_depth])
            / cam_intr[0, 0],
            (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2])
            * np.array([0, max_depth, max_depth, max_depth, max_depth])
            / cam_intr[1, 1],
            np.array([0, max_depth, max_depth, max_depth, max_depth]),
        ]
    )
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts


def meshwrite(filename, verts, faces, norms, colors):
    """Save a 3D mesh to a polygon .ply file."""
    # Write header
    ply_file = open(filename, "w")
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write(
            "%f %f %f %f %f %f %d %d %d\n"
            % (
                verts[i, 0],
                verts[i, 1],
                verts[i, 2],
                norms[i, 0],
                norms[i, 1],
                norms[i, 2],
                colors[i, 0],
                colors[i, 1],
                colors[i, 2],
            )
        )

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()


def pcwrite(filename, xyzrgb):
    """Save a point cloud to a polygon .ply file."""
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename, "w")
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write(
            "%f %f %f %d %d %d\n"
            % (
                xyz[i, 0],
                xyz[i, 1],
                xyz[i, 2],
                rgb[i, 0],
                rgb[i, 1],
                rgb[i, 2],
            )
        )


def integrate(
    depth_im,
    color_im,
    feat_im,
    cam_intr,
    cam_pose,
    obs_weight,
    world_c,
    vox_coords,
    weight_vol,
    tsdf_vol,
    color_vol,
    feat_vol,
    sdf_trunc,
    im_h,
    im_w,
):
    # Convert world coordinates to camera coordinates
    world2cam = torch.inverse(cam_pose).float()
    cam_c = torch.matmul(world2cam, world_c.transpose(1, 0)).transpose(1, 0).float()

    # Convert camera coordinates to pixel coordinates
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    pix_z = cam_c[:, 2]
    pix_x = torch.round((cam_c[:, 0] * fx / cam_c[:, 2]) + cx).long()
    pix_y = torch.round((cam_c[:, 1] * fy / cam_c[:, 2]) + cy).long()

    # Eliminate pixels outside view frustum
    valid_pix = ((pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0))
    valid_vox_x = vox_coords[valid_pix, 0]
    valid_vox_y = vox_coords[valid_pix, 1]
    valid_vox_z = vox_coords[valid_pix, 2]
    depth_val = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

    # print('depth_val: ', depth_val.shape)

    # Integrate tsdf
    depth_diff = depth_val - pix_z[valid_pix]
    dist = torch.clamp(depth_diff / sdf_trunc, max=1)
    valid_pts = (depth_val > 0) & (depth_diff >= -sdf_trunc)
    valid_vox_x = valid_vox_x[valid_pts]
    valid_vox_y = valid_vox_y[valid_pts]
    valid_vox_z = valid_vox_z[valid_pts]
    valid_dist = dist[valid_pts]
    w_old = weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    tsdf_vals = tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]

    w_new = w_old + obs_weight
    tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = (w_old * tsdf_vals + obs_weight * valid_dist) / w_new
    weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = w_new

    # Integrate color
    old_color = color_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    new_color = color_im[pix_y[valid_pix][valid_pts], pix_x[valid_pix][valid_pts]]

    # print('w_old: ', w_old.shape)            # [N]
    # print('old_color: ', old_color.shape)    # [N, 3]
    # print('new_color: ', new_color.shape) 
    # print('w_new: ', w_new.shape)

    tmp = torch.clamp(torch.round((w_old.unsqueeze(-1) * old_color + obs_weight * new_color) / w_new.unsqueeze(-1)), 0, 255)
    color_vol[valid_vox_x, valid_vox_y, valid_vox_z, :] = tmp

    # Integrate feature
    new_feat = feat_im[pix_y[valid_pix][valid_pts], pix_x[valid_pix][valid_pts]]
    old_feat = feat_vol[valid_vox_x, valid_vox_y, valid_vox_z]
    feat_vol[valid_vox_x, valid_vox_y, valid_vox_z, :] = torch.clamp((w_old.unsqueeze(-1) * old_feat + obs_weight * new_feat) / w_new.unsqueeze(-1), 0, 255)

    return weight_vol, tsdf_vol, color_vol, feat_vol

class TSDFVolumeTorch:
    """Volumetric TSDF Fusion of RGB-D Images."""

    def __init__(self, voxel_dim, origin, voxel_size, feat_dim, margin=3):
        """Constructor.
        Args:
          vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
            xyz bounds (min/max) in meters.
          voxel_size (float): The volume discretization in meters.
        """
        self.device = torch.device("cpu")
        # self.device = torch.device("cuda")

        # Define voxel volume parameters
        self._voxel_size = float(voxel_size)
        self._sdf_trunc = margin * self._voxel_size
        self._integrate_func = integrate
        self._feat_dim = feat_dim

        # Adjust volume bounds
        self._vol_dim = voxel_dim.long()
        self._vol_origin = origin
        self._num_voxels = torch.prod(self._vol_dim).item()

        # Get voxel grid coordinates
        xv, yv, zv = torch.meshgrid(
            torch.arange(0, self._vol_dim[0]),
            torch.arange(0, self._vol_dim[1]),
            torch.arange(0, self._vol_dim[2]),
            indexing='ij'
        )
        self._vox_coords = (torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1).long().to(self.device))

        # Convert voxel coordinates to world coordinates
        self._world_c = self._vol_origin + (self._voxel_size * self._vox_coords)
        self._world_c = torch.cat(
            [self._world_c, torch.ones(len(self._world_c), 1, device=self.device)],
            dim=1,
        ).float()

        self.reset()

        # print("[*] voxel volume: {} x {} x {}".format(*self._vol_dim))
        # print("[*] num voxels: {:,}".format(self._num_voxels))

    def reset(self):
        self._tsdf_vol = torch.ones(*self._vol_dim).to(self.device)
        self._weight_vol = torch.zeros(*self._vol_dim).to(self.device)
        self._color_vol = torch.zeros((*self._vol_dim, 3)).to(self.device)
        self._feat_vol = torch.zeros((*self._vol_dim, self._feat_dim)).to(self.device)

    def integrate(self, depth_im, color_im, feat_im, cam_intr, cam_pose, obs_weight=1.0):
        """Integrate an RGB-D frame into the TSDF volume.
        Args:
          color_im (ndarray): An RGB image of shape (H, W, 3).
          depth_im (ndarray): A depth image of shape (H, W).
          cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
          cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
          obs_weight (float): The weight to assign to the current observation.
        """
        cam_pose = cam_pose.float().to(self.device)
        cam_intr = cam_intr.float().to(self.device)
        color_im = color_im.float().to(self.device)
        depth_im = depth_im.float().to(self.device)
        feat_im = feat_im.float().to(self.device)
        im_h, im_w = depth_im.shape
        weight_vol, tsdf_vol, color_vol, feat_vol = self._integrate_func(
            depth_im,
            color_im,
            feat_im,
            cam_intr,
            cam_pose,
            obs_weight,
            self._world_c,
            self._vox_coords,
            self._weight_vol,
            self._tsdf_vol,
            self._color_vol,
            self._feat_vol,
            self._sdf_trunc,
            im_h,
            im_w,
        )
        self._weight_vol = weight_vol
        self._tsdf_vol = tsdf_vol
        self._color_vol = color_vol
        self._feat_vol = feat_vol

    def get_mesh(self):
        """Compute a mesh from the voxel volume using marching cubes."""
        tsdf_vol, color_vol, weight_vol, feat_vol = self.get_volume()

        verts, faces, norms, vals = measure.marching_cubes(tsdf_vol.numpy())
        # verts, faces, norms, vals = measure.marching_cubes_lewiner(tsdf_vol.numpy(), level=0)
        verts_ind = np.round(verts).astype(int)
        verts = (verts * self._voxel_size + self._vol_origin.numpy())  # voxel grid coordinates to world coordinates

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]].numpy()

        # Get vertex features
        feats = feat_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]].numpy()

        colors = np.floor(rgb_vals)
        # print('colors:', colors.shape)
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors, feats


    def get_volume(self):
        return self._tsdf_vol, self._color_vol, self._weight_vol, self._feat_vol

    def save(self, path):
        vols = {
            'tsdf': self._tsdf_vol,
            'weight': self._weight_vol,
            'color': self._color_vol,
            'feat': self._feat_vol,
        }

        torch.save(vols, path)
    
    def load(self, path):
        dicts = torch.load(path)

        self._tsdf_vol = dicts['tsdf']
        self._weight_vol = dicts['weight']
        self._color_vol = dicts['color']
        self._feat_vol = dicts['feat']

    @property
    def sdf_trunc(self):
        return self._sdf_trunc

    @property
    def voxel_size(self):
        return self._voxel_size