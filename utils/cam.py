import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F

def show_img(title, img_rgb):  # img - rgb image
    img_bgr = rgb2bgr(img_rgb)
    cv2.imshow(title, img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_POI(img_rgb, DEBUG=False): # img - RGB image in range 0...255
    img = np.copy(img_rgb)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img_gray, None)
    if DEBUG:
        img = cv2.drawKeypoints(img_gray, keypoints, img)
        show_img("Detected points", img)
    xy = [keypoint.pt for keypoint in keypoints]
    xy = np.array(xy).astype(int)
    # Remove duplicate points
    xy_set = set(tuple(point) for point in xy)
    xy = np.array([list(point) for point in xy_set]).astype(int)
    return xy # pixel coordinates


def transpose(R, t, X):
    """
    Pytorch batch version of computing transform of the 3D points
    :param R: rotation matrix in dimension of (N, 3, 3) or (3, 3)
    :param t: translation vector could be (N, 3, 1) or (3, 1)
    :param X: points with 3D position, a 2D array with dimension of (N, num_points, 3) or (num_points, 3)
    :return: transformed 3D points
    """
    keep_dim_n = False
    keep_dim_hw = False
    if R.dim() == 2:
        keep_dim_n = True
        R = R.unsqueeze(0)
        t = t.unsqueeze(0)
    if X.dim() == 2:
        X = X.unsqueeze(0)

    if X.dim() == 4:
        assert X.size(3) == 3
        keep_dim_hw = True
        N, H, W = X.shape[:3]
        X = X.view(N, H*W, 3)

    N = R.shape[0]
    M = X.shape[1]
    X_after_R = torch.bmm(R, torch.transpose(X, 1, 2))
    X_after_R = torch.transpose(X_after_R, 1, 2)
    trans_X = X_after_R + t.view(N, 1, 3).expand(N, M, 3)

    if keep_dim_hw:
        trans_X = trans_X.view(N, H, W, 3)
    if keep_dim_n:
        trans_X = trans_X.squeeze(0)

    return trans_X

def pi(K, X):
    """
    Projecting the X in camera coordinates to the image plane
    :param K: camera intrinsic matrix tensor (N, 3, 3) or (3, 3)
    :param X: point position in 3D camera coordinates system, is a 3D array with dimension of (N, num_points, 3), or (num_points, 3)
    :return: N projected 2D pixel position u (N, num_points, 2) and the depth X (N, num_points, 1)
    """
    keep_dim_n = False
    keep_dim_hw = False
    if K.dim() == 2:
        keep_dim_n = True
        K = K.unsqueeze(0)      # make dim (1, 3, 3)
    if X.dim() == 2:
        X = X.unsqueeze(0)      # make dim (1, num_points, 3)
    if X.dim() == 4:
        assert X.size(3) == 3
        keep_dim_hw = True
        N, H, W = X.shape[:3]
        X = X.view(N, H*W, 3)

    assert K.size(0) == X.size(0)
    N = K.shape[0]

    fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
    u_x = fx * X[:, :, 0:1] / (X[:, :, 2:3] + 0.0001) + cx
    u_y = fy * X[:, :, 1:2] / (X[:, :, 2:3] + 0.0001) + cy
    u = torch.cat([u_x, u_y], dim=-1)
    d = X[:, :, 2:3]

    if keep_dim_hw:
        u = u.view(N, H, W, 2)
        d = d.view(N, H, W)
    if keep_dim_n:
        u = u.squeeze(0)
        d = d.squeeze(0)

    return u, d

def pi_inv(K, x, d):
    """
    Projecting the pixel in 2D image plane and the depth to the 3D point in camera coordinate.
    :param x: 2d pixel position, a 2D array with dimension of (N, num_points, 2)
    :param d: depth at that pixel, a array with dimension of (N, num_points, 1)
    :param K: camera intrinsic matrix tensor (N, 3, 3)
    :return: 3D point in camera coordinate (N, num_points, 3)
    """
    keep_dim_n = False
    keep_dim_hw = False
    if K.dim() == 2:
        keep_dim_n = True
        K = K.unsqueeze(0)      # make dim (1, 3, 3)
    if x.dim() == 2:
        x = x.unsqueeze(0)      # make dim (1, num_points, 3)
    if d.dim() == 2:
        d = d.unsqueeze(0)      # make dim (1, num_points, 1)

    if x.dim() == 4:
        assert x.size(0) == d.size(0)
        assert x.size(1) == d.size(1)
        assert x.size(2) == d.size(2)
        assert x.size(3) == 2
        keep_dim_hw = True
        N, H, W = x.shape[:3]
        x = x.view(N, H*W, 2)
        d = d.view(N, H*W, 1)

    N = K.shape[0]
    fx, fy, cx, cy = K[:, 0:1, 0:1], K[:, 1:2, 1:2], K[:, 0:1, 2:3], K[:, 1:2, 2:3]
    X_x = d * (x[:, :, 0:1] - cx) / fx
    X_y = d * (x[:, :, 1:2] - cy) / fy
    X_z = d
    X = torch.cat([X_x, X_y, X_z], dim=-1)

    if keep_dim_hw:
        X = X.view(N, H, W, 3)
    if keep_dim_n:
        X = X.squeeze(0)

    return X

def skew_symmetric(v):
    """
    Convert a 3D vector to a skew-symmetric matrix
    """
    zero = torch.zeros_like(v[..., 0])
    return torch.stack((zero, -v[..., 2], v[..., 1], 
                        v[..., 2], zero, -v[..., 0], 
                        -v[..., 1], v[..., 0], zero), dim=-1).reshape(v.shape[:-1] + (3, 3))

def exp_map_so3(so3):
    """
    Convert a so(3) representation to a SO(3) representation
    """
    theta = torch.norm(so3, dim=-1, keepdim=True)
    normalized_so3 = so3 / theta
    cross_product_matrix = skew_symmetric(normalized_so3)
    
    I = torch.eye(3, device=so3.device).reshape((3, 3))
    theta = theta.reshape(1, 1)
    
    R = I + torch.sin(theta) * cross_product_matrix + (1 - torch.cos(theta)) * cross_product_matrix @ cross_product_matrix
    return R

def log_map_so3(R):
    """
    Convert a SO(3) representation to a so(3) representation
    """
    cos_theta = (torch.trace(R) - 1) / 2
    # Clamp cos(theta) to handle numerical issues
    cos_theta = torch.clamp(cos_theta, -1, 1)

    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)

    # Avoid division by zero
    small_angle = torch.isclose(theta, torch.tensor(0.0, device=theta.device), atol=1e-5)
    not_small_angle = ~small_angle

    theta_safe = torch.where(small_angle, torch.tensor(1.0, device=theta.device), theta)
    sin_theta_safe = torch.where(small_angle, torch.tensor(1.0, device=theta.device), sin_theta)

    log_R = torch.zeros_like(R)
    log_R[not_small_angle] = theta[not_small_angle] / (2 * sin_theta_safe[not_small_angle]) * (R[not_small_angle] - R[not_small_angle].transpose(-1, -2))
    log_R[small_angle] = 0.5 * (R[small_angle] - R[small_angle].transpose(-1, -2))

    return torch.stack((log_R[..., 2, 1], log_R[..., 0, 2], log_R[..., 1, 0]), dim=-1)

def batch_SO3_log_np(R: np.array):
    assert len(R.shape) == 3
    θ = np.arccos(0.5 * (np.trace(R, axis1=1, axis2=-1) - 1.0)).reshape([-1, 1, 1])
    S = θ * (R - np.transpose(R, [0, 2, 1])) / (2.0 * np.sin(θ))
    w1 = S[:, 2, 1].reshape([-1, 1])
    w2 = S[:, 0, 2].reshape([-1, 1])
    w3 = S[:, 1, 0].reshape([-1, 1])
    w = np.concatenate([w1, w2, w3], axis=-1).reshape([-1, 3])

    return w

def batch_SO3_exp_np(R_vec: np.array):
    assert len(R_vec.shape) == 2
    θ = np.linalg.norm(R_vec, axis=-1).reshape([-1, 1, 1])

    wx = R_vec[:, 0]
    wy = R_vec[:, 1]
    wz = R_vec[:, 2]

    S = np.zeros([R_vec.shape[0], 3, 3])
    S[:, 0, 1] = -wz
    S[:, 0, 2] = wy
    S[:, 1, 0] = wz
    S[:, 1, 2] = -wx
    S[:, 2, 0] = -wy
    S[:, 2, 1] = wx
    
    R = np.eye(3).reshape([1, 3, 3]).repeat(R_vec.shape[0], axis=0) \
      + S * np.sin(θ) / θ \
      + S @ S * (1.0 - np.cos(θ)) / θ**2

    return R

def quaternion_to_rotation(quaternion):
    '''
    The input quaternion is of the format [w, x, y, z] and the output
    is a 3X3 rotation matrix corresponding to the input quaternion.
    '''
    rotation = torch.zeros((3, 3), device=quaternion.device)
    q_0 = quaternion[0]
    q_1 = quaternion[1]
    q_2 = quaternion[2]
    q_3 = quaternion[3]

    rotation[0, 0] = 1 - (2 * q_2**2) - (2 * q_3**2)
    rotation[0, 1] = (2 * q_1 * q_2) - (2 * q_0 * q_3)
    rotation[0, 2] = (2 * q_1 * q_3) + (2 * q_0 * q_2)
    rotation[1, 0] = (2 * q_1 * q_2) + (2 * q_0 * q_3)
    rotation[1, 1] = 1 - (2 * q_1**2) - (2 * q_3**2)
    rotation[1, 2] = (2 * q_2 * q_3) - (2 * q_0 * q_1)
    rotation[2, 0] = (2 * q_1 * q_3) - (2 * q_0 * q_2)
    rotation[2, 1] = (2 * q_2 * q_3) + (2 * q_0 * q_1)
    rotation[2, 2] = 1 - (2 * q_1**2) - (2 * q_2**2)

    return rotation

def rotation_6d_to_matrix(cam_tensor):
    a1, a2 = cam_tensor[:3], cam_tensor[3:6]
    if isinstance(cam_tensor, torch.Tensor):
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        R = torch.stack((b1, b2, b3), dim=-2)
    else:
        b1 = a1 / np.linalg.norm(a1, axis=-1)
        b2 = a2 - (b1 * a2).sum(-1) * b1
        b2 = b2 / np.linalg.norm(b2, axis=-1)
        b3 = np.cross(b1, b2, axis=-1)
        R = np.stack((b1, b2, b3), axis=-2)

    return R