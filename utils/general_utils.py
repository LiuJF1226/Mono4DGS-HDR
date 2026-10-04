#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
import matplotlib.cm as cm
import torch.nn as nn
import math

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):

        points_warp = torch.matmul(T.unsqueeze(0), points)[:, :3, :]
        points_warp = points_warp.view(self.batch_size, 3, self.height, self.width)
        src_depth_warp = points_warp[:, 2:]

        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)

        proj_mask_x = (pix_coords[:, 0:1] >= 0) & (pix_coords[:, 0:1] <= self.width-1)
        proj_mask_y = (pix_coords[:, 1:2] >= 0) & (pix_coords[:, 1:2] <= self.height-1)
        inside_mask = proj_mask_x & proj_mask_y

        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords, src_depth_warp, inside_mask
    

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def batched_J_world(xyz_cam, H, W, CAM_K=None, fx=None, fy=None):
    if CAM_K is not None:
        fx, fy = CAM_K[0, 0], CAM_K[1, 1]
    else:
        assert fx is not None, "fx is not provided"
        if fy is None:
            fy = fx
    FoVx = focal2fov(fx, W)
    FoVy = focal2fov(fy, H)
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    limx = 1.3 * tanfovx
    limy = 1.3 * tanfovy

    z = xyz_cam[:, 2].clamp(min=1e-6) 
    txtz = xyz_cam[:, 0] / z
    tytz = xyz_cam[:, 1] / z
    xyz_cam[:, 0] = torch.clamp(txtz, -limx, limx) * z
    xyz_cam[:, 1] = torch.clamp(tytz, -limy, limy) * z
    fx_over_z = fx / z  # [N]
    fy_over_z = fy / z  # [N]
    x_over_z2 = xyz_cam[:, 0] / (z * z)  # [N]
    y_over_z2 = xyz_cam[:, 1] / (z * z)  # [N]

    J_world = torch.zeros(xyz_cam.shape[0], 3, 3, device=xyz_cam.device, dtype=torch.float32)
    J_world[:, 0, 0] = fx_over_z  # ∂u/∂X
    J_world[:, 0, 2] = -fx * x_over_z2  # ∂u/∂Z
    J_world[:, 1, 1] = fy_over_z  # ∂v/∂Y
    J_world[:, 1, 2] = -fy * y_over_z2  # ∂v/∂Z

    return J_world

def batched_quaternion_average(q: torch.Tensor) -> torch.Tensor:
    r"""Batched quaternion average based on Markley's method.
    
    Args:
        q: Tensor of shape [N, T, 4] where:
           - N: number of Gaussians
           - T: number of timesteps
           - 4: quaternion components (w,x,y,z)
           Each quaternion is assumed to be unit length.

    Returns:
        Tensor of shape [N, 4] containing averaged quaternions for each Gaussian.
    """
    # Handle antipodal configuration (q and -q represent same rotation)
    q[q[:, :, 0] < 0] *= -1  # Flip quaternions where w < 0
    
    # Reshape for batch computation: [N, T, 4, 1] @ [N, T, 1, 4] -> [N, T, 4, 4]
    q_outer = torch.matmul(q.unsqueeze(-1), q.unsqueeze(-2))
    
    # Compute mean of outer products along time dimension: [N, 4, 4]
    M = q_outer.mean(dim=1)
    
    # Compute eigenvectors (using torch.linalg.eigh for symmetric matrices)
    # Returns eigenvalues in ascending order, so take last eigenvector
    _, eigenvectors = torch.linalg.eigh(M)  # [N, 4, 4]
    avg_q = eigenvectors[:, :, -1]  # Take eigenvector for largest eigenvalue
    
    # Final antipodal handling
    avg_q[avg_q[:, 0] < 0] *= -1
    
    return avg_q


def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    import os; os.environ['PYTHONHASHSEED'] = str(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)


def vis_depth(depth):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    percentile = 97
    eps = 1e-10

    lo_auto, hi_auto = weighted_percentile(
        depth, np.ones_like(depth), [50 - percentile / 2, 50 + percentile / 2])
    lo = None or (lo_auto - eps)
    hi = None or (hi_auto + eps)
    curve_fn = lambda x: 1/(x+eps)
    depth, lo, hi = [curve_fn(x) for x in [depth, lo, hi]]
    depth = np.nan_to_num(
            np.clip((depth - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))
    
    colorized = cm.get_cmap('turbo')(depth)[:, :, :3]

    return np.uint8(colorized[..., ::-1] * 255)