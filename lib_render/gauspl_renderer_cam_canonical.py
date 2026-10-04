import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import math
from diff_gaussian_rasterization_cam import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from diff_gaussian_rasterization_cam_add3 import GaussianRasterizer as GaussianRasterizer_add3

from utils.sh_utils import eval_sh
import time
import torch
import numpy as np
import torch.nn.functional as F


def render_cam_canonical(
    xyz,
    rotation,
    scale,
    opacity,
    color_feat,
    H,
    W,
    color_mlp=None,
    add_buffer=None,
    verbose=False,
    scale_factor=1.0,
    bg_color=[0.0, 0.0, 0.0],
    colors_precomp=None,
    HDR_mode=False
):
    # ! 2024.Mar.16, remove the active_sph_order, auto detect
    # ! Camera is at origin, every input is in camera coordinate space

    S = torch.zeros_like(rotation)
    S[:, 0, 0] = scale[:, 0]
    S[:, 1, 1] = scale[:, 1]
    S[:, 2, 2] = scale[:, 2]
    actual_covariance = rotation @ (S**2) @ rotation.permute(0, 2, 1)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    device = xyz.device
    screenspace_points = (
        torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device=xyz.device)
        + 0
    )
    # screenspace_points.retain_grad()
    try:
        screenspace_points.retain_grad()
    except:
        pass


    # In camera canonical space, camera intrisic is not needed.
    # So we set fx=W and fy=H at will, just to be compatible with CUDA Rasterization
    fx, fy = W, H
    FoVx = focal2fov(fx, W)
    FoVy = focal2fov(fy, H)
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    viewmatrix = torch.from_numpy(
        getWorld2View2(np.eye(3), np.zeros(3)).transpose(0, 1)
    ).to(device)

    full_proj_transform = viewmatrix.clone()
    camera_center = viewmatrix.inverse()[3, :3]
    
    if add_buffer is not None:
        bg_color = bg_color + [0.0] * add_buffer.shape[-1]

    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.tensor(bg_color, dtype=torch.float32, device=device),
        scale_modifier=scale_factor,
        viewmatrix=viewmatrix,
        projmatrix=full_proj_transform,
        sh_degree=0,  # ! use pre-compute color!
        campos=camera_center,
        prefiltered=False,
        debug=False,
    )

    if add_buffer is not None:
        rasterizer = GaussianRasterizer_add3(raster_settings=raster_settings)
    else:
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    # JH
    cov3D_precomp = strip_lowerdiag(actual_covariance)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    dir_cam = torch.zeros_like(xyz).cuda()
    dir_cam[:, 2] = 1.0
    dir_local = torch.einsum("nji,nj->ni", rotation, dir_cam)  # note the transpose
    dir_local = F.normalize(
        dir_local, dim=-1
    )  # If frame is not SO(3) but Affinity, have to normalize

    if HDR_mode:
        log_gauss_rgb_h = color_mlp(torch.cat([color_feat, dir_local], dim=-1))
        gauss_rgb_h = torch.pow(torch.tensor(2).cuda(), log_gauss_rgb_h)
        if torch.isinf(gauss_rgb_h.mean()).sum() > 0:
            print("inf", log_gauss_rgb_h.max(), gauss_rgb_h.mean());exit()
        if colors_precomp is None:
            colors_precomp = gauss_rgb_h
        else:
            assert colors_precomp.shape == gauss_rgb_h.shape
    else:
        shs_view = color_feat.reshape(len(color_feat), 3, -1)  # N, Channels, Deg
        _deg = shs_view.shape[-1]
        if _deg == 1:
            active_sph_order = 0
        elif _deg == 4:
            active_sph_order = 1
        elif _deg == 9:
            active_sph_order = 2
        elif _deg == 16:
            active_sph_order = 3
        else:
            raise ValueError(f"Unexpected SH degree: {_deg}")
        sh2rgb = eval_sh(active_sph_order, shs_view, dir_local)
        sh2rgb = torch.clamp_min(sh2rgb + 0.5, 0.0)
        if colors_precomp is None:
            colors_precomp = sh2rgb
        else:
            assert colors_precomp.shape == sh2rgb.shape

    if add_buffer is not None:
        colors_precomp = torch.cat([colors_precomp, add_buffer], -1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).

    start_time = time.time()
    ret = rasterizer(
        means3D=means3D.float(),
        means2D=means2D.float(),
        shs=None,
        colors_precomp=colors_precomp.float(),  
        opacities=opacity.float(),
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp.float(),
    )
    if len(ret) == 2:
        rendered_image, radii = ret
        depth, alpha = None, None
    elif len(ret) == 4:
        rendered_image, radii, depth, alpha = ret
    else:
        raise ValueError(f"Unexpected return value from rasterizer with len={len(ret)}")
    if verbose:
        print(
            f"render time: {(time.time() - start_time)*1000:.3f}ms",
        )

    if add_buffer is not None:
        buf = rendered_image[3:]
    else:
        buf = None

    log_gs_rgb_h = color_mlp(torch.cat([color_feat, dir_local.detach()], dim=-1)) if HDR_mode else None

    ret = {
        "rgb": rendered_image[:3],
        "buf": buf,
        "dep": depth,
        "alpha": alpha,
        "log_gs_rgb_h": log_gs_rgb_h,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

    ret["dep"] = ret["dep"] / torch.clamp(ret["alpha"], 1e-6, 1.0)
    return ret


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty
