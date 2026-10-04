import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from diff_gaussian_rasterization_add3 import GaussianRasterizer as GaussianRasterizer_add3

from utils.sh_utils import eval_sh
import time
import torch
import numpy as np
import torch.nn.functional as F


def render_native(
    gs_param,
    H,
    W,
    K,
    T_cw,
    color_mlp=None,
    bg_color=[0.0, 0.0, 0.0],
    scale_factor=1.0,
    opa_replace=None,
    colors_precomp=None,
    add_buffer=None,
    HDR_mode=False
):
    # * Core render interface
    # prepare gs5 param in world system
    if torch.is_tensor(gs_param[0]):  # direct 5 tuple
        mu, rot, s, o, sph = gs_param
    else:
        mu, rot, s, o, sph = gs_param[0]
        for i in range(1, len(gs_param)):
            mu = torch.cat([mu, gs_param[i][0]], 0)
            rot = torch.cat([rot, gs_param[i][1]], 0)
            s = torch.cat([s, gs_param[i][2]], 0)
            o = torch.cat([o, gs_param[i][3]], 0)
            sph = torch.cat([sph, gs_param[i][4]], 0)
    if opa_replace is not None:
        assert isinstance(opa_replace, float)
        o = torch.ones_like(o) * opa_replace


    # cvt to cam frame
    assert T_cw.ndim == 2 and T_cw.shape[1] == T_cw.shape[0] == 4
    R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
    mu_cam = torch.einsum("ij,nj->ni", R_cw, mu) + t_cw[None]
    rot_cam = torch.einsum("ij,njk->nik", R_cw, rot)

    # render
    render_dict = render_cam_pcl(
        mu_cam,
        rot_cam,
        s,
        o,
        sph,
        H,
        W,
        color_mlp=color_mlp,
        CAM_K=K,
        scale_factor=scale_factor,
        bg_color=bg_color,
        colors_precomp=colors_precomp,
        add_buffer=add_buffer,
        HDR_mode=HDR_mode
    )

    return render_dict

def render_cam_pcl(
    xyz,
    rotation,
    scale,
    opacity,
    color_feat,
    H,
    W,
    color_mlp=None,
    # Multiple way to specify camera
    CAM_K=None,
    fx=None,
    fy=None,
    cx=None,
    cy=None,
    verbose=False,
    # active_sph_order=0,
    scale_factor=1.0,
    bg_color=[0.0, 0.0, 0.0],
    add_buffer=None,
    colors_precomp=None,
    HDR_mode=False
):
    # ! 2024.Mar.16, remove the active_sph_order, auto detect
    # ! Camera is at origin, every input is in camera coordinate space
    start_time = time.time()
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

    # ! 2023.12.31 camera different input
    if CAM_K is not None:
        fx, fy, cx, cy = CAM_K[0, 0], CAM_K[1, 1], CAM_K[0, 2], CAM_K[1, 2]
    else:
        assert fx is not None, "fx is not provided"
        if fy is None:
            fy = fx
        if cx is None:
            cx = W // 2
        if cy is None:
            cy = H // 2

    # * Specially handle the non-centered camera, using first padding and finally crop
    # ! fix this bug on 2023.10.28, use abs!!
    if abs(H // 2 - cy) > 1.0 or abs(W // 2 - cx) > 1.0:
        center_handling_flag = True
        left_w, right_w = cx, W - cx
        top_h, bottom_h = cy, H - cy
        new_W = int(2 * max(left_w, right_w))
        new_H = int(2 * max(top_h, bottom_h))
    else:
        center_handling_flag = False
        new_W, new_H = W, H

    # ! 2023.10.27 Fix this bug, change the order, should use the new_W, new_H to compute FoV
    # Set up rasterization configuration
    FoVx = focal2fov(fx, new_W)
    FoVy = focal2fov(fy, new_H)
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    # TODO: Check dynamic gaussian, they use projection matrix to handle non-centered K!, not using the stupid padding
    viewmatrix = torch.from_numpy(
        getWorld2View2(np.eye(3), np.zeros(3)).transpose(0, 1)
    ).to(device)
    projection_matrix = (
        getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy)
        .transpose(0, 1)
        .to(device)
    )
    full_proj_transform = (
        viewmatrix.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = viewmatrix.inverse()[3, :3]

    if add_buffer is not None:
        bg_color = bg_color + [0.0] * add_buffer.shape[-1]

    raster_settings = GaussianRasterizationSettings(
        image_height=new_H,
        image_width=new_W,
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
    # opacity = torch.ones_like(means3D[:, 0]) * sigma
    
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    # JH
    cov3D_precomp = strip_lowerdiag(actual_covariance)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # xyz are in camera frame, so the dir in camera frame is just their normalized direction
    dir_cam = F.normalize(xyz, dim=-1)
    # P_w = Frame @ P_local
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

    
    ret = rasterizer(
        means3D=means3D.float(),
        means2D=means2D.float(),
        shs=None,
        colors_precomp=colors_precomp.float(),  # TODO: support higher dimensional shading
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
        torch.cuda.synchronize()
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

    if center_handling_flag:
        for k in ["rgb", "dep", "alpha", "buf"]:
            if ret[k] is None:
                continue
            if left_w > right_w:
                ret[k] = ret[k][:, :, :W]
            else:
                ret[k] = ret[k][:, :, -W:]
            if top_h > bottom_h:
                ret[k] = ret[k][:, :H, :]
            else:
                ret[k] = ret[k][:, -H:, :]
    # ! to work with GOF, have to handle the dep inside!!
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
