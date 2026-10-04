# the GS controlling model for static scene

# given a colored pcl, construct GS models.

import sys, os, os.path as osp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import colorsys
import numpy as np
import scipy
import torch
from torch import nn
import torch.nn.functional as F

from utils.optim_utils import *
import logging

from pytorch3d.transforms import (
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from pytorch3d.ops import knn_points


def sph_order2nfeat(order):
    return (order + 1) ** 2


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


class StaticGaussian(nn.Module):
    def __init__(
        self,
        position,
        rot,
        scales, 
        features_dc, 
        opacities, 
        max_scale=0.1,  # use sigmoid activation, can't be too large
        min_scale=0.0
    ) -> None:
        super().__init__()

        self.op_update_exclude = []
        
        self.register_buffer("max_scale", torch.tensor(max_scale).squeeze())
        self.register_buffer("min_scale", torch.tensor(min_scale).squeeze())
        self._init_act(self.max_scale, self.min_scale)

        # # * init the parameters
        self._xyz = nn.Parameter(position)
        self._rot = nn.Parameter(rot)
        self._scaling = nn.Parameter(self.s_inv_act(scales))
        self._opacity = nn.Parameter(self.o_inv_act(opacities))
        self._features_dc = nn.Parameter(features_dc)

        # if init_id is None:
        #     init_id = torch.zeros(self.N, dtype=torch.int32)
        # self.register_buffer("group_id", init_id)

        # * init states
        # warning, our code use N, instead of (N,1) as in GS code
        self.register_buffer("xyz_gradient_accum", torch.zeros(self.N).float())
        self.register_buffer("xyz_gradient_denom", torch.zeros(self.N).long())
        self.register_buffer("max_radii2D", torch.zeros(self.N).float())

        # # ! dangerous flags
        # # * for viz the cate color
        # self.return_cate_colors_flag = False

        self.summary()
        return

    @classmethod
    def load_from_ckpt(cls, ckpt):
        position = ckpt["_xyz"]
        rot = ckpt["_rot"]
        scales = ckpt["_scaling"]
        features_dc = ckpt["_features_dc"]
        opacities = ckpt["_opacity"]
        model = cls(
                position = position,
                rot = rot,
                scales = scales,
                features_dc = features_dc,
                opacities = opacities,
        )
        model.load_state_dict(ckpt, strict=True)
        # ! important, must re-init the activation functions
        logging.info(
            f"Resume: Max scale: {model.max_scale}, Min scale: {model.min_scale}"
        )
        model._init_act(model.max_scale, model.min_scale)
        return model

    def summary(self):
        logging.info(f"StaticGaussian: {self.N/1000.0:.1f}K points")
        # logging.info number of parameters per pytorch sub module
        for name, param in self.named_parameters():
            logging.info(f"{name}, {param.numel()/1e6:.3f}M")
        logging.info("-" * 30)
        return

    def _init_act(self, max_s_value, min_s_value):
        max_s_value = max_s_value.item()
        min_s_value = min_s_value.item()

        def s_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            return min_s_value + torch.sigmoid(x) * (max_s_value - min_s_value)

        def s_inv_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            x = torch.clamp(
                x, min=min_s_value + 1e-6, max=max_s_value - 1e-6
            )  # ! clamp
            y = (x - min_s_value) / (max_s_value - min_s_value) + 1e-5
            y = torch.clamp(y, min=1e-5, max=1 - 1e-5)
            y = torch.logit(y)
            if torch.isnan(y).any():
                logging.error(f"{x.min()}, {x.max()}")
                logging.error(f"{y.min()}, {y.max()}")
            assert not torch.isnan(
                y
            ).any(), f"{x.min()}, {x.max()}, {y.min()}, {y.max()}"
            return y
        
        # def s_act(x):
        #     if isinstance(x, float):
        #         x = torch.tensor(x).squeeze()
        #     return torch.exp(x)

        # def s_inv_act(x):
        #     if isinstance(x, float):
        #         x = torch.tensor(x).squeeze()
        #     return torch.log(x)
        
        def o_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            return torch.sigmoid(x)

        def o_inv_act(x):
            if isinstance(x, float):
                x = torch.tensor(x).squeeze()
            return torch.logit(x)

        self.s_act = s_act
        self.s_inv_act = s_inv_act
        self.o_act = o_act
        self.o_inv_act = o_inv_act

        return

    @property
    def device(self):
        return self._xyz.device

    @property
    def N(self):
        try:  # for loading from file dummy init
            return len(self._xyz)
        except:
            return 0

    @property
    def get_position(self):
        return self._xyz

    @property
    def get_quaternion(self):
        return self._rot
    
    @property
    def get_rotation(self):
        return quaternion_to_matrix(self._rot)

    @property
    def get_opacity(self):
        return self.o_act(self._opacity)

    @property
    def get_scaling(self):
        return self.s_act(self._scaling)

    @property
    def get_feat(self):
        return self._features_dc

    # @property
    # def get_group(self):
    #     assert len(self.group_id) == self.N
    #     return self.group_id

    # @torch.no_grad()
    # def get_cate_color(self, color_plate=None, perm=None):
    #     gs_group_id = self.get_group
    #     unique_grouping = torch.unique(gs_group_id).sort()[0]
    #     if not hasattr(self, "group_colors"):
    #         if color_plate is None:
    #             n_cate = len(self.group_id.unique())
    #             hue = np.linspace(0, 1, n_cate + 1)[:-1]
    #             color_plate = torch.Tensor(
    #                 [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hue]
    #             ).to(self.device)
    #         self.group_colors = color_plate
    #         self.group_sphs = RGB2SH(self.group_colors)
    #     if perm is None:
    #         perm = torch.arange(len(unique_grouping))

    #     cate_sph = torch.zeros(self.N, 3).to(self.device)
    #     index_color_map = {}
    #     for ind in perm:
    #         gid = unique_grouping[ind]
    #         cate_sph[gs_group_id == gid] = self.group_sphs[ind].unsqueeze(0)
    #         index_color_map[gid] = self.group_colors[ind]
    #     return cate_sph, index_color_map

    def forward(self):
        xyz = self.get_position
        rot = self.get_rotation
        scale = self.get_scaling
        opa = self.get_opacity
        sph = self.get_feat
    
        # if self.return_cate_colors_flag:
        #     # logging.warning(f"VIZ purpose, return the cate-color")
        #     cate_sph, _ = self.get_cate_color()
        #     sph = torch.zeros_like(sph)
        #     sph[..., :3] = cate_sph  # zero pad

        return xyz, rot, scale, opa, sph

    def get_optimizable_list(
        self,
        lr_p=0.00016,
        lr_q=0.001,
        lr_s=0.005,
        lr_o=0.05,
        lr_sph=0.0025,
        lr_sph_rest=None,
    ):
        lr_sph_rest = lr_sph / 20 if lr_sph_rest is None else lr_sph_rest
        l = [
            {"params": [self._xyz], "lr": lr_p, "name": "xyz"},
            {"params": [self._opacity], "lr": lr_o, "name": "opacity"},
            {"params": [self._scaling], "lr": lr_s, "name": "scaling"},
            {"params": [self._rot], "lr": lr_q, "name": "rot"},
            {"params": [self._features_dc], "lr": lr_sph, "name": "f_dc"},
        ]
        return l

    ######################################################################
    # * Gaussian Control
    ######################################################################

    def record_xyz_grad_radii(self, viewspace_point_tensor_grad, radii, update_filter):
        # Record the gradient norm, invariant across different poses
        assert len(viewspace_point_tensor_grad) == self.N
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor_grad[update_filter, :2], dim=-1, keepdim=False
        )
        self.xyz_gradient_denom[update_filter] += 1
        self.max_radii2D[update_filter] = torch.max(
            self.max_radii2D[update_filter], radii[update_filter]
        )
        return

    def _densification_postprocess(
        self,
        optimizer,
        new_xyz,
        new_r,
        new_s,
        new_o,
        new_sph_dc,
        # new_group_id,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_sph_dc,
            "opacity": new_o,
            "scaling": new_s,
            "rot": new_r,
        }
        d = {k: v for k, v in d.items() if v is not None}

        # First cat to optimizer and then return to self
        optimizable_tensors = cat_tensors_to_optimizer(optimizer, d)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rot = optimizable_tensors["rot"]
        self._features_dc = optimizable_tensors["f_dc"]

        self.xyz_gradient_accum = torch.zeros(self._xyz.shape[0], device=self.device)
        self.xyz_gradient_denom = torch.zeros(self._xyz.shape[0], device=self.device)
        # self.max_radii2D = torch.zeros(self._xyz.shape[0], device=self.device)
        self.max_radii2D = torch.cat(
            [self.max_radii2D, torch.zeros_like(new_xyz[:, 0])], dim=0
        )

        # self.group_id = torch.cat([self.group_id, new_group_id], dim=0)
        return

    def clean_gs_control_record(self):
        self.xyz_gradient_accum = torch.zeros_like(self._xyz[:, 0])
        self.xyz_gradient_denom = torch.zeros_like(self._xyz[:, 0])
        self.max_radii2D = torch.zeros_like(self.max_radii2D)


    def _densify_and_clone(self, optimizer, grad_norm, grad_threshold, scale_th):
        # Extract points that satisfy the gradient condition
        # padding for enabling both call of clone and split
        padded_grad = torch.zeros((self.N), device=self.device)
        padded_grad[: grad_norm.shape[0]] = grad_norm.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= scale_th,
        )
        if selected_pts_mask.sum() == 0:
            return 0

        new_xyz = self._xyz[selected_pts_mask]
        new_rot = self._rot[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        # new_group_id = self.group_id[selected_pts_mask]

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rot,
            new_s=new_scaling,
            new_o=new_opacities,
            new_sph_dc=new_features_dc,
            # new_group_id=new_group_id,
        )

        return len(new_xyz)

    def _densify_and_split(
        self,
        optimizer,
        grad_norm,
        grad_threshold,
        scale_th,
        N=2,
    ):
        # Extract points that satisfy the gradient condition
        _scaling = self.get_scaling
        # padding for enabling both call of clone and split
        padded_grad = torch.zeros((self.N), device=self.device)
        padded_grad[: grad_norm.shape[0]] = grad_norm.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(_scaling, dim=1).values > scale_th,
        )
        if selected_pts_mask.sum() == 0:
            return 0

        stds = _scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = self.get_rotation[selected_pts_mask].repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = _scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        # new_scaling = torch.clamp(new_scaling, max=self.max_scale, min=self.min_scale)
        new_scaling = self.s_inv_act(new_scaling)
        new_rot = self._rot[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1)
        new_opacities = self._opacity[selected_pts_mask].repeat(N, 1)
        # new_group_id = self.group_id[selected_pts_mask].repeat(N)

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rot,
            new_s=new_scaling,
            new_o=new_opacities,
            new_sph_dc=new_features_dc,
            # new_group_id=new_group_id,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(), device=self.device, dtype=bool
                ),
            )
        )
        self._prune_points(optimizer, prune_filter)
        return len(new_xyz)

    
    def append_new_gs(self, optimizer, new_xyz,  new_rot, new_scaling, new_opacities, new_features_dc):
        new_scaling = self.s_inv_act(new_scaling)
        new_opacities = self.o_inv_act(new_opacities)

        self._densification_postprocess(
            optimizer,
            new_xyz=new_xyz,
            new_r=new_rot,
            new_s=new_scaling,
            new_o=new_opacities,
            new_sph_dc=new_features_dc,
            # new_group_id=new_group_id,
        )

    def densify(
        self,
        optimizer,
        max_grad,
        percent_dense,
        extent,
        verbose=True,
    ):
        grads = self.xyz_gradient_accum / self.xyz_gradient_denom
        grads[grads.isnan()] = 0.0

        x_clamped = max(10000, min(self.N, 80000))
        grad_ratio = 0.54 + (x_clamped/10000) * 0.0475
        # grad_ratio = 0.98
        grad_thr = torch.quantile(grads, torch.tensor([grad_ratio]).cuda())
        if grad_thr < max_grad:
            grad_thr = max_grad

        n_clone = self._densify_and_clone(
            optimizer, grads, grad_thr, percent_dense * extent
        )
        n_split = self._densify_and_split(
            optimizer, grads, grad_thr, percent_dense * extent, N=2
        )

        # drop_mask = torch.rand(self.N, device="cuda") < 0.01
        # self._prune_points(optimizer, drop_mask)
        if verbose:
            logging.info(f"Densify: Clone[+] {n_clone}, Split[+] {n_split//2}")
            # logging.info(f"Densify: Clone[+] {n_clone}")
        torch.cuda.empty_cache()
        return

    def prune_points(
        self,
        optimizer,
        min_opacity,
        extent,
        max_screen_size,
        verbose=True,
    ):
        opacity = self.o_act(self._opacity)
        prune_mask = (opacity < min_opacity).squeeze()
        logging.info(f"opacity_pruning {prune_mask.sum()}")
        if max_screen_size:  # if a point is too large
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)
            logging.info(f"radii2D_pruning {big_points_vs.sum()}")
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_ws)
            # * reset the maxRadii
            self.max_radii2D = torch.zeros_like(self.max_radii2D)
        self._prune_points(optimizer, prune_mask)
        if verbose:
            logging.info(f"Prune: {prune_mask.sum()}")

    def _prune_points(self, optimizer, mask):
        valid_points_mask = ~mask
        optimizable_tensors = prune_optimizer(
            optimizer,
            valid_points_mask,
            exclude_names=self.op_update_exclude,
        )

        self._xyz = optimizable_tensors["xyz"]
        if getattr(self, "color_memory", None) is None:
            self._features_dc = optimizable_tensors["f_dc"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rot = optimizable_tensors["rot"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_denom = self.xyz_gradient_denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        torch.cuda.empty_cache()
        # self.group_id = self.group_id[valid_points_mask]
        return

    def reset_opacity(self, optimizer, value=0.01, verbose=True):
        opacities_new = self.o_inv_act(
            torch.min(self.o_act(self._opacity), torch.ones_like(self._opacity) * value)
        )
        optimizable_tensors = replace_tensor_to_optimizer(
            optimizer, opacities_new, "opacity"
        )
        if verbose:
            logging.info(f"Reset opacity to {value}")
        self._opacity = optimizable_tensors["opacity"]

  