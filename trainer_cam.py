import os, glob, math, random, logging
import os.path as osp
import time
import numpy as np
import torch, torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from simple_knn._C import distCUDA2
from lib_prior.prior_loading import Saved2D
from mosca_precompute import auto_get_depth_dir_tap_mode
from scene.dynamic_gs import DynamicGaussian
from scene.tone_mapper import ToneMapper
from lib_moca.camera import MonocularCameras
from utils.optim_utils import *
from tqdm import tqdm
from lib_render.gauspl_renderer_cam_canonical import render_cam_canonical
from utils.loss_utils import compute_rgb_loss, compute_dep_loss, depth_correlation_loss, draw_CRF, unit_expos_loss, CRF_monotonic_loss, CRF_RGB_equal_loss, ssim, compute_HDR_TAE
from utils.geometry_utils import cal_connectivity_from_points, cal_arap_error, cal_smooth_error
from matplotlib import cm
import imageio, json
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from utils.image_utils import psnr
from lpipsPyTorch import lpips

tonemapReinhard = cv2.createTonemapReinhard(2.2, 0.5, 0.5 ,0)
tonemap_mu = lambda x : (torch.log(torch.clip(x,0,1) * 5000 + 1 ) / np.log(5000 + 1))

class CamGSTrainer:
    def __init__(self, args):
        self.args = args
        self.load_prior()
        self.tone_mapper_loss_gt = (0.5)**(1/2.2) if "syn" in self.args.ws else 0.5


        self.out_dir = osp.join(self.args.ws, self.args.exp_name, "cam_gs")

        os.makedirs(self.out_dir, exist_ok=True)

        self.color_feature_dim = 36
        self.cam_d_model = self.get_init_dyn_gs_from_tracks(tracks_3d=self.cam_tracks_info["tracks_3d"], base_time_idx=0, frames_per_control_point=self.args.frames_per_control_point).cuda()
        self.tone_mapper = self.get_tone_mapper(hidden=128, act="sp", pretrain_iters=1000).cuda()
        self.color_mlp = nn.Sequential(
                            nn.Linear(self.color_feature_dim+3, self.color_feature_dim),
                            nn.ReLU(True),
                            nn.Linear(self.color_feature_dim, 3),
                            # nn.Sigmoid()
                        ).cuda()

        draw_CRF(self.tone_mapper, self.out_dir)
       
        self.optimizer_cfg = OptimCFG(
            lr_cam_f=0.0,
            lr_cam_q=0.00003,
            lr_cam_t=0.00003,
            # gs
            lr_p=getattr(self.args, "photo_lr_p", 0.00016),
            lr_q=getattr(self.args, "photo_lr_q", 0.001),
            lr_s=getattr(self.args, "photo_lr_s", 0.005),
            lr_o=getattr(self.args, "photo_lr_o", 0.05),
            lr_sph=getattr(self.args, "photo_lr_sph", 0.0025),
            lr_sph_rest_factor=getattr(self.args, "photo_lr_sph_rest_factor", 20.0),
            lr_p_final=getattr(self.args, "photo_lr_p_final", 0.00016 / 100),
            # node
            lr_np=getattr(self.args, "photo_lr_np", 0.00016),
            lr_nq=getattr(self.args, "photo_lr_nq", 0.00016),
            lr_nsig=getattr(self.args, "photo_lr_nsig", 0.003),
            lr_np_final=getattr(self.args, "photo_lr_np_final", 0.00016 / 100.0),
            lr_nq_final=getattr(self.args, "photo_lr_nq_final", 0.00016 / 100.0),
            lr_w=getattr(self.args, "photo_lr_w", 0.0),
            lr_w_final=getattr(self.args, "photo_lr_w_final", None),
        )

        self.optimizer_tone_mapper = torch.optim.Adam(list(self.tone_mapper.parameters()), lr=5e-4, eps=1e-15)
        self.optimizer_color_mlp = torch.optim.Adam(list(self.color_mlp.parameters()), lr=5e-4, eps=1e-15)
        self.optimizer_cam_d_model = torch.optim.Adam(
            self.cam_d_model.get_optimizable_list(**self.optimizer_cfg.get_dynamic_lr_dict)
        )

        self.gs_model_scheduling_func_dict, _, _ = (
                self.optimizer_cfg.get_scheduler(total_steps=getattr(self.args, "photo_cam_total_steps", 10000))
            )

        self.total_steps=getattr(self.args, "photo_cam_total_steps", 20000)
        self.cam_d_ctrl_start = getattr(self.args, "photo_cam_d_ctrl_start", 500)
        self.cam_d_ctrl_end = getattr(self.args, "photo_cam_d_ctrl_end", 500)
        self.cam_d_gs_ctrl_cfg = GSControlCFG(
            densify_steps=getattr(self.args, "photo_cam_d_ctrl_densify_steps", 400),
            reset_steps=getattr(self.args, "photo_cam_d_ctrl_reset_steps", 1001),
            prune_steps=getattr(self.args, "photo_cam_d_ctrl_prune_steps", 200),
            densify_max_grad=getattr(self.args, "photo_cam_d_ctrl_densify_max_grad", 0.0002),  
            densify_percent_dense=getattr(self.args, "photo_cam_d_ctrl_densify_percent_dense", 0.01),
            prune_opacity_th=getattr(self.args, "photo_cam_d_ctrl_prune_opacity_th", 0.05),
            reset_opacity=getattr(self.args, "photo_cam_d_ctrl_reset_opacity", 0.01),
        )

        self.per_view_dict = {}
        self.full_dict = {}

        # self.tone_mapper = self.get_tone_mapper(hidden=128, act="sp", pretrain_iters=0).cuda()
        # self.tone_mapper.load_state_dict(torch.load(osp.join(self.out_dir, "ckpt", "tone_mapper.pth")))
        # self.cam_d_model = DynamicGaussian.load_from_ckpt(torch.load(osp.join(self.out_dir, "ckpt", "cam_d_model.pth"))).cuda()

        # with torch.no_grad():
        #     self.render_train()
        #     self.eval_ldr("train_ldr")
        #     if hasattr(self.s2d, "train_hdr_gt"):
        #         self.eval_hdr("train_hdr")
        #     if hasattr(self.s2d, "test_ldr_gt"):
        #         self.render_test()
        #         self.eval_ldr("test_ldr")
        #         if hasattr(self.s2d, "test_ne_ldr_gt"):
        #             self.eval_ldr("test_ne_ldr")
        #         if hasattr(self.s2d, "test_hdr_gt"):
        #             self.eval_hdr("test_hdr")           

        # with open(self.out_dir + "/results.json", 'a') as fp:
        #     json.dump(self.full_dict, fp, indent=True)
        # with open(self.out_dir + "/per_view.json", 'a') as fp:
        #     json.dump(self.per_view_dict, fp, indent=True)
        # exit()


    def load_prior(self):
        DEPTH_BOUNDARY_TH = getattr(self.args, "depth_boundary_th", 1.0)
        DEP_MEDIAN = getattr(self.args, "dep_median", 1.0)
        # DEPTH_DIR, TAP_MODE = auto_get_depth_dir_tap_mode(self.args.ws, self.args)
        DEPTH_MODE = self.args.dep_mode
        TAP_MODE = self.args.tap_mode
        DEPTH_DIR = f"{DEPTH_MODE}_depth"
        FLOW_MODE = self.args.flow_mode
        UNIFORM_TAP_MODE = f"dep={DEPTH_MODE}_{TAP_MODE}_tap"
        self.bundle_path = f"bundle_{self.args.dep_mode}_{self.args.flow_mode}_{self.args.tap_mode}"
        self.s2d = (
                Saved2D(self.args.ws)
                .load_epi(f"epi_{FLOW_MODE}")
                .load_dep(DEPTH_DIR, DEPTH_BOUNDARY_TH)
                .normalize_depth(median_depth=DEP_MEDIAN)
                .recompute_dep_mask(depth_boundary_th=DEPTH_BOUNDARY_TH)
                .load_track(UNIFORM_TAP_MODE, min_valid_cnt=getattr(self.args, "tap_loading_min_valid_cnt", 4))
                .rescale_perframe_depth_from_bundle(osp.join(self.args.ws, self.bundle_path, "bundle.pth"))
                .load_flow(f"flow_{FLOW_MODE}")
                .set_epi_mask_to_s2d(epi_th=getattr(self.args, "epi_th", 0.00005))
                .cuda()
            )

        tracks_3d = self.s2d.track.detach().clone().permute(1,0,2)
        visibles = self.s2d.track_mask.detach().clone().permute(1,0)
        tracks_3d[..., 0] = 2 * tracks_3d[..., 0] / self.s2d.W - 1
        tracks_3d[..., 1] = 2 * tracks_3d[..., 1] / self.s2d.H - 1
        self.cam_tracks_info = {
            "tracks_3d": tracks_3d,  # [N, T, 3]
            "visibles": visibles,  # [N, T]
        }


    def get_init_dyn_gs_from_tracks(self, tracks_3d, base_time_idx=0, poly_feature_dim=4, frames_per_control_point=1):
        ## tracks_3d are in camera canonical space
        tracks_3d = tracks_3d.permute(1,0,2)  # [T, N, 3]
        tracks_3d = torch.stack([x[~torch.isnan(x).any(dim=1)] for x in tracks_3d], dim=0)
        
        N = tracks_3d.shape[1]
        base_position, pos_cubic_node, control_points_idx = self.get_dyn_gs_pos_cubic(tracks_3d, base_time_idx, frames_per_control_point)

        # base_rot = torch.zeros((N, 4), dtype=torch.float32, device="cuda")
        # base_rot[:, 0] = 1
        rot_poly_feat = torch.zeros((N, poly_feature_dim, 4), dtype=torch.float32, device="cuda")
        rot_poly_feat[:, 0, 0] = 1
        rot_poly_feat = rot_poly_feat.reshape(N, -1)
        

        avg_dist = torch.clamp_min(distCUDA2(base_position.float().cuda()), 0.0000001)[..., None]
        scales = torch.sqrt(avg_dist).repeat(1, 3)
        
        features_dc = torch.randn(N, self.color_feature_dim).float().cuda()
        opacities = 0.5 * torch.ones((N, 1), dtype=torch.float, device="cuda")

        d_model = DynamicGaussian(base_position, pos_cubic_node, control_points_idx, rot_poly_feat, scales, features_dc, opacities, self.s2d.T, max_scale=getattr(self.args, "cam_gs_radius_max", 0.1), min_scale=0.0, base_time_idx=base_time_idx)

        return d_model
    
    def get_dyn_gs_pos_cubic(self, tracks_3d, base_time_idx=0, frames_per_control_point=1):
        T, N = tracks_3d.shape[0:2]
        base_position = tracks_3d[base_time_idx]
        delta_position = tracks_3d - base_position[None, ...]   # T, N, 3
        
        control_points_num = ((T - 1) // frames_per_control_point) + 1
        control_points_idx = torch.linspace(0, T-1, control_points_num).long().unique().cuda()
      
        # Use control-point positions as the initial values
        pos_cubic_node = delta_position[control_points_idx]  # [control_points_num, N, 3]
        pos_cubic_node = pos_cubic_node.permute(1, 0, 2).reshape(N, -1)  # [N, control_points_num * 3]

        # if control_points_num < T: 
        #     try:
        #         all_frame_idx = torch.arange(T, device=tracks_3d.device)
        #         indices = torch.searchsorted(control_points_idx, all_frame_idx, right=False) - 1
        #         left_indices = torch.clamp(indices, min=0).detach()
        #         right_indices = torch.clamp(left_indices + 1, max=control_points_num-1).detach()
                
        #         left_times = control_points_idx[left_indices]
        #         right_times = control_points_idx[right_indices]
                
        #         u = (all_frame_idx - left_times).float() / (right_times - left_times).float()
        #         u_2 = u * u
        #         u_3 = u_2 * u
                
        #         # Hermite basis functions
        #         h00 = 2 * u_3 - 3 * u_2 + 1  # start-point position
        #         h01 = -2 * u_3 + 3 * u_2     # end-point position
        #         h10 = u_3 - 2 * u_2 + u      # start-point derivative
        #         h11 = u_3 - u_2              # end-point derivative
                
        #         # Augmented coefficient matrix over all basis functions.
        #         # Each control point needs two indices:
        #         # 1. its own position
        #         # 2. its left/right neighbors, used to estimate the derivative
                
        #         # Position coefficient matrix A of shape (T, control_points_num)
        #         A = torch.zeros((T, control_points_num), device=tracks_3d.device)
        #         row_indices = torch.arange(T, device=tracks_3d.device)
                
        #         # Position terms: standard h00 and h01
        #         A[row_indices, left_indices] += h00  # start-point position
        #         A[row_indices, right_indices] += h01 # end-point position
                
        #         # Derivative terms, estimated by central differences at control points.
        #         # A derivative is a weighted combination of a control point and its neighbors.
                
        #         # Derivative weights at each control point, written as a linear combination of neighbors
        #         dt = (right_times - left_times).float()  # time gap, used to scale the derivative
                
        #         # Derivative contribution of every left control point, including boundaries
        #         if True:  # always run; interior and boundary points are handled separately
        #             # 1. Interior control points (central difference)
        #             left_internal = (left_indices > 0) & (left_indices < control_points_num - 1)
        #             if left_internal.any():
        #                 # Interior derivative is determined by the previous and next points
        #                 left_prev_indices = left_indices[left_internal] - 1
        #                 left_next_indices = left_indices[left_internal] + 1
                        
        #                 # Weights for the derivative (central difference)
        #                 dt_left_prev = control_points_idx[left_indices[left_internal]] - control_points_idx[left_prev_indices]
        #                 dt_left_next = control_points_idx[left_next_indices] - control_points_idx[left_indices[left_internal]]
        #                 dt_left_total = dt_left_prev + dt_left_next
                        
        #                 # Apply the left-derivative contribution
        #                 # The derivative is a weighted average of the previous and next points
        #                 weight_prev = dt_left_next / dt_left_total
        #                 weight_next = dt_left_prev / dt_left_total
                        
        #                 derivative_scale = 0.5 * dt[left_internal]  # scale factor
                        
        #                 # Apply h10 (start-point derivative)
        #                 # The derivative is a weighted combination of the previous and next points
        #                 h10_internal = h10[left_internal] * derivative_scale
        #                 A[row_indices[left_internal], left_prev_indices] -= weight_prev * h10_internal
        #                 A[row_indices[left_internal], left_next_indices] += weight_next * h10_internal
                    
        #             # 2. Left boundary (forward difference)
        #             left_boundary = left_indices == 0
        #             if left_boundary.any():
        #                 # Boundary derivative uses a forward difference
        #                 left_next_indices = torch.ones_like(left_indices[left_boundary])  # index 1
        #                 dt_left = control_points_idx[left_next_indices] - control_points_idx[0]
                        
        #                 # Apply the derivative contribution
        #                 derivative_scale = dt[left_boundary]  # boundary points use a larger scale
        #                 h10_boundary = h10[left_boundary] * derivative_scale
                        
        #                 # The boundary derivative comes entirely from the next point
        #                 A[row_indices[left_boundary], 0] += 0.0  # no self contribution
        #                 A[row_indices[left_boundary], left_next_indices] += 1.0 * h10_boundary  # positive contribution from the next point
                    
        #             # 3. Right boundary when the left point is the last control point
        #             left_right_boundary = left_indices == control_points_num - 1
        #             if left_right_boundary.any():
        #                 # Boundary derivative uses a backward difference
        #                 left_prev_indices = torch.ones_like(left_indices[left_right_boundary]) * (control_points_num - 2)
        #                 dt_right = control_points_idx[control_points_num-1] - control_points_idx[left_prev_indices]
                        
        #                 # Apply the derivative contribution
        #                 derivative_scale = dt[left_right_boundary]  # boundary points use a larger scale
        #                 h10_right_boundary = h10[left_right_boundary] * derivative_scale
                        
        #                 # The boundary derivative comes entirely from the previous point
        #                 A[row_indices[left_right_boundary], left_prev_indices] -= 1.0 * h10_right_boundary  # negative contribution from the previous point
        #                 A[row_indices[left_right_boundary], control_points_num-1] += 0.0  # no self contribution
                
        #         # Derivative contribution of every right control point, including boundaries
        #         if True:  # always run; interior and boundary points are handled separately
        #             # 1. Interior control points (central difference)
        #             right_internal = (right_indices > 0) & (right_indices < control_points_num - 1)
        #             if right_internal.any():
        #                 # Interior derivative is determined by the previous and next points
        #                 right_prev_indices = right_indices[right_internal] - 1
        #                 right_next_indices = right_indices[right_internal] + 1
                        
        #                 # Weights for the derivative
        #                 dt_right_prev = control_points_idx[right_indices[right_internal]] - control_points_idx[right_prev_indices]
        #                 dt_right_next = control_points_idx[right_next_indices] - control_points_idx[right_indices[right_internal]]
        #                 dt_right_total = dt_right_prev + dt_right_next
                        
        #                 # Apply the right-derivative contribution
        #                 weight_prev = dt_right_next / dt_right_total
        #                 weight_next = dt_right_prev / dt_right_total
                        
        #                 derivative_scale = 0.5 * dt[right_internal]  # scale factor
                        
        #                 # Apply h11 (end-point derivative)
        #                 h11_internal = h11[right_internal] * derivative_scale
        #                 A[row_indices[right_internal], right_prev_indices] -= weight_prev * h11_internal
        #                 A[row_indices[right_internal], right_next_indices] += weight_next * h11_internal
                    
        #             # 2. Left boundary when the right point is the first control point
        #             right_left_boundary = right_indices == 0
        #             if right_left_boundary.any():
        #                 # Boundary derivative uses a forward difference
        #                 right_next_indices = torch.ones_like(right_indices[right_left_boundary])  # index 1
        #                 dt_left = control_points_idx[right_next_indices] - control_points_idx[0]
                        
        #                 # Apply the derivative contribution
        #                 derivative_scale = dt[right_left_boundary]  # boundary points use a larger scale
        #                 h11_left_boundary = h11[right_left_boundary] * derivative_scale
                        
        #                 # The boundary derivative comes entirely from the next point
        #                 A[row_indices[right_left_boundary], 0] += 0.0  # no self contribution
        #                 A[row_indices[right_left_boundary], right_next_indices] += 1.0 * h11_left_boundary  # positive contribution from the next point
                    
        #             # 3. Right boundary (backward difference)
        #             right_boundary = right_indices == control_points_num - 1
        #             if right_boundary.any():
        #                 # Boundary derivative uses a backward difference
        #                 right_prev_indices = torch.ones_like(right_indices[right_boundary]) * (control_points_num - 2)
        #                 dt_right = control_points_idx[control_points_num-1] - control_points_idx[right_prev_indices]
                        
        #                 # Apply the derivative contribution
        #                 derivative_scale = dt[right_boundary]  # boundary points use a larger scale
        #                 h11_right_boundary = h11[right_boundary] * derivative_scale
                        
        #                 # The boundary derivative comes entirely from the previous point
        #                 A[row_indices[right_boundary], right_prev_indices] -= 1.0 * h11_right_boundary  # negative contribution from the previous point
        #                 A[row_indices[right_boundary], control_points_num-1] += 0.0  # no self contribution
                
        #         # Smoothness regularizer so neighboring control points stay smooth
        #         smooth_weight = 0.1
        #         S = torch.zeros((control_points_num, control_points_num), device=tracks_3d.device)
        #         for i in range(1, control_points_num-1):
        #             S[i, i-1] = -1
        #             S[i, i] = 2
        #             S[i, i+1] = -1
                
        #         # Boundary handling
        #         if control_points_num > 1:
        #             S[0, 0] = 1
        #             S[0, 1] = -1
        #             S[-1, -2] = -1
        #             S[-1, -1] = 1
                
        #         # Precompute the shared term and add the smoothness regularizer
        #         ATA = A.T @ A + smooth_weight * S.T @ S
                
        #         # Regularizer for numerical stability
        #         reg = 1e-6 * torch.eye(control_points_num, device=tracks_3d.device)
        #         ATA_reg = ATA + reg
                
        #         # Precompute the LU factorization so many linear systems can be solved efficiently
        #         ATA_LU, pivots = torch.linalg.lu_factor(ATA_reg)
                
        #         # Reshape delta_position to (T, N*3) for the matrix multiply
        #         y = delta_position.reshape(T, N * 3)
                
        #         # Compute A^T y, shape (control_points_num, N*3)
        #         ATy = A.T @ y
                
        #         # Solve ATA_reg * x = ATy
        #         # Result shape is (control_points_num, N*3)
        #         control_points = torch.linalg.lu_solve(ATy, ATA_LU, pivots)
                
        #         # Reshape back to (N, control_points_num*3)
        #         pos_cubic_node = control_points.reshape(control_points_num, N, 3).permute(1, 0, 2).reshape(N, -1)
                
        #     except Exception as e:
        #         print(f"Failed to optimize control points using tensor-based matrix method: {e}")
        #         # On failure, print the error and fall back to the original method
        #         pass

        # Return the base position, control-point positions, and control-point indices
        return base_position, pos_cubic_node, control_points_idx

    def get_tone_mapper(self, hidden=128, act="relu", pretrain_iters=1000):
        tone_mapper = ToneMapper(hidden, act).cuda()
        tone_mapper_optimizer = torch.optim.Adam(list(tone_mapper.parameters()), lr=5e-4, eps=1e-15)
        logging.info("Pre-train the tone mapper to ensure a proper initialization.")
        for i in range(pretrain_iters):
            # ln_x = (torch.rand(8096, 3, requires_grad=True).cuda() * 20.0 - 10.0)
            # rgb_l = tone_mapper(ln_x)
            # gt_rgb = torch.clip(torch.log(torch.exp(ln_x)+1), 0, 1)
            # loss = (rgb_l - gt_rgb).norm(dim=1).mean()

            loss = CRF_RGB_equal_loss(tone_mapper) + CRF_monotonic_loss(tone_mapper) + unit_expos_loss(tone_mapper, self.tone_mapper_loss_gt)
            loss.backward()
            tone_mapper_optimizer.step()
            tone_mapper_optimizer.zero_grad()
 
        return tone_mapper

    def apply_gs_control(self,
        render_dict,
        model,
        gs_control_cfg,
        step,
        optimizer_gs,
        record_flag=True,
        size_th=1e10
    ):
        grad = render_dict["viewspace_points"].grad
        radii = render_dict["radii"]
        visib = render_dict["visibility_filter"]
        if record_flag:
            model.record_xyz_grad_radii(grad, radii, visib)
        if (
            step in gs_control_cfg.densify_steps
            or step in gs_control_cfg.prune_steps
            or step in gs_control_cfg.reset_steps
        ):
            logging.info(f"GS Control at {step}")
        if step in gs_control_cfg.densify_steps and model.N < 150000:
            N_old = model.N
            model.densify(
                optimizer=optimizer_gs,
                max_grad=gs_control_cfg.densify_max_grad,
                percent_dense=gs_control_cfg.densify_percent_dense,
                extent=1,
                verbose=True,
            )
            logging.info(f"Densify: {N_old}->{model.N}")
        if step in gs_control_cfg.prune_steps:
            N_old = model.N
            size_threshold = size_th if step > gs_control_cfg.reset_steps[0] else None 
            model.prune_points(
                optimizer_gs,
                min_opacity=gs_control_cfg.prune_opacity_th,
                extent=1,
                max_screen_size=size_threshold
            )
            logging.info(f"Prune: {N_old}->{model.N}")
        if step in gs_control_cfg.reset_steps:
            model.reset_opacity(optimizer_gs, gs_control_cfg.reset_opacity)


    def render_test(self):
        save_dir = osp.join(self.out_dir, 'test')
        os.makedirs(save_dir+"_hdr_exr", exist_ok=True)
        os.makedirs(save_dir+"_hdr_tm", exist_ok=True)
        os.makedirs(save_dir+"_ldr", exist_ok=True)
        if hasattr(self.s2d, "test_ne_ldr_gt"):
            os.makedirs(save_dir+"_ne_ldr", exist_ok=True)

        images_h = []
        for i in range(len(self.s2d.test_exposures)):
            xyz, rot, scale, opa, sph = self.cam_d_model(i+0.5)
            render_dict = render_cam_canonical(xyz, rot, scale, opa, sph, self.s2d.H, self.s2d.W, self.color_mlp, HDR_mode=True)
            rgb_h = render_dict["rgb"]
            tmp = rgb_h.clone().reshape([3, -1]).transpose(0, 1)
            expos = self.s2d.test_exposures[i].detach().clone()
            pixel_lnx = torch.log2(tmp + 1e-5) + expos
            rgb_l = self.tone_mapper(pixel_lnx).transpose(0, 1).reshape(rgb_h.shape)  
            rgb_l = (rgb_l.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            imageio.imwrite(osp.join(save_dir+"_ldr", f'{self.s2d.test_frame_names[i]}.jpg'), rgb_l)
            images_h.append(rgb_h.permute(1,2,0).cpu().numpy())

            if hasattr(self.s2d, "test_ne_ldr_gt"):
                expos_ne = self.s2d.test_novel_exposures[i].detach().clone()
                pixel_lnx = torch.log2(tmp + 1e-5) + expos_ne
                rgb_l_ne = self.tone_mapper(pixel_lnx).transpose(0, 1).reshape(rgb_h.shape)  
                rgb_l_ne = (rgb_l_ne.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                imageio.imwrite(osp.join(save_dir+"_ne_ldr", f'{self.s2d.test_frame_names[i]}.jpg'), rgb_l_ne)
        
        for i, img in enumerate(images_h):
            cv2.imwrite(osp.join(save_dir+"_hdr_exr", f'{self.s2d.test_frame_names[i]}.exr'), img[:, :, ::-1])
        
        images_h = images_h/(np.max(images_h))
        Reinhards = [] 
        for i, h in enumerate(images_h):
            h = (tonemapReinhard.process(h) * 255).astype(np.uint8)  ## use Reinhard’s method
            imageio.imwrite(osp.join(save_dir+"_hdr_tm", f'{self.s2d.test_frame_names[i]}.jpg'), h)
            Reinhards.append(h)
        images_h = np.stack(Reinhards, 0)    


    def eval_ldr(self, mode="train_ldr"):
        self.per_view_dict[mode] = {}
        self.full_dict[mode] = {}
        ssims = []; psnrs = []; lpipss = []

        dir_name = mode
        if mode == "train_ldr":
            gts = self.s2d.rgb
            frame_names = self.s2d.train_frame_names
        if mode == "test_ldr":
            assert hasattr(self.s2d, "test_ldr_gt")
            gts = self.s2d.test_ldr_gt       
            frame_names = self.s2d.test_frame_names 
        if mode == "test_ne_ldr":
            assert hasattr(self.s2d, "test_ne_ldr_gt")
            gts = self.s2d.test_ne_ldr_gt       
            frame_names = self.s2d.test_frame_names           

        img_fns = [
                f
                for f in os.listdir(osp.join(self.out_dir, dir_name))
                if f.endswith(".jpg") or f.endswith(".png")
            ]
        img_fns.sort()
            
        images = [imageio.imread(osp.join(self.out_dir, dir_name, img_fn)) for img_fn in img_fns]
        images = torch.Tensor(np.stack(images)) / 255.0  # T,H,W,3
        images = images.cuda()

        for i in range(images.shape[0]):
            img = images[i].permute(2, 0, 1)
            gt = gts[i].permute(2, 0, 1)
            ssims.append(ssim(img.unsqueeze(0), gt.unsqueeze(0)))
            psnrs.append(psnr(img.unsqueeze(0), gt.unsqueeze(0)))
            lpipss.append(lpips(img.unsqueeze(0), gt.unsqueeze(0), net_type='alex'))
            
        self.per_view_dict[mode].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), frame_names)}, "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), frame_names)}, "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), frame_names)}})
        self.full_dict[mode].update({"SSIM": torch.tensor(ssims).mean().item(), "PSNR": torch.tensor(psnrs).mean().item(), "LPIPS": torch.tensor(lpipss).mean().item()})


    def eval_hdr(self, mode="train_hdr"):
        self.per_view_dict[mode] = {}
        self.full_dict[mode] = {}
        ssims = []; psnrs = []; lpipss = []

        dir_name = mode + "_exr"
        if mode == "train_hdr":
            assert hasattr(self.s2d, "train_hdr_gt")
            gts = self.s2d.train_hdr_gt
            frame_names = self.s2d.train_frame_names
        if mode == "test_hdr":
            assert hasattr(self.s2d, "test_hdr_gt")
            gts = self.s2d.test_hdr_gt       
            frame_names = self.s2d.test_frame_names 
         
        img_fns = [
                f
                for f in os.listdir(osp.join(self.out_dir, dir_name))
                if f.endswith(".exr")
            ]
        img_fns.sort()
            
        images = [cv2.imread(osp.join(self.out_dir, dir_name, img_fn), cv2.IMREAD_UNCHANGED)[:,:,::-1] for img_fn in img_fns]
        images = torch.Tensor(np.stack(images))   # T,H,W,3
        images = images.cuda()

        for i in range(images.shape[0]):
            img = images[i].permute(2, 0, 1)
            gt = gts[i].permute(2, 0, 1)
            img_tm = tonemap_mu(img / gt.max())
            gt_tm = tonemap_mu(gt / gt.max())
            ssims.append(ssim(img_tm.unsqueeze(0), gt_tm.unsqueeze(0)))
            psnrs.append(psnr(img_tm.unsqueeze(0), gt_tm.unsqueeze(0)))
            lpipss.append(lpips(img_tm.unsqueeze(0), gt_tm.unsqueeze(0), net_type='alex'))
            
        self.per_view_dict[mode].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), frame_names)}, "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), frame_names)}, "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), frame_names)}})
        self.full_dict[mode].update({"SSIM": torch.tensor(ssims).mean().item(), "PSNR": torch.tensor(psnrs).mean().item(), "LPIPS": torch.tensor(lpipss).mean().item()})

        self.per_view_dict[mode+"noGT"] = {}
        self.full_dict[mode+"noGT"] = {}
        ssims = []; psnrs = []; lpipss = []
        for i in range(images.shape[0]):
            img = images[i].permute(2, 0, 1)
            gt = gts[i].permute(2, 0, 1)
            img_tm = tonemap_mu(img / img.max())
            gt_tm = tonemap_mu(gt / gt.max())
            ssims.append(ssim(img_tm.unsqueeze(0), gt_tm.unsqueeze(0)))
            psnrs.append(psnr(img_tm.unsqueeze(0), gt_tm.unsqueeze(0)))
            lpipss.append(lpips(img_tm.unsqueeze(0), gt_tm.unsqueeze(0), net_type='alex'))
        self.per_view_dict[mode+"noGT"].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), frame_names)}, "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), frame_names)}, "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), frame_names)}})
        self.full_dict[mode+"noGT"].update({"SSIM": torch.tensor(ssims).mean().item(), "PSNR": torch.tensor(psnrs).mean().item(), "LPIPS": torch.tensor(lpipss).mean().item()})

    def eval_hdr_tc(self, include_test=False):
        img_fns_h = [
                osp.join(self.out_dir, "train_hdr_exr", f)
                for f in os.listdir(osp.join(self.out_dir, "train_hdr_exr"))
                if f.endswith(".exr")
            ]
        if os.path.exists(osp.join(self.args.ws, "train_hdr_tm_gt")):
            img_tm_dir = osp.join(self.args.ws, "train_hdr_tm_gt")
        else:
            img_tm_dir = osp.join(self.out_dir, "train_hdr_tm")
        img_fns_tm = [
                osp.join(img_tm_dir, f)
                for f in os.listdir(img_tm_dir)
                if f.endswith(".jpg") or f.endswith(".png")
            ]
        if include_test:
            try:
                img_fns_h_test = [
                    osp.join(self.out_dir, "test_hdr_exr", f)
                    for f in os.listdir(osp.join(self.out_dir, "test_hdr_exr"))
                    if f.endswith(".exr")
                ]
                if os.path.exists(osp.join(self.args.ws, "test_hdr_tm_gt")):
                    img_tm_test_dir = osp.join(self.args.ws, "test_hdr_tm_gt")
                else:
                    img_tm_test_dir = osp.join(self.out_dir, "test_hdr_tm")
                img_fns_tm_test = [
                        osp.join(img_tm_test_dir, f)
                        for f in os.listdir(img_tm_test_dir)
                        if f.endswith(".jpg") or f.endswith(".png")
                    ]
                img_fns_h += img_fns_h_test
                img_fns_tm += img_fns_tm_test
            except:
                pass

        img_fns_h.sort(key=lambda x: x.split('/')[-1])
        img_fns_tm.sort(key=lambda x: x.split('/')[-1])
        images_h = [cv2.imread(img_fn, cv2.IMREAD_UNCHANGED)[:,:,::-1] for img_fn in img_fns_h]
        images_h = torch.Tensor(np.stack(images_h))   # T,H,W,3
        images_h = images_h.cuda()

        images_tm = [cv2.imread(img_fn)[:,:,::-1] for img_fn in img_fns_tm]

        torch.cuda.empty_cache()
        HDR_TAE = compute_HDR_TAE(images_tm, images_h, flow_mode="gmflow")
        # print("HDR_TAE:", HDR_TAE)

        img_avgs = []
        images_norm = images_h / images_h.mean()
        for img in images_norm:
            img_avgs.append(img.mean())
        img_avgs = torch.tensor(img_avgs)
        # print("mean:", img_avgs.mean().item())
        # print("HDR_T_std:", img_avgs.std().item()) 

        if not include_test:
            mode = "HDR_TC(train)"
        else:
            mode = "HDR_TC(train+test)"
        self.full_dict[mode] = {}
        self.full_dict[mode].update({"HDR_TAE": HDR_TAE, "HDR_T_std": img_avgs.std().item()})

    def render_train(self):
        images_l, depths, images_h = [], [], []
        render_times = []
        for i in range(self.s2d.T):
            start_time = time.time()
            xyz, rot, scale, opa, sph = self.cam_d_model(i)
            render_dict = render_cam_canonical(xyz, rot, scale, opa, sph, self.s2d.H, self.s2d.W, self.color_mlp,HDR_mode=True)
            rgb_h = render_dict["rgb"]
            expos = self.s2d.train_exposures[i].detach().clone()
            tmp = rgb_h.clone().reshape([3, -1]).transpose(0, 1)
            pixel_lnx = torch.log2(tmp + 1e-5) + expos
            rgb_l = self.tone_mapper(pixel_lnx).transpose(0, 1).reshape(rgb_h.shape)  
            end_time = time.time()
            if i > 1:
                render_times.append(end_time - start_time)
            depth = render_dict['dep'].permute(1,2,0).cpu().numpy()
            images_l.append((rgb_l.permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            images_h.append(rgb_h.permute(1,2,0).cpu().numpy())
            depths.append(depth)
        FPS = 1/np.mean(render_times)
        self.full_dict['render_fps'] = FPS
        save_dir = osp.join(self.out_dir, 'train')
        os.makedirs(save_dir+"_hdr_exr", exist_ok=True)
        os.makedirs(save_dir+"_hdr_tm", exist_ok=True)
        os.makedirs(save_dir+"_ldr", exist_ok=True)
        for i, img in enumerate(images_h):
            cv2.imwrite(osp.join(save_dir+"_hdr_exr", f'{self.s2d.train_frame_names[i]}.exr'), img[:, :, ::-1])
        
        # images_h = images_h/(np.max(images_h))
        images_h = (images_h/np.percentile(images_h, 99.5)).clip(0,1)
     
        Reinhards = [] 
        for i, h in enumerate(images_h):
            h = (tonemapReinhard.process(h) * 255).astype(np.uint8)  ## use Reinhard’s method
            imageio.imwrite(osp.join(save_dir+"_hdr_tm", f'{self.s2d.train_frame_names[i]}.jpg'), h)
            Reinhards.append(h)
        images_h = np.stack(Reinhards, 0)    

        for i, l in enumerate(images_l):
            imageio.imwrite(osp.join(save_dir+"_ldr", f'{self.s2d.train_frame_names[i]}.jpg'), l)

        imageio.mimwrite(osp.join(self.out_dir, 'train_video_hdr.mp4'), images_h, quality=8)
        # imageio.mimwrite(osp.join(self.out_dir, 'train_video_ldr.mp4'), images_l, quality=8)

        dep_list = np.stack(depths, axis=0)
    
        # use disparity to viz, not depth
        dep_valid_mask = dep_list > 1e-6
        viz_quantile =3 
        # use robust min and max to visualize
        dep_max = np.percentile(dep_list[dep_valid_mask], 100 - viz_quantile)
        dep_min = np.percentile(dep_list[dep_valid_mask], viz_quantile)
        dep_list = np.clip(dep_list, dep_min, dep_max)
        dep_list = (dep_list - dep_min) / (dep_max - dep_min)
        dep_list[~dep_valid_mask] = 0
        viz_depth_list = []
        for dep in dep_list:
            viz = cm.viridis(dep[:,:,0])[:, :, :3]
            viz_depth_list.append((viz * 255).astype(np.uint8))
        imageio.mimwrite(osp.join(self.out_dir, 'train_video_depth.mp4'), viz_depth_list, quality=8)


    def compute_dyn_tc_loss(self, render_dict_dst, dst_id):
        if dst_id == 0:
            src_ids = [1]
        elif dst_id == self.s2d.T-1:
            src_ids = [self.s2d.T-2]
        else:
            src_ids = [dst_id-1, dst_id+1]

        dst_xyz, _, _, _, _ = self.cam_d_model(dst_id)
        add_buffer = dst_xyz
        depth_dst = render_dict_dst["dep"][0]
        rgb_h_dst = render_dict_dst["rgb"]

        loss_tlr = torch.tensor(0.0).cuda()
        for src_id in src_ids:
            xyz, rot, scale, opa, sph = self.cam_d_model(src_id)
            render_dict_src = render_cam_canonical(xyz, rot, scale, opa, sph, self.s2d.H, self.s2d.W, self.color_mlp, add_buffer, HDR_mode=True)
            rgb_h_src = render_dict_src["rgb"]
            rendered_xyz_map = render_dict_src["buf"].permute(1, 2, 0)  # H,W,3
            pix_coords = rendered_xyz_map[..., 0:2]    ## in camera canonical space
            depth_dst2src_proj = rendered_xyz_map[..., 2]

            proj_mask_x = (pix_coords[..., 0] >= -1) & (pix_coords[..., 0] <= 1)
            proj_mask_y = (pix_coords[..., 1] >= -1) & (pix_coords[..., 1] <= 1)
            inside_mask = proj_mask_x & proj_mask_y

            rgb_h_dst2src = torch.nn.functional.grid_sample(rgb_h_dst.unsqueeze(0), pix_coords.unsqueeze(0).detach(), padding_mode="border", align_corners=True)[0]
            depth_dst2src_sample = torch.nn.functional.grid_sample(depth_dst.unsqueeze(0).unsqueeze(0), pix_coords.unsqueeze(0), padding_mode="border", align_corners=True)[0][0]
            occ_mask = depth_dst2src_proj > 1.1 * depth_dst2src_sample
            mask = (inside_mask & (~occ_mask)).detach()

            diff = (rgb_h_src - rgb_h_dst2src).abs() / (rgb_h_src + rgb_h_dst2src + 1e-10).detach()
            loss_tlr += (diff * mask[None, ...]).sum() / (mask.sum() + 1e-10)
        loss_tlr = loss_tlr / len(src_ids)

        return loss_tlr
    
    def train(self):
        view_ind_list = list(range(self.s2d.T))
        random.shuffle(view_ind_list)

        base_u, base_v = np.meshgrid(np.arange(self.s2d.W), np.arange(self.s2d.H))
        base_uv = np.stack([base_u, base_v], -1)
        base_uv = torch.tensor(base_uv, device=self.s2d.rgb.device).long()       

        track_loss_interval = getattr(self.args, "photo_track_loss_interval", 4)
        latest_track_event = 0
        track_loss_protect_steps = 100
        loss_rgb_list, loss_dep_list, loss_rigid_list, loss_track_list = [], [], [], []
        loss_vel_xyz_list, loss_vel_rot_list, loss_acc_xyz_list, loss_acc_rot_list = [], [], [], []
        loss_opa_list = []
        loss_tlr_list = []
        loss_scale_var_list = []
        loss_ue_list = []
        cam_d_gs_n_count_list = []

        for step in tqdm(range(1, self.total_steps+1)):
            if not view_ind_list:
                view_ind_list = list(range(self.s2d.T))
                random.shuffle(view_ind_list)
            view_ind = view_ind_list.pop(0)
            exposure = self.s2d.train_exposures[view_ind].detach().clone()
            track_flow_interval_candidates = [len(torch.unique(self.s2d.train_exposures))]
            # track_flow_interval_candidates = [3]

            depth_gt = self.s2d.dep[view_ind].detach().clone()
            rgb_l_gt = self.s2d.rgb[view_ind].permute(2, 0, 1).detach().clone()

            # corr_exe_flag = (
            #     step > latest_track_event + track_loss_protect_steps
            #     and step % track_loss_interval == 0
            # )
            corr_exe_flag = True
            # select another ind different than the view_ind  
            flow_flag = np.random.rand() < getattr(self.args, "photo_track_flow_chance", 0.5)
            if flow_flag:  # contruct target by flow
                corr_dst_ind_candidates = []
                for flow_interval in track_flow_interval_candidates:
                    if view_ind + flow_interval < self.s2d.T:
                        corr_dst_ind_candidates.append(view_ind + flow_interval)
                    if view_ind - flow_interval >= 0:
                        corr_dst_ind_candidates.append(view_ind - flow_interval)
                corr_dst_ind = np.random.choice(corr_dst_ind_candidates)
                flow_ind = self.s2d.flow_ij_to_listind_dict[(view_ind, corr_dst_ind)]
                flow = self.s2d.flow[flow_ind].detach().clone()
                flow_mask = self.s2d.flow_mask[flow_ind].detach().clone().bool()
                track_src = base_uv.clone().detach()[flow_mask]
                flow = flow[flow_mask]
                track_dst = track_src.float() + flow
            else:   # contruct target by track
                corr_dst_ind = view_ind
                while corr_dst_ind == view_ind:
                    corr_dst_ind = np.random.choice(self.s2d.T)
                # left = max(view_ind-5, 0)
                # right = min(view_ind+5, self.s2d.T-1)
                # corr_dst_ind_candidates = list(range(left, view_ind))+list(range(view_ind+1, right+1))
                # corr_dst_ind = np.random.choice(corr_dst_ind_candidates)
                track_valid = self.s2d.track_mask[view_ind] & self.s2d.track_mask[corr_dst_ind]
                track_src = self.s2d.track[view_ind][track_valid][..., :2].detach().clone()
                track_dst = self.s2d.track[corr_dst_ind][track_valid][..., :2].detach().clone()


            xyz, rot, scale, opa, sph = self.cam_d_model(view_ind)
            dst_xyz, _, _, _, _ = self.cam_d_model(corr_dst_ind)
            if corr_exe_flag:
                add_buffer = dst_xyz
            else:
                add_buffer = torch.zeros_like(dst_xyz).cuda()

            ii, jj, nn, weight = cal_connectivity_from_points(points=xyz.clone(), K=5)
            pos = torch.stack([xyz.clone(), dst_xyz.clone()], dim=0)
            loss_rigid = cal_arap_error(pos, ii, jj, nn) 

            render_dict = render_cam_canonical(xyz, rot, scale, opa, sph, self.s2d.H, self.s2d.W, self.color_mlp, add_buffer, HDR_mode=True)
            alpha = render_dict["alpha"][0]
            # loss_opa = (torch.ones_like(alpha, dtype=torch.float, device="cuda").detach() - alpha).abs().mean()
            opa_diff = (torch.ones_like(alpha, dtype=torch.float, device="cuda").detach() - alpha).abs()
            loss_opa = (opa_diff * opa_diff.detach()).sum() / (opa_diff.detach().sum() + 1e-5)

            rgb_h = render_dict["rgb"]
            depth = render_dict["dep"][0]
            tmp = rgb_h.clone().reshape([3, -1]).transpose(0, 1)
            pixel_lnx = torch.log2(tmp + 1e-6) + exposure
            rgb_l = self.tone_mapper(pixel_lnx).transpose(0, 1).reshape(rgb_h.shape)  

            rgb_sup_mask = self.s2d.get_mask_by_key("all")[view_ind]
            loss_rgb, _ = compute_rgb_loss(rgb_l_gt.detach().clone(), rgb_l, rgb_sup_mask)
            # dep_sup_mask = rgb_sup_mask * self.s2d.dep_mask[view_ind]
            dep_sup_mask = rgb_sup_mask
            loss_dep_i = torch.abs((depth_gt.detach().clone() - depth)) * dep_sup_mask
            loss_dep = loss_dep_i.sum() / dep_sup_mask.sum()
            # loss_dep, _ = compute_dep_loss(depth_gt.detach().clone(), depth, dep_sup_mask)
            # loss_dep = depth_correlation_loss(depth_gt[..., None].detach().clone(), depth[..., None], patch_size=32, num_patches=64)
            if corr_exe_flag:
                rendered_xyz_map = render_dict["buf"].permute(1, 2, 0)  # H,W,3
                if len(track_src) == 0:
                    loss_track = torch.zeros_like(loss_rgb)
                else:
                    src_fetch_index = track_src[:, 1].long() * self.s2d.W + track_src[:, 0].long()
                    warped_xyz_cam = rendered_xyz_map.reshape(-1, 3)[src_fetch_index]
                    # filter the pred, only add loss to points that are infront of the camera
                    track_loss_mask = warped_xyz_cam[:, 2] > 1e-4
                    if track_loss_mask.sum() == 0:
                        loss_track = torch.zeros_like(loss_rgb)
                    else:
                        pred_track_dst = warped_xyz_cam[:, 0:2]    ## in camera canonical space
                        pred_track_dst[:, :1] = (pred_track_dst[:, :1]+1.0) * self.s2d.W/2 - 0.5
                        pred_track_dst[:, 1:] = (pred_track_dst[:, 1:]+1.0) * self.s2d.H/2 - 0.5
                        loss_track = (pred_track_dst - track_dst).norm(dim=-1)[track_loss_mask]
                        loss_track = loss_track.clamp(0.0, 100.0).mean()
            else:
                loss_track = torch.zeros_like(loss_rgb)
            
            loss_scale_var = torch.mean(torch.std(self.cam_d_model.get_scaling, dim=1))
            

            reg_radius = 2
            _l = max(0, view_ind - reg_radius)
            _r = min(self.s2d.T, view_ind + 1 + reg_radius)
            tids = torch.arange(_l, _r).cuda()
            loss_vel_xyz, loss_vel_rot, loss_acc_xyz, loss_acc_rot = self.cam_d_model.compute_vel_acc_loss(tids)

            if step >= self.cam_d_ctrl_end:
                loss_tlr = self.compute_dyn_tc_loss(render_dict, view_ind)
            else:
                loss_tlr = torch.tensor(0.0)
            # loss_rigid = torch.tensor(0)
            loss = loss_rgb + self.args.weight_dyn_tc*loss_tlr + 1.0 * loss_dep + 1.0 * loss_opa + 0.01*loss_track + 0.01 * loss_rigid + 10*unit_expos_loss(self.tone_mapper, self.tone_mapper_loss_gt) + 10 * CRF_monotonic_loss(self.tone_mapper) + 10 * loss_scale_var + 10 * (loss_vel_xyz + loss_vel_rot + loss_acc_xyz + loss_acc_rot)
 
            loss_rgb_list.append(loss_rgb.item())
            loss_dep_list.append(loss_dep.item())
            loss_opa_list.append(loss_opa.item())
            loss_tlr_list.append(loss_tlr.item())
            loss_track_list.append(loss_track.item())
            loss_rigid_list.append(loss_rigid.item())
            loss_vel_xyz_list.append(loss_vel_xyz.item()+loss_vel_rot.item())
            loss_vel_rot_list.append(loss_vel_rot.item())
            loss_acc_xyz_list.append(loss_acc_xyz.item()+loss_acc_rot.item())
            loss_acc_rot_list.append(loss_acc_rot.item())
            loss_scale_var_list.append(loss_scale_var.item())
            loss_ue_list.append(unit_expos_loss(self.tone_mapper, self.tone_mapper_loss_gt).item())
            cam_d_gs_n_count_list.append(self.cam_d_model.N)

            # import time
            # t = time.time()
            loss.backward()
            # torch.cuda.synchronize()
            # print("  pytorch runtime: ", (time.time() - t), " s")

            if self.cam_d_gs_ctrl_cfg is not None and step > self.cam_d_ctrl_start and step < self.cam_d_ctrl_end:
                self.apply_gs_control(
                    render_dict=render_dict,
                    model=self.cam_d_model,
                    gs_control_cfg=self.cam_d_gs_ctrl_cfg,
                    step=step,
                    optimizer_gs=self.optimizer_cam_d_model,
                    record_flag=True,
                    size_th=1e10
                    # record_flag=(not corr_exe_flag)
                )
                if step in self.cam_d_gs_ctrl_cfg.reset_steps:
                    latest_track_event = step

            self.optimizer_tone_mapper.step()
            self.optimizer_color_mlp.step()
            self.optimizer_cam_d_model.step()
            self.optimizer_tone_mapper.zero_grad(set_to_none = True)
            self.optimizer_color_mlp.zero_grad(set_to_none = True)
            self.optimizer_cam_d_model.zero_grad(set_to_none = True)

            if step > 0:
                for k, v in self.gs_model_scheduling_func_dict.items():
                    update_learning_rate(v(step), k, self.optimizer_cam_d_model)

            if step % 200 == 0:
                draw_CRF(self.tone_mapper, self.out_dir)

            if step % getattr(self.args, "photo_log_step", 5000) == 0:
                fig = plt.figure(figsize=(30, 5))
                for plt_i, plt_pack in enumerate(
                    [
                        ("loss_rgb", loss_rgb_list),
                        ("loss_ue", loss_ue_list),
                        ("loss_dep", loss_dep_list),
                        # ("loss_opa", loss_opa_list),
                        ("loss_track", loss_track_list),
                        ("loss_arap", loss_rigid_list),
                        ("loss_vel", loss_vel_xyz_list),
                        ("loss_acc", loss_acc_xyz_list),                        
                        ("loss_tlr", loss_tlr_list),
                        # ("loss_scale_var", loss_scale_var_list),
                        # ("loss_vel_xyz", loss_vel_xyz_list),
                        # ("loss_vel_rot", loss_vel_rot_list),
                        # ("loss_acc_xyz", loss_acc_xyz_list),
                        # ("loss_acc_rot", loss_acc_rot_list),
                        # ("CamD-GS-N", cam_d_gs_n_count_list),
                    ]
                ):
                    plt.subplot(1, 8, plt_i + 1)
                    plt.plot(plt_pack[1]), plt.title(plt_pack[0] + f" End={plt_pack[1][-1]:.6f}")
                plt.tight_layout()
                plt.savefig(
                    osp.join(self.out_dir, "optim_loss.jpg")
                )
                plt.close()
       
        with torch.no_grad():

            ckpt_dir = osp.join(self.out_dir, "ckpt")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(self.cam_d_model.state_dict(), osp.join(ckpt_dir, "cam_d_model.pth"))
            torch.save(self.tone_mapper.state_dict(), osp.join(ckpt_dir, "tone_mapper.pth"))
            torch.save(self.color_mlp.state_dict(), osp.join(ckpt_dir, "color_mlp.pth"))

            self.render_train()
            self.eval_ldr("train_ldr")
            if hasattr(self.s2d, "train_hdr_gt"):
                self.eval_hdr("train_hdr")
            if hasattr(self.s2d, "test_ldr_gt"):
                self.render_test()
                self.eval_ldr("test_ldr")
                if hasattr(self.s2d, "test_ne_ldr_gt"):
                    self.eval_ldr("test_ne_ldr")
                if hasattr(self.s2d, "test_hdr_gt"):
                    self.eval_hdr("test_hdr")  

            self.eval_hdr_tc(include_test=False)
            if hasattr(self.s2d, "test_ldr_gt"):
                self.eval_hdr_tc(include_test=True)

        with open(self.out_dir + "/results.json", 'a') as fp:
            json.dump(self.full_dict, fp, indent=True)
        with open(self.out_dir + "/per_view.json", 'a') as fp:
            json.dump(self.per_view_dict, fp, indent=True)

        