import os, glob, math, random, logging
import os.path as osp
import numpy as np
import torch, torchvision
import torch.nn as nn
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
    quaternion_to_axis_angle,
)
from pytorch3d.ops import knn_points
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from simple_knn._C import distCUDA2
from lib_prior.prior_loading import Saved2D
from mosca_precompute import auto_get_depth_dir_tap_mode
from scene.dynamic_gs import DynamicGaussian
from scene.static_gs import StaticGaussian
from scene.tone_mapper import ToneMapper
from lib_moca.camera import MonocularCameras
from utils.optim_utils import *
from utils.general_utils import batched_quaternion_average, batched_J_world, BackprojectDepth, Project3D
from tqdm import tqdm
from lib_render.gauspl_renderer_cam_canonical import render_cam_canonical
from lib_render.gauspl_renderer_native import render_native
from lib_moca.bundle import query_buffers_by_track
from lib_moca.epi_helpers import identify_tracks
from lib_prior.prior_loading import gather_track_from_buffer
from utils.loss_utils import compute_rgb_loss, compute_dep_loss, depth_correlation_loss, draw_CRF, unit_expos_loss, CRF_RGB_equal_loss, CRF_monotonic_loss, ssim, compute_HDR_TAE
from utils.geometry_utils import cal_connectivity_from_points, cal_arap_error, cal_smooth_error
from utils.viz_utils import *
from matplotlib import cm
import imageio, json
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from utils.image_utils import psnr
from lpipsPyTorch import lpips
import time
tonemapReinhard = cv2.createTonemapReinhard(2.2, 0.5, 0.5 ,0)
tonemap_mu = lambda x : (torch.log(torch.clip(x,0,1) * 5000 + 1 ) / np.log(5000 + 1))

class WorldGSTrainer:
    def __init__(self, args):
        self.args = args
        self.tone_mapper_loss_gt = (0.5)**(1/2.2) if "syn" in self.args.ws else 0.5
        
        self.color_feature_dim = 36
        
        self.color_mlp = nn.Sequential(
                            nn.Linear(self.color_feature_dim+3, self.color_feature_dim),
                            nn.ReLU(True),
                            nn.Linear(self.color_feature_dim, 3),
                            # nn.Sigmoid()
                        ).cuda()
        self.tone_mapper = self.get_tone_mapper(hidden=128, act="sp", pretrain_iters=0).cuda()
        self.bundle_path = f"bundle_{args.dep_mode}_{args.flow_mode}_{args.tap_mode}"

        self.cams: MonocularCameras = MonocularCameras.load_from_ckpt(
            torch.load(osp.join(self.args.ws, self.bundle_path, "bundle_cams.pth"))
        ).cuda()

        self.load_prior()
        self.world_gs_radius_max = getattr(self.args, "world_gs_radius_max", 0.1)

        # d_flag = self.s2d.dynamic_track_mask.sum().bool().item()
        # if d_flag:
        #     d_tracks_3d = self.world_tracks_info["tracks_3d"][self.s2d.dynamic_track_mask]
        #     self.d_model = self.get_init_dyn_gs_from_tracks(tracks_3d=d_tracks_3d.detach(), base_time_idx=0, frames_per_control_point=self.args.frames_per_control_point).cuda()
        # else:
        #     self.d_model = None
        # s_tracks_3d = self.world_tracks_info["tracks_3d"][self.s2d.static_track_mask]
        # self.s_model = self.get_init_static_gs_from_tracks(tracks_3d=s_tracks_3d.detach()).cuda()

        cam_gs_dir = osp.join(self.args.ws, self.args.exp_name, "cam_gs")
        cam_d_model = DynamicGaussian.load_from_ckpt(torch.load(osp.join(cam_gs_dir, "ckpt", "cam_d_model.pth")))
        self.tone_mapper.load_state_dict(torch.load(osp.join(cam_gs_dir, "ckpt", "tone_mapper.pth")))
        self.color_mlp.load_state_dict(torch.load(osp.join(cam_gs_dir, "ckpt", "color_mlp.pth")))
        with torch.no_grad():
            self.cam_gs_depths, self.cam_gs_imgs_h, self.cam_gs_imgs_h_tm = self.render_cam_gs(cam_d_model)
        self.s_model, self.d_model = self.get_init_world_gs_model_from_cam_gs(cam_d_model, frames_per_control_point=self.args.frames_per_control_point)

        self.out_dir = osp.join(self.args.ws, self.args.exp_name, "world_gs")
        os.makedirs(self.out_dir, exist_ok=True)
        self.per_view_dict = {}
        self.full_dict = {}
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
        self.optimizer_s_model = torch.optim.Adam(
            self.s_model.get_optimizable_list(**self.optimizer_cfg.get_static_lr_dict)
        )
        if self.d_model:
            self.optimizer_d_model = torch.optim.Adam(
                self.d_model.get_optimizable_list(**self.optimizer_cfg.get_dynamic_lr_dict)
            )
        cam_param_list = self.cams.get_optimizable_list(**self.optimizer_cfg.get_cam_lr_dict)[:2]
        if len(cam_param_list) > 0:
            self.optimizer_cams = torch.optim.Adam(
                self.cams.get_optimizable_list(**self.optimizer_cfg.get_cam_lr_dict)[:2]
            )
        else:
            self.optimizer_cams = None


        # self.remove_error_dyn_gs()
        # self.append_new_dyn_gs()
        # self.remove_error_dyn_gs()
        # self.append_new_sta_gs()
        # with torch.no_grad():
            # self.render_train()
            # self.eval_hdr_tc()
            # exit()

        self.gs_scheduling_func_dict, self.cams_scheduling_func_dict, _ = (
                self.optimizer_cfg.get_scheduler(total_steps=getattr(self.args, "photo_world_total_steps", 20000))
            )

        self.total_steps=getattr(self.args, "photo_world_total_steps", 10000)
        if self.d_model:
            self.d_ctrl_start_1 = getattr(self.args, "photo_d_ctrl_start_1", 500)
            self.d_ctrl_end_1 = getattr(self.args, "photo_d_ctrl_end_1", 10000)
            self.d_ctrl_start_2 = getattr(self.args, "photo_d_ctrl_start_2", 11000)
            self.d_ctrl_end_2 = getattr(self.args, "photo_d_ctrl_end_2", 15000)
            self.d_gs_ctrl_cfg = GSControlCFG(
                densify_steps=getattr(self.args, "photo_d_ctrl_densify_steps", 400),
                reset_steps=getattr(self.args, "photo_d_ctrl_reset_steps", 1001),
                prune_steps=getattr(self.args, "photo_d_ctrl_prune_steps", 200),
                densify_max_grad=getattr(self.args, "photo_d_ctrl_densify_max_grad", 0.0002),  
                densify_percent_dense=getattr(self.args, "photo_d_ctrl_densify_percent_dense", 0.01),
                prune_opacity_th=getattr(self.args, "photo_d_ctrl_prune_opacity_th", 0.05),
                reset_opacity=getattr(self.args, "photo_d_ctrl_reset_opacity", 0.01),
            )


        self.s_ctrl_start_1 = getattr(self.args, "photo_s_ctrl_start_1", 500)
        self.s_ctrl_end_1 = getattr(self.args, "photo_s_ctrl_end_1", 10000)
        self.s_ctrl_start_2 = getattr(self.args, "photo_s_ctrl_start_2", 11000)
        self.s_ctrl_end_2 = getattr(self.args, "photo_s_ctrl_end_2", 15000)
        self.s_gs_ctrl_cfg = GSControlCFG(
            densify_steps=getattr(self.args, "photo_s_ctrl_densify_steps", 400),
            reset_steps=getattr(self.args, "photo_s_ctrl_reset_steps", 1001),
            prune_steps=getattr(self.args, "photo_s_ctrl_prune_steps", 200),
            densify_max_grad=getattr(self.args, "photo_s_ctrl_densify_max_grad", 0.0002),  
            densify_percent_dense=getattr(self.args, "photo_s_ctrl_densify_percent_dense", 0.01),
            prune_opacity_th=getattr(self.args, "photo_s_ctrl_prune_opacity_th", 0.05),
            reset_opacity=getattr(self.args, "photo_s_ctrl_reset_opacity", 0.01),
        )



        # self.cams: MonocularCameras = MonocularCameras.load_from_ckpt(
        #     torch.load(osp.join(self.out_dir, "ckpt", "train_cams.pth"))
        # ).cuda()
        # self.tone_mapper = self.get_tone_mapper(hidden=128, act="sp", pretrain_iters=0).cuda()
        # self.tone_mapper.load_state_dict(torch.load(osp.join(self.out_dir, "ckpt", "tone_mapper.pth")))
        # self.d_model = DynamicGaussian.load_from_ckpt(torch.load(osp.join(self.out_dir, "ckpt", "d_model.pth"))).cuda()
        # self.s_model = StaticGaussian.load_from_ckpt(torch.load(osp.join(self.out_dir, "ckpt", "s_model.pth"))).cuda()
        # self.color_mlp.load_state_dict(torch.load(osp.join(self.out_dir, "ckpt", "color_mlp.pth")))


        # self.generate_flow_error_by_world_gs()


      
        # from utils.viz_utils import viz2d_flow_video
        # with torch.no_grad():
            # flow_frame_list, choice = viz2d_flow_video(
            #     0,
            #     self.cams.T-1,
            #     self.cams,
            #     self.s_model,
            #     self.d_model,
            #     self.color_mlp,
            #     self.tone_mapper,
            #     viz_bg=True,
            #     view_cam_id=0,
            #     traj_scale=5e-5,
            #     traj_opa=0.5,
            #     N=32,
            #     choice=None,
            #     skip_t=1,
            # )
            # imageio.mimwrite('train_video_hdr.mp4', flow_frame_list, quality=8)
            # exit()
        #     self.render_train()
        #     self.eval_ldr("train_ldr")
        #     if hasattr(self.s2d, "train_hdr_gt"):
        #         self.eval_hdr("train_hdr")

        #     if hasattr(self.s2d, "test_ldr_gt"):
        #         self.get_test_cam_pose()
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
            # self.render_eval_train(rtype="all")
            # self.render_eval_train(rtype="dynamic")
            # self.render_eval_train(rtype="static")
        #     with open(self.out_dir + "/results.json", 'a') as fp:
        #         json.dump(self.full_dict, fp, indent=True)
        #     with open(self.out_dir + "/per_view.json", 'a') as fp:
        #         json.dump(self.per_view_dict, fp, indent=True)
            # exit()
            # viz2d_total_video(
            #     viz_vid_fn=osp.join(self.out_dir, "world_gs_2dviz.mp4"),
            #     s2d=self.s2d,
            #     start_from=0,
            #     end_at=self.s2d.T-1,
            #     skip_t=1 if self.s2d.T < 120 else max(1, self.s2d.T // 50),
            #     cams=self.cams,
            #     s_model=self.s_model,
            #     d_model=self.d_model,
            #     color_mlp=self.color_mlp,
            #     tone_mapper=self.tone_mapper,
            #     move_around_angle_deg=getattr(self.args, "photo_viz_move_angle_deg", 1.0),
            #     print_text=False,
            # )
            # if self.d_model:
            #     viz3d_total_video(
            #         self.cams,
            #         self.d_model,
            #         0,
            #         self.s2d.T - 1,
            #         save_path=osp.join(self.out_dir, "world_gs_3dviz.mp4"),
            #         res=960,
            #         s_model=self.s_model,
            #         color_mlp=self.color_mlp,
            #         bg_color=[0,0,0]
            #     )
            #     exit()
            # viz_main(
            #     save_dir=osp.join(self.out_dir, "viz"),
            #     cams=self.cams,
            #     s_model=self.s_model,
            #     d_model=self.d_model,
            #     color_mlp=self.color_mlp,
            #     tone_mapper=self.tone_mapper,
            #     N=getattr(self.args, "viz_N", 5),
            #     move_angle_deg=getattr(self.args, "viz_move_angle_deg", 1.0),
            #     H_3d=getattr(self.args, "viz_H_3d", 960),
            #     W_3d=getattr(self.args, "viz_W_3d", 960),
            #     fov_3d=getattr(self.args, "viz_fov_3d", 70),
            #     back_ratio_3d=getattr(self.args, "viz_back_ratio_3d", 1.5),
            #     up_ratio=getattr(self.args, "viz_up_ratio", 0.05),
            #     bg_color=getattr(self.args, "photo_default_bg_color", [0.0, 0.0, 0.0]),
            # )
            # exit()

    def load_prior(self):
        DEPTH_BOUNDARY_TH = getattr(self.args, "depth_boundary_th", 1.0)
        DEP_MEDIAN = getattr(self.args, "dep_median", 1.0)
        # DEPTH_DIR, TAP_MODE = auto_get_depth_dir_tap_mode(self.args.ws, self.args)
        DEPTH_MODE = self.args.dep_mode
        TAP_MODE = self.args.tap_mode
        DEPTH_DIR = f"{DEPTH_MODE}_depth"
        FLOW_MODE = self.args.flow_mode
        EPI_TH = getattr(self.args, "epi_th", 0.00005)
        DYN_ID_CNT = getattr(self.args, "dyn_id_cnt", 2 * 4)
        min_curve_num=getattr(self.args, "min_curve_num", 32)
        UNIFORM_TAP_MODE = f"dep={DEPTH_MODE}_{TAP_MODE}_tap"
        self.s2d = (
                Saved2D(self.args.ws)
                .load_epi(f"epi_{FLOW_MODE}")
                .load_dep(DEPTH_DIR, DEPTH_BOUNDARY_TH)
                .normalize_depth(median_depth=DEP_MEDIAN)
                .recompute_dep_mask(depth_boundary_th=DEPTH_BOUNDARY_TH)
                .load_track(UNIFORM_TAP_MODE, min_valid_cnt=getattr(self.args, "tap_loading_min_valid_cnt", 4))
                .rescale_perframe_depth_from_bundle(osp.join(self.args.ws, self.bundle_path, "bundle.pth"))
                .load_flow(f"flow_{FLOW_MODE}")
                .set_epi_mask_to_s2d(epi_th=EPI_TH)
                .update_track_identification(self.args.ws, EPI_TH, DYN_ID_CNT, min_curve_num=min_curve_num)
                .cuda()
            )

        tracks_3d = self.s2d.track.detach().clone().permute(1,0,2)
        visibles = self.s2d.track_mask.detach().clone().permute(1,0)
        tracks_3d[..., 0] = 2 * tracks_3d[..., 0] / self.s2d.W - 1
        tracks_3d[..., 1] = 2 * tracks_3d[..., 1] / self.s2d.H - 1

        cam_K = self.cams.K(self.s2d.H, self.s2d.W)
        fx, fy = cam_K[0, 0], cam_K[1, 1]
        gs_world_track = []
        for i in range(self.s2d.T):
            ## xy at [-1,1], transform from camera canonical (pixel) space to camera space
            xyz = tracks_3d[:, i, :]
            xyz[:, 0] = xyz[:, 0] * self.s2d.W / 2 * xyz[:, 2] / fx
            xyz[:, 1] = xyz[:, 1] * self.s2d.H / 2 * xyz[:, 2] / fy

            ## transform to world space
            xyz_w = self.cams.trans_pts_to_world(i, xyz)
            gs_world_track.append(xyz_w)
        gs_world_track = torch.stack(gs_world_track, dim=1)  # [N, T, 3]
        self.world_tracks_info = {
            "tracks_3d": gs_world_track,  # [N, T, 3]
            "visibles": visibles,  # [N, T]
        }


    def render_cam_gs(self, cam_d_model):
        depths, images_h, images_h_tm = [], [], []

        for i in range(self.s2d.T):
            xyz, rot, scale, opa, sph = cam_d_model(i)
            render_dict = render_cam_canonical(xyz, rot, scale, opa, sph, self.s2d.H, self.s2d.W, self.color_mlp, HDR_mode=True)
            rgb_h = render_dict["rgb"]
            depth = render_dict['dep']
            images_h.append(rgb_h)
            images_h_tm.append(rgb_h.permute(1,2,0).cpu().numpy())
            depths.append(depth)

        cam_gs_depths = torch.cat(depths, dim=0)
        cam_gs_imgs_h = torch.stack(images_h, dim=0)   

        # images_h_tm = images_h_tm/(np.max(images_h_tm))
        images_h_tm = (images_h_tm/np.percentile(images_h_tm, 99.5)).clip(0,1)
        Reinhards = [] 
        for i, h in enumerate(images_h_tm):      
            Reinhards.append((tonemapReinhard.process(h)*255).astype(np.uint8))
        Reinhards = np.stack(Reinhards, 0) 
        cam_gs_imgs_h_tm = torch.tensor(Reinhards/255).float().cuda()

        return cam_gs_depths, cam_gs_imgs_h, cam_gs_imgs_h_tm

       

    def get_init_world_gs_model_from_cam_gs(self, cam_d_model, frames_per_control_point=1):
        cam_K = self.cams.K(self.s2d.H, self.s2d.W)
        fx, fy = cam_K[0, 0], cam_K[1, 1]
        gs_pixel_track = []
        gs_world_track = []
        gs_world_rot_q = []
        gs_cam_track = []
        gs_cam_cov2d = []
        J_cam = torch.tensor([
            [self.s2d.W/2.0, 0, 0],
            [0, self.s2d.H/2.0, 0],
            [0, 0, 0]
        ]).cuda().float()
        scale = cam_d_model.get_scaling
        S = torch.zeros(cam_d_model.N, 3, 3).cuda()
        S[:, 0, 0] = scale[:, 0]
        S[:, 1, 1] = scale[:, 1]
        S[:, 2, 2] = scale[:, 2]
        for i in range(self.s2d.T):
            xyz = cam_d_model.get_position_at_t(i)
            rot = cam_d_model.get_rotation_at_t(i)
            gs_pixel_track.append(xyz.clone().detach())

            cov3d = rot @ (S**2) @ rot.permute(0, 2, 1)
            J_cov = torch.einsum("ij,njk->nik", J_cam, cov3d)  
            cov2d = torch.einsum("nij,jk->nik", J_cov, J_cam.T)
            cov2d = cov2d[:, 0:2, 0:2].reshape(cam_d_model.N, -1)
            gs_cam_cov2d.append(cov2d)

            R_wc, t_wc = self.cams.Rt_wc(i)

            ## xy at [-1,1], transform from camera canonical (pixel) space to camera space
            xyz[:, 0] = xyz[:, 0] * self.s2d.W / 2 * xyz[:, 2] / fx
            xyz[:, 1] = xyz[:, 1] * self.s2d.H / 2 * xyz[:, 2] / fy
            gs_cam_track.append(xyz)

            ## transform to world space
            xyz_w = self.cams.trans_pts_to_world(i, xyz)
            gs_world_track.append(xyz_w)
            rot = torch.einsum("ij,njk->nik", R_wc, rot)
            rot_q = matrix_to_quaternion(rot)
            gs_world_rot_q.append(rot_q)

        gs_pixel_track = torch.stack(gs_pixel_track, dim=1)  # [N, T, 3]
        gs_world_track = torch.stack(gs_world_track, dim=1)  # [N, T, 3]
        gs_cam_track = torch.stack(gs_cam_track, dim=1)  # [N, T, 3]
        gs_world_rot_q = torch.stack(gs_world_rot_q, dim=1)  # [N, T, 4]
        gs_cam_cov2d = torch.stack(gs_cam_cov2d, dim=1)   # [N, T, 4]


        is_static, is_dynamic = self.get_init_gs_mask(gs_pixel_track.detach(), gs_world_track.detach())

        s_model = self.get_init_static_gs_from_cam_gs(cam_d_model, gs_world_track, gs_world_rot_q, gs_cam_track, gs_cam_cov2d, is_static).cuda()

        d_flag = is_dynamic.sum().bool().item()
        if d_flag:
            d_model = self.get_init_dynamic_gs_from_cam_gs(cam_d_model, gs_world_track, gs_world_rot_q, gs_cam_track, gs_cam_cov2d, is_dynamic, base_time_idx=0, frames_per_control_point=frames_per_control_point).cuda()
        else:
            d_model = None

        return s_model, d_model

    def generate_flow_error_by_cam_gs(self, cam_d_model):
        backproject_depth = BackprojectDepth(1, self.s2d.H, self.s2d.W).cuda()
        project_3d = Project3D(1, self.s2d.H, self.s2d.W).cuda()
        K = self.cams.K(self.s2d.H, self.s2d.W)
        K_44 = torch.eye(4, dtype=torch.float32).cuda()
        K_44[:3, :3] = K  
        K_44_inv = torch.inverse(K_44)

        for src_id in range(self.s2d.T):
            if src_id !=17:
                continue
            # if src_id == 0:
            #     dst_ids = [1]
            # elif src_id == self.s2d.T-1:
            #     dst_ids = [self.s2d.T-2]
            # else:
            #     dst_ids = [src_id-1, src_id+1]
            if src_id in [0, 1, 2]:
                dst_ids = [src_id+3]
            elif src_id in [self.s2d.T-3, self.s2d.T-2, self.s2d.T-1]:
                dst_ids = [src_id-3]
            else:
                dst_ids = [src_id-3, src_id+3]
                      
            depth_src = self.cam_gs_depths[src_id]
            xyz, rot, scale, opa, sph = cam_d_model(src_id)

            base_u, base_v = np.meshgrid(np.arange(self.s2d.W), np.arange(self.s2d.H))
            base_uv = np.stack([base_u, base_v], -1)
            base_uv = torch.tensor(base_uv, device=self.s2d.rgb.device).long() 

            for dst_id in dst_ids:
                depth_dst = self.cam_gs_depths[dst_id]
                dst_xyz, _, _, _, _ = cam_d_model(dst_id)
                add_buffer = dst_xyz
                render_dict = render_cam_canonical(xyz, rot, scale, opa, sph, self.s2d.H, self.s2d.W, self.color_mlp, add_buffer, HDR_mode=True)
                rendered_xyz_map = render_dict["buf"].permute(1, 2, 0)  # H,W,3

                pix_coords = rendered_xyz_map[..., 0:2]    ## in camera canonical space
                depth_dst2src_proj = rendered_xyz_map[..., 2]
                rgb_h_dst = (self.cam_gs_imgs_h[dst_id]/self.cam_gs_imgs_h[dst_id].max()).permute(1,2,0).cpu().numpy()
                h = tonemapReinhard.process(rgb_h_dst) 
                h = torch.tensor(h).cuda().permute(2,0,1)
                rgb_h_dst2src = torch.nn.functional.grid_sample(h.unsqueeze(0), pix_coords.unsqueeze(0).detach(), padding_mode="border", align_corners=True)[0]
                # torchvision.utils.save_image(rgb_h_dst2src, 'aa_{}'.format(dst_id) + ".png")
                # proj_mask_x = (pix_coords[..., 0] >= -1) & (pix_coords[..., 0] <= 1)
                # proj_mask_y = (pix_coords[..., 1] >= -1) & (pix_coords[..., 1] <= 1)
                # inside_mask = proj_mask_x & proj_mask_y

            # rgb_h_dst2src = torch.nn.functional.grid_sample(rgb_h_dst.unsqueeze(0), pix_coords.unsqueeze(0).detach(), padding_mode="border", align_corners=True)[0]
                # depth_dst2src_sample = torch.nn.functional.grid_sample(depth_dst.unsqueeze(0).unsqueeze(0), pix_coords.unsqueeze(0), padding_mode="border", align_corners=True)[0][0]
                # occ_mask = depth_dst2src_proj > 1.1 * depth_dst2src_sample
                # mask = (inside_mask & (~occ_mask)).detach()


                cam_points = backproject_depth(depth_src.unsqueeze(0), K_44_inv.detach().unsqueeze(0))
                T_rel = self.cams.T_cw(dst_id) @ self.cams.T_wc(src_id)
                pix_coords_pose, _, _ = project_3d(cam_points, K_44.detach().unsqueeze(0), T_rel)
                # rgb_h_dst2src = torch.nn.functional.grid_sample(h.unsqueeze(0), pix_coords_pose.detach(), padding_mode="border", align_corners=True)[0]
                # torchvision.utils.save_image(rgb_h_dst2src, 'aa_{}'.format(dst_id) + ".png")


                # flow_ind = self.s2d.flow_ij_to_listind_dict[(src_id, dst_id)]
                # flow = self.s2d.flow[flow_ind].detach().clone()
                # flow_mask = self.s2d.flow_mask[flow_ind].detach().clone().bool()
                # track_src = base_uv.clone().detach()

                # track_dst = track_src.float() + flow
                # pix_coords[..., 0] = 2*track_dst[..., 0] / self.s2d.W - 1
                # pix_coords[..., 1] = 2*track_dst[..., 1] / self.s2d.H - 1

                error = ((pix_coords_pose[0] - pix_coords)**2).sum(-1)
                error = torch.sqrt(error+1e-15)
                print(error.min(),error.max(),error.mean())
                mask = error>0.04
                torchvision.utils.save_image(mask.unsqueeze(0).float(), 'mask_{}'.format(dst_id) + ".png")



        exit()

    def generate_flow_error_by_world_gs(self):
        backproject_depth = BackprojectDepth(1, self.s2d.H, self.s2d.W).cuda()
        project_3d = Project3D(1, self.s2d.H, self.s2d.W).cuda()
        K = self.cams.K(self.s2d.H, self.s2d.W)
        K_44 = torch.eye(4, dtype=torch.float32).cuda()
        K_44[:3, :3] = K  
        K_44_inv = torch.inverse(K_44)

        err_list = []
        for src_id in range(self.s2d.T):
            # if src_id !=17:
            #     continue
            # if src_id == 0:
            #     dst_ids = [1]
            # elif src_id == self.s2d.T-1:
            #     dst_ids = [self.s2d.T-2]
            # else:
            #     dst_ids = [src_id-1, src_id+1]
            # if src_id in [0, 1, 2]:
            #     dst_ids = [src_id+3]
            # elif src_id in [self.s2d.T-3, self.s2d.T-2, self.s2d.T-1]:
            #     dst_ids = [src_id-3]
            # else:
            #     dst_ids = [src_id-3, src_id+3]
            if src_id in [0, 1]:
                dst_ids = [src_id+2]
            elif src_id in [self.s2d.T-2, self.s2d.T-1]:
                dst_ids = [src_id-2]
            else:
                dst_ids = [src_id-2, src_id+2]                      

            gs5_s = list(self.s_model())
            if self.d_model:
                gs5_src = [gs5_s, list(self.d_model(src_id))]
            else:
                gs5_src = [gs5_s]
            T_cw = self.cams.T_cw(src_id)

            errors = []
            for dst_id in dst_ids:
                dst_xyz = gs5_s[0]
                if self.d_model:
                    dst_xyz = torch.cat([dst_xyz, list(self.d_model(dst_id))[0]], 0)
                dst_xyz_cam = self.cams.trans_pts_to_cam(dst_id, dst_xyz)
                add_buffer = dst_xyz_cam
                render_dict = render_native(gs5_src, self.s2d.H, self.s2d.W, K, T_cw, self.color_mlp, add_buffer=add_buffer, HDR_mode=True)
                depth_src = render_dict["dep"][0]
                rendered_xyz_map = render_dict["buf"].permute(1, 2, 0)  # H,W,3
                warped_xyz_cam = rendered_xyz_map.reshape([-1, 3])

                pix_coords = self.cams.project(warped_xyz_cam).reshape([self.s2d.H, self.s2d.W, 2])
                L = min(self.s2d.W, self.s2d.H)
                pix_coords[..., :1] = (pix_coords[..., :1] + self.s2d.W / L) / 2.0 * L
                pix_coords[..., 1:] = (pix_coords[..., 1:] + self.s2d.H / L) / 2.0 * L

                pix_coords[..., :1] /= self.s2d.W - 1
                pix_coords[..., 1:] /= self.s2d.H - 1
                pix_coords = (pix_coords - 0.5) * 2

    
                # rgb_h_dst = (self.cam_gs_imgs_h[dst_id]/self.cam_gs_imgs_h[dst_id].max()).permute(1,2,0).cpu().numpy()
                # h = tonemapReinhard.process(rgb_h_dst) 
                # h = torch.tensor(h).cuda().permute(2,0,1)
                # rgb_h_dst2src = torch.nn.functional.grid_sample(h.unsqueeze(0), pix_coords.unsqueeze(0).detach(), padding_mode="border", align_corners=True)[0]
                # torchvision.utils.save_image(rgb_h_dst2src, 'aa_{}'.format(dst_id) + ".png")
                # proj_mask_x = (pix_coords[..., 0] >= -1) & (pix_coords[..., 0] <= 1)
                # proj_mask_y = (pix_coords[..., 1] >= -1) & (pix_coords[..., 1] <= 1)
                # inside_mask = proj_mask_x & proj_mask_y


                cam_points = backproject_depth(depth_src.unsqueeze(0), K_44_inv.detach().unsqueeze(0))
                T_rel = self.cams.T_cw(dst_id) @ self.cams.T_wc(src_id)
                pix_coords_pose, _, _ = project_3d(cam_points, K_44.detach().unsqueeze(0), T_rel)
                # rgb_h_dst2src = torch.nn.functional.grid_sample(h.unsqueeze(0), pix_coords_pose.detach(), padding_mode="border", align_corners=True)[0]
                # torchvision.utils.save_image(rgb_h_dst2src, 'aa_{}'.format(dst_id) + ".png")


                # flow_ind = self.s2d.flow_ij_to_listind_dict[(src_id, dst_id)]
                # flow = self.s2d.flow[flow_ind].detach().clone()
                # flow_mask = self.s2d.flow_mask[flow_ind].detach().clone().bool()
                # track_src = base_uv.clone().detach()

                # track_dst = track_src.float() + flow
                # pix_coords[..., 0] = 2*track_dst[..., 0] / self.s2d.W - 1
                # pix_coords[..., 1] = 2*track_dst[..., 1] / self.s2d.H - 1

                error = ((pix_coords_pose[0] - pix_coords)**2).sum(-1)
                error = torch.sqrt(error+1e-15)
                errors.append(error)
            errors = torch.stack(errors, dim=0)
            error = torch.min(errors, dim=0)[0]
            err_list.append(error)
        err_list = torch.stack(err_list, dim=0)

        return err_list
            # print(err_list.min(),err_list.max(),err_list.mean())
            # mask = err_list>0.02
            # torchvision.utils.save_image(mask.unsqueeze(0).float(), 'mask_{}'.format(src_id) + ".png")




    def get_init_gs_mask(self, tracks_pixel_3d, tracks_world_3d):
        ## tracks_pixel_3d are in camera canonical (pixel) space
        ## tracks_world_3d are in world space

        if self.s2d.has_epi:
            EPI_TH = getattr(self.args, "epi_th", 0.00005)
            DYN_ID_CNT = getattr(self.args, "dyn_id_cnt", 2 * 4)
            tracks_3d = tracks_pixel_3d.clone().permute(1, 0, 2)
            tracks_3d[..., 0] = (tracks_3d[..., 0] + 1.0) * self.s2d.W / 2 - 0.5
            tracks_3d[..., 1] = (tracks_3d[..., 1] + 1.0) * self.s2d.H / 2 - 0.5

            tracks_mask = (tracks_3d[..., 0] >= 0) * (tracks_3d[..., 1] >= 0) * (tracks_3d[..., 0] < self.s2d.W) * (tracks_3d[..., 1] < self.s2d.H) * (tracks_3d[..., 2] >= 0.02)

            flow_epi = self.s2d.epi.clone()
            # self.mask = flow_epi > 0.00001

            # depth = self.s2d.dep.clone()
            depth = self.cam_gs_depths.clone()

            with torch.no_grad():
                depth_list = query_buffers_by_track(depth[..., None], tracks_3d, tracks_mask, default_value=1e10).squeeze(-1)
                visible = tracks_3d[..., 2] <= 1.05 * depth_list 
                floater_mask = tracks_3d[..., 2] < 0.85 * depth_list 
                floater_mask = floater_mask.sum(0) == self.s2d.T

                occ_mask = tracks_mask.sum(0) == 0
                remain_mask = (~occ_mask) 

                epi_error_list = query_buffers_by_track(flow_epi[..., None], tracks_3d, tracks_mask).squeeze(-1)

            # DYN_ID_CNT = 0.15 * self.s2d.T
            is_static, is_dynamic = identify_tracks(epi_error_list, EPI_TH, static_cnt=1, dynamic_cnt=DYN_ID_CNT)
            is_static = is_static * remain_mask
            is_dynamic = is_dynamic * remain_mask
        else:
            std_per_gaussian = torch.std(tracks_world_3d, dim=1)  # [N_gaussians, 3]
            mean_std = torch.mean(std_per_gaussian, dim=1)  
            is_static = mean_std < 0.02
            is_dynamic = mean_std > 0.02


        return is_static, is_dynamic

    def s_act(self, x):
        max_s_value = self.world_gs_radius_max
        min_s_value = 0.0
        if isinstance(x, float):
            x = torch.tensor(x).squeeze()
        return min_s_value + torch.sigmoid(x) * (max_s_value - min_s_value)

    def s_inv_act(self, x):
        max_s_value = self.world_gs_radius_max
        min_s_value = 0.0
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

    def get_init_dynamic_gs_from_cam_gs(self, cam_d_model, gs_world_track, gs_world_rot_q, gs_cam_track, gs_cam_cov2d, is_dynamic, base_time_idx=0, frames_per_control_point=1):
        logging.info("Get initial dynamic world-gs model from camera-gs model (optimization for rot and scale).")

        features_dc = cam_d_model._features_dc[is_dynamic].detach().clone()
        opacities = cam_d_model.get_opacity[is_dynamic].detach().clone() 
        d_pos = gs_world_track[is_dynamic].detach().clone()
        base_position, pos_cubic_node, control_points_idx = self.get_dyn_gs_pos_cubic(d_pos.permute(1,0,2), base_time_idx, frames_per_control_point)
        rot_poly_feat = cam_d_model._rot_poly_feat[is_dynamic].detach().clone()
        scales = cam_d_model.get_scaling[is_dynamic].detach().clone()
        d_scales_op = nn.Parameter(self.s_inv_act(scales)) 

        rot_poly_feat_op = nn.Parameter(rot_poly_feat)  
        optimizer = torch.optim.Adam([{'params': rot_poly_feat_op, 'lr': 0.001}, {'params': d_scales_op, 'lr': 0.1}])
        loss_list = []
        for step in tqdm(range(1000)):
            time_idx = random.randint(0, self.s2d.T - 1)
            normed_time = time_idx / (self.s2d.T-1)
            _rot_poly_feat = rot_poly_feat_op.reshape(d_pos.shape[0], -1, 4)
            poly_feature_dim = _rot_poly_feat.shape[1]
            basis = torch.arange(poly_feature_dim).float().cuda()
            poly_basis = torch.pow(normed_time, basis.detach())[None, :, None]
            quaternion = torch.sum(_rot_poly_feat * poly_basis, dim=1)
            quaternion_norm = quaternion / (torch.norm(quaternion, p=2, dim=1, keepdim=True) + 1e-5)
            quaternion_prior = gs_world_rot_q[is_dynamic, time_idx, :].detach().clone()
            dot = torch.sum(quaternion_norm * quaternion_prior, dim=-1)  
            loss = 1 - dot.abs().mean()
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
      
        plt.figure(figsize=(10, 5))
        plt.plot(loss_list, label='Training Loss', color='blue', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        _log_dir = self.args.ws
        os.makedirs(_log_dir, exist_ok=True)
        plt.savefig(osp.join(_log_dir, "loss_curve2.png"))  

        K = self.cams.K(self.s2d.H, self.s2d.W)
        loss_list = []
        for step in tqdm(range(1000)):
            S = torch.zeros(d_pos.shape[0], 3, 3).cuda()
            S[:, 0, 0] = self.s_act(d_scales_op[:, 0])
            S[:, 1, 1] = self.s_act(d_scales_op[:, 1])
            S[:, 2, 2] = self.s_act(d_scales_op[:, 2])
            time_idx = random.randint(0, self.s2d.T - 1)
            normed_time = time_idx / (self.s2d.T-1)
            _rot_poly_feat = rot_poly_feat_op.reshape(d_pos.shape[0], -1, 4)
            poly_feature_dim = _rot_poly_feat.shape[1]
            basis = torch.arange(poly_feature_dim).float().cuda()
            poly_basis = torch.pow(normed_time, basis.detach())[None, :, None]
            quaternion = torch.sum(_rot_poly_feat * poly_basis, dim=1)
            d_rot_mat = quaternion_to_matrix(quaternion)
            xyz_cam = gs_cam_track[is_dynamic, time_idx, :]
            valid_mask = xyz_cam[:, -1] > 0.02
            with torch.no_grad():
                J_world = batched_J_world(xyz_cam[valid_mask], self.s2d.H, self.s2d.W, CAM_K=K)
                T_cw = self.cams.T_cw(time_idx)
                d_rot_cam = torch.einsum("ij,njk->nik", T_cw[:3, :3], d_rot_mat[valid_mask])
            cov3d = d_rot_cam @ (S[valid_mask]**2) @ d_rot_cam.permute(0, 2, 1)
            cov2d = J_world @ cov3d @ J_world.permute(0, 2, 1)
            cov2d = cov2d[:, 0:2, 0:2].reshape(cov3d.shape[0], -1)
            cov2d_cam = gs_cam_cov2d[is_dynamic, time_idx, :][valid_mask].detach()
            loss = (cov2d_cam - cov2d).abs().mean()
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        plt.figure(figsize=(10, 5))
        plt.plot(loss_list, label='Training Loss', color='blue', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        _log_dir = self.args.ws
        os.makedirs(_log_dir, exist_ok=True)
        plt.savefig(osp.join(_log_dir, "loss_curve3.png"))  

        scales = self.s_act(d_scales_op).detach().clone()
        rot_poly_feat = rot_poly_feat_op.detach().clone()
        d_model = DynamicGaussian(base_position, pos_cubic_node, control_points_idx, rot_poly_feat, scales, features_dc, opacities, self.s2d.T, max_scale=self.world_gs_radius_max, min_scale=0.0, base_time_idx=cam_d_model.base_time_idx.item())
        return d_model



    def get_init_static_gs_from_cam_gs(self, cam_d_model, gs_world_track, gs_world_rot_q, gs_cam_track, gs_cam_cov2d, is_static):
        logging.info("Get initial static world-gs model from camera-gs model (optimization for rot and scale).")

        s_pos = gs_world_track[is_static].detach().clone().mean(dim=1)
        s_opacities = cam_d_model.get_opacity[is_static].detach().clone()
        s_features_dc = cam_d_model._features_dc[is_static].detach().clone()


        # s_rot = gs_world_rot_q[:, 0, :][is_static].detach().clone()
        s_rot = batched_quaternion_average(gs_world_rot_q[is_static]).detach().clone()
        s_rot_op = nn.Parameter(s_rot)  
        s_scales = cam_d_model.get_scaling[is_static].detach().clone()
        s_scales_op = nn.Parameter(self.s_inv_act(s_scales))  
        optimizer = torch.optim.Adam([{'params': s_scales_op, 'lr': 0.1}, {'params': s_rot_op, 'lr': 0.001}])

        K = self.cams.K(self.s2d.H, self.s2d.W)
        loss_list = []
        for _ in tqdm(range(1000)):
            s_rot_mat = quaternion_to_matrix(s_rot_op)
            S = torch.zeros(s_pos.shape[0], 3, 3).cuda()
            S[:, 0, 0] = self.s_act(s_scales_op[:, 0])
            S[:, 1, 1] = self.s_act(s_scales_op[:, 1])
            S[:, 2, 2] = self.s_act(s_scales_op[:, 2])
            time_idx = random.randint(0, self.s2d.T - 1)
            xyz_cam = gs_cam_track[is_static, time_idx, :]
            valid_mask = xyz_cam[:, -1] > 0.02
            with torch.no_grad():
                J_world = batched_J_world(xyz_cam[valid_mask], self.s2d.H, self.s2d.W, CAM_K=K)
                T_cw = self.cams.T_cw(time_idx)
                s_rot_cam = torch.einsum("ij,njk->nik", T_cw[:3, :3], s_rot_mat[valid_mask])
            cov3d = s_rot_cam @ (S[valid_mask]**2) @ s_rot_cam.permute(0, 2, 1)
            cov2d = J_world @ cov3d @ J_world.permute(0, 2, 1)
            cov2d = cov2d[:, 0:2, 0:2].reshape(cov3d.shape[0], -1)
            cov2d_cam = gs_cam_cov2d[is_static, time_idx, :][valid_mask].detach()
            loss = (cov2d_cam - cov2d).abs().mean()
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        plt.figure(figsize=(10, 5))
        plt.plot(loss_list, label='Training Loss', color='blue', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        _log_dir = self.args.ws
        os.makedirs(_log_dir, exist_ok=True)
        plt.savefig(osp.join(_log_dir, "loss_curve1.png"))  

        s_scales = self.s_act(s_scales_op).detach().clone()
        s_rot = s_rot_op.detach().clone()
        s_model = StaticGaussian(s_pos, s_rot, s_scales, s_features_dc, s_opacities, max_scale=self.world_gs_radius_max, min_scale=0.0)

        return s_model


    def get_init_static_gs_from_tracks(self, tracks_3d):
        ## tracks_3d are in world space
        tracks_3d = tracks_3d.permute(1,0,2)  # [T, N, 3]
        tracks_3d = torch.stack([x[~torch.isnan(x).any(dim=1)] for x in tracks_3d], dim=0)
        
        xyz = tracks_3d.mean(0)
        N = xyz.shape[0]
        rot = torch.zeros((N, 4), dtype=torch.float32, device="cuda")
        rot[:, 0] = 1

        avg_dist = torch.clamp_min(distCUDA2(xyz.float().cuda()), 0.0000001)[..., None]
        scales = torch.sqrt(avg_dist).repeat(1, 3)
        
        features_dc = torch.randn(N, self.color_feature_dim).float().cuda()
        opacities = 0.5 * torch.ones((N, 1), dtype=torch.float, device="cuda")

        s_model = StaticGaussian(xyz, rot, scales, features_dc, opacities, max_scale=self.world_gs_radius_max, min_scale=0.0)

        return s_model
      
    
    def get_init_dyn_gs_from_tracks(self, tracks_3d, base_time_idx=0, poly_feature_dim=4, frames_per_control_point=1):
        ## tracks_3d are in world space
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

        d_model = DynamicGaussian(base_position, pos_cubic_node, control_points_idx, rot_poly_feat, scales, features_dc, opacities, self.s2d.T, max_scale=self.world_gs_radius_max, min_scale=0.0, base_time_idx=base_time_idx)

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

        return base_position, pos_cubic_node, control_points_idx

    def get_tone_mapper(self, hidden=128, act="relu", pretrain_iters=1000):
        tone_mapper = ToneMapper(hidden, act).cuda()
        tone_mapper_optimizer = torch.optim.Adam(list(tone_mapper.parameters()), lr=5e-4, eps=1e-15)
        logging.info("Pre-train the tone mapper to ensure a proper initialization.")
        for i in range(pretrain_iters):
            # ln_x = (torch.rand(1024, 3, requires_grad=True).cuda() * 6.0 - 3.0)
            # rgb_l = tone_mapper(ln_x)
            # gt_rgb = torch.clip(torch.log(torch.exp(ln_x)+1), 0, 1)
            # loss = (rgb_l - gt_rgb).norm(dim=1).mean()
            loss = CRF_RGB_equal_loss(tone_mapper) + CRF_monotonic_loss(tone_mapper) + unit_expos_loss(tone_mapper, self.tone_mapper_loss_gt)
            loss.backward()
            tone_mapper_optimizer.step()
            tone_mapper_optimizer.zero_grad()
        return tone_mapper

    def transfer_dyn_to_sta(self):
        EPI_TH = 0.02
        DYN_ID_CNT = getattr(self.args, "dyn_id_cnt", 2 * 4)
    
        gs_world_track = []
        gs_world_rot_q = []
        ## tracks_3d should be in pixel space
        tracks_3d = []
        for i in range(self.s2d.T):
            xyz_w = self.d_model.get_position_at_t(i)
            rot_q = self.d_model.get_quaternion_at_t(i)
            gs_world_track.append(xyz_w)
            gs_world_rot_q.append(rot_q)
            xyz = self.cams.trans_pts_to_cam(i, xyz_w)
            cam_K = self.cams.K(self.s2d.H, self.s2d.W)
            fx, fy = cam_K[0, 0], cam_K[1, 1]
            xyz[:, 0] = xyz[:, 0] / (xyz[:, 2] + 1e-5) * fx + self.s2d.W / 2
            xyz[:, 1] = xyz[:, 1] / (xyz[:, 2] + 1e-5) * fy + self.s2d.H / 2
            tracks_3d.append(xyz)
        tracks_3d = torch.stack(tracks_3d, dim=0)   # T N 3
        gs_world_track = torch.stack(gs_world_track, dim=1)   # N T 3
        gs_world_rot_q = torch.stack(gs_world_rot_q, dim=1)   # N T 4
        
        tracks_mask = (tracks_3d[..., 0] >= 0) * (tracks_3d[..., 1] >= 0) * (tracks_3d[..., 0] < self.s2d.W) * (tracks_3d[..., 1] < self.s2d.H) * (tracks_3d[..., 2] >= 0.02)

        with torch.no_grad():
            flow_epi = self.generate_flow_error_by_world_gs()
        depth = self.s2d.dep.clone()
        # depth = self.cam_gs_depths.clone()
       
        with torch.no_grad():
            depth_list = query_buffers_by_track(depth[..., None], tracks_3d, tracks_mask, default_value=1e10).squeeze(-1)
            visible = tracks_3d[..., 2] <= 1.04 * depth_list 
            # visible = (tracks_3d[..., 2] <= 1.04 * depth_list) * (tracks_3d[..., 2] >= 0.96 * depth_list)
            floater_mask = tracks_3d[..., 2] < 0.85 * depth_list 
            floater_mask = floater_mask.sum(0) == self.s2d.T

            tracks_mask = tracks_mask * visible
            # tracks_dep_mask, _ = gather_track_from_buffer(tracks_3d[..., :2].long(), tracks_mask, self.s2d.dep_mask)
            # tracks_mask = tracks_dep_mask.squeeze(-1) * tracks_mask

            occ_mask = tracks_mask.sum(0) == 0
            remain_mask = (~occ_mask) 
            # tracks_mask = tracks_mask * (tracks_3d[..., 2] >= 0.96 * depth_list)
            epi_error_list = query_buffers_by_track(flow_epi[..., None], tracks_3d, tracks_mask).squeeze(-1)

        # DYN_ID_CNT = 0.15 * self.s2d.T
        _, is_dynamic = identify_tracks(epi_error_list, EPI_TH, static_cnt=1, dynamic_cnt=DYN_ID_CNT)
        is_dynamic = is_dynamic * remain_mask


        s_pos = gs_world_track[~is_dynamic].detach().clone().mean(dim=1)
        s_opacities = self.d_model.get_opacity[~is_dynamic].detach().clone()
        s_features_dc = self.d_model._features_dc[~is_dynamic].detach().clone()
        s_rots = batched_quaternion_average(gs_world_rot_q[~is_dynamic]).detach().clone()
        s_scales = self.d_model.get_scaling[~is_dynamic].detach().clone()
        self.s_model.append_new_gs(self.optimizer_s_model, s_pos, s_rots, s_scales, s_opacities, s_features_dc)
        
        self.d_model._prune_points(self.optimizer_d_model, ~is_dynamic)



    def remove_error_dyn_gs(self):
        EPI_TH = getattr(self.args, "epi_th", 0.00005)
        DYN_ID_CNT = getattr(self.args, "dyn_id_cnt", 2 * 4)
    
        gs_world_track = []
        gs_world_rot_q = []
        ## tracks_3d should be in pixel space
        tracks_3d = []
        for i in range(self.s2d.T):
            xyz_w = self.d_model.get_position_at_t(i)
            rot_q = self.d_model.get_quaternion_at_t(i)
            gs_world_track.append(xyz_w)
            gs_world_rot_q.append(rot_q)
            xyz = self.cams.trans_pts_to_cam(i, xyz_w)
            cam_K = self.cams.K(self.s2d.H, self.s2d.W)
            fx, fy = cam_K[0, 0], cam_K[1, 1]
            xyz[:, 0] = xyz[:, 0] / (xyz[:, 2] + 1e-5) * fx + self.s2d.W / 2
            xyz[:, 1] = xyz[:, 1] / (xyz[:, 2] + 1e-5) * fy + self.s2d.H / 2
            tracks_3d.append(xyz)
        tracks_3d = torch.stack(tracks_3d, dim=0)   # T N 3
        gs_world_track = torch.stack(gs_world_track, dim=1)   # N T 3
        gs_world_rot_q = torch.stack(gs_world_rot_q, dim=1)   # N T 4
        
        tracks_mask = (tracks_3d[..., 0] >= 0) * (tracks_3d[..., 1] >= 0) * (tracks_3d[..., 0] < self.s2d.W) * (tracks_3d[..., 1] < self.s2d.H) * (tracks_3d[..., 2] >= 0.02)

        flow_epi = self.s2d.epi.clone()

        depths_before = []
        with torch.no_grad():
            for i in range(self.s2d.T):
                gs5 = [list(self.s_model()), list(self.d_model(i))]
                T_cw = self.cams.T_cw(i)
                K = self.cams.K(self.s2d.H, self.s2d.W)
                render_dict = render_native(gs5, self.s2d.H, self.s2d.W, K, T_cw, self.color_mlp, HDR_mode=True)
                depth = render_dict["dep"]
                depths_before.append(depth)
            depths_before = torch.cat(depths_before, dim=0)

        # depth = self.s2d.dep.clone()
        # depth = self.cam_gs_depths.clone()
        depth = depths_before.clone()

        a = (tracks_3d[-1, :, 0] < 350) * (tracks_3d[-1, :, 1] < 220) * (tracks_3d[-1, :, 0]> 285) * (tracks_3d[-1, :, 1] >180)
        # print(a.sum())
        with torch.no_grad():
            depth_list = query_buffers_by_track(depth[..., None], tracks_3d, tracks_mask, default_value=1e10).squeeze(-1)
            visible = tracks_3d[..., 2] <= 1.05 * depth_list
            # visible = (tracks_3d[..., 2] <= 1.04 * depth_list) * (tracks_3d[..., 2] >= 0.96 * depth_list)
            floater_mask = tracks_3d[..., 2] < 0.85 * depth_list 
            floater_mask = floater_mask.sum(0) == self.s2d.T

            new_tracks_mask = tracks_mask * visible
            # tracks_dep_mask, _ = gather_track_from_buffer(tracks_3d[..., :2].long(), tracks_mask, self.s2d.dep_mask)
            # new_tracks_mask = tracks_dep_mask.squeeze(-1) * new_tracks_mask

            occ_mask = new_tracks_mask.sum(0) == 0
            remain_mask = (~occ_mask) 
            epi_error_list = query_buffers_by_track(flow_epi[..., None], tracks_3d, new_tracks_mask, default_value=0.0).squeeze(-1)

        # DYN_ID_CNT = 0.15 * self.s2d.T
        _, is_dynamic = identify_tracks(epi_error_list, EPI_TH, static_cnt=1, dynamic_cnt=DYN_ID_CNT)
        is_dynamic = is_dynamic * remain_mask
       

        # s_pos = gs_world_track[~is_dynamic].detach().clone().mean(dim=1)
        # s_opacities = self.d_model.get_opacity[~is_dynamic].detach().clone()
        # s_features_dc = self.d_model._features_dc[~is_dynamic].detach().clone()
        # s_rots = batched_quaternion_average(gs_world_rot_q[~is_dynamic]).detach().clone()
        # s_scales = self.d_model.get_scaling[~is_dynamic].detach().clone()
        # self.s_model.append_new_gs(self.optimizer_s_model, s_pos, s_rots, s_scales, s_opacities, s_features_dc)
        
        # self.d_model._prune_points(self.optimizer_d_model, ~is_dynamic)

        radii2D = torch.zeros(self.s2d.T, self.d_model.N).float().cuda()
        # max_radii2D = torch.zeros(self.d_model.N).float().cuda()
        visibility_all_times = torch.zeros_like(radii2D).bool().cuda()
        with torch.no_grad():
            for i in range(self.s2d.T):
                gs5 = [list(self.d_model(i))]
                T_cw = self.cams.T_cw(i)
                K = self.cams.K(self.s2d.H, self.s2d.W)
                render_dict = render_native(gs5, self.s2d.H, self.s2d.W, K, T_cw, self.color_mlp, HDR_mode=True)
                radii = render_dict["radii"].float()
                visib = render_dict["visibility_filter"]
                # max_radii2D[visib] = torch.max(max_radii2D[visib], radii[visib])
                radii2D[i, visib] = radii[visib]
                visibility_all_times[i] = visib

        max_radii2D = torch.max(radii2D, dim=0).values
        median_radii2D = torch.zeros_like(max_radii2D)
        for j in range(self.d_model.N):
            visible_radii = radii2D[:, j][visibility_all_times[:, j]]
            if visible_radii.numel() > 0:
                median_radii2D[j] = torch.median(visible_radii)
        prune_mask = (~is_dynamic) | ((max_radii2D - median_radii2D) > 5) | (max_radii2D > 20)

        if prune_mask.sum() == 0:
            return
        
        depths_after = []
        depths_err = []
        with torch.no_grad():
            for i in range(self.s2d.T):
                xyz, rot, scale, opa, sph = self.d_model(i)
                xyz, rot, scale, opa, sph = xyz[~prune_mask], rot[~prune_mask], scale[~prune_mask], opa[~prune_mask], sph[~prune_mask]
                gs5 = [list(self.s_model()), [xyz, rot, scale, opa, sph]]
                T_cw = self.cams.T_cw(i)
                K = self.cams.K(self.s2d.H, self.s2d.W)
                render_dict = render_native(gs5, self.s2d.H, self.s2d.W, K, T_cw, self.color_mlp, HDR_mode=True)
                depth = render_dict["dep"]
                depths_after.append(depth)
            depths_after = torch.cat(depths_after, dim=0)

            for i in range(self.s2d.T):
                dep_err = (depths_after[i] - depths_before[i]).abs() / (depths_after[i] + depths_before[i] + 1e-5)
                depths_err.append(dep_err)
                # mask =  (dep_err > self.args.remove_dep_err_thr)
                # from PIL import Image
                # mask_np = (mask.float().cpu().numpy() * 255).astype(np.uint8)  # 0 or 255
                # Image.fromarray(mask_np).save(f'threshold_{i}.png')
     
            depths_err = torch.stack(depths_err, dim=0)
            depths_err_list = query_buffers_by_track(depths_err[..., None], tracks_3d, tracks_mask, default_value=0.0).squeeze(-1)
            over_th_cnt = (depths_err_list > self.args.remove_dep_err_thr).sum(0)
            preserve_mask = over_th_cnt >= 8
            prune_mask = prune_mask & (~preserve_mask)

        self.d_model._prune_points(self.optimizer_d_model, prune_mask)
        
    def append_new_dyn_gs(self):
        gs_world_track = []
        for i in range(self.s2d.T):
            gs_world_track.append(self.d_model.get_position_at_t(i))
        gs_world_track = torch.stack(gs_world_track, dim=0)
 
        new_d_gs_track = []
        with torch.no_grad():
            for i in range(self.s2d.T):
                gs5 = [list(self.s_model()), list(self.d_model(i))]
                T_cw = self.cams.T_cw(i)
                K = self.cams.K(self.s2d.H, self.s2d.W)
                render_dict = render_native(gs5, self.s2d.H, self.s2d.W, K, T_cw, self.color_mlp, HDR_mode=True)
                rgb_h = render_dict["rgb"]
                expos = self.s2d.train_exposures[i].detach().clone()
                tmp = rgb_h.clone().reshape([3, -1]).transpose(0, 1)
                pixel_lnx = torch.log2(tmp + 1e-5) + expos
                rgb_l = self.tone_mapper(pixel_lnx).transpose(0, 1).reshape(rgb_h.shape) 
                rgb_l_gt = self.s2d.rgb[i].permute(2, 0, 1).clone()

                error_map = torch.abs(rgb_l - rgb_l_gt)
                mask =  (error_map.mean(dim=0) > 0.1)
                mask = mask * self.s2d.dyn_mask[i]
                kernel = np.ones((3,3), dtype=np.uint8) 
                mask = cv2.morphologyEx(mask.cpu().numpy(), cv2.MORPH_CLOSE, kernel, iterations=1)
                mask = torch.from_numpy(mask).cuda()
                if mask.sum() == 0:
                    continue

                y_coords, x_coords = torch.where(mask == 1)
                z_values = self.s2d.dep[i][y_coords, x_coords]
                xyz = torch.stack((x_coords, y_coords, z_values), dim=1)
                cam_K = self.cams.K(self.s2d.H, self.s2d.W)
                fx, fy = cam_K[0, 0], cam_K[1, 1]

                ## transform to camera space
                xyz[:, 0] = (xyz[:, 0] - self.s2d.W / 2) * xyz[:, 2] / fx
                xyz[:, 1] = (xyz[:, 1] - self.s2d.H / 2) * xyz[:, 2] / fy
                ## transform to world space
                xyz_w = self.cams.trans_pts_to_world(i, xyz)
                max_points = 1000
                if xyz_w.shape[0] > max_points:
                    indices = torch.randperm(xyz_w.shape[0])[:max_points]
                    xyz_w = xyz_w[indices]
                d_gs = self.d_model.get_position_at_t(i)
                knn_res = knn_points(xyz_w[None], d_gs[None], None, None, K=5)
                idx = knn_res.idx[0]
                A, B = idx.shape
                idx = idx.reshape([A*B])
                select_gs_world_track = torch.index_select(gs_world_track, dim=1, index=idx)  
                select_gs_world_track = select_gs_world_track.reshape([-1, A, B, 3]).mean(-2)
                select_gs_world_track[i, :, :] = xyz_w
                
                new_d_gs_track.append(select_gs_world_track)

        if len(new_d_gs_track) == 0:
            return
        
        new_d_gs_track = torch.cat(new_d_gs_track, dim=1)
        max_points = 15000
        if new_d_gs_track.shape[1] > max_points:
            indices = torch.randperm(new_d_gs_track.shape[1])[:max_points]
            new_d_gs_track = new_d_gs_track[:, indices]

        N = new_d_gs_track.shape[1]
        base_position, pos_cubic_node, _ = self.get_dyn_gs_pos_cubic(new_d_gs_track, base_time_idx=0, frames_per_control_point=self.args.frames_per_control_point)

        # base_rot = torch.zeros((N, 4), dtype=torch.float32, device="cuda")
        # base_rot[:, 0] = 1
        poly_feature_dim = 4
        rot_poly_feat = torch.zeros((N, poly_feature_dim, 4), dtype=torch.float32, device="cuda")
        rot_poly_feat[:, 0, 0] = 1
        rot_poly_feat = rot_poly_feat.reshape(N, -1)
        
        scales = self.d_model.get_scaling.mean() * torch.ones((N, 3), dtype=torch.float, device="cuda")
        features_dc = torch.randn(N, self.color_feature_dim).float().cuda()
        opacities = 0.99 * torch.ones((N, 1), dtype=torch.float, device="cuda")
        
        self.d_model.append_new_gs(self.optimizer_d_model, base_position, pos_cubic_node, rot_poly_feat, scales, opacities, features_dc)


    def append_new_sta_gs(self):
        new_s_gs_pos = []
        with torch.no_grad():
            for i in range(self.s2d.T):
                gs5 = [list(self.s_model()), list(self.d_model(i))]
                T_cw = self.cams.T_cw(i)
                K = self.cams.K(self.s2d.H, self.s2d.W)
                render_dict = render_native(gs5, self.s2d.H, self.s2d.W, K, T_cw, self.color_mlp, HDR_mode=True)

                alpha = render_dict["alpha"][0]
                mask = alpha < 0.1
                mask = mask * self.s2d.sta_mask[i]
                if mask.sum() == 0:
                    continue

                y_coords, x_coords = torch.where(mask == 1)
                z_values = self.s2d.dep[i][y_coords, x_coords]
                xyz = torch.stack((x_coords, y_coords, z_values), dim=1)
                cam_K = self.cams.K(self.s2d.H, self.s2d.W)
                fx, fy = cam_K[0, 0], cam_K[1, 1]

                ## transform to camera space
                xyz[:, 0] = (xyz[:, 0] - self.s2d.W / 2) * xyz[:, 2] / fx
                xyz[:, 1] = (xyz[:, 1] - self.s2d.H / 2) * xyz[:, 2] / fy
                ## transform to world space
                xyz_w = self.cams.trans_pts_to_world(i, xyz)
                max_points = 1000
                if xyz_w.shape[0] > max_points:
                    indices = torch.randperm(xyz_w.shape[0])[:max_points]
                    xyz_w = xyz_w[indices]
                new_s_gs_pos.append(xyz_w)

        if len(new_s_gs_pos) == 0:
            return
        
        new_s_gs_pos = torch.cat(new_s_gs_pos, dim=0)
    
        max_points = 10000
        if new_s_gs_pos.shape[0] > max_points:
            indices = torch.randperm(new_s_gs_pos.shape[0])[:max_points]
            new_s_gs_pos = new_s_gs_pos[indices]   

        N = new_s_gs_pos.shape[0]
        rots = torch.zeros((N, 4), dtype=torch.float32, device="cuda")
        rots[:, 0] = 1
        scales = self.s_model.get_scaling.mean() * torch.ones((N, 3), dtype=torch.float, device="cuda")
        features_dc = torch.randn(N, self.color_feature_dim).float().cuda()
        opacities = 0.99 * torch.ones((N, 1), dtype=torch.float, device="cuda")

        self.s_model.append_new_gs(self.optimizer_s_model, new_s_gs_pos, rots, scales, opacities, features_dc)

    def apply_gs_control(self,
        render_dict,
        model,
        gs_control_cfg,
        step,
        optimizer_gs,
        first_N=None,
        last_N=None,
        record_flag=True,
        size_th=1e10,
        dyn_gs=False
    ):
        if first_N is not None:
            assert last_N is None
            grad = render_dict["viewspace_points"].grad[:first_N]
            radii = render_dict["radii"][:first_N]
            visib = render_dict["visibility_filter"][:first_N]
        elif last_N is not None:
            assert first_N is None
            grad = render_dict["viewspace_points"].grad[-last_N:]
            radii = render_dict["radii"][-last_N:]
            visib = render_dict["visibility_filter"][-last_N:]
        else:
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

        if dyn_gs and step in gs_control_cfg.reset_steps:
            N_old = model.N
            self.remove_error_dyn_gs()
            logging.info(f"Remove: {N_old}->{self.d_model.N}")
        
        if step in gs_control_cfg.densify_steps:
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
                max_screen_size=size_threshold,  
            )
            logging.info(f"Prune: {N_old}->{model.N}")

        if step in gs_control_cfg.reset_steps:
            model.reset_opacity(optimizer_gs, gs_control_cfg.reset_opacity)


    def get_test_cam_pose(self):
        assert hasattr(self.s2d, "test_ldr_gt")

        K = self.cams.K(self.s2d.H, self.s2d.W)
        q_test_cams = []
        t_test_cams = []
        for i in range(len(self.s2d.test_ldr_gt)):
            q1 = F.normalize(self.cams.q_wc[i:i+1], dim=-1)
            q2 = F.normalize(self.cams.q_wc[i+1:i+2], dim=-1)
            try:
                qs = torch.cat([q1, q2], dim=0)
                q = batched_quaternion_average(qs.unsqueeze(0))
            except:
                q1[q1[:, 0] < 0] *= -1  # Flip quaternions where w < 0
                q2[q2[:, 0] < 0] *= -1
                q = (q1 + q2) / 2
                q = F.normalize(q, dim=-1)
            t1 = self.cams.t_wc[i:i+1]
            t2 = self.cams.t_wc[i+1:i+2]
            t = (t1 + t2) / 2
            q_test_cams.append(q)
            t_test_cams.append(t)

        q_test_cams = torch.cat(q_test_cams)
        t_test_cams = torch.cat(t_test_cams)
        delta_flag = self.cams.delta_flag.item()
        iso_focal = self.cams.iso_focal.item()
        self.test_cams = MonocularCameras(
            n_time_steps=len(self.s2d.test_ldr_gt),
            default_H=self.s2d.H,
            default_W=self.s2d.W,
            K=K,
            delta_flag=delta_flag,
            iso_focal=iso_focal
        )
        self.test_cams.q_wc = nn.Parameter(q_test_cams.detach())
        self.test_cams.t_wc = nn.Parameter(t_test_cams.detach())
        # test_cam_param_list = self.test_cams.get_optimizable_list(lr_q=1e-4, lr_t=1e-4)[:2]
        # self.optimizer_test_cams = torch.optim.Adam(test_cam_param_list)
        # view_ind_list = None
        # loss_list = []
        # for step in tqdm(range(1000)):
        #     if not view_ind_list:
        #         view_ind_list = list(range(self.test_cams.T))
        #         random.shuffle(view_ind_list)
        #     view_ind = view_ind_list.pop(0)
        #     exposure = self.s2d.test_exposures[view_ind].detach().clone()
        #     rgb_l_gt = self.s2d.test_ldr_gt[view_ind].permute(2, 0, 1).detach().clone()

        #     gs5 = [list(self.s_model())]
        #     if self.d_model:
        #         gs5.append(list(self.d_model(view_ind+0.5)))

        #     T_cw = self.test_cams.T_cw(view_ind)
        #     K = self.test_cams.K(self.s2d.H, self.s2d.W)
        #     render_dict = render_native(gs5, self.s2d.H, self.s2d.W, K, T_cw, self.color_mlp, HDR_mode=True)
        #     rgb_h = render_dict["rgb"]
        #     depth = render_dict["dep"][0]
        #     tmp = rgb_h.clone().reshape([3, -1]).transpose(0, 1)
        #     pixel_lnx = torch.log2(tmp + 1e-6) + exposure
        #     rgb_l = self.tone_mapper(pixel_lnx).transpose(0, 1).reshape(rgb_h.shape)  
        #     rgb_sup_mask = torch.ones_like(depth)
        #     loss, _ = compute_rgb_loss(rgb_l_gt.detach().clone(), rgb_l, rgb_sup_mask)
        #     loss.backward()
        #     self.optimizer_test_cams.step()
        #     self.optimizer_test_cams.zero_grad(set_to_none = True)
        #     loss_list.append(loss.item())


        # plt.figure(figsize=(10, 5))
        # plt.plot(loss_list, label='Training Loss', color='blue', linewidth=2)
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # plt.title('Training Loss Curve')
        # plt.grid(True, linestyle='--', alpha=0.6)
        # plt.legend()
        # plt.savefig('loss_curve4.png')  

        ckpt_dir = osp.join(self.out_dir, "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(self.test_cams.state_dict(), osp.join(ckpt_dir, "test_cams.pth"))
        

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


    def render_test(self):
        if not hasattr(self, "test_cams"):
            self.test_cams: MonocularCameras = MonocularCameras.load_from_ckpt(
                torch.load(osp.join(self.out_dir, "ckpt", "test_cams.pth"))
            ).cuda()

        save_dir = osp.join(self.out_dir, 'test')
        os.makedirs(save_dir+"_hdr_exr", exist_ok=True)
        os.makedirs(save_dir+"_hdr_tm", exist_ok=True)
        os.makedirs(save_dir+"_ldr", exist_ok=True)
        if hasattr(self.s2d, "test_ne_ldr_gt"):
            os.makedirs(save_dir+"_ne_ldr", exist_ok=True)

        images_h = []
        for i in range(self.test_cams.T):
            if self.d_model:
                gs5 = [list(self.s_model()), list(self.d_model(i+0.5))]
            else:
                gs5 = [list(self.s_model())]
            T_cw = self.test_cams.T_cw(i)
            K = self.test_cams.K(self.s2d.H, self.s2d.W)
            render_dict = render_native(gs5, self.s2d.H, self.s2d.W, K, T_cw, self.color_mlp, HDR_mode=True)
            rgb_h = render_dict["rgb"]
            expos = self.s2d.test_exposures[i].detach().clone()
            tmp = rgb_h.clone().reshape([3, -1]).transpose(0, 1)
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
            
        # images_h = images_h/(np.max(images_h))
        images_h = (images_h/np.percentile(images_h, 99.5)).clip(0,1)
        Reinhards = [] 
        for i, h in enumerate(images_h):
            h = (tonemapReinhard.process(h) * 255).astype(np.uint8)  ## use Reinhard’s method
            imageio.imwrite(osp.join(save_dir+"_hdr_tm", f'{self.s2d.test_frame_names[i]}.jpg'), h)
            Reinhards.append(h)
        images_h = np.stack(Reinhards, 0)      


    def render_train(self):
        ## static + dynamic
        images_l, depths, images_h = [], [], []
        render_times = []
        for i in range(self.s2d.T):
            if self.d_model:
                gs5 = [list(self.s_model()), list(self.d_model(i))]
            else:
                gs5 = [list(self.s_model())]
            T_cw = self.cams.T_cw(i)
            K = self.cams.K(self.s2d.H, self.s2d.W)
            start_time = time.time()
            render_dict = render_native(gs5, self.s2d.H, self.s2d.W, K, T_cw, self.color_mlp, HDR_mode=True)
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
        hdr_max = np.percentile(images_h, 99.5)
        images_h = (images_h/np.percentile(images_h, 99.5)).clip(0,1)
        Reinhards = [] 
        for i, h in enumerate(images_h):
            h = (tonemapReinhard.process(h) * 255).astype(np.uint8)  ## use Reinhard’s method
            imageio.imwrite(osp.join(save_dir+"_hdr_tm", f'{self.s2d.train_frame_names[i]}.jpg'), h)
            Reinhards.append(h)
        Reinhards_all = Reinhards.copy()
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

        if self.d_model is None:
            return
        
        ## static
        images_l, depths, images_h = [], [], []
        gs5 = [list(self.s_model())]
        for i in range(self.s2d.T):
            T_cw = self.cams.T_cw(i)
            K = self.cams.K(self.s2d.H, self.s2d.W)
            render_dict = render_native(gs5, self.s2d.H, self.s2d.W, K, T_cw, self.color_mlp, HDR_mode=True)
            rgb_h = render_dict["rgb"]
            expos = self.s2d.train_exposures[i].detach().clone()
            tmp = rgb_h.clone().reshape([3, -1]).transpose(0, 1)
            pixel_lnx = torch.log2(tmp + 1e-5) + expos
            rgb_l = self.tone_mapper(pixel_lnx).transpose(0, 1).reshape(rgb_h.shape) 
            depth = render_dict['dep'].permute(1,2,0).cpu().numpy()
            images_l.append((rgb_l.permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            images_h.append(rgb_h.permute(1,2,0).cpu().numpy())
            depths.append(depth)

        save_dir = osp.join(self.out_dir, 'train_static')
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

        imageio.mimwrite(osp.join(self.out_dir, 'train_video_hdr_static.mp4'), images_h, quality=8)
        # imageio.mimwrite(osp.join(self.out_dir, 'train_video_ldr_static.mp4'), images_l, quality=8)

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
        imageio.mimwrite(osp.join(self.out_dir, 'train_video_depth_static.mp4'), viz_depth_list, quality=8)

        ## dynamic
        images_l, depths, images_h, masks = [], [], [], []
        for i in range(self.s2d.T):
            gs5 = [list(self.d_model(i))]
            T_cw = self.cams.T_cw(i)
            K = self.cams.K(self.s2d.H, self.s2d.W)
            add_buffer = torch.ones([gs5[0][0].shape[0], 3]).float().cuda()
            render_dict = render_native(gs5, self.s2d.H, self.s2d.W, K, T_cw, self.color_mlp, add_buffer=add_buffer, HDR_mode=True)
            mask = (render_dict["buf"][0] > 0.0).float()
            rgb_h = render_dict["rgb"]
            expos = self.s2d.train_exposures[i].detach().clone()
            tmp = rgb_h.clone().reshape([3, -1]).transpose(0, 1)
            pixel_lnx = torch.log2(tmp + 1e-5) + expos
            rgb_l = self.tone_mapper(pixel_lnx).transpose(0, 1).reshape(rgb_h.shape) 
            depth = render_dict['dep'].permute(1,2,0).cpu().numpy()
            images_l.append((rgb_l.permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
            images_h.append(rgb_h.permute(1,2,0).cpu().numpy())
            depths.append(depth)
            masks.append(mask.unsqueeze(-1).cpu().numpy())

        save_dir = osp.join(self.out_dir, 'train_dynamic')
        os.makedirs(save_dir+"_hdr_exr", exist_ok=True)
        os.makedirs(save_dir+"_hdr_tm", exist_ok=True)
        os.makedirs(save_dir+"_ldr", exist_ok=True)
        for i, img in enumerate(images_h):
            cv2.imwrite(osp.join(save_dir+"_hdr_exr", f'{self.s2d.train_frame_names[i]}.exr'), img[:, :, ::-1])
            
        # images_h = images_h/(np.max(images_h))
        images_h = (images_h/hdr_max).clip(0,1)
        Reinhards = [] 
        for i, h in enumerate(images_h):
            # h = (tonemapReinhard.process(h) * 255).astype(np.uint8)  ## use Reinhard’s method
            h = (Reinhards_all[i].astype(np.float32) * masks[i]).astype(np.uint8)
            imageio.imwrite(osp.join(save_dir+"_hdr_tm", f'{self.s2d.train_frame_names[i]}.jpg'), h)
            Reinhards.append(h)
        images_h = np.stack(Reinhards, 0)      

        for i, l in enumerate(images_l):
            imageio.imwrite(osp.join(save_dir+"_ldr", f'{self.s2d.train_frame_names[i]}.jpg'), l)

        imageio.mimwrite(osp.join(self.out_dir, 'train_video_hdr_dynamic.mp4'), images_h, quality=8)
        # imageio.mimwrite(osp.join(self.out_dir, 'train_video_ldr_dynamic.mp4'), images_l, quality=8)

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
        imageio.mimwrite(osp.join(self.out_dir, 'train_video_depth_dynamic.mp4'), viz_depth_list, quality=8)


    def compute_dyn_tc_loss(self, render_dict_dst, dst_id):
        if dst_id == 0:
            src_ids = [1]
        elif dst_id == self.s2d.T-1:
            src_ids = [self.s2d.T-2]
        else:
            src_ids = [dst_id-1, dst_id+1]
        

        gs5_s = list(self.s_model())
        dst_xyz = gs5_s[0].detach().clone()
        if self.d_model:
            dst_xyz = torch.cat([dst_xyz, list(self.d_model(dst_id))[0]], 0)
        dst_xyz_cam = self.cams.trans_pts_to_cam(dst_id, dst_xyz)
        add_buffer = dst_xyz_cam
        K = self.cams.K(self.s2d.H, self.s2d.W)

        depth_dst = render_dict_dst["dep"][0]
        rgb_h_dst = render_dict_dst["rgb"]

        loss_dyn_tc = torch.tensor(0.0).cuda()
        for src_id in src_ids:
            if self.d_model:
                gs5_src = [gs5_s, list(self.d_model(src_id))]
            else:
                gs5_src = [gs5_s]
        
            T_cw_src = self.cams.T_cw(src_id)
            render_dict_src = render_native(gs5_src, self.s2d.H, self.s2d.W, K, T_cw_src, self.color_mlp, add_buffer=add_buffer, HDR_mode=True)

            rgb_h_src = render_dict_src["rgb"]
            rendered_xyz_map = render_dict_src["buf"].permute(1, 2, 0)  # H,W,3
            warped_xyz_cam = rendered_xyz_map.reshape([-1, 3])

            pix_coords = self.cams.project(warped_xyz_cam).reshape([self.s2d.H, self.s2d.W, 2])
            depth_dst2src_proj = warped_xyz_cam[:, 2].reshape([self.s2d.H, self.s2d.W])
            L = min(self.s2d.W, self.s2d.H)
            pix_coords[..., :1] = (pix_coords[..., :1] + self.s2d.W / L) / 2.0 * L
            pix_coords[..., 1:] = (pix_coords[..., 1:] + self.s2d.H / L) / 2.0 * L

            proj_mask_x = (pix_coords[..., 0] >= 0) & (pix_coords[..., 0] <= self.s2d.W-1)
            proj_mask_y = (pix_coords[..., 1] >= 0) & (pix_coords[..., 1] <= self.s2d.H-1)
            inside_mask = proj_mask_x & proj_mask_y

            pix_coords[..., :1] /= self.s2d.W - 1
            pix_coords[..., 1:] /= self.s2d.H - 1
            pix_coords = (pix_coords - 0.5) * 2
            rgb_h_dst2src = torch.nn.functional.grid_sample(rgb_h_dst.unsqueeze(0), pix_coords.unsqueeze(0).detach(), padding_mode="border", align_corners=True)[0]
            depth_dst2src_sample = torch.nn.functional.grid_sample(depth_dst.unsqueeze(0).unsqueeze(0), pix_coords.unsqueeze(0), padding_mode="border", align_corners=True)[0][0]
            occ_mask = depth_dst2src_proj > 1.1 * depth_dst2src_sample
            mask = (inside_mask & (~occ_mask)).detach()

            diff = (rgb_h_src - rgb_h_dst2src).abs() / (rgb_h_src + rgb_h_dst2src + 1e-10).detach()
            loss_dyn_tc += (diff * mask[None, ...]).sum() / (mask.sum() + 1e-10)
        loss_dyn_tc = loss_dyn_tc / len(src_ids)

        return loss_dyn_tc
        

    def train(self):
        view_ind_list = list(range(self.s2d.T))
        random.shuffle(view_ind_list)

        backproject_depth = BackprojectDepth(1, self.s2d.H, self.s2d.W).cuda()
        project_3d = Project3D(1, self.s2d.H, self.s2d.W).cuda()

        base_u, base_v = np.meshgrid(np.arange(self.s2d.W), np.arange(self.s2d.H))
        base_uv = np.stack([base_u, base_v], -1)
        base_uv = torch.tensor(base_uv, device=self.s2d.rgb.device).long()       

        track_loss_interval = getattr(self.args, "photo_track_loss_interval", 4)
        latest_track_event = 0
        track_loss_protect_steps = 100
        loss_rgb_list, loss_dep_list, loss_rigid_list, loss_track_list = [], [], [], []
        loss_vel_xyz_list, loss_vel_rot_list, loss_acc_xyz_list, loss_acc_rot_list = [], [], [], []
        loss_scale_var_list = []
        loss_reproj_list = []
        loss_opa_list = []
        loss_dyn_tc_list = []
        d_gs_n_count_list = []
        s_gs_n_count_list = []

        for step in tqdm(range(1, self.total_steps+1)):
            if not view_ind_list:
                view_ind_list = list(range(self.s2d.T))
                random.shuffle(view_ind_list)
            view_ind = view_ind_list.pop(0)

            # if step <= 6000:
            #     view_ind = np.random.randint(0, self.s2d.T)
            # else:
            #     first = 300
            #     remain = max(self.total_steps - 6000 - first, 0)
            #     per = remain // (self.s2d.T - 1)
            #     step_ranges = [first] + [per] * (self.s2d.T - 1)
            #     step_ranges[-1] += (self.total_steps - 6000) - sum(step_ranges)
            #     acc = 0
            #     for vid, rng in enumerate(step_ranges):
            #         if (step - 6001) < acc + rng:
            #             view_ind = vid
            #             break
            #         acc += rng
            # if step <= self.total_steps - 3000:
            #     first = 300
            #     remain = max(self.total_steps -3000- first, 0)
            #     per = remain // (self.s2d.T - 1) 
            #     # Step range of each view
            #     step_ranges = [first] + [per] * (self.s2d.T - 1)
            #     # Fix the last view so the sum does not exceed total_step
            #     step_ranges[-1] += self.total_steps - sum(step_ranges)
            #     # Find which view this step belongs to
            #     acc = 0
            #     for vid, rng in enumerate(step_ranges):
            #         if step < acc + rng:
            #             view_ind = vid
            #             break
            #         acc += rng

            
            exposure = self.s2d.train_exposures[view_ind].detach().clone()
            track_flow_interval_candidates = [len(torch.unique(self.s2d.train_exposures))]

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

            gs5 = [list(self.s_model())]
            xyz = gs5[0][0]
            dst_xyz = xyz.detach().clone()
            if self.d_model:
                gs5.append(list(self.d_model(view_ind)))
                xyz_d = gs5[-1][0]
                # xyz = torch.cat([xyz, xyz_d], 0)
                dst_xyz_d = self.d_model(corr_dst_ind)[0]
                dst_xyz = torch.cat([dst_xyz, dst_xyz_d], 0)
                ii, jj, nn, weight = cal_connectivity_from_points(points=xyz_d.clone(), K=5)
                pos = torch.stack([xyz_d.clone(), dst_xyz_d.clone()], dim=0)
                loss_rigid = cal_arap_error(pos, ii, jj, nn) 
            else:
                loss_rigid = torch.tensor(0.0).cuda()
            dst_xyz_cam = self.cams.trans_pts_to_cam(corr_dst_ind, dst_xyz)
            if corr_exe_flag:
                add_buffer = dst_xyz_cam
            else:
                add_buffer = torch.zeros_like(dst_xyz_cam).cuda()

            
            T_cw = self.cams.T_cw(view_ind)
            K = self.cams.K(self.s2d.H, self.s2d.W)
            render_dict = render_native(gs5, self.s2d.H, self.s2d.W, K, T_cw, self.color_mlp, add_buffer=add_buffer, HDR_mode=True)

            rgb_h = render_dict["rgb"]

            alpha = render_dict["alpha"][0]
            # loss_opa = (torch.ones_like(alpha, dtype=torch.float, device="cuda").detach() - alpha).abs().mean()
            opa_diff = (torch.ones_like(alpha, dtype=torch.float, device="cuda").detach() - alpha).abs()
            loss_opa = (opa_diff * opa_diff.detach()).sum() / (opa_diff.detach().sum() + 1e-5)

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


            K_44 = torch.eye(4, dtype=torch.float32).cuda()
            K_44[:3, :3] = K  
            K_44_inv = torch.inverse(K_44)
            
            ## compute reprojection loss
            loss_reproj = torch.tensor(0.0).cuda()
            if step < 3000:
                left = max(view_ind-3, 0)
                right = min(view_ind+3, self.s2d.T-1)
                if len(range(left, view_ind)) > 0:
                    reproj_src_inds = [np.random.choice(list(range(left, view_ind)))]
                else:
                    reproj_src_inds = []
                if len(range(view_ind+1, right+1)) > 0:
                    reproj_src_inds.append(np.random.choice(list(range(view_ind+1, right+1))))
                # reproj_src_inds = [np.random.choice(list(range(left, view_ind))), np.random.choice(list(range(view_ind+1, right+1)))]
                # reproj_src_inds = list(range(left, left+1))
                # reproj_src_inds = list(range(left, view_ind))+list(range(view_ind+1, right+1))
                # reproj_src_inds = [np.random.choice(reproj_src_inds)]
                depth_p = depth_gt.detach().clone() if np.random.rand() < 0.5 else depth.clone()
                cam_points = backproject_depth(depth_p.unsqueeze(0), K_44_inv.detach().unsqueeze(0))
                for src_ind in reproj_src_inds:
                    T_rel = self.cams.T_cw(src_ind) @ self.cams.T_wc(view_ind)
                    pix_coords, src_depth_warp, inside_mask = project_3d(cam_points, K_44.detach().unsqueeze(0), T_rel)

                    # src = self.cam_gs_imgs_h_tm[src_ind].permute(2, 0, 1).detach()
                    # img_tgt = self.cam_gs_imgs_h_tm[view_ind].permute(2, 0, 1).detach()
                    src = self.cam_gs_imgs_h[src_ind].detach()
                    img_tgt = self.cam_gs_imgs_h[view_ind].detach()
                    src_depth = self.s2d.dep[src_ind].detach().clone()
                    src_sta_mask = self.s2d.sta_mask[src_ind].detach()
                    tgt_sta_mask = self.s2d.sta_mask[view_ind].detach()
                    # torchvision.utils.save_image(src, 'srcf_{}'.format(2) + ".png")

                    img_src2tgt = torch.nn.functional.grid_sample(src.unsqueeze(0), pix_coords, padding_mode="border", align_corners=True)[0]
                    mask_src2tgt = torch.nn.functional.grid_sample(src_sta_mask.unsqueeze(0).unsqueeze(0), pix_coords, mode="nearest", padding_mode="border", align_corners=True)[0][0]
                    src_depth_sample = torch.nn.functional.grid_sample(src_depth.unsqueeze(0).unsqueeze(0), pix_coords, padding_mode="border", align_corners=True)[0][0]

                    reproj_mask = mask_src2tgt.bool() & inside_mask[0][0] & tgt_sta_mask.bool()

                    diff = (img_tgt - img_src2tgt).abs() / (img_tgt + img_src2tgt + 1e-5).detach()
                    loss_reproj += (diff * reproj_mask[None, ...]).sum() / (reproj_mask.sum() + 1e-5)
                    # loss_reproj += compute_rgb_loss(img_tgt, img_src2tgt, reproj_mask.detach())[0]
                    loss_reproj_dep_i = torch.abs((src_depth_sample - src_depth_warp[0][0])) * reproj_mask
                    loss_reproj += loss_reproj_dep_i.sum() / reproj_mask.sum()

                    # torchvision.utils.save_image(mask.unsqueeze(0).float(), 'mask_{}'.format(3) + ".png")
                    # torchvision.utils.save_image(img_src2tgt, 'src_{}'.format(3) + ".png")
                    # torchvision.utils.save_image(self.s2d.pseudo_hdr_tm[view_ind].permute(2, 0, 1), 'tgt_{}'.format(3) + ".png")
                    # torchvision.utils.save_image(self.s2d.sta_mask[src_ind], 'src_{}'.format(2) + ".png")
                loss_reproj = loss_reproj / len(reproj_src_inds)
  
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
                        pred_track_dst = self.cams.project(warped_xyz_cam)
                        L = min(self.s2d.W, self.s2d.H)
                        pred_track_dst[:, :1] = (pred_track_dst[:, :1] + self.s2d.W / L) / 2.0 * L
                        pred_track_dst[:, 1:] = (pred_track_dst[:, 1:] + self.s2d.H / L) / 2.0 * L
                        loss_track = (pred_track_dst - track_dst).norm(dim=-1)[track_loss_mask]
                        loss_track = loss_track.clamp(0.0, 100.0).mean()
            else:
                loss_track = torch.zeros_like(loss_rgb)

            scale_var = self.s_model.get_scaling.std(dim=1)
            if self.d_model:
                scale_var = torch.cat([scale_var, self.d_model.get_scaling.std(dim=1)])
            loss_scale_var = torch.mean(scale_var)
            
            reg_radius = 2
            _l = max(0, view_ind - reg_radius)
            _r = min(self.s2d.T, view_ind + 1 + reg_radius)
            tids = torch.arange(_l, _r).cuda()
            loss_vel_xyz, loss_vel_rot, loss_acc_xyz, loss_acc_rot = self.d_model.compute_vel_acc_loss(tids)

            if step >= max(self.d_ctrl_end_2, self.s_ctrl_end_1):
                loss_dyn_tc = self.compute_dyn_tc_loss(render_dict, view_ind)
            else:
                loss_dyn_tc = torch.tensor(0.0)

            loss = loss_rgb + self.args.weight_dyn_tc*loss_dyn_tc + 1.0 * loss_dep + loss_opa + 0.01*loss_track + 0.01 * loss_rigid + 10*unit_expos_loss(self.tone_mapper, self.tone_mapper_loss_gt) + 10 * CRF_monotonic_loss(self.tone_mapper)  + 10 * loss_scale_var + 10 * (loss_vel_xyz + loss_vel_rot + loss_acc_xyz + loss_acc_rot) + loss_reproj
            # loss = loss_rgb + 0.1*loss_dyn_tc + 1.0 * loss_dep + loss_opa + 0*loss_HDR_tc + 0.000*loss_HDR_scale + 0.01*loss_track + 0.01 * loss_rigid + 10*unit_expos_loss(self.tone_mapper, self.tone_mapper_loss_gt) + 10 * CRF_monotonic_loss(self.tone_mapper) + 0.0 * loss_reproj + 1 * loss_scale_var + 1 * (loss_vel_xyz + loss_vel_rot + loss_acc_xyz + loss_acc_rot) + 0.0 * loss_rgb_h
            # loss = loss_rgb + 0.1*loss_dyn_tc + 1.0 * loss_dep + loss_opa+0.01*loss_track+ 0.01 * loss_rigid + 10*unit_expos_loss(self.tone_mapper, self.tone_mapper_loss_gt) + 10 * CRF_monotonic_loss(self.tone_mapper) + 1 * loss_scale_var 
            loss_rgb_list.append(loss_rgb.item())
            loss_dep_list.append(loss_dep.item())
            loss_opa_list.append(loss_opa.item())
            loss_dyn_tc_list.append(loss_dyn_tc.item())
            loss_track_list.append(loss_track.item())
            loss_rigid_list.append(loss_rigid.item())
            loss_reproj_list.append(loss_opa.item())
            loss_vel_xyz_list.append(loss_vel_xyz.item())
            loss_vel_rot_list.append(loss_vel_rot.item())
            loss_acc_xyz_list.append(loss_acc_xyz.item())
            loss_acc_rot_list.append(loss_acc_rot.item())
            loss_scale_var_list.append(loss_scale_var.item())
            if self.d_model:
                d_gs_n_count_list.append(self.d_model.N)
            else:
                d_gs_n_count_list.append(0)
            s_gs_n_count_list.append(self.s_model.N)


            # t = time.time()
            loss.backward()
            # torch.cuda.synchronize()
            # print("  pytorch runtime: ", (time.time() - t), " s")

            # if self.d_model and step  == self.total_steps-1500:
            #     self.transfer_dyn_to_sta()

            if self.s_gs_ctrl_cfg is not None:
                if (step > self.s_ctrl_start_1 and step < self.s_ctrl_end_1):
                    self.apply_gs_control(
                        render_dict=render_dict,
                        model=self.s_model,
                        gs_control_cfg=self.s_gs_ctrl_cfg,
                        step=step,
                        optimizer_gs=self.optimizer_s_model,
                        first_N=self.s_model.N,
                        record_flag=True,
                        size_th=1e10
                        # record_flag=(not corr_exe_flag)
                    )
                    if step in self.s_gs_ctrl_cfg.reset_steps:
                        latest_track_event = step
                if step == 5000:
                    N_old = self.s_model.N
                    self.append_new_sta_gs()
                    logging.info(f"Add: {N_old}->{self.s_model.N}")

            if self.d_model is not None:
                if (step > self.d_ctrl_start_1 and step < self.d_ctrl_end_1) or (step > self.d_ctrl_start_2 and step < self.d_ctrl_end_2):
                # if (step > self.d_ctrl_start_1 and step < self.d_ctrl_end_2):
                    size_th=20 if step > 2000 else 1e10
                    self.apply_gs_control(
                        render_dict=render_dict,
                        model=self.d_model,
                        gs_control_cfg=self.d_gs_ctrl_cfg,
                        step=step,
                        optimizer_gs=self.optimizer_d_model,
                        last_N=self.d_model.N,
                        record_flag=True,
                        size_th=size_th,
                        dyn_gs=True
                        # record_flag=(not corr_exe_flag)
                    )
                    if step in self.d_gs_ctrl_cfg.reset_steps:
                        latest_track_event = step
                if step == self.d_ctrl_start_2 - 300:
                # if step == 5100 or step == 7100:
                    N_old = self.d_model.N
                    self.append_new_dyn_gs()
                    logging.info(f"Add: {N_old}->{self.d_model.N}")
                if step in [self.d_ctrl_end_2 - 500, self.total_steps - 1000]:
                    self.remove_error_dyn_gs()

            self.optimizer_tone_mapper.step()
            self.optimizer_color_mlp.step()
            self.optimizer_s_model.step()
            self.optimizer_tone_mapper.zero_grad(set_to_none = True)
            self.optimizer_color_mlp.zero_grad(set_to_none = True)
            self.optimizer_s_model.zero_grad(set_to_none = True)
            if self.d_model:
                self.optimizer_d_model.step()
                self.optimizer_d_model.zero_grad(set_to_none = True)
            if self.optimizer_cams is not None:
                self.optimizer_cams.step()
                self.optimizer_cams.zero_grad(set_to_none = True)


            if step > 0:
                for k, v in self.gs_scheduling_func_dict.items():
                    if self.d_model:
                        update_learning_rate(v(step), k, self.optimizer_d_model)
                    update_learning_rate(v(step), k, self.optimizer_s_model)
                for k, v in self.cams_scheduling_func_dict.items():
                    update_learning_rate(v(step), k, self.optimizer_cams)

            if step % 200 == 0:
                draw_CRF(self.tone_mapper, self.out_dir)

            if step % getattr(self.args, "photo_log_step", 5000) == 0:
                fig = plt.figure(figsize=(30, 8))
                for plt_i, plt_pack in enumerate(
                    [
                        ("loss_rgb", loss_rgb_list),
                        ("loss_dep", loss_dep_list),
                        ("loss_opa", loss_opa_list),
                        ("loss_dyn_tc", loss_dyn_tc_list),
                        ("loss_track", loss_track_list),
                        ("loss_rigid", loss_rigid_list),
                        ("loss_reproj", loss_reproj_list),
                        ("loss_scale_var", loss_scale_var_list),
                        ("loss_vel_xyz", loss_vel_xyz_list),
                        ("loss_vel_rot", loss_vel_rot_list),
                        ("loss_acc_xyz", loss_acc_xyz_list),
                        ("loss_acc_rot", loss_acc_rot_list),
                        ("World-D-GS-N", d_gs_n_count_list),
                        ("World-S-GS-N", s_gs_n_count_list),
                    ]
                ):
                    plt.subplot(2, 10, plt_i + 1)
                    plt.plot(plt_pack[1]), plt.title(plt_pack[0] + f" End={plt_pack[1][-1]:.6f}")
                plt.savefig(
                    osp.join(self.out_dir, "optim_loss.jpg")
                )
                plt.close()
        self.save_render_eval()

    def save_render_eval(self):
        with torch.no_grad():
            ckpt_dir = osp.join(self.out_dir, "ckpt")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(self.d_model.state_dict(), osp.join(ckpt_dir, "d_model.pth"))
            torch.save(self.s_model.state_dict(), osp.join(ckpt_dir, "s_model.pth"))
            torch.save(self.cams.state_dict(), osp.join(ckpt_dir, "train_cams.pth"))
            torch.save(self.tone_mapper.state_dict(), osp.join(ckpt_dir, "tone_mapper.pth"))
            torch.save(self.color_mlp.state_dict(), osp.join(ckpt_dir, "color_mlp.pth"))
        
            self.render_train()
            self.eval_ldr("train_ldr")
            if hasattr(self.s2d, "train_hdr_gt"):
                self.eval_hdr("train_hdr")

            # if self.d_model:
            #     viz2d_total_video(
            #         viz_vid_fn=osp.join(self.out_dir, "world_gs_2dviz.mp4"),
            #         s2d=self.s2d,
            #         start_from=0,
            #         end_at=self.s2d.T-1,
            #         skip_t=1 if self.s2d.T < 120 else max(1, self.s2d.T // 50),
            #         cams=self.cams,
            #         s_model=self.s_model,
            #         d_model=self.d_model,
            #         color_mlp=self.color_mlp,
            #         tone_mapper=self.tone_mapper,
            #         move_around_angle_deg=getattr(self.args, "photo_viz_move_angle_deg", 1.0),
            #         print_text=False,
            #     )
            #     viz3d_total_video(
            #         self.cams,
            #         self.d_model,
            #         0,
            #         self.s2d.T - 1,
            #         save_path=osp.join(self.out_dir, "world_gs_3dviz.mp4"),
            #         res=480,
            #         s_model=self.s_model,
            #         color_mlp=self.color_mlp
            #     )
            viz_main(
                save_dir=osp.join(self.out_dir, "viz"),
                cams=self.cams,
                s_model=self.s_model,
                d_model=self.d_model,
                color_mlp=self.color_mlp,
                tone_mapper=self.tone_mapper,
                N=getattr(self.args, "viz_N", 5),
                move_angle_deg=getattr(self.args, "viz_move_angle_deg", 1.0),
                H_3d=getattr(self.args, "viz_H_3d", 480),
                W_3d=getattr(self.args, "viz_W_3d", 864),
                fov_3d=getattr(self.args, "viz_fov_3d", 70),
                back_ratio_3d=getattr(self.args, "viz_back_ratio_3d", 1.5),
                up_ratio=getattr(self.args, "viz_up_ratio", 0.05),
                bg_color=getattr(self.args, "photo_default_bg_color", [0.0, 0.0, 0.0]),
            )

        if hasattr(self.s2d, "test_ldr_gt"):
            self.get_test_cam_pose()
            with torch.no_grad():
                self.render_test()
                self.eval_ldr("test_ldr")
                if hasattr(self.s2d, "test_ne_ldr_gt"):
                    self.eval_ldr("test_ne_ldr")
                if hasattr(self.s2d, "test_hdr_gt"):
                    self.eval_hdr("test_hdr")  

        with torch.no_grad():
            self.eval_hdr_tc(include_test=False)
            if hasattr(self.s2d, "test_ldr_gt"):
                self.eval_hdr_tc(include_test=True)

        with open(self.out_dir + "/results.json", 'a') as fp:
            json.dump(self.full_dict, fp, indent=True)
        with open(self.out_dir + "/per_view.json", 'a') as fp:
            json.dump(self.per_view_dict, fp, indent=True)