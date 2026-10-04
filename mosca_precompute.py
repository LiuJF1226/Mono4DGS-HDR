import torch
import imageio
import os, os.path as osp
import numpy as np
import sys
import glob

from lib_prior.moca_processor import MoCaPrep
from lib_prior.preprocessor_utils import load_imgs, convert_from_mp4
from lib_prior.prior_loading import Saved2D, visualize_track
from lib_prior.moca_processor import mark_dynamic_region

# from lib_render.render_helper import GS_BACKEND

from lib_moca.moca import moca_solve
from lib_moca.epi_helpers import analyze_track_epi, identify_tracks
from lib_moca.camera import MonocularCameras
from lib_moca.moca_misc import make_pair_list

# from viz_utils import viz_list_of_colored_points_in_cam_frame
import logging
from lib_prior.moca_processor import *
from omegaconf import OmegaConf
import random
import json


def seed_everything(seed):
    logging.info(f"seed: {seed}")
    print(f"seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    logging.info(f"seed: {seed}")
    print(f"seed: {seed}")


def get_moca_processor(pre_cfg):
    moca_processor = MoCaPrep(
        dep_mode=getattr(
            pre_cfg, "dep_mode", "sensor"
        ),  # "depthcrafter", "metric3d", "unidepth"
        tap_mode=getattr(
            pre_cfg, "tap_mode", "bootstapir"
        ),  # "spatracker", "cotracker"
        flow_mode=getattr(pre_cfg, "flow_mode", "raft"),
        align_metric_flag=getattr(pre_cfg, "align_metric_flag", True),
        align_metric_model=getattr(pre_cfg, "align_metric_model", "unidepth"),
    )
    return moca_processor


def load_imgs_from_dir(src):
    ## 
    train_img_dir = osp.join(src, "train")
    train_img_fns = sorted(
        [it for it in os.listdir(train_img_dir) if it.endswith(".png") or it.endswith(".jpg") or it.endswith(".tif")]
    )
    # test_img_dir = osp.join(src, "test")
    # test_img_fns = sorted(
    #     [it for it in os.listdir(train_img_dir) if it.endswith(".png") or it.endswith(".jpg") or it.endswith(".tif")]
    # )
  

    with open(osp.join(src, "exposures.json"), 'r', encoding='utf-8') as f:
        exp_dict = json.load(f)

    train_exp_num = exp_dict["train_exp_num"]
    ref_expos = np.unique(list(exp_dict["train"].values()))
    ref_exposure_time = 2 ** np.mean(ref_expos)
    # ref_exposure_time = torch.mean(2 ** torch.tensor(ref_expos, dtype=torch.float)).item()

  
    img_list = []
    img_ori_list = []
    tonemapReinhard = cv2.createTonemapReinhard(2.2, 0.5, 0.5 ,0)
    for i, fn in enumerate(train_img_fns):
        ldr = cv2.imread(osp.join(train_img_dir, fn), cv2.IMREAD_COLOR)
        img_ori_list.append(ldr[:,:,::-1])
        ldr_normalized = ldr.astype(np.float32) / 255.0
        linear = ldr_normalized ** 2.2
        exposure_time = 2**(exp_dict["train"][fn])
        # hdr_linear = linear / exposure_time
        # h = (tonemapReinhard.process(hdr_linear) * 255).astype(np.uint8) 
        # l = h[:,:,::-1]
        l = (linear / exposure_time * ref_exposure_time)  ** (1/2.2)
        l = (l.clip(0,1) * 255).astype(np.uint8)
        l = l[:,:,::-1]
        img_list.append(l)
    img_fns = train_img_fns
    return img_list, img_fns, img_ori_list, train_exp_num


def load_imgs_from_mp4():
    raise RuntimeError("Not implemented yet")
    return


def auto_get_depth_dir_tap_mode(ws, fit_cfg):
    dep_dir = getattr(fit_cfg, "depth_dirname", None)
    if dep_dir is None:
        logging.info("Auto get depth dir")
        pattern = "*_depth"
        candidates = glob.glob(osp.join(ws, pattern))
        # ensure is dir
        candidates = [it for it in candidates if osp.isdir(it)]
        if len(candidates) > 1:
            # have a default order
            priority_key = ["gt", "sensor", "sharp", "depthcrafter"]
            for priority_it in priority_key:
                _candidates = [it for it in candidates if priority_it in it]
                if len(_candidates) == 1:
                    logging.warning(f"Multiple depth dir, use {priority_it} depth dir")
                    candidates = _candidates
                    break
        assert len(candidates) == 1, f"Found {len(candidates)} depth dir"
        dep_dir = osp.basename(candidates[0])
    tap_mode = getattr(fit_cfg, "tap_mode", None)
    if tap_mode is None:
        logging.info("Auto get tap mode")
        pattern = "*uniform*tap.npz"
        candidates = glob.glob(osp.join(ws, pattern))
        assert len(candidates) == 1, f"Found {len(candidates)} tap mode"
        tap_mode = osp.basename(candidates[0])
        tap_mode = tap_mode.split("_tap.npz")[0].split("_")[-1]
    return dep_dir, tap_mode


def static_reconstruct(ws, log_path, fit_cfg):
    seed_everything(12345)
    FLOW_MODE = fit_cfg.flow_mode
    DEPTH_MODE = fit_cfg.dep_mode
    TAP_MODE = fit_cfg.tap_mode
    DEPTH_DIR = f"{DEPTH_MODE}_depth"
    # DEPTH_DIR, TAP_MODE = auto_get_depth_dir_tap_mode(ws, fit_cfg)
    DEPTH_BOUNDARY_TH = getattr(fit_cfg, "depth_boundary_th", 1.0)
    INIT_GT_CAMERA_FLAG = getattr(fit_cfg, "init_gt_camera", False)
    DEP_MEDIAN = getattr(fit_cfg, "dep_median", 1.0)

    EPI_TH = getattr(fit_cfg, "ba_epi_th", getattr(fit_cfg, "epi_th", 1e-3))
    logging.info(f"Static BA with EPI_TH={EPI_TH}")
    print(f"Static BA with EPI_TH={EPI_TH}")
    device = torch.device("cuda")

    s2d: Saved2D = (
        Saved2D(ws)
        .load_epi(f"epi_{FLOW_MODE}")
        .load_dep(DEPTH_DIR, DEPTH_BOUNDARY_TH)
        .normalize_depth(median_depth=DEP_MEDIAN)
        .recompute_dep_mask(depth_boundary_th=DEPTH_BOUNDARY_TH)
        .load_track(
            f"uniform_dep={DEPTH_MODE}_{TAP_MODE}_tap",
            min_valid_cnt=getattr(fit_cfg, "ba_tap_loading_min_valid_cnt", 4),
        )
        .load_vos()
    )

    # if INIT_GT_CAMERA_FLAG:
    #     # if start form gt camera, load gt camera here
    #     logging.info(f"Initializing from GT camera")
    #     (
    #         gt_training_cam_T_wi,
    #         gt_testing_cam_T_wi_list,
    #         gt_testing_tids_list,
    #         gt_testing_fns_list,
    #         gt_training_fov,
    #         gt_testing_fov_list,
    #         gt_training_cxcy_ratio,
    #         gt_testing_cxcy_ratio_list,
    #     ) = load_gt_cam(ws, fit_cfg)
    #     gt_fovdeg = float(gt_training_fov)
    #     cxcy_ratio = gt_training_cxcy_ratio[0]  # gt camera center
    #     if getattr(fit_cfg, "init_gt_camera_focal_only", False):
    #         logging.info(f"Only init focal length")
    #         cams = MonocularCameras(
    #             n_time_steps=s2d.T,
    #             default_H=s2d.H,
    #             default_W=s2d.W,
    #             fxfycxcy=[gt_fovdeg, gt_fovdeg] + cxcy_ratio,
    #             delta_flag=True,
    #             init_camera_pose=torch.eye(4)
    #             .to(gt_training_cam_T_wi)[None]
    #             .expand(len(gt_training_cam_T_wi) - 1, -1, -1),
    #             iso_focal=getattr(fit_cfg, "iso_focal", False),
    #         )
    #     else:
    #         cams = MonocularCameras(
    #             n_time_steps=s2d.T,
    #             default_H=s2d.H,
    #             default_W=s2d.W,
    #             fxfycxcy=[gt_fovdeg, gt_fovdeg] + cxcy_ratio,
    #             delta_flag=False,
    #             init_camera_pose=gt_training_cam_T_wi,
    #             iso_focal=getattr(fit_cfg, "iso_focal", False),
    #         )
    # else:
    #     cams = None
    cams = None

    logging.info("*" * 20 + "MoCa BA" + "*" * 20)
    cams, s2d, _ = moca_solve(
        ws=log_path,
        s2d=s2d,
        fit_cfg=fit_cfg,
        device=device,
        epi_th=EPI_TH,
        ba_total_steps=getattr(fit_cfg, "ba_total_steps", 2000),
        ba_switch_to_ind_step=getattr(fit_cfg, "ba_switch_to_ind_step", 500),
        ba_depth_correction_after_step=getattr(
            fit_cfg, "ba_depth_correction_after_step", 500
        ),
        ba_max_frames_per_step=32,
        static_id_mode="flow" if s2d.has_epi else "track",
        # * robust setting
        robust_depth_decay_th=getattr(fit_cfg, "robust_depth_decay_th", 2.0),
        robust_depth_decay_sigma=getattr(fit_cfg, "robust_depth_decay_sigma", 1.0),
        robust_std_decay_th=getattr(fit_cfg, "robust_std_decay_th", 0.2),
        robust_std_decay_sigma=getattr(fit_cfg, "robust_std_decay_sigma", 0.2),
        #
        gt_cam=cams,
        iso_focal=getattr(fit_cfg, "iso_focal", False),
        rescale_gt_cam_transl=getattr(fit_cfg, "rescale_gt_cam_transl", False),
        ba_lr_cam_f=getattr(fit_cfg, "ba_lr_cam_f", 0.0003),
        ba_lr_dep_c=getattr(fit_cfg, "ba_lr_dep_c", 0.001),
        ba_lr_dep_s=getattr(fit_cfg, "ba_lr_dep_s", 0.001),
        ba_lr_cam_q=getattr(fit_cfg, "ba_lr_cam_q", 0.0003),
        ba_lr_cam_t=getattr(fit_cfg, "ba_lr_cam_t", 0.0003),
        #
        ba_lambda_flow=getattr(fit_cfg, "ba_lambda_flow", 1.0),
        ba_lambda_depth=getattr(fit_cfg, "ba_lambda_depth", 0.1),
        ba_lambda_small_correction=getattr(fit_cfg, "ba_lambda_small_correction", 0.03),
        ba_lambda_cam_smooth_trans=getattr(fit_cfg, "ba_lambda_cam_smooth_trans", 0.0),
        ba_lambda_cam_smooth_rot=getattr(fit_cfg, "ba_lambda_cam_smooth_rot", 0.0),
        #
        depth_filter_th=getattr(fit_cfg, "ba_depth_remove_th", -1.0),
        init_cam_with_optimal_fov_results=getattr(
            fit_cfg, "init_cam_with_optimal_fov_results", True
        ),
        # fov
        fov_search_fallback=getattr(fit_cfg, "ba_fov_search_fallback", 53.0),
        fov_search_N=getattr(fit_cfg, "ba_fov_search_N", 100),
        fov_search_start=getattr(fit_cfg, "ba_fov_search_start", 30.0),
        fov_search_end=getattr(fit_cfg, "ba_fov_search_end", 90.0),
        viz_valid_ba_points=getattr(fit_cfg, "ba_viz_valid_points", False),
    )  # ! S2D is changed becuase the depth is re-scaled

    # datamode = getattr(fit_cfg, "mode", "iphone")
    # if datamode == "sintel":
    #     test_func = test_sintel_cam
    # elif datamode == "tum":
    #     test_func = test_tum_cam
    # else:
    #     test_func = None
    # if test_func is not None:
    #     test_func(
    #         cam_pth_fn=osp.join(log_path, "bundle", "bundle_cams.pth"),
    #         ws=ws,
    #         save_path=osp.join(log_path, "cam_metrics_ba.txt"),
    #     )

    return s2d

def preprocess(
    img_list: list,
    img_fns: list,
    img_ori_list: list,
    train_exp_num: int,
    ws: str,
    moca_processor: MoCaPrep,
    pre_cfg: OmegaConf,
    resample_for_dynamic=True,
):
    seed_everything(getattr(pre_cfg, "seed", 12345))
    start_t = time.time()
    logging.info("*" * 20 + " Preprocessing " + "*" * 20)
    logging.info(f"Working on {ws}, start phase-1 preprocessing")
    logging.info("*" * 20 + " Preprocessing " + "*" * 20)

    BOUNDARY_EHNAHCE_TH = getattr(pre_cfg, "boundary_enhance_th", -1)
    DEPTH_DIR_POSTFIX = "_depth_sharp" if BOUNDARY_EHNAHCE_TH > 0 else "_depth"

    EPI_TH = getattr(pre_cfg, "epi_th", 1e-3)
    DEPTH_BOUNDARY_TH = getattr(
        pre_cfg, "prep_depth_boundary_th", 1.0
    )  # this is in the median=1.0 space

    TAP_CHUNK_SIZE = getattr(pre_cfg, "tap_chunk_size", 5000)

    moca_processor.process(
        t_list=None,
        img_list=img_list,
        img_name_list=img_fns,
        img_ori_list=img_ori_list,
        save_dir=ws,
        n_track=getattr(pre_cfg, "n_track_uniform", 8192),
        # metric alignment
        metric_alignment_frames=getattr(pre_cfg, "metric_alignment_frames", 10),
        metric_alignment_first_quantil=getattr(
            pre_cfg, "metric_alignment_first_quantil", 0.7
        ),
        metric_alignment_bias_flag=getattr(pre_cfg, "metric_alignment_bias_flag", True),
        metric_alignment_kernel=getattr(pre_cfg, "metric_alignment_kernel", "cauchy"),
        metric_alignment_fscale=getattr(pre_cfg, "metric_alignment_fscale", 0.001),
        # TAP
        compute_tap=True,
        tap_chunk_size=TAP_CHUNK_SIZE,
        # Flow
        flow_steps=[train_exp_num],
        epi_num_threads=getattr(pre_cfg, "epi_num_threads", 64),
        # Dep enhance for spatracker
        boundary_enhance_th=BOUNDARY_EHNAHCE_TH,  # if > 0 will create a sharp dir
        # boost
        compute_flow=getattr(pre_cfg, "compute_flow", True),
    )
    
    if not resample_for_dynamic:
        duration = (time.time() - start_t) / 60.0
        logging.info(
            f"Preprocessing done, SKIP DYN RESAMPLE! time cost: {duration:.3f}min"
        )
        return
  
    logging.info("*" * 20 + " Preprocessing " + "*" * 20)
    logging.info(f"Working on {ws}, start phase-2 preprocessing, densify the fg TAP")

    s2d = (
        Saved2D(ws)
        .load_epi(f"epi_{pre_cfg.flow_mode}")
        .load_dep(f"{pre_cfg.dep_mode}{DEPTH_DIR_POSTFIX}", DEPTH_BOUNDARY_TH)
        .normalize_depth(median_depth=1.0)
        .recompute_dep_mask(depth_boundary_th=DEPTH_BOUNDARY_TH)
        .load_track(f"uniform_dep={pre_cfg.dep_mode}_{pre_cfg.tap_mode}_tap", min_valid_cnt=4)
        .load_vos()
    )

    if hasattr(s2d, "epi"):
        sample_mask = s2d.epi > EPI_TH
    else:
        continuous_pair_list = make_pair_list(s2d.T, interval=[1, 4], dense_flag=True)
        F_list, epierr_list, _ = analyze_track_epi(
            continuous_pair_list, s2d.track, s2d.track_mask, H=s2d.H, W=s2d.W
        )
        track_static_selection, _ = identify_tracks(epierr_list, EPI_TH)
        sample_mask = mark_dynamic_region(
            s2d.track[:, ~track_static_selection],
            s2d.track_mask[:, ~track_static_selection],
            s2d.H,
            s2d.W,
            0.1,
        )

    resampling_mask_dilate_ksize = getattr(pre_cfg, "resampling_mask_dilate_ksize", 7)
    sample_mask = (
        torch.nn.functional.max_pool2d(
            sample_mask[:, None].float(),
            kernel_size=resampling_mask_dilate_ksize,
            stride=1,
            padding=(resampling_mask_dilate_ksize - 1) // 2,
        )[:, 0]
        > 0.5
    )

    epi_mask_dir = osp.join(ws, "epi_"+moca_processor.flow_mode, "mask")
    os.makedirs(epi_mask_dir, exist_ok=True)
    for i in range(sample_mask.shape[0]):
        imageio.imwrite(osp.join(epi_mask_dir, f'{s2d.train_frame_names[i]}.jpg'), sample_mask[i:i+1].permute(1,2,0).detach().cpu().numpy().astype(np.uint8)*255)

    imageio.mimsave(
        osp.join(ws, "epi_resample_mask_{}.mp4".format(moca_processor.flow_mode)),
        sample_mask.cpu().numpy().astype(np.uint8) * 255,
    )

    moca_processor.compute_tap(
        ws=ws,
        save_name=f"dynamic_dep={moca_processor.dep_mode}",
        # n_track=8192 * 3,
        n_track=getattr(pre_cfg, "n_track_dynamic", 8192 * 3),
        img_list=img_list,
        img_ori_list=img_ori_list,
        mask_list=sample_mask.detach().cpu().numpy() > 0,
        dep_list=moca_processor.load_dep_list(
            ws, f"{moca_processor.dep_mode}{DEPTH_DIR_POSTFIX}"
        ),
        # K=cams.default_K.detach().cpu().numpy(), # ! maintain the same K as the first infered static one
        max_viz_cnt=getattr(pre_cfg, "max_viz_cnt", 512),
        chunk_size=TAP_CHUNK_SIZE,
    )
 
    duration = (time.time() - start_t) / 60.0
    logging.info(f"Preprocessing done, time cost: {duration:.3f}min")
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("MoSca-V2 Preprocessing")
    parser.add_argument("--data_path", type=str, help="Source folder", required=True)
    parser.add_argument("--cfg", type=str, help="profile yaml file path", required=True)
    parser.add_argument(
        "--skip_dynamic_resample", action="store_true", help="skip dynamic resample"
    )
    parser.add_argument("--gpu", type=str, default='0')
    args, unknown = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    cfg = OmegaConf.load(args.cfg)
    cli_cfg = OmegaConf.from_dotlist([arg.lstrip('--') for arg in unknown])
    cfg = OmegaConf.merge(cfg, cli_cfg)
    assert 'defaults' in cfg, "cfg must have 'defaults' field"
    base_cfg = os.path.join(os.path.dirname(args.cfg), cfg.defaults)
    base_cfg = OmegaConf.load(base_cfg)
    cfg = OmegaConf.merge(base_cfg, cfg)

    cfg.ws = os.path.join(args.data_path, cfg.data_type, cfg.name)

    img_list, img_fns, img_ori_list, train_exp_num = load_imgs_from_dir(cfg.ws)

    moca_processor = get_moca_processor(cfg)

    preprocess(
        img_list=img_list,
        img_fns=img_fns,
        img_ori_list=img_ori_list,
        train_exp_num=train_exp_num,
        ws=cfg.ws,
        moca_processor=moca_processor,
        pre_cfg=cfg,
        resample_for_dynamic=not args.skip_dynamic_resample,
    )

    static_reconstruct(cfg.ws, cfg.ws, cfg)
