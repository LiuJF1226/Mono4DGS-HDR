import sys
import os, os.path as osp
from easydict import EasyDict as edict
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms
import logging
import time
import glob
import imageio

sys.path.append(osp.dirname(osp.abspath(__file__)))
from cotracker_visualizer import Visualizer

from SpaTrackV2.models.predictor import Predictor

from tracking_utils import (
    seed_everything,
    convert_img_list_to_cotracker_input,
    get_sampling_mask,
    get_uniform_random_queries,
    load_epi_error,
    load_vos,
    viz_queries,
    viz_coverage,
    tracker_get_query_uv,
)

def get_spatracker2(
    device,
    track_mode="offline",
):
    
    print("loading spatracker-v2...")
    if track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    model.eval()
    model.to(device)

    return model


def make_spatracker2_input(rgb_list, dep_list):
    # T,H,W,3; T,H,W
    assert rgb_list.ndim == 4 and rgb_list.shape[-1] == 3
    assert dep_list.ndim == 3
    assert len(rgb_list) == len(dep_list)

    input_video = (
        torch.from_numpy(rgb_list).permute(0, 3, 1, 2).float()[None].cuda()
    )  # 1,T,3,H,W
    input_depth = torch.from_numpy(dep_list).float().cuda()[:, None]  # T,1,H,W

    return input_video, input_depth


@torch.no_grad()
def infer_spatracker2(video, dep, queries, model, K=None):
    start_t = time.time()
    T = video.shape[1]

    torch.cuda.empty_cache()
    device = "cuda"

    assert (
        video.ndim == 5 and video.shape[0] == 1 and video.shape[2] == 3
    ), "video should have size: 1,T,3,H,W"
    assert dep.ndim == 4 and dep.shape[1] == 1, "depth should have size: T,1,H,W"
    assert dep.shape[0] == video.shape[1], "video and depth should have same length"
    assert (
        queries.ndim == 3 and queries.shape[0] == 1 and queries.shape[2] == 3
    ), "queries should have size: 1,N,3"
    if K is not None:
        K = torch.as_tensor(K).to(device).clone()
        if K.ndim == 2:
            assert K.shape[0] == 3 and K.shape[1] == 3, "K should have size: 3,3"
            K = K[None].repeat(len(dep), 1, 1)
        elif K.ndim == 3:
            assert (
                K.shape[0] == len(dep) and K.shape[1] == 3 and K.shape[2] == 3
            ), "K should have size: T,3,3"
        else:
            raise ValueError("K should have size: 3,3 or T,3,3 or None")
        # K = K[None]  # 1,T,3,3
    else:
        _, _, _, H, W = video.shape
        K = torch.tensor(
            [
                [W, 0.0, W // 2],
                [0.0, W, H // 2],
                [0.0, 0.0, 1.0],
            ]
        ).to(device)
        K = K[None].repeat(len(dep), 1, 1)

    video = video.to(device)
    dep = dep.to(device)
    queries = queries.to(device)

    pred_tracks, pred_visibility = __infer_one_pass__(
        video.detach().clone(),
        dep.detach().clone(),
        queries.detach().clone(),
        model,
        K=K,
    )
    # pred_tracks = pred_tracks[0]  # T,N,3
    # pred_visibility = pred_visibility[0]

    end_t = time.time()
    print(f"SpaT-v2 time cost: {(end_t - start_t)/60.0:.3f} min")

    return pred_tracks, pred_visibility

@torch.no_grad()
def __infer_one_pass__(
    video,
    dep,
    queries,
    model,
    K=None,
):
    # video: 1,T,3,H,W, [0,255] with float
    # depth: T,1,H,W
    # queries: 1,N,3 t,x(W),y(H)
    torch.cuda.empty_cache()
    device = "cuda"

    assert (
        video.ndim == 5 and video.shape[0] == 1 and video.shape[2] == 3
    ), "video should have size: 1,T,3,H,W"
    assert dep.ndim == 4 and dep.shape[1] == 1, "depth should have size: T,1,H,W"
    assert dep.shape[0] == video.shape[1], "video and depth should have same length"
    assert (
        queries.ndim == 3 and queries.shape[0] == 1 and queries.shape[2] == 3
    ), "queries should have size: 1,N,3"
    if K is not None:
        K = torch.as_tensor(K).to(device)
        if K.ndim == 2:
            assert K.shape[0] == 3 and K.shape[1] == 3, "K should have size: 3,3"
            K = K[None].repeat(len(dep), 1, 1)
        elif K.ndim == 3:
            assert (
                K.shape[0] == len(dep) and K.shape[1] == 3 and K.shape[2] == 3
            ), "K should have size: T,3,3"
        else:
            raise ValueError("K should have size: 3,3 or T,3,3 or None")
        # K = K[None]  # 1,T,3,3
    else:
        _, _, _, H, W = video.shape
        K = torch.tensor(
            [
                [W, 0.0, W // 2],
                [0.0, W, H // 2],
                [0.0, 0.0, 1.0],
            ]
        ).to(device)
        K = K[None].repeat(len(dep), 1, 1)

    video = video.to(device)
    dep = dep.to(device)
    # queries = queries.to(device)

    model.spatrack.track_num = queries.shape[1]
    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(video[0], depth=dep[:, 0],
                            intrs=K, extrs=None, 
                            queries=queries[0].cpu().numpy(),
                            fps=1, full_point=False, iters_track=4,
                            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=None,
                            support_frame=len(video)-1, replace_ratio=0.2) 

    pred_tracks = track2d_pred
    pred_visibility = vis_pred[..., 0] > 0.9

    torch.cuda.empty_cache()
    return pred_tracks.cpu(), pred_visibility.cpu()

@torch.no_grad()
def spatracker2_process_folder(
    working_dir,
    img_list,
    img_ori_list,
    dep_list,
    sample_mask_list,
    model,
    total_n_pts,
    chunk_size=10000,  # designed for 16GB GPU
    K=None,
    save_name="",
    max_viz_cnt=512,
    support_ratio=0.2,
):
    print(total_n_pts)
    viz_dir = osp.join(working_dir, "spatracker2_viz")
    os.makedirs(viz_dir, exist_ok=True)
    save_dir = working_dir
    os.makedirs(save_dir, exist_ok=True)
    vis = Visualizer(
        save_dir=working_dir,
        linewidth=2,
        draw_invisible=True,  # False
        tracks_leave_trace=4,
    )

    full_video_pt, full_dep_pt = make_spatracker2_input(img_list, dep_list)
    full_video_pt_ori = torch.from_numpy(img_ori_list).permute(0, 3, 1, 2).float()[None].cuda()
    _, T, _, H, W = full_video_pt.shape
    assert sample_mask_list.shape == (T, H, W), f"{sample_mask_list.shape} != {T,H,W}"
    depth_mask = full_dep_pt.squeeze(1) > 1e-6
    logging.info(f"T=[{T}], video shape: {full_video_pt.shape}")

    start_t = time.time()
    # viz the fg mask
    sample_mask_list = torch.as_tensor(sample_mask_list).cpu() > 0
    viz_sample_mask = sample_mask_list[..., None].cpu() * full_video_pt_ori[
        0
    ].cpu().permute(0, 2, 3, 1)
    imageio.mimsave(
        osp.join(viz_dir, f"{save_name}_sample_mask.mp4"),
        viz_sample_mask.cpu().numpy().astype(np.uint8),
    )

    imageio.mimsave(
        osp.join(viz_dir, f"{save_name}_depth_boundary_mask.mp4"),
        ((sample_mask_list * depth_mask.detach().cpu()).float().numpy() * 255).astype(np.uint8),
    )

    video_pt = full_video_pt.clone()
    dep_pt = full_dep_pt.clone()

    tracks, visibility = [], []

    num_slice = int(np.ceil(total_n_pts / chunk_size))
    chunk_size = int(np.ceil(total_n_pts / num_slice))

    for round in range(num_slice):
        logging.info(f"Round {round+1}/{num_slice} ...")
        masks = sample_mask_list * depth_mask.to(sample_mask_list)
        queries = get_uniform_random_queries(
            video_pt, int(chunk_size * (1.0 - support_ratio)), mask_list=masks
        )
        queries_uniform = get_uniform_random_queries(
            video_pt,
            int(chunk_size * support_ratio),
            mask_list=depth_mask.to(sample_mask_list),
        )
        queries = torch.cat([queries, queries_uniform], 1)

        viz_list = viz_queries(queries.squeeze(0), H, W, T)
        imageio.mimsave(
            osp.join(viz_dir, f"{save_name}_r={round}_quries.mp4"), viz_list
        )
        _tracks, _visibility = infer_spatracker2(
            video_pt, dep_pt, queries, model, K=K
        )  # T,N,3; T,N
        tracks.append(_tracks)
        visibility.append(_visibility)
    tracks = torch.cat(tracks, 1)
    visibility = torch.cat(visibility, 1)
    end_t = time.time()
    logging.info(f"Time cost: {(end_t - start_t)/60.0:.3f}min")
  
    # efficient viz
    viz_choice = np.random.choice(tracks.shape[1], min(tracks.shape[1], max_viz_cnt))
    vis.visualize(
        video=full_video_pt_ori,
        tracks=tracks[None, :, viz_choice, :2],
        visibility=visibility[None, :, viz_choice],
        filename=f"{save_name}_spatracker2_tap",
    )
    logging.info(f"Save to {save_dir} with tracks={tracks.shape}")

    np.savez_compressed(
        osp.join(save_dir, f"{save_name}_spatracker2_tap.npz"),
        queries=queries.numpy(),  # useless
        tracks=tracks.numpy(),
        visibility=visibility.numpy(),
        K=K,  # also save intrinsic for later use if necessary, but seems because the depth is aligned to input depth, so it is not necessary
    )
    return