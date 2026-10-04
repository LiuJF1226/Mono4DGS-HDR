import argparse
import os
import cv2
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import imageio
from copy import deepcopy

from .unimatch.unimatch import UniMatch
from .unimatch.geometry import forward_backward_consistency_check
from .RAFT.utils import flow_viz

from .flow_utils import *
from tqdm import tqdm
import os, os.path as osp



def warp_flow_torch(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

    res = cv2.remap(
        img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )
    return res


def compute_fwdbwd_mask(fwd_flow, bwd_flow):
    alpha_1 = 0.5
    alpha_2 = 0.5

    bwd2fwd_flow = warp_flow(bwd_flow, fwd_flow)
    fwd_lr_error = np.linalg.norm(fwd_flow + bwd2fwd_flow, axis=-1)
    fwd_mask = (
        fwd_lr_error
        < alpha_1
        * (np.linalg.norm(fwd_flow, axis=-1) + np.linalg.norm(bwd2fwd_flow, axis=-1))
        + alpha_2
    )

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = (
        bwd_lr_error
        < alpha_1
        * (np.linalg.norm(bwd_flow, axis=-1) + np.linalg.norm(fwd2bwd_flow, axis=-1))
        + alpha_2
    )

    return fwd_mask, bwd_mask


def get_neighboring_pair_list(name_list):
    names = deepcopy(name_list)
    names.sort()
    pair_list = []
    for i in range(len(names) - 1):
        pair_list.append((names[i], names[i + 1]))
    return pair_list


def get_dense_pair_list(name_list, jump_steps=[1]):
    names = deepcopy(name_list)
    names.sort()
    pair_list = []
    for step in jump_steps:
        for i in range(len(names) - step):
            pair_list.append((names[i], names[i + step]))
    return pair_list


@torch.no_grad()
def gmflow_process_folder(
    model, img_list, img_name_list, dst_dir, pair_list=None, step_list=[1], padding_factor=32, inference_size=None, attn_type="swin", attn_splits_list=[2,8], corr_radius_list=[-1,4], prop_radius_list=[-1,1], num_reg_refine=6
):
    if pair_list is None:
        pair_list = get_dense_pair_list(img_name_list, jump_steps=step_list)
    device = next(model.parameters()).device
    os.makedirs(dst_dir, exist_ok=True)
    flow_viz_fwd_list, flow_mask_viz_fwd_list = [], []
    flow_viz_bwd_list, flow_mask_viz_bwd_list = [], []

    fixed_inference_size = inference_size
    transpose_img = False

    for vi, vj in tqdm(pair_list):
        image1 = img_list[img_name_list.index(vi)].astype(np.uint8).copy()
        image2 = img_list[img_name_list.index(vj)].astype(np.uint8).copy()

        if len(image1.shape) == 2:  # gray image
            image1 = np.tile(image1[..., None], (1, 1, 3))
            image2 = np.tile(image2[..., None], (1, 1, 3))
        else:
            image1 = image1[..., :3]
            image2 = image2[..., :3]

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(device)
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(device)

        if image1.size(-2) > image1.size(-1):
            image1 = torch.transpose(image1, -2, -1)
            image2 = torch.transpose(image2, -2, -1)
            transpose_img = True

        nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                        int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

        # resize to nearest size or specified size
        inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

        assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
        ori_size = image1.shape[-2:]

        # resize before inference
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                   align_corners=True)
            image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                   align_corners=True)

        results_dict = model(image1, image2,
                             attn_type=attn_type,
                             attn_splits_list=attn_splits_list,
                             corr_radius_list=corr_radius_list,
                             prop_radius_list=prop_radius_list,
                             num_reg_refine=num_reg_refine,
                             task='flow',
                             pred_bidir_flow=True,
                            )
        
        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]
     
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                    align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]
        
        if transpose_img:
            flow_pr = torch.transpose(flow_pr, -2, -1)

        flow_fwd = flow_pr[0].permute(1, 2, 0).cpu().numpy() 
        assert flow_pr.size(0) == 2  # [2, H, W, 2]
        flow_bwd = flow_pr[1].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

        # fwd_occ, bwd_occ = forward_backward_consistency_check(flow_pr[:1], flow_pr[1:])  # [1, H, W] float
        # mask_fwd, mask_bwd = 1 - fwd_occ[0].cpu().numpy(), 1 - bwd_occ[0].cpu().numpy()

        mask_fwd, mask_bwd = compute_fwdbwd_mask(flow_fwd, flow_bwd)

        # from PIL import Image
        # Image.fromarray((mask_fwd * 255).astype(np.uint8)).save('fwd2.png')
        # Image.fromarray((mask_bwd * 255).astype(np.uint8)).save('bwd2.png')

        # Save flow
        np.savez_compressed(
            os.path.join(dst_dir, f"{vi}_to_{vj}.npz"),
            flow=flow_fwd.astype(np.float16),
            mask=mask_fwd.astype(np.float16),
        )
        np.savez_compressed(
            os.path.join(dst_dir, f"{vj}_to_{vi}.npz"),
            flow=flow_bwd.astype(np.float16),
            mask=mask_bwd.astype(np.float16),
        )
        # Save flow_img
        if vi < vj:
            flow_viz_fwd = flow_viz.flow_to_image(flow_fwd)
            flow_viz_fwd_list.append(flow_viz_fwd)
            flow_mask_viz_fwd_list.append(mask_fwd.astype(np.uint8) * 255)
            flow_viz_bwd = flow_viz.flow_to_image(flow_bwd)
            flow_viz_bwd_list.append(flow_viz_bwd)
            flow_mask_viz_bwd_list.append(mask_bwd.astype(np.uint8) * 255)
    n = step_list[-1]
    flow_viz_fwd_list = [flow_viz_fwd_list[i::n] for i in range(n)]
    flow_mask_viz_fwd_list = [flow_mask_viz_fwd_list[i::n] for i in range(n)]
    flow_viz_bwd_list = [flow_viz_bwd_list[i::n] for i in range(n)]
    flow_mask_viz_bwd_list = [flow_mask_viz_bwd_list[i::n] for i in range(n)]
    for i in range(n):
        imageio.mimsave(
            os.path.join(os.path.dirname(dst_dir), "flow_viz_fwd_{}.mp4".format(i)),
            flow_viz_fwd_list[i],
        )
        imageio.mimsave(
            os.path.join(os.path.dirname(dst_dir), "flow_mask_viz_fwd_{}.mp4".format(i)),
            flow_mask_viz_fwd_list[i],
        )
        imageio.mimsave(
            os.path.join(os.path.dirname(dst_dir), "flow_viz_bwd_{}.mp4".format(i)),
            flow_viz_bwd_list[i],
        )
        imageio.mimsave(
            os.path.join(os.path.dirname(dst_dir), "flow_mask_viz_bwd_{}.mp4".format(i)),
            flow_mask_viz_bwd_list[i],
        )
    return

def get_gmflow_model(ckpt_path, device, feature_channels=128, num_scales=2, upsample_factor=4, num_head=1, ffn_dim_expansion=4, num_transformer_layers=6, reg_refine=True, task="flow"):

    model = UniMatch(feature_channels=feature_channels,
                     num_scales=num_scales,
                     upsample_factor=upsample_factor,
                     num_head=num_head,
                     ffn_dim_expansion=ffn_dim_expansion,
                     num_transformer_layers=num_transformer_layers,
                     reg_refine=reg_refine,
                     task=task)
    _stat_dict = torch.load(ckpt_path, map_location="cpu")
    # remove the module prefix
    # state_dict = {}
    # for k, v in _stat_dict.items():
    #     if k.startswith("module."):
    #         state_dict[k[7:]] = v
    #     else:
    #         state_dict[k] = v
    model.load_state_dict(_stat_dict["model"], strict=False)

    # model = model.module
    model.to(device)
    model.eval()
    return model
