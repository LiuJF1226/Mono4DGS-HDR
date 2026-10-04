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
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch, torchvision
import imageio
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from matplotlib import pyplot as plt
import numpy as np


def compute_dep_loss(target_dep, pred_dep, sup_mask, st_invariant=True):
    if st_invariant:
        prior_t = torch.median(target_dep[sup_mask > 0.5])
        pred_t = torch.median(pred_dep[sup_mask > 0.5])
        prior_s = (target_dep[sup_mask > 0.5] - prior_t).abs().mean()
        pred_s = (pred_dep[sup_mask > 0.5] - pred_t).abs().mean()
        prior_dep_norm = (target_dep - prior_t) / prior_s
        pred_dep_norm = (pred_dep - pred_t) / pred_s
    else:
        prior_dep_norm = target_dep
        pred_dep_norm = pred_dep
    sup_mask = sup_mask.float()
    loss_dep_i = torch.abs(pred_dep_norm - prior_dep_norm) * sup_mask
    loss_dep = loss_dep_i.sum() / sup_mask.sum()
    return loss_dep, loss_dep_i

def depth_correlation_loss(gt_depth, rendered_depth, patch_size, num_patches):
    """
    Compute the depth correlation loss between the ground truth and rendered depth maps.

    Args:
        gt_depth (torch.Tensor): The ground truth depth map. [H, W, 1]
        rendered_depth (torch.Tensor): The rendered depth map. [H, W, 1]
        patch_size (int): The size of the patches to sample.
        num_patches (int): The number of patches to sample.
    """

    # Find the dimensions of the depth maps
    height, width, _ = gt_depth.size()
    grid_i, grid_j = torch.meshgrid([torch.arange(patch_size), torch.arange(patch_size)], indexing='ij')
    grid = torch.stack([grid_i, grid_j], dim=-1).float().to(gt_depth.device)  # [patch_size, patch_size, 2]

    
    # Sample patches from the depth maps and compute correlations
    ii = torch.randint(0, height - patch_size, (num_patches,))  # [N,]
    jj = torch.randint(0, width - patch_size, (num_patches,))
    sampled_indexes = torch.stack([ii, jj], dim=1).to(gt_depth.device)  # [N, 2]
    sampled_indexes = sampled_indexes[:, None, None, :] + grid[None]  # [N, patch_size, patch_size, 2]

    # Extract the patches from the depth maps
    sampled_indexes = sampled_indexes[:, :, :, 0] * width + sampled_indexes[:, :, :, 1]  # [N, patch_size, patch_size]
    sampled_indexes = sampled_indexes.reshape(num_patches, -1).long()
    sampled_gt_patches = torch.gather(gt_depth.reshape(1,height*width).repeat(num_patches,1), dim=1, index=sampled_indexes)

    sampled_rendered_patches = torch.gather(rendered_depth.reshape(1,height*width).repeat(num_patches,1), dim=1, index=sampled_indexes)

    pcc = (sampled_rendered_patches*sampled_gt_patches).mean(dim=1) - sampled_rendered_patches.mean(dim=1)*sampled_gt_patches.mean(dim=1)
    pcc = pcc / (sampled_rendered_patches.std(dim=1) * sampled_gt_patches.std(dim=1))

    return 1 - pcc.mean()

def compute_rgb_loss(
    gt_rgb, pred_rgb, sup_mask: torch.Tensor, ssim_lambda=0.2
):
    sup_mask = sup_mask.float()
    rgb_loss_i = l1_loss(pred_rgb, gt_rgb.detach()) * sup_mask[None, ...]
    rgb_loss = rgb_loss_i.sum() / sup_mask.sum()
    if ssim_lambda > 0:
        ssim_loss = 1.0 - ssim(
            (pred_rgb * sup_mask[None, ...])[None],
            (gt_rgb * sup_mask[None, ...])[None],
        )
        rgb_loss = rgb_loss * (1-ssim_lambda) + ssim_loss * ssim_lambda

    return rgb_loss, rgb_loss_i

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt))

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def unit_expos_loss(tone_mapper, gt=0.5):
    ln_x = torch.zeros([1,3]).cuda()
    rgb_l = tone_mapper(ln_x)
    
    return torch.mean((rgb_l - gt) ** 2) 

def CRF_monotonic_loss(tone_mapper):
    ln_x = torch.linspace(-10, 10, 10000).reshape([-1, 1]).cuda()
    ln_x.requires_grad_(True)
    rgb_ln_x = torch.cat([ln_x, ln_x, ln_x], -1)
    rgb_l = tone_mapper(rgb_ln_x)
    
    loss = 0.0
    for channel in range(3):  
        y = rgb_l[:, channel].sum() 
        grad = torch.autograd.grad(y, ln_x, create_graph=True)[0]  
        channel_loss = torch.mean(torch.relu(-grad))  #
        loss += channel_loss

    return loss 

def CRF_RGB_equal_loss(tone_mapper):
    ln_x = torch.linspace(-10, 10, 10000).reshape([-1, 1]).cuda()
    rgb_ln_x = torch.cat([ln_x, ln_x, ln_x], -1)
    rgb_l = tone_mapper(rgb_ln_x)
    
    loss = torch.mean((rgb_l[:, 0] - rgb_l[:, 1]) ** 2) + torch.mean((rgb_l[:, 0] - rgb_l[:, 2]) ** 2) + torch.mean((rgb_l[:, 1] - rgb_l[:, 2]) ** 2)

    return loss 

def draw_CRF(tone_mapper, basedir, data_type="real"):
    # simple tone mapper for synthetic dataset
    def tonemapSimple(x):
        return (x / (x + 1)) ** (1 / 2.2)
    
    ln_x = torch.linspace(-10, 10, 1000).reshape([-1, 1]).cuda()
    rgb_ln_x = torch.cat([ln_x, ln_x, ln_x], -1)
    rgb_l = tone_mapper(rgb_ln_x)
    x = ln_x.cpu().numpy()
    y = rgb_l.detach().cpu().numpy()
    # z_simple = np.clip(tonemapSimple(np.exp(x)), 0, 1)
    z_simple = np.clip(tonemapSimple(np.power(2, x)), 0, 1)

    plt.xlabel("lnE + lnt")
    plt.ylabel("pixel value")
    plt.plot(x,y[:,0], color='r', label='Red')
    plt.plot(x,y[:,1:2], color='g', label='Green')
    plt.plot(x,y[:,2:3], color='b', label='Blue')
    # if data_type == "synthetic":
    plt.plot(x, z_simple, color='y', label='GT')

    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(basedir, 'CRF.png'))
    plt.close()


def compute_HDR_TAE(imgs_tm, imgs_h, flow_mode="raft"):
    if flow_mode == "raft":
        from lib_prior.optical_flow.RAFT.raft import RAFT
        from lib_prior.optical_flow.RAFT.utils import flow_viz
        from lib_prior.optical_flow.raft_wrapper import resize_flow, compute_fwdbwd_mask, load_image, warp_flow_torch
        from lib_prior.optical_flow.RAFT.utils.utils import InputPadder
        import argparse

        args = argparse.Namespace()
        args.small = False
        args.mixed_precision = False
        model = RAFT(args)
        _stat_dict = torch.load('./prior_weights/raft-things.pth', map_location="cpu")
        # remove the module prefix
        state_dict = {}
        for k, v in _stat_dict.items():
            if k.startswith("module."):
                state_dict[k[7:]] = v
            else:
                state_dict[k] = v
        model.load_state_dict(state_dict)
        model.cuda().eval()

        errs = []
        for i in range(len(imgs_tm)-1):
            image1, img_shape = load_image(imgs_tm[i])
            image2, img_shape = load_image(imgs_tm[i+1])
            image1, image2 = image1.cuda(), image2.cuda()
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            _, flow_fwd = model(image1, image2, iters=20, test_mode=True)
            _, flow_bwd = model(image2, image1, iters=20, test_mode=True)

            flow_fwd = padder.unpad(flow_fwd[0]).cpu().numpy().transpose(1, 2, 0)
            flow_bwd = padder.unpad(flow_bwd[0]).cpu().numpy().transpose(1, 2, 0)

            flow_fwd = resize_flow(flow_fwd, img_shape[0], img_shape[1])
            flow_bwd = resize_flow(flow_bwd, img_shape[0], img_shape[1])
            
            # a=flow_viz.flow_to_image(flow_fwd)
            # imageio.imwrite('output.png', a)

            mask_fwd, mask_bwd = compute_fwdbwd_mask(flow_fwd, flow_bwd)

            flow_fwd = torch.from_numpy(flow_fwd).float().cuda().permute(2, 0, 1)
            mask_fwd = torch.from_numpy(mask_fwd).float().cuda()  
            flow_bwd = torch.from_numpy(flow_bwd).float().cuda().permute(2, 0, 1)
            mask_bwd = torch.from_numpy(mask_bwd).float().cuda()
            image1_h = imgs_h[i].permute(2, 0, 1)
            image2_h = imgs_h[i+1].permute(2, 0, 1)

            # image2_h =( torch.tensor(imgs_tm[i+1].copy()).cuda().float()/255).permute(2, 0, 1)
            # torchvision.utils.save_image(image2_h, '21_{}'.format(3) + ".png")
            rgb_h_2_to_1 = warp_flow_torch(image2_h.unsqueeze(0), flow_fwd.unsqueeze(0))[0]
            # torchvision.utils.save_image(rgb_h_2_to_1, 'srcdasda_{}'.format(3) + ".png")
   
            rgb_h_1_to_2 = warp_flow_torch(image1_h.unsqueeze(0), flow_bwd.unsqueeze(0))[0]

            err1 = (image1_h - rgb_h_2_to_1).abs() / (image1_h + rgb_h_2_to_1 + 1e-10).detach()
            err1 = (err1 * mask_fwd[None, ...]).sum() / (mask_fwd.sum() + 1e-10)
            err2 = (image2_h - rgb_h_1_to_2).abs() / (image2_h + rgb_h_1_to_2 + 1e-10).detach()
            err2 = (err2 * mask_bwd[None, ...]).sum() / (mask_bwd.sum() + 1e-10)
            err = (err1 + err2) / 2
            errs.append(err)
        return torch.tensor(errs).mean().item()

    elif flow_mode == "gmflow":
        from lib_prior.optical_flow.unimatch.unimatch import UniMatch
        from lib_prior.optical_flow.gmflow_wrapper import compute_fwdbwd_mask, warp_flow_torch
        from lib_prior.optical_flow.RAFT.utils import flow_viz
        model = UniMatch(feature_channels=128,
                        num_scales=2,
                        upsample_factor=4,
                        num_head=1,
                        ffn_dim_expansion=4,
                        num_transformer_layers=6,
                        reg_refine=True,
                        task="flow")
        _stat_dict = torch.load('./prior_weights/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth', map_location="cpu")
        model.load_state_dict(_stat_dict["model"], strict=False)
        model.cuda().eval()

        fixed_inference_size = None
        transpose_img = False
        errs = []
        for i in range(len(imgs_tm)-1):
            image1 = imgs_tm[i].astype(np.uint8).copy()
            image2 = imgs_tm[i+1].astype(np.uint8).copy()

            image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).cuda()
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).cuda()

            if image1.size(-2) > image1.size(-1):
                image1 = torch.transpose(image1, -2, -1)
                image2 = torch.transpose(image2, -2, -1)
                transpose_img = True

            padding_factor = 32
            nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                            int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

            # resize to nearest size or specified size
            inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

            assert isinstance(inference_size, list) or isinstance(inference_size, tuple)
            ori_size = image1.shape[-2:]

            # resize before inference
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                image1 = F.interpolate(image1, size=inference_size, mode='bilinear', align_corners=True)
                image2 = F.interpolate(image2, size=inference_size, mode='bilinear', align_corners=True)

            results_dict = model(image1, image2,
                                attn_type="swin",
                                attn_splits_list=[2,8],
                                corr_radius_list=[-1,4],
                                prop_radius_list=[-1,1],
                                num_reg_refine=6,
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
            # a=flow_viz.flow_to_image(flow_fwd)
            # imageio.imwrite('foutput.png', a)
            mask_fwd, mask_bwd = compute_fwdbwd_mask(flow_fwd, flow_bwd)
            flow_fwd = torch.from_numpy(flow_fwd).float().cuda().permute(2, 0, 1)
            mask_fwd = torch.from_numpy(mask_fwd).float().cuda()  
            flow_bwd = torch.from_numpy(flow_bwd).float().cuda().permute(2, 0, 1)
            mask_bwd = torch.from_numpy(mask_bwd).float().cuda()

            image1_h = imgs_h[i].permute(2, 0, 1)
            image2_h = imgs_h[i+1].permute(2, 0, 1)

            # image2_h =( torch.tensor(imgs_tm[i+1].copy()).cuda().float()/255).permute(2, 0, 1)
            # torchvision.utils.save_image(image2_h, 'f21_{}'.format(3) + ".png")
            rgb_h_2_to_1 = warp_flow_torch(image2_h.unsqueeze(0), flow_fwd.unsqueeze(0))[0]
            # torchvision.utils.save_image(rgb_h_2_to_1, 'fsrcdasda_{}'.format(3) + ".png")
      
            rgb_h_1_to_2 = warp_flow_torch(image1_h.unsqueeze(0), flow_bwd.unsqueeze(0))[0]

            err1 = (image1_h - rgb_h_2_to_1).abs() / (image1_h + rgb_h_2_to_1 + 1e-10).detach()
            err1 = (err1 * mask_fwd[None, ...]).sum() / (mask_fwd.sum() + 1e-10)
            err2 = (image2_h - rgb_h_1_to_2).abs() / (image2_h + rgb_h_1_to_2 + 1e-10).detach()
            err2 = (err2 * mask_bwd[None, ...]).sum() / (mask_bwd.sum() + 1e-10)
            err = (err1 + err2) / 2
            errs.append(err)
        return torch.tensor(errs).mean().item()
    else:
        raise NotImplementedError