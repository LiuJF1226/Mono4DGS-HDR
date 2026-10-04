from matplotlib import pyplot as plt
import torch, numpy as np
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
    quaternion_to_axis_angle,
)
import logging
import open3d as o3d
import imageio
import os, sys, os.path as osp
from tqdm import tqdm

sys.path.append(osp.dirname(osp.abspath(__file__)))
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))


from matplotlib.colors import hsv_to_rgb
from sklearn.decomposition import PCA

from utils.loss_utils import compute_rgb_loss, compute_dep_loss
# from lib_mosca.scaffold_utils.viz_helper import viz_curve
from lib_moca.camera import MonocularCameras
from lib_render.gauspl_renderer_native import render_cam_pcl, render_native
from sh_utils import RGB2SH, SH2RGB
from lib_prior.prior_loading import Saved2D

import torch.nn.functional as F
from matplotlib import cm
import cv2
import glob
from transforms3d.euler import mat2euler, euler2mat
import imageio.core.util

TEXTCOLOR = (255, 0, 0)
BORDER_COLOR = (100, 255, 100)
BG_COLOR1 = [1.0, 1.0, 1.0]

tonemapReinhard = cv2.createTonemapReinhard(2.2, 0.5, 0.5 ,0)

def make_video_from_pattern(pattern, dst):
    fns = glob.glob(pattern)
    fns.sort()
    frames = []
    for fn in fns:
        frames.append(imageio.imread(fn))
    __video_save__(dst, frames)
    return


@torch.no_grad()
def make_viz_np(
    gt,
    pred,
    error,
    error_cm=cv2.COLORMAP_WINTER,
    img_cm=cv2.COLORMAP_VIRIDIS,
    text0="target",
    text1="pred",
    text2="error",
    gt_margin=5,
    print_text=True,
):
    assert error.ndim == 2
    error = (error / error.max()).detach().cpu().numpy()
    error = (error * 255).astype(np.uint8)
    error = cv2.applyColorMap(error, error_cm)[:, :, ::-1]
    viz_frame = torch.cat([gt, pred], 1)
    if viz_frame.ndim == 2:
        viz_frame = viz_frame / viz_frame.max()
    viz_frame = viz_frame.detach().cpu().numpy()
    viz_frame = np.clip(viz_frame * 255, 0, 255).astype(np.uint8)
    if viz_frame.ndim == 2:
        viz_frame = cv2.applyColorMap(viz_frame, img_cm)[:, :, ::-1]
    viz_frame = np.concatenate([viz_frame, error], 1)
    # split the image to 3 draw the text onto the image
    viz_frame_list = np.split(viz_frame, 3, 1)
    # draw green border of GT target, don't pad, draw inside

    if print_text:
        viz_frame_list[0] = cv2.copyMakeBorder(
            viz_frame_list[0][gt_margin:-gt_margin, gt_margin:-gt_margin],
            gt_margin,
            gt_margin,
            gt_margin,
            gt_margin,
            cv2.BORDER_CONSTANT,
            value=BORDER_COLOR,
        )
        for i, text in enumerate([text0, text1, text2]):
            if len(text) > 0:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 30)
                fontScale = 1
                fontColor = TEXTCOLOR
                lineType = 2
                cv2.putText(
                    viz_frame_list[i],
                    text,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType,
                )
    viz_frame = np.concatenate(viz_frame_list, 1)
    return viz_frame


def __get_move_around_cam_T_cw__(
    s_model,
    d_model,
    color_mlp,
    cams: MonocularCameras,
    move_around_id,
    move_around_angle_deg,
    start_from,
    end_at,
):
    gs5 = [s_model()]
    if d_model is not None:
        gs5.append(d_model(move_around_id))
    render_dict = render_native(
        gs5,
        cams.default_H,
        cams.default_W,
        cams.default_K,
        cams.T_cw(move_around_id),
        color_mlp,
        HDR_mode=True
    )
    depth = render_dict["dep"][0]
    center_dep = depth[depth.shape[0] // 2, depth.shape[1] // 2].item()
    if center_dep < 1e-2:
        try:
            center_dep = depth[render_dict["alpha"][0] > 0.1].min().item()
        except:
            center_dep = 1.0
    focus_point = torch.Tensor([0.0, 0.0, center_dep]).to(depth)

    move_around_radius = np.tan(move_around_angle_deg) * focus_point[2].item()
    # in the xy plane, the new camera is forming a circle
    total_steps = end_at - start_from + 1
    move_around_view_list = []
    for i in range(total_steps):
        x = move_around_radius * np.cos(2 * np.pi * i / (total_steps - 1))
        y = move_around_radius * np.sin(2 * np.pi * i / (total_steps - 1))
        T_c_new = torch.eye(4).to(cams.T_wc(0))
        T_c_new[0, -1] = x
        T_c_new[1, -1] = y
        _z_dir = F.normalize(focus_point[:3] - T_c_new[:3, -1], dim=0)
        _x_dir = F.normalize(
            torch.cross(torch.Tensor([0.0, 1.0, 0.0]).to(_z_dir), _z_dir), dim=0
        )
        _y_dir = F.normalize(torch.cross(_z_dir, _x_dir), dim=0)
        T_c_new[:3, 0] = _x_dir
        T_c_new[:3, 1] = _y_dir
        T_c_new[:3, 2] = _z_dir
        T_w_new = cams.T_wc(move_around_id) @ T_c_new
        T_new_w = T_w_new.inverse()
        move_around_view_list.append(T_new_w)
    return move_around_view_list


def viz2d_total_video(
    s2d: Saved2D,
    viz_vid_fn,
    start_from,
    end_at,
    skip_t,
    cams,
    s_model,
    d_model,
    color_mlp,
    tone_mapper,
    subsample=1,
    mask_type="all",
    move_around_angle_deg=60.0,
    move_around_id=None,
    remove_redundant_flag=True,
    max_num_frames=500,  # 150,
    print_text=True,
):
    logging.info(f"Viz 2D video from {start_from} to {end_at} ...")

    frame_list = []
    fix_cam_id = start_from

    # prepare the novel camera poses
    move_around_angle = np.deg2rad(move_around_angle_deg)
    if move_around_id is None:
        move_around_id = (start_from + end_at) // 2
    move_around_view_list = __get_move_around_cam_T_cw__(
        s_model,
        d_model,
        color_mlp,
        cams,
        move_around_id,
        move_around_angle,
        start_from,
        end_at,
    )
    for _ind, view_ind in tqdm(enumerate(range(start_from, min(end_at + 1, cams.T)))):
        if view_ind % skip_t != 0:
            continue
        frame = viz2d_one_frame(
            view_ind,
            s2d,
            cams,
            s_model,
            d_model,
            color_mlp,
            tone_mapper,
            subsample=1,
            loss_mask_type=mask_type,
            print_text=print_text,
            append_graph=False,
        )
        # fix cam, vary time
        _frame = viz2d_one_frame(
            view_ind,
            s2d,
            cams,
            s_model,
            d_model,
            color_mlp,
            tone_mapper,
            subsample=1,
            loss_mask_type=mask_type,
            append_depth=False,
            append_dyn=False,
            append_graph=False,
            prefix_text=f"Fix={fix_cam_id} ",
            T_cw=cams.T_cw(fix_cam_id),
            print_text=print_text,
        )
        frame = np.concatenate([frame, _frame], 0)
        # fix time, vary cam
        _frame = viz2d_one_frame(
            move_around_id,
            s2d,
            cams,
            s_model,
            d_model,
            color_mlp,
            tone_mapper,
            subsample=1,
            loss_mask_type=mask_type,
            append_depth=False,
            append_dyn=False,
            append_graph=False,
            prefix_text=f"T={move_around_id} ",
            T_cw=move_around_view_list[_ind],
            print_text=print_text,
        )
        frame = np.concatenate([frame, _frame], 0)
        # vary cam and time
        _frame = viz2d_one_frame(
            view_ind,
            s2d,
            cams,
            s_model,
            d_model,
            color_mlp,
            tone_mapper,
            subsample=1,
            loss_mask_type=mask_type,
            append_depth=False,
            append_dyn=False,
            append_graph=False,
            prefix_text=f"Novel",
            T_cw=move_around_view_list[_ind],
            print_text=print_text,
        )
        frame = np.concatenate([frame, _frame], 0)
        frame_list.append(frame)

    # viz the flow
    if d_model is not None:
        flow_frame_list1, choice = viz2d_flow_video(
            start_from,
            end_at,
            cams,
            s_model,
            d_model,
            color_mlp,
            tone_mapper,
            viz_bg=True,
            text="WithBG",
            skip_t=skip_t,
        )
        flow_frame_list2, choice = viz2d_flow_video(
            start_from,
            end_at,
            cams,
            s_model,
            d_model,
            color_mlp,
            tone_mapper,
            viz_bg=False,
            text="NoBG",
            choice=choice,
            skip_t=skip_t,
        )
        flow_frame_list3, choice = viz2d_flow_video(
            start_from,
            end_at,
            cams,
            s_model,
            d_model,
            color_mlp,
            tone_mapper,
            viz_bg=False,
            view_cam_id=fix_cam_id,
            text=f"FixCam={fix_cam_id} ",
            choice=choice,
            skip_t=skip_t,
        )
        flow_frame_list = [
            np.concatenate(
                [flow_frame_list1[i], flow_frame_list2[i], flow_frame_list3[i]], 1
            )
            for i in range(len(flow_frame_list1))
        ]
        final_frame_list = [
            np.concatenate([flow_frame_list[i], frame_list[i]], 0)
            for i in range(len(flow_frame_list))
        ]
    else:
        final_frame_list = frame_list

    # save_path = "/export1/jliugk/project2/4DGS_HDR/aa.jpg"
    # imageio.imwrite(save_path,  final_frame_list[0])

    # ! always split to two columns
    if (final_frame_list[0].shape[0] // cams.default_H) % 2 != 0:
        dummy_frame = np.zeros((cams.default_H, cams.default_W * 3, 3))
        final_frame_list = [
            np.concatenate([f, dummy_frame], 0) for f in final_frame_list
        ]
        assert (final_frame_list[0].shape[0] // cams.default_H) % 2 == 0
    # * save
    re_aranged_list = []
    for frame in final_frame_list:
        H = frame.shape[0]
        frame_top = frame[: H // 2]
        frame_bottom = frame[H // 2 :]
        if remove_redundant_flag:
            # check whether the right image has all redundant gt and error
            first_col = frame_bottom[:, 0]
            if (first_col == first_col[:1]).all() and first_col[0].tolist() == list(
                BORDER_COLOR
            ):
                frame_bottom_w = frame_bottom.shape[1]
                frame_bottom = frame_bottom[
                    :, frame_bottom_w // 3 : -frame_bottom_w // 3
                ]

        frame = np.concatenate([frame_top, frame_bottom], 1)
        re_aranged_list.append(frame)

    cnt = 0
    cur = 0
    T = len(re_aranged_list)
    while cur < T:
        __video_save__(
            viz_vid_fn[:-4] + f"_{cnt}.mp4",
            [
                f[::subsample, ::subsample, :]
                for f in re_aranged_list[cur : cur + max_num_frames]
            ],
        )
        cnt += 1
        cur += max_num_frames



@torch.no_grad()
def viz2d_one_frame(
    model_tid,
    s2d: Saved2D,
    cams: MonocularCameras,
    s_model,
    d_model=None,
    color_mlp=None,
    tone_mapper=None,
    subsample=1,
    loss_mask_type="all",
    view_cam_id=None,
    prefix_text="",
    save_path=None,
    append_depth=True,
    append_dyn=True,
    append_graph=True,
    rgb_mask=None,
    dep_mask=None,
    T_cw=None,
    print_text=True,
):
    if T_cw is None:
        if view_cam_id is None:
            T_cw = cams.T_cw(model_tid)
        else:
            T_cw = cams.T_cw(view_cam_id)

    # * normal viz
    gs5 = [s_model()]
    if d_model is not None:
        gs5.append(d_model(model_tid))
    render_dict = render_native(gs5, cams.default_H, cams.default_W, cams.default_K, T_cw, color_mlp, HDR_mode=True)
    if rgb_mask is None:
        rgb_mask = s2d.get_mask_by_key(loss_mask_type)[model_tid].clone()
    exposure = s2d.train_exposures[model_tid]
    rgb_h = render_dict["rgb"]
    tmp = rgb_h.clone().reshape([3, -1]).transpose(0, 1)
    pixel_lnx = torch.log2(tmp + 1e-6) + exposure
    rgb_l = tone_mapper(pixel_lnx).transpose(0, 1).reshape(rgb_h.shape)  
    gt_rgb = s2d.rgb[model_tid]
    _, rgb_loss_i = compute_rgb_loss(gt_rgb.permute(2, 0, 1).clone(), rgb_l, rgb_mask)
    rgb_h = rgb_h.permute(1,2,0).cpu().numpy()
    # rgb_h = rgb_h / rgb_h.max()
    rgb_h = (rgb_h/np.percentile(rgb_h, 99.5)).clip(0,1)
    rgb_h = tonemapReinhard.process(rgb_h) 
    rgb_h = torch.tensor(rgb_h).cuda()
    viz_frame = make_viz_np(
        gt_rgb * rgb_mask[:, :, None],
        rgb_h,
        rgb_loss_i.permute(1,2,0).max(dim=-1).values,
        text0=f"{prefix_text}Fr={model_tid:03d} GT",
        text1=f"{prefix_text}Fr={model_tid:03d} Pred",
        text2=f"{prefix_text}Fr={model_tid:03d} Err",
        print_text=print_text,
    )

    # * ed viz
    if append_graph and d_model is not None:
        viz_frame_graph = make_viz_graph(d_model, model_tid, cams, view_cam_id)
        viz_frame = np.concatenate([viz_frame, viz_frame_graph], 0)

    # * depth viz
    if append_depth:
        if dep_mask is None:
            dep_mask = rgb_mask * s2d.dep_mask[model_tid]
        depth = render_dict["dep"][0]
        depth_gt = s2d.dep[model_tid].clone()
        _, dep_loss_i = compute_dep_loss(depth_gt, depth, dep_mask)
        viz_frame_dep = make_viz_np(
            depth_gt * dep_mask,
            depth,
            dep_loss_i,
            text0="DEP Target",
            text1="DEP Pred",
            text2="DEP Error",
            print_text=print_text,
        )
        viz_frame = np.concatenate([viz_frame, viz_frame_dep], 0)

    # * fg only viz
    if d_model is not None and append_dyn:
        dyn_render_dict = render_native(
            [d_model(model_tid)],
            cams.default_H,
            cams.default_W,
            cams.default_K,
            T_cw,
            color_mlp,
            # bg_color=[0.5, 0.5, 0.5],
            HDR_mode=True
        )

        dyn_rgb_h = dyn_render_dict["rgb"]
        tmp = dyn_rgb_h.clone().reshape([3, -1]).transpose(0, 1)
        pixel_lnx = torch.log2(tmp + 1e-6) + exposure
        dyn_rgb_l = tone_mapper(pixel_lnx).transpose(0, 1).reshape(dyn_rgb_h.shape)  
        _, dyn_rgb_loss_i = compute_rgb_loss(gt_rgb.permute(2, 0, 1).clone(), dyn_rgb_l, rgb_mask)
        dyn_rgb_h = dyn_rgb_h.permute(1,2,0).cpu().numpy()
        # dyn_rgb_h = dyn_rgb_h / dyn_rgb_h.max()
        dyn_rgb_h = (dyn_rgb_h/np.percentile(dyn_rgb_h, 99.5)).clip(0,1)
        dyn_rgb_h = tonemapReinhard.process(dyn_rgb_h)
        dyn_rgb_h = torch.tensor(dyn_rgb_h).cuda()

        viz_frame_dyn = make_viz_np(
            gt_rgb,
            dyn_rgb_h,
            dyn_rgb_loss_i.permute(1,2,0).max(dim=-1).values,
            text0="FG Only",
            text1="FG Pred",
            text2="FG Error",
            print_text=print_text,
        )
        viz_frame = np.concatenate([viz_frame, viz_frame_dyn], 0)

    viz_frame = viz_frame[::subsample, ::subsample, :]
    if save_path is not None:
        imageio.imwrite(save_path, viz_frame)
    return viz_frame


@torch.no_grad()
def make_viz_graph(d_model, view_ind, cams, view_cam_id=None, max_radius=0.001):
    node_mu_w = d_model.scf._node_xyz[d_model.get_tlist_ind(view_ind)]
    if view_cam_id is None:
        render_cam_id = view_ind
    else:
        render_cam_id = view_cam_id
    R_cw, t_cw = cams.Rt_cw(render_cam_id)
    node_mu = node_mu_w @ R_cw.T + t_cw[None]
    order = torch.arange(len(node_mu))
    c_id = torch.from_numpy(cm.hsv(order / len(node_mu))[:, :3]).to(node_mu)

    c_time = torch.from_numpy(cm.hsv(torch.rand(len(node_mu)))[:, :3]).to(node_mu)
    acc_w = d_model.get_node_sinning_w_acc().detach().cpu().numpy()
    acc_w_binary = (acc_w > float(d_model.scf.skinning_k) / 2.0).astype(np.float32)
    acc_w = acc_w / acc_w.max()
    c_w = torch.from_numpy(cm.viridis(acc_w)[:, :3]).to(node_mu)
    c_wb = torch.from_numpy(cm.viridis(acc_w_binary)[:, :3]).to(node_mu)

    H, W = cams.default_H, cams.default_W
    pf = cams.rel_focal.mean() / 2 * min(H, W)

    viz_frames = []
    # for color in [c_id, c_time, c_w]:
    viz_r = min(max_radius, d_model.scf.spatial_unit / 10.0)
    for color, text in zip(
        [c_id, c_time, c_w], ["Nodes-id", "Nodes-rand-color", "Nodes-acc-w"]
    ):
        sph = RGB2SH(color)
        fr = torch.eye(3).to(node_mu)[None].expand(len(node_mu), -1, -1)
        s = torch.ones(len(node_mu), 3).to(node_mu) * viz_r
        o = torch.ones(len(node_mu), 1).to(node_mu) * 1.0

        render_dict = render_cam_pcl(
            node_mu, fr, s, o, sph, H, W, CAM_K=cams.default_K, bg_color=BG_COLOR1
        )
        rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        rgb = np.clip(rgb, 0, 1)
        rgb = (rgb * 255).astype(np.uint8).copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = TEXTCOLOR
        lineType = 2
        cv2.putText(
            rgb,
            text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType,
        )
        viz_frames.append(rgb)
    ret = np.concatenate(viz_frames, 1).copy()

    return ret

from sh_utils import *
@torch.no_grad()
def viz2d_flow_video(
    start_from,
    end_at,
    cams,
    s_model,
    d_model,
    color_mlp,
    tone_mapper,
    subsample=1,
    view_cam_id=None,
    N=128,
    line_n=32,
    traj_scale=1e-4,
    traj_opa=0.5,
    viz_bg=False,
    choice=None,
    text="",
    skip_t=1,
):

    H, W = cams.default_H, cams.default_W
    frame_list = []
    if viz_bg:
        s_mu_w, s_fr_w, s_s, s_o, s_sph = s_model()
        frame_list_bg = []

    
    prev_mu, _, s_traj, o_traj, sph_traj = d_model(start_from)  # get init coloring
    s_traj = torch.ones_like(s_traj) * traj_scale
    o_traj = torch.ones_like(o_traj) * traj_opa

    # sph_int = torch.rand_like(s_traj)
    # sph_traj = RGB2SH(sph_int)
    # make new color, first sort the prev mu and then use a HSV in the order
    color_id_coord = d_model._xyz
    color_id = (
        color_id_coord[:, 2] + color_id_coord[:, 1] * 1e6 + color_id_coord[:, 0] * 1e12
    )
    color_id = color_id.argsort().float() / len(color_id)
    # track_color = cm.rainbow(color_id.cpu())[:, :3]
    # track_color = cm.hsv(color_id.cpu())[:, :3]
    track_color = cm.get_cmap("gist_rainbow")(color_id.cpu())[:, :3]
    track_color = torch.from_numpy(track_color).to(prev_mu)
    # _track_color = torch.zeros_like(sph_traj)
    # _track_color[:, :3] = track_color
    # track_color = _track_color
    sph_traj = RGB2SH(track_color)


    # only viz the first frame visible points!!
    render_dict = render_native(
        [s_model(), d_model(start_from)],
        H,
        W,
        cams.default_K,
        cams.T_cw(start_from),
        color_mlp,
        HDR_mode=True
    )
    visibility_mask = render_dict["visibility_filter"][-d_model.N :]
    valid_ind = torch.arange(len(visibility_mask)).to(visibility_mask.device)[
        visibility_mask
    ]
    if choice is None:
        choice = valid_ind[torch.randperm(len(valid_ind))[:N]]
    s_traj = s_traj[choice]
    o_traj = o_traj[choice]
    sph_traj = sph_traj[choice]
    prev_mu = prev_mu[choice]

    mu_w = torch.zeros(0, 3).to(prev_mu)
    fr_w = torch.zeros(0, 3, 3).to(prev_mu)
    s = torch.zeros(0, 3).to(prev_mu)
    o = torch.zeros(0, 1).to(prev_mu)
    sph = torch.zeros(0, sph_traj.shape[-1]).to(prev_mu)
    for view_ind in tqdm(range(start_from, min(end_at + 1, cams.T))):
        if view_ind % skip_t != 0:
            continue
        d_mu_w, d_fr_w, d_s, d_o, d_sph = d_model(view_ind)

        # draw the line
        src_mu = prev_mu
        dst_mu = d_mu_w[choice]
        line_dir = dst_mu - src_mu  # N,3
        intermediate_mu = (
            src_mu[:, None]
            + torch.linspace(0, 1, line_n)[None, :, None].to(line_dir)
            * line_dir[:, None]
        ).reshape(-1, 3)
        intermediate_fr = (
            torch.eye(3)[None].expand(len(intermediate_mu), -1, -1).to(intermediate_mu)
        )
        intermediate_s = torch.ones_like(intermediate_mu) * traj_scale * 0.1
        intermediate_o = torch.ones_like(intermediate_mu[:, :1]) * traj_opa
        intermediate_sph = (
            sph_traj[:, None].expand(-1, line_n, -1).reshape(-1, sph_traj.shape[-1])
        )
        prev_mu = dst_mu

        mu_w = torch.cat([mu_w, intermediate_mu.clone(), d_mu_w[choice].clone()], 0)
        fr_w = torch.cat([fr_w, intermediate_fr.clone(), d_fr_w[choice].clone()], 0)
        s = torch.cat([s, intermediate_s.clone(), s_traj.clone()], 0)
        o = torch.cat([o, intermediate_o.clone(), o_traj.clone()], 0)
        sph = torch.cat([sph, intermediate_sph.clone(), sph_traj.clone()], 0)
        # transform
        if view_cam_id is None:
            _render_cam_id = view_ind
        else:
            _render_cam_id = view_cam_id


        # if viz_bg:
        #     working_mu_w = torch.cat([s_mu_w, mu_w, d_mu_w], 0)
        #     working_fr_w = torch.cat([s_fr_w, fr_w, d_fr_w], 0)
        #     working_s = torch.cat([s_s, s, d_s], 0)
        #     working_o = torch.cat([s_o, o, d_o], 0)
        #     working_sph = torch.cat([s_sph, sph, d_sph], 0)
        # else:
        working_mu_w = mu_w
        working_fr_w = fr_w
        working_s = s
        working_o = o
        working_sph = sph
        R_cw, t_cw = cams.Rt_cw(_render_cam_id)
        working_mu_cur = (
            torch.einsum("ij, nj->ni", R_cw, working_mu_w.clone()) + t_cw[None]
        )
        working_fr_cur = torch.einsum("ij, njk->nik", R_cw, working_fr_w.clone())
        # render
        assert (
            len(working_mu_cur)
            == len(working_fr_cur)
            == len(working_s)
            == len(working_o)
            == len(working_sph)
        )
        render_dict = render_cam_pcl(
            working_mu_cur.contiguous(),
            working_fr_cur.contiguous(),
            working_s.contiguous(),
            working_o.contiguous(),
            working_sph.contiguous(),
            H,
            W,
            CAM_K=cams.default_K,
            color_mlp=color_mlp,
            HDR_mode=False,
        )
        pred_rgb = render_dict["rgb"].permute(1, 2, 0)
        viz_frame = pred_rgb.detach().cpu().numpy()
        viz_frame = (np.clip(viz_frame, 0.0, 1.0) * 255).astype(np.uint8).copy()
        if len(text) > 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 30)
            fontScale = 1
            fontColor = TEXTCOLOR
            lineType = 2
            cv2.putText(
                viz_frame,
                text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType,
            )
        viz_frame = viz_frame[::subsample, ::subsample, :]
        frame_list.append(viz_frame)
        if viz_bg:
            working_mu_w = torch.cat([s_mu_w, d_mu_w], 0)
            working_fr_w = torch.cat([s_fr_w, d_fr_w], 0)
            working_s = torch.cat([s_s, d_s], 0)
            working_o = torch.cat([s_o, d_o], 0)
            working_sph = torch.cat([s_sph, d_sph], 0)
            working_mu_cur = (
                torch.einsum("ij, nj->ni", R_cw, working_mu_w.clone()) + t_cw[None]
            )
            working_fr_cur = torch.einsum("ij, njk->nik", R_cw, working_fr_w.clone())
            # render
            assert (
                len(working_mu_cur)
                == len(working_fr_cur)
                == len(working_s)
                == len(working_o)
                == len(working_sph)
            )
            render_dict = render_cam_pcl(
                working_mu_cur.contiguous(),
                working_fr_cur.contiguous(),
                working_s.contiguous(),
                working_o.contiguous(),
                working_sph.contiguous(),
                H,
                W,
                CAM_K=cams.default_K,
                color_mlp=color_mlp,
                HDR_mode=True,
            )
            pred_rgb = render_dict["rgb"].permute(1, 2, 0)
            viz_frame = pred_rgb.detach().cpu().numpy()
            frame_list_bg.append(viz_frame)

    if viz_bg: 
        frame_list_bg = (frame_list_bg/np.percentile(frame_list_bg, 99.5)).clip(0,1)
        for i, h in enumerate(frame_list_bg):
            h = (tonemapReinhard.process(h) * 255).astype(np.uint8)  ## use Reinhard’s method
            h = h[::subsample, ::subsample, :]
            traj = frame_list[i]
            blended = np.where(traj.sum(axis=2, keepdims=True) == 0, h, traj)
            frame_list[i] = blended

    return frame_list, choice


@torch.no_grad()
def viz3d_total_video(
    cams,
    d_model,
    start_tid,
    end_tid,
    save_path,
    res=240,
    s_model=None,
    color_mlp=None,
    time_interval=5,
    bg_color=BG_COLOR1,
    max_num_frames=500,  # 150,
):
    logging.info(f"Viz 3D scene from {start_tid} to {end_tid} ...")

    # frames_node = viz_curve(
    #     d_model.scf._node_xyz,
    #     torch.ones_like(d_model.scf._node_xyz[..., 0]).bool(),
    #     cams,
    #     res=res,
    #     text="All ED Nodes",
    #     viz_n=-1,
    #     # viz_n=128,
    #     time_window=30,
    # )

    # first row is all
    frames1 = _combine_frames(
        viz3d_scene_video(
            cams,
            d_model,
            start_tid=start_tid,
            end_tid=end_tid,
            res=res,
            s_model=s_model,
            color_mlp=color_mlp,
            bg_color=bg_color,
        )
    )
    frames2 = _combine_frames(
        viz3d_scene_flow_video(
            cams,
            d_model,
            start_tid=start_tid,
            end_tid=end_tid,
            res=res,
            s_model=s_model,
            color_mlp=color_mlp,
            bg_color=BG_COLOR1,
        )
    )
    frames3 = _combine_frames(
        viz3d_scene_video(
            cams,
            d_model,
            start_tid=start_tid,
            end_tid=end_tid,
            res=res,
            color_mlp=color_mlp,
            bg_color=bg_color,
        )
    )
    frames4 = _combine_frames(
        viz3d_scene_flow_video(
            cams,
            d_model,
            start_tid=start_tid,
            end_tid=end_tid,
            res=res,
            color_mlp=color_mlp,
            bg_color=BG_COLOR1,
        )
    )

    frames_up = [
        np.concatenate([frames1[i], frames2[i]], 1) for i in range(len(frames1))
    ]
    frames_down = [
        np.concatenate([frames3[i], frames4[i]], 1) for i in range(len(frames3))
    ]
    frames = [
        np.concatenate([frames_up[i], frames_down[i]], 0) for i in range(len(frames1))
    ]
    # frames = [
    #     np.concatenate(
    #         [
    #             frames[i],
    #             np.concatenate([frames_node[i], np.ones_like(frames_node[i])], 0) * 255,
    #         ],
    #         1,
    #     )
    #     for i in range(len(frames1))
    # ]

    # __video_save__(save_path, frames)
    cnt = 0
    cur = 0
    T = len(frames)
    while cur < T:
        __video_save__(
            save_path[:-4] + f"_{cnt}.mp4", frames[cur : cur + max_num_frames]
        )
        cnt += 1
        cur += max_num_frames
    return


def _combine_frames(frames: dict):
    T = len(frames[list(frames.keys())[0]])
    for v in frames.values():
        assert len(v) == T
    ret = []
    for i in range(T):
        ret.append(np.concatenate([frames[k][i] for k in frames.keys()], 1))
    return ret


def q2R(q):
    nq = F.normalize(q, dim=-1, p=2)
    R = quaternion_to_matrix(nq)
    return R


@torch.no_grad()
def viz3d_scene_video(
    cams,
    d_model,
    start_tid,
    end_tid,
    res=480,
    prefix="",
    save_dir=None,
    s_model=None,
    color_mlp=None,
    bg_color=BG_COLOR1,
):
    viz_cam_R = q2R(cams.q_wc[start_tid : end_tid + 1])
    viz_cam_t = cams.t_wc[start_tid : end_tid + 1]
    viz_cam_R, viz_cam_t = cams.Rt_wc_list()
    viz_cam_R = viz_cam_R[start_tid : end_tid + 1].clone()
    viz_cam_t = viz_cam_t[start_tid : end_tid + 1].clone()

    viz_f = 1.0 / np.tan(np.deg2rad(90.0) / 2.0)
    frames = {}
    for viz_time in tqdm(range(start_tid, end_tid + 1)):
        gs5_param = d_model(viz_time)
        if s_model is not None:
            gs5_param = cat_gs(*gs5_param, *s_model())
        viz_dict = viz_scene_hdr(
            res,
            res,
            viz_cam_R,
            viz_cam_t,
            viz_f=viz_f,
            color_mlp=color_mlp,
            gs5_param=gs5_param,
            draw_camera_frames=False,
            bg_color=bg_color,
        )
        for k, v in viz_dict.items():
            if k not in frames.keys():
                frames[k] = []
            v = np.clip(v, 0.0, 1.0)
            v = (v * 255).astype(np.uint8)
            frames[k].append(v)
    if save_dir is not None:
        for k, v in frames.items():
            __video_save__(
                osp.join(save_dir, f"{prefix}dyn_{k}_{start_tid}-{end_tid}.mp4"),
                v,
            )
    return frames


@torch.no_grad()
def viz3d_scene_flow_video(
    cams,
    d_model,
    start_tid,
    end_tid,
    res=480,
    prefix="",
    save_dir=None,
    s_model=None,
    color_mlp=None,
    N=128,
    line_n=16,
    time_window=10,
    bg_color=BG_COLOR1,
):
    viz_R = quaternion_to_matrix(cams.q_wc[start_tid : end_tid + 1])
    viz_t = cams.t_wc[start_tid : end_tid + 1]
    viz_f = 1.0 / np.tan(np.deg2rad(90.0) / 2.0)
    frames = {}

    # prepare flow gaussians
    prev_mu, _, s_traj, o_traj, sph_traj = d_model(start_tid)  # get init coloring
    s_traj = torch.ones_like(s_traj) * 0.0015
    o_traj = torch.ones_like(o_traj) * 0.999
    sph_int = torch.rand_like(sph_traj)
    sph_traj = RGB2SH(sph_int)

    choice = torch.randperm(len(prev_mu))[:N]
    s_traj = s_traj[choice]
    o_traj = o_traj[choice]
    sph_traj = sph_traj[choice]
    prev_mu = prev_mu[choice]

    # dummy
    mu_w, fr_w, s, o, sph = d_model(start_tid)
    s = s * 0.0
    o = o * 0.0

    for viz_time in range(start_tid, end_tid + 1):

        _mu_w, _fr_w, _, _, _ = d_model(viz_time)
        # draw the line
        src_mu = prev_mu
        dst_mu = _mu_w[choice]
        line_dir = dst_mu - src_mu  # N,3
        intermediate_mu = (
            src_mu[:, None]
            + torch.linspace(0, 1, line_n)[None, :, None].to(line_dir)
            * line_dir[:, None]
        ).reshape(-1, 3)
        intermediate_fr = (
            torch.eye(3)[None].expand(len(intermediate_mu), -1, -1).to(intermediate_mu)
        )
        intermediate_s = torch.ones_like(intermediate_mu) * 0.0015 * 0.3
        intermediate_o = torch.ones_like(intermediate_mu[:, :1]) * 0.999
        intermediate_sph = sph_traj[:, None].expand(-1, line_n, -1).reshape(-1, 3)
        prev_mu = dst_mu
        # pad
        if sph.shape[1] > 3:
            intermediate_sph = torch.cat(
                [
                    intermediate_sph,
                    torch.zeros(len(intermediate_sph), sph.shape[1] - 3).to(sph),
                ],
                1,
            )

        one_time_N = len(src_mu) * (line_n + 1)
        max_N = time_window * N

        mu_w = torch.cat([mu_w, intermediate_mu.clone(), _mu_w[choice].clone()], 0)[
            -max_N:
        ]
        fr_w = torch.cat([fr_w, intermediate_fr.clone(), _fr_w[choice].clone()], 0)[
            -max_N:
        ]
        s = torch.cat([s, intermediate_s.clone(), s_traj.clone()], 0)[-max_N:]
        o = torch.cat([o, intermediate_o.clone(), o_traj.clone()], 0)[-max_N:]
        sph = torch.cat([sph, intermediate_sph.clone(), sph_traj.clone()], 0)[-max_N:]
        gs5_param = (mu_w, fr_w, s, o, sph)

        if s_model is not None:
            gs5_param = cat_gs(*gs5_param, *s_model())
        viz_dict = viz_scene_hdr(
            res, res, viz_R, viz_t, viz_f=viz_f, color_mlp=color_mlp, gs5_param=gs5_param, bg_color=bg_color
        )
        for k, v in viz_dict.items():
            if k not in frames.keys():
                frames[k] = []
            v = np.clip(v, 0.0, 1.0)
            v = (v * 255).astype(np.uint8)
            frames[k].append(v)
    if save_dir is not None:
        for k, v in frames.items():
            __video_save__(
                osp.join(save_dir, f"{prefix}dyn_{k}_{start_tid}-{end_tid}.mp4"),
                v,
            )
    return frames


def cat_gs(m1, f1, s1, o1, c1, m2, f2, s2, o2, c2):
    m = torch.cat([m1, m2], dim=0).contiguous()
    f = torch.cat([f1, f2], dim=0).contiguous()
    s = torch.cat([s1, s2], dim=0).contiguous()
    o = torch.cat([o1, o2], dim=0).contiguous()
    c = torch.cat([c1, c2], dim=0).contiguous()
    return m, f, s, o, c


def viz_o_hist(model, save_path, title_text=""):
    o = model.get_o.detach().cpu().numpy()
    fig = plt.figure(figsize=(10, 5))
    plt.hist(o, bins=100)
    plt.title(f"{title_text} o hist")
    plt.savefig(save_path)
    plt.close()
    return


def viz_s_hist(model, save_path, title_text=""):
    s = model.get_s.detach()
    s = s.sort(dim=-1).values
    s = s.cpu().numpy()
    fig = plt.figure(figsize=(20, 3))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.hist(s[..., i], bins=100)
        plt.title(f"{title_text} s hist")
    plt.savefig(save_path)
    plt.close()
    return


def viz_sigma_hist(scf, save_path, title_text=""):
    sig = scf.node_sigma.abs().detach().cpu().numpy()
    fig = plt.figure(figsize=(10, 5))
    # sig = sig.reshape(-1)
    if sig.shape[1] == 1:
        plt.hist(sig, bins=100)
        plt.title(f"{title_text} Node Sigma hist (Total {scf.M} nodes)")
    else:
        C = sig.shape[1]
        for i in range(C):
            plt.subplot(1, C, i + 1)
            plt.hist(sig[:, i], bins=100)
            plt.title(f"{title_text} Node Sigma [{scf.M}] dim={i}")
    plt.savefig(save_path)
    plt.close()
    return


def viz_dyn_o_hist(model, save_path, title_text=""):
    dyn_o = model.get_d.detach().cpu().numpy()
    fig = plt.figure(figsize=(10, 5))
    plt.hist(dyn_o, bins=100)
    plt.title(f"{title_text} dyn_o hist")
    plt.savefig(save_path)
    plt.close()
    return


def viz_hist(d_model, viz_dir, postfix):
    viz_s_hist(d_model, osp.join(viz_dir, f"s_hist_{postfix}.jpg"))
    viz_o_hist(d_model, osp.join(viz_dir, f"o_hist_{postfix}.jpg"))


def viz_dyn_hist(scf, viz_dir, postfix):
    viz_sigma_hist(scf, osp.join(viz_dir, f"sigma_hist_{postfix}.jpg"))
    # viz_dyn_o_hist(d_model, osp.join(viz_dir, f"dyn_o_hist_{postfix}.jpg"))
    # viz the skinning K count
    valid_sk_count = scf.topo_knn_mask.sum(-1).detach().cpu().numpy()
    fig = plt.figure(figsize=(10, 5))
    plt.hist(valid_sk_count), plt.title(f"Valid node neighbors count {scf.M}")
    plt.savefig(osp.join(viz_dir, f"valid_sk_count_{postfix}.jpg"))
    plt.close()
    return


def viz_N_count(N_count_list, path):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(N_count_list), plt.title("Noodle Count")
    plt.savefig(path)
    plt.close()


def viz_depth_list(depth_list_pt, save_path):
    assert isinstance(depth_list_pt, torch.Tensor)

    # depth_min = depth_list_pt.min()
    # depth_max = depth_list_pt.max()
    # depth_list_pt = (depth_list_pt - depth_min) / (depth_max - depth_min)
    viz_list = []
    for dep in tqdm(depth_list_pt):
        depth_min = dep.min()
        depth_max = dep.max()
        dep = (dep - depth_min) / (depth_max - depth_min)
        viz = cm.viridis(dep.detach().cpu().numpy())[:, :, :3]
        viz_list.append(viz)
    __video_save__(save_path, viz_list)
    return





def viz_plt_missing_slot(track_mask, path, max_viz=2048):
    # T,N
    T, N = track_mask.shape
    choice = torch.randperm(N)[:max_viz]

    viz_mask = track_mask[:, choice].clone().float()
    resort = torch.argsort(viz_mask.sum(0), descending=True)
    viz_mask = viz_mask[:, resort]
    plt.figure(figsize=(2.0 * max_viz / T, 3.0))
    plt.imshow((viz_mask * 255.0).cpu().numpy(), cmap="viridis")
    plt.title("MissingSlot=0"), plt.xlabel("Sorted Noodles"), plt.ylabel("T")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    # also viz how many valid slot in each frame
    valid_count = viz_mask.sum(1)
    plt.figure(figsize=(5, 3))
    # use bar plot
    plt.bar(range(T), valid_count.cpu().numpy())
    plt.title("ValidSlotCount"), plt.xlabel("T"), plt.ylabel("ValidCount")
    plt.tight_layout()
    plt.savefig(path.replace(".jpg", "_count_perframe.jpg"))
    plt.close()

    return


@torch.no_grad()
def viz_mv_model_frame(
    prior2d, cams, s_model, scf, mv_model, t, support_t, T_cw=None, support_m=None
):
    if s_model is None:
        gs5 = []
        order = 0
    else:
        gs5 = [s_model()]
        order = s_model.max_sph_order
    if support_m is not None:
        d_gs5, param_mask = mv_model.forward(
            scf,
            t,
            support_mask=support_m,
            pad_sph_order=order,
        )
    else:
        d_gs5, param_mask = mv_model.forward(
            scf,
            t,
            torch.Tensor([support_t]).long(),
            pad_sph_order=order,
        )
    gs5.append(d_gs5)
    if T_cw is None:
        T_cw = cams.T_cw(t)
    render_dict = render_native(
        gs5,
        prior2d.H,
        prior2d.W,
        cams.defualt_K,
        T_cw,
    )
    # do the rendering loss
    rgb_sup_mask = prior2d.get_mask_by_key("all", t)
    loss_rgb, rgb_loss_i, pred_rgb, gt_rgb = compute_rgb_loss(
        prior2d, t, render_dict, rgb_sup_mask
    )
    dep_sup_mask = prior2d.get_mask_by_key("all_dep", t)
    loss_dep, dep_loss_i, pred_dep, prior_dep = compute_dep_loss(
        prior2d, t, render_dict, dep_sup_mask
    )
    viz_figA = make_viz_np(
        gt_rgb,
        pred_rgb,
        rgb_loss_i.max(-1).values,
        text0=f"q={t},s={support_t}",
    )
    viz_figB = make_viz_np(
        prior_dep,
        pred_dep,
        dep_loss_i,
        text0=f"q={t},s={support_t}",
    )
    ret = np.concatenate([viz_figA, viz_figB], 0)
    return ret


def __video_save__(fn, imgs, fps=10):
    logging.info(f"Saving video to {fn} ...")
    # H, W = imgs[0].shape[:2]
    # image_size = (W, H)
    # out = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*"MP4V"), fps, image_size)
    # for img in tqdm(imgs):
    #     out.write(img[..., ::-1])
    # out.release()
    imageio.mimsave(fn, imgs)
    logging.info(f"Saved!")
    return


def silence_imageio_warning(*args, **kwargs):
    pass


imageio.core.util._precision_warn = silence_imageio_warning


def make_video(src_pattern, dst):
    image_fns = glob(src_pattern)
    image_fns.sort()
    if len(image_fns) == 0:
        print(f"no image found in {src_pattern}")
        return
    frames = []
    for i, fn in enumerate(image_fns):
        img = cv2.imread(fn)[..., ::-1]
        frames.append(img)
    imageio.mimwrite(dst, frames)


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def cat_gs(m1, f1, s1, o1, c1, m2, f2, s2, o2, c2):
    m = torch.cat([m1, m2], dim=0).contiguous()
    f = torch.cat([f1, f2], dim=0).contiguous()
    s = torch.cat([s1, s2], dim=0).contiguous()
    o = torch.cat([o1, o2], dim=0).contiguous()
    c = torch.cat([c1, c2], dim=0).contiguous()
    return m, f, s, o, c


def draw_line(start, end, radius, rgb, opa=1.0):
    if not isinstance(start, torch.Tensor):
        start = torch.as_tensor(start)
    if not isinstance(end, torch.Tensor):
        end = torch.as_tensor(end)
    line_len = torch.norm(end - start)
    assert line_len > 0
    N = line_len / radius * 3
    line_dir = (end - start) / line_len
    # draw even points on the line
    mu = torch.linspace(0, float(line_len), int(N)).to(start)
    mu = start + mu[:, None] * line_dir[None]
    fr = torch.eye(3)[None].to(mu).expand(len(mu), -1, -1)
    s = radius * torch.ones(len(mu), 3).to(mu)
    o = opa * torch.ones(len(mu), 1).to(mu)
    assert len(rgb) == 3
    c = torch.as_tensor(rgb)[None].to(mu) * torch.ones(len(mu), 3).to(mu)
    c = RGB2SH(c)
    return mu, fr, s, o, c


def draw_frame(R_wc, t_wc, size=0.1, weight=0.01, color=None, opa=1.0):
    if not isinstance(R_wc, torch.Tensor):
        R_wc = torch.as_tensor(R_wc)
    if not isinstance(t_wc, torch.Tensor):
        t_wc = torch.as_tensor(t_wc)
    origin = t_wc
    for i in range(3):
        end = t_wc + size * R_wc[:, i]
        if color is None:
            _color = torch.eye(3)[i].to(R_wc)
        else:
            _color = torch.as_tensor(color).to(R_wc)
        _mu, _fr, _s, _o, _c = draw_line(origin, end, weight, _color, opa)
        if i == 0:
            mu, fr, s, o, rgb = _mu, _fr, _s, _o, _c
        else:
            mu, fr, s, o, rgb = cat_gs(mu, fr, s, o, rgb, _mu, _fr, _s, _o, _c)
    return mu, fr, s, o, rgb


def look_at_R(look_at, cam_center, right_dir=None):
    if right_dir is None:
        right_dir = torch.tensor([1.0, 0.0, 0.0]).to(look_at)
    z_dir = F.normalize(look_at - cam_center, dim=0)
    y_dir = F.normalize(torch.cross(z_dir, right_dir), dim=0)
    x_dir = F.normalize(torch.cross(y_dir, z_dir), dim=0)
    R = torch.stack([x_dir, y_dir, z_dir], 1)
    return R


def add_camera_frame(
    gs5_param, cam_R_wc, cam_t_wc, viz_first_n_cam=-1, add_global=False
):
    mu_w, fr_w, s, o, sph = gs5_param
    N_scene = len(mu_w)
    if viz_first_n_cam <= 0:
        viz_first_n_cam = len(cam_R_wc)
    for i in range(viz_first_n_cam):
        if cam_R_wc.ndim == 2:
            R_wc = quaternion_to_matrix(F.normalize(cam_R_wc[i : i + 1], dim=-1))[0]
        else:
            assert cam_R_wc.ndim == 3
            R_wc = cam_R_wc[i]
        t_wc = cam_t_wc[i]
        _mu, _fr, _s, _o, _sph = draw_frame(
            R_wc.clone(), t_wc.clone(), size=0.1, weight=0.0003
        )
        # pad the _sph to have same order with the input
        if sph.shape[1] > 3:
            _sph = torch.cat(
                [_sph, torch.zeros(len(_sph), sph.shape[1] - 3).to(_sph)], dim=1
            )
        mu_w, fr_w, s, o, sph = cat_gs(mu_w, fr_w, s, o, sph, _mu, _fr, _s, _o, _sph)
    if add_global:
        _mu, _fr, _s, _o, _sph = draw_frame(
            torch.eye(3).to(s),
            torch.zeros(3).to(s),
            size=0.3,
            weight=0.001,
            color=[0.5, 0.5, 0.5],
            opa=0.3,
        )
        mu_w, fr_w, s, o, sph = cat_gs(mu_w, fr_w, s, o, sph, _mu, _fr, _s, _o, _sph)
    cam_pts_mask = torch.zeros_like(o.squeeze(-1)).bool()
    cam_pts_mask[N_scene:] = True
    return mu_w, fr_w, s, o, sph, cam_pts_mask


def get_global_viz_cam_Rt(
    mu_w,
    param_cam_R_wc,
    param_cam_t_wc,
    viz_f,
    z_downward_deg=0.0,
    factor=1.0,
    auto_zoom_mask=None,
    scene_center_mode="mean",
    shift_margin_ratio=1.5,
):
    # always looking towards the scene center
    if scene_center_mode == "mean":
        scene_center = mu_w.mean(0)
    else:
        scene_bound_max, scene_bound_min = mu_w.max(0)[0], mu_w.min(0)[0]
        scene_center = (scene_bound_max + scene_bound_min) / 2.0
    cam_center = param_cam_t_wc.mean(0)

    cam_z_direction = F.normalize(scene_center - cam_center, dim=0)
    cam_y_direction = F.normalize(
        torch.cross(cam_z_direction, param_cam_R_wc[0, :, 0]),
        dim=0,
    )
    cam_x_direction = F.normalize(
        torch.cross(cam_y_direction, cam_z_direction),
        dim=0,
    )
    R_wc = torch.stack([cam_x_direction, cam_y_direction, cam_z_direction], 1)
    additional_R = euler2mat(-np.deg2rad(z_downward_deg), 0, 0, "rxyz")
    additional_R = torch.as_tensor(additional_R).to(R_wc)
    R_wc = R_wc @ additional_R
    # transform the mu to cam_R and then identify the distance
    mu_viz_cam = (mu_w - scene_center[None, :]) @ R_wc.T
    desired_shift = (
        viz_f / factor * mu_viz_cam[:, :2].abs().max(-1)[0] - mu_viz_cam[:, 2]
    )
    # # the nearest point should be in front of camera!
    # nearest_shift =   mu_viz_cam[:, :2].mean()
    # desired_shift = max(desired_shift)
    if auto_zoom_mask is not None:
        desired_shift = desired_shift[auto_zoom_mask]
    shift = desired_shift.max() * shift_margin_ratio
    t_wc = -R_wc[:, -1] * shift + scene_center
    return R_wc, t_wc


@torch.no_grad()
def viz_scene(
    H,
    W,
    param_cam_R_wc,
    param_cam_t_wc,
    model=None,
    viz_f=40.0,
    save_name=None,
    viz_first_n_cam=-1,
    gs5_param=None,
    bg_color=[1.0, 1.0, 1.0],
    draw_camera_frames=False,
    return_full=False,
):
    # auto select viewpoint
    # manually add the camera viz to to
    if model is None:
        assert gs5_param is not None
        mu_w, fr_w, s, o, sph = gs5_param
    else:
        mu_w, fr_w, s, o, sph = model()
    # add the cameras to the GS
    if draw_camera_frames:
        mu_w, fr_w, s, o, sph, cam_viz_mask = add_camera_frame(
            (mu_w, fr_w, s, o, sph), param_cam_R_wc, param_cam_t_wc, viz_first_n_cam
        )

    # * prepare the viz camera
    # * (1) global scene viz
    # viz camera set manually
    # global_R_wc, global_t_wc = get_global_viz_cam_Rt(
    #     mu_w, param_cam_R_wc, param_cam_t_wc, viz_f
    # )
    global_down20_R_wc, global_down20_t_wc = get_global_viz_cam_Rt(
        mu_w, param_cam_R_wc, param_cam_t_wc, viz_f, 20, shift_margin_ratio=1.1
    )
    if draw_camera_frames:
        # camera_R_wc, camera_t_wc = get_global_viz_cam_Rt(
        #     mu_w,
        #     param_cam_R_wc,
        #     param_cam_t_wc,
        #     viz_f,
        #     factor=0.5,
        #     auto_zoom_mask=cam_viz_mask,
        # )
        camera_down20_R_wc, camera_down20_t_wc = get_global_viz_cam_Rt(
            mu_w,
            param_cam_R_wc,
            param_cam_t_wc,
            viz_f,
            20,
            factor=0.5,
            auto_zoom_mask=cam_viz_mask,
            shift_margin_ratio=1.1,
        )

    ret = {}
    ret_full = {}
    todo = {  # "scene_global": (global_R_wc, global_t_wc),
        "scene_global_20deg": (global_down20_R_wc, global_down20_t_wc)
    }
    if draw_camera_frames:
        # todo["scene_camera"] = (camera_R_wc, camera_t_wc)
        todo["scene_camera_20deg"] = (camera_down20_R_wc, camera_down20_t_wc)
    for name, Rt in todo.items():
        viz_cam_R_wc, viz_cam_t_wc = Rt
        viz_cam_R_cw = viz_cam_R_wc.transpose(1, 0)
        viz_cam_t_cw = -viz_cam_R_cw @ viz_cam_t_wc
        viz_mu = torch.einsum("ij,nj->ni", viz_cam_R_cw, mu_w) + viz_cam_t_cw[None]
        viz_fr = torch.einsum("ij,njk->nik", viz_cam_R_cw, fr_w)

        pf = viz_f / 2 * min(H, W)
        render_dict = render_cam_pcl(
            viz_mu, viz_fr, s, o, sph, H=H, W=W, fx=pf, bg_color=bg_color
        )
        rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        ret[name] = rgb
        if return_full:
            ret_full[name] = render_dict
        if save_name is not None:
            base_name = osp.basename(save_name)
            dir_name = osp.dirname(save_name)
            os.makedirs(dir_name, exist_ok=True)
            save_img = np.clip(ret[name] * 255, 0, 255).astype(np.uint8)
            imageio.imwrite(osp.join(dir_name, f"{name}_{base_name}.jpg"), save_img)
    if return_full:
        return ret, ret_full
    return ret


@torch.no_grad()
def viz_scene_hdr(
    H,
    W,
    param_cam_R_wc,
    param_cam_t_wc,
    model=None,
    color_mlp=None,
    viz_f=40.0,
    save_name=None,
    viz_first_n_cam=-1,
    gs5_param=None,
    bg_color=[1.0, 1.0, 1.0],
    draw_camera_frames=False,
    return_full=False,
):
    # auto select viewpoint
    # manually add the camera viz to to
    if model is None:
        assert gs5_param is not None
        mu_w, fr_w, s, o, sph = gs5_param
    else:
        mu_w, fr_w, s, o, sph = model()
    # add the cameras to the GS
    if draw_camera_frames:
        mu_w, fr_w, s, o, sph, cam_viz_mask = add_camera_frame(
            (mu_w, fr_w, s, o, sph), param_cam_R_wc, param_cam_t_wc, viz_first_n_cam
        )

    # * prepare the viz camera
    # * (1) global scene viz
    # viz camera set manually
    # global_R_wc, global_t_wc = get_global_viz_cam_Rt(
    #     mu_w, param_cam_R_wc, param_cam_t_wc, viz_f
    # )
    global_down20_R_wc, global_down20_t_wc = get_global_viz_cam_Rt(
        mu_w, param_cam_R_wc, param_cam_t_wc, viz_f, 20, shift_margin_ratio=1.1
    )
    if draw_camera_frames:
        # camera_R_wc, camera_t_wc = get_global_viz_cam_Rt(
        #     mu_w,
        #     param_cam_R_wc,
        #     param_cam_t_wc,
        #     viz_f,
        #     factor=0.5,
        #     auto_zoom_mask=cam_viz_mask,
        # )
        camera_down20_R_wc, camera_down20_t_wc = get_global_viz_cam_Rt(
            mu_w,
            param_cam_R_wc,
            param_cam_t_wc,
            viz_f,
            20,
            factor=0.5,
            auto_zoom_mask=cam_viz_mask,
            shift_margin_ratio=1.1,
        )

    ret = {}
    ret_full = {}
    todo = {  # "scene_global": (global_R_wc, global_t_wc),
        "scene_global_20deg": (global_down20_R_wc, global_down20_t_wc)
    }
    if draw_camera_frames:
        # todo["scene_camera"] = (camera_R_wc, camera_t_wc)
        todo["scene_camera_20deg"] = (camera_down20_R_wc, camera_down20_t_wc)
    for name, Rt in todo.items():
        viz_cam_R_wc, viz_cam_t_wc = Rt
        viz_cam_R_cw = viz_cam_R_wc.transpose(1, 0)
        viz_cam_t_cw = -viz_cam_R_cw @ viz_cam_t_wc
        viz_mu = torch.einsum("ij,nj->ni", viz_cam_R_cw, mu_w) + viz_cam_t_cw[None]
        viz_fr = torch.einsum("ij,njk->nik", viz_cam_R_cw, fr_w)

        pf = viz_f / 2 * min(H, W)
        render_dict = render_cam_pcl(
            viz_mu, viz_fr, s, o, sph, H=H, W=W, color_mlp=color_mlp, fx=pf, bg_color=bg_color, HDR_mode=True
        )

        rgb_h = render_dict["rgb"]
        rgb_h = rgb_h.permute(1,2,0).cpu().numpy()
        # rgb_h = rgb_h / rgb_h.max()
        rgb_h = (rgb_h/np.percentile(rgb_h, 99.5)).clip(0,1)

        rgb_h = tonemapReinhard.process(rgb_h) 
        # tonemap_mu = lambda x : (np.log(np.clip(x,0,1) * 5000 + 1 ) / np.log(5000 + 1))
        # rgb = tonemap_mu(rgb / rgb.max())

        ret[name] = rgb_h
        if return_full:
            ret_full[name] = render_dict
        if save_name is not None:
            base_name = osp.basename(save_name)
            dir_name = osp.dirname(save_name)
            os.makedirs(dir_name, exist_ok=True)
            save_img = np.clip(ret[name] * 255, 0, 255).astype(np.uint8)
            imageio.imwrite(osp.join(dir_name, f"{name}_{base_name}.jpg"), save_img)
    if return_full:
        return ret, ret_full
    return ret

def get_global_3D_cam_T_cw(
    s_model,
    d_model,
    color_mlp,
    cams,
    H,
    W,
    ref_tid,
    back_ratio=1.0,
    up_ratio=0.2,
):
    render_dict = render_native(
        [s_model(), d_model(ref_tid)],
        H,
        W,
        K=cams.K(H, W),
        T_cw=cams.T_cw(ref_tid),
        color_mlp=color_mlp,
        HDR_mode=True
    )
    depth = render_dict["dep"][0]
    center_dep = depth[depth.shape[0] // 2, depth.shape[1] // 2].item()
    if center_dep < 1e-2:
        center_dep = depth[render_dict["alpha"][0] > 0.1].min().item()
    focus_point = torch.Tensor([0.0, 0.0, center_dep]).to(depth)  # in cam frame

    T_c_new = torch.eye(4).to(cams.T_wc(0))
    T_c_new[2, -1] = -center_dep * back_ratio  # z
    T_c_new[1, -1] = -center_dep * up_ratio  # y
    _z_dir = F.normalize(focus_point[:3] - T_c_new[:3, -1], dim=0)
    _x_dir = F.normalize(
        torch.cross(torch.Tensor([0.0, 1.0, 0.0]).to(_z_dir), _z_dir), dim=0
    )
    _y_dir = F.normalize(torch.cross(_z_dir, _x_dir), dim=0)
    T_c_new[:3, 0] = _x_dir
    T_c_new[:3, 1] = _y_dir
    T_c_new[:3, 2] = _z_dir
    T_base = cams.T_wc(ref_tid)
    T_w_new = T_base @ T_c_new
    T_new_w = T_w_new.inverse()
    # T_new_w[0, -1] -= 0.2
    # print(T_new_w)
    return T_new_w

def map_colors(points, mod=1):
    # normalized_points = (points - np.min(points, axis=0)) / (np.max(points, axis=0) - np.min(points, axis=0))

    # do pca for the points
    pca = PCA(n_components=3)
    pca_points = pca.fit_transform(points)
    # normalzie
    pca_points = (pca_points - np.min(pca_points, axis=0)) / (
        np.max(pca_points, axis=0) - np.min(pca_points, axis=0)
    )

    # Map coordinates to HSV colors
    # # H: X-coordinate, S: 1 (high saturation), V: Z-coordinate
    hsv_colors = np.zeros_like(pca_points)
    hue = pca_points[:, 0]
    if mod > 1:
        # set periodical mod times
        hue = hue * mod
        hue = hue - np.floor(hue)
    hsv_colors[:, 0] = hue
    hsv_colors[:, 1] = 0.9
    hsv_colors[:, 2] = 0.9
    rgb_colors = hsv_to_rgb(hsv_colors)
    return rgb_colors

def outlier_removal_o3d(xyz, nb_neighbors=20, std_ratio=2.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
    _, inlier_ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    inlier_mask = torch.zeros_like(xyz[:, 0]).bool()
    inlier_mask[inlier_ind] = True
    return inlier_mask

def draw_gs_point_line(start, end, n=32):
    # start, end is N,3 tensor
    line_dir = end - start
    xyz = (
        start[:, None]
        + torch.linspace(0, 1, n)[None, :, None].to(start) * line_dir[:, None]
    )
    return xyz

def viz_single_2d_flow_video(
    H,
    W,
    cams,
    s_model,
    d_model,
    color_mlp,
    save_fn,
    pose_list,
    N_max=128,
    color_mod=5,
    max_T=1000,
    gray_scale_bg_flag=True,
    #
    node_r_factor=0.0001,  # 0.05,
    # line
    line_N=32,
    line_opa=0.5,
    line_r_factor=0.0001,
    rel_focal=None,
    bg_color=[0.0, 0.0, 0.0],
):
    rgb_viz_list = []

    # ! color the node
    pts_first = d_model(0)[0]
    if len(pts_first) > N_max:
        # ! do a filtering for viz purpose, only viz dense area
        # use open3d
        inlier_mask = outlier_removal_o3d(pts_first, std_ratio=1.0)
        print(f"Filtered {len(pts_first) - inlier_mask.sum()} points")
        candidates = torch.arange(len(pts_first))[inlier_mask.cpu()]
        step = max(1, len(candidates) // N_max)
        # viz_choice = candidates[torch.randperm(len(candidates))[:N_max]]
        viz_choice = candidates[::step][:N_max]
        pts_first = pts_first[viz_choice]
    node_colors = map_colors(pts_first.detach().cpu().numpy(), mod=color_mod)

    flow_sph = RGB2SH(torch.from_numpy(node_colors).to(pts_first.device).float())
    # pad_sph_dim = s_model()[-1].shape[1]
    # if pad_sph_dim > flow_sph.shape[1]:
    #     flow_sph = F.pad(flow_sph, (0, pad_sph_dim - flow_sph.shape[1], 0, 0))

    flow_mu = pts_first
    flow_fr = (
        torch.eye(3).to(flow_mu.device).unsqueeze(0).expand(flow_mu.shape[0], -1, -1)
    )
    flow_s = (
        torch.ones(len(flow_mu), 3).to(flow_mu)
        * node_r_factor
    )
    flow_o = torch.ones_like(flow_s[:, :1]) * 0.99
    last_flow_mu = flow_mu
    last_flow_sph = flow_sph

    # ! gray-scale the bg
    gs5_bg = list(s_model())
    # if gray_scale_bg_flag:
    #     bg_rgb = SH2RGB(gs5_bg[-1][:, :3])
    #     bg_gray = torch.mean(bg_rgb, dim=1, keepdim=True).expand(-1, 3)
    #     # convert to gray scale
    #     bg_sph = RGB2SH(bg_gray)
    #     if pad_sph_dim > bg_sph.shape[1]:
    #         bg_sph = F.pad(bg_sph, (0, pad_sph_dim - bg_sph.shape[1], 0, 0))
    #     gs5_bg[-1] = bg_sph

    max_buffer_size = len(flow_mu) * (line_N + 1) * max_T
    frame_list_bg = []
    rgb_viz_list = []
    for cam_tid in tqdm(range(len(pose_list))):
        # working_t = cam_tid if model_t is None else model_t
        working_t = cam_tid

        ##################################################
        # make GS
        gs5 = [gs5_bg]
        d_gs5 = list(d_model(working_t))
        # d_gs5[-2] = 0.2 * d_gs5[-2]
        gs5.append(d_gs5)
        if rel_focal is None:
            rel_focal = cams.rel_focal
        render_dict = render_native(
            gs5,
            H,
            W,
            K=cams.K(H, W),
            T_cw=pose_list[cam_tid],
            color_mlp=color_mlp,
            HDR_mode=True,
            bg_color=bg_color,
        )
        pred_rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        frame_list_bg.append(pred_rgb)
        if cam_tid > 0:
            new_xyz = d_gs5[0][viz_choice]
            new_flow_sph = last_flow_sph
            # first draw lines of the flow
            if line_N > 0:
                line_xyz = draw_gs_point_line(new_xyz, last_flow_mu, n=line_N).reshape(
                    -1, 3
                )
                line_fr = (
                    torch.eye(3)
                    .to(flow_mu.device)
                    .unsqueeze(0)
                    .expand(line_xyz.shape[0], -1, -1)
                )
                line_s = (
                    torch.ones_like(line_xyz) * line_r_factor
                )
                line_o = torch.ones_like(line_s[:, :1]) * line_opa
                line_sph = draw_gs_point_line(
                    new_flow_sph,
                    last_flow_sph,
                    n=line_N,
                ).reshape(-1, flow_sph.shape[-1])
                flow_mu = torch.cat([flow_mu, line_xyz], dim=0)
                flow_fr = torch.cat([flow_fr, line_fr], dim=0)
                flow_s = torch.cat([flow_s, line_s], dim=0)
                flow_o = torch.cat([flow_o, line_o], dim=0)
                flow_sph = torch.cat([flow_sph, line_sph], dim=0)
                last_flow_mu = new_xyz
                last_flow_sph = new_flow_sph
            flow_mu = torch.cat([flow_mu, new_xyz], dim=0)
            new_fr = (
                torch.eye(3)
                .to(new_xyz.device)
                .unsqueeze(0)
                .expand(new_xyz.shape[0], -1, -1)
            )
            flow_fr = torch.cat([flow_fr, new_fr], dim=0)
            flow_s = torch.cat(
                [
                    flow_s,
                    torch.ones_like(new_xyz) * node_r_factor,
                ],
                dim=0,
            )
            flow_o = torch.cat([flow_o, torch.ones_like(flow_s[:, :1]) * 0.99], dim=0)
            flow_sph = torch.cat([flow_sph, new_flow_sph], dim=0)
        if len(flow_mu) > max_buffer_size:
            flow_mu = flow_mu[-max_buffer_size:]
            flow_fr = flow_fr[-max_buffer_size:]
            flow_s = flow_s[-max_buffer_size:]
            flow_o = flow_o[-max_buffer_size:]
            flow_sph = flow_sph[-max_buffer_size:]

        # gs5.append([flow_mu, flow_fr, flow_s, flow_o, flow_sph])
        gs5 = [[flow_mu, flow_fr, flow_s, flow_o, flow_sph]]
        ##################################################
        if rel_focal is None:
            rel_focal = cams.rel_focal
        render_dict = render_native(
            gs5,
            H,
            W,
            K=cams.K(H, W),
            T_cw=pose_list[cam_tid],
            bg_color=bg_color,
            color_mlp=color_mlp,
            HDR_mode=False
        )
        rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        rgb_viz = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        rgb_viz_list.append(rgb_viz)

    frame_list_bg = (frame_list_bg/np.percentile(frame_list_bg, 99.5)).clip(0,1)
    for i, h in enumerate(frame_list_bg):
        h = (tonemapReinhard.process(h) * 255).astype(np.uint8)  ## use Reinhard’s method
        traj = rgb_viz_list[i]
        blended = np.where(traj.sum(axis=2, keepdims=True) == 0, h, traj)
        rgb_viz_list[i] = blended

    save_frame_list(rgb_viz_list, save_fn + "_rgb")
    return


def viz_single_2d_video(
    H,
    W,
    cams,
    s_model,
    d_model,
    color_mlp,
    save_fn,
    pose_list,
    model_t=None,
    rel_focal=None,
    bg_flag=True,
    fg_flag=True,
    bg_color=[0.0, 0.0, 0.0],
    d_mask=None,
):
    rgb_viz_list, dep_viz_list, normal_viz_list = [], [], []
    if rel_focal is None:
        rel_focal = cams.rel_focal
    for cam_tid in tqdm(range(len(pose_list))):
        gs5 = []
        assert bg_flag or fg_flag
        if bg_flag:
            gs5.append(s_model())
        # if fg_flag:
        #     gs5.append(d_model(cam_tid if model_t is None else model_t))
        if fg_flag:
            if d_mask is None:
                gs5.append(d_model(cam_tid if model_t is None else model_t))
            else:
                _d_gs5 = d_model(cam_tid if model_t is None else model_t)
                gs5.append([it[d_mask] for it in _d_gs5])
        render_dict = render_native(
            gs5,
            H,
            W,
            K=cams.K(H, W),
            T_cw=pose_list[cam_tid],
            color_mlp=color_mlp,
            bg_color=bg_color,
            HDR_mode=True
        )
    
        rgb_h = render_dict["rgb"]
        rgb_h = rgb_h.permute(1,2,0).cpu().numpy()
        # rgb_h = rgb_h / rgb_h.max()
        rgb_h = (rgb_h/np.percentile(rgb_h, 99.5)).clip(0,1)
        rgb_h = tonemapReinhard.process(rgb_h) 
        rgb_viz = (rgb_h * 255).astype(np.uint8)

        rgb_viz_list.append(rgb_viz)
        dep = render_dict["dep"].detach().cpu().numpy().squeeze(0)
        dep_viz_list.append(dep)
        if "normal" in render_dict:
            normal = render_dict["normal"].detach().cpu().numpy()
            normal_viz = (1 - normal) / 2
            normal_viz_list.append(normal_viz.transpose(1, 2, 0))

    # # use disp map to viz the depth!
    # viz_dep = np.stack(dep_viz_list, axis=0)
    # valid_mask = viz_dep > 0
    # max_dep, min_dep = viz_dep[valid_mask].max(), viz_dep[valid_mask].min()
    # viz_dep[valid_mask] = (viz_dep[valid_mask] - min_dep) / (max_dep - min_dep)
    # # viz_dep = [plt.cm.plasma(it)[:,:,:3] * 255 for it in viz_dep]
    # viz_dep = [plt.cm.viridis(it)[:, :, :3] * 255 for it in viz_dep]
    # save_frame_list(viz_dep, save_fn + "_dep")

    save_frame_list(rgb_viz_list, save_fn + "_rgb")
    if len(normal_viz_list) > 0:
        print(normal_viz_list[0].shape)
        save_frame_list(normal_viz_list, save_fn + "_normal")
    return

def __draw_camera_pyramid__(n_pts_per_line=100, H=1.0, W=1.0, F=1.0):
    # get a list of xyz position of opencv camera pyramid
    # the forward z is facing the scene
    cam_points = np.array(
        [
            [0, 0, 0],  # camera center
            [W / 2, H / 2, F],  # top-right
            [W / 2, -H / 2, F],  # bottom-right
            [-W / 2, -H / 2, F],  # bottom-left
            [-W / 2, H / 2, F],  # top-left
        ]
    )
    lines = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),  # from center to corners
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),  # between corners
    ]
    xyz = []
    for start, end in lines:
        line_dir = cam_points[end] - cam_points[start]
        line_points = (
            cam_points[start][None, :]
            + np.linspace(0, 1, n_pts_per_line)[:, None] * line_dir[None, :]
        )
        xyz.append(line_points)
    return np.concatenate(xyz, axis=0)

def viz_single_2d_camera_video(
    H,
    W,
    cams,
    s_model,
    d_model,
    color_mlp,
    save_fn,
    pose_list,
    model_t=None,
    rel_focal=None,
    bg_flag=True,
    fg_flag=True,
    bg_color=[0.0, 0.0, 0.0],
    invisble_opa_factor=1.0,  # 0.05,
    cam_draw_scale=0.2,
    inivisble_red_ratio=0.8,
    # K=32,
):
    device = cams.T_wc(0).device
    rgb_viz_list, dep_viz_list, normal_viz_list = [], [], []
    if rel_focal is None:
        rel_focal = cams.rel_focal

    cam_H, cam_W = cams.default_H, cams.default_W
    L = float(max(cam_H, cam_W))
    cam_F = float(cams.K()[0, 0] / L)
    cam_H, cam_W = float(cam_H / L), float(cam_W / L)
    camera_mu = __draw_camera_pyramid__(H=cam_H, W=cam_W, F=cam_F)
    camera_mu = camera_mu * cam_draw_scale
    camera_mu = torch.from_numpy(camera_mu).to(device).float()

    # middle_T = cams.T // 2
    # _T_cw = pose_list[middle_T]
    # mid_cam_ori_w = cams.T_wc(middle_T)[:3, -1]
    # mid_cam_ori_c = _T_cw[:3,:3] @ mid_cam_ori_w + _T_cw[:3,-1]
    # # distance to the camera
    frame_list_bg = []
    rgb_viz_list = []
    for cam_tid in tqdm(range(len(pose_list))):
        working_t = cam_tid if model_t is None else model_t

        gs5 = []
        assert bg_flag or fg_flag
        if bg_flag:
            gs5.append(s_model())
        if fg_flag:
            gs5.append(d_model(working_t))

        # * identyfy the visible GS
        visible_render_dict = render_native(
            gs5,
            cams.default_H,
            cams.default_W,
            K=cams.K(),
            T_cw=cams.T_cw(cam_tid),
            bg_color=bg_color,
            color_mlp=color_mlp,
            HDR_mode=True
        )
        # mu_cat = torch.cat([it[0] for it in gs5], 0)
        # dep = visible_render_dict["dep"].detach()[0]
        # mask = visible_render_dict["alpha"].detach()[0] > 0.5
        # back_pts = cams.backproject(cams.homo()[mask], dep[mask])
        # back_pts_world = cams.trans_pts_to_world(working_t, back_pts)
        # dist_sq, nearest_id, _ = knn_points(back_pts_world[None], mu_cat[None], K=K)
        # dist_sq = dist_sq[0, :].reshape(-1)
        # nearest_id = nearest_id[0, :].reshape(-1)
        # valid_nn_mask = dist_sq < (d_model.scf.spatial_unit * 3.0) ** 2
        # nearest_id = nearest_id[valid_nn_mask]
        # visibl_emask = torch.zeros_like(mu_cat[:, 0]).bool()
        # if len(nearest_id) > 0:
        #     visible_mask[nearest_id] = True
        visible_mask = visible_render_dict["visibility_filter"]

        gs5_cat = []
        for i in range(5):
            gs5_cat.append(torch.cat([it[i] for it in gs5], 0))
        new_opa = gs5_cat[-2]
        new_opa[~visible_mask] = new_opa[~visible_mask] * invisble_opa_factor
        gs5_cat[-2] = new_opa

        render_dict = render_native(
            gs5_cat,
            H,
            W,
            K=cams.K(H, W),
            T_cw=pose_list[cam_tid],
            color_mlp=color_mlp,
            HDR_mode=True,
            bg_color=bg_color
        )
        pred_rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        frame_list_bg.append(pred_rgb)
        # # convert to gray scale
        # gray_sph = RGB2SH(
        #     torch.mean(
        #         SH2RGB(gs5_cat[-1][~visible_mask, :3]), dim=1, keepdim=True
        #     ).expand(-1, 3)
        # )
        # gray_sph[:, 0] = (
        #     gray_sph[:, 0] * (1.0 - inivisble_red_ratio) + inivisble_red_ratio
        # )
        # gray_sph[:, 1:] = gray_sph[:, 1:] * (1.0 - inivisble_red_ratio) + 0.0
        # pad_sph_dim = s_model()[-1].shape[1]
        # if pad_sph_dim > gray_sph.shape[1]:
        #     gray_sph = F.pad(gray_sph, (0, pad_sph_dim - gray_sph.shape[1], 0, 0))
        # gs5_cat[-1][~visible_mask] = gray_sph

        # * draw also the current camera frame in the scene
        add_mu = cams.trans_pts_to_world(working_t, camera_mu)
        add_fr = (
            torch.eye(3).to(add_mu.device).unsqueeze(0).expand(add_mu.shape[0], -1, -1)
        )
        add_s = torch.ones_like(add_mu) * 0.001
        add_o = torch.ones_like(add_s[:, :1]) * 1.0  # 0.4
        add_sph = torch.ones_like(add_s) * 0.0
        add_sph[:, 1] = 1.0
        # if pad_sph_dim > add_sph.shape[1]:
        #     add_sph = F.pad(add_sph, (0, pad_sph_dim - add_sph.shape[1], 0, 0))

        render_dict = render_native(
            [
                [
                    add_mu.to(device),
                    add_fr.to(device),
                    add_s.to(device),
                    add_o.to(device),
                    add_sph.to(device),
                ],
            ],
            H,
            W,
            K=cams.K(H, W),
            T_cw=pose_list[cam_tid],
            color_mlp=color_mlp,
            HDR_mode=False,
            bg_color=bg_color,
        )
        rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        rgb_viz = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        rgb_viz_list.append(rgb_viz)
        # dep = render_dict["dep"].detach().cpu().numpy().squeeze(0)
        # dep_viz_list.append(dep)
        # if "normal" in render_dict:
        #     normal = render_dict["normal"].detach().cpu().numpy()
        #     normal_viz = (1 - normal) / 2
        #     normal_viz_list.append(normal_viz.transpose(1, 2, 0))

        # imageio.imsave("./debug/dbg.jpg", rgb_viz)

    frame_list_bg = (frame_list_bg/np.percentile(frame_list_bg, 99.5)).clip(0,1)
    for i, h in enumerate(frame_list_bg):
        h = (tonemapReinhard.process(h) * 255).astype(np.uint8)  ## use Reinhard’s method
        viz_cam = rgb_viz_list[i]
        blended = np.where(viz_cam.sum(axis=2, keepdims=True) == 0, h, viz_cam)
        rgb_viz_list[i] = blended

    # # use disp map to viz the depth!
    # viz_dep = np.stack(dep_viz_list, axis=0)
    # valid_mask = viz_dep > 0
    # max_dep, min_dep = viz_dep[valid_mask].max(), viz_dep[valid_mask].min()
    # viz_dep[valid_mask] = (viz_dep[valid_mask] - min_dep) / (max_dep - min_dep)
    # # viz_dep = [plt.cm.plasma(it)[:,:,:3] * 255 for it in viz_dep]
    # viz_dep = [plt.cm.viridis(it)[:, :, :3] * 255 for it in viz_dep]

    # save_frame_list(viz_dep, save_fn + "_dep")

    save_frame_list(rgb_viz_list, save_fn + "_rgb")
    # if len(normal_viz_list) > 0:
    #     print(normal_viz_list[0].shape)
    #     save_frame_list(normal_viz_list, save_fn + "_normal")
    return



def save_frame_list(frame_list, name):
    os.makedirs(name, exist_ok=True)
    imageio.mimsave(name + ".mp4", frame_list)
    for i, frame in enumerate(frame_list):
        imageio.imwrite(osp.join(name, f"{i:04d}.jpg"), frame)
    return


def viz_main22(
    save_dir,
    cams,
    s_model,
    d_model=None,
    color_mlp=None,
    tone_mapper=None,
    N=5,
    move_angle_deg=20.0,
    H_3d=960,
    # H_3d=640,
    W_3d=960,
    fov_3d=70,
    back_ratio_3d=0.8,
    up_ratio=0.4,
    bg_color=[0.0, 0.0, 0.0],
):
    # H_3d=480
    # W_3d=854
    back_ratio_3d = 0.5
    # up_ratio=0.01

    H, W = cams.default_H, cams.default_W

    rel_focal_3d = 1.0 / np.tan(np.deg2rad(fov_3d) / 2.0)

    key_steps = [cams.T // 2, cams.T - 1, 0, cams.T // 4, 3 * cams.T // 4][:N]

    # * Get pose
    global_pose_list = get_global_3D_cam_T_cw(
        s_model,
        d_model,
        color_mlp,
        cams,
        H,
        W,
        cams.T // 2,
        back_ratio=back_ratio_3d,
        up_ratio=up_ratio,
    )
    global_pose_list = global_pose_list[None].expand(cams.T, -1, -1)
    training_pose_list = [cams.T_cw(t) for t in range(cams.T)]


    for key_time_step in key_steps:
        fixed_pose_list = [cams.T_cw(key_time_step) for _ in range(cams.T)]
        round_pose_list = get_move_around_cam_T_cw_large(
            s_model,
            d_model,
            color_mlp,
            cams,
            H,
            W,
            np.deg2rad(move_angle_deg),
            total_steps=cams.T,  # cams.T
            center_id=key_time_step,
        )

        # Viz rgb
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_moving_large")
        viz_single_2d_video(
            H,
            W,
            cams,
            s_model,
            d_model,
            color_mlp,
            save_fn_prefix,
            round_pose_list,
            bg_color=bg_color,
        )
        round_pose_list = get_move_around_cam_T_cw_large_add(
            s_model,
            d_model,
            color_mlp,
            cams,
            H,
            W,
            np.deg2rad(move_angle_deg),
            total_steps=cams.T,  # cams.T
            center_id=key_time_step,
        )
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_freezing_large")
        viz_single_2d_video(
            H,
            W,
            cams,
            s_model,
            d_model,
            color_mlp,
            save_fn_prefix,
            round_pose_list,
            model_t=key_time_step,
            bg_color=bg_color,
        )


def viz_main(
    save_dir,
    cams,
    s_model,
    d_model=None,
    color_mlp=None,
    tone_mapper=None,
    N=5,
    move_angle_deg=20.0,
    H_3d=960,
    # H_3d=640,
    W_3d=960,
    fov_3d=70,
    back_ratio_3d=0.8,
    up_ratio=0.4,
    bg_color=[0.0, 0.0, 0.0],
):
    # H_3d=480
    # W_3d=854
    back_ratio_3d = 0.5
    # up_ratio=0.01

    H, W = cams.default_H, cams.default_W

    rel_focal_3d = 1.0 / np.tan(np.deg2rad(fov_3d) / 2.0)

    key_steps = [cams.T // 2, cams.T - 1, 0, cams.T // 4, 3 * cams.T // 4][:N]

    # * Get pose
    global_pose_list = get_global_3D_cam_T_cw(
        s_model,
        d_model,
        color_mlp,
        cams,
        H,
        W,
        cams.T // 2,
        back_ratio=back_ratio_3d,
        up_ratio=up_ratio,
    )
    global_pose_list = global_pose_list[None].expand(cams.T, -1, -1)
    training_pose_list = [cams.T_cw(t) for t in range(cams.T)]

    # * #############################################################################

    # save_fn_prefix = osp.join(save_dir, f"3D_moving_flow_and_cam")
    # viz_single_2d_flow_camera_video22(
    #     H_3d,
    #     W_3d,
    #     cams,
    #     s_model,
    #     d_model,
    #     color_mlp,
    #     save_fn_prefix,
    #     global_pose_list,
    #     rel_focal=rel_focal_3d,
    #     bg_color=bg_color,
    #     N_max=16,
    #     node_r_factor=0.00001,  # 0.05,
    #     # line
    #     line_N=32,
    #     line_opa=0.1,
    #     line_r_factor=0.00001,
    #     cam_draw_scale=0.06,
    # )
    # exit()

    # save_fn_prefix = osp.join(save_dir, f"3D_moving_cam")
    # viz_single_2d_camera_video(
    #     H_3d,
    #     W_3d,
    #     cams,
    #     s_model,
    #     d_model,
    #     color_mlp,
    #     save_fn_prefix,
    #     global_pose_list,
    #     rel_focal=rel_focal_3d,
    #     bg_color=bg_color,
    # )

    save_fn_prefix = osp.join(save_dir, f"3D_moving_flow")
    viz_single_2d_flow_video(
        H_3d,
        W_3d,
        cams,
        s_model,
        d_model,
        color_mlp,
        save_fn_prefix,
        global_pose_list,
        rel_focal=rel_focal_3d,
        bg_color=bg_color,
    )
   
    save_fn_prefix = osp.join(save_dir, f"3D_moving")
    viz_single_2d_video(
        H_3d,
        W_3d,
        cams,
        s_model,
        d_model,
        color_mlp,
        save_fn_prefix,
        global_pose_list,
        rel_focal=rel_focal_3d,
        bg_color=bg_color,
    )


    for key_time_step in key_steps:
        fixed_pose_list = [cams.T_cw(key_time_step) for _ in range(cams.T)]
        round_pose_list = get_move_around_cam_T_cw(
            s_model,
            d_model,
            color_mlp,
            cams,
            H,
            W,
            np.deg2rad(move_angle_deg),
            total_steps=cams.T,  # cams.T
            center_id=key_time_step,
        )

        # viz flow
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_fixed_moving_flow")
        viz_single_2d_flow_video(
            H, W, cams, s_model, d_model, color_mlp, save_fn_prefix, fixed_pose_list
        )
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_moving_flow")
        viz_single_2d_flow_video(
            H, W, cams, s_model, d_model, color_mlp, save_fn_prefix, round_pose_list
        )

        # Viz rgb
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_moving")
        viz_single_2d_video(
            H,
            W,
            cams,
            s_model,
            d_model,
            color_mlp,
            save_fn_prefix,
            round_pose_list,
            bg_color=bg_color,
        )
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_freezing")
        viz_single_2d_video(
            H,
            W,
            cams,
            s_model,
            d_model,
            color_mlp,
            save_fn_prefix,
            round_pose_list,
            model_t=key_time_step,
            bg_color=bg_color,
        )
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_fixed_moving")
        viz_single_2d_video(
            H,
            W,
            cams,
            s_model,
            d_model,
            color_mlp,
            save_fn_prefix,
            fixed_pose_list,
            bg_color=bg_color,
        )

def viz_single_2d_flow_camera_video(
    H,
    W,
    cams,
    s_model,
    d_model,
    color_mlp,
    save_fn,
    pose_list,
    N_max=128,
    color_mod=5,
    max_T=1000,
    gray_scale_bg_flag=True,
    #
    node_r_factor=0.0001,  # 0.05,
    # line
    line_N=32,
    line_opa=0.5,
    line_r_factor=0.0001,
    cam_draw_scale=0.2,
    rel_focal=None,
    bg_color=[0.0, 0.0, 0.0],
):
    rgb_viz_list = []

    # ! color the node
    pts_first = d_model(0)[0]
    if len(pts_first) > N_max:
        # ! do a filtering for viz purpose, only viz dense area
        # use open3d
        inlier_mask = outlier_removal_o3d(pts_first, std_ratio=1.0)
        print(f"Filtered {len(pts_first) - inlier_mask.sum()} points")
        candidates = torch.arange(len(pts_first))[inlier_mask.cpu()]
        step = max(1, len(candidates) // N_max)
        # viz_choice = candidates[torch.randperm(len(candidates))[:N_max]]
        viz_choice = candidates[::step][:N_max]
        pts_first = pts_first[viz_choice] 
    # node_colors = map_colors(pts_first.detach().cpu().numpy(), mod=color_mod)

    color_id_coord = pts_first
    color_id = (
        color_id_coord[:, 2] + color_id_coord[:, 1] * 1e6 + color_id_coord[:, 0] * 1e12
    )
    color_id = color_id.argsort().float() / len(color_id)
    node_colors = cm.get_cmap("gist_rainbow")(color_id.cpu())[:, :3]

    flow_sph = RGB2SH(torch.from_numpy(node_colors).to(pts_first.device).float())
    # pad_sph_dim = s_model()[-1].shape[1]
    # if pad_sph_dim > flow_sph.shape[1]:
    #     flow_sph = F.pad(flow_sph, (0, pad_sph_dim - flow_sph.shape[1], 0, 0))

    flow_mu = pts_first
    flow_fr = (
        torch.eye(3).to(flow_mu.device).unsqueeze(0).expand(flow_mu.shape[0], -1, -1)
    )
    flow_s = (
        torch.ones(len(flow_mu), 3).to(flow_mu)
        * node_r_factor
    )
    flow_o = torch.ones_like(flow_s[:, :1]) * 0.99
    last_flow_mu = flow_mu
    last_flow_sph = flow_sph

    cam_H, cam_W = cams.default_H, cams.default_W
    L = float(max(cam_H, cam_W))
    cam_F = float(cams.K()[0, 0] / L)
    cam_H, cam_W = float(cam_H / L), float(cam_W / L)
    camera_mu = __draw_camera_pyramid__(H=cam_H, W=cam_W, F=cam_F)
    camera_mu = camera_mu * cam_draw_scale
    camera_mu = torch.from_numpy(camera_mu).cuda().float()

    # ! gray-scale the bg
    gs5_bg = list(s_model())
    # if gray_scale_bg_flag:
    #     bg_rgb = SH2RGB(gs5_bg[-1][:, :3])
    #     bg_gray = torch.mean(bg_rgb, dim=1, keepdim=True).expand(-1, 3)
    #     # convert to gray scale
    #     bg_sph = RGB2SH(bg_gray)
    #     if pad_sph_dim > bg_sph.shape[1]:
    #         bg_sph = F.pad(bg_sph, (0, pad_sph_dim - bg_sph.shape[1], 0, 0))
    #     gs5_bg[-1] = bg_sph

    max_buffer_size = len(flow_mu) * (line_N + 1) * max_T
    frame_list_bg = []
    rgb_viz_list = []
    cam_viz_list = []
    for cam_tid in tqdm(range(len(pose_list))):
        # working_t = cam_tid if model_t is None else model_t
        working_t = cam_tid

        ##################################################
        # make GS
        gs5 = [gs5_bg]
        d_gs5 = list(d_model(working_t))
        # d_gs5[-2] = 0.2 * d_gs5[-2]
        gs5.append(d_gs5)
        if rel_focal is None:
            rel_focal = cams.rel_focal
        render_dict = render_native(
            gs5,
            H,
            W,
            K=cams.K(H, W),
            T_cw=pose_list[cam_tid],
            color_mlp=color_mlp,
            HDR_mode=True,
            bg_color=bg_color,
        )
        pred_rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        frame_list_bg.append(pred_rgb)
        if cam_tid > 0:
            new_xyz = d_gs5[0][viz_choice]
            new_flow_sph = last_flow_sph
            # first draw lines of the flow
            if line_N > 0:
                line_xyz = draw_gs_point_line(new_xyz, last_flow_mu, n=line_N).reshape(
                    -1, 3
                )
                line_fr = (
                    torch.eye(3)
                    .to(flow_mu.device)
                    .unsqueeze(0)
                    .expand(line_xyz.shape[0], -1, -1)
                )
                line_s = (
                    torch.ones_like(line_xyz) * line_r_factor
                )
                line_o = torch.ones_like(line_s[:, :1]) * line_opa
                line_sph = draw_gs_point_line(
                    new_flow_sph,
                    last_flow_sph,
                    n=line_N,
                ).reshape(-1, flow_sph.shape[-1])
                flow_mu = torch.cat([flow_mu, line_xyz], dim=0)
                flow_fr = torch.cat([flow_fr, line_fr], dim=0)
                flow_s = torch.cat([flow_s, line_s], dim=0)
                flow_o = torch.cat([flow_o, line_o], dim=0)
                flow_sph = torch.cat([flow_sph, line_sph], dim=0)
                last_flow_mu = new_xyz
                last_flow_sph = new_flow_sph
            flow_mu = torch.cat([flow_mu, new_xyz], dim=0)
            new_fr = (
                torch.eye(3)
                .to(new_xyz.device)
                .unsqueeze(0)
                .expand(new_xyz.shape[0], -1, -1)
            )
            flow_fr = torch.cat([flow_fr, new_fr], dim=0)
            flow_s = torch.cat(
                [
                    flow_s,
                    torch.ones_like(new_xyz) * node_r_factor,
                ],
                dim=0,
            )
            flow_o = torch.cat([flow_o, torch.ones_like(flow_s[:, :1]) * 0.99], dim=0)
            flow_sph = torch.cat([flow_sph, new_flow_sph], dim=0)
        if len(flow_mu) > max_buffer_size:
            flow_mu = flow_mu[-max_buffer_size:]
            flow_fr = flow_fr[-max_buffer_size:]
            flow_s = flow_s[-max_buffer_size:]
            flow_o = flow_o[-max_buffer_size:]
            flow_sph = flow_sph[-max_buffer_size:]

        # gs5.append([flow_mu, flow_fr, flow_s, flow_o, flow_sph])
        gs5 = [[flow_mu, flow_fr, flow_s, flow_o, flow_sph]]
        ##################################################
        if rel_focal is None:
            rel_focal = cams.rel_focal
        render_dict = render_native(
            gs5,
            H,
            W,
            K=cams.K(H, W),
            T_cw=pose_list[cam_tid],
            bg_color=bg_color,
            color_mlp=color_mlp,
            HDR_mode=False
        )
        rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        rgb_viz = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        rgb_viz_list.append(rgb_viz)

        add_mu = cams.trans_pts_to_world(working_t, camera_mu)
        add_fr = (
            torch.eye(3).to(add_mu.device).unsqueeze(0).expand(add_mu.shape[0], -1, -1)
        )
        add_s = torch.ones_like(add_mu) * 0.00005
        add_o = torch.ones_like(add_s[:, :1]) * 1.0  # 0.4
        add_sph = torch.ones_like(add_s) * 0.0
        add_sph[:, 2] = 1.0
        # if pad_sph_dim > add_sph.shape[1]:
        #     add_sph = F.pad(add_sph, (0, pad_sph_dim - add_sph.shape[1], 0, 0))

        render_dict = render_native(
            [
                [
                    add_mu.cuda(),
                    add_fr.cuda(),
                    add_s.cuda(),
                    add_o.cuda(),
                    add_sph.cuda(),
                ],
            ],
            H,
            W,
            K=cams.K(H, W),
            T_cw=pose_list[cam_tid],
            color_mlp=color_mlp,
            HDR_mode=False,
        )
        rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        rgb_viz = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        # rgb_viz[rgb_viz.sum(axis=2) != 0] = [0, 0, 255]
        cam_viz_list.append(rgb_viz)

    frame_list_bg = (frame_list_bg/np.percentile(frame_list_bg, 99.5)).clip(0,1)
    for i, h in enumerate(frame_list_bg):
        h = (tonemapReinhard.process(h) * 255).astype(np.uint8)  ## use Reinhard’s method
        traj = rgb_viz_list[i]
        cam_viz = cam_viz_list[i]
        traj_cam = np.where(cam_viz.sum(axis=2, keepdims=True) == 0, traj, cam_viz)
        blended = np.where(traj_cam.sum(axis=2, keepdims=True) == 0, h, traj_cam)
        rgb_viz_list[i] = blended

    save_frame_list(rgb_viz_list, save_fn + "_rgb")
    return


def viz_single_2d_flow_camera_video22(
    H,
    W,
    cams,
    s_model,
    d_model,
    color_mlp,
    save_fn,
    pose_list,
    N_max=128,
    color_mod=5,
    max_T=1000,
    gray_scale_bg_flag=True,
    #
    node_r_factor=0.0001,  # 0.05,
    # line
    line_N=32,
    line_opa=0.5,
    line_r_factor=0.0001,
    cam_draw_scale=0.2,
    rel_focal=None,
    bg_color=[0.0, 0.0, 0.0],
):
    rgb_viz_list = []

    # ! color the node
    pts_first = d_model(0)[0]
    if len(pts_first) > N_max:
        # ! do a filtering for viz purpose, only viz dense area
        # use open3d
        inlier_mask = outlier_removal_o3d(pts_first, std_ratio=1.0)
        print(f"Filtered {len(pts_first) - inlier_mask.sum()} points")
        candidates = torch.arange(len(pts_first))[inlier_mask.cpu()]
        step = max(1, len(candidates) // N_max)
        # viz_choice = candidates[torch.randperm(len(candidates))[:N_max]]
        viz_choice = candidates[::step][:N_max]
        pts_first = pts_first[viz_choice] 
    # node_colors = map_colors(pts_first.detach().cpu().numpy(), mod=color_mod)

    color_id_coord = pts_first
    color_id = (
        color_id_coord[:, 2] + color_id_coord[:, 1] * 1e6 + color_id_coord[:, 0] * 1e12
    )
    color_id = color_id.argsort().float() / len(color_id)
    node_colors = cm.get_cmap("gist_rainbow")(color_id.cpu())[:, :3]

    flow_sph = RGB2SH(torch.from_numpy(node_colors).to(pts_first.device).float())
    # pad_sph_dim = s_model()[-1].shape[1]
    # if pad_sph_dim > flow_sph.shape[1]:
    #     flow_sph = F.pad(flow_sph, (0, pad_sph_dim - flow_sph.shape[1], 0, 0))

    flow_mu = pts_first
    flow_fr = (
        torch.eye(3).to(flow_mu.device).unsqueeze(0).expand(flow_mu.shape[0], -1, -1)
    )
    flow_s = (
        torch.ones(len(flow_mu), 3).to(flow_mu)
        * node_r_factor
    )
    flow_o = torch.ones_like(flow_s[:, :1]) * 0.99
    last_flow_mu = flow_mu
    last_flow_sph = flow_sph

    cam_H, cam_W = cams.default_H, cams.default_W
    L = float(max(cam_H, cam_W))
    cam_F = float(cams.K()[0, 0] / L)
    cam_H, cam_W = float(cam_H / L), float(cam_W / L)
    camera_mu = __draw_camera_pyramid__(H=cam_H, W=cam_W, F=cam_F)
    camera_mu = camera_mu * cam_draw_scale
    camera_mu = torch.from_numpy(camera_mu).cuda().float()

    # ! gray-scale the bg
    gs5_bg = list(s_model())
    # if gray_scale_bg_flag:
    #     bg_rgb = SH2RGB(gs5_bg[-1][:, :3])
    #     bg_gray = torch.mean(bg_rgb, dim=1, keepdim=True).expand(-1, 3)
    #     # convert to gray scale
    #     bg_sph = RGB2SH(bg_gray)
    #     if pad_sph_dim > bg_sph.shape[1]:
    #         bg_sph = F.pad(bg_sph, (0, pad_sph_dim - bg_sph.shape[1], 0, 0))
    #     gs5_bg[-1] = bg_sph

    max_buffer_size = len(flow_mu) * (line_N + 1) * max_T
    frame_list_bg = []
    rgb_viz_list = []
    cam_viz_list = []
    t1=  0
    t2 = (len(pose_list)-1) // 2 
    for cam_tid in tqdm(range(len(pose_list))):
        # working_t = cam_tid if model_t is None else model_t
        working_t = cam_tid

        ##################################################
        # make GS
        gs5 = [gs5_bg]
        d_gs5 = list(d_model(working_t))
        # d_gs5[-2] = 0.2 * d_gs5[-2]
        gs5.append(d_gs5)

        dd_gs5 = list(d_model(t2))
        dd_gs5[-2] = 0.2 * dd_gs5[-2]
        gs5.append(dd_gs5)

        dd_gs5 = list(d_model(t1))
        dd_gs5[-2] = 0.2 * dd_gs5[-2]
        gs5.append(dd_gs5)

        if rel_focal is None:
            rel_focal = cams.rel_focal
        render_dict = render_native(
            gs5,
            H,
            W,
            K=cams.K(H, W),
            T_cw=pose_list[cam_tid],
            color_mlp=color_mlp,
            HDR_mode=True,
            bg_color=bg_color,
        )
        pred_rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        frame_list_bg.append(pred_rgb)
        if cam_tid > 0:
            new_xyz = d_gs5[0][viz_choice]
            new_flow_sph = last_flow_sph
            # first draw lines of the flow
            if line_N > 0:
                line_xyz = draw_gs_point_line(new_xyz, last_flow_mu, n=line_N).reshape(
                    -1, 3
                )
                line_fr = (
                    torch.eye(3)
                    .to(flow_mu.device)
                    .unsqueeze(0)
                    .expand(line_xyz.shape[0], -1, -1)
                )
                line_s = (
                    torch.ones_like(line_xyz) * line_r_factor
                )
                line_o = torch.ones_like(line_s[:, :1]) * line_opa
                line_sph = draw_gs_point_line(
                    new_flow_sph,
                    last_flow_sph,
                    n=line_N,
                ).reshape(-1, flow_sph.shape[-1])
                flow_mu = torch.cat([flow_mu, line_xyz], dim=0)
                flow_fr = torch.cat([flow_fr, line_fr], dim=0)
                flow_s = torch.cat([flow_s, line_s], dim=0)
                flow_o = torch.cat([flow_o, line_o], dim=0)
                flow_sph = torch.cat([flow_sph, line_sph], dim=0)
                last_flow_mu = new_xyz
                last_flow_sph = new_flow_sph
            flow_mu = torch.cat([flow_mu, new_xyz], dim=0)
            new_fr = (
                torch.eye(3)
                .to(new_xyz.device)
                .unsqueeze(0)
                .expand(new_xyz.shape[0], -1, -1)
            )
            flow_fr = torch.cat([flow_fr, new_fr], dim=0)
            flow_s = torch.cat(
                [
                    flow_s,
                    torch.ones_like(new_xyz) * node_r_factor,
                ],
                dim=0,
            )
            flow_o = torch.cat([flow_o, torch.ones_like(flow_s[:, :1]) * 0.99], dim=0)
            flow_sph = torch.cat([flow_sph, new_flow_sph], dim=0)
        if len(flow_mu) > max_buffer_size:
            flow_mu = flow_mu[-max_buffer_size:]
            flow_fr = flow_fr[-max_buffer_size:]
            flow_s = flow_s[-max_buffer_size:]
            flow_o = flow_o[-max_buffer_size:]
            flow_sph = flow_sph[-max_buffer_size:]

        # gs5.append([flow_mu, flow_fr, flow_s, flow_o, flow_sph])
        gs5 = [[flow_mu, flow_fr, flow_s, flow_o, flow_sph]]
        ##################################################
        if rel_focal is None:
            rel_focal = cams.rel_focal
        render_dict = render_native(
            gs5,
            H,
            W,
            K=cams.K(H, W),
            T_cw=pose_list[cam_tid],
            bg_color=bg_color,
            color_mlp=color_mlp,
            HDR_mode=False
        )
        rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        rgb_viz = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        rgb_viz_list.append(rgb_viz)

        add_mu = cams.trans_pts_to_world(working_t, camera_mu)
        add_fr = (
            torch.eye(3).to(add_mu.device).unsqueeze(0).expand(add_mu.shape[0], -1, -1)
        )
        add_s = torch.ones_like(add_mu) * 0.0005
        add_o = torch.ones_like(add_s[:, :1]) * 1.0  # 0.4
        add_sph = torch.ones_like(add_s) * 0.0
        add_sph[:, 2] = 1.0
        # if pad_sph_dim > add_sph.shape[1]:
        #     add_sph = F.pad(add_sph, (0, pad_sph_dim - add_sph.shape[1], 0, 0))
        # add_sph = RGB2SH(add_sph)
        render_dict = render_native(
            [
                [
                    add_mu.cuda(),
                    add_fr.cuda(),
                    add_s.cuda(),
                    add_o.cuda(),
                    add_sph.cuda(),
                ],
            ],
            H,
            W,
            K=cams.K(H, W),
            T_cw=pose_list[cam_tid],
            color_mlp=color_mlp,
            HDR_mode=False,
        )
        rgb = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        rgb_viz = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        # rgb_viz[rgb_viz.sum(axis=2) != 0] = [0, 0, 255]
        cam_viz_list.append(rgb_viz)

    frame_list_bg = (frame_list_bg/np.percentile(frame_list_bg, 99.5)).clip(0,1)
    for i, h in enumerate(frame_list_bg):
        h = (tonemapReinhard.process(h) * 255).astype(np.uint8)  ## use Reinhard’s method
        traj = rgb_viz_list[i]
        cam_viz = cam_viz_list[i]
        aa = np.where(cam_viz_list[t2].sum(axis=2, keepdims=True) == 0, cam_viz_list[t1], cam_viz_list[t2])
        cam_viz = np.where(cam_viz.sum(axis=2, keepdims=True) == 0, aa, cam_viz)
        # traj_cam = np.where(cam_viz.sum(axis=2, keepdims=True) == 0, traj, cam_viz)
        traj_cam = cam_viz
        blended = np.where(traj_cam.sum(axis=2, keepdims=True) == 0, h, traj_cam)
        rgb_viz_list[i] = blended

    save_frame_list(rgb_viz_list, save_fn + "_rgb")
    return

def get_move_around_cam_T_cw(
    s_model,
    d_model,
    color_mlp,
    cams,
    H,
    W,
    move_around_angle_deg,
    total_steps,
    center_id=None,
):

    # in the xy plane, the new camera is forming a circle
    move_around_view_list = []
    for i in tqdm(range(total_steps)):
        if center_id is None:
            move_around_id = i
            assert total_steps - 1 < cams.T
            render_dict = render_native(
                [s_model(), d_model(move_around_id)],
                H,
                W,
                K=cams.K(H, W),
                T_cw=cams.T_cw(move_around_id),
                color_mlp=color_mlp,
                HDR_mode=True
            )
            # depth = (render_dict["dep"] / (render_dict["alpha"] + 1e-6))[0]
            depth = render_dict["dep"][0]
            center_dep = depth[depth.shape[0] // 2, depth.shape[1] // 2].item()
            if center_dep < 1e-2:
                center_dep = depth[render_dict["alpha"][0] > 0.1].min().item()
            focus_point = torch.Tensor([0.0, 0.0, center_dep]).to(depth)
            move_around_radius = np.tan(move_around_angle_deg) * focus_point[2].item()
        else:
            move_around_id = center_id
            if i == 0:
                render_dict = render_native(
                    [s_model(), d_model(move_around_id)],
                    H,
                    W,
                    K=cams.K(H, W),
                    T_cw=cams.T_cw(move_around_id),
                    color_mlp=color_mlp,
                    HDR_mode=True
                )
                # depth = (render_dict["dep"] / (render_dict["alpha"] + 1e-6))[0]
                depth = render_dict["dep"][0]
                center_dep = depth[depth.shape[0] // 2, depth.shape[1] // 2].item()
                if center_dep < 1e-2:
                    center_dep = depth[render_dict["alpha"][0] > 0.1].min().item()
                focus_point = torch.Tensor([0.0, 0.0, center_dep]).to(depth)
                move_around_radius = (
                    np.tan(move_around_angle_deg) * focus_point[2].item()
                )

        x = (
            move_around_radius * np.cos(2 * np.pi * i / (total_steps - 1))
            - move_around_radius
        )
        y = move_around_radius * np.sin(2 * np.pi * i / (total_steps - 1))
        T_c_new = torch.eye(4).to(cams.T_wc(0))
        T_c_new[0, -1] = x
        T_c_new[1, -1] = y
        _z_dir = F.normalize(focus_point[:3] - T_c_new[:3, -1], dim=0)
        _x_dir = F.normalize(
            torch.cross(torch.Tensor([0.0, 1.0, 0.0]).to(_z_dir), _z_dir), dim=0
        )
        _y_dir = F.normalize(torch.cross(_z_dir, _x_dir), dim=0)
        T_c_new[:3, 0] = _x_dir
        T_c_new[:3, 1] = _y_dir
        T_c_new[:3, 2] = _z_dir

        T_base = cams.T_wc(move_around_id)

        T_w_new = T_base @ T_c_new
        T_new_w = T_w_new.inverse()
        move_around_view_list.append(T_new_w)
    return move_around_view_list

def get_move_around_cam_T_cw_large(
    s_model,
    d_model,
    color_mlp,
    cams,
    H,
    W,
    move_around_angle_deg,
    total_steps,
    center_id=None,
    large_radius_factor=2.0,
    height_variation_factor=1.0,
    forward_backward_factor=1.5,
):
    """
    Generate camera poses with larger view changes for novel view rendering.
    Camera rotation matrix is kept identical to the base camera while the translation
    component follows a user-controlled trajectory expressed directly in camera space.

    Motion design (executed in the original camera's local frame):
        1. X/Y axes (left-right & up-down) trace a circle/ellipse
        2. Z axis (forward/backward) oscillates to mimic zoom in/out

    Args:
        large_radius_factor: Scales left/right excursion along camera-right axis
        height_variation_factor: Scales up/down excursion along camera-up axis
        forward_backward_factor: Scales forward/back excursion along camera-forward axis

    Note: No scene-center estimation or extra rendering is involved—poses are built
    purely from the reference camera extrinsics, making the trajectory lightweight
    yet expressive for demonstrating wide view changes.
    """
    
    move_around_view_list = []
    # Use move_around_angle_deg directly as a scale for translation distance
    base_translation_scale = move_around_angle_deg if move_around_angle_deg > 0 else 1.0
    
    # Split total_steps into two segments
    # First half: XY elliptical motion
    # Second half: Z-axis forward/backward motion
    half_steps = total_steps // 2
    first_segment_steps = half_steps
    second_segment_steps = total_steps - half_steps

    for i in tqdm(range(total_steps)):
        move_around_id = center_id if center_id is not None else i
        move_around_id = int(np.clip(move_around_id, 0, cams.T - 1))

        T_base = cams.T_wc(move_around_id)
        R_base = T_base[:3, :3]

        if i < first_segment_steps:
            # First segment: Elliptical motion in camera XY plane
            # Horizontal (X) has larger amplitude, Vertical (Y) has smaller amplitude
            # Start from origin (offset by -π/2 so cos(-π/2)=0, sin(-π/2)=-1, then shift up)
            angle = 2 * np.pi * i / first_segment_steps - np.pi / 2
            
            # Ellipse: larger radius for horizontal (X), smaller for vertical (Y)
            horizontal_scale = large_radius_factor * 2.4  # Increase horizontal amplitude
            vertical_scale = height_variation_factor * 0.5  # Decrease vertical amplitude
            
            circle_offset_local = torch.tensor(
                [
                    base_translation_scale * horizontal_scale * (np.cos(angle) - np.cos(-np.pi/2)),
                    base_translation_scale * vertical_scale * (np.sin(angle) - np.sin(-np.pi/2)),
                    0.0,
                ],
                device=T_base.device,
                dtype=T_base.dtype,
            )
            total_offset_local = circle_offset_local
        else:
            # Second segment: Forward/backward motion along camera Z axis
            # Start from origin, move forward and back
            segment_idx = i - first_segment_steps
            z_angle = 2 * np.pi * segment_idx / second_segment_steps
            
            zoom_offset_local = torch.tensor(
                [
                    0.0,
                    0.0,
                    base_translation_scale * forward_backward_factor * 4.0 * np.sin(z_angle),
                ],
                device=T_base.device,
                dtype=T_base.dtype,
            )
            total_offset_local = zoom_offset_local

        # Transform offset to world frame
        translation_offset_world = R_base @ total_offset_local

        # Apply translation while keeping rotation unchanged
        T_w_new = T_base.clone()
        T_w_new[:3, 3] = T_base[:3, 3] + translation_offset_world

        T_new_w = T_w_new.inverse()
        move_around_view_list.append(T_new_w)
    
    return move_around_view_list


def get_move_around_cam_T_cw_large_add(
    s_model,
    d_model,
    color_mlp,
    cams,
    H,
    W,
    move_around_angle_deg,
    total_steps,
    center_id=None,
    large_radius_factor=2.0,
    height_variation_factor=1.0,
    forward_backward_factor=1.5,
):
    """
    Generate camera poses with larger view changes for novel view rendering.
    Camera rotation matrix is kept identical to the base camera while the translation
    component follows a user-controlled trajectory expressed directly in camera space.

    Motion design (executed in the original camera's local frame):
        1. X/Y axes (left-right & up-down) trace a circle/ellipse
        2. Z axis (forward/backward) oscillates to mimic zoom in/out

    Args:
        large_radius_factor: Scales left/right excursion along camera-right axis
        height_variation_factor: Scales up/down excursion along camera-up axis
        forward_backward_factor: Scales forward/back excursion along camera-forward axis

    Note: No scene-center estimation or extra rendering is involved—poses are built
    purely from the reference camera extrinsics, making the trajectory lightweight
    yet expressive for demonstrating wide view changes.
    """
    
    move_around_view_list = []
    # Use move_around_angle_deg directly as a scale for translation distance
    base_translation_scale = move_around_angle_deg if move_around_angle_deg > 0 else 1.0
    
    # Two segments: each segment has total_steps frames
    # First segment: XY elliptical motion (total_steps frames)
    # Second segment: Z-axis forward/backward motion (total_steps frames)
    first_segment_steps = total_steps
    second_segment_steps = total_steps
    total_frames = first_segment_steps + second_segment_steps

    for i in tqdm(range(total_frames)):
        # Map frame index to valid time index in [0, cams.T-1]
        if i < first_segment_steps:
            # First segment: use time index i % cams.T
            time_id = center_id if center_id is not None else (i % cams.T)
        else:
            # Second segment: use time index (i - first_segment_steps) % cams.T
            time_id = center_id if center_id is not None else ((i - first_segment_steps) % cams.T)
        
        move_around_id = int(np.clip(time_id, 0, cams.T - 1))

        T_base = cams.T_wc(move_around_id)
        R_base = T_base[:3, :3]

        if i < first_segment_steps:
            # First segment: Elliptical motion in camera XY plane
            # Horizontal (X) has larger amplitude, Vertical (Y) has smaller amplitude
            # Start from origin (offset by -π/2 so cos(-π/2)=0, sin(-π/2)=-1, then shift up)
            angle = 2 * np.pi * i / first_segment_steps - np.pi / 2
            
            # Ellipse: larger radius for horizontal (X), smaller for vertical (Y)
            horizontal_scale = large_radius_factor * 2.4  # Increase horizontal amplitude
            vertical_scale = height_variation_factor * 0.5  # Decrease vertical amplitude
            
            circle_offset_local = torch.tensor(
                [
                    base_translation_scale * horizontal_scale * (np.cos(angle) - np.cos(-np.pi/2)),
                    base_translation_scale * vertical_scale * (np.sin(angle) - np.sin(-np.pi/2)),
                    0.0,
                ],
                device=T_base.device,
                dtype=T_base.dtype,
            )
            total_offset_local = circle_offset_local
        else:
            # Second segment: Forward/backward motion along camera Z axis
            # Start from origin, move forward and back
            segment_idx = i - first_segment_steps
            z_angle = 2 * np.pi * segment_idx / second_segment_steps
            
            zoom_offset_local = torch.tensor(
                [
                    0.0,
                    0.0,
                    base_translation_scale * forward_backward_factor * 4.0 * np.sin(z_angle),
                ],
                device=T_base.device,
                dtype=T_base.dtype,
            )
            total_offset_local = zoom_offset_local

        # Transform offset to world frame
        translation_offset_world = R_base @ total_offset_local

        # Apply translation while keeping rotation unchanged
        T_w_new = T_base.clone()
        T_w_new[:3, 3] = T_base[:3, 3] + translation_offset_world

        T_new_w = T_w_new.inverse()
        move_around_view_list.append(T_new_w)
    
    return move_around_view_list