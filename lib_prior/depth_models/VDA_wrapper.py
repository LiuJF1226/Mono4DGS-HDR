import sys, os, os.path as osp

sys.path.append(osp.abspath(osp.dirname(__file__)))
import torch
from video_depth_anything.video_depth import VideoDepthAnything
from depth_utils import viz_depth_list, save_depth_list

def get_VDA_model(ckpt_path, device, encoder="vitl"):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    video_depth_anything = VideoDepthAnything(**model_configs[encoder])
    video_depth_anything.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=True)

    video_depth_anything.to(device)
    video_depth_anything.eval()
    return video_depth_anything

def VDA_process_folder(
    model,
    img_list,
    fn_list,
    dst,
    invalid_mask_list=None,
):
    depths, _ = model.infer_video_depth(img_list, target_fps=-1, input_size=518, fp32=True)

    save_depth_list(depths, fn_list, dst, invalid_mask_list)
    viz_depth_list(depths, dst + ".mp4")
    return
