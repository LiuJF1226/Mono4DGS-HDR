import sys, os, os.path as osp

sys.path.append(osp.abspath(osp.dirname(__file__)))
import torch
from dav.pipelines import DAVPipeline
from dav.models import UNetSpatioTemporalRopeConditionModel
from diffusers import AutoencoderKLTemporalDecoder, FlowMatchEulerDiscreteScheduler
from dav.utils import img_utils
import numpy as np
import cv2
from depth_utils import viz_depth_list, save_depth_list


class DAVDemo:
    def __init__(
        self,
        model_base: str,
        local_files_only=False,
    ):
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
                model_base, 
                subfolder="vae",      
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                local_files_only=local_files_only
            )
    
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                model_base, 
                subfolder="scheduler",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                local_files_only=local_files_only
            )
        unet = UNetSpatioTemporalRopeConditionModel.from_pretrained(
                model_base, 
                subfolder="unet",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                local_files_only=local_files_only
            )
        unet_interp = UNetSpatioTemporalRopeConditionModel.from_pretrained(
                model_base, 
                subfolder="unet_interp",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                local_files_only=local_files_only
            )

        self.pipe = DAVPipeline(
                vae=vae,
                unet=unet,
                unet_interp=unet_interp,
                scheduler=scheduler,
            )
      
        self.pipe.to("cuda")

        # enable attention slicing and xformers memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
            print("Xformers is not enabled")
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        img_list,
        num_frames=16,
        num_overlap_frames=6,
        num_interp_frames=16,
        decode_chunk_size=8,
        denoise_steps=3,
    ):
        assert num_frames % 2 == 0, "num_frames should be even."
        assert (
            2 <= num_overlap_frames <= (num_interp_frames + 2 + 1) // 2
        ), "Invalid frame overlap."
        max_frames = (num_interp_frames + 2 - num_overlap_frames) * (num_frames // 2)
        assert len(img_list) <= max_frames, f"Too many frames {len(img_list)} > {max_frames}"

        image_tensor = np.ascontiguousarray(
            [_img.transpose(2, 0, 1) / 255.0 for _img in img_list]
        )
        image_tensor = torch.from_numpy(image_tensor).cuda()

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            pipe_out = self.pipe(
                image_tensor,
                num_frames=num_frames,
                num_overlap_frames=num_overlap_frames,
                num_interp_frames=num_interp_frames,
                decode_chunk_size=decode_chunk_size,
                num_inference_steps=denoise_steps,
            )
        disparity = pipe_out.disparity

        return disparity

def get_DAV_model(model_base):
    try:
        model = DAVDemo(
            model_base=model_base,
            local_files_only=True,
        )
        return model
    except:
        print("Failed to load model from cache, try to download from the internet")
        model = DAVDemo(
            model_base=model_base,
        )
        return model

def DAV_process_folder(
    model,
    img_list,
    fn_list,
    dst,
    invalid_mask_list=None,
):
    img_list = np.asarray(img_list)
    # if img_list.dtype == np.uint8:
    #     img_list = img_list.astype(np.float32) / 255.0
    T, H, W, C = img_list.shape

    # adjust the maximum resolution according to GPU memory
    img_list = img_utils.imresize_max(img_list, 750)
    H2, W2, C = img_list[0].shape

    # reshape to 32 base size
    new_H = H2 // 32 * 32
    new_W = W2 // 32 * 32
    # use opencv to resize
    working_img_list = []
    for img in img_list:
        img = cv2.resize(img.copy(), (new_W, new_H))
        working_img_list.append(img)
    working_img_list = np.asarray(working_img_list)
    _dep_list = model.infer(
        working_img_list,
        num_frames=16,
        num_overlap_frames=6,  
        num_interp_frames=16,
        decode_chunk_size=8,
        denoise_steps=3
    )

    dep_list = []
    for dep in _dep_list:
        dep = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST_EXACT)
        dep_list.append(dep)
    dep_list = np.asarray(dep_list)

    save_depth_list(dep_list, fn_list, dst, invalid_mask_list)
    viz_depth_list(dep_list, dst + ".mp4")
    return

