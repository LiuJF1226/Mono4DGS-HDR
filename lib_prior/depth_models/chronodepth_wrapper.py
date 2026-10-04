import sys, os, os.path as osp

sys.path.append(osp.abspath(osp.dirname(__file__)))
import torch

from chronodepth.unet_chronodepth import DiffusersUNetSpatioTemporalConditionModelChronodepth
from chronodepth.chronodepth_pipeline import ChronoDepthPipeline
import numpy as np
import cv2
from depth_utils import viz_depth_list, save_depth_list


class ChronoDepthDemo:
    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload=None,
        local_files_only=False,
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelChronodepth.from_pretrained(
                unet_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                local_files_only=local_files_only,
            )
        self.pipe = ChronoDepthPipeline.from_pretrained(
                pre_train_path,
                unet=unet,
                torch_dtype=torch.float16,
                variant="fp16",
                local_files_only=local_files_only,
            )
        self.pipe.n_tokens = 10
        self.pipe.chunk_size = 5


        # for saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
        if cpu_offload is not None:
            if cpu_offload == "sequential":
                # This will slow, but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to("cuda")
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
            print("Xformers is not enabled")
            
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        img_list,
        infer_mode="ours",
        sigma_epsilon=-4.0,
        decode_chunk_size=8,
        denoise_steps=5,
    ):
        generator = torch.Generator(device="cuda").manual_seed(12345)
        with torch.inference_mode():
            pipe_out = self.pipe(
                img_list,
                num_inference_steps=denoise_steps,
                decode_chunk_size=decode_chunk_size,
                motion_bucket_id=127,
                fps=7,
                noise_aug_strength=0.0,
                generator=generator,
                infer_mode=infer_mode,
                sigma_epsilon=sigma_epsilon,
            )
        depth = pipe_out.frames[:, 0]
        disparity = 1.0 / (depth + 1e-8)

        return disparity

def get_chronodepth_model():
    try:
        model = ChronoDepthDemo(
            unet_path="jhshao/ChronoDepth-v1",
            pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
            cpu_offload="model",
            local_files_only=True,
        )
        return model
    except:
        print("Failed to load model from cache, try to download from the internet")
        model = ChronoDepthDemo(
            unet_path="jhshao/ChronoDepth-v1",
            pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
            cpu_offload="model"
        )
        return model


def chronodepth_process_folder(
    model,
    img_list,
    fn_list,
    dst,
    invalid_mask_list=None,
):

    img_list = np.asarray(img_list)
    if img_list.dtype == np.uint8:
        img_list = img_list.astype(np.float32) / 255.0
    T, H, W, C = img_list.shape

    # reshape to 64 base size
    new_H = H // 64 * 64
    new_W = W // 64 * 64
    # use opencv to resize
    working_img_list = []
    for img in img_list:
        img = cv2.resize(img.copy(), (new_W, new_H))
        working_img_list.append(img)
    working_img_list = np.asarray(working_img_list)
    _dep_list = model.infer(
        working_img_list, 
        infer_mode="ours",
        sigma_epsilon=-4.0,
        decode_chunk_size=8,
        denoise_steps=5,
    )

    dep_list = []
    for dep in _dep_list:
        dep = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST_EXACT)
        dep_list.append(dep)
    dep_list = np.asarray(dep_list)
    save_depth_list(dep_list, fn_list, dst, invalid_mask_list)
    viz_depth_list(dep_list, dst + ".mp4")
    return


