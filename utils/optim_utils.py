import numpy as np
import torch
from torch import nn
import logging


class GSControlCFG:
    def __init__(
        self,
        densify_steps=300,
        reset_steps=900,
        prune_steps=300,
        densify_max_grad=0.0002,
        densify_percent_dense=0.01,
        prune_opacity_th=0.012,
        reset_opacity=0.01
    ):
        if isinstance(densify_steps, int):
            densify_steps = [densify_steps * i for i in range(1000)]
            # densify_steps = [step for step in densify_steps if step >=start and step < end]
        if isinstance(reset_steps, int):
            reset_steps = [reset_steps * i for i in range(1000)]
            # reset_steps = [step for step in reset_steps if step >=start and step < end]
        if isinstance(prune_steps, int):
            prune_steps = [prune_steps * i for i in range(1000)]
            # prune_steps = [step for step in prune_steps if step >=start and step < end]

        self.densify_steps = densify_steps
        self.reset_steps = reset_steps
        self.prune_steps = prune_steps
        self.densify_max_grad = densify_max_grad
        self.densify_percent_dense = densify_percent_dense
        self.prune_opacity_th = prune_opacity_th
        self.reset_opacity = reset_opacity
        self.summary()

    def summary(self):
        logging.info("GSControlCFG: Summary")
        logging.info(
            f"GSControlCFG: densify_steps={self.densify_steps[:min(5, len(self.densify_steps))]}..."
        )
        logging.info(
            f"GSControlCFG: reset_steps={self.reset_steps[:min(5, len(self.densify_steps))]}..."
        )
        logging.info(
            f"GSControlCFG: prune_steps={self.prune_steps[:min(5, len(self.densify_steps))]}..."
        )
        logging.info(f"GSControlCFG: densify_max_grad={self.densify_max_grad}")
        logging.info(
            f"GSControlCFG: densify_percent_dense={self.densify_percent_dense}"
        )
        logging.info(f"GSControlCFG: prune_opacity_th={self.prune_opacity_th}")
        logging.info(f"GSControlCFG: reset_opacity={self.reset_opacity}")
   
    

class OptimCFG:
    def __init__(
        self,
        # GS
        lr_p=0.00016,
        lr_q=0.001,
        lr_s=0.005,
        lr_o=0.05,
        lr_sph=0.0025,
        lr_sph_rest_factor=20.0,
        lr_p_final=None,
        lr_cam_q=0.0001,
        lr_cam_t=0.0001,
        lr_cam_f=0.00,
        lr_cam_q_final=None,
        lr_cam_t_final=None,
        lr_cam_f_final=None,
        # # dyn
        lr_np=0.00016,
        lr_nq=0.001,
        lr_nsig=0.00001,
        lr_w=0.0,  # ! use 0.0
        lr_dyn=0.01,
        lr_np_final=None,
        lr_nq_final=None,
        lr_w_final=None,
    ) -> None:
        # gs
        self.lr_p = lr_p
        self.lr_q = lr_q
        self.lr_s = lr_s
        self.lr_o = lr_o
        self.lr_sph = lr_sph
        self.lr_sph_rest = lr_sph / lr_sph_rest_factor
        # cam
        self.lr_cam_q = lr_cam_q
        self.lr_cam_t = lr_cam_t
        self.lr_cam_f = lr_cam_f
        # # dyn
        self.lr_np = lr_np
        self.lr_nq = lr_nq
        self.lr_w = lr_w
        self.lr_dyn = lr_dyn
        self.lr_nsig = lr_nsig

        # gs scheduler
        self.lr_p_final = lr_p_final if lr_p_final is not None else lr_p / 100.0
        self.lr_cam_q_final = (
            lr_cam_q_final if lr_cam_q_final is not None else lr_cam_q / 10.0
        )
        self.lr_cam_t_final = (
            lr_cam_t_final if lr_cam_t_final is not None else lr_cam_t / 10.0
        )
        self.lr_cam_f_final = (
            lr_cam_f_final if lr_cam_f_final is not None else lr_cam_f / 10.0
        )
        self.lr_np_final = lr_np_final if lr_np_final is not None else lr_np / 100.0
        self.lr_nq_final = lr_nq_final if lr_nq_final is not None else lr_nq / 10.0
        if lr_w is not None:
            self.lr_w_final = lr_w_final if lr_w_final is not None else lr_w / 10.0
        else:
            self.lr_w_final = 0.0
        return

    def summary(self):
        logging.info("OptimCFG: Summary")
        logging.info(f"OptimCFG: lr_p={self.lr_p}")
        logging.info(f"OptimCFG: lr_q={self.lr_q}")
        logging.info(f"OptimCFG: lr_s={self.lr_s}")
        logging.info(f"OptimCFG: lr_o={self.lr_o}")
        logging.info(f"OptimCFG: lr_sph={self.lr_sph}")
        logging.info(f"OptimCFG: lr_sph_rest={self.lr_sph_rest}")
        logging.info(f"OptimCFG: lr_cam_q={self.lr_cam_q}")
        logging.info(f"OptimCFG: lr_cam_t={self.lr_cam_t}")
        logging.info(f"OptimCFG: lr_cam_f={self.lr_cam_f}")
        logging.info(f"OptimCFG: lr_p_final={self.lr_p_final}")
        logging.info(f"OptimCFG: lr_cam_q_final={self.lr_cam_q_final}")
        logging.info(f"OptimCFG: lr_cam_t_final={self.lr_cam_t_final}")
        logging.info(f"OptimCFG: lr_cam_f_final={self.lr_cam_f_final}")
        logging.info(f"OptimCFG: lr_np={self.lr_np}")
        logging.info(f"OptimCFG: lr_nq={self.lr_nq}")
        logging.info(f"OptimCFG: lr_w={self.lr_w}")
        logging.info(f"OptimCFG: lr_dyn={self.lr_dyn}")
        logging.info(f"OptimCFG: lr_nsig={self.lr_nsig}")
        logging.info(f"OptimCFG: lr_np_final={self.lr_np_final}")
        logging.info(f"OptimCFG: lr_nq_final={self.lr_nq_final}")
        logging.info(f"OptimCFG: lr_w_final={self.lr_w_final}")
        return

    @property
    def get_static_lr_dict(self):
        return {
            "lr_p": self.lr_p,
            "lr_q": self.lr_q,
            "lr_s": self.lr_s,
            "lr_o": self.lr_o,
            "lr_sph": self.lr_sph,
            "lr_sph_rest": self.lr_sph_rest,
        }

    @property
    def get_dynamic_lr_dict(self):
        return {
            "lr_p": self.lr_p,
            "lr_q": self.lr_q,
            "lr_s": self.lr_s,
            "lr_o": self.lr_o,
            "lr_sph": self.lr_sph,
            "lr_sph_rest": self.lr_sph_rest,
            # "lr_np": self.lr_np,
            # "lr_nq": self.lr_nq,
            # "lr_w": self.lr_w,
            # "lr_dyn": self.lr_dyn,
            # "lr_nsig": self.lr_nsig,
        }

    @property
    def get_cam_lr_dict(self):
        return {
            "lr_q": self.lr_cam_q,
            "lr_t": self.lr_cam_t,
            "lr_f": self.lr_cam_f,
        }

    def get_scheduler(self, total_steps):
        # todo: decide whether to decay skinning weights
        tone_mapper_scheduling = get_expon_lr_func(lr_init=5e-4, lr_final=5e-5, max_steps=total_steps)

        gs_scheduling_dict = {
            "xyz": get_expon_lr_func(
                lr_init=self.lr_p,
                lr_final=self.lr_p_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
            "xyz_cubic_node": get_expon_lr_func(
                lr_init=self.lr_p,
                lr_final=self.lr_p_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
            "rot": get_expon_lr_func(
                lr_init=self.lr_q,
                lr_final=self.lr_q*0.01,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
            "rot_poly_feat": get_expon_lr_func(
                lr_init=self.lr_q,
                lr_final=self.lr_q*0.01,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
            "node_rotation": get_expon_lr_func(
                lr_init=self.lr_nq,
                lr_final=self.lr_nq_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
        }
        cam_scheduling_dict = {
            "R": get_expon_lr_func(
                lr_init=self.lr_cam_q,
                lr_final=self.lr_cam_q_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
            "t": get_expon_lr_func(
                lr_init=self.lr_cam_t,
                lr_final=self.lr_cam_t_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
            "f": get_expon_lr_func(
                lr_init=self.lr_cam_f,
                lr_final=self.lr_cam_f_final,
                lr_delay_mult=0.01,  # 0.02
                max_steps=total_steps,
            ),
        }
        return gs_scheduling_dict, cam_scheduling_dict, tone_mapper_scheduling
    

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def get_expon_lr_func_interval(
    init_step, final_step, lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0
):
    def helper(step):
        if (
            step < init_step
            or step > final_step
            or (lr_init == 0.0 and lr_final == 0.0)
        ):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip((step - init_step) / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip((step - init_step) / (final_step - init_step), 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def update_learning_rate(lr, names, optimizer):
    if not isinstance(names, list):
        names = [names]
    for param_group in optimizer.param_groups:
        if param_group["name"] in names:
            param_group["lr"] = lr
            # print("debug")
            # print(f"Update {name} lr to {lr}")
    # ! have to iterate over all param_groups, because some param_groups may not have name
    return lr


@torch.no_grad()
def cat_tensors_to_optimizer(optimizer, tensors_dict, specified_dim_dict={}):
    for k in specified_dim_dict.keys():
        assert k in tensors_dict.keys(), f"{k} not in tensors_dict"
    optimizable_tensors = {}
    N = -1
    for group in optimizer.param_groups:
        if group["name"] not in tensors_dict.keys():
            # print(f"Warning: {group['name']} not in optimizer, skip")
            continue
        assert len(group["params"]) == 1, f"{group['name']} has more than one param"
        extension_tensor = tensors_dict[group["name"]]
        # print(group["name"])
        stored_state = optimizer.state.get(group["params"][0], None)
        working_dim = 0
        if group["name"] in specified_dim_dict.keys():
            working_dim = specified_dim_dict[group["name"]]
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat(
                (stored_state["exp_avg"].clone(), torch.zeros_like(extension_tensor)),
                dim=working_dim,
            )
            stored_state["exp_avg_sq"] = torch.cat(
                (
                    stored_state["exp_avg_sq"].clone(),
                    torch.zeros_like(extension_tensor),
                ),
                dim=working_dim,
            )

            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(
                torch.cat(
                    (
                        group["params"][0].clone().contiguous(),
                        extension_tensor.contiguous(),
                    ),
                    dim=working_dim,
                )
                .contiguous()
                .requires_grad_(True)
            )
            optimizable_tensors[group["name"]] = group["params"][0]
            optimizer.state[group["params"][0]] = stored_state
        else:
            group["params"][0] = nn.Parameter(
                torch.cat(
                    (
                        group["params"][0].clone().contiguous(),
                        extension_tensor.contiguous(),
                    ),
                    dim=working_dim,
                ).requires_grad_(True)
            )
            optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


@torch.no_grad()
def cat_tensors_to_optimizer_maintain_last(optimizer, tensors_dict):
    # ! specific for maintaining the ref frame at last, so will append at not hte last but the second last
    optimizable_tensors = {}
    N = -1
    for group in optimizer.param_groups:
        if group["name"] not in tensors_dict.keys():
            # print(f"Warning: {group['name']} not in optimizer, skip")
            continue
        assert len(group["params"]) == 1, f"{group['name']} has more than one param"
        extension_tensor = tensors_dict[group["name"]]
        # print(group["name"])
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat(
                (
                    stored_state["exp_avg"][:-1].clone(),
                    torch.zeros_like(extension_tensor),
                    stored_state["exp_avg"][-1:].clone(),
                ),
                dim=0,
            )
            stored_state["exp_avg_sq"] = torch.cat(
                (
                    stored_state["exp_avg_sq"][:-1].clone(),
                    torch.zeros_like(extension_tensor),
                    stored_state["exp_avg_sq"][-1:].clone(),
                ),
                dim=0,
            )

            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(
                torch.cat(
                    (
                        group["params"][0][:-1].clone().contiguous(),
                        extension_tensor.contiguous(),
                        group["params"][0][-1:].clone().contiguous(),
                    ),
                    dim=0,
                )
                .contiguous()
                .requires_grad_(True)
            )
            optimizable_tensors[group["name"]] = group["params"][0]
            optimizer.state[group["params"][0]] = stored_state
        else:
            group["params"][0] = nn.Parameter(
                torch.cat(
                    (
                        group["params"][0][:-1].clone().contiguous(),
                        extension_tensor.contiguous(),
                        group["params"][0][-1:].clone().contiguous(),
                    ),
                    dim=0,
                ).requires_grad_(True)
            )
            optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


def prune_optimizer(optimizer, mask, exclude_names=[], specific_names=[]):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        # print(group["name"])
        if group["name"] in exclude_names or len(group["params"]) == 0:
            continue
        if len(specific_names) > 0 and group["name"] not in specific_names:
            continue
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            if stored_state["exp_avg"].ndim == 3:
                stored_state["exp_avg"] = stored_state["exp_avg"][:, mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][:, mask]
            else:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

            del optimizer.state[group["params"][0]]
            if group["params"][0].ndim == 3:
                group["params"][0] = nn.Parameter(
                    (group["params"][0][:, mask].requires_grad_(True))
                )
            else:
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
            optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            if group["params"][0].ndim == 3:
                group["params"][0] = nn.Parameter(
                    group["params"][0][:, mask].requires_grad_(True)
                )
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
            optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


def replace_tensor_to_optimizer(optimizer, tensor, name):
    if not isinstance(tensor, list):
        tensor = [tensor]
    if not isinstance(name, list):
        name = [name]
    optimizable_tensors = {}
    for _tensor, _name in zip(tensor, name):
        for group in optimizer.param_groups:
            if group["name"] == _name:
                stored_state = optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(_tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(_tensor)
                    del optimizer.state[group["params"][0]]
                    optimizer.state[group["params"][0]] = stored_state
                group["params"][0] = nn.Parameter(_tensor.requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors