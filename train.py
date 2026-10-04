import os
from omegaconf import OmegaConf
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("MoSca-V2 Preprocessing")
    parser.add_argument("--data_path", type=str, help="Source folder", required=True)
    parser.add_argument("--cfg", type=str, help="profile yaml file path", required=True)
    parser.add_argument("--gpu", type=str, default='0', help="which gpu to use")
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

    from trainer_cam import CamGSTrainer
    from trainer_world import WorldGSTrainer

    cam_trainer = CamGSTrainer(cfg)
    cam_trainer.train()
    world_trainer = WorldGSTrainer(cfg)
    world_trainer.train()
   
