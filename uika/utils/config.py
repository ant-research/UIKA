import os
import argparse
from omegaconf import OmegaConf


def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args, unknown = parser.parse_known_args()
    cli_overrides = [item for item in unknown if '=' in item or item.startswith('-')]

    if args.config is None:
        raise ValueError("`--config` is required")
    cfg = OmegaConf.load(args.config)
    if 'base_config' in cfg:
        base_cfg = OmegaConf.load(cfg.base_config)
        del cfg['base_config']
        cfg = OmegaConf.merge(base_cfg, cfg)

    is_inference_config = 'inference' in cfg
    cli_cfg = OmegaConf.from_cli(cli_overrides)
    if not is_inference_config and os.environ.get('APP_MODEL_NAME') is not None:
        cli_cfg.model_name = os.environ.get('APP_MODEL_NAME')
    cfg = OmegaConf.merge(cfg, cli_cfg)

    if 'inference' in cfg:
        cfg._config_path = args.config
        cfg._cli_overrides = list(cli_overrides)
        return cfg

    if hasattr(cfg, 'model_name'):
        step = cfg.model_name.split('_')[-1]
        assert 'k' in step, f"step {step} must end with `k`"
        step = f'_{step}'
    else:
        step = ''
    
    # hard code
    cfg.source_size = cfg.data.dataset.source_image_res
    cfg.render_size = cfg.data.dataset.render_image.high
    _relative_path = os.path.join(cfg.experiment.parent, cfg.experiment.child + step)
    
    cfg.save_tmp_dump = os.path.join('dumps', 'save_tmp', _relative_path)
    cfg.image_dump = os.path.join('dumps', 'images', _relative_path)
    cfg.video_dump = os.path.join('dumps', 'videos', _relative_path)
    cfg.mesh_dump = os.path.join('dumps', 'meshes', _relative_path)
    cfg.blender_path = 'blender'
    cfg.motion_video_read_fps = 30

    return cfg
