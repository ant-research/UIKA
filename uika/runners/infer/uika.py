from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from .base_inferrer import Inferrer
from .checkpoint import CheckpointLoadReport, build_model_from_config, load_checkpoint_into_model
from .config import validate_inference_config
from .motion import MotionBatch, load_motion_sequence
from .output import build_debug_video_grid, save_inference_outputs
from .reference import (
    ReferenceBatch,
    infer_encoder_patch_size,
    load_references,
    tile_images_to_square,
)
from uika.runners import REGISTRY_RUNNERS
from uika.utils.config import parse_configs


@REGISTRY_RUNNERS.register("infer.uika")
class UIKAInferrer(Inferrer):
    def __init__(self):
        super().__init__()

        self.cfg = validate_inference_config(parse_configs())
        self.inf_cfg = self.cfg.inference

        torch._dynamo.config.disable = not bool(self.inf_cfg.compile)

        self.dtype = torch.float32
        self.model = build_model_from_config(
            self.cfg,
            skip_pretrained_weights=True,
        ).to(self.device)
        self.load_report = load_checkpoint_into_model(
            self.model,
            self.inf_cfg.checkpoint,
            strict=False,
            fail_on_shape_mismatch=True,
        )
        self.model.to(self.dtype)
        self.model.eval()

    def infer(self):
        patch_size = infer_encoder_patch_size(str(self.cfg.model.encoder_type))
        references = load_references(
            self.inf_cfg.image_input,
            source_size=int(self.inf_cfg.source_size),
            patch_size=patch_size,
            matting_weights=self.inf_cfg.matting.weights,
            head_detection_weights=self.inf_cfg.head_detection.weights,
            head_expand_scale=float(self.inf_cfg.head_detection.expand_scale),
            head_confidence_threshold=float(
                self.inf_cfg.head_detection.confidence_threshold
            ),
            device=self.device,
            background_color=1.0,
        )

        motion = load_motion_sequence(
            self.inf_cfg.motion_dir,
            shape_param_dim=int(self.cfg.model.shape_param_dim),
            render_size=int(self.inf_cfg.render_size),
            camera_path=str(self.inf_cfg.camera_path),
            orbit_cfg=self.inf_cfg.orbit,
            debug_cfg=self.inf_cfg.debug,
            teeth_bs_required=bool(self.cfg.model.get("teeth_bs_flag", False)),
            background_color=1.0,
        )

        rgb, mask = self._render_model(references, motion)
        ref_grid = tile_images_to_square(references.images) if (
            self.inf_cfg.debug.ref_grid or self.inf_cfg.debug.video_grid
        ) else None
        debug_video = None
        if self.inf_cfg.debug.video_grid:
            debug_video = build_debug_video_grid(
                rgb=rgb,
                ref_grid=ref_grid,
                driving_rgbs=motion.driving_rgbs,
                motion_rgbs=motion.motion_rgbs,
                blend_motion=bool(self.inf_cfg.debug.blend_motion),
            )

        metadata = self._build_metadata(references, motion)
        save_inference_outputs(
            output_dir=self.inf_cfg.output_dir,
            rgb=rgb,
            mask=mask,
            fps=int(self.inf_cfg.render_fps),
            save_frames=bool(self.inf_cfg.save_frames),
            save_video=bool(self.inf_cfg.save_video),
            metadata=metadata,
            debug_video=debug_video,
            debug_ref_grid=ref_grid if self.inf_cfg.debug.ref_grid else None,
        )
        print(f"inference outputs saved to: {Path(str(self.inf_cfg.output_dir)).expanduser()}")

    def _render_model(
        self,
        references: ReferenceBatch,
        motion: MotionBatch,
    ) -> tuple[np.ndarray, np.ndarray]:
        start_time = time.time()
        total_frames = motion.num_frames
        chunk_size = int(self.inf_cfg.render_chunk_size or 0)

        images = references.images.unsqueeze(0).to(self.device, self.dtype)
        masks = references.masks.unsqueeze(0).to(self.device, self.dtype)

        rgbs: list[torch.Tensor] = []
        masks_out: list[torch.Tensor] = []

        print("start inference ...")
        with torch.no_grad():
            for start, end in _iter_frame_chunks(total_frames, chunk_size):
                chunk_inputs = _slice_motion_inputs(
                    motion.model_inputs,
                    start=start,
                    end=end,
                    total_frames=total_frames,
                    device=self.device,
                )
                result = self.model(
                    images,
                    masks,
                    render_c2ws=chunk_inputs["render_c2ws"],
                    render_intrs=chunk_inputs["render_intrs"],
                    render_bg_colors=chunk_inputs["render_bg_colors"],
                    flame_params=chunk_inputs["flame_params"],
                    render_h=int(self.inf_cfg.render_size),
                    render_w=int(self.inf_cfg.render_size),
                )
                rgbs.append(result["comp_rgb"].detach().cpu())
                masks_out.append(result["comp_mask"].detach().cpu())

        rgb_tensor = torch.cat(rgbs, dim=1)[0]
        mask_tensor = torch.cat(masks_out, dim=1)[0]
        rgb = rgb_tensor.permute(0, 2, 3, 1).numpy()
        mask = mask_tensor.permute(0, 2, 3, 1).numpy()

        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        mask = ((mask > 0.7) * 255).astype(np.uint8)
        print(f"time elapsed: {round(time.time() - start_time, 2)}s")
        return rgb, mask

    def _build_metadata(
        self,
        references: ReferenceBatch,
        motion: MotionBatch,
    ) -> dict[str, Any]:
        metadata = {
            "config": str(Path(str(self.cfg.get("_config_path", ""))).expanduser()),
            "checkpoint": str(Path(str(self.inf_cfg.checkpoint)).expanduser()),
            "image_input": str(Path(str(self.inf_cfg.image_input)).expanduser()),
            "motion_dir": str(Path(str(self.inf_cfg.motion_dir)).expanduser()),
            "output_dir": str(Path(str(self.inf_cfg.output_dir)).expanduser()),
            "num_reference_images": len(references.paths),
            "reference_images": references.paths,
            "reference_mask_sources": references.mask_sources,
            "reference_preprocess": references.preprocess_records,
            "num_frames": motion.num_frames,
            "source_size": int(self.inf_cfg.source_size),
            "render_size": int(self.inf_cfg.render_size),
            "render_fps": int(self.inf_cfg.render_fps),
            "resolved_dtype": str(self.dtype).replace("torch.", ""),
            "camera_path": str(self.inf_cfg.camera_path),
            "save_video": bool(self.inf_cfg.save_video),
            "save_frames": bool(self.inf_cfg.save_frames),
            "debug": OmegaConf.to_container(self.inf_cfg.debug, resolve=True),
            "checkpoint_load": _checkpoint_report_counts(self.load_report),
        }
        if self.inf_cfg.camera_path == "orbit":
            metadata["orbit"] = OmegaConf.to_container(self.inf_cfg.orbit, resolve=True)
        return metadata

def _iter_frame_chunks(total_frames: int, chunk_size: int):
    if chunk_size <= 0 or chunk_size >= total_frames:
        yield 0, total_frames
        return
    for start in range(0, total_frames, chunk_size):
        yield start, min(start + chunk_size, total_frames)


def _slice_motion_inputs(
    model_inputs: dict[str, Any],
    *,
    start: int,
    end: int,
    total_frames: int,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "render_c2ws": model_inputs["render_c2ws"][:, start:end].to(device),
        "render_intrs": model_inputs["render_intrs"][:, start:end].to(device),
        "render_bg_colors": model_inputs["render_bg_colors"][:, start:end].to(device),
        "flame_params": {
            key: _slice_flame_param(
                value,
                start=start,
                end=end,
                total_frames=total_frames,
            ).to(device)
            for key, value in model_inputs["flame_params"].items()
        },
    }


def _slice_flame_param(
    value: torch.Tensor,
    *,
    start: int,
    end: int,
    total_frames: int,
) -> torch.Tensor:
    if value.ndim >= 3 and value.shape[1] == total_frames:
        return value[:, start:end]
    return value


def _checkpoint_report_counts(report: CheckpointLoadReport) -> dict[str, int]:
    return {
        "loaded": len(report.loaded),
        "missing": len(report.missing),
        "unexpected": len(report.unexpected),
        "mismatched": len(report.mismatched),
    }
