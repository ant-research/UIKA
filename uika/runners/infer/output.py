from __future__ import annotations

import os
import json
import torch
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def images_to_video(images, output_path, fps, gradio_codec: bool, verbose=False):
    import imageio
    # images: torch.tensor (T, C, H, W), 0-1  or numpy: (T, H, W, 3) 0-255
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = []
    for i in range(images.shape[0]):
        if isinstance(images, torch.Tensor):
            frame = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            assert frame.shape[0] == images.shape[2] and frame.shape[1] == images.shape[3], \
                f"Frame shape mismatch: {frame.shape} vs {images.shape}"
            assert frame.min() >= 0 and frame.max() <= 255, \
                f"Frame value out of range: {frame.min()} ~ {frame.max()}"
        else:
            frame = images[i]
        frames.append(frame)
    frames = np.stack(frames)
    if gradio_codec:
        imageio.mimwrite(output_path, frames, fps=fps, quality=10)
    else:
        # imageio.mimwrite(output_path, frames, fps=fps, codec='mpeg4', quality=10)
        imageio.mimwrite(output_path, frames, fps=fps, quality=10)

    if verbose:
        print(f"Using gradio codec option {gradio_codec}")
        print(f"Saved video to {output_path}")


def save_inference_outputs(
    *,
    output_dir: str | Path,
    rgb: np.ndarray,
    mask: np.ndarray,
    fps: int,
    save_frames: bool,
    save_video: bool,
    metadata: dict[str, Any],
    debug_video: np.ndarray | None = None,
    debug_ref_grid: np.ndarray | None = None,
) -> None:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_frames:
        _save_rgba_frames(output_dir / "frames", rgb, mask)

    if save_video:
        images_to_video(rgb, str(output_dir / "video.mp4"), fps, gradio_codec=False)

    if debug_ref_grid is not None:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(debug_ref_grid).save(debug_dir / "ref_grid.png")

    if debug_video is not None:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        images_to_video(debug_video, str(debug_dir / "video_grid.mp4"), fps, gradio_codec=False)

    _save_metadata(output_dir / "metadata.json", metadata)


def build_debug_video_grid(
    *,
    rgb: np.ndarray,
    ref_grid: np.ndarray | None,
    driving_rgbs: np.ndarray | None,
    motion_rgbs: np.ndarray | None,
    blend_motion: bool,
) -> np.ndarray:
    clips: list[np.ndarray] = []
    if ref_grid is not None:
        ref_grid = cv2.resize(ref_grid, (rgb.shape[2], rgb.shape[1]), interpolation=cv2.INTER_AREA)
        clips.append(np.tile(ref_grid[None], (rgb.shape[0], 1, 1, 1)))

    if driving_rgbs is not None:
        clips.append(_ensure_video_size(driving_rgbs, rgb.shape[1], rgb.shape[2]))

    clips.append(rgb)

    if motion_rgbs is not None:
        motion_rgbs = _ensure_video_size(motion_rgbs, rgb.shape[1], rgb.shape[2])
        if blend_motion:
            clips.append(((0.3 * rgb + 0.7 * motion_rgbs).clip(0, 255)).astype(np.uint8))
        clips.append(motion_rgbs)

    return np.concatenate(clips, axis=2)


def _save_rgba_frames(frame_dir: Path, rgb: np.ndarray, mask: np.ndarray) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in range(rgb.shape[0]):
        rgba = np.concatenate([rgb[frame_idx], mask[frame_idx]], axis=2)
        Image.fromarray(rgba).save(frame_dir / f"{frame_idx:04d}.png")


def _save_metadata(path: Path, metadata: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _ensure_video_size(video: np.ndarray, height: int, width: int) -> np.ndarray:
    if video.shape[1] == height and video.shape[2] == width:
        return video
    return np.stack(
        [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in video],
        axis=0,
    )
