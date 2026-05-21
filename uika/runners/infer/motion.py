from __future__ import annotations

import json
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


REQUIRED_FRAME_FIELDS = {
    "transform_matrix",
    "fl_x",
    "fl_y",
    "cx",
    "cy",
    "flame_param_path",
}

REQUIRED_FLAME_FIELDS = {
    "expr",
    "rotation",
    "neck_pose",
    "jaw_pose",
    "eyes_pose",
    "translation",
}


@dataclass
class MotionBatch:
    model_inputs: dict[str, Any]
    num_frames: int
    driving_rgbs: np.ndarray | None
    motion_rgbs: np.ndarray | None


def load_motion_sequence(
    motion_dir: str | Path,
    *,
    shape_param_dim: int,
    render_size: int,
    camera_path: str,
    orbit_cfg: Any,
    debug_cfg: Any,
    teeth_bs_required: bool = False,
    background_color: float = 1.0,
) -> MotionBatch:
    motion_dir = Path(motion_dir).expanduser()
    data = _read_transforms(motion_dir)
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"`{motion_dir / 'transforms.json'}` must contain a non-empty `frames` list")

    teeth_bs_values = _load_teeth_bs(motion_dir, required=teeth_bs_required)
    c2ws: list[torch.Tensor] = []
    intrs: list[torch.Tensor] = []
    flame_params: list[dict[str, torch.Tensor]] = []
    driving_rgbs: list[np.ndarray] = []

    include_driving_rgb = bool(debug_cfg.get("include_driving_rgb", False))

    print("loading motion sequence ...")
    for frame_idx, frame_info in enumerate(tqdm(frames, total=len(frames))):
        _validate_frame(frame_info, frame_idx)
        flame_path = _resolve_motion_file(motion_dir, frame_info["flame_param_path"])
        teeth_bs = (
            teeth_bs_values[frame_idx]
            if teeth_bs_values is not None and frame_idx < len(teeth_bs_values)
            else None
        )

        flame_params.append(load_flame_params(flame_path, teeth_bs=teeth_bs))
        c2w, intrinsic = load_camera_pose(frame_info)
        c2ws.append(c2w)
        intrs.append(intrinsic)

        if include_driving_rgb:
            driving_rgbs.append(_load_driving_rgb(frame_info, motion_dir, render_size))

    print("motion sequence loaded")
    c2ws_tensor = torch.stack(c2ws, dim=0)
    intrs_tensor = torch.stack(intrs, dim=0)
    flame_tensors = _stack_flame_params(flame_params)

    if camera_path == "orbit":
        c2ws_tensor = build_orbit_cameras(
            num_frames=len(frames),
            orbit_cfg=orbit_cfg,
        )
        flame_tensors["rotation"] = torch.zeros_like(flame_tensors["rotation"])
        flame_tensors["translation"] = torch.zeros_like(flame_tensors["translation"])

    flame_tensors["betas"] = torch.zeros(shape_param_dim, dtype=torch.float32)

    motion_rgbs = None
    if bool(debug_cfg.get("video_grid", False)) and bool(debug_cfg.get("vis_motion", False)):
        motion_rgbs = _try_render_motion_mesh(
            flame_tensors,
            intrs_tensor,
            c2ws_tensor,
            render_size=render_size,
        )

    for key, value in list(flame_tensors.items()):
        flame_tensors[key] = value.unsqueeze(0)

    bg_colors = torch.full((len(frames), 3), float(background_color), dtype=torch.float32)
    model_inputs = {
        "render_c2ws": c2ws_tensor.unsqueeze(0),
        "render_intrs": intrs_tensor.unsqueeze(0),
        "render_bg_colors": bg_colors.unsqueeze(0),
        "flame_params": flame_tensors,
    }

    return MotionBatch(
        model_inputs=model_inputs,
        num_frames=len(frames),
        driving_rgbs=np.stack(driving_rgbs, axis=0) if driving_rgbs else None,
        motion_rgbs=motion_rgbs,
    )


def load_camera_pose(frame_info: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    c2w = np.array(frame_info["transform_matrix"], dtype=np.float32)
    if c2w.shape != (4, 4):
        raise ValueError(f"`transform_matrix` must have shape [4, 4], got: {c2w.shape}")
    c2w[:3, 1:3] *= -1

    intrinsic = torch.eye(4, dtype=torch.float32)
    intrinsic[0, 0] = float(frame_info["fl_x"])
    intrinsic[1, 1] = float(frame_info["fl_y"])
    intrinsic[0, 2] = float(frame_info["cx"])
    intrinsic[1, 2] = float(frame_info["cy"])
    return torch.from_numpy(c2w), intrinsic


def load_flame_params(flame_path: str | Path, *, teeth_bs: np.ndarray | None = None) -> dict[str, torch.Tensor]:
    flame_path = Path(flame_path).expanduser()
    if not flame_path.is_file():
        raise FileNotFoundError(f"FLAME parameter file not found: {flame_path}")

    flame_data = dict(np.load(flame_path, allow_pickle=True))
    missing = sorted(REQUIRED_FLAME_FIELDS - set(flame_data.keys()))
    if missing:
        raise ValueError(f"FLAME file missing required keys {missing}: {flame_path}")

    flame_param_tensor = {
        key: _to_1d_tensor(flame_data[key], key, flame_path)
        for key in REQUIRED_FLAME_FIELDS
    }
    if teeth_bs is not None:
        flame_param_tensor["teeth_bs"] = torch.as_tensor(teeth_bs, dtype=torch.float32)
    return flame_param_tensor


def build_orbit_cameras(num_frames: int, orbit_cfg: Any) -> torch.Tensor:
    from dreifus.vector import Vec3

    from uika.utils.interpolate_camera import ellipse_around_axis

    poses = ellipse_around_axis(
        n_poses=num_frames,
        a_len=float(orbit_cfg.radius_x),
        b_len=float(orbit_cfg.radius_y),
        axis=_axis_to_vec3(orbit_cfg.axis, Vec3),
        up=_axis_to_vec3(orbit_cfg.up, Vec3),
        move=Vec3(*[float(value) for value in orbit_cfg.center]),
        look_at=Vec3(*[float(value) for value in orbit_cfg.look_at]),
    )
    return torch.stack([torch.from_numpy(pose.numpy()).float() for pose in poses], dim=0)


def _read_transforms(motion_dir: Path) -> dict[str, Any]:
    transforms_path = motion_dir / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Missing motion transforms file: {transforms_path}")
    with transforms_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_frame(frame_info: dict[str, Any], frame_idx: int) -> None:
    missing = sorted(REQUIRED_FRAME_FIELDS - set(frame_info.keys()))
    if missing:
        raise ValueError(f"`frames[{frame_idx}]` missing required fields: {missing}")


def _resolve_motion_file(motion_dir: Path, file_path: str) -> Path:
    path = Path(file_path).expanduser()
    if path.is_absolute():
        return path
    return motion_dir / path


def _load_teeth_bs(motion_dir: Path, *, required: bool) -> np.ndarray | None:
    path = motion_dir / "tracked_teeth_bs.npz"
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                "`cfg.model.teeth_bs_flag=true` requires `tracked_teeth_bs.npz` "
                f"in `inference.motion_dir`, missing: {path}"
            )
        return None
    data = np.load(path)
    if "expr_teeth" not in data:
        if required:
            raise ValueError(f"`tracked_teeth_bs.npz` must contain `expr_teeth`: {path}")
        warnings.warn(f"Ignoring {path}; missing `expr_teeth` key", stacklevel=2)
        return None
    return data["expr_teeth"]


def _to_1d_tensor(value: np.ndarray, key: str, flame_path: Path) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 0:
        raise ValueError(f"`{key}` in {flame_path} must not be scalar")
    if tensor.ndim > 1 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 1:
        raise ValueError(
            f"`{key}` in {flame_path} must describe one motion frame as a 1D vector, "
            f"got shape {tuple(tensor.shape)}"
        )
    return tensor


def _stack_flame_params(flame_params: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    stacked: dict[str, list[torch.Tensor]] = defaultdict(list)
    for flame_param in flame_params:
        for key, value in flame_param.items():
            stacked[key].append(value)
    return {key: torch.stack(values, dim=0) for key, values in stacked.items()}


def _axis_to_vec3(axis: str, vec3_cls):
    if axis == "x":
        return vec3_cls(1, 0, 0)
    if axis == "y":
        return vec3_cls(0, 1, 0)
    if axis == "z":
        return vec3_cls(0, 0, 1)
    raise ValueError(f"Unsupported orbit axis: {axis}")


def _load_driving_rgb(frame_info: dict[str, Any], motion_dir: Path, render_size: int) -> np.ndarray:
    if "file_path" not in frame_info or not frame_info["file_path"]:
        warnings.warn("Motion frame has no `file_path`; using gray debug driving RGB", stacklevel=2)
        return _gray_rgb(render_size)

    image_path = _resolve_motion_file(motion_dir, frame_info["file_path"])
    if not image_path.is_file():
        warnings.warn(f"Driving RGB missing; using gray placeholder: {image_path}", stacklevel=2)
        return _gray_rgb(render_size)

    try:
        image = np.asarray(Image.open(image_path).convert("RGB")).astype(np.uint8)
        image = _center_crop_square(image)
        return cv2.resize(image, (render_size, render_size), interpolation=cv2.INTER_AREA)
    except Exception as exc:  # noqa: BLE001 - debug output must not block public rendering.
        warnings.warn(
            f"Failed to read driving RGB; using gray placeholder: {image_path}. Error: {exc}",
            stacklevel=2,
        )
        return _gray_rgb(render_size)


def _center_crop_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return image[top:top + side, left:left + side]


def _gray_rgb(render_size: int) -> np.ndarray:
    return np.full((render_size, render_size, 3), 127, dtype=np.uint8)


def _try_render_motion_mesh(
    flame_tensors: dict[str, torch.Tensor],
    intrs: torch.Tensor,
    c2ws: torch.Tensor,
    *,
    render_size: int,
) -> np.ndarray:
    try:
        return _render_motion_mesh(flame_tensors, intrs, c2ws)
    except Exception as exc:  # noqa: BLE001 - debug output must not block public rendering.
        warnings.warn(
            f"Failed to render debug motion mesh; using gray placeholders. Error: {exc}",
            stacklevel=2,
        )
        return np.stack([_gray_rgb(render_size) for _ in range(c2ws.shape[0])], axis=0)


def _render_motion_mesh(
    flame_tensors: dict[str, torch.Tensor],
    intrs: torch.Tensor,
    c2ws: torch.Tensor,
) -> np.ndarray:
    from uika.models.rendering.flame_model.flame_subdivide import FlameHeadSubdivided
    from uika.models.rendering.utils.vis_utils import render_mesh

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flame_sub_model = FlameHeadSubdivided(
        300,
        100,
        add_teeth=True,
        add_shoulder=False,
        flame_model_path="model_zoo/human_parametric_models/flame2023.pkl",
        flame_lmk_embedding_path="model_zoo/human_parametric_models/landmark_embedding_with_eyes.npy",
        flame_template_mesh_path="model_zoo/human_parametric_models/head_template_mesh.obj",
        flame_parts_path="model_zoo/human_parametric_models/FLAME_masks.pkl",
        subdivide_num=0,
    ).to(device)

    betas = flame_tensors["betas"].to(device)
    v_cano = flame_sub_model.get_cano_verts(betas.unsqueeze(0))
    num_frames = flame_tensors["expr"].shape[0]
    ret = flame_sub_model.animation_forward(
        v_cano.repeat(num_frames, 1, 1),
        betas.unsqueeze(0).repeat(num_frames, 1),
        flame_tensors["expr"].to(device),
        flame_tensors["rotation"].to(device),
        flame_tensors["neck_pose"].to(device),
        flame_tensors["jaw_pose"].to(device),
        flame_tensors["eyes_pose"].to(device),
        flame_tensors["translation"].to(device),
        zero_centered_at_root_node=False,
        return_landmarks=False,
        return_verts_cano=True,
        static_offset=None,
    )

    flame_face = flame_sub_model.faces.cpu().squeeze().numpy()
    mesh_render_list = []
    for frame_idx in range(num_frames):
        intr = intrs[frame_idx]
        cam_param = {
            "focal": torch.tensor([intr[0, 0], intr[1, 1]]),
            "princpt": torch.tensor([intr[0, 2], intr[1, 2]]),
        }
        render_shape = int(cam_param["princpt"][1] * 2), int(cam_param["princpt"][0] * 2)

        vertices = ret["animated"][frame_idx].cpu().squeeze()
        w2c = torch.inverse(c2ws[frame_idx])
        # render_mesh expects camera-space vertices; apply the row-vector
        # equivalent of x_cam = w2c @ x_world.
        vertices = vertices @ w2c[:3, :3].T + w2c[:3, 3]
        mesh_render, _ = render_mesh(
            vertices,
            flame_face,
            cam_param,
            np.ones((render_shape[0], render_shape[1], 3), dtype=np.float32) * 255,
            return_bg_mask=True,
        )
        mesh_render_list.append(mesh_render.astype(np.uint8))

    return np.stack(mesh_render_list, axis=0)
