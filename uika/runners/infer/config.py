from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


DEPRECATED_TOP_LEVEL_FIELDS = {
    "model_name": "inference.checkpoint",
    "image_input": "inference.image_input",
    "motion_seqs_dir": "inference.motion_dir",
    "video_dump": "inference.output_dir",
    "image_dump": "inference.output_dir",
    "vis_rgb": "inference.debug.include_driving_rgb",
    "vis_motion": "inference.debug.vis_motion",
    "vis_blend_res": "inference.debug.blend_motion",
    "vis_ref": "inference.debug.ref_grid / inference.debug.video_grid",
    "save_img": "inference.save_frames",
    "export_video": "inference.save_video",
    "save_img_type": "removed",
    "cycle_sample_mv_cam": "inference.camera_path=orbit",
    "infer_mode": "removed",
    "camera_index": "removed",
    "camera_stride": "removed",
}

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SUPPORTED_CAMERA_PATHS = {"motion", "orbit"}
SUPPORTED_ORBIT_AXES = {"x", "y", "z"}
REMOVED_INFERENCE_FIELDS = {
    "dtype": "`inference.dtype` has been removed; UIKA inference currently runs in fp32 only.",
}


def validate_inference_config(cfg: DictConfig) -> DictConfig:
    if "inference" not in cfg:
        raise ValueError(
            "Missing required config section `inference`. "
            "Use `--config configs/infer_uika.yaml` or add an `inference:` block."
        )

    _validate_cli_overrides(cfg)
    _reject_deprecated_top_level_fields(cfg)
    inf = cfg.inference
    _reject_removed_inference_fields(inf)

    checkpoint = _required_path(inf, "checkpoint")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"`inference.checkpoint` must be an existing checkpoint file, got: {checkpoint}"
        )
    if checkpoint.suffix.lower() not in {".safetensors", ".pt", ".pth"}:
        raise ValueError(
            "`inference.checkpoint` must end with .safetensors, .pt, or .pth, "
            f"got: {checkpoint}"
        )

    image_input = _required_path(inf, "image_input")
    if not image_input.exists():
        raise FileNotFoundError(f"`inference.image_input` does not exist: {image_input}")
    if image_input.is_file() and image_input.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(
            "`inference.image_input` file must be .png, .jpg, or .jpeg, "
            f"got: {image_input}"
        )
    if not image_input.is_file() and not image_input.is_dir():
        raise ValueError(
            "`inference.image_input` must be a file or directory, "
            f"got: {image_input}"
        )

    motion_dir = _required_path(inf, "motion_dir")
    if not motion_dir.is_dir():
        raise FileNotFoundError(
            f"`inference.motion_dir` must be an existing directory, got: {motion_dir}"
        )
    transforms_path = motion_dir / "transforms.json"
    if not transforms_path.is_file():
        raise FileNotFoundError(
            "`inference.motion_dir` must contain `transforms.json`, "
            f"missing: {transforms_path}"
        )

    output_dir = _required_path(inf, "output_dir")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(
            f"`inference.output_dir` must be a directory path, got existing file: {output_dir}"
        )

    _validate_positive_int(inf, "source_size")
    _validate_positive_int(inf, "render_size")
    _validate_positive_int(inf, "render_fps")
    render_chunk_size = int(inf.get("render_chunk_size", 0) or 0)
    if render_chunk_size < 0:
        raise ValueError(
            "`inference.render_chunk_size` must be >= 0, "
            f"got: {inf.render_chunk_size}"
        )
    inf.render_chunk_size = render_chunk_size

    camera_path = str(inf.get("camera_path", "motion"))
    if camera_path not in SUPPORTED_CAMERA_PATHS:
        raise ValueError(
            "`inference.camera_path` must be one of motion, orbit, "
            f"got: {camera_path}"
        )
    inf.camera_path = camera_path

    _validate_bool(inf, "compile")
    _validate_bool(inf, "save_video")
    _validate_bool(inf, "save_frames")
    _validate_debug_config(inf)
    _validate_matting_config(inf)
    _validate_head_detection_config(inf)
    if camera_path == "orbit":
        _validate_orbit_config(inf)

    return cfg


def _reject_deprecated_top_level_fields(cfg: DictConfig) -> None:
    present = [field for field in DEPRECATED_TOP_LEVEL_FIELDS if field in cfg]
    if not present:
        return

    migrations = [
        f"{field} -> {DEPRECATED_TOP_LEVEL_FIELDS[field]}"
        for field in sorted(present)
    ]
    raise ValueError(
        "Deprecated top-level inference fields are not supported. "
        "Move settings under `inference.*`: " + "; ".join(migrations)
    )


def _reject_removed_inference_fields(inf: DictConfig) -> None:
    present = [field for field in REMOVED_INFERENCE_FIELDS if field in inf]
    if not present:
        return

    messages = [REMOVED_INFERENCE_FIELDS[field] for field in sorted(present)]
    raise ValueError(" ".join(messages))


def _validate_cli_overrides(cfg: DictConfig) -> None:
    overrides = cfg.get("_cli_overrides", [])
    invalid: list[str] = []
    for override in overrides:
        key = str(override).split("=", 1)[0]
        if not key.startswith("inference."):
            invalid.append(str(override))
    if invalid:
        raise ValueError(
            "Inference CLI overrides must target `inference.*` fields only. "
            f"Invalid overrides: {', '.join(invalid)}"
        )


def _required_path(inf: DictConfig, key: str) -> Path:
    value = inf.get(key)
    if value in (None, ""):
        raise ValueError(f"`inference.{key}` is required")
    return Path(str(value)).expanduser()


def _validate_positive_int(inf: DictConfig, key: str) -> None:
    value = inf.get(key)
    if value is None:
        raise ValueError(f"`inference.{key}` is required")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`inference.{key}` must be a positive integer, got: {value}") from exc
    if parsed <= 0:
        raise ValueError(f"`inference.{key}` must be > 0, got: {value}")
    inf[key] = parsed


def _validate_bool(container: DictConfig, key: str) -> None:
    value = container.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"`inference.{key}` must be true or false, got: {value}")


def _validate_debug_config(inf: DictConfig) -> None:
    if "debug" not in inf or inf.debug is None:
        inf.debug = OmegaConf.create({})
    for key in ("ref_grid", "video_grid", "include_driving_rgb", "vis_motion", "blend_motion"):
        if key not in inf.debug:
            inf.debug[key] = False
        if not isinstance(inf.debug[key], bool):
            raise ValueError(f"`inference.debug.{key}` must be true or false")


def _validate_matting_config(inf: DictConfig) -> None:
    if "matting" not in inf or inf.matting is None:
        inf.matting = OmegaConf.create({})
    if "weights" not in inf.matting or inf.matting.weights in (None, ""):
        inf.matting.weights = "./model_zoo/tools/stylematte_synth.pt"


def _validate_head_detection_config(inf: DictConfig) -> None:
    if "head_detection" not in inf or inf.head_detection is None:
        inf.head_detection = OmegaConf.create({})

    head_detection = inf.head_detection
    if "weights" not in head_detection or head_detection.weights in (None, ""):
        head_detection.weights = "./model_zoo/tools/vgg_heads_l.trcd"
    weights = Path(str(head_detection.weights)).expanduser()
    if not weights.is_file():
        raise FileNotFoundError(
            "`inference.head_detection.weights` must be an existing VGGHeadDetector "
            f"checkpoint file, got: {weights}"
        )
    head_detection.weights = str(weights)

    expand_scale = _as_float(
        head_detection.get("expand_scale", 1.5),
        "inference.head_detection.expand_scale",
    )
    if expand_scale <= 1.0:
        raise ValueError(
            "`inference.head_detection.expand_scale` must be > 1.0, "
            f"got: {expand_scale}"
        )
    head_detection.expand_scale = expand_scale

    confidence_threshold = _as_float(
        head_detection.get("confidence_threshold", 0.5),
        "inference.head_detection.confidence_threshold",
    )
    if not 0.0 < confidence_threshold < 1.0:
        raise ValueError(
            "`inference.head_detection.confidence_threshold` must be in (0, 1), "
            f"got: {confidence_threshold}"
        )
    head_detection.confidence_threshold = confidence_threshold


def _validate_orbit_config(inf: DictConfig) -> None:
    if "orbit" not in inf:
        raise ValueError("`inference.orbit` is required when `inference.camera_path=orbit`")

    orbit = inf.orbit
    for key in ("radius_x", "radius_y"):
        value = _as_float(orbit.get(key), f"inference.orbit.{key}")
        if value <= 0:
            raise ValueError(f"`inference.orbit.{key}` must be > 0, got: {value}")
        orbit[key] = value

    for key in ("center", "look_at"):
        value = orbit.get(key)
        if not _is_number_triplet(value):
            raise ValueError(
                f"`inference.orbit.{key}` must be a list of three numbers, got: {value}"
            )
        orbit[key] = [float(item) for item in value]

    axis = str(orbit.get("axis", "z"))
    up = str(orbit.get("up", "y"))
    if axis not in SUPPORTED_ORBIT_AXES:
        raise ValueError("`inference.orbit.axis` must be one of x, y, z")
    if up not in SUPPORTED_ORBIT_AXES:
        raise ValueError("`inference.orbit.up` must be one of x, y, z")
    if axis == up:
        raise ValueError("`inference.orbit.axis` and `inference.orbit.up` must differ")
    orbit.axis = axis
    orbit.up = up


def _as_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{field}` must be a number, got: {value}") from exc


def _is_number_triplet(value: Any) -> bool:
    try:
        if value is None or len(value) != 3:
            return False
    except TypeError:
        return False
    try:
        [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return True
