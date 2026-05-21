from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from .config import SUPPORTED_IMAGE_SUFFIXES


@dataclass
class ReferenceBatch:
    images: torch.Tensor
    masks: torch.Tensor
    paths: list[str]
    mask_sources: list[str]
    preprocess_records: list[dict[str, Any]]


class LazyMattingEngine:
    def __init__(self, weights_path: str | Path, device: torch.device | str):
        self.weights_path = Path(weights_path).expanduser()
        self.device = str(device)
        self._engine = None

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        if self._engine is None:
            if not self.weights_path.is_file():
                raise FileNotFoundError(
                    "Reference matting is required, but `inference.matting.weights` "
                    f"does not exist: {self.weights_path}"
                )
            from tools.human_matting import StyleMatteEngine as HumanMattingEngine

            self._engine = HumanMattingEngine(self.device, str(self.weights_path))

        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        alpha = self._engine(rgb_tensor, return_type="alpha")
        return alpha.detach().cpu().numpy()


class LazyHeadDetector:
    def __init__(
        self,
        weights_path: str | Path,
        device: torch.device | str,
        confidence_threshold: float,
    ):
        self.weights_path = Path(weights_path).expanduser()
        self.device = str(device)
        self.confidence_threshold = float(confidence_threshold)
        self._detector = None

    def detect(self, rgb: np.ndarray, image_key: str) -> np.ndarray | None:
        if self._detector is None:
            if not self.weights_path.is_file():
                raise FileNotFoundError(
                    "`inference.head_detection.weights` does not exist: "
                    f"{self.weights_path}"
                )
            from tools.vgghead_detector import VGGHeadDetector

            self._detector = VGGHeadDetector(
                device=self.device,
                vggheadmodel_path=str(self.weights_path),
            )

        rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        frame = torch.from_numpy(rgb_u8).permute(2, 0, 1)
        _, bbox, _ = self._detector(
            frame,
            image_key,
            conf_threshold=self.confidence_threshold,
        )
        if bbox is None:
            return None
        return bbox.detach().cpu().numpy().astype(np.float32)


def load_references(
    image_input: str | Path,
    *,
    source_size: int,
    patch_size: int,
    matting_weights: str | Path,
    head_detection_weights: str | Path,
    head_expand_scale: float,
    head_confidence_threshold: float,
    device: torch.device | str,
    background_color: float = 1.0,
) -> ReferenceBatch:
    image_paths = _resolve_reference_paths(image_input)
    matting = LazyMattingEngine(matting_weights, device)
    head_detector = LazyHeadDetector(
        head_detection_weights,
        device,
        confidence_threshold=head_confidence_threshold,
    )

    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    mask_sources: list[str] = []
    preprocess_records: list[dict[str, Any]] = []

    for image_path in image_paths:
        image, mask, mask_source, preprocess_record = _load_reference_image(
            image_path,
            matting,
            head_detector,
            head_expand_scale=head_expand_scale,
        )
        image, mask = preprocess_reference_image(
            image,
            mask,
            source_size=source_size,
            patch_size=patch_size,
            background_color=background_color,
        )
        images.append(image)
        masks.append(mask)
        mask_sources.append(mask_source)
        preprocess_records.append(preprocess_record)

    return ReferenceBatch(
        images=torch.cat(images, dim=0),
        masks=torch.cat(masks, dim=0),
        paths=[str(path) for path in image_paths],
        mask_sources=mask_sources,
        preprocess_records=preprocess_records,
    )


def infer_encoder_patch_size(encoder_type: str) -> int:
    if encoder_type == "dinov2_fusion":
        return 14
    if encoder_type == "dinov3_fusion":
        return 16
    raise ValueError(
        "`cfg.model.encoder_type` must be dinov2_fusion or dinov3_fusion for inference, "
        f"got: {encoder_type}"
    )


def tile_images_to_square(
    images: torch.Tensor,
    *,
    fill_color: float = 0.5,
    padding: int = 2,
) -> np.ndarray:
    if images.ndim != 4:
        raise ValueError(
            f"Expected reference images shaped [V, C, H, W], got: {tuple(images.shape)}"
        )

    num_images, channels, height, width = images.shape
    grid_size = int(np.ceil(np.sqrt(num_images)))
    final_h = grid_size * height + (grid_size + 1) * padding
    final_w = grid_size * width + (grid_size + 1) * padding
    grid = torch.full(
        (channels, final_h, final_w),
        fill_color,
        dtype=images.dtype,
        device=images.device,
    )

    for idx, image in enumerate(images):
        row = idx // grid_size
        col = idx % grid_size
        top = padding + row * (height + padding)
        left = padding + col * (width + padding)
        grid[:, top:top + height, left:left + width] = image

    return (grid.permute(1, 2, 0).detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def _resolve_reference_paths(image_input: str | Path) -> list[Path]:
    path = Path(image_input).expanduser()
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported reference image suffix: {path}")
        return [path]

    image_paths = [
        child for child in sorted(path.iterdir())
        if child.is_file() and child.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    if not image_paths:
        raise FileNotFoundError(
            f"`inference.image_input` directory contains no .png/.jpg/.jpeg files: {path}"
        )
    return image_paths


def _load_reference_image(
    image_path: Path,
    matting: LazyMattingEngine,
    head_detector: LazyHeadDetector,
    *,
    head_expand_scale: float,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any]]:
    image = Image.open(image_path)
    rgb = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0

    head_bbox = head_detector.detect(rgb, str(image_path))
    if head_bbox is None:
        warnings.warn(
            "Failed to detect a reference head; falling back to the full image: "
            f"{image_path}",
            stacklevel=2,
        )
        expanded_bbox = None
        crop_bbox = (0, 0, rgb.shape[1], rgb.shape[0])
        head_detection_status = "fallback_full_image"
    else:
        expanded_bbox = expand_bbox_xyxy(head_bbox, head_expand_scale)
        crop_bbox = clip_bbox_xyxy(expanded_bbox, width=rgb.shape[1], height=rgb.shape[0])
        head_detection_status = "ok"
    rgb = crop_to_bbox(rgb, crop_bbox)

    mask = matting(rgb)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask_source = "matting+head_crop"

    preprocess_record = {
        "path": str(image_path),
        "mask_source": mask_source,
        "head_detection_status": head_detection_status,
        "head_bbox_xyxy": _bbox_to_float_list(head_bbox) if head_bbox is not None else None,
        "expanded_bbox_xyxy": (
            _bbox_to_float_list(expanded_bbox) if expanded_bbox is not None else None
        ),
        "crop_bbox_xyxy": list(crop_bbox),
        "head_expand_scale": float(head_expand_scale),
        "head_confidence_threshold": float(head_detector.confidence_threshold),
    }
    return rgb, mask.astype(np.float32), mask_source, preprocess_record


def expand_bbox_xyxy(bbox: np.ndarray, scale: float) -> np.ndarray:
    bbox = np.asarray(bbox, dtype=np.float32)
    if bbox.shape != (4,):
        raise ValueError(f"Expected bbox shape [4], got: {bbox.shape}")
    if not np.isfinite(bbox).all():
        raise ValueError(f"Reference head bbox contains non-finite values: {bbox}")

    x_min, y_min, x_max, y_max = bbox
    width = max(float(x_max - x_min), 1.0)
    height = max(float(y_max - y_min), 1.0)
    side = max(width, height) * float(scale)
    center_x = float(x_min + x_max) * 0.5
    center_y = float(y_min + y_max) * 0.5
    half_side = side * 0.5
    return np.asarray(
        [
            center_x - half_side,
            center_y - half_side,
            center_x + half_side,
            center_y + half_side,
        ],
        dtype=np.float32,
    )


def clip_bbox_xyxy(bbox: np.ndarray, *, width: int, height: int) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got: width={width}, height={height}")

    x_min = max(0, int(np.floor(float(bbox[0]))))
    y_min = max(0, int(np.floor(float(bbox[1]))))
    x_max = min(width, int(np.ceil(float(bbox[2]))))
    y_max = min(height, int(np.ceil(float(bbox[3]))))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "Reference head crop is empty after clipping: "
            f"bbox={_bbox_to_float_list(bbox)}, image_size=({width}, {height})"
        )
    return x_min, y_min, x_max, y_max


def crop_to_bbox(array: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = bbox
    return array[y_min:y_max, x_min:x_max]


def _bbox_to_float_list(bbox: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(bbox, dtype=np.float32).tolist()]


def preprocess_reference_image(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    source_size: int,
    patch_size: int,
    background_color: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Reference RGB must have shape [H, W, 3], got: {rgb.shape}")
    if mask.ndim != 2:
        raise ValueError(f"Reference mask must have shape [H, W], got: {mask.shape}")

    mask = (mask > 0.7).astype(np.float32)
    if mask.mean() < 0.01:
        raise ValueError("Reference mask is near-empty after thresholding")

    rgb = rgb * mask[:, :, None] + background_color * (1.0 - mask[:, :, None])
    rgb, mask = pad_to_square(rgb, mask, background_color)
    rgb, mask = center_crop_according_to_mask(rgb, mask, aspect_standard=1.0)
    rgb, mask = resize_to_model_size(rgb, mask, source_size, patch_size)

    image_tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)
    return image_tensor, mask_tensor


def resize_to_model_size(
    rgb: np.ndarray,
    mask: np.ndarray,
    source_size: int,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_h = round(source_size / patch_size) * patch_size
    target_w = round(source_size / patch_size) * patch_size
    rgb = cv2.resize(rgb, dsize=(target_w, target_h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, dsize=(target_w, target_h), interpolation=cv2.INTER_AREA)
    return rgb, mask


def pad_to_square(
    rgb: np.ndarray,
    mask: np.ndarray,
    background_color: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = rgb.shape[:2]
    if height == width:
        return rgb, mask

    target_size = max(height, width)
    pad_top = (target_size - height) // 2
    pad_bottom = target_size - height - pad_top
    pad_left = (target_size - width) // 2
    pad_right = target_size - width - pad_left

    rgb = np.pad(
        rgb,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=background_color,
    )
    mask = np.pad(
        mask,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0.0,
    )
    return rgb, mask


def center_crop_according_to_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    aspect_standard: float,
) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) / max(mask.size, 1) < 0.01:
        raise ValueError("Reference mask is near-empty before crop")

    x_min, x_max = int(np.min(xs)), int(np.max(xs))
    y_min, y_max = int(np.min(ys)), int(np.max(ys))

    center_x = rgb.shape[1] // 2
    center_y = rgb.shape[0] // 2

    half_w = max(abs(center_x - x_min), abs(center_x - x_max), 1)
    half_h = max(abs(center_y - y_min), abs(center_y - y_max), 1)

    if half_h / half_w >= aspect_standard:
        half_w = round(half_h / aspect_standard)
    else:
        half_h = round(half_w * aspect_standard)

    half_h = min(max(half_h, 1), center_y)
    half_w = min(max(half_w, 1), center_x)

    top = center_y - half_h
    left = center_x - half_w
    return (
        rgb[top:top + 2 * half_h, left:left + 2 * half_w],
        mask[top:top + 2 * half_h, left:left + 2 * half_w],
    )
