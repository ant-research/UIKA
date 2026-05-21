from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToTensor


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def collect_frame_pairs(pred_dir: str | Path, gt_dir: str | Path) -> list[tuple[Path, Path]]:
    pred_dir = Path(pred_dir).expanduser()
    gt_dir = Path(gt_dir).expanduser()
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"`pred_dir` does not exist or is not a directory: {pred_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"`gt_dir` does not exist or is not a directory: {gt_dir}")

    pred_paths = list_image_files(pred_dir)
    gt_paths = list_image_files(gt_dir)
    if len(pred_paths) != len(gt_paths):
        raise ValueError(
            "`pred_dir` and `gt_dir` must contain the same number of image files, "
            f"got {len(pred_paths)} and {len(gt_paths)}"
        )
    if not pred_paths:
        raise ValueError(f"No image files found in `{pred_dir}`")
    return list(zip(pred_paths, gt_paths))


def list_image_files(path: Path) -> list[Path]:
    return sorted(
        child for child in path.iterdir()
        if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
    )


def load_image_as_tensor(path: str | Path, device: str) -> torch.Tensor:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    return ToTensor()(Image.open(path).convert("RGB")).to(device)


def save_results_to_csv(
    *,
    results: list[dict[str, float | str]],
    metric_cols: list[str],
    output_path: str | Path,
) -> None:
    if not results:
        raise ValueError("Cannot save an empty metrics result")

    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df = df[["filename"] + metric_cols]

    avg_metrics = df[metric_cols].mean(numeric_only=True).to_dict()
    avg_metrics["filename"] = "average"
    df = pd.concat([df, pd.DataFrame([avg_metrics])], ignore_index=True)

    df.to_csv(output_path, index=False, float_format="%.4f")
    print(f"Metrics saved to: {output_path}")


def validate_image_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise ValueError(f"`{name}` must have shape [1, 3, H, W], got: {tuple(tensor.shape)}")
