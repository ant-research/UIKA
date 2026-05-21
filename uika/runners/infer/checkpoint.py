from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from uika.models import model_dict


@dataclass
class CheckpointLoadReport:
    loaded: list[str]
    missing: list[str]
    unexpected: list[str]
    mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]]


def build_model_from_config(
    cfg,
    *,
    skip_pretrained_weights: bool = False,
) -> torch.nn.Module:
    exp_type = cfg.experiment.type
    if exp_type not in model_dict:
        raise ValueError(
            f"Unknown experiment type `{exp_type}`. "
            f"Available model types: {sorted(model_dict.keys())}"
        )
    model_class = model_dict[exp_type]
    model_kwargs = dict(cfg.model)
    if skip_pretrained_weights:
        model_kwargs.update(
            {
                "encoder_pretrained": False,
                "fuvt_ckpt_path": None,
                "fuvt_load_pretrained": False,
                "fuvt_pretrained_patch_embed": False,
            }
        )
    print(f"model class: {model_class}")
    return model_class(**model_kwargs)


def load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = False,
    fail_on_shape_mismatch: bool = True,
) -> CheckpointLoadReport:
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print("=" * 100)
    print(f"loading inference checkpoint from: {checkpoint_path}")
    checkpoint = _load_checkpoint_file(checkpoint_path)
    ckpt_state = _extract_state_dict(checkpoint)

    model_state = model.state_dict()
    loaded: list[str] = []
    unexpected: list[str] = []
    mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    for key, value in ckpt_state.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            mismatched.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        model_state[key].copy_(value)
        loaded.append(key)

    missing = sorted(set(model_state.keys()) - set(loaded))
    report = CheckpointLoadReport(
        loaded=loaded,
        missing=missing,
        unexpected=unexpected,
        mismatched=mismatched,
    )

    if not loaded:
        raise RuntimeError(
            f"Loaded 0 matching tensors from checkpoint: {checkpoint_path}. "
            "Check that `inference.checkpoint` points to a UIKA model state dict."
        )
    if fail_on_shape_mismatch and mismatched:
        sample = _format_mismatch_sample(mismatched)
        raise RuntimeError(
            f"Checkpoint contains {len(mismatched)} shape-mismatched tensors. "
            f"Examples: {sample}"
        )
    if strict and (missing or unexpected):
        raise RuntimeError(
            "Strict checkpoint loading failed: "
            f"{len(missing)} missing keys, {len(unexpected)} unexpected keys."
        )

    print(
        "checkpoint load summary: "
        f"loaded={len(loaded)}, missing={len(missing)}, "
        f"unexpected={len(unexpected)}, mismatched={len(mismatched)}"
    )
    if missing:
        print("missing keys sample:", ", ".join(missing[:10]))
    if unexpected:
        print("unexpected keys sample:", ", ".join(unexpected[:10]))
    print("=" * 100)
    return report


def _load_checkpoint_file(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return load_file(str(path), device="cpu")
    if suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu")
    raise ValueError(f"Unsupported checkpoint suffix `{suffix}` for {path}")


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state dict or contain a nested state dict")

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint

    for key in ("state_dict", "model", "module"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict) and nested and all(torch.is_tensor(value) for value in nested.values()):
            return nested

    raise ValueError(
        "Could not find a tensor state dict in checkpoint. "
        "Expected raw state dict or one of keys: state_dict, model, module."
    )


def _format_mismatch_sample(
    mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]],
    limit: int = 5,
) -> str:
    return "; ".join(
        f"{key}: checkpoint{ckpt_shape} != model{model_shape}"
        for key, ckpt_shape, model_shape in mismatched[:limit]
    )
