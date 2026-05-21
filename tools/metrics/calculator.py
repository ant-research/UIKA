from __future__ import annotations

from pathlib import Path
from typing import Literal

import face_alignment
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fused_ssim import fused_ssim
from insightface.app import FaceAnalysis
from torch.nn.functional import cosine_similarity
from torchmetrics.image import PeakSignalNoiseRatio

from tools.metrics.Deep3DFaceRecon import Deep3DFaceRecon
from tools.metrics.io import (
    collect_frame_pairs,
    load_image_as_tensor,
    save_results_to_csv,
    validate_image_tensor,
)


SELF_METRICS = ["PSNR", "SSIM", "LPIPS", "L1", "AKD", "CSIM", "AED", "APD"]
CROSS_METRICS = ["CSIM", "AED", "APD"]
DEFAULT_DEEP3D_CHECKPOINT = "model_zoo/tools/deep3dface_recon_2023ver_epoch_20.pth"


class MetricsCalculator(nn.Module):
    """Compute UIKA evaluation metrics for paired rendered and ground-truth frames."""

    def __init__(
        self,
        *,
        mode: Literal["self", "cross"] = "self",
        device: str | None = None,
        deep3d_checkpoint: str | Path = DEFAULT_DEEP3D_CHECKPOINT,
        insightface_name: str = "buffalo_l",
    ):
        super().__init__()
        if mode not in {"self", "cross"}:
            raise ValueError(f"`mode` must be self or cross, got: {mode}")

        self.mode = mode
        self.metric_cols = SELF_METRICS if mode == "self" else CROSS_METRICS
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        self.psnr = None
        self.lpips_model = None
        self.fa = None

        if self.mode == "self":
            self.psnr = PeakSignalNoiseRatio(data_range=(0.0, 1.0)).to(self.device)
            self.lpips_model = lpips.LPIPS(net="alex").to(self.device).eval()
            self.fa = face_alignment.FaceAlignment(
                _face_alignment_2d_landmark_type(),
                flip_input=False,
                device=self.device,
            )

        self.arcface = FaceAnalysis(name=insightface_name)
        self.arcface.prepare(ctx_id=_insightface_ctx_id(self.device), det_size=(512, 512))

        deep3d_checkpoint = Path(deep3d_checkpoint).expanduser()
        if not deep3d_checkpoint.is_file():
            raise FileNotFoundError(
                "Deep3DFaceRecon checkpoint is required for AED/APD metrics: "
                f"{deep3d_checkpoint}"
            )
        self.deep3dface_recon = Deep3DFaceRecon(
            checkpoint_path=str(deep3d_checkpoint),
            device=self.device,
        )

        print(f"MetricsCalculator initialized on device: {self.device}")

    def calculate_metrics_for_tensors(
        self,
        pred_tensor: torch.Tensor,
        gt_tensor: torch.Tensor,
        ref_tensor: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Compute metrics for one paired frame."""
        validate_image_tensor(pred_tensor, "pred_tensor")
        validate_image_tensor(gt_tensor, "gt_tensor")

        metrics: dict[str, float] = {}

        if self.mode == "self":
            assert self.psnr is not None
            assert self.lpips_model is not None
            metrics["PSNR"] = self.psnr(pred_tensor, gt_tensor).item()
            metrics["SSIM"] = fused_ssim(pred_tensor, gt_tensor, train=False).item()
            metrics["LPIPS"] = self.lpips_model(
                pred_tensor * 2.0 - 1.0,
                gt_tensor * 2.0 - 1.0,
            ).mean().item()
            metrics["L1"] = F.l1_loss(pred_tensor, gt_tensor).item()
            metrics["AKD"] = self.calculate_average_keypoint_distance(pred_tensor, gt_tensor)
            metrics["CSIM"] = self.calculate_arcface_similarity(pred_tensor, gt_tensor)
        else:
            if ref_tensor is None:
                raise ValueError("`ref_tensor` is required when `mode='cross'`")
            validate_image_tensor(ref_tensor, "ref_tensor")
            metrics["CSIM"] = self.calculate_arcface_similarity(pred_tensor, ref_tensor)

        metrics["AED"], metrics["APD"] = self.calculate_expression_pose_distance(
            pred_tensor,
            gt_tensor,
        )
        return metrics

    def process_directories(
        self,
        *,
        pred_dir: str | Path,
        gt_dir: str | Path,
        output_path: str | Path,
        ref_image: str | Path | None = None,
        on_error: Literal["fail", "skip"] = "fail",
    ) -> list[dict[str, float | str]]:
        """Evaluate sorted frame pairs from two directories and write a CSV report."""
        if on_error not in {"fail", "skip"}:
            raise ValueError(f"`on_error` must be fail or skip, got: {on_error}")

        frame_pairs = collect_frame_pairs(pred_dir, gt_dir)
        ref_tensor = None
        if self.mode == "cross":
            if ref_image is None:
                raise ValueError("`ref_image` is required when `mode='cross'`")
            ref_tensor = load_image_as_tensor(ref_image, self.device).unsqueeze(0)

        results: list[dict[str, float | str]] = []
        skipped: list[tuple[str, str]] = []

        print(f"Found {len(frame_pairs)} frame pairs to process")
        for pred_path, gt_path in frame_pairs:
            try:
                pred_tensor = load_image_as_tensor(pred_path, self.device).unsqueeze(0)
                gt_tensor = load_image_as_tensor(gt_path, self.device).unsqueeze(0)

                if pred_tensor.shape != gt_tensor.shape:
                    raise ValueError(
                        "Prediction and ground-truth frame shapes differ: "
                        f"{tuple(pred_tensor.shape)} vs {tuple(gt_tensor.shape)}. "
                        "Prepare predictions and ground truth at the same resolution before scoring."
                    )

                frame_metrics = self.calculate_metrics_for_tensors(
                    pred_tensor,
                    gt_tensor,
                    ref_tensor,
                )
                row: dict[str, float | str] = {"filename": pred_path.name, **frame_metrics}
                results.append(row)
                print(_format_frame_metrics(pred_path.name, frame_metrics))

            except Exception as exc:
                if on_error == "fail":
                    raise RuntimeError(
                        f"Failed to process frame pair `{pred_path}` and `{gt_path}`"
                    ) from exc
                skipped.append((pred_path.name, str(exc)))
                print(f"  Skipped {pred_path.name}: {exc}")

        if not results:
            raise RuntimeError("No valid frame pairs were evaluated")

        self.save_results_to_csv(results, output_path)
        if skipped:
            print(f"Skipped {len(skipped)} frame pair(s); first skipped error: {skipped[0][1]}")
        return results

    def save_results_to_csv(
        self,
        results: list[dict[str, float | str]],
        output_path: str | Path,
    ) -> None:
        """Save per-frame metrics plus an average row."""
        save_results_to_csv(
            results=results,
            metric_cols=self.metric_cols,
            output_path=output_path,
        )

    def calculate_arcface_similarity(self, pred_img: torch.Tensor, gt_img: torch.Tensor) -> float:
        emb_pred = self._get_arcface_embedding(pred_img)
        emb_gt = self._get_arcface_embedding(gt_img)
        return cosine_similarity(
            torch.from_numpy(emb_pred).unsqueeze(0),
            torch.from_numpy(emb_gt).unsqueeze(0),
        ).item()

    def calculate_average_keypoint_distance(
        self,
        pred_img: torch.Tensor,
        gt_img: torch.Tensor,
    ) -> float:
        if self.fa is None:
            raise RuntimeError("AKD is only available in self mode")

        pred_landmarks = self._get_landmarks(pred_img)
        gt_landmarks = self._get_landmarks(gt_img)
        if gt_landmarks.shape != pred_landmarks.shape:
            raise ValueError(
                "Landmark shape mismatch: "
                f"{gt_landmarks.shape} vs {pred_landmarks.shape}"
            )
        return float(np.linalg.norm(gt_landmarks - pred_landmarks, axis=1).mean())

    def calculate_expression_pose_distance(
        self,
        pred_img: torch.Tensor,
        gt_img: torch.Tensor,
    ) -> tuple[float, float]:
        exp_pred, pose_pred = self.deep3dface_recon(pred_img)
        exp_gt, pose_gt = self.deep3dface_recon(gt_img)
        aed = torch.mean(torch.abs(exp_pred - exp_gt))
        apd = torch.mean(torch.abs(pose_pred - pose_gt))
        return aed.item(), apd.item()

    def _get_arcface_embedding(self, img_t: torch.Tensor) -> np.ndarray:
        img_rgb = self._tensor_to_uint8_rgb(img_t)
        img_bgr = img_rgb[..., ::-1]
        faces = self.arcface.get(img_bgr)
        if len(faces) == 0:
            raise RuntimeError("No face detected by InsightFace")

        face = max(faces, key=lambda item: _bbox_area(item.bbox))
        return face.normed_embedding.astype(np.float32)

    def _get_landmarks(self, img_t: torch.Tensor) -> np.ndarray:
        if self.fa is None:
            raise RuntimeError("face_alignment is not initialized")

        img_rgb = self._tensor_to_uint8_rgb(img_t)
        predictions = self.fa.get_landmarks(img_rgb)
        if predictions is None or len(predictions) == 0:
            raise RuntimeError("No face detected by face_alignment")
        return predictions[0]

    @staticmethod
    def _tensor_to_uint8_rgb(img_t: torch.Tensor) -> np.ndarray:
        validate_image_tensor(img_t, "img_t")
        x = img_t.detach().cpu().squeeze(0).float().clamp(0.0, 1.0) * 255.0
        return x.permute(1, 2, 0).numpy().astype(np.uint8)


def _bbox_area(bbox: np.ndarray) -> float:
    return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))


def _face_alignment_2d_landmark_type():
    landmark_type = getattr(face_alignment.LandmarksType, "TWO_D", None)
    if landmark_type is not None:
        return landmark_type
    return face_alignment.LandmarksType._2D


def _insightface_ctx_id(device: str) -> int:
    if not str(device).startswith("cuda"):
        return -1
    if ":" not in str(device):
        return 0
    try:
        return int(str(device).split(":", 1)[1])
    except ValueError:
        return 0


def _format_frame_metrics(filename: str, metrics: dict[str, float]) -> str:
    values = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    return f"  Processed {filename}: {values}"
