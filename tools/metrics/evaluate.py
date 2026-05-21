from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DEEP3D_CHECKPOINT = "model_zoo/tools/deep3dface_recon_2023ver_epoch_20.pth"


def _print_dependency_notes(mode: str, deep3d_checkpoint: str | Path, insightface_name: str) -> None:
    print("Metric dependencies:")
    print(f"- AED/APD: {Path(deep3d_checkpoint).expanduser()}")
    print(f"- CSIM: InsightFace `{insightface_name}` cache under ~/.insightface/models")
    if mode == "self":
        print("- AKD: face_alignment cache under ~/.cache/torch/hub/checkpoints")
        print("- LPIPS: lpips package weights plus torchvision AlexNet cache when needed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute UIKA image quality and face metrics.")
    parser.add_argument("--pred-dir", required=True, help="Directory containing predicted frames")
    parser.add_argument("--gt-dir", required=True, help="Directory containing ground-truth frames")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--ref-image", default=None, help="Reference image for cross reenactment CSIM")
    parser.add_argument("--mode", choices=["self", "cross"], default="self", help="Evaluation mode")
    parser.add_argument("--device", default=None, help="Torch device, default: cuda:0 when available else cpu")
    parser.add_argument(
        "--deep3d-checkpoint",
        default=DEFAULT_DEEP3D_CHECKPOINT,
        help="Deep3DFaceRecon checkpoint path used for AED/APD",
    )
    parser.add_argument("--insightface-name", default="buffalo_l", help="InsightFace model pack name")
    parser.add_argument(
        "--on-error",
        choices=["fail", "skip"],
        default="fail",
        help="Whether to fail immediately or skip frame pairs when one metric fails",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _print_dependency_notes(args.mode, args.deep3d_checkpoint, args.insightface_name)

    from tools.metrics.calculator import MetricsCalculator

    calculator = MetricsCalculator(
        mode=args.mode,
        device=args.device,
        deep3d_checkpoint=args.deep3d_checkpoint,
        insightface_name=args.insightface_name,
    )
    calculator.process_directories(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        output_path=args.output,
        ref_image=args.ref_image,
        on_error=args.on_error,
    )


if __name__ == "__main__":
    main()
