#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


REPO_ID = "Yuukki/UIKA"

UIKA_SHA256 = "a1d7f56e0e8073de5699f9ad247c08c88f203c79bff37babdfdc84e7641eb46a"
FUVT_SHA256 = "9316302564e0ebfeaf63031af98870b9718228dc4223c394b8f6c4904f554e2a"
STYLEMATTE_SHA256 = "5ce985571e909b6677d7d25e560216fa3f620e5cd337a8382ee0799c6d9af16c"
VGGHEAD_SHA256 = "18acb79c53032db11e8f502c12fdd34b5f642e9bc9041bce152c7b716c1b6f74"
DEEP3D_SHA256 = "62e7e6a6bc4e16fb567643182ccabf55f8222746269cce3392109a6c592babc1"
P3DMM_SHA256 = "dff9d73feec47914b704759f57ebffb8c58d2aef550b426013fe31eae21707b8"

HUMAN_PARAMETRIC_MODEL_FILES = [
    "landmark_embedding_with_eyes.npy",
    "flame_w_mouth.obj",
    "head_template_mesh.obj",
    "oral_jawopen0p5.obj",
    "shoulder_mesh.obj",
    "teeth_blendshape.json",
]

MANUAL_FLAME_FILES = [
    "flame2023.pkl",
    "FLAME_masks.pkl",
]

DINO_FILES = [
    "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
]


@dataclass
class CheckResult:
    ok: list[str]
    missing_required: list[str]
    missing_manual: list[str]


class Console:
    def __init__(self, use_color: bool):
        self.use_color = use_color

    def color(self, text: str, code: str) -> str:
        if not self.use_color:
            return text
        return f"\033[{code}m{text}\033[0m"

    def ok(self, message: str) -> None:
        print(self.color(f"OK: {message}", "32"))

    def info(self, message: str) -> None:
        print(message)

    def warn(self, message: str) -> None:
        print(self.color(f"WARNING: {message}", "33"))

    def error(self, message: str) -> None:
        print(self.color(f"ERROR: {message}", "31"), file=sys.stderr)

    def action(self, message: str) -> None:
        print(self.color(f"ACTION REQUIRED: {message}", "1;33"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify UIKA weights and demo assets.",
    )
    parser.add_argument(
        "--repo-id",
        default=REPO_ID,
        help=f"Hugging Face repo for UIKA release files. Default: {REPO_ID}",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Also prepare UIKA training assets: fuvt_15k and DINOv3 checks.",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Also download and verify metric assets.",
    )
    parser.add_argument(
        "--fuvt-training",
        action="store_true",
        help="Also prepare FUVT-from-scratch assets: Pixel3DMM and DINOv3-B check.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Enable --training, --metrics, and --fuvt-training.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify local files; do not download or extract anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download/re-install downloadable assets even when local files exist.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored terminal output.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_hf_file(
    *,
    repo_id: str,
    filename: str,
    output_path: Path,
    expected_sha256: str | None = None,
    verify_only: bool,
    force: bool,
    console: Console,
) -> bool:
    if output_path.is_file() and output_path.stat().st_size > 0 and (not force or verify_only):
        if expected_sha256 is None:
            console.ok(str(output_path))
            return True
        actual_sha256 = sha256_file(output_path)
        if actual_sha256 == expected_sha256:
            console.ok(str(output_path))
            return True
        if verify_only:
            console.warn(f"checksum mismatch: {output_path}")
            return False
        console.warn(f"checksum mismatch, re-downloading: {output_path}")

    if verify_only:
        console.warn(f"missing: {output_path}")
        return False

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install dependencies with "
            "`pip install -r install/requirements.txt`."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    console.info(f"Downloading {repo_id}/{filename} -> {output_path}")
    cached_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model")
    shutil.copy2(cached_path, output_path)
    if expected_sha256 is not None:
        actual_sha256 = sha256_file(output_path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {output_path}\n"
                f"expected: {expected_sha256}\n"
                f"actual:   {actual_sha256}"
            )
    console.ok(str(output_path))
    return True


def ensure_url_file(
    *,
    url: str,
    output_path: Path,
    expected_sha256: str,
    verify_only: bool,
    force: bool,
    console: Console,
) -> bool:
    if output_path.is_file() and (not force or verify_only):
        actual_sha256 = sha256_file(output_path)
        if actual_sha256 == expected_sha256:
            console.ok(str(output_path))
            return True
        console.warn(f"checksum mismatch, re-downloading: {output_path}")

    if verify_only:
        console.warn(f"missing or invalid: {output_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    console.info(f"Downloading {url} -> {output_path}")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    actual_sha256 = sha256_file(tmp_path)
    if actual_sha256 != expected_sha256:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {output_path}\n"
            f"expected: {expected_sha256}\n"
            f"actual:   {actual_sha256}"
        )

    tmp_path.replace(output_path)
    console.ok(str(output_path))
    return True


def ensure_gdrive_file(
    *,
    file_id: str,
    output_path: Path,
    expected_sha256: str,
    verify_only: bool,
    force: bool,
    console: Console,
) -> bool:
    if output_path.is_file() and (not force or verify_only):
        actual_sha256 = sha256_file(output_path)
        if actual_sha256 == expected_sha256:
            console.ok(str(output_path))
            return True
        console.warn(f"checksum mismatch, re-downloading: {output_path}")

    if verify_only:
        console.warn(f"missing or invalid: {output_path}")
        return False

    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required for this asset. Install dependencies with "
            "`pip install -r install/requirements.txt`."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    console.info(f"Downloading Google Drive file {file_id} -> {output_path}")
    result = gdown.download(id=file_id, output=str(tmp_path), quiet=False)
    if result is None:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Google Drive download failed for {output_path}")

    actual_sha256 = sha256_file(tmp_path)
    if actual_sha256 != expected_sha256:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {output_path}\n"
            f"expected: {expected_sha256}\n"
            f"actual:   {actual_sha256}"
        )

    tmp_path.replace(output_path)
    console.ok(str(output_path))
    return True


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            member_path = destination / member.name
            resolved_member_path = member_path.resolve()
            try:
                resolved_member_path.relative_to(destination_resolved)
            except ValueError as exc:
                raise RuntimeError(f"Unsafe tar path: {member.name}") from exc
            if member.issym() or member.islnk():
                raise RuntimeError(f"Tar links are not allowed: {member.name}")

        for member in tar.getmembers():
            member_path = destination / member.name
            if member.isdir():
                member_path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                member_path.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Failed to read tar member: {member.name}")
                with source, member_path.open("wb") as target:
                    shutil.copyfileobj(source, target)


def copy_tree_merge(source: Path, destination: Path, *, force: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.rglob("*"):
        relative_path = item.relative_to(source)
        target = destination / relative_path
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if item.is_file():
            if target.exists() and not force:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def download_hf_archive(
    *,
    repo_id: str,
    filename: str,
    verify_only: bool,
    force: bool,
    console: Console,
) -> Path | None:
    if verify_only:
        return None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required. Install dependencies with "
            "`pip install -r install/requirements.txt`."
        ) from exc
    console.info(f"Downloading archive {repo_id}/{filename}")
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="model"))


def find_human_assets_root(extracted_root: Path) -> Path:
    candidates = [
        extracted_root / "model_zoo" / "human_parametric_models",
        extracted_root / "human_parametric_models",
        extracted_root,
    ]
    for candidate in candidates:
        if all((candidate / name).is_file() for name in HUMAN_PARAMETRIC_MODEL_FILES):
            return candidate
    raise RuntimeError(
        "human_parametric_models.tar does not contain the expected auxiliary "
        "FLAME assets."
    )


def install_human_parametric_archive(
    *,
    repo_id: str,
    root: Path,
    verify_only: bool,
    force: bool,
    console: Console,
) -> None:
    target_dir = root / "model_zoo" / "human_parametric_models"
    if all((target_dir / name).is_file() for name in HUMAN_PARAMETRIC_MODEL_FILES) and not force:
        console.ok(str(target_dir))
        return
    if verify_only:
        console.warn(f"missing auxiliary human parametric assets under {target_dir}")
        return

    archive_path = download_hf_archive(
        repo_id=repo_id,
        filename="human_parametric_models.tar",
        verify_only=verify_only,
        force=force,
        console=console,
    )
    assert archive_path is not None
    with tempfile.TemporaryDirectory(prefix="uika_human_assets_") as tmp:
        tmp_root = Path(tmp)
        safe_extract_tar(archive_path, tmp_root)
        source_root = find_human_assets_root(tmp_root)
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in source_root.rglob("*"):
            if item.is_dir() or not item.is_file():
                continue
            if item.name in MANUAL_FLAME_FILES:
                console.warn(
                    f"Skipping {item.name} from archive. Users must download "
                    "this file from FLAME directly."
                )
                continue
            target = target_dir / item.relative_to(source_root)
            if target.exists() and not force:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
    console.ok(str(target_dir))


def find_example_roots(extracted_root: Path) -> tuple[Path, Path]:
    candidates = [
        (extracted_root / "assets" / "ref", extracted_root / "assets" / "motion"),
        (extracted_root / "ref", extracted_root / "motion"),
    ]
    for ref_root, motion_root in candidates:
        if ref_root.is_dir() and motion_root.is_dir():
            return ref_root, motion_root
    raise RuntimeError("ref_motion_example.tar must contain ref/ and motion/ examples.")


def examples_are_present(root: Path) -> bool:
    ref_root = root / "assets" / "ref"
    motion_root = root / "assets" / "motion"
    has_ref = ref_root.is_dir() and any(
        path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        for path in ref_root.rglob("*")
        if path.is_file()
    )
    has_motion = motion_root.is_dir() and any(
        path.name == "transforms.json" for path in motion_root.rglob("transforms.json")
    )
    return has_ref and has_motion


def install_ref_motion_examples(
    *,
    repo_id: str,
    root: Path,
    verify_only: bool,
    force: bool,
    console: Console,
) -> None:
    if examples_are_present(root) and not force:
        console.ok("assets/ref and assets/motion")
        return
    if verify_only:
        console.warn("missing reference/motion examples under assets/")
        return

    archive_path = download_hf_archive(
        repo_id=repo_id,
        filename="ref_motion_example.tar",
        verify_only=verify_only,
        force=force,
        console=console,
    )
    assert archive_path is not None
    with tempfile.TemporaryDirectory(prefix="uika_examples_") as tmp:
        tmp_root = Path(tmp)
        safe_extract_tar(archive_path, tmp_root)
        ref_source, motion_source = find_example_roots(tmp_root)
        copy_tree_merge(ref_source, root / "assets" / "ref", force=force)
        copy_tree_merge(motion_source, root / "assets" / "motion", force=force)
    console.ok("assets/ref and assets/motion")


def check_file(
    result: CheckResult,
    path: Path,
    *,
    label: str | None = None,
    manual: bool = False,
    expected_sha256: str | None = None,
) -> None:
    display = label or str(path)
    if path.is_file() and path.stat().st_size > 0:
        if expected_sha256 is not None:
            actual_sha256 = sha256_file(path)
            if actual_sha256 != expected_sha256:
                result.missing_required.append(f"{display} (checksum mismatch)")
                return
        result.ok.append(display)
    elif manual:
        result.missing_manual.append(display)
    else:
        result.missing_required.append(display)


def check_dir_files(
    result: CheckResult,
    directory: Path,
    filenames: list[str],
    *,
    manual: bool = False,
) -> None:
    for filename in filenames:
        check_file(result, directory / filename, manual=manual)


def verify_layout(root: Path, *, training: bool, metrics: bool, fuvt_training: bool) -> CheckResult:
    result = CheckResult(ok=[], missing_required=[], missing_manual=[])

    check_file(
        result,
        root / "model_zoo" / "uika" / "uika.safetensors",
        expected_sha256=UIKA_SHA256,
    )
    check_file(result, root / "model_zoo" / "tools" / "stylematte_synth.pt")
    check_file(result, root / "model_zoo" / "tools" / "vgg_heads_l.trcd")
    check_dir_files(
        result,
        root / "model_zoo" / "human_parametric_models",
        HUMAN_PARAMETRIC_MODEL_FILES,
    )
    check_dir_files(
        result,
        root / "model_zoo" / "human_parametric_models",
        MANUAL_FLAME_FILES,
        manual=True,
    )

    if examples_are_present(root):
        result.ok.append("assets/ref and assets/motion")
    else:
        result.missing_required.append("assets/ref and assets/motion")

    if training:
        check_file(
            result,
            root / "model_zoo" / "uv_modules" / "fuvt_15k.safetensors",
            expected_sha256=FUVT_SHA256,
        )
        check_dir_files(
            result,
            root / "model_zoo" / "feature_extractor",
            DINO_FILES,
            manual=True,
        )

    if metrics:
        check_file(
            result,
            root / "model_zoo" / "tools" / "deep3dface_recon_2023ver_epoch_20.pth",
        )

    if fuvt_training:
        check_file(result, root / "model_zoo" / "uv_modules" / "p3dmm.ckpt")
        check_file(
            result,
            root / "model_zoo" / "feature_extractor" / DINO_FILES[0],
            manual=True,
        )

    return result


def print_verification(result: CheckResult, console: Console) -> None:
    console.info("\nLocal verification")
    for item in result.ok:
        console.ok(item)
    for item in result.missing_required:
        console.error(f"missing required downloadable asset: {item}")
    for item in result.missing_manual:
        console.action(f"missing manually licensed asset: {item}")


def print_manual_instructions(result: CheckResult, console: Console) -> None:
    if not result.missing_manual:
        return

    missing_text = "\n".join(result.missing_manual)
    console.info("\nManual assets")
    if "flame2023.pkl" in missing_text or "FLAME_masks.pkl" in missing_text:
        console.action(
            "Download FLAME 2023 from https://flame.is.tue.mpg.de/index.html after "
            "registration/login. Place `flame2023.pkl` and `FLAME_masks.pkl` "
            "under `model_zoo/human_parametric_models/`. Use "
            "`flame2023.pkl`, not `flame2023_no_jaw.pkl`."
        )
    if "feature_extractor" in missing_text or "dinov3_" in missing_text:
        console.action(
            "DINOv3 weights are not redistributed by Yuukki/UIKA. Accept the "
            "official Meta DINOv3 terms on Hugging Face and place the required "
            "files under `model_zoo/feature_extractor/`."
        )


def main() -> int:
    args = parse_args()
    if args.all:
        args.training = True
        args.metrics = True
        args.fuvt_training = True

    root = repo_root()
    console = Console(use_color=(not args.no_color and sys.stdout.isatty()))

    try:
        ensure_hf_file(
            repo_id=args.repo_id,
            filename="uika.safetensors",
            output_path=root / "model_zoo" / "uika" / "uika.safetensors",
            expected_sha256=UIKA_SHA256,
            verify_only=args.verify_only,
            force=args.force,
            console=console,
        )
        install_human_parametric_archive(
            repo_id=args.repo_id,
            root=root,
            verify_only=args.verify_only,
            force=args.force,
            console=console,
        )
        install_ref_motion_examples(
            repo_id=args.repo_id,
            root=root,
            verify_only=args.verify_only,
            force=args.force,
            console=console,
        )
        ensure_url_file(
            url="https://github.com/chroneus/stylematte/releases/download/weights/stylematte_synth.pth",
            output_path=root / "model_zoo" / "tools" / "stylematte_synth.pt",
            expected_sha256=STYLEMATTE_SHA256,
            verify_only=args.verify_only,
            force=args.force,
            console=console,
        )
        ensure_url_file(
            url="https://huggingface.co/okupyn/vgg_heads/resolve/main/vgg_heads_l.trcd",
            output_path=root / "model_zoo" / "tools" / "vgg_heads_l.trcd",
            expected_sha256=VGGHEAD_SHA256,
            verify_only=args.verify_only,
            force=args.force,
            console=console,
        )

        if args.training:
            ensure_hf_file(
                repo_id=args.repo_id,
                filename="fuvt_15k.safetensors",
                output_path=root / "model_zoo" / "uv_modules" / "fuvt_15k.safetensors",
                expected_sha256=FUVT_SHA256,
                verify_only=args.verify_only,
                force=args.force,
                console=console,
            )

        if args.metrics:
            ensure_gdrive_file(
                file_id="15-38Iqv7vmZou8fDVBAt_c9PwgJn1BHo",
                output_path=root / "model_zoo" / "tools" / "deep3dface_recon_2023ver_epoch_20.pth",
                expected_sha256=DEEP3D_SHA256,
                verify_only=args.verify_only,
                force=args.force,
                console=console,
            )

        if args.fuvt_training:
            ensure_gdrive_file(
                file_id="1SDV_8_qWTe__rX_8e4Fi-BE3aES0YzJY",
                output_path=root / "model_zoo" / "uv_modules" / "p3dmm.ckpt",
                expected_sha256=P3DMM_SHA256,
                verify_only=args.verify_only,
                force=args.force,
                console=console,
            )

        result = verify_layout(
            root,
            training=args.training,
            metrics=args.metrics,
            fuvt_training=args.fuvt_training,
        )
        print_verification(result, console)
        print_manual_instructions(result, console)

        if result.missing_required:
            return 1
        if args.verify_only and result.missing_manual:
            return 1
        return 0
    except Exception as exc:  # noqa: BLE001 - installer should surface one clear message.
        console.error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
