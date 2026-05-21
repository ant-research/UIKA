# Preparing Weights and Assets

Run commands from the repository root.

The release helper prepares the public inference demo assets and then verifies
the local layout:

```bash
python install/prepare_assets.py
```

Existing files are skipped. Use `--force` to re-download downloadable assets, or
`--verify-only` to only check the local layout.

## Release Boundary

[UIKA](https://huggingface.co/Yuukki/UIKA) is a public, non-gated Hugging Face repository. It distributes only
the assets that this project can redistribute directly:

```text
UIKA
|-- uika.safetensors
|-- fuvt_15k.safetensors
|-- human_parametric_models.tar      # excludes flame2023.pkl and FLAME_masks.pkl
`-- ref_motion_example.tar
```

Release artifact checks:

| File | Required for inference | Size | Verification |
| --- | --- | --- | --- |
| `uika.safetensors` | yes | ~4.6 GB | SHA256 `a1d7f56e0e8073de5699f9ad247c08c88f203c79bff37babdfdc84e7641eb46a` |
| `fuvt_15k.safetensors` | no, training only | ~1.4 GB | SHA256 `9316302564e0ebfeaf63031af98870b9718228dc4223c394b8f6c4904f554e2a` |
| `human_parametric_models.tar` | yes | release dependent | Extracted layout is verified; `flame2023.pkl` and `FLAME_masks.pkl` are manual FLAME downloads. |
| `ref_motion_example.tar` | demo only | release dependent | Extracted `assets/ref/` and `assets/motion/` layout is verified. |

The repository does **not** redistribute DINOv3 feature extractor weights or
FLAME assets that require the [FLAME website](https://flame.is.tue.mpg.de/) registration/login flow:
`flame2023.pkl` and `FLAME_masks.pkl`.

## Local Layout

After setup, the expected tree is:

```text
UIKA/
|-- assets/
|   |-- ref/                         # from ref_motion_example.tar
|   `-- motion/                      # from ref_motion_example.tar
`-- model_zoo/
    |-- uika/
    |   `-- uika.safetensors
    |-- human_parametric_models/
    |   |-- flame2023.pkl            # user downloads from FLAME manually
    |   |-- FLAME_masks.pkl          # user downloads from FLAME manually
    |   |-- landmark_embedding_with_eyes.npy
    |   |-- flame_w_mouth.obj
    |   |-- head_template_mesh.obj
    |   |-- oral_jawopen0p5.obj
    |   |-- shoulder_mesh.obj
    |   `-- teeth_blendshape.json
    |-- tools/
    |   |-- stylematte_synth.pt
    |   |-- vgg_heads_l.trcd
    |   `-- deep3dface_recon_2023ver_epoch_20.pth    # --metrics
    |-- uv_modules/
    |   |-- fuvt_15k.safetensors                    # --training
    |   `-- p3dmm.ckpt                              # --fuvt-training
    `-- feature_extractor/
        |-- dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
        `-- dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

## Installer Modes

```bash
python install/prepare_assets.py
python install/prepare_assets.py --training
python install/prepare_assets.py --metrics
python install/prepare_assets.py --fuvt-training
python install/prepare_assets.py --all
python install/prepare_assets.py --verify-only
```

| Mode | Downloads | Verifies / Highlights |
| --- | --- | --- |
| default | `uika.safetensors`, `human_parametric_models.tar`, `ref_motion_example.tar`, StyleMatte, VGGHead | `flame2023.pkl` and `FLAME_masks.pkl` manual FLAME step |
| `--training` | default + `fuvt_15k.safetensors` | DINOv3-B/L manual setup |
| `--metrics` | default + Deep3DFaceRecon checkpoint | InsightFace/LPIPS runtime caches are external |
| `--fuvt-training` | default + Pixel3DMM `p3dmm.ckpt` | DINOv3-B manual setup |
| `--all` | all downloadable groups | all manual steps |
| `--verify-only` | nothing | exits non-zero if required or manual files are missing |

The installer prints highlighted `ACTION REQUIRED` messages for assets that must
be obtained after registration/login or license acceptance.

## Manual FLAME Step

`human_parametric_models.tar` does not include `flame2023.pkl` or
`FLAME_masks.pkl`.

Register and log in at:

```text
https://flame.is.tue.mpg.de/index.html
```

Download the regular FLAME 2023 package, not the `FLAME 2023 Open
(for commercial use, CC-BY-4.0)` package. The regular archive contains:

```text
flame2023_no_jaw.pkl
flame2023.pkl
```

Use `flame2023.pkl`, not `flame2023_no_jaw.pkl`. Also download
`FLAME_masks.pkl` from the FLAME website. Place both files at:

```text
model_zoo/human_parametric_models/flame2023.pkl
model_zoo/human_parametric_models/FLAME_masks.pkl
```

## Public Inference

Public inference needs:

```text
model_zoo/uika/uika.safetensors
model_zoo/human_parametric_models/flame2023.pkl
model_zoo/human_parametric_models/FLAME_masks.pkl
model_zoo/human_parametric_models/landmark_embedding_with_eyes.npy
model_zoo/human_parametric_models/flame_w_mouth.obj
model_zoo/human_parametric_models/head_template_mesh.obj
model_zoo/tools/stylematte_synth.pt
model_zoo/tools/vgg_heads_l.trcd
assets/ref/
assets/motion/
```

The default inference config already points to the released checkpoint:

```yaml
inference:
  checkpoint: ./model_zoo/uika/uika.safetensors
```

The released `uika.safetensors` contains the UIKA image encoder and frozen FUVT
weights needed for inference. Standalone DINOv3 and `fuvt_15k.safetensors` are
not required for inference.

## Training

UIKA training from scratch additionally needs:

```text
model_zoo/uv_modules/fuvt_15k.safetensors
model_zoo/feature_extractor/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
model_zoo/feature_extractor/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

`fuvt_15k.safetensors` is downloaded by:

```bash
python install/prepare_assets.py --training
```

DINOv3 weights are not redistributed by [UIKA](https://huggingface.co/Yuukki/UIKA). Accept the official Meta
DINOv3 access terms on Hugging Face and place the files under
`model_zoo/feature_extractor/`. The code searches this directory by filename
substring, so the names must contain `dinov3_vitb16` and `dinov3_vitl16`.

Official DINOv3 pages:

```text
https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m
https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
```

## Metrics

Metric calculation additionally needs:

```text
model_zoo/tools/deep3dface_recon_2023ver_epoch_20.pth
```

Download it with:

```bash
python install/prepare_assets.py --metrics
```

Other metric dependencies are outside `model_zoo`: CSIM uses InsightFace
`buffalo_l` under `~/.insightface/models`, and LPIPS uses package/torchvision
caches.

## FUVT From Scratch

Only use this if you are training the FUVT UV-estimation module itself:

```text
model_zoo/uv_modules/p3dmm.ckpt
model_zoo/feature_extractor/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

Download Pixel3DMM with:

```bash
python install/prepare_assets.py --fuvt-training
```

The DINOv3-B weight remains a manual official-DINOv3 setup item.
