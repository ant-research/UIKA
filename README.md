<h1 align="center">UIKA: Fast Universal Head Avatar from Pose-Free Images</h1>

<p align="center">
  <a href="https://zijian-wu.github.io/">Zijian Wu</a><sup>1,2,*</sup>,
  <a href="https://yaourtb.github.io/">Boyao Zhou</a><sup>2,†</sup>,
  <a href="https://huliangxiao.github.io/">Liangxiao Hu</a><sup>2</sup>,
  <a href="https://kumapowerliu.github.io/">Hongyu Liu</a><sup>2,3</sup>,
  <a href="https://github.com/YuanSun-XJTU/">Yuan Sun</a><sup>2,4</sup>,
  <a href="https://xuanwangvc.github.io/">Xuan Wang</a><sup>2,4</sup>,
  <a href="https://cite.nju.edu.cn/People/Faculty/20190621/i5054.html/">Xun Cao</a><sup>1</sup>,
  <a href="https://shenyujun.github.io/">Yujun Shen</a><sup>2</sup>,
  <a href="http://zhuhao.cc/home/">Hao Zhu</a><sup>1,✉</sup>
</p>

<p align="center">
  <sup>1</sup>Nanjing University,
  <sup>2</sup>Ant Group,
  <sup>3</sup>HKUST,
  <sup>4</sup>Xi'an Jiaotong University
</p>

<p align="center">
  <sup>*</sup>Work done during an internship at Ant Group,
  <sup>†</sup>Project lead,
  <sup>✉</sup>Corresponding author
</p>

<p align="center">
  <strong>CVPR 2026 Highlight</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.07603">
    <img src="https://img.shields.io/badge/arXiv-2601.07603-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://zijian-wu.github.io/uika-page/">
    <img src="https://img.shields.io/badge/Project-Homepage-blue.svg" alt="Project Page">
  </a>
  <a href="https://huggingface.co/Yuukki/UIKA">
    <img src="https://img.shields.io/badge/HuggingFace-Model-yellow.svg" alt="Hugging Face Model">
  </a>
  <a href="https://github.com/Zijian-Wu/HeadEngine">
    <img src="https://img.shields.io/badge/Synthetic_Data-Pipeline-green.svg" alt="Synthetic Data Pipeline">
  </a>
</p>

<div align=center>
  <img src="./assets/teaser.gif">
</div>

> We present UIKA, a feed-forward 3D reconstruction model for creating animatable Gaussian head avatars from an arbitrary number of inputs, including a single image, multi-view captures, and smartphone-captured videos.

## Installation

The default setup targets CUDA 11.8:

```bash
conda create -n uika python=3.10 -y
conda activate uika
conda install -c "nvidia/label/cuda-11.8.0" cuda=11.8.0 -y

pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu118
pip install -r install/requirements.txt --no-build-isolation
```

## Weights and Assets

The default inference config uses the released checkpoint at
`./model_zoo/uika/uika.safetensors`. Use
`inference.checkpoint=/path/to/model.safetensors` only when testing a custom
checkpoint.

Auxiliary dependency assets are stored under `model_zoo/`, including UV module
weights, FLAME assets, and tool weights under `model_zoo/tools/` for metrics,
matting, and head detection. Use
[install/prepare_assets.py](install/prepare_assets.py) for the public
downloadable assets and local layout verification, then place separately
licensed assets such as `flame2023.pkl`, `FLAME_masks.pkl`, and DINOv3 weights
in the expected `model_zoo/` subdirectories. See
[install/prepare_assets.md](install/prepare_assets.md) for the full expected
layout.

## Training

UIKA training expects prepared training data produced by
[VHAP](https://github.com/ShenhanQian/VHAP) processing.
Configure dataset roots and train/val metadata in
[configs/uika_base.yaml](configs/uika_base.yaml). The default config references
prepared **Nersemble**, **VFHQ**, **HDTF**, and **Synthetic Multi-View** data under
`train_data/`.

Each entry listed in `label/train_ids.json` or `label/val_ids.json` should map
to a sequence directory under the matching `export/` directory. **Nersemble**
uses an extra identity directory under `export/`; the other datasets use each
sequence directory directly as the ID.

```text
train_data/
|-- nersemble_v2/                         # Nersemble
|   |-- export/
|   |   |-- <identity_id>/
|   |   |   |-- <sequence_id>/
|   |   |   |   |-- transforms.json
|   |   |   |   |-- canonical_flame_param.npz
|   |   |   |   |-- images/
|   |   |   |   |-- fg_masks/
|   |   |   |   `-- flame_param/
|   |   |   `-- ...
|   |   `-- ...
|   `-- label/
|       |-- train_ids.json
|       `-- val_ids.json
|-- vfhq/                                 # VFHQ
|   |-- export/
|   |   |-- <sequence_id>/
|   |   |   |-- transforms.json
|   |   |   |-- canonical_flame_param.npz
|   |   |   |-- images/
|   |   |   |-- fg_masks/
|   |   |   `-- flame_param/
|   |   `-- ...
|   `-- label/
|       |-- train_ids.json
|       `-- val_ids.json
|-- hdtf/                                 # HDTF, same layout as VFHQ
|   |-- export/
|   `-- label/
`-- synth_mv/                             # Synthetic Multi-View, same layout as VFHQ
    |-- export/
    `-- label/
```

```bash
NUM_GPUS=8 TRAIN_CONFIG=./configs/uika_base.yaml ./train.sh
```

To train the FUVT UV-estimation module from scratch, use
[configs/fuvt_base.yaml](configs/fuvt_base.yaml) with the `train.fuvt` runner:

```bash
NUM_GPUS=8 TRAIN_RUNNER=train.fuvt TRAIN_CONFIG=./configs/fuvt_base.yaml ./train.sh
```

FUVT training produces the FUVT module checkpoint consumed by UIKA training and
inference. It requires the DINOv3-B/16 feature extractor weights and
`model_zoo/uv_modules/p3dmm.ckpt`; see
[install/prepare_assets.md](install/prepare_assets.md) for the full expected
weight layout.

For **synthetic data generation**, use the companion **[HeadEngine](https://github.com/Zijian-Wu/HeadEngine)**
project.

## Inference

After preparing assets, run inference with a reference image or image directory
and a motion directory containing `transforms.json`:

```bash
python -m uika.launch infer.uika --config configs/infer_uika.yaml \
  inference.image_input=/path/to/ref.png_or_ref_dir \
  inference.motion_dir=/path/to/motion_dir \
  inference.output_dir=outputs/demo
```

For `inference.motion_dir`, use one of the sample sequences under
`assets/motion/`, or process a custom monocular driving video with
[VHAP monocular tracking](https://github.com/ShenhanQian/VHAP/blob/main/doc/monocular.md)
and use its exported NeRF/3DGS-style sequence folder.

For `inference.image_input`, use a single image from `assets/ref/` or a folder
of multiple images of the same identity. Reference images **DO NOT** require FLAME
pose/shape estimation or camera estimation.

Useful inference overrides:

| Override | Values | Description |
| --- | --- | --- |
| `inference.camera_path` | `orbit`, `motion` | `orbit` uses the generated orbit camera path; `motion` uses the tracked cameras from `motion_dir/transforms.json`. |
| `inference.orbit.radius_x`, `radius_y` | float | Orbit ellipse size when `camera_path=orbit`. |
| `inference.orbit.center`, `look_at` | `[x, y, z]` | Orbit center and look-at point. |
| `inference.orbit.axis`, `up` | `x`, `y`, `z` | Orbit rotation axis and up direction; they must differ. |
| `inference.render_size` | int | Output render resolution. |
| `inference.render_chunk_size` | int | Frames per model forward; use a smaller value to reduce VRAM. `0` renders all frames at once. |
| `inference.save_frames`, `save_video` | `true`, `false` | Save RGBA PNG frames and/or RGB video. |
| `inference.debug.ref_grid` | `true`, `false` | Save `debug/ref_grid.png`. |
| `inference.debug.video_grid` | `true`, `false` | Save `debug/video_grid.mp4` with reference, driving, render, and motion views when enabled. |
| `inference.debug.include_driving_rgb` | `true`, `false` | Include the driving RGB frames from `motion_dir` in `video_grid`. |
| `inference.debug.vis_motion` | `true`, `false` | Render a motion mesh panel in `debug/video_grid.mp4` when `video_grid=true`. |
| `inference.debug.blend_motion` | `true`, `false` | Add a render/motion overlay panel in `video_grid` when `vis_motion=true`. |

To choose a GPU, prefix the command, for example `CUDA_VISIBLE_DEVICES=2 ...`.
Inside the process this appears as `cuda:0`, mapped to physical GPU `2`.

Outputs are written under `inference.output_dir`: RGBA frames in `frames/`, RGB
video at `video.mp4`, and run metadata at `metadata.json`. Public inference runs
in FP32; `inference.dtype` is not a supported override.

Reference images are head-cropped with `inference.head_detection.weights` before
masking, then all reference masks are generated with `inference.matting.weights`.
Input PNG alpha channels are ignored. If the head detector returns no bounding
box for a reference image, inference emits a warning and falls back to the full
image for that reference.

## Metrics

Use [tools/metrics/evaluate.py](tools/metrics/evaluate.py) to evaluate
rendered frames against prepared ground-truth frames. Predicted and ground-truth
directories are matched by sorted image order and must contain the same number of
image files.

Self-reenactment evaluation reports `PSNR`, `SSIM`, `LPIPS`, `L1`, `AKD`,
`CSIM`, `AED`, and `APD`:

```bash
python tools/metrics/evaluate.py \
  --mode self \
  --pred-dir outputs/demo/frames \
  --gt-dir /path/to/gt_frames \
  --output metrics/demo_self.csv
```

Cross-reenactment evaluation reports `CSIM`, `AED`, and `APD`; `CSIM` compares
the prediction with the reference image, while `AED` and `APD` compare the
prediction with the driving ground truth:

```bash
python tools/metrics/evaluate.py \
  --mode cross \
  --pred-dir outputs/demo/frames \
  --gt-dir /path/to/gt_frames \
  --ref-image /path/to/reference.png \
  --output metrics/demo_cross.csv
```

Use `--on-error skip` only when you want to skip frames where face detection or a
metric model fails. Predictions and ground truth must already have the same
resolution.

Metrics require
`model_zoo/tools/deep3dface_recon_2023ver_epoch_20.pth`. `CSIM` uses
InsightFace's `buffalo_l` cache under `~/.insightface/models`; `AKD` uses
`face_alignment` caches under `~/.cache/torch/hub/checkpoints`; `LPIPS` uses
the `lpips` package weights and may use the torchvision AlexNet cache.

## Acknowledgement

This work is built on many amazing research works and open-source projects:

- [LAM](https://github.com/aigc3d/LAM)
- [OpenLRM](https://github.com/3DTopia/OpenLRM)
- [VHAP](https://github.com/ShenhanQian/VHAP)
- [Pixel3DMM](https://github.com/SimonGiebenhain/pixel3dmm)
- [VGGT](https://github.com/facebookresearch/vggt)

Thanks for their excellent works and great contribution.

## License

The UIKA-authored code and UIKA-released checkpoints, including the UIKA
checkpoint and FUVT module checkpoint, are released under the
[MIT License](LICENSE), copyright (c) 2025-2026, Zijian Wu. Third-party components,
model code, model weights, datasets, and parametric model assets remain subject
to their own licenses and access terms. See
[LICENSES/THIRD_PARTY_LICENSES.md](LICENSES/THIRD_PARTY_LICENSES.md) for known
third-party notices in this repository, with vendored third-party license texts
under [LICENSES/](LICENSES/).

## Citation

If you find this project useful, please cite:

```bibtex
@inproceedings{wu2026uika,
    title     = {UIKA: Fast Universal Head Avatar from Pose-Free Images},
    author    = {Wu, Zijian and Zhou, Boyao and Hu, Liangxiao and Liu, Hongyu and Sun, Yuan and Wang, Xuan and Cao, Xun and Shen, Yujun and Zhu, Hao},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2026}
}
```
