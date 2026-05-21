# Third-Party Licenses and Notices

The UIKA-authored code and UIKA-released checkpoints, including the UIKA
checkpoint and FUVT module checkpoint, are released under the MIT License.
Third-party source code, model code, model weights, datasets, and parametric
model assets remain subject to their own licenses and access terms.

This file summarizes known third-party license notices found in the source tree.
Where a full license text is vendored, it is placed in this directory.

## Source Code Components

| Component | Paths | License / notice found |
|---|---|---|
| Zexin He-authored training, dataset, loss, and utility code | `uika/runners/`, selected `uika/datasets/`, `uika/losses/`, `uika/utils/`, selected package `__init__.py` files | Apache-2.0 with copyright notice by Zexin He; see `Apache-2.0.txt` |
| DINOv3 code from Meta | `uika/models/encoders/dinov3/` | DINOv3 License; see `DINOv3-LICENSE.md` |
| DINOv2 and related Meta code | `uika/models/encoders/dinov2/`, selected `uika/models/uv_modules/layers/` | Apache-2.0; see `Apache-2.0.txt` |
| VGGT-derived data and prediction-head utilities | `uika/datasets/dynamic_dataloader.py`, `uika/datasets/worker_fn.py`, `uika/models/uv_modules/heads/` | VGGT License; see `VGGT-LICENSE.txt` |
| FLAME-related implementation | `uika/models/rendering/flame_model/` | Max Planck / FLAME proprietary notice |
| Spherical harmonics utilities | `uika/models/rendering/utils/sh_utils.py` | BSD-style PlenOctree notice |
| VGGHead detector wrapper | `tools/vgghead_detector/` | Copyright notice by Xuangeng Chu; no license text found in this repository |
| Deep3DFaceRecon metric helper | `tools/metrics/Deep3DFaceRecon.py` | Derived from Deep3DFaceRecon-style model code; no standalone license header found in this file |

## Model Weights and Assets

The UIKA checkpoint and FUVT module checkpoint are released under the repository
MIT License. Other model weights and assets are not covered by the repository
MIT License unless explicitly stated by their upstream providers. This includes,
but is not limited to:

- DINOv3 feature extractor weights.
- Pixel3DMM weights.
- FLAME parametric model assets.
- StyleMatte, VGGHead, InsightFace, face-alignment, LPIPS, and Deep3DFaceRecon
  weights or cache files.

Users are responsible for obtaining each model asset from its official source
and complying with its license, terms of use, and redistribution restrictions.

## Current Audit Summary

The current source tree contains multiple license families:

- MIT: UIKA-authored code and UIKA-released checkpoints.
- Apache-2.0: selected upstream-derived files and Zexin He-authored utility,
  dataset, loss, and runner code.
- Meta DINOv3 License Agreement: DINOv3 source files.
- VGGT License: selected Meta VGGT-derived utility files.
- FLAME / Max Planck proprietary notice: FLAME-related source files and assets.
- BSD-style PlenOctree notice: spherical harmonics helper.
- Copyright-only files without full license text: selected detector and metric
  helper code.

Before redistribution, review any copyright-only or proprietary-notice files and
confirm that their inclusion and redistribution are permitted.
