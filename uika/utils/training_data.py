import logging
from typing import Optional, Tuple

import numpy as np
import torch


def scale_intrs(intrs, ratio_x, ratio_y):
    if len(intrs.shape) >= 3:
        intrs[:, 0] = intrs[:, 0] * ratio_x
        intrs[:, 1] = intrs[:, 1] * ratio_y
    else:
        intrs[0] = intrs[0] * ratio_x
        intrs[1] = intrs[1] * ratio_y
    return intrs


def calc_new_tgt_size_by_aspect(cur_hw, aspect_standard, tgt_size, multiply):
    assert abs(cur_hw[0] / cur_hw[1] - aspect_standard) < 0.03
    tgt_size = tgt_size * aspect_standard, tgt_size
    tgt_size = round(tgt_size[0] / multiply) * multiply, round(tgt_size[1] / multiply) * multiply
    ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
    return tgt_size, ratio_y, ratio_x


def img_center_padding(img_np, pad_ratio):
    ori_w, ori_h = img_np.shape[:2]

    w = round((1 + pad_ratio) * ori_w)
    h = round((1 + pad_ratio) * ori_h)

    if len(img_np.shape) > 2:
        img_pad_np = np.zeros((w, h, img_np.shape[2]), dtype=np.uint8)
    else:
        img_pad_np = np.zeros((w, h), dtype=np.uint8)
    offset_h, offset_w = (w - img_np.shape[0]) // 2, (h - img_np.shape[1]) // 2
    img_pad_np[offset_h: offset_h + img_np.shape[0]:, offset_w: offset_w + img_np.shape[1]] = img_np

    return img_pad_np


def center_crop_according_to_mask(img, mask, aspect_standard, enlarge_ratio):
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]
    ys, xs = np.where(mask > 0)

    fg_ratio = len(xs) / max(mask.size, 1)
    if fg_ratio < 0.01:
        raise ValueError(f"Degenerate mask (near-empty): fg_ratio={fg_ratio:.4f}")

    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)

    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

    half_w = max(abs(center_x - x_min), abs(center_x - x_max), 1)
    half_h = max(abs(center_y - y_min), abs(center_y - y_max), 1)

    aspect = half_h / half_w
    if aspect >= aspect_standard:
        half_w = round(half_h / aspect_standard)
    else:
        half_h = round(half_w * aspect_standard)

    half_h = min(half_h, center_y)
    half_w = min(half_w, center_x)
    half_h = max(half_h, 1)
    half_w = max(half_w, 1)

    if half_h / half_w > aspect_standard:
        half_h = max(round(half_w * aspect_standard), 1)
    else:
        half_w = max(round(half_h / aspect_standard), 1)

    if abs(enlarge_ratio[0] - 1) > 0.01 or abs(enlarge_ratio[1] - 1) > 0.01:
        enlarge_ratio_min, enlarge_ratio_max = enlarge_ratio
        enlarge_ratio_max_real = min(center_y / half_h, center_x / half_w)
        enlarge_ratio_max = min(enlarge_ratio_max_real, enlarge_ratio_max)
        enlarge_ratio_min = min(enlarge_ratio_max_real, enlarge_ratio_min)
        if enlarge_ratio_min > enlarge_ratio_max:
            enlarge_ratio_min = enlarge_ratio_max
        enlarge_ratio_cur = np.random.rand() * (enlarge_ratio_max - enlarge_ratio_min) + enlarge_ratio_min
        half_h, half_w = max(round(enlarge_ratio_cur * half_h), 1), max(round(enlarge_ratio_cur * half_w), 1)

    half_h = min(half_h, center_y)
    half_w = min(half_w, center_x)
    if half_w > 0 and abs(half_h / half_w - aspect_standard) >= 0.03:
        logging.warning(f"Aspect drift after crop: {half_h/half_w:.3f} vs {aspect_standard}")

    offset_x = center_x - half_w
    offset_y = center_y - half_h

    new_img = img[offset_y: offset_y + 2 * half_h, offset_x: offset_x + 2 * half_w]
    new_mask = mask[offset_y: offset_y + 2 * half_h, offset_x: offset_x + 2 * half_w]

    return new_img, new_mask, offset_x, offset_y


def load_cam_pose(frame_info, transpose_R=False):
    c2w = np.array(frame_info["transform_matrix"])
    c2w[:3, 1:3] *= -1
    c2w = torch.FloatTensor(c2w)

    # TODO: CHECK HERE!
    # if transpose_R:
    #     w2c = torch.inverse(c2w)
    #     w2c[:3, :3] = w2c[:3, :3].transpose(1, 0).contiguous()
    #     c2w = torch.inverse(w2c)

    intrinsic = torch.eye(4)
    intrinsic[0, 0] = frame_info["fl_x"]
    intrinsic[1, 1] = frame_info["fl_y"]
    intrinsic[0, 2] = frame_info["cx"]
    intrinsic[1, 2] = frame_info["cy"]
    intrinsic = intrinsic.float()

    return c2w, intrinsic


def load_flame_params(flame_file_path, teeth_bs=None):
    flame_param = dict(np.load(flame_file_path, allow_pickle=True))
    flame_param_tensor = {}
    flame_param_tensor["expr"] = torch.FloatTensor(flame_param["expr"])[0]
    flame_param_tensor["rotation"] = torch.FloatTensor(flame_param["rotation"])[0]
    flame_param_tensor["neck_pose"] = torch.FloatTensor(flame_param["neck_pose"])[0]
    flame_param_tensor["jaw_pose"] = torch.FloatTensor(flame_param["jaw_pose"])[0]
    flame_param_tensor["eyes_pose"] = torch.FloatTensor(flame_param["eyes_pose"])[0]
    flame_param_tensor["translation"] = torch.FloatTensor(flame_param["translation"])[0]
    if teeth_bs is not None:
        flame_param_tensor["teeth_bs"] = torch.FloatTensor(teeth_bs)

    return flame_param_tensor


def pad_to_square(
    rgb: np.ndarray,
    mask: np.ndarray,
    bg_color: float,
    intrinsic: Optional[torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, Optional[torch.Tensor]]:
    h, w = rgb.shape[:2]
    if h == w:
        return rgb, mask, intrinsic

    target_size = max(h, w)
    pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0

    if h < w:
        pad_total_h = w - h
        pad_top = pad_total_h // 2
        pad_bottom = pad_total_h - pad_top
    else:
        pad_total_w = h - w
        pad_left = pad_total_w // 2
        pad_right = pad_total_w - pad_left

    if intrinsic is not None:
        new_intrinsic = intrinsic.clone()
        new_intrinsic[0, 2] += pad_left
        new_intrinsic[1, 2] += pad_top
    else:
        new_intrinsic = None

    padded_rgb = np.pad(
        rgb,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=bg_color,
    )
    padded_mask = np.pad(
        mask,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0.0,
    )

    return padded_rgb, padded_mask, new_intrinsic
