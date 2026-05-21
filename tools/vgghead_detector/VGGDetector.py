#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)
# Modified based on code from Orest Kupyn (University of Oxford).

import torch
import numpy as np
import torchvision


def nms(boxes_xyxy, scores, flame_params,
        confidence_threshold: float = 0.5, iou_threshold: float = 0.5, 
        top_k: int = 1000, keep_top_k: int = 100
    ):
    for pred_bboxes_xyxy, pred_bboxes_conf, pred_flame_params in zip(
            boxes_xyxy.detach().float(),
            scores.detach().float(),
            flame_params.detach().float(),
    ):
        pred_bboxes_conf = pred_bboxes_conf.squeeze(-1)  # [Anchors]
        conf_mask = pred_bboxes_conf >= confidence_threshold

        pred_bboxes_conf = pred_bboxes_conf[conf_mask]
        pred_bboxes_xyxy = pred_bboxes_xyxy[conf_mask]
        pred_flame_params = pred_flame_params[conf_mask]

        # Filter all predictions by self.nms_top_k
        if pred_bboxes_conf.size(0) > top_k:
            topk_candidates = torch.topk(pred_bboxes_conf, k=top_k, largest=True, sorted=True)
            pred_bboxes_conf = pred_bboxes_conf[topk_candidates.indices]
            pred_bboxes_xyxy = pred_bboxes_xyxy[topk_candidates.indices]
            pred_flame_params = pred_flame_params[topk_candidates.indices]

        # NMS
        idx_to_keep = torchvision.ops.boxes.nms(boxes=pred_bboxes_xyxy, scores=pred_bboxes_conf, iou_threshold=iou_threshold)

        final_bboxes = pred_bboxes_xyxy[idx_to_keep][: keep_top_k]  # [Instances, 4]
        final_scores = pred_bboxes_conf[idx_to_keep][: keep_top_k]  # [Instances, 1]
        final_params = pred_flame_params[idx_to_keep][: keep_top_k]  # [Instances, Flame Params]
        return final_bboxes, final_scores, final_params


class VGGHeadDetector(torch.nn.Module):
    def __init__(self, device,
                 vggheadmodel_path=None):
        super().__init__()
        self.image_size = 640
        self._device = device
        self.vggheadmodel_path = vggheadmodel_path
        self._init_models()

    def _init_models(self,):
        # vgg_heads_l
        self.model = torch.load(self.vggheadmodel_path, map_location='cpu')
        self.model.to(self._device).eval()

    @torch.no_grad()
    def forward(self, image_tensor, image_key, conf_threshold=0.5):
        if not hasattr(self, 'model'):
            self._init_models()
        image_tensor = image_tensor.to(self._device).float()
        image, padding, scale = self._preprocess(image_tensor)
        bbox, scores, flame_params = self.model(image)
        bbox, vgg_results = self._postprocess(bbox, scores, flame_params, conf_threshold)

        if bbox is None:
            print('VGGHeadDetector: No face detected: {}!'.format(image_key))
            return None, None, None
        vgg_results['normalize'] = {'padding': padding, 'scale': scale}

        # bbox
        bbox = bbox.clip(0, self.image_size)
        bbox[[0, 2]] -= padding[0]; bbox[[1, 3]] -= padding[1]; bbox /= scale
        bbox = bbox.clip(0, self.image_size / scale)

        return vgg_results, bbox, None

    def _preprocess(self, image):
        _, h, w = image.shape
        if h > w:
            new_h, new_w = self.image_size, int(w * self.image_size / h)
        else:
            new_h, new_w = int(h * self.image_size / w), self.image_size
        scale = self.image_size / max(h, w)
        image = torchvision.transforms.functional.resize(image, (new_h, new_w), antialias=True)
        pad_w = self.image_size - image.shape[2]
        pad_h = self.image_size - image.shape[1]
        image = torchvision.transforms.functional.pad(
            image,
            (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2),
            fill=127,
        )
        image = image.unsqueeze(0).float() / 255.0
        return image, np.array([pad_w // 2, pad_h // 2]), scale

    def _postprocess(self, bbox, scores, flame_params, conf_threshold):
        bbox, scores, flame_params = nms(
            bbox,
            scores,
            flame_params,
            confidence_threshold=conf_threshold,
        )
        if bbox.shape[0] == 0:
            return None, None
        max_idx = ((bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])).argmax().long()
        bbox, flame_params = bbox[max_idx], flame_params[max_idx]
        if bbox[0] < 5 and bbox[1] < 5 and bbox[2] > 635 and bbox[3] > 635:
            return None, None
        # flame
        posecode = torch.cat([flame_params.new_zeros(3), flame_params[400:403]])
        vgg_results = {
            'rotation_6d': flame_params[403:409],
            'translation': flame_params[409:412],
            'scale': flame_params[412:],
            'shapecode': flame_params[:300],
            'expcode': flame_params[300:400],
            'posecode': posecode,
        }
        return bbox, vgg_results
