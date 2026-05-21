import os
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from safetensors.torch import load_file

from uika.models.uv_modules import PixelUVWrapper
from uika.models.uv_modules.aggregator import Aggregator_Backbone
from uika.models.uv_modules.heads.dpt_head import DPTHead


logger = logging.getLogger(__name__)


class FUVT(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        pretrained_patch_embed: bool = True,
    ):
        super().__init__()

        self.aggregator = Aggregator_Backbone(
            pretrained_patch_embed=pretrained_patch_embed,
        )
        self.uv_head = DPTHead(
            dim_in=2 * embed_dim,
            patch_size=16,
            output_dim=3,
            activation="sigmoid",
            conf_activation="expp1",
            intermediate_layer_idx=[0, 1, 2, 3],
        )
    
    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        img: [B, V, C, H, W], value range: 0 ~ 1
        """
        B, V, C, H, W = images.shape
        assert H == 512 and W == 512

        aggregated_tokens_list = self.aggregator(images)

        with torch.cuda.amp.autocast(enabled=False):
            uv, uv_conf = self.uv_head(aggregated_tokens_list, images=images, patch_start_idx=0)

        return uv


class ModelFUVT(nn.Module):
    def __init__(self):
        super().__init__()

        self.fuvt = FUVT()
        self.uv_wrapper = PixelUVWrapper()
    
    @torch.compile
    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
        """
        image: [B, V, 3, H, W], value range: 0 ~ 1, 512x512
        mask: [B, V, 1, H, W], value range: 0 ~ 1, 512x512
        """
        B, V, _, H, W = image.shape

        output_uv = self.uv_wrapper(image.view(B * V, 3, H, W)[:, None])  # [B*V, 1, 2, H, W], value range: 0 ~ 1

        supervised_uv = output_uv[:, 0].view(B, V, 2, 512, 512) * mask

        pred_uv = self.fuvt(image)
        # uv: [B, V, 512, 512, 2]

        pred_uv_ori = pred_uv.permute(0, 1, 4, 2, 3).clone()
        pred_uv = pred_uv.permute(0, 1, 4, 2, 3) * mask
        
        gt_uv_color = torch.cat([supervised_uv, torch.zeros_like(supervised_uv[:, :, :1])], dim=2)  # [B, V, 3, 512, 512]
        pred_uv_color = torch.cat([pred_uv.detach(), torch.zeros_like(supervised_uv[:, :, :1])], dim=2)  # [B, V, 3, 512, 512]

        ret = {
            'pred_uv': pred_uv,  # [B, V, 2, 512, 512]
            'supervised_uv': supervised_uv,  # [B, V, 2, 512, 512]
            'gt_uv_color': gt_uv_color,  # [B, V, 3, 512, 512]
            'pred_uv_color': pred_uv_color,  # [B, V, 3, 512, 512]
            'pred_uv_ori': pred_uv_ori,  # [B, V, 2, 512, 512]
        }

        return ret


class FUVTWrapper(nn.Module):
    def __init__(
        self,
        ckpt_path: str | None = 'model_zoo/uv_modules/fuvt_15k.safetensors',
        load_pretrained: bool = True,
        pretrained_patch_embed: bool = True,
    ):
        super().__init__()
        if load_pretrained:
            if ckpt_path is None:
                raise ValueError("`ckpt_path` is required when `load_pretrained=True`")
            self.model = self._load_pretrained(
                ckpt_path,
                pretrained_patch_embed=pretrained_patch_embed,
            )
        else:
            self.model = FUVT(pretrained_patch_embed=pretrained_patch_embed)
        self._freeze()
    
    def _load_pretrained(
        self,
        ckpt_path: str,
        *,
        pretrained_patch_embed: bool,
    ):
        state_dict = load_file(ckpt_path, device='cpu')
        fuvt_state = {}
        for k, v in state_dict.items():
            if k.startswith('fuvt.'):
                new_k = k[len('fuvt.'):]
                fuvt_state[new_k] = v

        model = FUVT(pretrained_patch_embed=pretrained_patch_embed)
        model.load_state_dict(fuvt_state)
        return model
    
    def _freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.compile
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: [B, V, 3, H, W], value range: 0 ~ 1, 512x512
        """
        pred_uv = self.model(image)
        # uv: [B, V, 512, 512, 2]
        
        pred_uv = pred_uv.detach().permute(0, 1, 4, 2, 3)  # [B, V, 2, 512, 512]

        return pred_uv
