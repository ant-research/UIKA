import os
import torch
import torch.nn as nn

from torchvision.transforms import v2
# from accelerate.logging import get_logger

# logger = get_logger(__name__)


class LightFsuionHead(nn.Module):
    def __init__(
        self,
        in_channels,
        inner_channels,
        out_channel: int = 1024,
        use_clstoken: bool = False,
    ):
        super().__init__()
        
        self.use_clstoken = use_clstoken
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=inner_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for inner_channel in inner_channels
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.output_conv = nn.Conv2d(sum(inner_channels) , out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                assert isinstance(x, tuple)
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            out.append(x)
        
        fusion_feats = torch.cat(out, dim=1)        
        fusion_feats = self.output_conv(fusion_feats)
        return fusion_feats


class Dinov2FusionWrapper(nn.Module):
    """
    Dinov2FusionWrapper using original implementation, hacked with modulation.
    """
    def __init__(
        self,
        model_name: str,
        modulation_dim: int = None,
        freeze: bool = True,
        pretrained: bool = True,
        encoder_feat_dim: int = 384,
        use_clstoken: bool = False,
    ):
        super().__init__()
        self.modulation_dim = modulation_dim
        self.model = self._build_dinov2(
            model_name,
            modulation_dim=modulation_dim,
            pretrained=pretrained,
        )
        self.use_clstoken = use_clstoken
        
        self.intermediate_layer_idx_info = {
            'dinov2_vits14_reg': [2, 5, 8, 11],
            'dinov2_vitb14_reg': [2, 5, 8, 11], 
            'dinov2_vitl14_reg': [4, 11, 17, 23], 
            'dinov2_vitg14_reg': [9, 19, 29, 39]
        }
        
        self.intermediate_layer_idx = self.intermediate_layer_idx_info[model_name]
        self.fusion_head = LightFsuionHead(
            in_channels=self.model.embed_dim,
            inner_channels=[self.model.embed_dim] * 4,
            out_channel=encoder_feat_dim,
            use_clstoken=use_clstoken,
        )

        if freeze:
            if modulation_dim is not None:
                raise ValueError("Modulated Dinov2 requires training, freezing is not allowed.")
            self._freeze()


    def _freeze(self):
        # logger.warning(f"======== Freezing Dinov2FusionWrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @staticmethod
    def _build_dinov2(model_name: str, modulation_dim: int = None, pretrained: bool = True):
        from importlib import import_module
        dinov2_hub = import_module(".dinov2.hub.backbones", package=__package__)
        model_fn = getattr(dinov2_hub, model_name)
        # logger.debug(f"Modulation dim for Dinov2 is {modulation_dim}.")
        model = model_fn(modulation_dim=modulation_dim, pretrained=pretrained)
        return model

    @torch.compile
    def forward(self, image: torch.Tensor, mod: torch.Tensor = None):
        # image: [N, C, H, W]
        # mod: [N, D] or None
        # RGB image with [0,1] scale and properly sized
        
        patch_h, patch_w = image.shape[-2] // self.model.patch_size, image.shape[-1] // self.model.patch_size
        
        features = self.model.get_intermediate_layers(image, self.intermediate_layer_idx, return_class_token=self.use_clstoken)
        
        out_local = self.fusion_head(features,  patch_h, patch_w)

        out_global = None
        if out_global is not None:
            ret = torch.cat([out_local.permute(0, 2, 3, 1).flatten(1, 2), out_global.unsqueeze(1)], dim=1)
        else:
            ret = out_local.permute(0, 2, 3, 1).flatten(1, 2)  # (B, D, H, W) -> (B, H, W, D) -> (B, H*W, D)
        return ret


class Dinov3FusionWrapper(nn.Module):
    def __init__(
        self,
        model_name: str,
        modulation_dim: int = None,
        freeze: bool = True,
        pretrained: bool = True,
        encoder_feat_dim: int = 384,
        dual_fusion: bool = True,  # use different fusion block for screen/uv space dino token
        use_clstoken: bool = False,  # whether return class token from dino 
    ):
        super().__init__()
        self.modulation_dim = modulation_dim
        self.model = self._build_dinov3(
            model_name,
            modulation_dim=modulation_dim,
            pretrained=pretrained,
        )
        self.use_clstoken = use_clstoken
        self.dual_fusion = dual_fusion
        
        self.intermediate_layer_idx_info = {
            'dinov3_vits16': [2, 5, 8, 11],
            'dinov3_vits16plus': [2, 5, 8, 11],
            'dinov3_vitb16': [2, 5, 8, 11],
            'dinov3_vitl16': [4, 11, 17, 23],  # defined in dinov3/hub/depthers.py
            'dinov3_vith16plus': [7, 15, 23, 31],
            'dinov3_vit7b16': [9, 19, 29, 39]
        }
        
        self.intermediate_layer_idx = self.intermediate_layer_idx_info[model_name]
        self.normalize_transform = v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.fusion_head = LightFsuionHead(
            in_channels=self.model.embed_dim,
            inner_channels=[self.model.embed_dim] * 4,
            out_channel=encoder_feat_dim,
            use_clstoken=use_clstoken,
        )
        if dual_fusion:
            self.uv_fusion_head = LightFsuionHead(
                in_channels=self.model.embed_dim,
                inner_channels=[self.model.embed_dim] * 4,
                out_channel=encoder_feat_dim,
                use_clstoken=use_clstoken,
            )

        if freeze:
            if modulation_dim is not None:
                raise ValueError("Modulated Dinov3 requires training, freezing is not allowed.")
            self._freeze()


    def _freeze(self):
        # logger.warning(f"======== Freezing Dinov3FusionWrapper ========")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @staticmethod
    def _build_dinov3(model_name: str, modulation_dim: int = None, pretrained: bool = True):
        from importlib import import_module
        dinov3_hub = import_module(".dinov3.hub.backbones", package=__package__)
        model_fn = getattr(dinov3_hub, model_name)

        weight_path = None
        if pretrained:
            # load weight from local path
            WEIGHTS_BASE_PATH = 'model_zoo/feature_extractor'
            weight_names = os.listdir(WEIGHTS_BASE_PATH)
            for weight_name in weight_names:
                if model_name in weight_name:
                    weight_path = os.path.join(WEIGHTS_BASE_PATH, weight_name)
                    break
            if weight_path is None:
                raise ValueError(f"Weight path for {model_name} not found.")
        
        # logger.debug(f"Modulation dim for Dinov3 is {modulation_dim}.")
        model = model_fn(modulation_dim=modulation_dim, pretrained=pretrained, weights=weight_path)
        return model

    @torch.compile
    def forward(self, image: torch.Tensor, flag_screen_space: bool = True):
        # image: [N, C, H, W]
        # flag_screen_space: True for screen space dino token, False for uv space dino token
        # RGB image with [0, 1] scale and properly sized

        # Apply Normalize
        image = self.normalize_transform(image)
        
        patch_h, patch_w = image.shape[-2] // self.model.patch_size, image.shape[-1] // self.model.patch_size
        
        features = self.model.get_intermediate_layers(x=image, n=self.intermediate_layer_idx, return_class_token=self.use_clstoken)
        
        if flag_screen_space:
            out_local = self.fusion_head(features, patch_h, patch_w)
        else:
            out_local = self.uv_fusion_head(features, patch_h, patch_w)
        
        out_global = None
        if out_global is not None:
            ret = torch.cat([out_local.permute(0, 2, 3, 1).flatten(1, 2), out_global.unsqueeze(1)], dim=1)
        else:
            ret = out_local.permute(0, 2, 3, 1).flatten(1, 2)  # (B, D, H, W) -> (B, H, W, D) -> (B, H*W, D)
        return ret
