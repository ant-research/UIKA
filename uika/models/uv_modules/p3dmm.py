import torch, timm
import torch.nn as nn
import numpy as np
import pytorch_lightning as L

from torchvision import transforms
from torch.nn import MultiheadAttention
from torch.nn import functional as F


class DinoWrapper(L.LightningModule):
    """
    Dino v1 wrapper using huggingface transformer implementation.
    """

    def __init__(self, model_name: str, is_train: bool = False):
        super().__init__()
        self.model, self.processor = self._build_dino(model_name)
        self.freeze(is_train)

    def forward(self, image):
        outputs = self.model.forward_features(self.processor(image))
        return outputs[:, 1:]

    def freeze(self, is_train: bool = False):
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = is_train

    @staticmethod
    def _build_dino(model_name: str, proxy_error_retries: int = 3, proxy_error_cooldown: int = 5):
        import requests
        try:
            model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True)
            data_config = timm.data.resolve_model_data_config(model)
            processor = transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
            return model, processor
        except requests.exceptions.ProxyError as err:
            if proxy_error_retries > 0:
                import time
                time.sleep(proxy_error_cooldown)
                return DinoWrapper._build_dino(model_name, proxy_error_retries - 1, proxy_error_cooldown)
            else:
                raise err


class GroupAttBlock(L.LightningModule):
    def __init__(self, inner_dim: int, input_dim: int,
                 num_heads: int, eps: float,
                 attn_drop: float = 0., attn_bias: bool = False,
                 mlp_ratio: float = 4., mlp_drop: float = 0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(inner_dim)
        self.self_attn = MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=inner_dim, vdim=inner_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.self_attn2 = MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=inner_dim, vdim=inner_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)

        self.norm2 = norm_layer(inner_dim)
        self.norm3 = norm_layer(inner_dim)
        self.norm4 = norm_layer(inner_dim)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x, facial_components=None):
        B, V, C, H, W = x.shape

        x = x.permute(0, 1, 3, 4, 2).view(B, V * H * W, C)
        if facial_components is not None:
            n_facial_components = facial_components.shape[1]
            x = torch.cat([x, facial_components], dim=1)
        patches = self.norm1(x)
        patches = patches + self.self_attn(patches, patches, patches, need_weights=False)[0]
        patches = patches + self.mlp(self.norm2(patches))

        patches = self.norm3(patches)
        patches = patches + self.self_attn2(patches, patches, patches, need_weights=False)[0]
        patches = patches + self.mlp2(self.norm4(patches))

        if facial_components is not None:
            facial_components = patches[:, -n_facial_components:, :]
            patches = patches[:, :-n_facial_components, :]
        else:
            facial_components = None

        patches = patches.reshape(B, V, H, W, C).permute(0, 1, 4, 2, 3)

        return patches, facial_components


class VolTransformer(L.LightningModule):
    def __init__(self, embed_dim: int, image_feat_dim: int, n_groups: list,
                 vol_low_res: int, vol_high_res: int, out_dim: int,
                 num_layers: int, num_heads: int,
                 eps: float = 1e-6):
        super().__init__()

        self.vol_low_res = vol_low_res
        self.vol_high_res = vol_high_res
        self.out_dim = out_dim
        self.n_groups = n_groups
        self.embed_dim = embed_dim

        self.down_proj = torch.nn.Linear(image_feat_dim, embed_dim)

        self.layers = nn.ModuleList([
            GroupAttBlock(
                inner_dim=embed_dim, input_dim=image_feat_dim, num_heads=num_heads, eps=eps)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=eps)

    def forward(self, image_feats, facial_components=None):
        B, V, C, H, W = image_feats.shape

        image_feats = self.down_proj(image_feats.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)

        for i, layer in enumerate(self.layers):
            image_feats, facial_components = layer(image_feats, facial_components)

        x = image_feats
        return x, facial_components


def unpatchify(x, batch_size, channels=3, patch_size=16, n_views: int = 1):
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]
    x = x.reshape(shape=(batch_size, n_views, h, w, patch_size, patch_size, channels))
    x = torch.einsum('nvhwpqc->nvchpwq', x)
    imgs = x.reshape(shape=(batch_size, n_views, channels, h * patch_size, h * patch_size))
    return imgs


class Network(L.LightningModule):
    def __init__(self, cfg, white_bkgd=True):
        super().__init__()

        self.cfg = cfg
        if not hasattr(cfg.model, 'pred_disentangled'):
            cfg.model.pred_disentangled = False

        self.scene_size = 0.5
        self.white_bkgd = white_bkgd

        self.img_encoder = DinoWrapper(
            model_name=cfg.model.encoder_backbone,
            is_train=self.cfg.model.finetune_backbone,
        )
        self.feat_map_size = 32

        encoder_feat_dim = self.img_encoder.model.num_features

        if self.cfg.model.use_pos_enc:
            self.patch_pos_enc = nn.Parameter(
                torch.randn(1, encoder_feat_dim, self.feat_map_size, self.feat_map_size) * (1 / encoder_feat_dim) ** 0.5
            )

        if self.cfg.n_views > 1:
            self.view_embed = nn.Parameter(
                torch.randn(1, self.cfg.n_views, self.cfg.model.view_embed_dim, 1, 1) * (
                        1 / cfg.model.view_embed_dim) ** 0.5
            )
            inp_dim_transformer = encoder_feat_dim + cfg.model.view_embed_dim
        else:
            inp_dim_transformer = encoder_feat_dim

        embedding_dim = cfg.model.embedding_dim
        self.vol_decoder = VolTransformer(
            embed_dim=embedding_dim, image_feat_dim=inp_dim_transformer,
            vol_low_res=None, vol_high_res=None, out_dim=cfg.model.vol_embedding_out_dim, n_groups=None,
            num_layers=cfg.model.num_layers, num_heads=cfg.model.num_heads,
        )

        self.prediction_dim = 0
        for prediction_type in ['pos_map', 'normals', 'albedo', 'uv_map', 'depth', 'nocs']:
            if prediction_type in self.cfg.model.prediction_type:
                if prediction_type in ['pos_map', 'normals', 'albedo', 'nocs']:
                    self.prediction_dim += 3
                    if prediction_type in ['pos_map', 'normals'] and self.cfg.model.pred_disentangled:
                        self.prediction_dim += 3
                elif prediction_type == 'uv_map':
                    self.prediction_dim += 2
                    if self.cfg.model.pred_disentangled:
                        self.prediction_dim += 2
                elif prediction_type in ['depth', 'depth_si']:
                    self.prediction_dim += 1
        self.pred_disentangled = self.cfg.model.pred_disentangled

        self.t_conv1 = nn.ConvTranspose2d(embedding_dim, embedding_dim, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(embedding_dim, embedding_dim, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(embedding_dim, embedding_dim, 2, stride=2)

        if self.cfg.model.conv_dec:
            remaining_patch_size = 2
        elif self.cfg.model.feature_map_type == 'DINO':
            remaining_patch_size = 16
        else:
            remaining_patch_size = 8

        self.patch_size = remaining_patch_size
        self.token_2_patch_content = nn.Linear(embedding_dim, remaining_patch_size ** 2 * self.prediction_dim)

        if self.cfg.model.pred_conf:
            self.t_conv3_conf = nn.ConvTranspose2d(embedding_dim, embedding_dim, 2, stride=2)
            self.token_2_patch_conf = nn.Linear(embedding_dim, remaining_patch_size ** 2 * 1)

        self.n_facial_components = 0

    def forward(self, batch):
        B, N, H, W, C = batch['tar_rgb'].shape
        n_views_sel = N

        facial_components = None
        _inps = batch['tar_rgb'][:, :n_views_sel].reshape(B * n_views_sel, H, W, C)
        _inps = torch.einsum('bhwc->bchw', _inps)

        if self.cfg.model.feature_map_type == 'sapiens':
            if self.cfg.model.finetune_backbone:
                _inps = self.bicubic_up(_inps)
                img_feats = self.img_encoder(_inps)
            else:
                with torch.no_grad():
                    _inps = self.bicubic_up(_inps)
                    img_feats = self.img_encoder(_inps)

        elif self.cfg.model.feature_map_type == 'DINO':
            if self.cfg.model.finetune_backbone:
                img_feats = torch.einsum('blc->bcl', self.img_encoder(_inps))
            else:
                with torch.no_grad():
                    img_feats = torch.einsum('blc->bcl', self.img_encoder(_inps))
            token_size = int(np.sqrt(H * W / img_feats.shape[-1]))
            img_feats = img_feats.reshape(*img_feats.shape[:2], H // token_size, W // token_size)

        elif self.cfg.model.feature_map_type == 'FaRL':
            if self.cfg.model.finetune_backbone:
                img_feats, facial_components = self.img_encoder(_inps, facial_components=self.facial_components)
            else:
                with torch.no_grad():
                    img_feats, facial_components = self.img_encoder(_inps, facial_components=self.facial_components)
            token_size = int(np.sqrt(224 * 224 / img_feats.shape[-1]))

        if self.cfg.model.use_pos_enc:
            img_feats = img_feats + self.patch_pos_enc

        img_feats = img_feats.reshape(B, N, img_feats.shape[1], img_feats.shape[2], img_feats.shape[3])

        if self.cfg.n_views > 1:
            img_feats = torch.cat((img_feats,
                                   self.view_embed[:, :n_views_sel].repeat(B, 1, 1, img_feats.shape[-2],
                                                                           img_feats.shape[-1])), dim=2)

        img_feats, facial_components = self.vol_decoder(img_feats, facial_components=facial_components)

        out_dict = {}
        conf = None

        img_feats = img_feats.reshape(-1, img_feats.shape[2], img_feats.shape[3], img_feats.shape[4])

        if self.cfg.model.conv_dec:
            if self.cfg.model.feature_map_type == 'DINO':
                img_feats = F.gelu(self.t_conv1(img_feats, output_size=(64, 64)))
            img_feats = F.gelu(self.t_conv2(img_feats, output_size=(128, 128)))
            if self.cfg.model.pred_conf:
                conf_feats = F.gelu(self.t_conv3_conf(img_feats, output_size=(256, 256)))
            img_feats = F.gelu(self.t_conv3(img_feats, output_size=(256, 256)))

        img_feats = img_feats.permute(0, 2, 3, 1)
        img_feats = img_feats.reshape(img_feats.shape[0], -1, img_feats.shape[-1])
        img_feats = self.token_2_patch_content(img_feats)
        img = unpatchify(img_feats, batch_size=B, channels=self.prediction_dim, patch_size=self.patch_size,
                         n_views=n_views_sel)

        if self.cfg.model.pred_conf:
            conf_feats = conf_feats.permute(0, 2, 3, 1)
            conf_feats = conf_feats.reshape(img_feats.shape[0], -1, conf_feats.shape[-1])
            conf_feats = self.token_2_patch_conf(conf_feats)
            conf = unpatchify(conf_feats, batch_size=B, channels=1, patch_size=self.patch_size,
                              n_views=n_views_sel)

        cur_dim = 0
        if 'pos_map' in self.cfg.model.prediction_type:
            out_dict['pos_map'] = img[:, :, cur_dim:cur_dim + 3, ...]
            cur_dim += 3
            if self.pred_disentangled:
                out_dict['pos_map_can'] = img[:, :, cur_dim:cur_dim + 3, ...]
                cur_dim += 3
        if 'uv_map' in self.cfg.model.prediction_type:
            out_dict['uv_map'] = img[:, :, cur_dim:cur_dim + 2, ...]
            cur_dim += 2
            if self.pred_disentangled:
                out_dict['disps'] = img[:, :, cur_dim:cur_dim + 2, ...]
                cur_dim += 2
        if 'normals' in self.cfg.model.prediction_type:
            out_dict['normals'] = img[:, :, cur_dim:cur_dim + 3, ...]
            cur_dim += 3
            if self.pred_disentangled:
                out_dict['normals_can'] = img[:, :, cur_dim:cur_dim + 3, ...]
                cur_dim += 3
        if 'albedo' in self.cfg.model.prediction_type:
            out_dict['albedo'] = img[:, :, cur_dim:cur_dim + 3, ...]
            cur_dim += 3
        if 'nocs' in self.cfg.model.prediction_type:
            out_dict['nocs'] = img[:, :, cur_dim:cur_dim + 3, ...]
            cur_dim += 3

        return out_dict, conf


class pixel3dmm(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.net = Network(cfg)

