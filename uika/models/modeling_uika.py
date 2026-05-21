import torch
import torch.nn as nn

from diffusers.utils import is_torch_version

from uika.models.mm_transformer import MMTransformer
from uika.models.uvgs_decoder import UVGSDecoder
from uika.models.rendering.uvgs_renderer import UVGSRenderer
from uika.models.rendering.utils.typing import *
from uika.models.uv_modules import uv_reproject
from uika.models.modeling_fuvt import FUVTWrapper


class ModelUIKA(nn.Module):
    """
    Full model of the basic arbitrary view large reconstruction model.
    """
    def __init__(
        self,
        # transformer
        transformer_dim: int = 1024,
        transformer_layers: int = 12,
        transformer_heads: int = 16,
        tf_grad_ckpt: bool = True,
        encoder_grad_ckpt: bool = True,

        # image encoder
        encoder_freeze: bool = True,
        encoder_pretrained: bool = True,
        encoder_type: str = 'dinov3_fusion',
        encoder_model_name: str = 'dinov3_vitl16',
        encoder_feat_dim: int = 1024,

        # uv setting
        uv_token_dim: int = 1024,
        uv_token_size: Literal[64, 96, 128] = 96,
        uv_attr_map_size: Literal[256, 384, 512] = 384,
        fuvt_ckpt_path: str | None = 'model_zoo/uv_modules/fuvt_15k.safetensors',
        fuvt_load_pretrained: bool = True,
        fuvt_pretrained_patch_embed: bool = True,

        # gs decoder
        cano_shape: bool = True,
        uv_dpt_inner_dim: int = 256,
        gs_decode_dim: int = 512,
        gs_xyz_offset_max_step: float = 0.2,
        gs_clip_scaling: float = 0.01,
        gs_mlp_network_config=None,
        scale_sphere: bool = False,
        fix_opacity: bool = False,
        fix_rotation: bool = False,
        gs_type: Literal['3dgs', '2dgs'] = '3dgs',
        rot_type: Literal['quat', 'angle'] = 'quat',

        # gs renderer
        human_model_path: str = "./model_zoo/human_parametric_models",
        gs_use_rgb: bool = True,
        gs_sh: int = 0,
        shape_param_dim: int = 300,
        expr_param_dim: int = 100,
        add_teeth: bool = False,
        teeth_bs_flag: bool = False,
        oral_mesh_flag: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.gradient_checkpointing = tf_grad_ckpt
        self.encoder_gradient_checkpointing = encoder_grad_ckpt
        self.encoder_feat_dim = encoder_feat_dim
        self.uv_token_size = uv_token_size
        self.uv_token_dim = uv_token_dim
        self.uv_attr_map_size = uv_attr_map_size
        self.cano_shape = cano_shape

        # image encoder
        self.encoder = self._encoder_fn(encoder_type)(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
            pretrained=encoder_pretrained,
            encoder_feat_dim=encoder_feat_dim,
            dual_fusion=True,
            use_clstoken=False,
        )

        # uv wrapper
        self.uv_wrapper = FUVTWrapper(
            ckpt_path=fuvt_ckpt_path,
            load_pretrained=fuvt_load_pretrained,
            pretrained_patch_embed=fuvt_pretrained_patch_embed,
        )

        # transformer
        self.transformer = MMTransformer(
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            inner_dim=transformer_dim,
            cond_dim=encoder_feat_dim,
            cond_dim2=encoder_feat_dim,
            gradient_checkpointing=tf_grad_ckpt,
            use_dual_attention=True,
            uv_token_size=uv_token_size,
        )
        
        # renderer
        self.renderer = UVGSRenderer(
            uv_attr_map_size=uv_attr_map_size,
            human_model_path=human_model_path,
            use_rgb=gs_use_rgb,
            sh_degree=gs_sh,
            shape_param_dim=shape_param_dim,
            expr_param_dim=expr_param_dim,
            add_teeth=add_teeth,
            teeth_bs_flag=teeth_bs_flag,
            oral_mesh_flag=oral_mesh_flag,
            gs_type=gs_type,
        )

        # decoder
        self.gs_decoder = UVGSDecoder(
            uv_valid_mask_flatten=self.renderer.uv_valid_mask_flatten,
            uv_token_size=uv_token_size,
            uv_token_dim=uv_token_dim,
            uv_attr_map_size=uv_attr_map_size,
            inner_dim=uv_dpt_inner_dim,
            decode_dim=gs_decode_dim,
            use_reproj_rgb=True,
            use_rgb=gs_use_rgb,
            sh_degree=gs_sh,
            xyz_offset_max_step=gs_xyz_offset_max_step,
            clip_scaling=gs_clip_scaling,
            mlp_network_config=gs_mlp_network_config,
            scale_sphere=scale_sphere,
            fix_opacity=fix_opacity,
            fix_rotation=fix_rotation,
            gs_type=gs_type,
            rot_type=rot_type,
        )
    
    @staticmethod
    def _encoder_fn(encoder_type: Literal['dinov2_fusion', 'dinov3_fusion']):
        if encoder_type == 'dinov2_fusion':
            from .encoders.dino_fusion_wrapper import Dinov2FusionWrapper
            return Dinov2FusionWrapper
        else:
            from .encoders.dino_fusion_wrapper import Dinov3FusionWrapper
            return Dinov3FusionWrapper
    
    def forward_encode_image(self, image: torch.Tensor, flag_screen_space: bool = True):
        # encode image
        if self.training and self.encoder_gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            image_feats = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.encoder),
                image,
                flag_screen_space,
                **ckpt_kwargs,
            )
        else:
            image_feats = self.encoder(image, flag_screen_space)
        return image_feats

    def forward_uv_projection(
        self,
        image: torch.Tensor,  # [B, N_ref, C_img, H_img, W_img]
        mask: torch.Tensor,  # [B, N_ref, 1, H_img, W_img]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pixel_uv = self.uv_wrapper(image)  # [B, V, 2, H, W], value range: 0 ~ 1
        per_view_uv, aggregated_uv = uv_reproject(
            rgb_image=image,
            mask=mask,
            uv_map=pixel_uv,
            per_view_uv_size=int(image.shape[-1]),
            aggregated_uv_size=self.uv_attr_map_size,
        )  # [B, V, 3, H, W] and [B, 4, uv_map_size, uv_map_size]

        return per_view_uv, aggregated_uv
    
    @torch.compile
    def forward_uv_token(
        self,
        image: torch.Tensor,  # [B, N_ref, C_img, H_img, W_img]
        per_view_reproj_rgb: torch.Tensor,  # [B, N_ref, C_img, H_img, W_img]
    ) -> List[torch.Tensor]:  # [B, L, D] * 4
        B, V, C, H, W = image.shape
        
        # encode image
        image_feats = self.forward_encode_image(image.view(B*V, C, H, W), True)  # [B*V, h*w, D]
        image_feats = image_feats.view(B, -1, self.encoder_feat_dim)  # [B, V*h*w, D]
        
        uv_feats = self.forward_encode_image(per_view_reproj_rgb.view(B*V, C, H, W), False)  # [B*V, h*w, D]
        uv_feats = uv_feats.view(B, -1, self.encoder_feat_dim)  # [B, V*h*w, D]
        cond2 = uv_feats.to(image_feats.dtype)

        x_list = self.transformer(cond=image_feats, cond2=cond2)
        return x_list

    def forward(
        self, image, mask, render_c2ws, render_intrs, render_bg_colors,
        flame_params,
        render_h: Optional[int] = None, render_w: Optional[int] = None,
    ):
        assert image.shape[0] == mask.shape[0], "Batch size mismatch for image and mask"
        assert image.shape[0] == render_c2ws.shape[0], "Batch size mismatch for image and render_c2ws"
        assert image.shape[0] == render_bg_colors.shape[0], "Batch size mismatch for image and render_bg_colors"
        assert image.shape[0] == flame_params["betas"].shape[0], "Batch size mismatch for image and flame_params"
        assert image.shape[0] == flame_params["expr"].shape[0], "Batch size mismatch for image and flame_params"
        assert len(flame_params["betas"].shape) == 2
        
        if render_h is None or render_w is None:
            render_h, render_w = int(render_intrs[0, 0, 1, 2] * 2), int(render_intrs[0, 0, 0, 2] * 2)

        per_view_uv, aggregated_uv = self.forward_uv_projection(image, mask)

        uv_token_list = self.forward_uv_token(image, per_view_uv)

        shape = torch.zeros_like(flame_params["betas"]) if self.cano_shape else flame_params["betas"]
        flame_params = {**flame_params, "betas": shape}
        v_cano = self.renderer.get_shaped_cano_verts(shape, image.device)

        gs_list = self.gs_decoder(uv_token_list, v_cano, aggregated_uv)

        render_results = self.renderer(
            gs_model_list=gs_list,
            flame_data=flame_params,
            c2w=render_c2ws,
            intrinsic=render_intrs,
            height=render_h,
            width=render_w,
            background_color=render_bg_colors,
        )

        ret = {**render_results}
        return ret
