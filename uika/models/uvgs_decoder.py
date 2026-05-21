import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

os.environ["PYOPENGL_PLATFORM"] = "egl"

from PIL import Image
from pytorch3d.transforms import matrix_to_quaternion
from uika.models.rendering.utils.typing import *
from uika.models.rendering.gaussian_model import GaussianModel, inverse_sigmoid, trunc_exp


RE_PROJ_DIM: int = 4  # rgba -> 4 channels


def get_activation(name):
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "lin2srgb":
        return lambda x: torch.where(
            x > 0.0031308,
            torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "exp":
        return lambda x: torch.exp(x)
    elif name == "shifted_exp":
        return lambda x: torch.exp(x - 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name == "shifted_trunc_exp":
        return lambda x: trunc_exp(x - 1.0)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: F.softplus(x - 1.0)
    elif name == "scale_-11_01":
        return lambda x: x * 0.5 + 0.5
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_neurons: int,
        n_hidden_layers: int,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        bias: bool = True,
    ):
        super().__init__()
        layers = [
            self.make_linear(dim_in, n_neurons, is_first=True, is_last=False, bias=bias),
            self.make_activation(activation),
        ]
        for i in range(n_hidden_layers - 1):
            layers += [
                self.make_linear(n_neurons, n_neurons, is_first=False, is_last=False, bias=bias),
                self.make_activation(activation),
            ]
        layers += [
            self.make_linear(n_neurons, dim_out, is_first=False, is_last=True, bias=bias)
        ]
        self.layers = nn.Sequential(*layers)
        self.output_activation = get_activation(output_activation)

    def forward(self, x):
        x = self.layers(x)
        x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last, bias=True):
        layer = nn.Linear(dim_in, dim_out, bias=bias)
        return layer

    def make_activation(self, activation):
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "silu":
            return nn.SiLU(inplace=True)
        else:
            raise NotImplementedError


class GSHead(nn.Module):
    def __init__(
        self,
        in_channels,
        uv_attr_map_size,
        uv_valid_mask_flatten,
        use_rgb=True,
        pred_res=True,
        clip_scaling=0.2,
        init_scaling=-5.0,
        init_density=0.1,
        gs_type: Literal['3dgs', '2dgs'] = '3dgs',
        rot_type: Literal['quat', 'angle'] = 'quat',
        scale_sphere=False,
        sh_degree=None,
        xyz_offset=True,
        restrict_offset=True,
        xyz_offset_max_step=None,
        fix_opacity=False,
        fix_rotation=False,
    ):
        super().__init__()
        self.clip_scaling = clip_scaling
        self.use_rgb = use_rgb
        self.restrict_offset = restrict_offset
        self.xyz_offset = xyz_offset
        self.xyz_offset_max_step = xyz_offset_max_step  # 1.2 / 32
        self.fix_opacity = fix_opacity
        self.fix_rotation = fix_rotation
        self.scale_sphere = scale_sphere
        self.gs_type = gs_type
        self.rot_type = rot_type
        self.pred_res = pred_res
        self.uv_valid_mask_flatten = uv_valid_mask_flatten
        self.uv_attr_map_size = uv_attr_map_size
        
        self.attr_dict = {
            "xyz": 3,
            "opacity": None,
            "shs": None,
            "scaling": None,
            "rotation": None,
        }
        if use_rgb:
            self.attr_dict["shs"] = 4  # rgb + pred weight
        else:
            self.attr_dict["shs"] = (sh_degree + 1) ** 2 * 3
        if scale_sphere:
            self.attr_dict['scaling'] = 1
        else:
            if self.gs_type == '2dgs':
                self.attr_dict['scaling'] = 2
            else:
                self.attr_dict['scaling'] = 3
        if not self.fix_opacity:
            self.attr_dict["opacity"] = 1
        if not self.fix_rotation:
            if self.rot_type == 'quat':
                self.attr_dict["rotation"] = 4
            elif self.rot_type == 'angle':
                self.attr_dict["rotation"] = 3

        self.out_layers = nn.ModuleDict()
        for key, out_ch in self.attr_dict.items():
            if out_ch is None:
                layer = nn.Identity()
            else:
                if key == 'shs' and pred_res:
                    layer = nn.Linear(in_channels + RE_PROJ_DIM, out_ch)
                else:
                    layer = nn.Linear(in_channels, out_ch)
            
            # initialize
            if not (key == "shs" and use_rgb):
                if key == "opacity" and self.fix_opacity:
                    pass
                elif key == "rotation" and self.fix_rotation:
                    pass
                else:
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                nn.init.constant_(layer.bias, init_scaling)
            elif key == "rotation":
                if not self.fix_rotation:
                    nn.init.constant_(layer.bias, 0)
                    nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                if not self.fix_opacity:
                    nn.init.constant_(layer.bias, inverse_sigmoid(init_density))
            self.out_layers[key] = layer
    
    def forward(
        self,
        x: Float[Tensor, "B N D"],
        pts: Float[Tensor, "B N 3"],
        colors: Float[Tensor, "B N RE_PROJ_DIM"],
    ) -> List[GaussianModel]:
        
        bs = x.shape[0]
        ret = {}
        gs_list = []

        opacity_temp = None
        
        for k in self.attr_dict:
            # forward
            layer = self.out_layers[k]
            if k == 'shs' and self.pred_res:
                v = layer(torch.cat([x, colors], dim=-1))
                # v = colors + v
            else:
                v = layer(x)
            
            # activation
            if k == "rotation":
                if self.fix_rotation:
                    v = matrix_to_quaternion(torch.eye(3).type_as(x)[None].repeat(x.shape[0], 1, 1))  # constant rotation
                else:
                    v = F.normalize(v, dim=-1)
            elif k == "scaling":
                v = trunc_exp(v)
                if self.scale_sphere:
                    assert v.shape[-1] == 1
                    v = torch.cat([v, v, v], dim=-1)
                if self.clip_scaling is not None:
                    v = torch.clamp(v, min=0, max=self.clip_scaling)
            elif k == "opacity":
                if self.fix_opacity:
                    v = torch.ones_like(x)[..., 0:1]
                else:
                    v = torch.sigmoid(v)
                    opacity_temp = v
            elif k == "shs":
                assert v.shape[-1] == 4, "use_rgb must be True!"
                v = torch.sigmoid(v)
                fuse_w = v[..., 3:]
                # ----- Visualization weight here -----
                VIZ_UV: bool = False
                if VIZ_UV:
                    save_dir = './debug_vis/uv_fuse_weight/'
                    os.makedirs(save_dir, exist_ok=True)
                    num_png = os.listdir(save_dir)
                    save_name = os.path.join(save_dir, f'{len(num_png):04d}.png')

                    def _seq_to_grid(tensor: torch.Tensor):
                        b, _, channels = tensor.shape
                        assert b == 1, 'only support batch size = 1 for visualization during inference'
                        assert channels == 1 or channels == 3
                        grid_tensor = torch.zeros(b, self.uv_attr_map_size, self.uv_attr_map_size, channels).type_as(tensor)
                        grid_tensor = grid_tensor.view(b, -1, channels)
                        grid_tensor[:, self.uv_valid_mask_flatten, :] = tensor
                        grid_tensor = grid_tensor.view(b, self.uv_attr_map_size, self.uv_attr_map_size, channels)
                        if channels == 1:
                            grid_tensor = grid_tensor.repeat(1, 1, 1, 3)
                        return grid_tensor

                    # viz reproj color / pred color / final color / reproj conf / pred weight / opacity
                    input_list = [colors[..., :3], v[..., :3],
                                    fuse_w * v[..., :3] + (1 - fuse_w) * colors[..., :3],
                                    colors[..., 3:], fuse_w, opacity_temp]
                    ret_list = []
                    for inp in input_list:
                        grid_inp = _seq_to_grid(inp.detach().cpu())
                        grid_inp = (grid_inp[0].numpy() * 255).astype(np.uint8)
                        ret_list.append(grid_inp)
                    # ret_list.append(np.zeros_like(ret_list[0]))  # black separator
                    concat_img = np.concatenate(ret_list, axis=1)
                    concat_img = np.concatenate([concat_img[:, :self.uv_attr_map_size * 3], concat_img[:, self.uv_attr_map_size * 3:]], axis=0)
                    img_pil = Image.fromarray(concat_img)
                    img_pil.save(save_name)
                # -------------------------------------
                v = fuse_w * v[..., :3] + (1 - fuse_w) * colors[..., :3]
            elif k == "xyz":
                if self.restrict_offset:
                    max_step = self.xyz_offset_max_step
                    v = (torch.sigmoid(v) - 0.5) * max_step
                assert self.xyz_offset
                ret["offset"] = v
                v = pts + v
            ret[k] = v
        
        for b in range(bs):
            gs_dict = {k: v[b] for k, v in ret.items()}
            gs_list.append(GaussianModel(**gs_dict))

        return gs_list


class UV_DPT(nn.Module):
    def __init__(
        self,
        dim_in: int = 1024,
        final_output_dim: int = 256,
        hidden_dims: int = 256,
        out_dims: List[int] = [256, 512, 1024, 1024],
        use_reproj_rgb: bool = True,
    ):
        super().__init__()
        self.use_reproj_rgb = use_reproj_rgb
        exrtra_dim = RE_PROJ_DIM if use_reproj_rgb else 0

        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, out_dim, kernel_size=1, stride=1, padding=0) for out_dim in out_dims
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_dims[0], out_dims[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_dims[1], out_dims[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_dims[3], out_dims[3], kernel_size=3, stride=2, padding=1)
        ])
        self.layer_rn = nn.ModuleList([
            nn.Conv2d(out_dims[0] + exrtra_dim, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(out_dims[1] + exrtra_dim, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(out_dims[2] + exrtra_dim, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(out_dims[3] + exrtra_dim, hidden_dims, kernel_size=3, stride=1, padding=1, bias=False),
        ])
        self.refinenet = nn.ModuleList([
            FeatureFusionBlock(features=hidden_dims, activation=nn.ReLU(False), fuse=False),
            FeatureFusionBlock(features=hidden_dims, activation=nn.ReLU(False), fuse=True),
            FeatureFusionBlock(features=hidden_dims, activation=nn.ReLU(False), fuse=True),
            FeatureFusionBlock(features=hidden_dims, activation=nn.ReLU(False), fuse=True),
        ])
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dims + exrtra_dim, hidden_dims, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dims, final_output_dim, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(
        self,
        token_list: List[torch.Tensor],  # [B, L, D] * 4
        uv_token_size: Literal[64, 96, 128],
        reproj_rgb: torch.Tensor,  # [B, RE_PROJ_DIM, H, W]
    ) -> torch.Tensor:
        
        out_features = []
        for i, feature in enumerate(token_list):
            feature = feature.permute(0, 2, 1).reshape(
                (feature.shape[0], feature.shape[-1], uv_token_size, uv_token_size)
            ).contiguous()
            feature = self.projects[i](feature)
            feature = self.resize_layers[i](feature)
            
            if self.use_reproj_rgb:
                feature = torch.cat([
                        feature,
                        T.functional.resize(reproj_rgb, (feature.shape[-2], feature.shape[-1]), antialias=True).detach(),
                    ], dim=1
                )
            out_features.append(feature)
        
        layer_rns = []
        for i, feature in enumerate(out_features):
            layer_rns.append(self.layer_rn[i](feature))
        
        path_ = self.refinenet[0](layer_rns[3], size=layer_rns[2].shape[2:])
        path_ = self.refinenet[1](path_, layer_rns[2], size=layer_rns[1].shape[2:])
        path_ = self.refinenet[2](path_, layer_rns[1], size=layer_rns[0].shape[2:])
        path_ = self.refinenet[3](path_, layer_rns[0], size=layer_rns[0].shape[2:])
        
        if self.use_reproj_rgb:
            path_ = torch.cat([path_, reproj_rgb], dim=1)
        out = self.output_conv(path_)
        return out


class UVGSDecoder(nn.Module):
    def __init__(
        self,
        uv_valid_mask_flatten: torch.Tensor,  # [uv_map_res^2,]
        uv_token_size: Literal[64, 96, 128] = 64,
        uv_token_dim: int = 1024,
        uv_attr_map_size: Literal[256, 384, 512] = 256,
        inner_dim: int = 256,
        decode_dim: int = 512,
        use_reproj_rgb: bool = True,
        use_rgb: bool = True,
        sh_degree: int = 0,
        xyz_offset_max_step: float = 0.2,
        mlp_network_config=None,
        clip_scaling=0.2,
        scale_sphere=False,
        fix_opacity=False,
        fix_rotation=False,
        gs_type: Literal['3dgs', '2dgs'] = '3dgs',
        rot_type: Literal['quat', 'angle'] = 'quat',
    ):
        super().__init__()
        self.uv_valid_mask_flatten = uv_valid_mask_flatten
        assert uv_valid_mask_flatten.shape[0] == uv_attr_map_size * uv_attr_map_size
        self.feat_upsample_ratio = uv_attr_map_size // uv_token_size
        assert self.feat_upsample_ratio in [2, 4], "feat_upsample_ratio must be 2 or 4"

        self.uv_token_size = uv_token_size
        self.uv_token_dim = uv_token_dim
        self.uv_attr_map_size = uv_attr_map_size
        print("==="*16*3, "\n UV token size: ", self.uv_token_size, "\n"+"==="*16*3)
        print("UV token dim: ", self.uv_token_dim, "\n"+"==="*16*3)
        print("UV attr map size: ", self.uv_attr_map_size, "\n"+"==="*16*3)

        self.use_reproj_rgb = use_reproj_rgb
        self.use_rgb = use_rgb
        self.sh_degree = 0 if use_rgb else sh_degree
        
        self.dpt_decoder = UV_DPT(
            dim_in=uv_token_dim,
            final_output_dim=inner_dim,
            use_reproj_rgb=use_reproj_rgb,
        )

        assert mlp_network_config is not None
        self.mlp_net = MLP(inner_dim, decode_dim, **mlp_network_config)

        self.gs_net = GSHead(
            in_channels=decode_dim,
            uv_attr_map_size=uv_attr_map_size,
            uv_valid_mask_flatten=uv_valid_mask_flatten,
            use_rgb=use_rgb,
            pred_res=use_reproj_rgb,
            sh_degree=self.sh_degree,
            clip_scaling=clip_scaling,
            scale_sphere=scale_sphere,
            init_scaling=-5.0,
            init_density=0.1,
            xyz_offset=True,
            restrict_offset=True,
            xyz_offset_max_step=xyz_offset_max_step,
            fix_opacity=fix_opacity,
            fix_rotation=fix_rotation,
            gs_type=gs_type,
            rot_type=rot_type,
        )
    
    def forward(
            self,
            uv_token_list: List[torch.Tensor],  # [B, L, D] * 4
            vert_base_pos: Float[Tensor, "B Np_q 3"],
            re_proj_rgb: Float[Tensor, "B RE_PROJ_DIM H W"],
    ) -> List[GaussianModel]:
        
        uv_map_feat = self.dpt_decoder(
            token_list=uv_token_list,
            uv_token_size=self.uv_token_size,
            reproj_rgb=re_proj_rgb,
        )  # [B, inner_dim, uv_attr_map_size, uv_attr_map_size]

        valid_uv_feat = uv_map_feat.view(uv_map_feat.shape[0], uv_map_feat.shape[1], -1)[:, :, self.uv_valid_mask_flatten].permute(0, 2, 1)  # [B, N, inner_dim]
        valid_reproj_rgb = re_proj_rgb.view(re_proj_rgb.shape[0], re_proj_rgb.shape[1], -1)[:, :, self.uv_valid_mask_flatten].permute(0, 2, 1)  # [B, N, RE_PROJ_DIM]

        gs_feat = self.mlp_net(valid_uv_feat)  # [B, N, decode_dim]
        gs_list = self.gs_net(x=gs_feat, pts=vert_base_pos, colors=valid_reproj_rgb.detach())

        return gs_list


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups=1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, activation, fuse=True, deconv=False, bn=False, expand=False, align_corners=False, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1
        self.expand = expand
        out_features = features
        if self.expand:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        if fuse:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()
        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}
        output = F.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output


if __name__ == "__main__":
    import cv2
    from accelerate.utils import set_seed
    from uika.datasets.mv_video_head import MV_VideoHeadDataset
    from uika.models.rendering.uvgs_renderer import UVGSRenderer

    def get_flame_params(data):
        flame_params = {}        
        flame_keys = ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose',\
                      'lhand_pose', 'rhand_pose', 'expr', 'trans', 'betas',\
                      'rotation', 'neck_pose', 'eyes_pose', 'translation']
        for k, v in data.items():
            if k in flame_keys:
                # print(k, v.shape)
                flame_params[k] = data[k]
        return flame_params

    set_seed(1234)
    human_model_path = "./model_zoo/human_parametric_models"
    device = "cuda:0"
    os.makedirs("./debug_vis/gs_render", exist_ok=True)

    # root_dir = "./train_data/nersemble_v2/export"
    # meta_path = "./train_data/nersemble_v2/label/local_total_ids.json"
    root_dir = "./train_data/synth_mv/export"
    meta_path = "./train_data/synth_mv/label/local_total_ids.json"
    
    dataset = MV_VideoHeadDataset(
        root_dirs=root_dir, meta_path=meta_path, sample_side_views=7,
        render_image_res_low=512, render_image_res_high=512,
        render_region_size=(512, 512), source_image_res=512,
        enlarge_ratio=[0.8, 1.2],
        debug=False, is_val=False
    )

    data = dataset[0]
    flame_data = get_flame_params(data)
    flame_data_tmp = {}
    for k, v in flame_data.items():
        flame_data_tmp[k] = v.unsqueeze(0).to(device)
        print(k, v.shape)
    flame_data = flame_data_tmp
    
    c2ws = data["c2ws"].unsqueeze(0).to(device)
    intrs = data["intrs"].unsqueeze(0).to(device)
    render_images = data["render_image"].numpy()
    render_h = data["render_full_resolutions"][0, 0]
    render_w= data["render_full_resolutions"][0, 1]
    render_bg_colors = data["render_bg_colors"].unsqueeze(0).to(device)
    print("c2ws", c2ws.shape, "intrs", intrs.shape, intrs)

    uv_map_size: int = 256
    uv_token_size: int = 64
    uv_token_dim: int = 1024
    mlp_network_config = {
        'n_neurons': 512,
        'n_hidden_layers': 2,
        'activation': 'silu',
    }
    
    gs_renderer = UVGSRenderer(
        uv_attr_map_size=uv_map_size,
        human_model_path=human_model_path,
        use_rgb=True,
        sh_degree=0,
        shape_param_dim=300,
        expr_param_dim=100,
        add_teeth=False,
        teeth_bs_flag=False,
        oral_mesh_flag=False,
        gs_type='3dgs',
    ).to(device)

    gs_decoder = UVGSDecoder(
        gs_renderer.uv_valid_mask_flatten,
        uv_token_size=uv_token_size,
        uv_token_dim=uv_token_dim,
        uv_attr_map_size=uv_map_size,
        inner_dim=256,
        decode_dim=512,
        use_rgb=True,
        sh_degree=0,
        xyz_offset_max_step=0.2,
        clip_scaling=0.01,
        mlp_network_config=mlp_network_config,
        scale_sphere=False,
        fix_opacity=False,
        fix_rotation=False,
        gs_type='3dgs',
        rot_type='quat',
    ).to(device)
    
    v_cano = gs_renderer.get_shaped_cano_verts(flame_data['betas'], torch.device(device))  # [1, N, 3]
    gs_list = gs_decoder(
        [torch.randn((1, uv_token_size*uv_token_size, uv_token_dim)).float().to(device) for _ in range(4)],
        v_cano,
        torch.randn((1, RE_PROJ_DIM, uv_map_size, uv_map_size)).float().to(device)
    )
    out = gs_renderer(
        gs_list,
        flame_data=flame_data,
        c2w=c2ws,
        intrinsic=intrs,
        height=render_h,
        width=render_w,
        background_color=render_bg_colors,
    )

    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            if k == "comp_rgb_bg":
                print("comp_rgb_bg", v)
                continue
            for b_idx in range(len(v)):
                if k == "3dgs":
                    for v_idx in range(len(v[b_idx])):
                        v[b_idx][v_idx].save_ply(f"./debug_vis/gs_render/{b_idx}_{v_idx}.ply")
                    continue
                for v_idx in range(v.shape[1]):
                    save_path = os.path.join("./debug_vis/gs_render", f"{b_idx}_{v_idx}_{k}.jpg")
                    if "normal" in k:
                        img = ((v[b_idx, v_idx].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2. * 255).astype(np.uint8)
                    else:
                        img = (v[b_idx, v_idx].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                    print(v[b_idx, v_idx].shape, img.shape, save_path)
                    if "mask" in k:
                        render_img = render_images[v_idx].transpose(1, 2, 0) * 255
                        blend_img = (render_images[v_idx].transpose(1, 2, 0) * 255 * 0.5 + np.tile(img, (1, 1, 3)) * 0.5).clip(0, 255).astype(np.uint8)
                        cv2.imwrite(save_path, np.hstack([np.tile(img, (1, 1, 3)), render_img.astype(np.uint8), blend_img])[:, :, (2, 1, 0)])
                    else:
                        print(save_path, k)
                        cv2.imwrite(save_path, img)
