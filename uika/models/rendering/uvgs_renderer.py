import os
import math
import torch
import numpy as np
import torch.nn as nn

os.environ["PYOPENGL_PLATFORM"] = "egl"

from collections import defaultdict
from diff_gaussian_rasterization import GaussianRasterizationSettings as GSRS_gaussian
from diff_gaussian_rasterization import GaussianRasterizer as GSR_gaussian

from uika.models.rendering.utils.typing import *
from uika.models.rendering.camera import Camera
from uika.models.rendering.gaussian_model import GaussianModel
from uika.models.rendering.flame_model.uv_flame import UVFlameHead
from uika.models.rendering.utils.point_utils import depth_to_normal
from uika.models.rendering.utils.template_utils import get_sing_batch_smpl_data


AVATAR_SURFEL_DEPTH_RATIO = 1.0


class UVGSRenderer(nn.Module):
    def __init__(
        self,
        uv_attr_map_size: Literal[256, 384, 512] = 256,
        human_model_path: str = "./model_zoo/human_parametric_models",
        use_rgb: bool = True,
        sh_degree: int = 0,
        shape_param_dim: int = 300,
        expr_param_dim: int = 100,
        add_teeth=False,
        teeth_bs_flag=False,
        oral_mesh_flag=False,
        gs_type: Literal['3dgs', '2dgs'] = '3dgs',
    ):
        super().__init__()

        self.teeth_bs_flag = teeth_bs_flag
        self.oral_mesh_flag = oral_mesh_flag
        self.gs_type = gs_type
        self.scaling_modifier = 1.0
        self.sh_degree = sh_degree
        self.use_rgb = use_rgb
        if use_rgb:
            self.sh_degree = 0
        
        self.flame_model = UVFlameHead(
            shape_params=shape_param_dim,
            expr_params=expr_param_dim,
            uv_resolution=uv_attr_map_size,
            flame_model_path=f"{human_model_path}/flame2023.pkl",
            flame_lmk_embedding_path=f"{human_model_path}/landmark_embedding_with_eyes.npy",
            flame_template_mesh_path=f"{human_model_path}/flame_w_mouth.obj",
            flame_parts_path=f"{human_model_path}/FLAME_masks.pkl",
            include_mask=False,
            add_teeth=add_teeth,
            add_shoulder=False,
            teeth_bs_flag=teeth_bs_flag,
            oral_mesh_flag=oral_mesh_flag,
        )
    
    @property
    def uv_valid_mask_flatten(self):
        return self.flame_model.uv_valid_mask_flatten

    def get_shaped_cano_verts(
        self,
        shape: torch.Tensor,  # [B, shape_dim]
        device: torch.device
    ) -> torch.Tensor:  # [B, N, 3]
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float32):
                positions = self.flame_model.get_cano_verts(shape_params=shape)  # [B, N, 3]
        return positions
    
    def render_single_view(
            self,
            gs: GaussianModel,
            viewpoint_camera: Camera,
            background_color: Optional[Float[Tensor, "3"]],
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=self.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        bg_color = background_color
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        assert self.gs_type == '3dgs'
        GSRS = GSRS_gaussian
        GSR = GSR_gaussian

        raster_settings = GSRS(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform.float(),
            sh_degree=self.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GSR(raster_settings=raster_settings)

        means3D = gs.xyz
        means2D = screenspace_points
        opacity = gs.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = gs.scaling
        rotations = gs.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if self.use_rgb:
            colors_precomp = gs.shs.squeeze(1)
            RENDER_UV_COLOR = False
            if RENDER_UV_COLOR:
                colors_precomp = self.flame_model.uv_color.to(colors_precomp.device)
        else:
            shs = gs.shs
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        # torch.cuda.synchronize()
        # with boxx.timeit():
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            raster_ret = rasterizer(
                means3D = means3D.float(),
                means2D = means2D.float(),
                shs = shs.float() if not self.use_rgb else None,
                colors_precomp = colors_precomp.float() if colors_precomp is not None else None,
                opacities = opacity.float(),
                scales = scales.float(),
                rotations = rotations.float(),
                cov3D_precomp = cov3D_precomp
            )
        
        if self.gs_type == '3dgs':
            rendered_image, radii, rendered_depth, rendered_alpha = raster_ret
            ret = {
                "comp_rgb": rendered_image,  # [3, H, W] or [32, H, W]
                "comp_rgb_bg": bg_color,
                'comp_mask': rendered_alpha,
                'comp_depth': rendered_depth,
            }
        elif self.gs_type == '2dgs':
            rendered_image, radii, allmap = raster_ret
            ret = {
                "comp_rgb": rendered_image,  # [3, H, W]
                "comp_rgb_bg": bg_color,
            }
            # additional regularizations
            render_alpha = allmap[1:2]

            # get normal map
            # transform normal from view space to world space
            render_normal = allmap[2:5]
            render_normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_view_transform[:3, :3].T)).permute(2, 0, 1)
            
            # get median depth map
            render_depth_median = allmap[5:6]
            render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

            # get expected depth map
            render_depth_expected = allmap[0:1]
            render_depth_expected = (render_depth_expected / render_alpha)
            render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
            
            # get depth distortion map
            render_dist = allmap[6:7]

            # pseudo surface attributes
            # surf depth is either median or expected by setting depth_ratio to 1 or 0
            # for bounded scene, use median depth, i.e., depth_ratio = 1; 
            # for unbounded scene, use expected depth, i.e., depth_ratio = 0, to reduce disk aliasing.
            surf_depth = (1 - AVATAR_SURFEL_DEPTH_RATIO) * render_depth_expected + AVATAR_SURFEL_DEPTH_RATIO * render_depth_median
            
            # assume the depth points form the 'surface' and generate pseudo surface normal for regularizations.
            surf_normal = depth_to_normal(viewpoint_camera, surf_depth, self.device)
            surf_normal = surf_normal.permute(2, 0, 1)
            # remember to multiply with accum_alpha since render_normal is unnormalized.
            surf_normal = surf_normal * (render_alpha).detach()

            ret.update({
                'comp_mask': render_alpha,
                'rend_normal': render_normal,
                'surf_normal': surf_normal,
                'rend_dist': render_dist,
                'comp_depth': surf_depth,
            })
        return ret
    
    def render_single_batch(
        self,
        gs_list: List[GaussianModel],
        c2ws: Float[Tensor, "Nv 4 4"],
        intrinsics: Float[Tensor, "Nv 4 4"],
        height: int,
        width: int,
        background_color: Optional[Float[Tensor, "Nv 3"]],
    ):
        out_list = []
        self.device = gs_list[0].xyz.device

        # measure render FPS
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        for v_idx, (c2w, intrinsic) in enumerate(zip(c2ws, intrinsics)):
            # start.record()
            ret = self.render_single_view(
                gs_list[v_idx],
                Camera.from_c2w(c2w, intrinsic, height, width),
                background_color[v_idx],
            )
            # end.record()
            # torch.cuda.synchronize()
            # elapsed_time_ms = start.elapsed_time(end)
            # fps = 1000.0 / elapsed_time_ms
            # print(f"FPS: {fps:.2f} for view {v_idx}")
            out_list.append(ret)
        
        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        out = {k: torch.stack(v, dim=0) for k, v in out.items()}
        out["3dgs"] = gs_list
        return out

    def animate_gs(self, gs_attr: GaussianModel, flame_data: Dict):
        device = gs_attr.xyz.device
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            mean_3d = gs_attr.xyz  # [N, 3]
            
            num_view = flame_data["expr"].shape[0]  # [V, 100]
            mean_3d = mean_3d.unsqueeze(0).repeat(num_view, 1, 1)  # [V, N, 3]

            if self.teeth_bs_flag:
                expr = torch.cat([flame_data['expr'], flame_data['teeth_bs']], dim=-1)
            else:
                expr = flame_data["expr"]
            ret = self.flame_model(
                v_cano=mean_3d,
                shape=flame_data["betas"].repeat(num_view, 1),
                expr=expr,
                rotation=flame_data["rotation"],
                neck=flame_data["neck_pose"],
                jaw=flame_data["jaw_pose"],
                eyes=flame_data["eyes_pose"],
                translation=flame_data["translation"],
                zero_centered_at_root_node=False,
                return_landmarks=False,
                return_verts_cano=False,
                static_offset=None
            )
            mean_3d = ret["animated"]
            
        gs_attr_list = []
        for i in range(num_view):
            gs_attr_copy = GaussianModel(
                xyz=mean_3d[i],             # [N, 3]
                opacity=gs_attr.opacity,    # [N, 1]
                rotation=gs_attr.rotation,  # [N, 4]
                scaling=gs_attr.scaling,    # [N, 3]
                shs=gs_attr.shs,            # [N, 3]
                offset=gs_attr.offset,      # [N, 3]
            )
            gs_attr_list.append(gs_attr_copy)
        return gs_attr_list

    def forward(
        self,
        gs_model_list: List[GaussianModel],
        flame_data,  # e.g., body_pose:[B, Nv, 21, 3], betas:[B, 100]
        c2w: Float[Tensor, "B Nv 4 4"],
        intrinsic: Float[Tensor, "B Nv 4 4"],
        height: int,
        width: int,
        background_color: Optional[Float[Tensor, "B Nv 3"]] = None,
    ):
        batch_size = len(gs_model_list)
        out_list = []

        for b in range(batch_size):
            gs_model = gs_model_list[b]

            animatable_gs_model_list: list[GaussianModel] =\
                self.animate_gs(gs_model, get_sing_batch_smpl_data(flame_data, b))
            
            assert len(animatable_gs_model_list) == c2w.shape[1]
            out_list.append(
                self.render_single_batch(
                    animatable_gs_model_list,
                    c2w[b],
                    intrinsic[b],
                    height,
                    width,
                    background_color[b] if background_color is not None else None,
                )
            )
        
        out = defaultdict(list)
        out['cano_gs_lst'] = gs_model_list
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        for k, v in out.items():
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.stack(v, dim=0)
            else:
                out[k] = v
        
        return out


if __name__ == "__main__":
    import cv2
    from accelerate.utils import set_seed
    from uika.datasets.mv_video_head import MV_VideoHeadDataset
    from uika.models.uvgs_decoder import UVGSDecoder

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
        torch.randn((1, 4, uv_map_size, uv_map_size)).float().to(device)
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
