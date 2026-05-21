# Code heavily inspired by https://github.com/HavenFeng/photometric_optimization/blob/master/models/FLAME.py.
# Please consider citing their work if you find this code useful. The code is subject to the license available via
# https://github.com/vchoutas/flame/edit/master/LICENSE

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


import torch
import numpy as np

from PIL import Image
from typing import Literal

from uika.models.rendering.flame_model.base_flame import FlameHead
from uika.models.rendering.utils.uv_utils import Pytorch3dRasterizer
from uika.models.rendering.flame_model.lbs import batch_rigid_transform, batch_rodrigues
from uika.models.rendering.flame_model.lbs import vertices2landmarks, blend_shapes, vertices2joints


class UVFlameHead(FlameHead):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(
        self,
        shape_params: int = 300,
        expr_params: int = 100,
        uv_resolution: Literal[256, 384, 512] = 256,
        flame_model_path=None,
        flame_lmk_embedding_path=None,
        flame_template_mesh_path=None,
        flame_parts_path=None,
        include_mask=True,
        add_teeth=False,
        add_shoulder=False,
        teeth_bs_flag = False,
        oral_mesh_flag = False,
    ):
        super().__init__(
            shape_params=shape_params,
            expr_params=expr_params,
            flame_model_path=flame_model_path,
            flame_lmk_embedding_path=flame_lmk_embedding_path,
            flame_template_mesh_path=flame_template_mesh_path,
            include_mask=include_mask,
            add_teeth=add_teeth,
            add_shoulder=add_shoulder,
            flame_parts_path=flame_parts_path,
            teeth_bs_flag=teeth_bs_flag,
            oral_mesh_flag=oral_mesh_flag,
        )

        # uv setting
        self.uv_resolution = uv_resolution
        uv_coords = self.verts_uvs[None, ...] * 2 - 1  # self.verts_uvs = aux.verts_uvs
        uv_coords[..., 1] = -uv_coords[..., 1]  # [1, VT_NUM, 2]
        uv_coords = torch.cat([uv_coords, uv_coords[:, :, 0:1] * 0.0 + 1.0], -1)

        uv_faces = self.textures_idx[None, ...]  # self.textures_idx = faces.textures_idx
        uv_rasterizer = Pytorch3dRasterizer(uv_resolution)
        pix_to_face, bary_coords = uv_rasterizer(uv_coords.expand(1, -1, -1), uv_faces.expand(1, -1, -1))
        valid_pix_mask = pix_to_face > -1

        # ----------- for debug viz -----------
        DEBUG_VIZ = False
        if DEBUG_VIZ:
            uv_mask_viz = valid_pix_mask[0, :, :, 0].cpu().numpy() * 255
            uv_mask_viz_save_path = f'debug_vis/uv_mask_viz_{uv_resolution}.png'
            valid_pix_num = valid_pix_mask.sum()
            valid_pix_rate = valid_pix_num / (uv_resolution * uv_resolution) * 100.0

            print(f'uv_coords: {uv_coords.shape}')  # [1, VT_NUM, 3] (the last channel of the coordinates is 1)
            print(f'uv_faces: {uv_faces.shape}')  # [1, F_NUM_wo, 3]
            print(f'pix_to_face: {pix_to_face.shape}')  # [1, UV_RES, UV_RES, 1]
            print(f'bary_coords: {bary_coords.shape}')  # [1, UV_RES, UV_RES, 1, 3]
            print(f'valid_pix_mask: {valid_pix_mask.shape}')  # [1, UV_RES, UV_RES, 1] BOOL
            print(f'valid_pix_num: {valid_pix_num}, valid_rate: {valid_pix_rate:.2f}%')  # 89.73% for 512x512
            print(f'Visualizing uv mask at {uv_resolution}x{uv_resolution} to {uv_mask_viz_save_path}')
            Image.fromarray(uv_mask_viz.astype(np.uint8)).save(uv_mask_viz_save_path)
        # ----------- for debug viz -----------

        # flatten
        valid_mask_flatten = valid_pix_mask[0, :, :, 0].view(-1)  # [UV_RES * UV_RES,] BOOL
        pix_to_face_flatten = pix_to_face[0].view(-1)[valid_mask_flatten]  # [VALID_PIX_NUM,]
        pix_to_v_idx_flatten = self.faces[pix_to_face_flatten]  # [VALID_PIX_NUM, 3], range: 0 ~ 5022
        pix_bary_flatten = bary_coords[0].view(-1, 3)[valid_mask_flatten]  # [VALID_PIX_NUM, 3]

        self.register_buffer('valid_mask_flatten', valid_mask_flatten.clone().contiguous())
        self.register_buffer('pix_to_face_flatten', pix_to_face_flatten.clone().contiguous(), persistent=False)
        self.register_buffer('pix_to_v_idx_flatten', pix_to_v_idx_flatten.clone().contiguous())
        self.register_buffer('pix_bary_flatten', pix_bary_flatten.clone().contiguous())

        # uv color
        u, v = torch.meshgrid(
            torch.arange(uv_resolution, dtype=torch.float32),
            torch.arange(uv_resolution, dtype=torch.float32),
            indexing="xy"
        )
        u = (u + 0.5) / uv_resolution
        v = (v + 0.5) / uv_resolution
        uv_color = torch.stack([u, v, torch.zeros_like(u)], dim=-1)  # [UV_RES, UV_RES, 3]
        uv_color_flatten = uv_color.view(-1, 3)[valid_mask_flatten]  # [VALID_PIX_NUM, 3]
        self.register_buffer('uv_color_flatten', uv_color_flatten.clone().contiguous())

        # prepare attributes for bary-sample
        self.vertex_num = self.v_template.shape[0]
        self.joint_num = self.J_regressor.shape[0]

        # attributes
        v_template = self.v_template.float()  # [5023, 3]
        lbs_weights = self.lbs_weights.float()  # [5023, 5]
        J_regressor = self.J_regressor.permute(1, 0)  # [5023, 5]
        posedirs = self.posedirs.permute(1, 0).reshape(self.vertex_num, 3 * (self.joint_num - 1) * 9)  # [5023, 108]
        shapedirs = self.shapedirs.view(self.vertex_num, 3 * (self.n_shape_params + self.n_expr_params + (4 if self.teeth_bs_flag else 0)))  # [5023, 1200]

        # barycentric sample
        v_template_pix = self._barycentric_sample(v_template)
        lbs_weights_pix = self._barycentric_sample(lbs_weights)
        J_regressor_pix = self._barycentric_sample(J_regressor)
        posedirs_pix = self._barycentric_sample(posedirs)
        shapedirs_pix = self._barycentric_sample(shapedirs)

        valid_pix_num = v_template_pix.shape[0]
        J_regressor_pix = J_regressor_pix.permute(1, 0)
        posedirs_pix = posedirs_pix.reshape(valid_pix_num * 3, (self.joint_num - 1) * 9).permute(1, 0)
        shapedirs_pix = shapedirs_pix.view(valid_pix_num, 3, (self.n_shape_params + self.n_expr_params + (4 if self.teeth_bs_flag else 0)))

        self.register_buffer('v_template_pix', v_template_pix.contiguous())     # [VALID_PIX_NUM, 3]
        self.register_buffer('lbs_weights_pix', lbs_weights_pix.contiguous())   # [VALID_PIX_NUM, 5]
        self.register_buffer('J_regressor_pix', J_regressor_pix.contiguous())   # [5, VALID_PIX_NUM]
        self.register_buffer('posedirs_pix', posedirs_pix.contiguous())         # [36, VALID_PIX_NUM * 3]
        self.register_buffer('shapedirs_pix', shapedirs_pix.contiguous())       # [VALID_PIX_NUM, 3, 400]
    
    @property
    def uv_valid_mask_flatten(self):
        return self.valid_mask_flatten

    @property
    def uv_color(self):
        """
        uv_color: [VALID_PIX_NUM, 3], the last channel is 0, range: 0 ~ 1
        """
        return self.uv_color_flatten
    
    def _barycentric_sample(self, attr: torch.Tensor) -> torch.Tensor:
        """
        Interpolate vertex attributes with precomputed barycentric coordinates.

        Args:
            attr: vertex attributes shaped [vertex_num, D]
        Returns:
            interpolated attributes shaped [VALID_PIX_NUM, D]
        """
        # attr: [V, D] -> gather by indices [N, 3] -> [N, 3, D]
        attr_per_face = attr[self.pix_to_v_idx_flatten]  # [N, 3, D]
        # bary: [N, 3] -> [N, 3, 1]
        bary = self.pix_bary_flatten.unsqueeze(-1)  # [N, 3, 1]
        # Weighted sum: [N, 3, D] * [N, 3, 1] -> [N, D]
        interpolated = torch.sum(attr_per_face * bary, dim=1)
        return interpolated

    def get_cano_verts(self, shape_params):
        assert self.add_shoulder == False
        batch_size = shape_params.shape[0]
        template_vertices = self.v_template_pix.unsqueeze(0).expand(batch_size, -1, -1)
        v_shaped = template_vertices + blend_shapes(shape_params, self.shapedirs_pix[:, :, :self.n_shape_params])
        return v_shaped

    def forward(
        self,
        v_cano,
        shape,
        expr,
        rotation,
        neck,
        jaw,
        eyes,
        translation,
        zero_centered_at_root_node=False,  # otherwise, zero centered at the face
        return_landmarks=False,
        return_verts_cano=False,
        static_offset=None,
    ):
        assert self.add_shoulder == False
        assert static_offset is None

        batch_size = shape.shape[0]

        # step1. get animated_joint and corresponding transformed mat (Note not in upsampled space)
        betas = torch.cat([shape, expr], dim=1)
        full_pose = torch.cat([rotation, neck, jaw, eyes], dim=1)

        if(self.add_shoulder):
            template_vertices = self.v_template[:(self.v_template.shape[0]-self.v_shoulder.shape[0])].unsqueeze(0).expand(batch_size, -1, -1)
        else:
            template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add shape contribution
        v_shaped = template_vertices + blend_shapes(betas, self.shapedirs)

        # Add personal offsets
        if static_offset is not None:
            if (self.add_shoulder):
                v_shaped += static_offset[:, :(self.v_template.shape[0] - self.v_shoulder.shape[0])]
            else:
                v_shaped += static_offset

        A, J = self.get_transformed_mat(
            pose=full_pose, v_shaped=v_shaped, posedirs=self.posedirs, parents=self.parents,
            J_regressor=self.J_regressor, pose2rot=True, dtype=self.dtype
        )

        # step2. v_cano_with_expr
        v_cano_with_expr = v_cano + blend_shapes(expr, self.shapedirs_pix[:, :, self.n_shape_params:])
        
        # step3. lbs
        vertices = self.skinning(
            v_posed=v_cano_with_expr, A=A, lbs_weights=self.lbs_weights_pix, batch_size=batch_size,
            num_joints=self.joint_num, dtype=self.dtype, device=full_pose.device
        )
        
        if (self.add_shoulder):
            v_shaped = torch.cat([v_shaped, self.v_template[(self.v_template.shape[0] - self.v_shoulder.shape[0]):].unsqueeze(0).expand(batch_size, -1, -1)], dim=1)
            vertices = torch.cat([vertices, self.v_template[(self.v_template.shape[0] - self.v_shoulder.shape[0]):].unsqueeze(0).expand(batch_size, -1, -1)], dim=1)

        if zero_centered_at_root_node:
            vertices = vertices - J[:, [0]]
            J = J - J[:, [0]]

        vertices = vertices + translation[:, None, :]
        J = J + translation[:, None, :]

        ret_vals = {}
        ret_vals["animated"] =vertices

        if return_verts_cano:
            ret_vals["cano"] = v_cano
            ret_vals["cano_with_expr"] = v_cano_with_expr

        # compute landmarks if desired
        if return_landmarks:
            bz = vertices.shape[0]
            landmarks = vertices2landmarks(
                vertices,
                self.faces,
                self.full_lmk_faces_idx.repeat(bz, 1),
                self.full_lmk_bary_coords.repeat(bz, 1, 1),
            )
            ret_vals["landmarks"] = landmarks
        
        return ret_vals
    
    def get_transformed_mat(self, pose, v_shaped, posedirs, parents, J_regressor, pose2rot, dtype):
        batch_size = pose.shape[0]
        device = pose.device

        # Get the joints
        # NxJx3 array
        J = vertices2joints(J_regressor, v_shaped)

        # 3. Add pose blend shapes
        # N x J x 3 x 3
        ident = torch.eye(3, dtype=dtype, device=device)
        if pose2rot:
            rot_mats = batch_rodrigues(pose.view(-1, 3), dtype=dtype).view(
                [batch_size, -1, 3, 3]
            )

            pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
            # (N x P) x (P, V * 3) -> N x V x 3
            pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
        else:
            pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
            rot_mats = pose.view(batch_size, -1, 3, 3)

            pose_offsets = torch.matmul(pose_feature.view(batch_size, -1), posedirs).view(
                batch_size, -1, 3
            )

        v_posed = pose_offsets + v_shaped

        # 4. Get the global joint location
        J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)
        
        return A, J_transformed
    
    def skinning(self, v_posed, A, lbs_weights, batch_size, num_joints, dtype, device):
        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        # num_joints = J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
        verts = v_homo[:, :, :3, 0]
        
        return verts


if __name__ == '__main__':
    human_model_path = "./model_zoo/human_parametric_models"
    flame_model = UVFlameHead(
        shape_params=300,
        expr_params=100,
        uv_resolution=256,
        flame_model_path=f"{human_model_path}/flame2023.pkl",
        flame_lmk_embedding_path=f"{human_model_path}/landmark_embedding_with_eyes.npy",
        flame_template_mesh_path=f"{human_model_path}/flame_w_mouth.obj",
        flame_parts_path=f"{human_model_path}/FLAME_masks.pkl",
        include_mask=False,
        add_teeth=False,
        add_shoulder=False,
        teeth_bs_flag=False,
        oral_mesh_flag=False
    )
