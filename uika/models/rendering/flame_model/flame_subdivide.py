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


import os
import json
import torch
import trimesh
import numpy as np

from tqdm import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.ops import SubdivideMeshes

from uika.models.rendering.flame_model.base_flame import FlameHead
from uika.models.rendering.flame_model.lbs import vertices2landmarks, blend_shapes, vertices2joints
from uika.models.rendering.flame_model.lbs import batch_rigid_transform, batch_rodrigues


class FlameHeadSubdivided(FlameHead):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(
        self,
        shape_params,
        expr_params,
        flame_model_path=None,
        flame_lmk_embedding_path=None,
        flame_template_mesh_path=None,
        flame_parts_path=None,
        include_mask=True,
        add_teeth=True,
        add_shoulder=False,
        subdivide_num=0,
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
            teeth_bs_flag = teeth_bs_flag,
            oral_mesh_flag = oral_mesh_flag
        )
        
        # subdivider
        self.subdivide_num = subdivide_num
        self.subdivider_list = self.get_subdivider(subdivide_num)
        self.subdivider_cpu_list = self.get_subdivider_cpu(subdivide_num)
        self.face_upsampled = self.subdivider_list[-1]._subdivided_faces.cpu().numpy() if self.subdivide_num > 0 else self.faces.numpy()
        self.vertex_num_upsampled = int(np.max(self.face_upsampled) + 1)
        
        self.vertex_num = self.v_template.shape[0]
        self.joint_num = self.J_regressor.shape[0]
        print(f"face_upsampled:{self.face_upsampled.shape}, face_ori:{self.faces.shape}, \
                vertex_num_upsampled:{self.vertex_num_upsampled}, vertex_num_ori:{self.vertex_num}")
        
        lbs_weights = self.lbs_weights.float()
        posedirs = self.posedirs.permute(1, 0).reshape(self.vertex_num, 3 * (self.joint_num - 1) * 9)
        shapedirs = self.shapedirs.view(self.vertex_num, 3 * (self.n_shape_params + self.n_expr_params + (4 if self.teeth_bs_flag else 0)))
        J_regressor = self.J_regressor.permute(1, 0)
        
        attributes = [lbs_weights, posedirs, shapedirs, J_regressor]
        ret = self.upsample_mesh_cpu(self.v_template.float(), attributes,) # upsample with dummy vertex
        v_template_upsampled, lbs_weights, posedirs, shapedirs, J_regressor = ret
    
        posedirs = posedirs.reshape(self.vertex_num_upsampled * 3, (self.joint_num-1) * 9).permute(1, 0)
        shapedirs = shapedirs.view(self.vertex_num_upsampled, 3 , (self.n_shape_params + self.n_expr_params + (4 if self.teeth_bs_flag else 0)))
        J_regressor = J_regressor.permute(1, 0)

        self.register_buffer('J_regressor_up', J_regressor.contiguous())
        self.register_buffer('faces_up', torch.from_numpy(self.face_upsampled).to(shapedirs.device))
        self.register_buffer('v_template_up', v_template_upsampled.contiguous())
        self.register_buffer('lbs_weights_up', lbs_weights.contiguous())
        self.register_buffer('shapedirs_up', shapedirs.contiguous())

    def save_bone_tree(self, v_shaped, output_json_path):
        bone_tree = {"bones": [
            {"name": "root", "position": [0, 0, 0], "children": [{"name": "neck", "position": [0, 0, 0], "children":
                [{"name": "jaw", "position": [0, 0, 0]}, {"name": "leftEye", "position": [0, 0, 0]},
                {"name": "rightEye", "position": [0, 0, 0]}]}]}]}

        J = vertices2joints(self.J_regressor, v_shaped)

        bone_tree['bones'][0]['position'] = J[0, 0, :].cpu().tolist()
        bone_tree['bones'][0]['children'][0]['position'] = J[0, 1, :].cpu().tolist()
        bone_tree['bones'][0]['children'][0]['children'][0]['position'] = J[0, 2, :].cpu().tolist()
        bone_tree['bones'][0]['children'][0]['children'][1]['position'] = J[0, 3, :].cpu().tolist()
        bone_tree['bones'][0]['children'][0]['children'][2]['position'] = J[0, 4, :].cpu().tolist()

        with open(output_json_path, 'w') as f:
            json.dump(bone_tree, f, indent=2)

        return 0

    def save_h5_info(self, shape_params, fd="./runtime_data/"):
        if not os.path.exists(fd):
            os.system(f"mkdir -p {fd}")
        faces = self.faces_up.cpu().numpy()
        batch_size = shape_params.shape[0]
        template_vertices = self.v_template_up.unsqueeze(0).expand(batch_size, -1, -1)
        v_shaped = template_vertices + blend_shapes(shape_params, self.shapedirs_up[:, :, :self.n_shape_params])

        with open(os.path.join(fd, "lbs_weight_20k.json"), 'w') as of:
            json.dump(self.lbs_weights_up.cpu().numpy().tolist(), of)

        v_shaped_ori = self.v_template.unsqueeze(0).expand(batch_size, -1, -1) + blend_shapes(shape_params, self.shapedirs[:, :, :self.n_shape_params])
        self.save_bone_tree(v_shaped_ori, os.path.join(fd, "bone_tree.json"))

        mesh = trimesh.Trimesh(vertices=v_shaped.squeeze(0).cpu().numpy(), faces=faces)
        mesh.export(os.path.join(fd, "nature.obj"))
        
        bs_fd = os.path.join(fd, "bs")
        if not os.path.exists(bs_fd):
            os.system(f"mkdir -p {bs_fd}")
        for i in tqdm(range(100), desc="Saving_100_expr_mesh"):
            expr = torch.zeros((1, 100)).to(v_shaped.device)
            expr[:, i] = 1.
            v_shaped_expr = v_shaped + blend_shapes(expr, self.shapedirs_up[:, :, self.n_shape_params:])
            v_shaped_expr = v_shaped_expr.cpu().numpy().squeeze(0)

            mesh = trimesh.Trimesh(vertices=v_shaped_expr, faces=faces)
            mesh.export(os.path.join(bs_fd, f"expr{i}.obj"))

    def save_shaped_mesh(self, shape_params, fd="./runtime_data/"):
        if not os.path.exists(fd):
            os.system(f"mkdir -p {fd}")
        faces = self.faces_up.cpu().numpy()
        batch_size = shape_params.shape[0]
        template_vertices = self.v_template_up.unsqueeze(0).expand(batch_size, -1, -1)
        v_shaped = template_vertices + blend_shapes(shape_params, self.shapedirs_up[:, :, :self.n_shape_params])

        mesh = trimesh.Trimesh(vertices=v_shaped.squeeze(0).cpu().numpy(), faces=faces)
        saved_path = os.path.join(fd, "nature.obj")
        mesh.export(saved_path)

        return saved_path

    def get_cano_verts(self, shape_params):
        # TODO check
        assert self.add_shoulder == False
        batch_size = shape_params.shape[0]

        template_vertices = self.v_template_up.unsqueeze(0).expand(batch_size, -1, -1)

        v_shaped = template_vertices + blend_shapes(shape_params, self.shapedirs_up[:, :, :self.n_shape_params])
        
        return v_shaped

    def animation_forward(self,
        v_cano,
        shape,
        expr,
        rotation,
        neck,
        jaw,
        eyes,
        translation,
        zero_centered_at_root_node=False,  # otherwise, zero centered at the face
        return_landmarks=True,
        return_verts_cano=False,
        static_offset=None,
        dynamic_offset=None,
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
                v_shaped += static_offset[:,:(self.v_template.shape[0]-self.v_shoulder.shape[0])]
            else:
                v_shaped += static_offset

        A, J = self.get_transformed_mat(pose=full_pose, v_shaped=v_shaped, posedirs=self.posedirs,
                                        parents=self.parents, J_regressor=self.J_regressor, pose2rot=True, 
                                        dtype=self.dtype)

        # step2. v_cano_with_expr
        v_cano_with_expr = v_cano + blend_shapes(expr, self.shapedirs_up[:, :, self.n_shape_params:])
        
        # step3. lbs
        vertices = self.skinning(v_posed=v_cano_with_expr, A=A, lbs_weights=self.lbs_weights_up, batch_size=batch_size,
                                 num_joints=self.joint_num, dtype=self.dtype, device=full_pose.device)
        
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

        homogen_coord = torch.ones(
            [batch_size, v_posed.shape[1], 1], dtype=dtype, device=device
        )
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
        verts = v_homo[:, :, :3, 0]
        
        return verts

    def inverse_animation(self,
        v_pose,
        shape,
        expr,
        rotation,
        neck,
        jaw,
        eyes,
        translation,
        zero_centered_at_root_node=False,  # otherwise, zero centered at the face
        return_landmarks=True,
        return_verts_cano=False,
        static_offset=None,
        dynamic_offset=None,
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
                v_shaped += static_offset[:,:(self.v_template.shape[0]-self.v_shoulder.shape[0])]
            else:
                v_shaped += static_offset

        A, J = self.get_transformed_mat(pose=full_pose, v_shaped=v_shaped, posedirs=self.posedirs,
                                        parents=self.parents, J_regressor=self.J_regressor, pose2rot=True, 
                                        dtype=self.dtype)

        v_pose = v_pose - translation[:, None, :]

        # inverse lbs
        v_cano_with_expr = self.inverse_skinning(v_posed=v_pose, A=A, lbs_weights=self.lbs_weights_up, batch_size=batch_size,
                                 num_joints=self.joint_num, dtype=self.dtype, device=full_pose.device)

        # step2. v_cano
        v_cano = v_cano_with_expr - blend_shapes(expr, self.shapedirs_up[:, :, self.n_shape_params:])

        # step3. lbs
        if (self.add_shoulder):
            v_shaped = torch.cat([v_shaped, self.v_template[(self.v_template.shape[0] - self.v_shoulder.shape[0]):].unsqueeze(0).expand(batch_size, -1, -1)], dim=1)
            v_cano = torch.cat([v_cano, self.v_template[(self.v_template.shape[0] - self.v_shoulder.shape[0]):].unsqueeze(0).expand(batch_size, -1, -1)], dim=1)

        if zero_centered_at_root_node:
            v_cano = v_cano - J[:, [0]]
            J = J - J[:, [0]]


        ret_vals = {}
        ret_vals["cano"] = v_cano

        if return_verts_cano:
            ret_vals["cano_with_expr"] = v_cano_with_expr

        # compute landmarks if desired
        if return_landmarks:
            bz = v_cano.shape[0]
            landmarks = vertices2landmarks(
                v_cano,
                self.faces,
                self.full_lmk_faces_idx.repeat(bz, 1),
                self.full_lmk_bary_coords.repeat(bz, 1, 1),
            )
            ret_vals["landmarks"] = landmarks
        
        return ret_vals

    def inverse_skinning(self, v_posed, A, lbs_weights, batch_size, num_joints, dtype, device):
        
        # 5. Do skinning:
        # W is N x V x (J + 1)
        W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        # num_joints = J_regressor.shape[0]
        T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

        homogen_coord = torch.ones(
            [batch_size, v_posed.shape[1], 1], dtype=dtype, device=device
        )
        v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
        v_homo = torch.matmul(torch.inverse(T), torch.unsqueeze(v_posed_homo, dim=-1))
        verts = v_homo[:, :, :3, 0]
        
        return verts 
        
    def forward(
        self,
        shape,
        expr,
        rotation,
        neck,
        jaw,
        eyes,
        translation,
        zero_centered_at_root_node=False,  # otherwise, zero centered at the face
        return_landmarks=True,
        return_verts_cano=False,
        static_offset=None,
        dynamic_offset=None,
    ):
        """
        Input:
            shape_params: N X number of shape parameters
            expression_params: N X number of expression parameters
            pose_params: N X number of pose parameters (6)
        return:d
            vertices: N X V X 3
            landmarks: N X number of landmarks X 3
        """
        batch_size = shape.shape[0]

        betas = torch.cat([shape, expr], dim=1)
        full_pose = torch.cat([rotation, neck, jaw, eyes], dim=1)

        if(self.add_shoulder):
            template_vertices = self.v_template[:(self.v_template.shape[0]-self.v_shoulder.shape[0])].unsqueeze(0).expand(batch_size, -1, -1)
        else:
            template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        # Add shape contribution
        v_shaped_woexpr = template_vertices + blend_shapes(betas[:, :self.n_shape_params], self.shapedirs[:, :, :self.n_shape_params])
        v_shaped = template_vertices + blend_shapes(betas, self.shapedirs)


        # Add personal offsets
        if static_offset is not None:
            if (self.add_shoulder):
                v_shaped += static_offset[:,:(self.v_template.shape[0]-self.v_shoulder.shape[0])]
            else:
                v_shaped += static_offset

        A, J = self.get_transformed_mat(pose=full_pose, v_shaped=v_shaped, posedirs=self.posedirs,
                                        parents=self.parents, J_regressor=self.J_regressor, pose2rot=True, 
                                        dtype=self.dtype)

        v_shaped_up = self.v_template_up.unsqueeze(0).expand(batch_size, -1, -1) + blend_shapes(betas, self.shapedirs_up)
        vertices = self.skinning(v_posed=v_shaped_up, A=A, lbs_weights=self.lbs_weights_up, batch_size=batch_size,
                                 num_joints=self.joint_num, dtype=self.dtype, device=full_pose.device)
        
        
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
            ret_vals["cano"] = self.v_template_up.unsqueeze(0).expand(batch_size, -1, -1) + blend_shapes(betas[:, :self.n_shape_params], self.shapedirs_up[:, :, :self.n_shape_params])
            ret_vals["cano_with_expr"] = v_shaped_up

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

    def get_subdivider(self, subdivide_num):
        vert = self.v_template.float().cuda()
        face = torch.LongTensor(self.faces).cuda()
        mesh = Meshes(vert[None,:,:], face[None,:,:])

        if subdivide_num > 0:
            subdivider_list = [SubdivideMeshes(mesh)]
            for i in range(subdivide_num-1):
                mesh = subdivider_list[-1](mesh)
                subdivider_list.append(SubdivideMeshes(mesh))
        else:
            subdivider_list = [mesh]
        return subdivider_list

    def get_subdivider_cpu(self, subdivide_num):
        vert = self.v_template.float()
        face = torch.LongTensor(self.faces)
        mesh = Meshes(vert[None,:,:], face[None,:,:])

        if subdivide_num > 0:
            subdivider_list = [SubdivideMeshes(mesh)]
            for i in range(subdivide_num-1):
                mesh = subdivider_list[-1](mesh)
                subdivider_list.append(SubdivideMeshes(mesh))
        else:
             subdivider_list = [mesh]
        return subdivider_list
    
    def upsample_mesh_cpu(self, vert, feat_list=None):
        face = torch.LongTensor(self.faces)
        mesh = Meshes(vert[None,:,:], face[None,:,:])
        if self.subdivide_num > 0:
            if feat_list is None:
                for subdivider in self.subdivider_cpu_list:
                    mesh = subdivider(mesh)
                vert = mesh.verts_list()[0]
                return vert
            else:
                feat_dims = [x.shape[1] for x in feat_list]
                feats = torch.cat(feat_list,1)
                for subdivider in self.subdivider_cpu_list:
                    mesh, feats = subdivider(mesh, feats)
                vert = mesh.verts_list()[0]
                feats = feats[0]
                feat_list = torch.split(feats, feat_dims, dim=1)
                return vert, *feat_list
        else:
            if feat_list is None:
                return vert
            else:
                return vert, *feat_list
        
    def upsample_mesh(self, vert, feat_list=None, device="cuda"):
        face = torch.LongTensor(self.faces).to(device)
        mesh = Meshes(vert[None,:,:], face[None,:,:])
        if self.subdivide_num > 0:
            if feat_list is None:
                for subdivider in self.subdivider_list:
                    mesh = subdivider(mesh)
                vert = mesh.verts_list()[0]
                return vert
            else:
                feat_dims = [x.shape[1] for x in feat_list]
                feats = torch.cat(feat_list,1)
                for subdivider in self.subdivider_list:
                    mesh, feats = subdivider(mesh, feats)
                vert = mesh.verts_list()[0]
                feats = feats[0]
                feat_list = torch.split(feats, feat_dims, dim=1)
                return vert, *feat_list
        else:
            if feat_list is None:
                return vert
            else:
                return vert, *feat_list

    def upsample_mesh_batch(self, vert, device="cuda"):
        if self.subdivide_num > 0:
            face = torch.LongTensor(self.faces).to(device).unsqueeze(0).repeat(vert.shape[0], 1, 1)
            mesh = Meshes(vert, face)
            for subdivider in self.subdivider_list:
                mesh = subdivider(mesh)
            vert = torch.stack(mesh.verts_list(), dim=0)
        else:
            pass
        return vert


if __name__ == '__main__':
    add_teeth = True
    subdivide_num = 0 
    teeth_bs_flag = False
    oral_mesh_flag = False
    human_model_path = "./model_zoo/human_parametric_models"
    flame_model = FlameHeadSubdivided(
        300,
        100,
        add_teeth=add_teeth,
        add_shoulder=False,
        flame_model_path=f"{human_model_path}/flame2023.pkl",
        flame_lmk_embedding_path=f"{human_model_path}/landmark_embedding_with_eyes.npy",
        flame_template_mesh_path=f"{human_model_path}/head_template_mesh.obj",
        flame_parts_path=f"{human_model_path}/FLAME_masks.pkl",
        subdivide_num=subdivide_num,
        teeth_bs_flag=teeth_bs_flag,
        oral_mesh_flag=oral_mesh_flag
    )
