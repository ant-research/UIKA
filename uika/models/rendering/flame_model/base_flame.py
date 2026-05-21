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
import pickle
import numpy as np
import torch.nn as nn

from einops import rearrange, repeat
from pytorch3d.io import load_obj
from uika.models.rendering.flame_model.flame_mask import FlameMask
from uika.models.rendering.flame_model.lbs import lbs, vertices2landmarks, blend_shapes


def to_tensor(array, dtype=torch.float32):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


class FlameHead(nn.Module):
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
        add_teeth=False,
        add_shoulder=False,
        teeth_bs_flag=False,
        oral_mesh_flag=False,
    ):
        super().__init__()

        self.n_shape_params = shape_params
        self.n_expr_params = expr_params
        self.use_teeth = add_teeth
        self.flame_model_dir = os.path.dirname(flame_model_path)

        with open(flame_model_path, "rb") as f:
            ss = pickle.load(f, encoding="latin1")
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        # The vertices of the template model
        self.register_buffer("v_template", to_tensor(to_np(flame_model.v_template), dtype=self.dtype))

        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:, :, :shape_params], shapedirs[:, :, 300 : 300 + expr_params]], 2)
        self.register_buffer("shapedirs", shapedirs)

        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))
        self.register_buffer("J_regressor", to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer("lbs_weights", to_tensor(to_np(flame_model.weights), dtype=self.dtype))

        # Landmark embeddings for FLAME
        lmk_embeddings = np.load(flame_lmk_embedding_path, allow_pickle=True, encoding="latin1")
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer("full_lmk_faces_idx", torch.tensor(lmk_embeddings["full_lmk_faces_idx"], dtype=torch.long))
        self.register_buffer("full_lmk_bary_coords", torch.tensor(lmk_embeddings["full_lmk_bary_coords"], dtype=self.dtype))

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))

        # add faces and uvs
        verts, faces, aux = load_obj(flame_template_mesh_path, load_textures=False)

        vertex_uvs = aux.verts_uvs
        face_uvs_idx = faces.textures_idx  # index into verts_uvs

        pad = torch.ones(vertex_uvs.shape[0], 1)
        vertex_uvs = torch.cat([vertex_uvs, pad], dim=-1)

        face_uv_coords = face_vertices(vertex_uvs[None], face_uvs_idx[None])[0]
        self.register_buffer("face_uvcoords", face_uv_coords, persistent=False)
        self.register_buffer("faces", faces.verts_idx, persistent=False)

        self.register_buffer("verts_uvs", aux.verts_uvs, persistent=False)
        self.register_buffer("textures_idx", faces.textures_idx, persistent=False)

        # Cal vertex mean uvs from faces for vertex uvs, so as to use FLAME subdivision.
        vtx_ids = rearrange(self.faces, "nf nv -> (nf nv)")
        vtx_ids = repeat(vtx_ids, "n -> n c", c=3)
        uvs = rearrange(self.face_uvcoords, "nf nv c-> (nf nv) c")
        N = self.v_template.shape[0]
        sums = torch.zeros((N, 3), dtype=uvs.dtype, device=uvs.device)
        counts = torch.zeros((N), dtype=torch.int64, device=uvs.device)
        sums.scatter_add_(0, vtx_ids, uvs)
        one_hot = torch.ones_like(vtx_ids[:, 0], dtype=torch.int64).to(uvs.device)
        counts.scatter_add_(0, vtx_ids[:, 0], one_hot)
        clamp_counts = counts.clamp(min=1)
        vtx_uvs = sums / clamp_counts.view(-1, 1)
        
        # Check our template mesh faces match those of FLAME:
        # assert (self.faces == torch.from_numpy(flame_model.f.astype('int64'))).all()
        if include_mask:
            self.mask = FlameMask(
                flame_parts_path=flame_parts_path,
                faces=self.faces, 
                faces_t=self.textures_idx,
                num_verts=self.v_template.shape[0], 
                num_faces=self.faces.shape[0], 
            )

        if self.use_teeth:
            self.add_teeth()

        self.teeth_bs_flag = teeth_bs_flag
        if self.teeth_bs_flag:
            self.add_teeth_bs()

        if self.use_teeth:
            pad = torch.ones(self.teeth_verts_uvs.shape[0], 1)
            teeth_vtx_uvs = torch.cat([self.teeth_verts_uvs, pad], dim=-1)
            vtx_uvs = torch.cat((vtx_uvs, teeth_vtx_uvs), dim=0)

        self.add_shoulder = add_shoulder
        if (add_shoulder):
            shoulder_mesh = trimesh.load(os.path.join(self.flame_model_dir, 'shoulder_mesh.obj'))
            self.v_shoulder = torch.tensor(shoulder_mesh.vertices).float()
            self.f_shoulder = torch.tensor(shoulder_mesh.faces) + self.v_template.shape[0]

            self.v_template = torch.cat([self.v_template, self.v_shoulder], dim=0)
            self.faces = torch.cat([self.faces,self.f_shoulder])

        self.oral_mesh_flag = oral_mesh_flag
        if (self.oral_mesh_flag):
            oral_mesh_path = os.path.join(self.flame_model_dir, 'oral_jawopen0p5.obj')
            assert os.path.exists(oral_mesh_path), "oral_mesh_path {} is not exist!".format(oral_mesh_path)
            oral_mesh = trimesh.load(oral_mesh_path)
            v_oral = torch.tensor(oral_mesh.vertices).float()
            f_oral = torch.tensor(oral_mesh.faces) + self.v_template.shape[0]

            num_verts_oral = v_oral.shape[0]

            shapedirs_shoulder = torch.zeros((num_verts_oral, 3, self.shapedirs.shape[2])).float()
            self.shapedirs = torch.concat([self.shapedirs, shapedirs_shoulder], dim=0)

            # posedirs set to zero
            num_verts_orig = self.v_template.shape[0]
            posedirs = self.posedirs.reshape(len(self.parents) - 1, 9, num_verts_orig, 3)  # (J*9, V*3) -> (J, 9, V, 3)
            posedirs = torch.cat([posedirs, torch.zeros_like(posedirs[:, :, :num_verts_oral])], dim=2)  # (J, 9, V+num_verts_teeth, 3)
            self.posedirs = posedirs.reshape((len(self.parents) - 1) * 9, (num_verts_orig + num_verts_oral) * 3)  # (J*9, (V+num_verts_teeth)*3)

            # J_regressor set to zero
            self.J_regressor = torch.cat([self.J_regressor, torch.zeros_like(self.J_regressor[:, :num_verts_oral])], dim=1)  # (5, J) -> (5, J+num_verts_teeth)

            # lbs_weights manually set
            self.lbs_weights = torch.cat([self.lbs_weights, torch.zeros_like(self.lbs_weights[:num_verts_oral])], dim=0)  # (V, 5) -> (V+num_verts_teeth, 5)

            vid_oral = torch.arange(0, num_verts_oral) + num_verts_orig
            self.lbs_weights[vid_oral, 1] = 1

            self.v_template = torch.cat([self.v_template, v_oral], dim=0)
            self.faces = torch.cat([self.faces, f_oral], dim=0)

    def add_teeth_bs(self):
        teeth_bs_path = os.path.join(self.flame_model_dir, 'teeth_blendshape.json')
        assert os.path.exists(teeth_bs_path), "Path {} is not exist!".format(teeth_bs_path)
        with open(teeth_bs_path, 'r') as f:
            bs_data = json.load(f)
        sorted_keys = sorted(bs_data)
        bs_data = {key: bs_data[key] for key in sorted_keys}
        all_bs = []
        for bs_name in bs_data:
            current_bs = torch.from_numpy(np.array(bs_data[bs_name])).float()
            all_verts_bs = torch.zeros((5023,3))
            all_verts_bs = torch.cat([all_verts_bs,current_bs],dim=0)[None,...]
            all_bs.append(all_verts_bs)
        all_bs = torch.cat(all_bs,dim=0).permute(1,2,0)
        self.shapedirs = torch.cat([self.shapedirs,all_bs],dim=2)

    def add_teeth(self):
        # get reference vertices from lips
        vid_lip_outside_ring_upper = self.mask.get_vid_by_region(['lip_outside_ring_upper'], keep_order=True)

        vid_lip_outside_ring_lower = self.mask.get_vid_by_region(['lip_outside_ring_lower'], keep_order=True)

        v_lip_upper = self.v_template[vid_lip_outside_ring_upper]
        v_lip_lower = self.v_template[vid_lip_outside_ring_lower]

        # construct vertices for teeth
        mean_dist = (v_lip_upper - v_lip_lower).norm(dim=-1, keepdim=True).mean()
        v_teeth_middle = (v_lip_upper + v_lip_lower) / 2
        v_teeth_middle[:, 1] = v_teeth_middle[:, [1]].mean(dim=0, keepdim=True)
        # v_teeth_middle[:, 2] -= mean_dist * 2.5  # how far the teeth are from the lips
        # v_teeth_middle[:, 2] -= mean_dist * 2  # how far the teeth are from the lips
        v_teeth_middle[:, 2] -= mean_dist * 1.5  # how far the teeth are from the lips

        # upper, front
        v_teeth_upper_edge = v_teeth_middle.clone() + torch.tensor([[0, mean_dist, 0]])*0.1
        v_teeth_upper_root = v_teeth_upper_edge + torch.tensor([[0, mean_dist, 0]]) * 2  # scale the height of teeth

        # lower, front
        v_teeth_lower_edge = v_teeth_middle.clone() - torch.tensor([[0, mean_dist, 0]])*0.1
        # v_teeth_lower_edge -= torch.tensor([[0, 0, mean_dist]]) * 0.2  # slightly move the lower teeth to the back
        v_teeth_lower_edge -= torch.tensor([[0, 0, mean_dist]]) * 0.4  # slightly move the lower teeth to the back
        v_teeth_lower_root = v_teeth_lower_edge - torch.tensor([[0, mean_dist, 0]]) * 2  # scale the height of teeth

        # thickness = mean_dist * 0.5
        thickness = mean_dist * 1.
        # upper, back
        v_teeth_upper_root_back = v_teeth_upper_root.clone()
        v_teeth_upper_edge_back = v_teeth_upper_edge.clone()
        v_teeth_upper_root_back[:, 2] -= thickness  # how thick the teeth are
        v_teeth_upper_edge_back[:, 2] -= thickness  # how thick the teeth are

        # lower, back
        v_teeth_lower_root_back = v_teeth_lower_root.clone()
        v_teeth_lower_edge_back = v_teeth_lower_edge.clone()
        v_teeth_lower_root_back[:, 2] -= thickness  # how thick the teeth are
        v_teeth_lower_edge_back[:, 2] -= thickness  # how thick the teeth are

        # concatenate to v_template
        num_verts_orig = self.v_template.shape[0]
        v_teeth = torch.cat([
            v_teeth_upper_root,  # num_verts_orig + 0-14 
            v_teeth_lower_root,  # num_verts_orig + 15-29
            v_teeth_upper_edge,  # num_verts_orig + 30-44
            v_teeth_lower_edge,  # num_verts_orig + 45-59
            v_teeth_upper_root_back,  # num_verts_orig + 60-74
            v_teeth_upper_edge_back,  # num_verts_orig + 75-89
            v_teeth_lower_root_back,  # num_verts_orig + 90-104
            v_teeth_lower_edge_back,  # num_verts_orig + 105-119
        ], dim=0)
        num_verts_teeth = v_teeth.shape[0]
        self.v_template = torch.cat([self.v_template, v_teeth], dim=0)

        vid_teeth_upper_root = torch.arange(0, 15) + num_verts_orig
        vid_teeth_lower_root = torch.arange(15, 30) + num_verts_orig
        vid_teeth_upper_edge = torch.arange(30, 45) + num_verts_orig
        vid_teeth_lower_edge = torch.arange(45, 60) + num_verts_orig
        vid_teeth_upper_root_back = torch.arange(60, 75) + num_verts_orig
        vid_teeth_upper_edge_back = torch.arange(75, 90) + num_verts_orig
        vid_teeth_lower_root_back = torch.arange(90, 105) + num_verts_orig
        vid_teeth_lower_edge_back = torch.arange(105, 120) + num_verts_orig
        
        vid_teeth_upper = torch.cat([vid_teeth_upper_root, vid_teeth_upper_edge, vid_teeth_upper_root_back, vid_teeth_upper_edge_back], dim=0)
        vid_teeth_lower = torch.cat([vid_teeth_lower_root, vid_teeth_lower_edge, vid_teeth_lower_root_back, vid_teeth_lower_edge_back], dim=0)
        vid_teeth = torch.cat([vid_teeth_upper, vid_teeth_lower], dim=0)

        # update vertex masks
        self.mask.v.register_buffer("teeth_upper", vid_teeth_upper)
        self.mask.v.register_buffer("teeth_lower", vid_teeth_lower)
        self.mask.v.register_buffer("teeth", vid_teeth)
        self.mask.v.left_half = torch.cat([
            self.mask.v.left_half, 
            torch.tensor([
                5023, 5024, 5025, 5026, 5027, 5028, 5029, 5030, 5038, 5039, 5040, 5041, 5042, 5043, 5044, 5045, 5053, 5054, 5055, 5056, 5057, 5058, 5059, 5060, 5068, 5069, 5070, 5071, 5072, 5073, 5074, 5075, 5083, 5084, 5085, 5086, 5087, 5088, 5089, 5090, 5098, 5099, 5100, 5101, 5102, 5103, 5104, 5105, 5113, 5114, 5115, 5116, 5117, 5118, 5119, 5120, 5128, 5129, 5130, 5131, 5132, 5133, 5134, 5135, 
            ])], dim=0)

        self.mask.v.right_half = torch.cat([
            self.mask.v.right_half, 
            torch.tensor([
                5030, 5031, 5032, 5033, 5034, 5035, 5036, 5037, 5045, 5046, 5047, 5048, 5049, 5050, 5051, 5052, 5060, 5061, 5062, 5063, 5064, 5065, 5066, 5067, 5075, 5076, 5077, 5078, 5079, 5080, 5081, 5082, 5090, 5091, 5092, 5093, 5094, 5095, 5097, 5105, 5106, 5107, 5108, 5109, 5110, 5111, 5112, 5120, 5121, 5122, 5123, 5124, 5125, 5126, 5127, 5135, 5136, 5137, 5138, 5139, 5140, 5141, 5142, 
            ])], dim=0)

        # construct uv vertices for teeth
        u = torch.linspace(0.62, 0.38, 15)
        v = torch.linspace(1-0.0083, 1-0.0425, 7)
        # v = v[[0, 2, 1, 1]]
        # v = v[[0, 3, 1, 4, 3, 2, 6, 5]]
        v = v[[3, 2, 0, 1, 3, 4, 6, 5]]  # TODO: with this order, teeth_lower is not rendered correctly in the uv space
        uv = torch.stack(torch.meshgrid(u, v, indexing='ij'), dim=-1).permute(1, 0, 2).reshape(num_verts_teeth, 2)  # (#num_teeth, 2)
        num_verts_uv_orig = self.verts_uvs.shape[0]
        num_verts_uv_teeth = uv.shape[0]
        self.verts_uvs = torch.cat([self.verts_uvs, uv], dim=0)
        self.teeth_verts_uvs = uv

        # shapedirs copy from lips
        self.shapedirs = torch.cat([self.shapedirs, torch.zeros_like(self.shapedirs[:num_verts_teeth])], dim=0)
        shape_dirs_mean = (self.shapedirs[vid_lip_outside_ring_upper, :, :self.n_shape_params] + self.shapedirs[vid_lip_outside_ring_lower, :, :self.n_shape_params]) / 2
        self.shapedirs[vid_teeth_upper_root, :, :self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_root, :, :self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_upper_edge, :, :self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_edge, :, :self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_upper_root_back, :, :self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_upper_edge_back, :, :self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_root_back, :, :self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_edge_back, :, :self.n_shape_params] = shape_dirs_mean

        # posedirs set to zero
        posedirs = self.posedirs.reshape(len(self.parents)-1, 9, num_verts_orig, 3)  # (J*9, V*3) -> (J, 9, V, 3)
        posedirs = torch.cat([posedirs, torch.zeros_like(posedirs[:, :, :num_verts_teeth])], dim=2)  # (J, 9, V+num_verts_teeth, 3)
        self.posedirs = posedirs.reshape((len(self.parents)-1)*9, (num_verts_orig+num_verts_teeth)*3)  # (J*9, (V+num_verts_teeth)*3)

        # J_regressor set to zero
        self.J_regressor = torch.cat([self.J_regressor, torch.zeros_like(self.J_regressor[:, :num_verts_teeth])], dim=1)  # (5, J) -> (5, J+num_verts_teeth)

        # lbs_weights manually set
        self.lbs_weights = torch.cat([self.lbs_weights, torch.zeros_like(self.lbs_weights[:num_verts_teeth])], dim=0)  # (V, 5) -> (V+num_verts_teeth, 5)
        self.lbs_weights[vid_teeth_upper, 1] += 1  # move with neck
        self.lbs_weights[vid_teeth_lower, 2] += 1  # move with jaw

        # add faces for teeth
        f_teeth_upper = torch.tensor([
            [0, 31, 30],  #0
            [0, 1, 31],  #1
            [1, 32, 31],  #2
            [1, 2, 32],  #3
            [2, 33, 32],  #4
            [2, 3, 33],  #5
            [3, 34, 33],  #6
            [3, 4, 34],  #7
            [4, 35, 34],  #8
            [4, 5, 35],  #9
            [5, 36, 35],  #10
            [5, 6, 36],  #11
            [6, 37, 36],  #12
            [6, 7, 37],  #13
            [7, 8, 37],  #14
            [8, 38, 37],  #15
            [8, 9, 38],  #16
            [9, 39, 38],  #17
            [9, 10, 39],  #18
            [10, 40, 39],  #19
            [10, 11, 40],  #20
            [11, 41, 40],  #21
            [11, 12, 41],  #22
            [12, 42, 41],  #23
            [12, 13, 42],  #24
            [13, 43, 42],  #25
            [13, 14, 43],  #26
            [14, 44, 43],  #27
            [60, 75, 76],  # 56
            [60, 76, 61],  # 57
            [61, 76, 77],  # 58
            [61, 77, 62],  # 59
            [62, 77, 78],  # 60
            [62, 78, 63],  # 61
            [63, 78, 79],  # 62
            [63, 79, 64],  # 63
            [64, 79, 80],  # 64
            [64, 80, 65],  # 65
            [65, 80, 81],  # 66
            [65, 81, 66],  # 67
            [66, 81, 82],  # 68
            [66, 82, 67],  # 69
            [67, 82, 68],  # 70
            [68, 82, 83],  # 71
            [68, 83, 69],  # 72
            [69, 83, 84],  # 73
            [69, 84, 70],  # 74
            [70, 84, 85],  # 75
            [70, 85, 71],  # 76
            [71, 85, 86],  # 77
            [71, 86, 72],  # 78
            [72, 86, 87],  # 79
            [72, 87, 73],  # 80
            [73, 87, 88],  # 81
            [73, 88, 74],  # 82
            [74, 88, 89],  # 83
            [75, 30, 76],  # 84
            [76, 30, 31],  # 85
            [76, 31, 77],  # 86
            [77, 31, 32],  # 87
            [77, 32, 78],  # 88
            [78, 32, 33],  # 89
            [78, 33, 79],  # 90
            [79, 33, 34],  # 91
            [79, 34, 80],  # 92
            [80, 34, 35],  # 93
            [80, 35, 81],  # 94
            [81, 35, 36],  # 95
            [81, 36, 82],  # 96
            [82, 36, 37],  # 97
            [82, 37, 38],  # 98
            [82, 38, 83],  # 99
            [83, 38, 39],  # 100
            [83, 39, 84],  # 101
            [84, 39, 40],  # 102
            [84, 40, 85],  # 103
            [85, 40, 41],  # 104
            [85, 41, 86],  # 105
            [86, 41, 42],  # 106
            [86, 42, 87],  # 107
            [87, 42, 43],  # 108
            [87, 43, 88],  # 109
            [88, 43, 44],  # 110
            [88, 44, 89],  # 111
        ])
        f_teeth_lower = torch.tensor([
            [45, 46, 15],  # 28           
            [46, 16, 15],  # 29
            [46, 47, 16],  # 30
            [47, 17, 16],  # 31
            [47, 48, 17],  # 32
            [48, 18, 17],  # 33
            [48, 49, 18],  # 34
            [49, 19, 18],  # 35
            [49, 50, 19],  # 36
            [50, 20, 19],  # 37
            [50, 51, 20],  # 38
            [51, 21, 20],  # 39
            [51, 52, 21],  # 40
            [52, 22, 21],  # 41
            [52, 23, 22],  # 42
            [52, 53, 23],  # 43
            [53, 24, 23],  # 44
            [53, 54, 24],  # 45
            [54, 25, 24],  # 46
            [54, 55, 25],  # 47
            [55, 26, 25],  # 48
            [55, 56, 26],  # 49
            [56, 27, 26],  # 50
            [56, 57, 27],  # 51
            [57, 28, 27],  # 52
            [57, 58, 28],  # 53
            [58, 29, 28],  # 54
            [58, 59, 29],  # 55
            [90, 106, 105],  # 112
            [90, 91, 106],  # 113
            [91, 107, 106],  # 114
            [91, 92, 107],  # 115
            [92, 108, 107],  # 116
            [92, 93, 108],  # 117
            [93, 109, 108],  # 118
            [93, 94, 109],  # 119
            [94, 110, 109],  # 120
            [94, 95, 110],  # 121
            [95, 111, 110],  # 122
            [95, 96, 111],  # 123
            [96, 112, 111],  # 124
            [96, 97, 112],  # 125
            [97, 98, 112],  # 126
            [98, 113, 112],  # 127
            [98, 99, 113],  # 128
            [99, 114, 113],  # 129
            [99, 100, 114],  # 130
            [100, 115, 114],  # 131
            [100, 101, 115],  # 132
            [101, 116, 114],  # 133
            [101, 102, 116],  # 134
            [102, 117, 116],  # 135
            [102, 103, 117],  # 136
            [103, 118, 117],  # 137
            [103, 104, 118],  # 138
            [104, 119, 118],  # 139
            [105, 106, 45],  # 140
            [106, 46, 45],  # 141
            [106, 107, 46],  # 142
            [107, 47, 46],  # 143
            [107, 108, 47],  # 144
            [108, 48, 47],  # 145
            [108, 109, 48],  # 146
            [109, 49, 48],  # 147
            [109, 110, 49],  # 148
            [110, 50, 49],  # 149
            [110, 111, 50],  # 150
            [111, 51, 50],  # 151
            [111, 112, 51],  # 152
            [112, 52, 51],  # 153
            [112, 53, 52],  # 154
            [112, 113, 53],  # 155
            [113, 54, 53],  # 156
            [113, 114, 54],  # 157
            [114, 55, 54],  # 158
            [114, 115, 55],  # 159
            [115, 56, 55],  # 160
            [115, 116, 56],  # 161
            [116, 57, 56],  # 162
            [116, 117, 57],  # 163
            [117, 58, 57],  # 164
            [117, 118, 58],  # 165
            [118, 59, 58],  # 166
            [118, 119, 59],  # 167
        ])
        self.faces = torch.cat([self.faces, f_teeth_upper+num_verts_orig, f_teeth_lower+num_verts_orig], dim=0)
        self.textures_idx = torch.cat([self.textures_idx, f_teeth_upper+num_verts_uv_orig, f_teeth_lower+num_verts_uv_orig], dim=0)

        self.mask.update(self.faces, self.textures_idx)

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
        return_landmarks=False,
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
        v_shaped_woexpr = template_vertices + blend_shapes(torch.cat([betas[:, :self.n_shape_params], 
                                                                      torch.zeros_like(betas[:, self.n_shape_params:])],
                                                                      dim=1), self.shapedirs)
        v_shaped = template_vertices + blend_shapes(betas, self.shapedirs)

        # Add personal offsets
        if static_offset is not None:
            if (self.add_shoulder):
                v_shaped += static_offset[:,:(self.v_template.shape[0]-self.v_shoulder.shape[0])]
            else:
                v_shaped += static_offset

        vertices, J, mat_rot = lbs(
            full_pose,
            v_shaped,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            dtype=self.dtype,
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
            ret_vals["cano"] = v_shaped_woexpr
            ret_vals["cano_with_expr"] = v_shaped

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
