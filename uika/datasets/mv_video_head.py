import os
import cv2
import json
import torch
import logging
import numpy as np

from PIL import Image
from typing import Optional, Union
from collections import defaultdict

from uika.datasets.base import BaseDataset
from uika.utils.proxy import no_proxy
from uika.utils.training_data import (
    load_cam_pose, img_center_padding,
    pad_to_square, scale_intrs,
    calc_new_tgt_size_by_aspect,
    load_flame_params,
    center_crop_according_to_mask
)


__all__ = ['MV_VideoHeadDataset']


class MV_VideoHeadDataset(BaseDataset):
    def __init__(self, root_dirs: str, meta_path: Optional[Union[str, list]],
                 sample_side_views: int,
                 render_image_res_low: int, render_image_res_high: int, render_region_size: int,
                 source_image_res: int,
                 repeat_num=1,
                 crop_range_ratio_hw=[1.0, 1.0],
                 aspect_standard=1.0,  # h/w
                 enlarge_ratio=[0.8, 1.2],
                 multiply=16,
                 debug=False,
                 is_val=False,
                 **kwargs):
        super().__init__(root_dirs, meta_path)
        self.sample_side_views = sample_side_views
        self.render_image_res_low = render_image_res_low
        self.render_image_res_high = render_image_res_high
        if not (isinstance(render_region_size, list) or isinstance(render_region_size, tuple)): 
            render_region_size = render_region_size, render_region_size  # [H, W]
        self.render_region_size = render_region_size
        self.source_image_res = source_image_res
        
        self.uids = self.uids * repeat_num
        # print(self.uids)
        self.crop_range_ratio_hw = crop_range_ratio_hw
        self.debug = debug
        self.aspect_standard = aspect_standard
        
        assert self.render_image_res_low == self.render_image_res_high
        self.render_image_res = self.render_image_res_low
        self.enlarge_ratio = enlarge_ratio
        print(f"MV_VideoHeadDataset, data_len:{len(self.uids)}, repeat_num:{repeat_num}, debug:{debug}, is_val:{is_val}, multiply:{multiply}")
        self.multiply = multiply
        # set data deterministic
        self.is_val = is_val

        if 'nersemble' in self.root_dirs:
            self.dataset_id: int = 0
        elif 'synth' in self.root_dirs:
            self.dataset_id: int = 2
        else:  # monocular video
            self.dataset_id: int = 1

        self._fallback_uid_idx = None

    def load_rgb_image_with_aug_bg(self, rgb_path, mask_path, bg_color, pad_ratio,
                                   max_tgt_size, aspect_standard, enlarge_ratio, 
                                   render_tgt_size, multiply, intr):
        rgb = np.array(Image.open(rgb_path))

        interpolation = cv2.INTER_AREA
        if rgb.shape[0] != self.source_image_res and rgb.shape[0] == rgb.shape[1]:
            rgb = cv2.resize(rgb, (self.source_image_res, self.source_image_res), interpolation=interpolation)
        if pad_ratio > 0:
            rgb = img_center_padding(rgb, pad_ratio)
        rgb = rgb / 255.0

        if mask_path is not None:
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path)) > 180
                if len(mask.shape) == 3:
                    mask = mask[..., 0]
                assert pad_ratio == 0
                # if pad_ratio > 0:
                #     mask = img_center_padding(mask, pad_ratio)
                # mask = mask / 255.0
            else:
                # print("no mask file")
                mask = (rgb >= 0.99).sum(axis=2) == 3
                mask = np.logical_not(mask)
                # erode
                mask = (mask * 255).astype(np.uint8)
                kernel_size, iterations = 3, 7
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask = cv2.erode(mask, kernel, iterations=iterations) / 255.0
        else:
            # rgb: [H, W, 4]
            assert rgb.shape[2] == 4
            mask = rgb[:, :, 3]   # [H, W]
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
            
        mask = (mask > 0.5).astype(np.float32)
        rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

        # if h != w, padding to [a, a] for a = max(h, w)
        rgb, mask, intr = pad_to_square(rgb, mask, bg_color, intr)
    
        # crop image to enlarge face area.
        rgb, mask, offset_x, offset_y = center_crop_according_to_mask(
            rgb, mask, aspect_standard, enlarge_ratio
        )
        intr[0, 2] -= offset_x
        intr[1, 2] -= offset_y

        # resize to render_tgt_size for training
        tgt_hw_size, ratio_y, ratio_x = calc_new_tgt_size_by_aspect(
            cur_hw=rgb.shape[:2], aspect_standard=aspect_standard,
            tgt_size=render_tgt_size, multiply=multiply
        )
        rgb = cv2.resize(rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=interpolation)
        mask = cv2.resize(mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=interpolation)
        intr = scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)
        
        assert abs(intr[0, 2] * 2 - rgb.shape[1]) < 2.5, f"{intr[0, 2] * 2}, {rgb.shape[1]}"
        assert abs(intr[1, 2] * 2 - rgb.shape[0]) < 2.5, f"{intr[1, 2] * 2}, {rgb.shape[0]}"
        intr[0, 2] = rgb.shape[1] // 2
        intr[1, 2] = rgb.shape[0] // 2
        
        rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
        mask = torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)
        
        return rgb, mask, intr
    
    def uniform_sample_in_chunk(self, sample_num, sample_data):
        chunks = np.array_split(sample_data, sample_num)
        select_list = []
        for chunk in chunks:
            select_list.append(np.random.choice(chunk))
        return select_list

    def uniform_sample_in_chunk_det(self, sample_num, sample_data):
        chunks = np.array_split(sample_data, sample_num)
        select_list = []
        for chunk in chunks:
            select_list.append(chunk[len(chunk)//2])
        return select_list

    @no_proxy
    def inner_get_item(self, idx):
        """
        Lightweight robustness wrapper.
        """
        try:
            return self._inner_get_item_core(idx)
        except Exception as e:
            # Catch exceptions only at the outermost layer to keep training running.
            logging.warning(f"Data loading failed for idx {idx}: {str(e)}")
            return self._get_fallback_sample(idx)

    def _inner_get_item_core(self, idx):
        """
        Core sampling logic with checks only where they are needed.
        """
        if isinstance(idx, tuple):
            idx, num_images = idx
        else:
            num_images = 1
        if self.is_val:
            print(f'[Debug] num_images of this batch: {num_images}, other views: {self.sample_side_views}')
        
        # 1. Basic path handling. Keep the original behavior unchanged.
        uid = self.uids[idx]
        if len(uid.split('/')) == 1:
            if self.dataset_id == 0:  # nersemble_v2
                parent_id, child_seq = uid.split('_')
                uid = os.path.join(self.root_dirs, parent_id, child_seq)
            else:
                uid = os.path.join(self.root_dirs, uid)
        
        # 2. Check only the critical files.
        transforms_json = os.path.join(uid, "transforms.json")
        if not os.path.exists(transforms_json):
            raise FileNotFoundError(f"transforms.json missing: {transforms_json}")
        
        # 3. Load transforms.json and let JSON parsing fail naturally.
        with open(transforms_json) as fp:
            data = json.load(fp)
        
        # 4. Load canonical FLAME parameters with a lightweight existence check.
        cor_flame_path = transforms_json.replace('transforms.json','canonical_flame_param.npz')
        if os.path.exists(cor_flame_path):
            flame_param = np.load(cor_flame_path)
            shape_param = torch.FloatTensor(flame_param.get('shape', np.zeros(100)))
        else:
            raise FileNotFoundError(f"canonical_flame_param.npz missing: {cor_flame_path}")
        
            # Use default shape parameters.
            # shape_param = torch.zeros(100, dtype=torch.float32)
            # logging.warning(f"Using default shape param for {uid}")
        
        # 5. Validate frames with only basic checks.
        all_frames = data["frames"]
        if not all_frames:
            raise ValueError("No frames available")
        
        # 6. Sampling logic. Keep the original behavior.
        sample_total_views = self.sample_side_views + num_images
        # frame_id_list = self._sample_frames(all_frames, sample_total_views)
        if len(all_frames) >= self.sample_side_views:
            if not self.is_val:
                if np.random.rand() < 0.7 and len(all_frames) > sample_total_views:
                    frame_id_list = self.uniform_sample_in_chunk(sample_total_views, np.arange(len(all_frames)))
                else:
                    replace = len(all_frames) < sample_total_views
                    frame_id_list = np.random.choice(len(all_frames), size=sample_total_views, replace=replace)
            else:
                if len(all_frames) > sample_total_views:
                    frame_id_list = self.uniform_sample_in_chunk_det(sample_total_views, np.arange(len(all_frames)))
                else:
                    frame_id_list = np.random.choice(len(all_frames), size=sample_total_views, replace=True)
        else:
            if not self.is_val:
                replace = len(all_frames) < sample_total_views
                frame_id_list = np.random.choice(len(all_frames), size=sample_total_views, replace=replace)
            else:
                if len(all_frames) > 1:
                    frame_id_list = np.linspace(0, len(all_frames) - 1, num=sample_total_views, endpoint=True)
                    frame_id_list = [round(e) for e in frame_id_list]
                else:
                    frame_id_list = [0 for i in range(sample_total_views)]

        assert self.sample_side_views + num_images == len(frame_id_list)
        
        # 7. Optional teeth parameter loading; failures should not affect the main path.
        # teeth_bs_lst = self._load_teeth_safely(uid)
        teeth_bs_pth = os.path.join(uid, "tracked_teeth_bs.npz")
        use_teeth = False
        if os.path.exists(teeth_bs_pth) and use_teeth:
            teeth_bs_lst = np.load(teeth_bs_pth)['expr_teeth']
        else:
            teeth_bs_lst = None
        
        # 8. Batch-load image data and handle per-image failures here.
        render_data = self._load_images_batch(
            all_frames, frame_id_list, uid, teeth_bs_lst, shape_param, 
            flag_src_render=0, target_size=self.render_image_res,
        )

        source_data = self._load_images_batch(
            all_frames, frame_id_list[:num_images], uid, teeth_bs_lst, shape_param, 
            flag_src_render=1, target_size=self.source_image_res,
        )
        
        # 9. Assemble the final result.
        return self._assemble_final_result(render_data, source_data, uid)

    def _load_images_batch(self, all_frames, frame_id_list, uid, teeth_bs_lst, shape_param, flag_src_render, target_size):
        """
        Batch-load images and retry when individual images fail.
        flag_src_render: 1 for src and 0 for render
        """
        c2ws, intrs, rgbs, bg_colors, masks, flame_params, dataset_ids = [], [], [], [], [], [], []

        failed_indices = []

        # for src, multiply = 14 (dinov2) or 16 (dinov3); for render, multiply = 16 (cal loss in 512x512)
        multiply = self.multiply if flag_src_render else 16
        prefix = 'source_' if flag_src_render else ''
        
        for i, frame_id in enumerate(frame_id_list):
            try:
                frame_info = all_frames[frame_id]
                frame_path = os.path.join(uid, frame_info["file_path"])
                mask_path = os.path.join(uid, frame_info["fg_mask_path"])

                teeth_bs = teeth_bs_lst[frame_id] if teeth_bs_lst is not None else None
                flame_path = os.path.join(uid, frame_info["flame_param_path"])
                flame_param = load_flame_params(flame_path, teeth_bs)

                c2w, ori_intrinsic = load_cam_pose(frame_info, transpose_R=(self.dataset_id != 1))

                # make sure that the rest src inputs have the same bg_color as the first src input
                if flag_src_render and i != 0:
                    bg_color = bg_colors[0]
                else:
                    bg_color = np.random.choice([0.0, 0.5, 1.0])
                
                rgb, mask, intrinsic = self.load_rgb_image_with_aug_bg(
                    frame_path, mask_path=mask_path, bg_color=bg_color,
                    pad_ratio=0, max_tgt_size=None, aspect_standard=self.aspect_standard,
                    enlarge_ratio=[1.0, 1.0] if self.is_val else self.enlarge_ratio,
                    render_tgt_size=target_size, multiply=multiply, intr=ori_intrinsic.clone()
                )
                
                c2ws.append(c2w)
                intrs.append(intrinsic)
                rgbs.append(rgb)
                masks.append(mask)
                bg_colors.append(bg_color)
                dataset_ids.append(self.dataset_id)
                flame_params.append(flame_param)
                
            except Exception as e:
                # Record failed indices and replace them later.
                failed_indices.append(i)
                logging.warning(f"Failed to load frame {frame_id}: {str(e)}")
        
        # Replace failed images with randomly selected successfully loaded images.
        if failed_indices and len(c2ws) > 0:
            for failed_idx in reversed(failed_indices):
                success_idx = np.random.randint(0, len(c2ws))
                c2ws.insert(failed_idx, c2ws[success_idx].clone())
                intrs.insert(failed_idx, intrs[success_idx].clone())
                rgbs.insert(failed_idx, rgbs[success_idx].clone())
                masks.insert(failed_idx, masks[success_idx].clone())
                bg_colors.insert(failed_idx, bg_colors[success_idx])
                dataset_ids.insert(failed_idx, dataset_ids[success_idx])
                flame_params.insert(failed_idx, flame_params[success_idx].copy())
        
        # If every image failed, raise and let the caller handle fallback.
        if not c2ws:
            raise RuntimeError("All frames failed to load")
        
        flame_params_tmp = defaultdict(list)
        for flame in flame_params:
            for k, v in flame.items():
                flame_params_tmp[prefix + k].append(v)
        for k, v in flame_params_tmp.items():
            flame_params_tmp[k] = torch.stack(v)
        flame_params = flame_params_tmp
        flame_params[prefix + 'betas'] = shape_param
        
        ret = {
            prefix + 'c2ws': torch.stack(c2ws, dim=0),  # stack: add a dim at 0 and cat
            prefix + 'intrs': torch.stack(intrs, dim=0),
            prefix + 'rgbs': torch.cat(rgbs, dim=0),  # cat: just cat at the existing dim
            prefix + 'masks': torch.cat(masks, dim=0),
            prefix + 'bg_colors': torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3),  # [N, 3]
            prefix + 'dataset_ids': torch.tensor(dataset_ids, dtype=torch.int32),  # [N,]
            prefix + 'flame_params': flame_params,
        }

        return ret

    def _assemble_final_result(self, render_data, source_data, uid):
        """Assemble the final result."""
        # Get image size information.
        render_image = render_data['rgbs']
        render_mask = render_data['masks']
        tgt_size = render_image.shape[2:4]   # [H, W]
        
        # Basic sanity checks.
        intrs = render_data['intrs']
        assert abs(intrs[0, 0, 2] * 2 - render_image.shape[3]) <= 1.1, f"{intrs[0, 0, 2] * 2}, {render_image.shape}"
        assert abs(intrs[0, 1, 2] * 2 - render_image.shape[2]) <= 1.1, f"{intrs[0, 1, 2] * 2}, {render_image.shape}"

        # Assemble the return dictionary.
        ret = {
            'uid': uid,
            # src
            'source_c2ws': source_data['source_c2ws'],  # [N_i, 4, 4]
            'source_intrs': source_data['source_intrs'],  # [N_i, 4, 4]
            'source_rgbs': source_data['source_rgbs'].clamp(0, 1),   # [N_i, 3, H, W]
            'source_masks': source_data['source_masks'].clamp(0, 1), # [N_i, 1, H, W]
            'source_bg_colors': source_data['source_bg_colors'], # [N_i, 3]
            'source_dataset_ids': source_data['source_dataset_ids'],  # [N_i,]
            # render
            'c2ws': render_data['c2ws'],  # [N, 4, 4]
            'intrs': render_data['intrs'],  # [N, 4, 4]
            'render_image': render_image.clamp(0, 1), # [N, 3, H, W]  (N = N_i + sample_side_views)
            'render_mask': render_mask.clamp(0, 1), #[ N, 1, H, W]
            'render_bg_colors': render_data['bg_colors'], # [N, 3]
            'render_dataset_ids': render_data['dataset_ids'],  # [N,]
            'render_full_resolutions': torch.tensor([tgt_size], dtype=torch.float32).repeat(self.sample_side_views + len(source_data['source_c2ws']), 1),  # [N, 2]
        }
        
        # Add FLAME parameters.
        ret.update(render_data['flame_params'])
        ret.update(source_data['source_flame_params'])
        
        return ret
    
    def _get_fallback_sample(self, idx):
        """Get a fallback sample to keep training running."""
        if isinstance(idx, tuple):
            orig_idx, num_images = idx[0], idx[1]
        else:
            orig_idx, num_images = idx, 1

        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                fallback_idx = np.random.randint(0, len(self.uids))
                if fallback_idx == orig_idx:
                    fallback_idx = (fallback_idx + 1) % len(self.uids)
                result = self._inner_get_item_core((fallback_idx, num_images))
                self._fallback_uid_idx = fallback_idx
                return result
            except Exception as e:
                logging.warning(f"Fallback attempt {attempt+1}/{max_attempts} failed: {e}")
                continue

        if self._fallback_uid_idx is not None:
            logging.error("All random fallbacks failed, retrying with known-good uid")
            return self._inner_get_item_core((self._fallback_uid_idx, num_images))

        raise RuntimeError(f"No fallback available after {max_attempts} attempts")
    
    # ==========================================================================================================================================


if __name__ == "__main__":
    import trimesh
    import cv2
    # root_dir = "./train_data/nersemble_v2/export"
    # meta_path = "./train_data/nersemble_v2/label/local_total_ids.json"
    root_dir = "./train_data/synth_mv/export"
    meta_path = "./train_data/synth_mv/label/local_total_ids.json"
    dataset = MV_VideoHeadDataset(root_dirs=root_dir, meta_path=meta_path, sample_side_views=15,
                    render_image_res_low=512, render_image_res_high=512,
                    render_region_size=(512, 512), source_image_res=512,
                    enlarge_ratio=[0.8, 1.2],
                    debug=False, is_val=False)

    from uika.models.rendering.flame_model.flame_subdivide import FlameHeadSubdivided

    # subdivided flame 
    subdivide = 1
    flame_sub_model = FlameHeadSubdivided(
        300,
        100,
        add_teeth=False,
        add_shoulder=False,
        flame_model_path="model_zoo/human_parametric_models/flame2023.pkl",
        flame_lmk_embedding_path="model_zoo/human_parametric_models/landmark_embedding_with_eyes.npy",
        flame_template_mesh_path="model_zoo/human_parametric_models/head_template_mesh.obj",
        flame_parts_path="model_zoo/human_parametric_models/FLAME_masks.pkl",
        subdivide_num=subdivide,
        teeth_bs_flag=False,
    ).cuda()
    
    source_key = "source_rgbs"
    render_key = "render_image"
        
    for idx, data in enumerate(dataset):
        import boxx
        boxx.tree(data)
        if idx > 0:
            exit(0)
        os.makedirs("debug_vis/dataloader", exist_ok=True)
        for i in range(data[source_key].shape[0]):
            cv2.imwrite(f"debug_vis/dataloader/{source_key}_{i}_b{idx}.jpg", ((data[source_key][i].permute(1, 2, 0).numpy()[:, :, (2, 1, 0)] * 255).astype(np.uint8)))
            
        for i in range(data[render_key].shape[0]):
            cv2.imwrite(f"debug_vis/dataloader/rgbs{i}_b{idx}.jpg", ((data[render_key][i].permute(1, 2, 0).numpy()[:, :, (2, 1, 0)] * 255).astype(np.uint8)))
            

        save_root = "./debug_vis/dataloader"
        os.makedirs(save_root, exist_ok=True)

        shape = data['betas'].to('cuda')
        flame_param = {}
        flame_param['expr'] = data['expr'].to('cuda')
        flame_param['rotation'] = data['rotation'].to('cuda')
        flame_param['neck'] = data['neck_pose'].to('cuda')
        flame_param['jaw'] = data['jaw_pose'].to('cuda')
        flame_param['eyes'] = data['eyes_pose'].to('cuda')
        flame_param['translation'] = data['translation'].to('cuda')


        v_cano = flame_sub_model.get_cano_verts(
            shape.unsqueeze(0)
        )
        ret = flame_sub_model.animation_forward(
            v_cano.repeat(flame_param['expr'].shape[0], 1, 1),
            shape.unsqueeze(0).repeat(flame_param['expr'].shape[0], 1),
            flame_param['expr'],
            flame_param['rotation'],
            flame_param['neck'],
            flame_param['jaw'],
            flame_param['eyes'],
            flame_param['translation'],
            zero_centered_at_root_node=False,
            return_landmarks=False,
            return_verts_cano=True,
            # static_offset=batch_data['static_offset'].to('cuda'),
            static_offset=None,
        )

        import boxx
        boxx.tree(data)
        boxx.tree(ret)
        
        for i in range(ret["animated"].shape[0]):
            mesh = trimesh.Trimesh()
            mesh.vertices = np.array(ret["animated"][i].cpu().squeeze())
            mesh.faces = np.array(flame_sub_model.faces.cpu().squeeze())
            mesh.export(f'{save_root}/animated_sub{subdivide}_{i}.obj')

            intr = data["intrs"][i]
            from uika.models.rendering.utils.vis_utils import render_mesh
            cam_param = {"focal": torch.tensor([intr[0, 0], intr[1, 1]]), 
                        "princpt": torch.tensor([intr[0, 2], intr[1, 2]])}
            render_shape = data[render_key].shape[2:] # int(cam_param['princpt'][1]* 2), int(cam_param['princpt'][0] * 2)
            
            face = flame_sub_model.faces.cpu().squeeze().numpy()
            vertices = ret["animated"][i].cpu().squeeze()
            
            c2ws = data["c2ws"][i]
            w2cs = torch.inverse(c2ws)
            if data['render_dataset_ids'][0] != 1:
                R = w2cs[:3, :3].transpose(1, 0)
            else:
                R = w2cs[:3, :3]
            T = w2cs[:3, 3]
            vertices = vertices @ R + T
            mesh_render, is_bkg = render_mesh(vertices, face, cam_param=cam_param, 
                                            bkg=np.ones((render_shape[0],render_shape[1], 3), dtype=np.float32) * 255, 
                                            return_bg_mask=True)
            
            rgb_mesh = mesh_render.astype(np.uint8)
            t_image = (data[render_key][i].permute(1, 2, 0)*255).numpy().astype(np.uint8)
            
            blend_ratio = 0.7
            vis_img = np.concatenate([rgb_mesh, t_image, (blend_ratio * rgb_mesh + (1 -  blend_ratio) * t_image).astype(np.uint8)], axis=1)
            cam_idx = int(data.get('cam_idxs', [i for j in range(15 + 8)])[i])

            cv2.imwrite(os.path.join(save_root, f"render_{cam_idx}.jpg"), vis_img[:, :, (2, 1, 0)])
