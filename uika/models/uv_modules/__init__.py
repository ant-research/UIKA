import os
import torch
import numpy as np
from PIL import Image
from torch import nn
from typing import Tuple

from uika.models.uv_modules.p3dmm import pixel3dmm


def uv_reproject(
    rgb_image: torch.Tensor,
    mask: torch.Tensor,
    uv_map: torch.Tensor,
    per_view_uv_size: int,
    aggregated_uv_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Back-project multi-view, batched RGB colors onto UV planes and aggregate in two ways.
    Vectorized for efficient GPU execution.

    Args:
        rgb_image: Input RGB images, shape [B, V, 3, H, W], value range 0~1.
        mask: Mask images, shape [B, V, 1, H, W], valid regions > 0.
        uv_map: Per-pixel UV coordinate maps, shape [B, V, 2, H, W], value range 0~1.
        per_view_uv_size: Side length of the per-view UV plane output.
        aggregated_uv_size: Side length of the aggregated UV plane output.

    Returns:
        per_view_planes: Per-view UV planes, shape [B, V, 3, per_view_uv_size, per_view_uv_size].
        aggregated_plane: Batch-aggregated UV plane, shape [B, 4, aggregated_uv_size, aggregated_uv_size].
                          First 3 channels are averaged RGB, 4th channel is hit count.
    """
    B, V, _, H, W = rgb_image.shape
    device = rgb_image.device
    bv_batch_size = B * V

    # 1. Merge B and V dims for batched processing
    rgb_image_flat = rgb_image.reshape(bv_batch_size, 3, H, W)
    mask_flat = mask.reshape(bv_batch_size, 1, H, W)
    uv_map_flat = uv_map.reshape(bv_batch_size, 2, H, W)

    # 2. Permute dims and filter valid pixels
    rgb_permuted = rgb_image_flat.permute(0, 2, 3, 1)  # [BV, H, W, 3]
    mask_squeezed = mask_flat.squeeze(1)              # [BV, H, W]
    uv_permuted = uv_map_flat.permute(0, 2, 3, 1)    # [BV, H, W, 2]

    valid_mask = mask_squeezed > 0
    if not torch.any(valid_mask):
        # No valid pixels, return black planes
        per_view_planes = torch.zeros((B, V, 3, per_view_uv_size, per_view_uv_size), dtype=rgb_image.dtype, device=device)
        aggregated_plane = torch.zeros((B, 4, aggregated_uv_size, aggregated_uv_size), dtype=rgb_image.dtype, device=device)
        return per_view_planes, aggregated_plane

    source_colors = rgb_permuted[valid_mask].float()  # [N_total, 3]
    valid_uvs = uv_permuted[valid_mask]               # [N_total, 2]

    # Clamp UV coordinates to [0, 1]
    valid_uvs = torch.clamp(valid_uvs, 0.0, 1.0)

    # 3. Build indices mapping each valid pixel back to its batch/view index
    batch_indices = torch.arange(bv_batch_size, device=device).view(bv_batch_size, 1, 1).expand(-1, H, W)
    valid_bv_indices = batch_indices[valid_mask]  # [N_total] (range 0 to BV-1)

    # --- Path 1: Compute per-view UV planes ---

    # Coordinate transform
    target_coords1 = valid_uvs * (per_view_uv_size - 1)
    target_indices1 = target_coords1.round().long()
    target_u1, target_v1 = target_indices1[:, 0], target_indices1[:, 1]
    
    # Clamp coordinate range
    target_u1 = torch.clamp(target_u1, 0, per_view_uv_size - 1)
    target_v1 = torch.clamp(target_v1, 0, per_view_uv_size - 1)
    
    # Flatten 3D index (bv, v, u) to 1D for index_add_
    flat_indices1 = valid_bv_indices * (per_view_uv_size * per_view_uv_size) + target_v1 * per_view_uv_size + target_u1

    # Index range check
    max_flat_index1 = bv_batch_size * per_view_uv_size * per_view_uv_size - 1
    if torch.max(flat_indices1) > max_flat_index1 or torch.min(flat_indices1) < 0:
        print(f"Error: flat_indices1 out of range! Range: [{torch.min(flat_indices1)}, {torch.max(flat_indices1)}], Expected: [0, {max_flat_index1}]")
        print(f"valid_bv_indices range: [{torch.min(valid_bv_indices)}, {torch.max(valid_bv_indices)}]")
        print(f"target_u1 range: [{torch.min(target_u1)}, {torch.max(target_u1)}]")
        print(f"target_v1 range: [{torch.min(target_v1)}, {torch.max(target_v1)}]")
        # Force clamp range
        flat_indices1 = torch.clamp(flat_indices1, 0, max_flat_index1)

    # Accumulate colors and counts
    flat_color_sum1 = torch.zeros(bv_batch_size * per_view_uv_size * per_view_uv_size, 3, dtype=torch.float32, device=device)
    flat_count1 = torch.zeros(bv_batch_size * per_view_uv_size * per_view_uv_size, dtype=torch.int32, device=device)
    
    flat_color_sum1.index_add_(0, flat_indices1, source_colors)
    flat_count1.index_add_(0, flat_indices1, torch.ones_like(flat_indices1, dtype=torch.int32))

    # Reshape and compute average
    color_sum1 = flat_color_sum1.view(bv_batch_size, per_view_uv_size, per_view_uv_size, 3)
    count1 = flat_count1.view(bv_batch_size, per_view_uv_size, per_view_uv_size, 1)
    
    avg_colors1 = color_sum1 / count1.float().clamp(min=1)
    
    # Place results on black background
    hit_mask1 = count1 > 0
    per_view_planes_flat = torch.where(hit_mask1, avg_colors1, torch.zeros_like(avg_colors1))
    
    # Reshape to [B, V, C, H, W]
    per_view_planes = per_view_planes_flat.view(B, V, per_view_uv_size, per_view_uv_size, 3).permute(0, 1, 4, 2, 3)

    # --- Path 2: Compute batch-aggregated UV plane ---

    # Get original batch index (B) instead of view index (BV)
    valid_b_indices = valid_bv_indices // V # [N_total] (range 0 to B-1)

    target_coords2 = valid_uvs * (aggregated_uv_size - 1)
    target_indices2 = target_coords2.round().long()
    target_u2, target_v2 = target_indices2[:, 0], target_indices2[:, 1]

    # Clamp coordinate range
    target_u2 = torch.clamp(target_u2, 0, aggregated_uv_size - 1)
    target_v2 = torch.clamp(target_v2, 0, aggregated_uv_size - 1)

    flat_indices2 = valid_b_indices * (aggregated_uv_size * aggregated_uv_size) + target_v2 * aggregated_uv_size + target_u2
    
    # Index range check
    max_flat_index2 = B * aggregated_uv_size * aggregated_uv_size - 1
    if torch.max(flat_indices2) > max_flat_index2 or torch.min(flat_indices2) < 0:
        print(f"Error: flat_indices2 out of range! Range: [{torch.min(flat_indices2)}, {torch.max(flat_indices2)}], Expected: [0, {max_flat_index2}]")
        print(f"valid_b_indices range: [{torch.min(valid_b_indices)}, {torch.max(valid_b_indices)}]")
        print(f"target_u2 range: [{torch.min(target_u2)}, {torch.max(target_u2)}]")
        print(f"target_v2 range: [{torch.min(target_v2)}, {torch.max(target_v2)}]")
        # Force clamp range
        flat_indices2 = torch.clamp(flat_indices2, 0, max_flat_index2)

    # Accumulate colors and counts
    flat_color_sum2 = torch.zeros(B * aggregated_uv_size * aggregated_uv_size, 3, dtype=torch.float32, device=device)
    flat_count2 = torch.zeros(B * aggregated_uv_size * aggregated_uv_size, dtype=torch.int32, device=device)
    
    flat_color_sum2.index_add_(0, flat_indices2, source_colors)
    flat_count2.index_add_(0, flat_indices2, torch.ones_like(flat_indices2, dtype=torch.int32))
    
    # Reshape and compute average
    color_sum2 = flat_color_sum2.view(B, aggregated_uv_size, aggregated_uv_size, 3)
    count2 = flat_count2.view(B, aggregated_uv_size, aggregated_uv_size, 1)
    
    avg_colors2 = color_sum2 / count2.float().clamp(min=1)
    
    # Place results on black background
    hit_mask2 = count2 > 0
    aggregated_rgb = torch.where(hit_mask2, avg_colors2, torch.zeros_like(avg_colors2))

    # Concatenate RGB and hit count into 4-channel output
    aggregated_rgb_chw = aggregated_rgb.permute(0, 3, 1, 2)
    aggregated_count_chw = count2.float().permute(0, 3, 1, 2)

    # log1p handles zero values stably: log1p(x) = log(1+x)
    log_scaled = torch.log1p(aggregated_count_chw)
    max_vals, _ = torch.max(log_scaled.flatten(1), dim=1, keepdim=True)
    max_vals = max_vals.view(log_scaled.shape[0], 1, 1, 1)
    max_vals = torch.clamp(max_vals, min=1e-6)  # Prevent division by zero
    aggregated_count_chw = log_scaled / max_vals  # Log-normalized hit count, range: 0 ~ 1

    aggregated_plane = torch.cat([aggregated_rgb_chw, aggregated_count_chw], dim=1)

    return per_view_planes.to(rgb_image.dtype), aggregated_plane.to(rgb_image.dtype)


class PixelUVWrapper(nn.Module):
    def __init__(self, ckpt_path: str = 'model_zoo/uv_modules/p3dmm.ckpt'):
        super().__init__()
        self.model = self._load_pretrained(ckpt_path)
        self._freeze()
    
    def _load_pretrained(self, ckpt_path: str):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Found keys that are not in the model state dict")
            p3dmm = pixel3dmm.load_from_checkpoint(ckpt_path, strict=False, map_location='cpu')
        return p3dmm.net
    
    def _freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: [B, V, C, H, W], value range: 0 ~ 1
        """
        img_mirrored = torch.flip(img, dims=[-1]).to(img.device)
        img = torch.cat([img, img_mirrored], dim=0)
        # [B, V, C, H, W] -> [B, V, H, W, C]
        img = img.permute(0, 1, 3, 4, 2)
        batch = {'tar_rgb': img}
        
        pred_uvs = self.model(batch)[0]['uv_map']
        
        uv_map, mirrored_uv_maps = pred_uvs.chunk(2, dim=0)
        mirrored_uv_maps = torch.flip(mirrored_uv_maps, dims=[-1])

        mirrored_uv_maps[:, :, 0, :, :] *= -1
        mirrored_uv_maps[:, :, 0, :, :] += 2 * 0.0075
        
        uv_map = (uv_map + mirrored_uv_maps) / 2
        uv_query_map = torch.clamp((uv_map + 1) / 2, 0.0, 1.0)

        return uv_query_map  # [B, V, 2, H, W], value range: 0 ~ 1


if __name__ == '__main__':
    def proc_tensor(t: torch.Tensor):
        """
        t: [C, H, W] on device, 0 ~ 1
        """
        return (t * 255).permute(1, 2, 0).detach().cpu().numpy().astype('uint8')

    def proc_uv_tensor(t: torch.Tensor):
        """
        t: [C, H, W] on device, 0 ~ 1
        """
        t = t.permute(1, 2, 0).detach().cpu().numpy()
        res = np.concatenate([t, np.zeros_like(t[..., :1])], axis=-1)
        return (res * 255).astype('uint8')


    from uika.datasets.mv_video_head import MV_VideoHeadDataset
    device = 'cuda:1'

    pixel_uv_wrapper = PixelUVWrapper().to(device)

    root_dir = "./train_data/nersemble_v2/export"
    meta_path = "./train_data/nersemble_v2/label/local_total_ids.json"
    # root_dir = "./train_data/synth_mv/export"
    # meta_path = "./train_data/synth_mv/label/local_total_ids.json"
    
    dataset = MV_VideoHeadDataset(
        root_dirs=root_dir, meta_path=meta_path, sample_side_views=7,
        render_image_res_low=512, render_image_res_high=512,
        render_region_size=(512, 512), source_image_res=512,
        enlarge_ratio=[0.8, 1.2],
        debug=False, is_val=False
    )

    os.makedirs('debug_vis/p3dmm', exist_ok=True)

    for idx, data in enumerate(dataset):
        render_image = data['render_image'].unsqueeze(0).to(device)  # [B, N, 3, H, W]
        render_mask = data['render_mask'].unsqueeze(0)  # [B, N, 1, H, W]
        render_bg_colors = data['render_bg_colors'].unsqueeze(0)  # [B, N, 3]

        B, V, _, H, W = render_image.shape

        input_rgb = render_image.view(B * V, 3, H, W)[:, None]
        output_uv = pixel_uv_wrapper(input_rgb)
        uv_color = output_uv[:, 0].view(B, V, 2, H, W)

        per_view_uv, aggregated_uv = uv_reproject(
            rgb_image=render_image,
            mask=render_mask.to(device),
            uv_map=uv_color,
            per_view_uv_size=int(render_image.shape[-1]),
            aggregated_uv_size=256,
        )

        view = -1

        for b in range(B):
            multi_proj = aggregated_uv[b]
            multi_proj_rgb = proc_tensor(multi_proj[:3])
            Image.fromarray(multi_proj_rgb).save(f'debug_vis/p3dmm/{idx:02d}_views_{V}_rgb.png')

            hit_mask = multi_proj[-1].detach().cpu().numpy()
            Image.fromarray((hit_mask * 255).astype('uint8')).save(f'debug_vis/p3dmm/{idx:02d}_views_{V}_mask.png')

            for v in range(V):
                view += 1

                rgb = proc_tensor(render_image[b, v])
                single_uv = proc_uv_tensor(uv_color[b, v])
                
                projected_rgb = per_view_uv[b, v]
                projected_rgb = proc_tensor(projected_rgb)

                mask_uv = uv_color[b, v].detach().cpu()
                mask_uv = torch.cat([mask_uv, torch.zeros_like(mask_uv[:1])], dim=0)
                mask_uv = mask_uv * render_mask[b, v] + (1 - render_mask[b, v]) * render_bg_colors[b, v][:, None, None]
                mask_uv = proc_tensor(mask_uv)

                all_viz = np.concatenate([rgb, single_uv, mask_uv, projected_rgb], axis=1)
                Image.fromarray(all_viz).save(f'debug_vis/p3dmm/{idx:02d}_{view:02d}.png')

