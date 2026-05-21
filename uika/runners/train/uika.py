import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim

from .dynamic_view_trainer import DynamicViewTrainer
from uika.runners import REGISTRY_RUNNERS


@REGISTRY_RUNNERS.register('train.uika')
class UIKATrainer(DynamicViewTrainer):
    loss_name_dict = [
        'total_loss',
        'pixel_loss',
        'ssim_loss',
        'perceptual_loss',
        'offset_reg_loss',
        'opacity_entropy_loss',
    ]

    def _build_model(self, cfg):
        from uika.models import model_dict
        model_class = model_dict[cfg.experiment.type]
        model = model_class(**cfg.model)
        return model

    def _build_loss_fn(self, cfg):
        from uika.losses import PixelLoss, LPIPSLoss, OffsetReg, OpacityEntropyLoss
        self.pixel_loss_fn = PixelLoss(option=cfg.train.loss.pixel_loss_fn)
        with self.accelerator.main_process_first():
            self.perceptual_loss_fn = LPIPSLoss(device=self.device, prefech=True)
        self.offset_reg_loss_fn = OffsetReg(sigma=cfg.train.loss.offset_sigma)
        self.opacity_entropy_loss_fn = OpacityEntropyLoss(
            eps=cfg.train.loss.get('opacity_entropy_eps', 1e-6),
        )

    def forward_loss_local_step(self, data):
        render_c2ws = data['c2ws']
        render_intrs = data['intrs']

        source_image = data['source_rgbs']
        source_mask = data['source_masks']
        render_image = data['render_image']  # [B, N, 3, H, W]
        render_bg_colors = data['render_bg_colors']  # [B, N, 3]

        flame_params = dict(
            expr=data['expr'],
            rotation=data['rotation'],
            neck_pose=data['neck_pose'],
            jaw_pose=data['jaw_pose'],
            eyes_pose=data['eyes_pose'],
            translation=data['translation'],
            betas=data['betas'],
        )

        B, N, C, H, W = render_image.shape

        outputs = self.model(
            image=source_image,
            mask=source_mask,
            render_c2ws=render_c2ws,
            render_intrs=render_intrs,
            render_bg_colors=render_bg_colors,
            flame_params=flame_params,
        )

        cano_gs_lst = outputs['cano_gs_lst']
        offset_list = [_gs.offset.unsqueeze(0) for _gs in cano_gs_lst]
        offsets = torch.cat(offset_list, dim=0)

        zero_loss = outputs['comp_rgb'].new_tensor(0.0)
        loss_dict = {loss_name: zero_loss for loss_name in self.loss_name_dict}
        total_loss = None

        if self.cfg.train.loss.pixel_weight > 0.:
            loss_pixel = self.pixel_loss_fn(outputs['comp_rgb'], render_image)
            loss_dict['pixel_loss'] = loss_pixel * self.cfg.train.loss.pixel_weight
            total_loss = self._accumulate_loss(total_loss, loss_dict['pixel_loss'])

        if self.cfg.train.loss.ssim_weight > 0.:
            ssim_value = fused_ssim(outputs['comp_rgb'].view(B*N, C, H, W), render_image.view(B*N, C, H, W))
            loss_dict['ssim_loss'] = (1.0 - ssim_value) * self.cfg.train.loss.ssim_weight
            total_loss = self._accumulate_loss(total_loss, loss_dict['ssim_loss'])

        if self.cfg.train.loss.perceptual_weight > 0.:
            pred = outputs['comp_rgb'].view(B*N, C, H, W)
            gt = render_image.view(B*N, C, H, W)
            pred_224 = F.interpolate(pred, size=(224, 224), mode='area')
            gt_224 = F.interpolate(gt, size=(224, 224), mode='area')
            loss_perceptual = self.perceptual_loss_fn(pred, gt) + self.perceptual_loss_fn(pred_224, gt_224)
            loss_dict['perceptual_loss'] = loss_perceptual * self.cfg.train.loss.perceptual_weight
            total_loss = self._accumulate_loss(total_loss, loss_dict['perceptual_loss'])

        if self.cfg.train.loss.offset_reg_weight > 0.:
            loss_offset_reg = self.offset_reg_loss_fn(offsets)
            loss_dict['offset_reg_loss'] = loss_offset_reg * self.cfg.train.loss.offset_reg_weight
            total_loss = self._accumulate_loss(total_loss, loss_dict['offset_reg_loss'])

        opacity_entropy_weight = self.cfg.train.loss.get('opacity_entropy_weight', 0.0)
        if opacity_entropy_weight > 0. and not self.cfg.model.get('fix_opacity', False):
            opacity = torch.cat([_gs.opacity for _gs in cano_gs_lst], dim=0)
            loss_opacity_entropy = self.opacity_entropy_loss_fn(opacity)
            loss_dict['opacity_entropy_loss'] = loss_opacity_entropy * opacity_entropy_weight
            total_loss = self._accumulate_loss(total_loss, loss_dict['opacity_entropy_loss'])

        if total_loss is None:
            total_loss = outputs['comp_rgb'].sum() * 0.0
        loss_dict['total_loss'] = total_loss

        return outputs, loss_dict

    @staticmethod
    def _accumulate_loss(total_loss, loss):
        return loss if total_loss is None else total_loss + loss

    def _image_monitor_kwargs(self, data, outs) -> dict:
        return {
            'renders': outs['comp_rgb'],
            'gts': data['render_image'],
        }
