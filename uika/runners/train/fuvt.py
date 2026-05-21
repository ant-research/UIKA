from .dynamic_view_trainer import DynamicViewTrainer
from uika.runners import REGISTRY_RUNNERS


@REGISTRY_RUNNERS.register('train.fuvt')
class FUVTTrainer(DynamicViewTrainer):
    loss_name_dict = ['total_loss', 'pixel_loss', 'tv_loss']

    def _build_model(self, cfg):
        from uika.models import model_dict
        model_class = model_dict[cfg.experiment.type]
        model = model_class()
        return model

    def _build_loss_fn(self, cfg):
        from uika.losses import PixelLoss, TVLoss
        self.pixel_loss_fn = PixelLoss(option='l1')
        self.tv_loss_fn = TVLoss()

    def forward_loss_local_step(self, data):
        source_image = data['source_rgbs']  # [B, V, 3, 512, 512]
        source_mask = data['source_masks']  # [B, V, 1, 512, 512]

        outputs = self.model(source_image, source_mask)

        masked_gt = outputs['supervised_uv']
        masked_pred = outputs['pred_uv']

        loss_pixel = self.pixel_loss_fn(masked_gt, masked_pred) * self.cfg.train.loss.pixel_weight
        tv_loss = self.tv_loss_fn(outputs['pred_uv_ori']) * self.cfg.train.loss.tv_weight

        loss_dict = {
            'total_loss': loss_pixel + tv_loss,
            'pixel_loss': loss_pixel,
            'tv_loss': tv_loss,
        }

        return outputs, loss_dict

    def _image_monitor_kwargs(self, data, outs) -> dict:
        return {
            'renders': outs['pred_uv_color'],
            'renders_refine': outs['gt_uv_color'],
            'gts': data['source_rgbs'],
        }
