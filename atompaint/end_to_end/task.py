import lightning.pytorch as L
import torch
import torch.nn.functional as F
import torchyield as ty

from .data import make_amino_acid_crops
from atompaint.classifiers.amino_acid import (
        BlosumMetric,
)
from atompaint.checkpoints import EvalModeCheckpointMixin
from atompaint.type_hints import OptFactory, LrFactory
from atompaint.metrics import TrainValTestMetrics
from torchmetrics import MeanMetric, Accuracy

from typing import Optional, Callable

class EndToEndTask(EvalModeCheckpointMixin, L.LightningModule):

    def __init__(
            self,
            denoiser: ty.Layer, 
            classifier: ty.Layer,
            *,
            loss_agg: Callable[[float, float], float],
            opt_factory: OptFactory,
            lr_factory: Optional[LrFactory] = None,
    ):
        super().__init__()

        self.denoiser = ty.module_from_layer(denoiser)
        self.classifier = ty.module_from_layer(classifier)
        self.loss_agg = loss_agg
        self.optimizer = opt_factory([
            self.denoiser, 
            self.classifier,
            self.loss_agg,
        ])
        self.lr_scheduler = lr_factory(self.optimizer) if lr_factory else None

        self.aa_metrics = TrainValTestMetrics(
                lambda: {
                    'aa/accuracy': Accuracy(
                        task='multiclass',
                        num_classes=len(classifier.amino_acids),
                    ),
                    'aa/blosum62': BlosumMetric(
                        n=62,
                        labels=classifier.amino_acids,
                    ),
                }
        )
        self.use_x_pred_metrics = TrainValTestMetrics(
                lambda: {
                    'aa/x_pred': MeanMetric(),
                }
        )

    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return self.optimizer
        else:
            return {
                'optimizer': self.optimizer,
                'lr_scheduler': {
                    'scheduler': self.lr_scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                    'monitor': 'val/loss',
                },
            }

    def training_step(self, batch):
        return self._step(batch, 'train')

    def validation_step(self, batch):
        return self._step(batch, 'val')

    def test_step(self, batch):
        return self._step(batch, 'test')

    def _step(self, batch, loop):
        x_clean = batch['x_clean']
        x_noise = batch['x_noise']
        sigma = batch['sigma']
        aa_labels = batch['aa_labels']

        x_pred = self.denoiser(x_clean + x_noise, sigma)

        sigma = sigma.reshape(-1, 1, 1, 1, 1)
        sigma_data = self.denoiser.sigma_data

        weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
        mse_loss = torch.mean(weight * (x_pred - x_clean)**2)

        x_crop, use_x_pred = make_amino_acid_crops(
                aa_crops=batch['aa_crops'],
                aa_channels=batch['aa_channels'],
                x_pred=x_pred,
                x_clean=x_clean,
                use_x_pred=batch['use_x_pred'],
        )

        aa_pred = self.classifier(x_crop)

        aa_loss = F.cross_entropy(aa_pred, aa_labels)

        loss = self.loss_agg(mse_loss, aa_loss)

        self.log(f'{loop}/loss', loss)
        self.log(f'{loop}/loss/mse', mse_loss)
        self.log(f'{loop}/loss/aa', aa_loss)

        for name, metric in self.aa_metrics.get(loop):
            metric(aa_pred, aa_labels)
            self.log(f'{loop}/{name}', metric)

        for name, metric in self.use_x_pred_metrics.get(loop):
            metric(use_x_pred)
            self.log(f'{loop}/{name}', metric)

        return loss
