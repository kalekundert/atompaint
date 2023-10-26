"""\
Usage:
    ap_transform_pred <config>

Arguments:
    <config>
        A YAML file specifying all the hyperparameters for a training run.  The 
        following keys should be present: 

        trainer: Arguments to the `Trainer` class.
        model: Arguments to the `PredictorModule` class.
        data: Arguments to the `DataModule` class.
"""

import lightning.pytorch as pl
import os

from .models import TransformationPredictor
from .datasets.origins import SqliteOriginSampler
from .datasets.classification import (
        CnnViewIndexDataset, make_cube_face_frames_ab,
)
from atompaint.config import load_config
from atompaint.datasets.voxelize import ImageParams, Grid
from atompaint.datasets.samplers import RangeSampler, InfiniteSampler
from atompaint.checkpoints import EvalModeCheckpointMixin
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from docopt import docopt
from pathlib import Path
from typing import Optional

# Not going to use docker/singularity for now.  It'll be good to make a 
# container when I'm not making changes to atompaint anymore, but until then, 
# I'd have to make a new container for every commit.  

class PredictorModule(EvalModeCheckpointMixin, pl.LightningModule):

    def __init__(
            self, *,
            frequencies: int,
            conv_channels: list[int],
            conv_field_of_view: int | list[int],
            conv_stride: int | list[int],
            conv_padding: int | list[int],
            mlp_channels: int | list[int],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = TransformationPredictor(
            frequencies=frequencies,
            conv_channels=conv_channels,
            conv_field_of_view=conv_field_of_view,
            conv_stride=conv_stride,
            conv_padding=conv_padding,
            mlp_channels=mlp_channels,
        )
        self.loss = CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=6)
        self.optimizer = Adam(self.model.parameters())

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"val/loss": 0})

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, batch):
        x, y = batch
        y_hat = self.model(x)

        loss = self.loss(y_hat, y)
        acc = self.accuracy(y_hat, y)

        return loss, acc

    def training_step(self, batch, _):
        loss, acc = self.forward(batch)
        self.log('train/loss', loss, on_epoch=True)
        self.log('train/accuracy', acc, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        loss, acc = self.forward(batch)
        self.log('val/loss', loss)
        self.log('val/accuracy', acc)
        return loss

    def test_step(self, batch, _):
        loss, acc = self.forward(batch)
        self.log('test/loss', loss)
        self.log('test/accuracy', acc)
        return loss

class DataModule(pl.LightningDataModule):

    def __init__(
            self, *,
            # Origin parameters
            origins_path: Path,

            # Image parameters
            grid_length_voxels: int,
            grid_resolution_A: float,
            element_channels: list[str],
            element_radii_A: Optional[float],

            # View pair parameters
            view_padding_A: float,
            recording_path: Optional[Path] = None,

            # Data loader parameters
            batch_size: int,
            train_epoch_size: int,
            val_epoch_size: int = 0,
            test_epoch_size: int = 0,
            num_workers: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.origin_sampler = SqliteOriginSampler(origins_path)
        img_params = ImageParams(
                grid=Grid(
                    length_voxels=grid_length_voxels,
                    resolution_A=grid_resolution_A,
                ),
                channels=element_channels,
                element_radii_A=element_radii_A,
        )
        view_frames_ab = make_cube_face_frames_ab(
                img_params.grid.length_A,
                view_padding_A,
        )

        if num_workers is None:
            try:
                num_workers = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            except KeyError:
                num_workers = os.cpu_count()

        dataset = CnnViewIndexDataset(
                frames_ab=view_frames_ab,
                origin_sampler=self.origin_sampler,
                img_params=img_params,
                recording_path=recording_path,
        )

        def make_dataloader(sampler):
            return DataLoader(
                    dataset=dataset,
                    sampler=sampler,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
            )

        i = 0
        j = i + val_epoch_size
        k = j + test_epoch_size

        self._val_dataloader = make_dataloader(RangeSampler(i, j))
        self._test_dataloader = make_dataloader(RangeSampler(j, k))
        self._train_dataloader = make_dataloader(
                InfiniteSampler(train_epoch_size, start_index=k),
        )

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def teardown(self, stage):
        self.origin_sampler.teardown()

def main():
    args = docopt(__doc__)
    config_path = Path(args['<config>'])
    c = load_config(config_path, PredictorModule, DataModule)
    c.trainer.fit(c.model, c.data, ckpt_path='last')

if __name__ == '__main__':
    main()


