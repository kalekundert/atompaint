import torch
import lightning.pytorch as pl
import os

from .models import TransformationPredictor
from .loss import CoordFrameMseLoss
from .datasets import NeighborCountDatasetForCnn, SqliteOriginSampler, ParquetOriginSampler
from atompaint.datasets.voxelize import ImageParams, Grid
from lightning.pytorch.cli import LightningCLI
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional

# Not going to use docker/singularity for now.  It'll be good to make a 
# container when I'm not making changes to atompaint anymore, but until then, 
# I'd have to make a new container for every commit.  

class PredictorModule(pl.LightningModule):

    def __init__(self, model: TransformationPredictor, loss_radius_A: float):
        super().__init__()
        self.model = model
        self.loss = CoordFrameMseLoss(loss_radius_A)

    def forward(self, batch):
        x, y = batch
        y_hat = self.model(x)
        return self.loss(y_hat, y)

    def training_step(self, batch, _):
        loss = self.forward(batch)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        loss = self.forward(batch)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def test_step(self, batch, _):
        loss = self.forward(batch)
        self.log('test_loss', loss, sync_dist=True)
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
            max_dist_between_views_A: float,

            # Data loader parameters
            batch_size: int,
            train_epoch_size: int,
            val_epoch_size: int = None,
            test_epoch_size: int = None,
            num_workers: Optional[int] = None,
    ):
        super().__init__()

        self.origin_sampler = SqliteOriginSampler(origins_path)
        #self.origin_sampler = ParquetOriginSampler(origins_path)
        img_params = ImageParams(
                grid=Grid(
                    length_voxels=grid_length_voxels,
                    resolution_A=grid_resolution_A,
                ),
                channels=element_channels,
                element_radii_A=element_radii_A,
        )

        if num_workers is None:
            try:
                num_workers = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            except KeyError:
                num_workers = os.cpu_count()

        def make_dataset(low_seed, high_seed):
            return NeighborCountDatasetForCnn(
                    origin_sampler=self.origin_sampler,
                    img_params=img_params,
                    max_dist_A=max_dist_between_views_A,
                    low_seed=low_seed,
                    high_seed=high_seed,
            )

        def make_dataloader(low_seed, high_seed):
            return DataLoader(
                    make_dataset(low_seed, high_seed),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    # Think this might help with data transfer speeds, but 
                    # don't want to enable it until I've tested SQLite on its 
                    # own.
                    pin_memory=True,
            )

        i = train_epoch_size
        self._train_dataloader = make_dataloader(0, i)
        self._val_dataloader = None
        self._test_dataloader = None

        if val_epoch_size is not None:
            j = i + val_epoch_size
            self._val_dataloader = make_dataloader(i, j)

            if test_epoch_size is not None:
                k = j + test_epoch_size
                self._test_dataloader = make_dataloader(j, k)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def teardown(self, stage):
        self.origin_sampler.teardown()


def main():
    from lightning.pytorch.profilers import PyTorchProfiler
    from atompaint.diagnostics.shared_mem.profiler import SharedMemoryProfiler

    # Lightning recommends setting this to either 'medium' or 'high' (as
    # opposed to 'highest', which is the default) when training on GPUs with
    # support for the necessary acceleration.  I don't think there's a good way
    # of knowing a priori what the best setting should be; so I chose the
    # 'high' setting as a compromise to be optimized later.
    
    torch.set_float32_matmul_precision('high')

    LightningCLI(
            PredictorModule, DataModule,
            save_config_kwargs=dict(
                overwrite=True,
            ),
            trainer_defaults=dict( 
                #profiler=SharedMemoryProfiler(),
                #profiler=PyTorchProfiler(profile_memory=True),
            ),
    )


if __name__ == '__main__':
    main()


