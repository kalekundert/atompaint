import lightning.pytorch as pl
import os

from .models import TransformationPredictor
from .loss import CoordFrameMseLoss
from .datasets import NeighborCountDatasetForCnn, load_origins
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

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
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
            epoch_size: int,
            num_workers: Optional[int] = None,
    ):
        super().__init__()
        self.dataset = NeighborCountDatasetForCnn(
                origins=load_origins(origins_path),
                img_params=ImageParams(
                    grid=Grid(
                        length_voxels=grid_length_voxels,
                        resolution_A=grid_resolution_A,
                    ),
                    channels=element_channels,
                    element_radii_A=element_radii_A,
                ),
                max_dist_A=max_dist_between_views_A,
                epoch_size=epoch_size,
        )

        if num_workers is None:
            try:
                # Don't exceed the number of cores allocated to the job, and
                # don't forget to count the main process.
                num_workers = int(os.environ['SLURM_JOB_CPUS_PER_NODE']) - 1
            except KeyError:
                num_workers = os.cpu_count()

        self.data_loader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                num_workers=num_workers,
        )

    def train_dataloader(self):
        return self.data_loader

def main():
    from lightning.pytorch.profilers import PyTorchProfiler

    LightningCLI(
            PredictorModule, DataModule,
            save_config_kwargs=dict(
                overwrite=True,
            ),
            #trainer_defaults=dict( 
            #    profiler=PyTorchProfiler(profile_memory=True),
            #),
    )


if __name__ == '__main__':
    main()


