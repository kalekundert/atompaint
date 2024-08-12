import lightning as L
import torch
import numpy as np
import macromol_voxelize as mmvox
import logging

from macromol_gym_unsupervised import ImageParams, get_num_workers
from macromol_gym_unsupervised.torch import (
        MacromolImageDataset, MapDataset, InfiniteSampler,
)
from torch.utils.data import DataLoader
from pathlib import Path

from typing import Optional
from numpy.typing import ArrayLike

log = logging.getLogger('atompaint')

class MacromolImageDiffusionData(L.LightningDataModule):

    def __init__(
            self,
            db_path: Path,
            *,

            # Image parameters:
            grid_length_voxels: int,
            grid_resolution_A: float,
            atom_radius_A: Optional[float] = None,
            element_channels: list[str],
            normalize_mean: ArrayLike = 0,
            normalize_std: ArrayLike = 1,

            # Data loader parameters:
            batch_size: int,
            train_epoch_size: Optional[int] = None,
            val_epoch_size: Optional[int] = None,
            test_epoch_size: Optional[int] = None,
            identical_epochs: bool = False,
            num_workers: Optional[int] = None,
    ):
        super().__init__()

        grid = mmvox.Grid(
                length_voxels=grid_length_voxels,
                resolution_A=grid_resolution_A,
        )

        if atom_radius_A is None:
            atom_radius_A = grid_resolution_A / 2

        datasets = {
                split: MacromolImageDataset(
                    db_path=db_path,
                    split=split,
                    img_params=ImageParams(
                        grid=grid,
                        atom_radius_A=atom_radius_A,
                        element_channels=element_channels,
                        normalize_mean=normalize_mean,
                        normalize_std=normalize_std,
                    ),
                )
                for split in ['train', 'val', 'test']
        }
        num_workers = get_num_workers(num_workers)

        def make_dataloader(split, epoch_size):
            log.info("configure dataloader: split=%s num_workers=%d", split, num_workers)

            dataset = MapDataset(
                    dataset=datasets[split],
                    func=_prepare_inputs,
            )
            sampler = InfiniteSampler(
                    epoch_size or len(datasets[split]),
                    shuffle=True,
                    shuffle_size=len(datasets[split]),
                    increment_across_epochs=(
                        (split == 'train') and (not identical_epochs)
                    ),
            )

            return DataLoader(
                    dataset=dataset,
                    sampler=sampler,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    
                    # For some reason I don't understand, my worker processes
                    # get killed by SIGABRT if I use the 'fork' context.  The
                    # behavior is very sensitive to all sorts of small changes
                    # in the code (e.g. `debug()` calls), which makes me think
                    # it's some sort of race condition.
                    multiprocessing_context='spawn' if num_workers else None,

                    pin_memory=True,
                    drop_last=True,
            )

        self._train_dataloader = make_dataloader('train', train_epoch_size)
        self._val_dataloader = make_dataloader('val', val_epoch_size)
        self._test_dataloader = make_dataloader('test', test_epoch_size)

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader


def _prepare_inputs(x):
    rng = x['rng']
    x_clean = x['image']

    noise = rng.normal(size=x_clean.shape)
    noise = torch.from_numpy(noise).to(dtype=x_clean.dtype)

    # It's important to use `rng.random()` instead of `rng.uniform()` here.  
    # The reason is that the [Karras2022] algorithm requires uniform random 
    # 32-bit floating point values in the range [0, 1) (i.e. excluding 1), and 
    # only `rng.random()` can satisfy this invariant.
    #
    # The reason why [Karras2022] requires values <1 is that, when trying to 
    # convert a randomly distributed value to a normally distributed one, 1 
    # becomes +inf.  Needless to say, +inf leads to more problems and 
    # ultimately crashes the training run.  (Specifically, it leads to NaNs in 
    # the batch normalization routines.)  Note that 0, which becomes -inf, 
    # doesn't cause problems, because the algorithm uses the exponent of this 
    # value.
    #
    # The reason why `rng.uniform()` fails to satisfy the above invariant is 
    # that it can only generate 64-bit values.  Since we need 32-bit values, we 
    # have to downcast the output of this function.  Some 64-bit values that 
    # are less than 1 become exactly 1 when downcasted to 32-bits, thus 
    # violating the invariant.
    t_uniform = rng.random(dtype=np.float32)
    t_uniform = torch.tensor(t_uniform)

    assert torch.all(t_uniform < 1)

    return x_clean, noise, t_uniform
