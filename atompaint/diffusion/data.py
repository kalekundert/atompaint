import lightning as L
import torch
import numpy as np
import macromol_voxelize as mmvox
import logging

from macromol_gym_unsupervised import ImageParams, get_num_workers
from macromol_gym_unsupervised.torch import (
        MacromolImageDataset, MapDataset, InfiniteSampler,
)
from torch.utils.data import DataLoader, default_collate
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
                    collate_fn=collate_rngs,
                    
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

class BatchGenerator:
    """
    Wrap a collection of NumPy pseudorandom-number generators (PRNGs) such that 
    samples can easily be drawn from all of them at once.

    Instantiate this class with a list of NumPy PRNGs.  Then, any method 
    invoked on this class will automatically be invoked on all of those PRNGs, 
    and the results will collated into a PyTorch tensor.  For example::

        >>> from atompaint.diffusion.data import BatchGenerator
        >>> bg = BatchGenerator([
        ...     np.random.default_rng(0),
        ...     np.random.default_rng(1),
        ... ])
        >>> bg.uniform()
        tensor([0.6370, 0.5118], dtype=torch.float64)

    This class is meant to facilitate the idea that all of the randomness in 
    each training step should come a PRNG seeded based on the index of the 
    corresponding training example.  This PRNG would be created by the dataset, 
    used to build the training example, then returned in case the training loop 
    itself requires any more randomness.

    The benefit of this approach is that it's very robust.  The randomness does 
    not depend on the number of data loader processes, and every training 
    example can be reproduced without having to replay the whole dataset or 
    constantly log the PRNG state.  However, it's worth noting that from the 
    point-of-view of trying to get the best possible distribution of random 
    numbers, this approach is suboptimal.  PRNGs are only designed to output 
    high-quality randomness if seeded once.  There's no guarantee that two 
    PRNGs with different seeds won't output correlated values.  In practice, 
    though, this doesn't seem to be a significant issue.

    The `collate_rngs()` function can be used to make PyTorch dataloaders 
    automatically wrap collections of NumPy PRNGs with this class.
    """

    def __init__(self, rngs):
        self._rngs = rngs

    def __repr__(self):
        return f'<{self.__class__.__name__} n={len(self._rngs)}>'

    def __len__(self):
        return len(self._rngs)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)

        def method_wrapper(*args, **kwargs):
            return default_collate([
                getattr(rng, name)(*args, **kwargs)
                for rng in self._rngs
            ])

        return method_wrapper

    def pin_memory(self):
        return self

def collate_rngs(x):
    from torch.utils.data._utils.collate import collate, default_collate_fn_map

    def collate_rng_fn(x, *, collate_fn_map=None):
        return BatchGenerator(x)

    collate_fn_map = {
            np.random.Generator: collate_rng_fn,
            **default_collate_fn_map,
    }
    return collate(x, collate_fn_map=collate_fn_map)

def _prepare_inputs(x):
    rng = x['rng']
    x_clean = x['image']

    noise = rng.normal(size=x_clean.shape)
    noise = torch.from_numpy(noise).to(dtype=x_clean.dtype)

    return x_clean, noise, rng

