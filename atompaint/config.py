import os
import yaml
import time
import torch
import lightning.pytorch as pl
import logging

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from statistics import fmean
from more_itertools import pairwise
from dataclasses import dataclass
from reprfunc import repr_from_init

from typing import Optional

@dataclass
class Config:
    trainer: pl.Trainer
    model: pl.LightningModule
    data: pl.LightningDataModule

@dataclass
class ComputeConfig:
    train_command: Optional[str]
    num_cpus: int
    time_h: int
    memory_gb: int


class ConfigError(Exception):
    pass

class SlurmTimeoutCallback:
    """\
    Monitor how long it takes to process each epoch, and terminate gracefully 
    when it seems like there's not enough time for another.
    """

    def __init__(self, margin_factor=1):
        self.times = []
        self.time_limit = float(os.environ['SLURM_JOB_END_TIME'])
        self.margin_factor = margin_factor

    def on_validation_end(self, trainer, _):
        self.times.append(time.monotonic())
        if len(self.times) < 2:
            return

        time_per_epoch_s = fmean([y - x for x, y in pairwise(self.times)])
        time_per_epoch_s *= self.margin_factor
        time_remaining_s = self.time_limit - time.time()

        if time_per_epoch_s > time_remaining_s:
            # Copy requeue logic from lightning.
            trainer.should_stop = True

    __repr__ = repr_from_init

def load_config(path, model_cls, data_cls, **trainer_kwargs):
    conf = yaml.safe_load(path.read_text())
    conf.pop('compute', None)

    log_levels = {
            'info': logging.INFO,
            'debug': logging.DEBUG,
    }
    log_level = log_levels[conf.pop('log', 'info')]
    logging.basicConfig(level=log_level)

    trainer_kwargs |= conf.pop('trainer', {})

    # Lightning recommends setting this to either 'medium' or 'high' (as
    # opposed to 'highest', which is the default) when training on GPUs with
    # support for the necessary acceleration.  I don't think there's a good way
    # of knowing a priori what the best setting should be; so I chose the
    # 'high' setting as a compromise to be optimized later.
    
    prec = trainer_kwargs.pop('float32_precision', 'high')
    torch.set_float32_matmul_precision(prec)

    trainer = pl.Trainer(
            callbacks=[
                ModelCheckpoint(
                    save_last=True,
                    every_n_epochs=1,
                ),
            ],
            logger=TensorBoardLogger(
                save_dir=path.parent,
                name='workspace',
                version=path.stem,
                default_hp_metric=False,
            ),
            **trainer_kwargs,
    )
    model = model_cls(**conf.pop('model'))
    data = data_cls(**conf.pop('data'))

    if conf:
        raise ConfigError(f"{path}: unexpected keys: {', '.join(map(repr, conf))}")

    return Config(trainer, model, data)

def load_compute_config(path):
    # These config options are read when submitting a job, not when running a 
    # job.  Loading the main config can take a while, because we have to 
    # initialize the whole model.
    conf = yaml.safe_load(path.read_text()).get('compute', {})
    compute_conf = ComputeConfig(
            train_command=conf.pop('train_cmd'),
            num_cpus=conf.pop('cpus', 16),
            time_h=conf.pop('time_h', 2),
            memory_gb=conf.pop('memory_gb', 16),
    )

    if conf:
        raise ConfigError(f"{path}: unexpected keys: {', '.join(map(repr, conf))}")

    return compute_conf

