"""\
Usage:
    ap_transform_pred <config> [-d]

Arguments:
    <config>
        A YAML file specifying all the hyperparameters for a training run.  The 
        following keys should be present: 

        trainer: Arguments to the `Trainer` class.
        model: Arguments to the `PredictorModule` class.
        data: Arguments to the `DataModule` class.

Options:
    -d --dry-run
        Run only a few iterations, and don't record any logs.  This is meant to 
        help test that the program runs before launching a long run.  Note that 
        this option doesn't affect the batch size, so even a small run might 
        use a lot of resources.
"""

import lightning.pytorch as pl
import torch.nn.functional as F
import os

from .models import (
        TransformationPredictor, ViewPairEncoder, ViewPairClassifier,
        NonequivariantViewPairEncoder, NonequivariantViewPairClassifier,
        make_fourier_classifier_field_types, make_linear_fourier_layer,
        make_nonequivariant_linear_relu_dropout_layer,
)
from .datasets.origins import SqliteOriginSampler, copy_origins_to_tmp
from .datasets.classification import (
        CnnViewIndexDataset, make_cube_face_frames_ab,
)
from atompaint.config import load_train_config
from atompaint.datasets.voxelize import ImageParams, Grid
from atompaint.datasets.samplers import RangeSampler, InfiniteSampler
from atompaint.encoders.cnn import FourierCnn, NonequivariantCnn
from atompaint.encoders.resnet import (
        ResNet, make_escnn_example_block, make_alpha_block,
)
from atompaint.encoders.densenet import (
        DenseNet, make_fourier_growth_type,
)
from atompaint.encoders.layers import (
        make_conv_layer, make_conv_fourier_layer, make_gated_nonlinearity,
        make_top_level_field_types, make_polynomial_field_types,
        make_fourier_field_types,
)
from atompaint.pooling import FourierExtremePool3D
from atompaint.nonlinearities import leaky_hard_shrink, first_hermite
from atompaint.checkpoints import EvalModeCheckpointMixin
from atompaint.utils import parse_so3_grid
from escnn.nn import FourierPointwise
from escnn.gspaces import rot3dOnR3
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from docopt import docopt
from pathlib import Path
from functools import partial
from typing import Optional

# Not going to use docker/singularity for now.  It'll be good to make a 
# container when I'm not making changes to atompaint anymore, but until then, 
# I'd have to make a new container for every commit.  

PREDICTOR_MODULES = {}
NONLINEARITIES = {
        # rectifier
        'relu': F.relu,
        'gelu': F.gelu,
        'elu': F.elu,
        'silu': F.silu,
        'mish': F.mish,

        # linear
        'hardshrink': F.hardshrink,
        'leaky_hardshrink': leaky_hard_shrink,
        'softshrink': F.softshrink,
        'tanhshrink': F.tanhshrink,

        # sigmoid
        'hardtanh': F.hardtanh,
        'softsign': F.softsign,
        'sigmoid': F.sigmoid,
        'tanh': F.tanh,
        'hermite': first_hermite,
}

class PredictorModule(EvalModeCheckpointMixin, pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters(ignore='model')

        self.model = model
        self.loss = CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=6)
        self.optimizer = Adam(model.parameters())

    def __init_subclass__(cls, *, factory_key, **kwargs):
        super().__init_subclass__(**kwargs)
        PREDICTOR_MODULES[factory_key] = cls

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

    @property
    def in_type(self):
        return self.model.in_type

class CnnPredictor(PredictorModule, factory_key='cnn'):

    def __init__(
            self, *,
            frequencies: int,
            conv_channels: list[int],
            conv_field_of_view: int | list[int],
            conv_stride: int | list[int],
            conv_padding: int | list[int],
            mlp_channels: int | list[int],
    ):
        encoder = ViewPairEncoder(
                FourierCnn(
                    channels=conv_channels,
                    conv_field_of_view=conv_field_of_view,
                    conv_stride=conv_stride,
                    conv_padding=conv_padding,
                    frequencies=frequencies,
                ),
        )
        so3 = encoder.out_type.fibergroup
        classifier = ViewPairClassifier(
                layer_types=make_fourier_classifier_field_types(
                    in_type=encoder.out_type,
                    channels=mlp_channels,
                    max_frequencies=frequencies,
                ),
                layer_factory=partial(
                    make_linear_fourier_layer,
                    ift_grid=so3.grid('thomson_cube', N=4),
                ),
                logits_max_freq=frequencies,

                # This has to be the same as the grid used to construct the 
                # dataset.  For now, I've just hard-coded the 'cube' grid.
                logits_grid=so3.sphere_grid('cube'),
        )
        model = TransformationPredictor(
                encoder=encoder,
                classifier=classifier,
        )
        super().__init__(model)

class NonequivariantCnnPredictor(PredictorModule, factory_key='cnn-noneq'):

    def __init__(
            self,
            conv_channels: list[int],
            conv_kernel_sizes: int | list[int],
            conv_strides: int | list[int],
            conv_paddings: int | list[int],
            pool_sizes: int | list[int],
            mlp_channels: int | list[int],
    ):
        assert mlp_channels[-1] == 6

        encoder = NonequivariantViewPairEncoder(
                NonequivariantCnn(
                    channels=conv_channels,
                    kernel_sizes=conv_kernel_sizes,
                    strides=conv_strides,
                    paddings=conv_paddings,
                    pool_sizes=pool_sizes,
                ),
        )
        classifier = NonequivariantViewPairClassifier(
                channels=mlp_channels,
                layer_factory=partial(
                    make_nonequivariant_linear_relu_dropout_layer,
                    drop_rate=0.2,
                ),
        )
        model = TransformationPredictor(
                encoder=encoder,
                classifier=classifier,
        )
        super().__init__(model)


class ResNetPredictor(PredictorModule, factory_key='resnet'):

    def __init__(
            self,
            *,
            block_type: str,
            resnet_outer_channels: list[int],
            resnet_inner_channels: list[int],
            polynomial_terms: int | list[int] = 0,
            max_frequency: int,
            grid: str,
            block_repeats: int,
            pool_factors: int | list[int],
            final_conv: int = 0,
            mlp_channels: list[int],
            equivariant_mlp: bool = True,
    ):
        gspace = rot3dOnR3()
        so3 = gspace.fibergroup
        grid_elements = parse_so3_grid(so3, grid)

        if block_type == 'escnn':
            outer_types = make_top_level_field_types(
                    gspace=gspace, 
                    channels=resnet_outer_channels,
                    make_nontrivial_field_types=partial(
                        make_polynomial_field_types,
                        terms=polynomial_terms,
                    ),
            )
            inner_types = make_fourier_field_types(
                    gspace=gspace, 
                    channels=resnet_inner_channels,
                    max_frequencies=max_frequency,
            )
            initial_layer_factory = make_conv_layer
            block_factory = partial(
                    make_escnn_example_block,
                    grid=grid_elements,
            )

        elif block_type == 'alpha':
            outer_types = make_top_level_field_types(
                    gspace=gspace, 
                    channels=resnet_outer_channels,
                    make_nontrivial_field_types=partial(
                        make_fourier_field_types,
                        max_frequencies=max_frequency,
                    ),
            )
            inner_types = make_fourier_field_types(
                    gspace=gspace, 
                    channels=resnet_inner_channels,
                    max_frequencies=max_frequency,
            )
            initial_layer_factory = partial(
                    make_conv_fourier_layer,
                    ift_grid=grid_elements,
            )
            block_factory = partial(
                    make_alpha_block,
                    grid=grid_elements,
            )
            assert polynomial_terms == 0

        else:
            raise ValueError(f"unknown block type: {block_type}")

        if final_conv:
            final_layer_factory = partial(
                    make_conv_layer,
                    kernel_size=final_conv,
            )
        else:
            final_layer_factory = None

        encoder = ViewPairEncoder(
                ResNet(
                    outer_types=outer_types,
                    inner_types=inner_types,
                    initial_layer_factory=initial_layer_factory,
                    final_layer_factory=final_layer_factory,
                    block_factory=block_factory,
                    block_repeats=block_repeats,
                    pool_factors=pool_factors,
                ),
        )

        if equivariant_mlp:
            classifier = ViewPairClassifier(
                    layer_types=make_fourier_classifier_field_types(
                        in_type=encoder.out_type,
                        channels=mlp_channels,
                        max_frequencies=max_frequency,
                    ),
                    layer_factory=partial(
                        make_linear_fourier_layer,
                        ift_grid=grid_elements,
                    ),
                    logits_max_freq=max_frequency,

                    # This has to be the same as the grid used to construct the 
                    # dataset.  For now, I've just hard-coded the 'cube' grid.
                    logits_grid=so3.sphere_grid('cube'),
            )
        else:
            assert mlp_channels[-1] == 6
            classifier = NonequivariantViewPairClassifier(
                    channels=mlp_channels,
                    layer_factory=partial(
                        make_nonequivariant_linear_relu_dropout_layer,
                        drop_rate=0.2,
                    ),
            )

        model = TransformationPredictor(
                encoder=encoder,
                classifier=classifier,
        )
        super().__init__(model)

class DenseNetPredictor(PredictorModule, factory_key='densenet'):

    def __init__(
            self,
            *,
            transition_channels: list[int],
            growth_channels: int,
            grid: str,
            max_frequency: int,
            fourier_nonlinearity: str = 'leaky_hardshrink',
            block_depth: int | list[int],
            pool_factors: int | list[int],
            final_conv: int = 0,
            mlp_channels: list[int],
            equivariant_mlp: bool = True,
    ):
        gspace = rot3dOnR3()
        so3 = gspace.fibergroup
        grid_elements = parse_so3_grid(so3, grid)

        # Things that are hard-coded for now:
        #
        # - Using spectral regular representations at all levels.  Where 
        #   Fourier transforms are required, I could also try spectral induced 
        #   representations (i.e. quotient space Fourier transforms).  
        #   Everywhere else, I could try polynomial representations.
        #
        # - Using the same max frequency everywhere.
        #
        # - Dense layer nonlinearities: gated first, Fourier second.  It might 
        #   be better to use only one or the other, or even something else.

        if final_conv:
            final_layer_factory = partial(
                    make_conv_layer,
                    kernel_size=final_conv,
            )
        else:
            final_layer_factory = None

        encoder = ViewPairEncoder(
                DenseNet(
                    transition_types=make_top_level_field_types(
                        gspace=gspace,
                        channels=transition_channels,
                        make_nontrivial_field_types=partial(
                            make_fourier_field_types,
                            max_frequencies=max_frequency,
                            unpack=True,
                        ),
                    ),
                    growth_type_factory=partial(
                        make_fourier_growth_type,
                        gspace=gspace,
                        channels=growth_channels,
                        max_frequency=max_frequency,
                        unpack=True,
                    ),
                    initial_layer_factory=partial(
                            make_conv_fourier_layer,
                            ift_grid=grid_elements,
                    ),
                    final_layer_factory=final_layer_factory,
                    nonlin1_factory=make_gated_nonlinearity,
                    nonlin2_factory=partial(
                        FourierPointwise,
                        grid=grid_elements,
                        function=NONLINEARITIES[fourier_nonlinearity],
                    ),
                    pool_factory=lambda in_type, pool_factor: \
                            FourierExtremePool3D(
                                in_type,
                                grid=grid_elements,
                                kernel_size=pool_factor,
                            ),
                    pool_factors=pool_factors,
                    block_depth=block_depth,
                ),
        )

        if equivariant_mlp:
            classifier = ViewPairClassifier(
                    layer_types=make_fourier_classifier_field_types(
                        in_type=encoder.out_type,
                        channels=mlp_channels,
                        max_frequencies=max_frequency,
                    ),
                    layer_factory=partial(
                        make_linear_fourier_layer,
                        ift_grid=grid_elements,
                    ),
                    logits_max_freq=max_frequency,

                    # This has to be the same as the grid used to construct the 
                    # dataset.  For now, I've just hard-coded the 'cube' grid.
                    logits_grid=so3.sphere_grid('cube'),
            )
        else:
            assert mlp_channels[-1] == 6
            classifier = NonequivariantViewPairClassifier(
                    channels=mlp_channels,
                    layer_factory=partial(
                        make_nonequivariant_linear_relu_dropout_layer,
                        drop_rate=0.2,
                    ),
            )

        model = TransformationPredictor(
                encoder=encoder,
                classifier=classifier,
        )
        super().__init__(model)

class DataModule(pl.LightningDataModule):

    def __init__(
            self, *,
            # Origin parameters
            origins_path: str,

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

        # - We aren't going to manually clean up the temporary directory that
        #   we're copying the origins into; we're relying on it cleaning itself
        #   up when the whole data module is finalized (i.e. right before the
        #   interpreter shuts down).  This is not guaranteed to happen, so the
        #   temporary file might persist until the OS cleans it up.
        #
        # - We have to store the temporary file in an instance attribute, even
        #   though we won't reference it again.  If we used a local variable,
        #   the file would get cleaned up as soon as that variable went out of 
        #   scope (i.e. at the end of the constructor).
        self._tmp_origins_path = copy_origins_to_tmp(Path(origins_path))

        origin_sampler = SqliteOriginSampler(self._tmp_origins_path.name)
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
                origin_sampler=origin_sampler,
                img_params=img_params,
                recording_path=recording_path,
        )

        def make_dataloader(sampler):

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
                    persistent_workers=True,

                    pin_memory=True,
                    drop_last=True,
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

def main():
    args = docopt(__doc__)
    config_path = Path(args['<config>'])
    c = load_train_config(
            config_path, predictor_factory, DataModule,
            dry_run=args['--dry-run'],
    )
    c.trainer.fit(c.model, c.data, ckpt_path='last')

def predictor_factory(**kwargs):
    factory_key = kwargs.pop('architecture', 'cnn')
    factory = PREDICTOR_MODULES[factory_key]
    return factory(**kwargs)

if __name__ == '__main__':
    main()


