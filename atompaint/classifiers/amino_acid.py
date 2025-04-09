import lightning.pytorch as L
import torch
import torchyield as ty
import polars as pl
import numpy as np
import macromol_voxelize as mmvox

from atompaint.checkpoints import EvalModeCheckpointMixin
from atompaint.type_hints import OptFactory
from atompaint.metrics import TrainValTestMetrics
from macromol_gym_unsupervised import (
        ImageParams, make_unsupervised_image_sample, 
)
from macromol_dataframe import (
        assign_residue_ids,
)
from visible_residues import sample_visible_residues, Sphere
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Adam
from torchmetrics import MeanMetric, Accuracy
from blosum import BLOSUM
from itertools import chain
from more_itertools import one
from functools import cache

from typing import Optional, Literal

class OneStepAminoAcidClassificationTask(EvalModeCheckpointMixin, L.LightningModule):

    def __init__(
            self,
            classifier: Module,
            *,
            opt_factory: OptFactory = Adam,
    ):
        super().__init__()

        self.classifier = classifier
        self.optimizer = opt_factory(classifier.parameters())

        self.loss = CrossEntropyLoss()
        self.metrics = TrainValTestMetrics(lambda: {
            'accuracy': Accuracy(task='multiclass', num_classes=21),
            'blosum62': BlosumMetric(
                n=62,
                labels=get_amino_acid_labels(include_gap=True),
            ),
        })

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')

    def _step(self, batch, loop):
        x, y = batch
        y_hat = self.classifier(x)
        loss = self.loss(y_hat, y)

        self.log(f'{loop}/loss', loss, on_epoch=True)

        for name, metric in self.metrics.get(loop):
            metric(y_hat, y)
            self.log(f'{loop}/{name}', metric, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return self.optimizer


class TwoStepAminoAcidClassificationTask(EvalModeCheckpointMixin, L.LightningModule):

    def __init__(
            self,
            encoder: Module,
            classifier: Module,
            *,
            opt_factory: OptFactory = Adam,
    ):
        super().__init__()

        self.encoder = encoder
        self.classifier = classifier

        self.optimizer = opt_factory(chain(
            classifier.parameters(),
            encoder.parameters(),
        ))

        self.loss = CrossEntropyLoss()
        self.metrics = TrainValTestMetrics(lambda: {
            'accuracy': Accuracy(task='multiclass', num_classes=20),
            'blosum62': BlosumMetric(62),
        })

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')

    def _step(self, batch, loop):
        """
        The encoder will be called with a minibatch of images as input.  These 
        images can either be genuine macromolecular images, or the output of a 
        diffusion model.

        The classifier will be called with one image and a variably-sized batch 
        of coordinates.
        """

        x_latent = self.encoder(batch['image'])

        batch_size = batch['image'].shape[0]
        y, y_hat = [], []

        for b in range(batch_size):
            if not batch['coords'][b].numel():
                continue

            y_hat_b = self.classifier(x_latent[b], batch['coords'][b])
            y_b = batch['labels'][b]

            y.append(y_b)
            y_hat.append(y_hat_b)

        if not y:
            return torch.tensor(0.0)

        y = torch.cat(y)
        y_hat = torch.cat(y_hat)

        loss = self.loss(y_hat, y)

        self.log(f'{loop}/loss', loss, on_epoch=True)

        for name, metric in self.metrics.get(loop):
            metric(y_hat, y)
            self.log(f'{loop}/{name}', metric, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return self.optimizer


class CoordinateClassifier(Module):
    """
    Identify an amino acid given (i) the latent representation of an image and 
    (ii) an 3D vector giving the location of that amino acid.
    """

    def __init__(
            self,
            *,
            flatten_atoms: ty.LayerFactory,
            embed_coords: ty.LayerFactory,
            mlp: ty.LayerFactory,
    ):
        super().__init__()
        self.flatten_atoms = ty.module_from_layer(flatten_atoms)
        self.embed_coords = ty.module_from_layer(embed_coords)
        self.mlp = ty.module_from_layer(mlp)

    def forward(self, z_atoms, x_coords):
        z_atoms = self.flatten_atoms(z_atoms)
        z_coords = self.embed_coords(x_coords)
        return self.mlp(z_atoms + z_coords)


class BlosumMetric(MeanMetric):
    """
    Score predictions using a BLOSUM matrix.
    """

    def __init__(self, n=90, *, labels=None):
        super().__init__(nan_strategy='error')

        if labels is None:
            labels = get_amino_acid_labels()

        self.name1_from_label = dict(zip(labels['label'], labels['name1']))
        self.blosum = BLOSUM(n, default=0)

    def update(self, preds, target):
        b, n = preds.shape

        scores = torch.zeros((b, n), device=self.device)
        name1_from_label = self.name1_from_label

        for i, aa_i in enumerate(target):
            scores_i = self.blosum[name1_from_label[int(aa_i)]]
            for j in range(n):
                scores[i, j] = scores_i[name1_from_label[j]]

        probs = torch.softmax(preds.float(), dim=1)

        super().update(scores, probs)

def make_amino_acid_coords_full(
        db, db_cache, rng, zone_id,
        *,
        img_params: ImageParams,
        amino_acids: pl.DataFrame,
        bounding_sphere: Optional[Sphere] = None,
        max_residues: int = 10,
        coord_type: Literal['CA', 'sidechain'] = 'CA',
):
    """
    Arguments:
        amino_acids:
            A dataframe that described the amino acids in the dataset.  The 
            following columns are required:

            - `name3`: The three-letter code for the amino acid.
            - `name1`: The one-letter code for the amino acid.
            - `label`: The index in the "label" tensor for the amino acid.
            - `pick_prob`: The probability of including any particular amino 
              acid of this type in the dataset.
    """
    x = make_unsupervised_image_sample(
        db, db_cache, rng, zone_id,
        img_params=img_params,
    )

    atoms = x['image_atoms_a']

    # It's important that the residue id assignment is deterministic (i.e.  
    # `maintain_order=True`), because at several points we rely on sorting by 
    # the residue id to keep the whole algorithm deterministic.
    atoms = assign_residue_ids(atoms, maintain_order=True)
    atoms = balance_amino_acids(rng, atoms, amino_acids)
    atoms = remove_ambiguous_labels(atoms)

    visible = sample_visible_residues(
            rng=rng,
            atoms=atoms,
            grid=img_params.grid,
            n=max_residues,
            bounding_sphere=bounding_sphere,
    )

    if coord_type == 'CA':
        visible_ids = (
                visible
                .select(
                    pl.struct(
                        'residue_id',
                        pl.col('alt_ids').struct.field('CA').alias('alt_id'),
                    ),
                )
                .to_series()
        )
        coord_resns = (
                atoms
                .filter(
                    pl.col('atom_id') == 'CA',
                    pl.col('element') == 'C',
                    pl.struct('residue_id', 'alt_id').is_in(visible_ids),
                )
        )

    elif coord_type == 'sidechain':
        coord_resns = (
                visible
                .join(
                    atoms.select('residue_id', 'comp_id').unique(),
                    on='residue_id',
                )
        )

    else:
        raise ValueError(f"unknown coordinate type: {coord_type}")

    coord_labels = (
            coord_resns
            .join(
                amino_acids.select('name3', 'label'),
                left_on='comp_id',
                right_on='name3',
            )
            .sort('residue_id')
    )

    coords = coord_labels['x', 'y', 'z'].to_numpy().astype(np.float32)
    labels = coord_labels['label'].to_numpy().astype(np.uint8)

    return {
        **x,
        'coords': coords,
        'labels': labels,
    }

def make_amino_acid_coords(*args, **kwargs):
    x = make_amino_acid_coords_full(*args, **kwargs)
    return {
        'coords': x['coords'],
        'labels': x['labels'],
    }

def make_amino_acid_image_full(
        db, db_cache, rng, zone_id,
        *,
        img_params: ImageParams,
        amino_acids: pl.DataFrame,
        bounding_sphere: Optional[Sphere] = None,
        coord_type: Literal['CA', 'sidechain'] = 'CA',
        coord_radius_A: float = 1,
):
    x = make_amino_acid_coords_full(
        db=db,
        db_cache=db_cache,
        rng=rng,
        zone_id=zone_id,
        img_params=img_params,
        amino_acids=amino_acids,
        bounding_sphere=bounding_sphere,
        max_residues=1,
        coord_type=coord_type,
    )

    if len(x['coords']) == 0:
        img_aa = np.zeros(
                (1, *img_params.grid.shape),
                dtype=x['image'].dtype,
        )
        label = len(amino_acids)

    else:
        pseudoatoms = (
                pl.DataFrame({'xyz': x['coords']})
                .select(
                    pl.col('xyz')
                        .arr.to_struct(['x', 'y', 'z'])
                        .struct.unnest(),
                    radius_A=coord_radius_A,
                    channels=[0],
                )
        )
        img_aa = mmvox.image_from_all_atoms(
                pseudoatoms,
                mmvox.ImageParams(
                    channels=1,
                    grid=img_params.grid,
                    fill_algorithm=mmvox.FillAlgorithm.FractionVoxel,
                ),
        )
        label = one(x['labels'])

    img = np.concatenate((x['image'], img_aa))
    label = torch.tensor(label, dtype=torch.uint8)

    return {
            **x,
            'image': img,
            'label': label,
    }

def make_amino_acid_image(*args, **kwargs):
    x = make_amino_acid_image_full(*args, **kwargs)
    return x['image'], x['label']

def collate_amino_acid_coords(batch):
    from torch.utils.data import default_collate
    from torch.nested import nested_tensor

    return {
            'image': default_collate([
                x['image']
                for x in batch
            ]),
            'coords': nested_tensor([
                torch.from_numpy(x['coords'].copy())
                for x in batch
            ]),
            'labels': nested_tensor([
                # As of torch==2.5.1, nested tensors of unsigned integers can't 
                # be collated.  But signed integers smaller than 64-bits can't 
                # be used as the targets for cross-entropy loss.  So we have to 
                # cast to 64-bit signed integers here.
                torch.from_numpy(x['labels'].astype(np.int64))
                for x in batch
            ]),
    }


def balance_amino_acids(
        rng: np.random.Generator,
        atoms: pl.DataFrame,
        amino_acids: pl.DataFrame,
):
    """
    Randomly drop amino acids in order to, on average, get equal numbers of 
    each type.

    Arguments:
        rng:
            A pseudo-random number generator.

        atoms:
            A dataframe of atom coordinates.  The dataframe must contain the 
            following columns:

            - `residue_id`, e.g. as set by 
              :func:`macromol_dataframe.assign_residue_ids()`
            - `comp_id`

        amino_acids:
            A dataframe of amino acid features, particularly the probability of 
            using each amino acid as a training example.  The dataframe must 
            contain the following columns:

            - `name3`: The three-letter code for the amino acid.
            - `pick_prob`: The probability of picking an amino acid of the 
              given type.

    Returns:
        A dataframe matching the structure of the *atoms* argument, but without 
        any atoms belonging to residues that were dropped.  Atoms that don't 
        belong to any of the residue types contained in *amino_acid* are also 
        dropped.
    """
    residues = (
            atoms
            .group_by('residue_id', 'comp_id')
            .agg(cols=pl.struct(pl.col('*')))
            .join(
                amino_acids,
                left_on='comp_id',
                right_on='name3',
            )

            # This isn't strictly necessary, but it makes the algorithm 
            # deterministic, which is useful for debugging.
            .sort('residue_id')
    )
    atoms = (
            residues
            .with_columns(
                uniform=rng.uniform(0, 1, len(residues))
            )
            .filter(
                pl.col('uniform') < pl.col('pick_prob')
            )
            .explode('cols')
            .select(
                pl.col('cols').struct.unnest(),
            )
    )
    return atoms

def remove_ambiguous_labels(atoms: pl.DataFrame):
    # There are at least a few examples in the PDB (e.g. /6ycg/B/B/210) where a 
    # single residue has multiple amino acid types, via alternate atom 
    # locations.  There's no good way to assign a label to such residues, so we 
    # have to filter them out.
    return (
            atoms
            .filter(
                pl.col('comp_id').n_unique().over('residue_id') == 1,
            )
    )

@cache
def get_amino_acid_labels(include_gap=False):
    names = [
            ('A', 'ALA'),
            ('C', 'CYS'),
            ('D', 'ASP'),
            ('E', 'GLU'),
            ('G', 'GLY'),
            ('H', 'HIS'),
            ('I', 'ILE'),
            ('K', 'LYS'),
            ('L', 'LEU'),
            ('M', 'MET'),
            ('N', 'ASN'),
            ('F', 'PHE'),
            ('P', 'PRO'),
            ('Q', 'GLN'),
            ('R', 'ARG'),
            ('S', 'SER'),
            ('T', 'THR'),
            ('V', 'VAL'),
            ('W', 'TRP'),
            ('Y', 'TYR'),
    ]
    if include_gap:
        names += [('-', '---')]

    name1, name3 = zip(*names)
    return (
            pl.DataFrame({'name1': name1, 'name3': name3})
            .with_row_index('label')
    )

@cache
def get_unbiased_amino_acid_labels():
    return (
            get_amino_acid_labels()
            .with_columns(pick_prob=1)
    )

@cache
def get_uniprot_amino_acid_labels():
    return (
            get_amino_acid_labels()
            .join(
                get_uniprot_amino_acid_pick_probs(),
                on='name1',
            )
    )

@cache
def get_uniprot_amino_acid_pick_probs():
    # These weights are derived from the UniProt/Swiss-Prot database.  See 
    # experiment #128 for details.
    p = {
            'L': 0.11453502159289752,
            'A': 0.13386337793145517,
            'G': 0.15624600177894662,
            'V': 0.16118581342478136,
            'E': 0.1645282383034253,
            'S': 0.16600105778820984,
            'I': 0.18706163473325987,
            'K': 0.1905725723883488,
            'R': 0.199919079690039,
            'D': 0.2023106475552744,
            'T': 0.20600006984383082,
            'P': 0.23282767977849814,
            'N': 0.2719957773417831,
            'Q': 0.28109168278842744,
            'F': 0.2856307622332959,
            'Y': 0.377903200412642,
            'M': 0.45816883255612895,
            'H': 0.48499835139750763,
            'C': 0.7963700322377675,
            'W': 1.0,
    }
    return pl.DataFrame({
        'name1': p.keys(),
        'pick_prob': p.values(),
    })

@cache
def get_expt_107_amino_acid_labels():
    return (
            get_amino_acid_labels()
            .join(
                get_expt_107_amino_acid_pick_probs(),
                on='name1',
            )
    )

@cache
def get_expt_107_amino_acid_pick_probs():
    # These weights are derived from the version of the `macromol_gym` database 
    # created in experiment #107.  These numbers come specifically from one 
    # pass through the training set with 35Å images, but I found that I get 
    # similar results with 15Å and 25Å images as well.
    n = {
            'L': 1422900,
            'A': 1215235,
            'G': 1126967,
            'V': 1098345,
            'I': 940774,
            'S': 868068,
            'T': 796237,
            'F': 738772,
            'R': 715943,
            'Y': 612452,
            'P': 585516,
            'E': 560579,
            'D': 558698,
            'N': 548372,
            'K': 506354,
            'Q': 431299,
            'M': 392060,
            'H': 389110,
            'W': 257452,
            'C': 232430,
    }
    df = pl.DataFrame({
        'name1': n.keys(),
        'n': n.values(),
    })
    return pick_probs_from_counts(df)

def pick_probs_from_counts(df):
    return (
            df
            .with_columns(
                weight=pl.sum('n') / pl.col('n'),
            )
            .with_columns(
                pick_prob=pl.col('weight') / pl.col('weight').max(),
            )
            .drop('weight')
    )
