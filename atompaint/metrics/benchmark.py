import macromol_voxelize as mmvox
import polars as pl
import numpy as np
import opensimplex
import smartcall
import os
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm
from matplotlib.ticker import LogFormatterMathtext
from macromol_gym_unsupervised import (
        make_unsupervised_sample, make_unsupervised_image_sample,
        image_from_atoms,
)
from macromol_gym_unsupervised.torch import MacromolDataset
from macromol_dataframe import transform_atom_coords
from torch.utils.data import DataLoader
from smartcall import KwOnly
from dataclasses import dataclass
from collections import defaultdict
from functools import partial
from itertools import islice
from tqdm import tqdm
from pathlib import Path
from math import sqrt

class Perturbation:
    algorithm_name: str
    parameter_name: str
    parameter_value: str

    def __init__(self, algorithm_name, parameter_name, parameter_value):
        self.algorithm_name = algorithm_name
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value

    def iter_image_batches(self):
        raise NotImplementedError

    @property
    def num_batches(self):
        return len(self.iter_image_batches())


class PerturbAtoms(Perturbation):

    def __init__(
            self,
            *,
            perturb_atoms,
            parameter_name,
            parameter_value,
            db_path,
            img_params,
            batch_size=16,
    ):
        super().__init__(
                perturb_atoms.__name__,
                parameter_name,
                parameter_value,
        )
        self.perturb_atoms = perturb_atoms
        self.db_path = db_path
        self.img_params = img_params
        self.batch_size = batch_size

    def iter_image_batches(self):
        data = MacromolDataset(
                db_path=self.db_path,
                split='val',
                make_sample=partial(
                    _make_perturbed_atom_sample,
                    perturb_atoms=partial(
                        self.perturb_atoms,
                        **{self.parameter_name: self.parameter_value},
                    ),
                    img_params=self.img_params,
                ),
        )
        return DataLoader(
                dataset=data,
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=os.cpu_count() // 2,
                drop_last=True,
        )


class PerturbVoxels(Perturbation):

    def __init__(
            self,
            *,
            perturb_voxels,
            parameter_name,
            parameter_value,
            db_path,
            img_params,
            batch_size=16,
    ):
        super().__init__(
                perturb_voxels.__name__,
                parameter_name,
                parameter_value,
        )
        self.perturb_voxels = perturb_voxels
        self.db_path = db_path
        self.img_params = img_params
        self.batch_size = batch_size

    def iter_image_batches(self):
        perturb = partial(
                self.perturb_voxels,
                **{self.parameter_name: self.parameter_value},
        )
        data = MacromolDataset(
                db_path=self.db_path,
                split='val',
                make_sample=partial(
                    _make_perturbed_voxel_sample,
                    perturb_voxels=perturb,
                    img_params=self.img_params,
                ),
        )
        return DataLoader(
                dataset=data,
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=os.cpu_count() // 2,
                drop_last=True,
        )


class PerturbBatches(Perturbation):

    def __init__(
            self,
            *,
            perturb_batch,
            parameter_name,
            parameter_value,
            db_path,
            img_params,
            batch_size=16,
    ):
        super().__init__(
                perturb_batch.__name__,
                parameter_name,
                parameter_value,
        )
        self.perturb_batch = perturb_batch
        self.db_path = db_path
        self.img_params = img_params
        self.batch_size = batch_size

    def iter_image_batches(self):
        data = MacromolDataset(
                db_path=self.db_path,
                split='val',
                make_sample=partial(
                    _make_unperturbed_sample,
                    img_params=self.img_params,
                ),
        )
        data_loader = DataLoader(
                dataset=data,
                batch_size=self.batch_size,
                pin_memory=True,
                num_workers=os.cpu_count() // 2,
                drop_last=True,
        )

        def _generate():
            for x in data_loader:
                yield smartcall.call(
                        self.perturb_batch,
                        KwOnly('seeds', x['seed']),
                        KwOnly('imgs', x['image']),
                        KwOnly(self.parameter_name, self.parameter_value, required=True),
                )

        return _SizedIterable(_generate(), len(data_loader))


class TruncatedPerturbation(Perturbation):

    def __init__(self, perturbation, max_batches):
        # Don't call super().__init__() — attributes are delegated to _inner.
        self._inner = perturbation
        self.max_batches = max_batches

    @property
    def algorithm_name(self): return self._inner.algorithm_name
    @property
    def parameter_name(self):  return self._inner.parameter_name
    @property
    def parameter_value(self): return self._inner.parameter_value
    @property
    def batch_size(self):      return self._inner.batch_size
    @property
    def img_params(self):      return self._inner.img_params

    def iter_image_batches(self):
        it = self._inner.iter_image_batches()
        n = min(self.max_batches, len(it))
        return _SizedIterable(islice(it, self.max_batches), n)



@dataclass(order=True)
class Interval:
    start: int  # inclusive minibatch index
    end: int    # exclusive minibatch index


class MetricInterval:
    def __init__(self, interval, replicate, metric):
        self.interval = interval
        self.replicate = replicate
        self.metric = metric



class _SizedIterable:
    def __init__(self, it, length):
        self._it = it
        self._len = length

    def __len__(self):
        return self._len

    def __iter__(self):
        return iter(self._it)



def record_metrics(
        perturbations,
        intervals,
        metric_factories,
        *,
        save_images=False,
        img_dir=Path('images'),
):
    rows = []

    for perturbation in tqdm(perturbations, desc='all perturbations'):
        replicate_counter = defaultdict(int)
        active = []
        first_batch = True

        for batch_idx, x in enumerate(
                tqdm(perturbation.iter_image_batches(),
                     desc=perturbation.algorithm_name,
                     leave=False)
        ):
            if save_images and first_batch:
                img_dir.mkdir(exist_ok=True)
                filename = (
                        f'{perturbation.algorithm_name}'
                        f'_{perturbation.parameter_name}'
                        f'_{perturbation.parameter_value:.4g}.npz'
                )
                mmvox.write_npz(
                        img_dir / filename,
                        np.asarray(x),
                        perturbation.img_params.grid,
                )
                first_batch = False

            for length, ivs in intervals.items():
                r = replicate_counter[length]
                if r < len(ivs) and ivs[r].start == batch_idx:
                    for factory in metric_factories:
                        active.append(MetricInterval(ivs[r], r, factory()))
                    replicate_counter[length] += 1

            for mi in active:
                mi.metric.update(x)

            still_active = []
            for mi in active:
                if batch_idx + 1 == mi.interval.end:
                    rows.append(dict(
                            algorithm_name=perturbation.algorithm_name,
                            parameter_name=perturbation.parameter_name,
                            parameter_value=perturbation.parameter_value,
                            metric_name=_get_metric_name(mi.metric),
                            metric_value=mi.metric.compute().item(),
                            n=(mi.interval.end - mi.interval.start)
                              * perturbation.batch_size,
                            replicate=mi.replicate,
                    ))
                else:
                    still_active.append(mi)
            active = still_active

    return pl.DataFrame(rows)

def plot_metrics(df, *, perturbation_titles=None, metric_titles=None):
    algorithms = df['algorithm_name'].unique().sort().to_list()
    metrics = df['metric_name'].unique().sort().to_list()

    n_rows = len(metrics)
    n_cols = len(algorithms)
    cmap = 'rainbow'

    fig, axes = plt.subplots(
            figsize=(3 * n_cols, 5 * n_rows),
            nrows=n_rows,
            ncols=n_cols,
            sharex='all',
            sharey='row',
            layout='constrained',
            squeeze=False,
    )

    for col, algorithm in enumerate(algorithms):
        alg_df = df.filter(pl.col('algorithm_name') == algorithm)
        param_name = alg_df['parameter_name'][0]
        ticks = alg_df['parameter_value'].unique().sort()
        norm = _make_norm(ticks.to_numpy())

        alg_title = (perturbation_titles or {}).get(algorithm, algorithm)
        is_log = isinstance(norm, LogNorm)
        fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                ax=list(axes[:, col]),
                label=f'{alg_title}\n({param_name})',
                ticks=ticks if not is_log else None,
                format=LogFormatterMathtext() if is_log else None,
                location='top',
        )

        for row, metric in enumerate(metrics):
            metric_df = alg_df.filter(pl.col('metric_name') == metric)
            metric_title = (metric_titles or {}).get(metric, metric)
            ax = axes[row, col]

            sns.lineplot(
                    metric_df,
                    x='n',
                    y='metric_value',
                    hue='parameter_value',
                    palette=cmap,
                    hue_norm=norm,
                    errorbar=('ci', 95),
                    ax=ax,
                    legend=False,
            )
            ax.set_xscale('log')
            ax.set_xlabel('# samples')
            ax.set_ylabel(metric_title if col == 0 else '')

def make_standard_perturbations(
        db_path,
        img_params,
        *,
        batch_size=16,
):
    perturbations = []

    perturbations += [
            PerturbAtoms(
                perturb_atoms=jitter_atoms,
                parameter_name='noise_A',
                parameter_value=x,
                db_path=db_path,
                img_params=img_params,
                batch_size=batch_size,
            )
            for x in np.linspace(0, 2, 11)
    ]
    perturbations += [
            PerturbAtoms(
                perturb_atoms=delete_random_atoms,
                parameter_name='num_deletions',
                parameter_value=x,
                db_path=db_path,
                img_params=img_params,
                batch_size=batch_size,
            )
            for x in range(0, 20, 2)
    ]
    perturbations += [
            PerturbAtoms(
                perturb_atoms=insert_random_atoms,
                parameter_name='num_insertions',
                parameter_value=x,
                db_path=db_path,
                img_params=img_params,
                batch_size=batch_size,
            )
            for x in range(0, 20, 2)
    ]
    perturbations += [
            PerturbAtoms(
                perturb_atoms=change_random_elements,
                parameter_name='num_swaps',
                parameter_value=x,
                db_path=db_path,
                img_params=img_params,
                batch_size=batch_size,
            )
            for x in range(0, 20, 2)
    ]
    perturbations += [
            PerturbVoxels(
                perturb_voxels=add_noise_to_image,
                parameter_name='noise_std',
                parameter_value=x,
                db_path=db_path,
                img_params=img_params,
                batch_size=batch_size,
            )
            for x in np.logspace(-2, 1, 10)
    ]
    perturbations += [
            PerturbVoxels(
                perturb_voxels=blur_image,
                parameter_name='blur_std',
                parameter_value=x,
                db_path=db_path,
                img_params=img_params,
                batch_size=batch_size,
            )
            for x in np.logspace(-1, 1, 10)
    ]
    perturbations += [
            PerturbVoxels(
                perturb_voxels=mix_element_channels,
                parameter_name='mix_fraction',
                parameter_value=x,
                db_path=db_path,
                img_params=img_params,
                batch_size=batch_size,
            )
            for x in np.linspace(0, 1, 11)
    ]
    perturbations += [
            PerturbBatches(
                perturb_batch=blend_images,
                parameter_name='fraction_swapped',
                parameter_value=x,
                db_path=db_path,
                img_params=img_params,
                batch_size=batch_size,
            )
            for x in np.linspace(0, 1, 11)
    ]

    return perturbations, min(p.num_batches for p in perturbations)

def make_fast_debug_perturbations(
        db_path,
        img_params,
        *,
        batch_size=16,
):
    perturbations = [
            TruncatedPerturbation(
                PerturbVoxels(
                    perturb_voxels=blur_image,
                    parameter_name='blur_std',
                    parameter_value=x,
                    db_path=db_path,
                    img_params=img_params,
                    batch_size=batch_size,
                ),
                max_batches=100,
            )
            for x in [0.1, 10.0]
    ]
    return perturbations, min(p.num_batches for p in perturbations)

def make_dry_run_perturbations(
        db_path,
        img_params,
        *,
        batch_size=16,
):
    def _trunc(p):
        return TruncatedPerturbation(p, max_batches=100)

    perturbations = []

    perturbations += [
            _trunc(PerturbAtoms(
                perturb_atoms=jitter_atoms,
                parameter_name='noise_A',
                parameter_value=x,
                db_path=db_path,
                img_params=img_params,
                batch_size=batch_size,
            ))
            for x in np.linspace(0, 2, 3)
    ]
    perturbations += [
            _trunc(PerturbAtoms(
                perturb_atoms=delete_random_atoms,
                parameter_name='num_deletions',
                parameter_value=x,
                db_path=db_path,
                img_params=img_params,
                batch_size=batch_size,
            ))
            for x in np.linspace(0, 18, 3, dtype=int)
    ]
    perturbations += [
            _trunc(PerturbAtoms(
                perturb_atoms=insert_random_atoms,
                parameter_name='num_insertions',
                parameter_value=x,
                db_path=db_path,
                img_params=img_params,
                batch_size=batch_size,
            ))
            for x in np.linspace(0, 18, 3, dtype=int)
    ]
    perturbations += [
            _trunc(PerturbAtoms(
                perturb_atoms=change_random_elements,
                parameter_name='num_swaps',
                parameter_value=x,
                db_path=db_path,
                img_params=img_params,
                batch_size=batch_size,
            ))
            for x in np.linspace(0, 18, 3, dtype=int)
    ]
    perturbations += [
            _trunc(PerturbVoxels(
                perturb_voxels=add_noise_to_image,
                parameter_name='noise_std',
                parameter_value=x,
                db_path=db_path,
                img_params=img_params,
                batch_size=batch_size,
            ))
            for x in np.logspace(-2, 1, 3)
    ]
    perturbations += [
            _trunc(PerturbVoxels(
                perturb_voxels=blur_image,
                parameter_name='blur_std',
                parameter_value=x,
                db_path=db_path,
                img_params=img_params,
                batch_size=batch_size,
            ))
            for x in np.logspace(-1, 1, 3)
    ]
    perturbations += [
            _trunc(PerturbVoxels(
                perturb_voxels=mix_element_channels,
                parameter_name='mix_fraction',
                parameter_value=x,
                db_path=db_path,
                img_params=img_params,
                batch_size=batch_size,
            ))
            for x in np.linspace(0, 1, 3)
    ]
    perturbations += [
            _trunc(PerturbBatches(
                perturb_batch=blend_images,
                parameter_name='fraction_swapped',
                parameter_value=x,
                db_path=db_path,
                img_params=img_params,
                batch_size=batch_size,
            ))
            for x in np.linspace(0, 1, 3)
    ]

    return perturbations, min(p.num_batches for p in perturbations)

def make_intervals(rng, num_batches, num_lengths, num_replicates):
    lengths = np.unique(
            np.geomspace(1, num_batches, num_lengths).astype(int)
    )
    intervals_by_length = {}
    for length in lengths:
        num_possible = num_batches // length
        if num_possible < num_replicates:
            continue
        selected = rng.choice(num_possible, size=num_replicates, replace=False)
        intervals_by_length[length] = sorted(
                Interval(start=idx * length, end=(idx + 1) * length)
                for idx in selected
        )
    return intervals_by_length


def jitter_atoms(rng, atoms, *, noise_A):
    noise = rng.normal(
            scale=noise_A,
            size=(3, atoms.height),
    )
    return (
            atoms
            .with_columns(
                x=pl.col('x') + noise[0],
                y=pl.col('y') + noise[1],
                z=pl.col('z') + noise[2],
            )
    )

def delete_random_atoms(rng, atoms, *, img_params, num_deletions: int):
    if num_deletions == 0:
        return atoms

    img_atoms = mmvox.discard_atoms_outside_image(
            atoms, img_params.get_mmvox_image_params(),
    )

    mask = np.ones(img_atoms.height, dtype=bool)
    mask[0:num_deletions] = False
    rng.shuffle(mask)

    return img_atoms.filter(mask)

def insert_random_atoms(rng, atoms, *, img_params, num_insertions: int):
    if num_insertions == 0:
        return atoms

    mask = np.zeros(atoms.height, dtype=bool)
    mask[0:num_insertions] = True
    rng.shuffle(mask)

    atoms_copy = atoms.filter(mask)

    n = atoms_copy.height
    grid = img_params.grid
    xyz = rng.uniform(
            low=grid.center_A - grid.length_A / 2,
            high=grid.center_A + grid.length_A / 2,
            size=(n, 3),
    )

    atoms_insert = (
            atoms_copy
            .with_columns(
                x=xyz[:,0],
                y=xyz[:,1],
                z=xyz[:,2],
            )
    )

    return pl.concat([atoms, atoms_insert])

def change_random_elements(rng, atoms, *, img_params, num_swaps: int):
    if num_swaps == 0:
        return atoms

    img_atoms = mmvox.discard_atoms_outside_image(
            atoms, img_params.get_mmvox_image_params(),
    )

    mask = np.zeros(img_atoms.height, dtype=bool)
    mask[0:num_swaps] = True
    rng.shuffle(mask)

    element_counts = (
            atoms
            .group_by('element')
            .len()
            .sort('element')
    )
    elements = element_counts['element'].to_numpy()
    counts = element_counts['len'].to_numpy().astype(float)

    swap_atom_dicts = img_atoms.filter(mask).to_dicts()

    for atom in swap_atom_dicts:
        excl = elements != atom['element']
        weights = counts[excl] / counts[excl].sum()
        atom['element'] = rng.choice(elements[excl], p=weights)

    orig_atoms = img_atoms.filter(~mask)
    swap_atoms = pl.DataFrame(swap_atom_dicts, schema=img_atoms.schema)

    return pl.concat([orig_atoms, swap_atoms])

def add_noise_to_image(rng, img, *, noise_std, dataset_std=1):
    noise = rng.normal(
            scale=noise_std,
            size=img.shape,
    )
    var_dataset = dataset_std**2
    var_noise = noise_std**2
    scale_factor = sqrt(var_dataset / (var_dataset + var_noise))
    return scale_factor * (img + noise)

def blur_image(rng, img, *, blur_std):
    from scipy.ndimage import gaussian_filter

    img_blur = np.empty_like(img)

    for c in range(img.shape[0]):
        img_blur[c] = gaussian_filter(img[c], sigma=blur_std)

    return img_blur

def mix_element_channels(rng, img, *, mix_fraction):
    img_mean = np.mean(img, axis=0)
    return (1 - mix_fraction) * img + (mix_fraction) * img_mean

def blend_images(seeds, imgs, *, noise_scale=5, fraction_swapped=0.0):
    if fraction_swapped == 0:
        return imgs

    B, C, W, H, D = imgs.shape
    assert B == len(seeds)

    x = np.linspace(0, noise_scale, W)
    y = np.linspace(0, noise_scale, H)
    z = np.linspace(0, noise_scale, D)

    noise = np.zeros((B, W, H, D))

    for b in range(B):
        opensimplex.seed(seeds[b])
        noise[b] = opensimplex.noise3array(x, y, z)

    imgs_perturbed = np.empty_like(imgs)
    voxels_swapped = int(fraction_swapped * W * H * D)

    for b in range(B):
        threshold = np.sort(noise[b].flatten())[-voxels_swapped]

        mask = noise[b] > threshold

        fill_imgs = np.concatenate([imgs[:b], imgs[b+1:]])
        fill_noise = np.concatenate([noise[:b], noise[b+1:]])
        fill_idx = np.argmax(fill_noise, axis=0)
        fill_img = np.choose(fill_idx, fill_imgs)

        imgs_perturbed[b] = np.where(mask, fill_img, imgs[b])

    return imgs_perturbed


def _make_perturbed_atom_sample(sample, *, perturb_atoms, img_params):
    x = make_unsupervised_sample(sample)
    atoms_a = transform_atom_coords(x['atoms_i'], x['frame_ia'])
    perturbed_atoms_a = smartcall.call(
            perturb_atoms,
            KwOnly('sample', sample),
            KwOnly('rng', sample.rng),
            KwOnly('atoms', atoms_a),
            KwOnly('img_params', img_params),
    )
    img, _ = image_from_atoms(perturbed_atoms_a, img_params)
    return img

def _make_perturbed_voxel_sample(sample, *, perturb_voxels, img_params):
    x = make_unsupervised_image_sample(sample, img_params=img_params)
    return smartcall.call(
            perturb_voxels,
            KwOnly('sample', sample),
            KwOnly('rng', sample.rng),
            KwOnly('img', x['image']),
            KwOnly('img_params', img_params),
    )

def _make_unperturbed_sample(sample, *, img_params):
    x = make_unsupervised_image_sample(sample, img_params=img_params)
    return {
            'seed': sample.i,
            'image': x['image'],
    }
def _get_metric_name(metric):
    if hasattr(metric, 'name'):
        return metric.name
    return repr(metric)

def _make_norm(values):
    vals = np.sort(values)
    if vals[0] > 0 and len(vals) >= 3:
        log_diffs = np.diff(np.log10(vals))
        if (np.ptp(np.log10(vals)) > 1
                and np.std(log_diffs) / np.mean(log_diffs) < 0.01):
            return LogNorm(vmin=vals[0], vmax=vals[-1])
    return Normalize(vmin=vals[0], vmax=vals[-1])




