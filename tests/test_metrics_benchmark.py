import numpy as np
import polars as pl
import macromol_voxelize as mmvox

from matplotlib.colors import Normalize, LogNorm
from macromol_gym_unsupervised import ImageParams
from more_itertools import flatten, all_unique
from functools import partial


from atompaint.metrics.benchmark import (
    Perturbation, TruncatedPerturbation, Interval,
    make_intervals, record_metrics, _get_metric_name, _make_norm,
    jitter_atoms, delete_random_atoms, insert_random_atoms,
    change_random_elements, add_noise_to_image, blur_image,
    mix_element_channels, blend_images,
)

IMG_PARAMS = ImageParams(
        grid=mmvox.Grid(length_voxels=19, resolution_A=1),
        element_channels=[['C'], ['N'], ['O'], ['*']],
        normalize_std=0.05,
)


def _make_atoms(n=20, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-7.0, 7.0, size=(n, 3))
    elements = rng.choice(['C', 'N', 'O'], size=n).tolist()
    return pl.DataFrame({
        'x': coords[:, 0].tolist(),
        'y': coords[:, 1].tolist(),
        'z': coords[:, 2].tolist(),
        'element': elements,
    })

def _make_image(rng, channels=4, size=5):
    return rng.normal(size=(channels, size, size, size)).astype(np.float32)

def _make_images(rng, batch=4, channels=4, size=5):
    return rng.normal(size=(batch, channels, size, size, size)).astype(np.float32)

class FakePerturbation(Perturbation):
    def __init__(
            self,
            n_batches,
            *,
            algorithm_name='test_alg',
            parameter_name='test_param',
            parameter_value=1.0,
            batch_size=4,
    ):
        super().__init__(algorithm_name, parameter_name, parameter_value)
        self.batch_size = batch_size
        self.img_params = None
        self._n_batches = n_batches

    def iter_image_batches(self):
        return list(range(self._n_batches))

class FakeMetric:

    def __init__(self, name='fake'):
        self.name = name
        self.seen = []

    def update(self, x):
        self.seen.append(x)

    def compute(self):
        return FakeTensor(float(len(self.seen)))

class FakeTensor:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value



def test_make_intervals_interval_lengths():
    rng = np.random.default_rng(0)
    intervals = make_intervals(rng, num_batches=100, num_lengths=10, num_replicates=3)
    for length, ivs in intervals.items():
        for iv in ivs:
            assert iv.end - iv.start == length

def test_make_intervals_within_bounds():
    rng = np.random.default_rng(0)
    num_batches = 100
    intervals = make_intervals(rng, num_batches, num_lengths=10, num_replicates=3)
    for length, ivs in intervals.items():
        for iv in ivs:
            assert iv.start >= 0
            assert iv.end <= num_batches

def test_make_intervals_sorted():
    rng = np.random.default_rng(0)
    intervals = make_intervals(rng, num_batches=100, num_lengths=10, num_replicates=3)
    for length, ivs in intervals.items():
        assert ivs == sorted(ivs)

def test_make_intervals_replicate_count():
    rng = np.random.default_rng(0)
    intervals = make_intervals(rng, num_batches=200, num_lengths=10, num_replicates=3)
    for length, ivs in intervals.items():
        assert len(ivs) == 3

def test_make_intervals_non_overlapping():
    rng = np.random.default_rng(0)
    intervals = make_intervals(rng, num_batches=100, num_lengths=10, num_replicates=3)
    for length, ivs in intervals.items():
        indices = flatten(range(iv.start, iv.end) for iv in ivs)
        assert all_unique(indices)

def test_make_intervals_length_filter():
    # Lengths where num_batches // length < num_replicates must be excluded.
    rng = np.random.default_rng(0)
    num_batches = 100
    num_replicates = 5
    intervals = make_intervals(rng, num_batches, num_lengths=20, num_replicates=num_replicates)
    for length in intervals:
        assert num_batches // length >= num_replicates

    # Sanity-check: large lengths (e.g. 100) must be excluded (100//100=1 < 5).
    candidate_lengths = np.unique(np.geomspace(1, num_batches, 20).astype(int))
    excluded = [l for l in candidate_lengths if l not in intervals]
    assert len(excluded) > 0

def test_make_intervals_empty():
    # No length L satisfies 3 // L >= 5, so result must be empty.
    rng = np.random.default_rng(0)
    intervals = make_intervals(rng, num_batches=3, num_lengths=5, num_replicates=5)
    assert intervals == {}


def test_truncated_perturbation_delegation():
    inner = FakePerturbation(
            20,
            algorithm_name='blur_image',
            parameter_name='blur_std',
            parameter_value=0.5,
            batch_size=8,
    )
    tp = TruncatedPerturbation(inner, max_batches=10)
    assert tp.algorithm_name == 'blur_image'
    assert tp.parameter_name == 'blur_std'
    assert tp.parameter_value == 0.5
    assert tp.batch_size == 8
    assert tp.img_params is inner.img_params

def test_truncated_perturbation_truncates():
    tp = TruncatedPerturbation(FakePerturbation(20), max_batches=7)
    batches = tp.iter_image_batches()
    assert len(batches) == 7
    assert list(batches) == list(range(7))

def test_truncated_perturbation_no_truncation():
    tp = TruncatedPerturbation(FakePerturbation(5), max_batches=10)
    batches = tp.iter_image_batches()
    assert len(batches) == 5
    assert list(batches) == list(range(5))

def test_truncated_perturbation_num_batches():
    assert TruncatedPerturbation(FakePerturbation(20), max_batches=7).num_batches == 7
    assert TruncatedPerturbation(FakePerturbation(5), max_batches=10).num_batches == 5


def test_get_metric_name_with_name_attr():
    class M:
        name = 'my_metric'
    assert _get_metric_name(M()) == 'my_metric'

def test_get_metric_name_repr_fallback():
    class M:
        pass
    m = M()
    assert _get_metric_name(m) == repr(m)


def test_make_norm_linear():
    vals = np.linspace(1, 10, 10)
    assert isinstance(_make_norm(vals), Normalize)
    assert not isinstance(_make_norm(vals), LogNorm)

def test_make_norm_log_spaced():
    vals = np.logspace(0, 2, 10)  # 1 to 100, evenly log-spaced
    assert isinstance(_make_norm(vals), LogNorm)

def test_make_norm_too_few_values():
    # len < 3 → always linear
    vals = np.logspace(0, 2, 2)
    assert not isinstance(_make_norm(vals), LogNorm)

def test_make_norm_small_range():
    # Range ≤ 1 decade → linear even if positive and log-spaced
    vals = np.logspace(0, 0.5, 5)
    assert not isinstance(_make_norm(vals), LogNorm)


def test_record_metrics_schema():
    perturbations = [FakePerturbation(10)]
    intervals = {5: [Interval(0, 5)]}
    metrics = [FakeMetric]

    df = record_metrics(perturbations, intervals, metrics)

    assert set(df.columns) == {
        'algorithm_name', 'parameter_name', 'parameter_value',
        'metric_name', 'metric_value', 'n', 'replicate',
    }

def test_record_metrics_row_count():
    perturbations = [
            FakePerturbation(100, algorithm_name='perturb_1'),
            FakePerturbation(100, algorithm_name='perturb_2'),
    ]
    intervals = {
            5: [Interval(0, 5), Interval(5, 10)],
            10: [Interval(0, 10), Interval(10, 20)],
    }
    metrics = [
            partial(FakeMetric, 'metric_1'),
            partial(FakeMetric, 'metric_2'),
    ]

    df = record_metrics(perturbations, intervals, metrics)

    n_perturbations = 2
    n_intervals = 4
    n_metrics = 2

    assert len(df) == n_perturbations * n_intervals * n_metrics == 16

def test_record_metrics_n_value():
    perturbations = [FakePerturbation(20, batch_size=8)]
    intervals = {5: [Interval(0, 5), Interval(5, 10)]}
    metrics = [FakeMetric]

    df = record_metrics(perturbations, intervals, metrics)

    assert (df['n'] == 5 * 8).all()

def test_record_metrics_replicate_values():
    perturbations = [FakePerturbation(50)]
    intervals = {5: [Interval(0, 5), Interval(10, 15), Interval(20, 25)]}
    metrics = [FakeMetric]

    df = record_metrics(perturbations, intervals, metrics)

    assert sorted(df['replicate'].to_list()) == [0, 1, 2]

def test_record_metrics_batch_assignment():
    # Metric for Interval(5, 10) must receive exactly batches 5, 6, 7, 8, 9.
    created = []
    def factory():
        m = FakeMetric()
        created.append(m)
        return m

    perturbations = [FakePerturbation(20)]
    intervals = {5: [Interval(5, 10)]}
    metrics = [factory]

    record_metrics(perturbations, intervals, metrics)

    assert len(created) == 1
    assert created[0].seen == [5, 6, 7, 8, 9]

def test_record_metrics_metric_isolation():
    # Two intervals in the same length-bucket must get independent metric instances.
    created = []
    def factory():
        m = FakeMetric()
        created.append(m)
        return m

    perturbations = [FakePerturbation(50)]
    intervals = {5: [Interval(0, 5), Interval(10, 15)]}
    metrics = [factory]

    record_metrics(perturbations, intervals, metrics)

    assert len(created) == 2
    assert created[0].seen == [0, 1, 2, 3, 4]
    assert created[1].seen == [10, 11, 12, 13, 14]

def test_record_metrics_multiple_factories():
    perturbations = [FakePerturbation(10)]
    intervals = {5: [Interval(0, 5)]}
    metrics = [
            lambda: FakeMetric('metric_1'),
            lambda: FakeMetric('metric_2'),
    ]

    df = record_metrics(perturbations, intervals, metrics)

    assert len(df) == 2
    assert set(df['metric_name'].to_list()) == {'metric_1', 'metric_2'}

def test_record_metrics_multiple_perturbations():
    perturbations = [
            FakePerturbation(10, algorithm_name='perturb_1', parameter_value=1.0),
            FakePerturbation(10, algorithm_name='perturb_2', parameter_value=2.0),
    ]
    intervals = {5: [Interval(0, 5)]}
    metrics = [FakeMetric]

    df = record_metrics(perturbations, intervals, metrics)

    assert len(df) == 2
    assert set(df['algorithm_name'].to_list()) == {'perturb_1', 'perturb_2'}


def test_add_noise_to_image_zero_noise():
    rng = np.random.default_rng(0)
    img = _make_image(rng)
    result = add_noise_to_image(rng, img, noise_std=0)
    np.testing.assert_array_equal(result, img)

def test_add_noise_to_image_shape_preserved():
    rng = np.random.default_rng(0)
    img = _make_image(rng)
    result = add_noise_to_image(rng, img, noise_std=0.5)
    assert result.shape == img.shape

def test_blur_image_shape_preserved():
    rng = np.random.default_rng(0)
    img = _make_image(rng)
    result = blur_image(rng, img, blur_std=1.0)
    assert result.shape == img.shape

def test_blur_image_reduces_std():
    rng = np.random.default_rng(0)
    img = _make_image(rng)
    result = blur_image(rng, img, blur_std=2.0)
    assert np.std(result) < np.std(img)

def test_blur_image_soften():
    rng = np.random.default_rng(0)
    img = _make_image(rng)
    result = blur_image(rng, img, blur_std=1.0)
    assert img.max() >= result.max()
    assert img.min() <= result.min()

def test_mix_element_channels_zero_fraction():
    rng = np.random.default_rng(0)
    img = _make_image(rng)
    result = mix_element_channels(rng, img, mix_fraction=0)
    np.testing.assert_array_equal(result, img)

def test_mix_element_channels_full_fraction():
    rng = np.random.default_rng(0)
    img = _make_image(rng)
    result = mix_element_channels(rng, img, mix_fraction=1.0)
    img_mean = np.mean(img, axis=0)
    for c in range(img.shape[0]):
        np.testing.assert_allclose(result[c], img_mean)

def test_blend_images_zero_fraction():
    rng = np.random.default_rng(0)
    imgs = _make_images(rng)
    seeds = [0, 1, 2, 3]
    result = blend_images(seeds, imgs, fraction_swapped=0.0)
    np.testing.assert_allclose(imgs, result)

def test_blend_images_shape_preserved():
    rng = np.random.default_rng(0)
    imgs = _make_images(rng)
    seeds = [10, 20, 30, 40]
    result = blend_images(seeds, imgs, fraction_swapped=0.5)
    assert result.shape == imgs.shape

def test_blend_images_voxels_from_batch():
    # Every output voxel must be a copy of the same spatial voxel from one of
    # the input images — the mask is spatial, so channels are never mixed.
    rng = np.random.default_rng(0)
    B, C, W = 4, 3, 5
    imgs = rng.normal(size=(B, C, W, W, W)).astype(np.float32)
    seeds = list(range(B))
    result = blend_images(seeds, imgs, fraction_swapped=0.5)

    for b in range(B):
        for w, h, d in np.ndindex(W, W, W):
            voxel = result[b, :, w, h, d]
            assert any(
                np.array_equal(voxel, imgs[b_src, :, w, h, d])
                for b_src in range(B)
            )


def test_jitter_atoms_schema_preserved():
    rng = np.random.default_rng(0)
    atoms = _make_atoms()
    result = jitter_atoms(rng, atoms, noise_A=1.0)
    assert result.schema == atoms.schema
    assert result.height == atoms.height

def test_jitter_atoms_zero_noise():
    rng = np.random.default_rng(0)
    atoms = _make_atoms()
    result = jitter_atoms(rng, atoms, noise_A=0.0)
    assert (result['x'] == atoms['x']).all()
    assert (result['y'] == atoms['y']).all()
    assert (result['z'] == atoms['z']).all()

def test_jitter_atoms_shifts_coords():
    rng = np.random.default_rng(0)
    atoms = _make_atoms()
    result = jitter_atoms(rng, atoms, noise_A=1.0)
    assert not (result['x'] == atoms['x']).all()

def test_delete_random_atoms_zero_deletions():
    rng = np.random.default_rng(0)
    atoms = _make_atoms()
    result = delete_random_atoms(rng, atoms, img_params=IMG_PARAMS, num_deletions=0)
    assert result is atoms

def test_delete_random_atoms_removes_atoms():
    rng = np.random.default_rng(0)
    atoms = _make_atoms(n=20)
    result = delete_random_atoms(rng, atoms, img_params=IMG_PARAMS, num_deletions=5)
    assert result.height == 15

def test_insert_random_atoms_zero_insertions():
    rng = np.random.default_rng(0)
    atoms = _make_atoms()
    result = insert_random_atoms(rng, atoms, img_params=IMG_PARAMS, num_insertions=0)
    assert result is atoms

def test_insert_random_atoms_adds_atoms():
    rng = np.random.default_rng(0)
    atoms = _make_atoms(n=20)
    result = insert_random_atoms(rng, atoms, img_params=IMG_PARAMS, num_insertions=5)
    assert result.height == 25

def test_change_random_elements_zero_swaps():
    rng = np.random.default_rng(0)
    atoms = _make_atoms()
    result = change_random_elements(rng, atoms, img_params=IMG_PARAMS, num_swaps=0)
    assert result is atoms

def test_change_random_elements_swaps_elements():
    rng = np.random.default_rng(0)
    atoms = pl.DataFrame({
        'x': [0.0] * 12,
        'y': [0.0] * 12,
        'z': [0.0] * 12,
        'element': ['C'] * 8 + ['N'] * 4,
    })
    result = change_random_elements(rng, atoms, img_params=IMG_PARAMS, num_swaps=3)
    assert result.height == atoms.height
    assert result.schema == atoms.schema
    # With 3 swaps between 2 element types, net count must change (3 is odd).
    orig_counts = dict(atoms.group_by('element').len().iter_rows())
    result_counts = dict(result.group_by('element').len().iter_rows())
    assert orig_counts != result_counts
