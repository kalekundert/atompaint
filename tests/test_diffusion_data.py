import atompaint.diffusion.data as _ap
import parametrize_from_file as pff
import polars as pl
import torch
import numpy as np

with_py = pff.Namespace()
POLYMER_LABELS = [
        'polypeptide(L)',
        'polydeoxyribonucleotide',
        'polyribonucleotide',
]
CATH_LABELS = [
        '1.10', '1.20',
        '2.30', '2.40', '2.60',
        '3.10', '3.20', '3.30', '3.40', '3.60', '3.90',
]

def labeled_atoms(*label_cols):

    def cast(x):
        from macromol_dataframe.testing import dataframe
        return (
                dataframe(x)
                .with_columns(
                    pl.col(label_cols).str.to_integer(strict=False),
                )
        )

    return cast

def label(x):
    return np.array(with_py.eval(x.split()))


def test_random_label_factory():
    rng = np.random.default_rng(0)

    labels = _ap.random_label_factory(
            rng=rng,
            batch_size=1024, 
            polymer_labels=POLYMER_LABELS,
            cath_labels=CATH_LABELS,
    )

    assert labels.shape == (1024, 14)

    # Over 1024 samples, we would expect to see each label 73 times on average.  
    # I arbitrarily reduced the threshold to 50, to account for random 
    # variance.
    assert torch.all(labels.sum(axis=0) > 50)

    for label in labels:
        # There should be at most one polymer and one domain.
        assert sum(label[:3]) == 1
        assert sum(label[3:]) <= 1

        # For DNA/RNA molecules, none of the domain labels should be set.
        if label[1:3].any():
            assert not label[0]
            assert not label[3:].any()

        # If one of the domain labels is set, this must be a protein molecule.
        if label[3:].any():
            assert label[0]

@pff.parametrize(
        schema=[
            pff.cast(
                atoms=labeled_atoms('polymer_label', 'cath_label'),
                n_labels=pff.cast(polymer=int, cath=int),
                expected=label,
            ),
        ],
)
def test_get_polymer_cath_label(atoms, n_labels, expected):
    label = _ap._get_polymer_cath_label(
            atoms,
            n_polymer_labels=n_labels['polymer'],
            n_cath_labels=n_labels['cath'],
    )
    np.testing.assert_allclose(label, expected)

@pff.parametrize(
        schema=[
            pff.cast(
                atoms=labeled_atoms('label'),
                n_labels=int,
                expected=label,
            ),
        ],
)
def test_get_label(atoms, n_labels, expected):
    np.testing.assert_allclose(
            _ap._get_label(atoms, 'label', n_labels),
            expected,
    )

def test_get_all_polymer_cath_labels():
    labels = _ap._get_all_polymer_cath_labels(POLYMER_LABELS, CATH_LABELS)
    expected = [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # protein, no domain
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # DNA
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # RNA
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # protein, 1st domain
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # protein, 2nd domain
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # ...
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]

    # The order of the rows is enforced by this test, but it doesn't really 
    # matter.
    np.testing.assert_allclose(labels, expected)
