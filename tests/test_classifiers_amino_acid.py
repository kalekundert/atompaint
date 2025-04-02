import atompaint.classifiers.amino_acid as ap
import torch
import polars as pl
import numpy as np

from polars.testing import assert_frame_equal
from pytest import approx
from math import log

def test_blosum_metric():
    # To make the inputs and outputs for this test smaller and easier to 
    # understand, I'm going to configure the metric as if Ala, Arg, and Asn are 
    # the only amino acids.  Below are the relevant BLOSUM90 scores:
    #
    #      A  R  N
    #   A  5 -2 -2
    #   R -2  6 -1
    #   N -2 -1  7

    labels = pl.DataFrame({
        'name1': ['A', 'R', 'N'],
        'name3': ['ALA', 'ARG', 'ASN'],
        'label': [0, 1, 2],
    })

    metric = ap.BlosumMetric(n=90, labels=labels)

    # Best predictions

    y = torch.tensor([0, 1, 2])
    y_hat = torch.tensor([
        [100,   0,   0],
        [  0, 100,   0],
        [  0,   0, 100],
    ])

    metric.update(y_hat, y)
    assert metric.compute() == approx(6)
    metric.reset()

    # Worst predictions

    y = torch.tensor([0, 1, 2])
    y_hat = torch.tensor([
        [  0, 100,   0],
        [100,   0,   0],
        [100,   0,   0],
    ])

    metric.update(y_hat, y)
    assert metric.compute() == approx(-2)
    metric.reset()

    # Uniform predictions

    y = torch.tensor([0, 1, 2])
    y_hat = torch.ones((3,3))

    metric.update(y_hat, y)
    assert metric.compute() == approx(8/9)
    metric.reset()

    # 60% correct predictions

    # [z, 0, 0] corresponds to 60/20/20 probabilities, after softmax.
    z = log(3)

    y = torch.tensor([0, 1, 2])
    y_hat = torch.tensor([
        [z, 0, 0],
        [0, z, 0],
        [0, 0, z],
    ])

    metric.update(y_hat, y)
    assert metric.compute() == approx(88/30)
    metric.reset()

def test_balance_amino_acids():
    rng = np.random.default_rng(0)

    # Only the `residue_id` and `comp_id` columns are needed for the balancing 
    # calculation, but the `atom_id` column let's use check that no columns are 
    # lost.

    atoms = pl.DataFrame([
        dict(residue_id=1, comp_id='ALA', atom_id='CA'),
        dict(residue_id=2, comp_id='VAL', atom_id='CA'),
        dict(residue_id=3, comp_id='ALA', atom_id='CA'),
        dict(residue_id=4, comp_id='VAL', atom_id='CA'),
    ])

    amino_acids = pl.DataFrame([
        dict(name1='A', name3='ALA', pick_prob=1),
        dict(name1='V', name3='VAL', pick_prob=0),
    ])

    actual = ap.balance_amino_acids(rng, atoms, amino_acids)

    expected = pl.DataFrame([
        dict(residue_id=1, comp_id='ALA', atom_id='CA'),
        dict(residue_id=3, comp_id='ALA', atom_id='CA'),
    ])

    assert_frame_equal(actual, expected)


