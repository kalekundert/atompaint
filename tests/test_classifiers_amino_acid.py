import atompaint.classifiers.amino_acid as ap
import torch
import polars as pl
import numpy as np
import macromol_voxelize as mmvox

from polars.testing import assert_frame_equal
from pytest import approx
from utils import IMAGE_DIR, require_apw
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

def test_sample_uniform_crop():
    from scipy.stats import chisquare

    # The idea for this test is to (i) label each voxel with a unique id and 
    # (ii) check that those ids are sampled uniformly.

    I = 5
    C = 3

    rng = np.random.default_rng(0)

    img = np.full((1, I, I, I), 3**3)
    img[0, 0:C, 0:C, 0:C] = np.arange(3**3).reshape((C, C, C))

    counts = np.zeros(3**3)

    for i in range(1000):
        crop = ap.sample_uniform_crop(
                rng=rng,
                grid=mmvox.Grid(length_voxels=I, resolution_A=1),
                crop_length_voxels=C,
        )
        img_i = img[crop]

        assert img_i.shape == (1, C, C, C)

        counts[img_i[0, 0, 0, 0]] += 1

    test = chisquare(counts)
    assert test.pvalue > 0.05

def test_sample_targeted_crop():
    # The idea behind this test is to sample a bunch of crops, then to check 
    # for the following properties:
    #
    # - All the crops are the right size.
    # - All of the crops completely contain the target sphere.
    # - The target sphere touches every edge of both the image and the crops, 
    #   over all of the samples.

    grid = mmvox.Grid(
            length_voxels=(I := 11),
            resolution_A=1,
    )
    img_params = mmvox.ImageParams(
            grid=grid,
            channels=1,
    )
    crop_length_voxels = C = 7
    target_radius_A = 2

    all_images = np.zeros((1, I, I, I))
    all_crops = np.zeros((1, C, C, C))

    for i in range(1000):
        rng = np.random.default_rng(i)

        L = grid.length_A / 2 - target_radius_A
        target_center_A = rng.uniform(-L, L, size=3)

        atom = pl.DataFrame([
            dict(
                x=target_center_A[0],
                y=target_center_A[1],
                z=target_center_A[2],
                radius_A=target_radius_A,
                channels=[0],
            ),
        ])

        img = mmvox.image_from_all_atoms(atom, img_params)

        assert img.shape == (1, I, I, I)
        assert img.sum() == approx(1)

        crop = ap.sample_targeted_crop(
                rng=rng,
                grid=grid,
                crop_length_voxels=crop_length_voxels,
                target_center_A=target_center_A,
                target_radius_A=target_radius_A,
        )
        img_i = img[crop]

        assert img_i.shape == (1, C, C, C)
        assert img_i.sum() == approx(1)

        all_images += img
        all_crops += img_i

    for i in range(I):
        assert all_images[0, i, :, :].sum() > 0
        assert all_images[0, :, i, :].sum() > 0
        assert all_images[0, :, :, i].sum() > 0

    for c in range(C):
        assert all_crops[0, c, :, :].sum() > 0
        assert all_crops[0, :, c, :].sum() > 0
        assert all_crops[0, :, :, c].sum() > 0

    #np.save('all_images.npy', all_images)
    #np.save('all_crops.npy', all_crops)

@require_apw
def test_load_expt_131_classifier_1qjg_n38():
    classifier = ap.load_expt_131_classifier()

    img = np.load(IMAGE_DIR / '1qjg_n38.npz')['image']
    img = torch.from_numpy(img).unsqueeze(0).float()

    label_preds = classifier(img)[0]
    label_pred = torch.argmax(label_preds).item()

    assert classifier.amino_acids[label_pred, 'name1'] == 'N'

