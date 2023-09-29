import atompaint.transform_pred.datasets.recording as ap
import numpy as np
import sqlite3
import pytest

from more_itertools import one
from atompaint.datasets.voxelize import ImageParams, Grid

def test_record_training_example():
    # Use random integers just to make it relatively easy to tell each array 
    # from all the others.
    rng = np.random.default_rng(0)
    mock_frame_ia = rng.integers(10, size=(4, 4))
    mock_frames_ab = rng.integers(10, size=(6, 4, 4))
    mock_input_ab = rng.integers(10, size=(2, 4, 21, 21, 21))

    db = ap.init_recording(':memory:', mock_frames_ab)
    np.testing.assert_array_equal(ap.load_frames_ab(db), mock_frames_ab)

    ap.record_training_example(
            db, 99, '1abc', mock_frame_ia, np.int64(0), mock_input_ab)

    seed, tag, frame_ia, b, input_ab = one(ap.iter_training_examples(db))
    assert seed == 99
    assert tag == '1abc'
    np.testing.assert_array_equal(frame_ia, mock_frame_ia)
    assert b == 0
    np.testing.assert_array_equal(input_ab, mock_input_ab)

    tag, frame_ia, b, input_ab = ap.load_training_example(db, 99)
    assert tag == '1abc'
    np.testing.assert_array_equal(frame_ia, mock_frame_ia)
    assert b == 0
    np.testing.assert_array_equal(input_ab, mock_input_ab)

    assert ap.has_training_example(db, mock_input_ab)

    ap.drop_training_example(db, mock_input_ab)

    assert not ap.has_training_example(db, mock_input_ab)

    # Only 6 frames; illegal to reference a frame beyond that.  Note that we're 
    # also implicitly testing 0-indexing here, which is useful because SQLite 
    # is 1-indexed by default.
    with pytest.raises(sqlite3.IntegrityError):
        ap.record_training_example(
                db, 99, '1abc', mock_frame_ia, np.int64(6), mock_input_ab)

def test_record_img_params():
    db = ap.init_recording(':memory:', [])

    with pytest.raises(ValueError):
        ap.load_img_params(db)

    img_params_in = ImageParams(
            grid=Grid(
                length_voxels=21,
                resolution_A=0.75,
                center_A=np.zeros(3),
            ),
            channels=['C', 'N', 'O', '.*'],
            element_radii_A=0.375,
    )

    ap.record_img_params(db, img_params_in)
    img_params_out = ap.load_img_params(db)

    assert img_params_out.grid.length_voxels == 21
    assert img_params_out.grid.resolution_A == 0.75
    np.testing.assert_array_equal(img_params_out.grid.center_A, np.zeros(3))
    assert img_params_out.channels == ['C', 'N', 'O', '.*']
    assert img_params_out.element_radii_A == 0.375

