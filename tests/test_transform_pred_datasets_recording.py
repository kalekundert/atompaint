import atompaint.transform_pred.datasets.recording as ap
import numpy as np
import pandas as pd
import sqlite3
import pytest

from more_itertools import one
from atompaint.datasets.voxelize import ImageParams, Grid

def test_training_example():
    n = 3

    # Use random integers just to make it relatively easy to tell each array 
    # from all the others when debugging.
    rng = np.random.default_rng(0)
    mock_frames_ab = rng.integers(10, size=(6, 4, 4))
    mock_frame_ia = rng.integers(10, size=(n, 4, 4))
    mock_input_ab = rng.integers(10, size=(n, 2, 4, 21, 21, 21))

    db = ap.init_recording(':memory:', mock_frames_ab)
    np.testing.assert_array_equal(ap.load_frames_ab(db), mock_frames_ab)

    ap.record_training_example(
            db, 1, '1abc', mock_frame_ia[0], np.int64(0), mock_input_ab[0])

    example_ids = ap.load_all_training_example_ids(db)
    assert len(example_ids) == 1

    ap.record_training_example(
            db, 2, '2def', mock_frame_ia[1], np.int64(5), mock_input_ab[1])

    example_ids = ap.load_all_training_example_ids(db)
    assert len(example_ids) == 2

    training_example = ap.load_training_example(db, example_ids[0])

    assert training_example['seed'] == 1
    assert training_example['tag'] == '1abc'
    assert training_example['b'] == 0
    np.testing.assert_array_equal(
            training_example['frame_ia'],
            mock_frame_ia[0],
    )
    np.testing.assert_array_equal(
            training_example['frame_ab'],
            mock_frames_ab[0],
    )
    np.testing.assert_array_equal(
            training_example['input'],
            mock_input_ab[0],
    )

    training_example = ap.load_training_example(db, example_ids[1])

    assert training_example['seed'] == 2
    assert training_example['tag'] == '2def'
    assert training_example['b'] == 5
    np.testing.assert_array_equal(
            training_example['frame_ia'],
            mock_frame_ia[1],
    )
    np.testing.assert_array_equal(
            training_example['frame_ab'],
            mock_frames_ab[5],
    )
    np.testing.assert_array_equal(
            training_example['input'],
            mock_input_ab[1],
    )

    with pytest.raises(ap.NotFound):
        ap.load_training_example(db, max(example_ids) + 1)

    # Only 6 frames; illegal to reference a frame beyond that.  Note that we're 
    # also implicitly testing 0-indexing here, which is useful because SQLite 
    # is 1-indexed by default.
    with pytest.raises(sqlite3.IntegrityError):
        ap.record_training_example(
                db, 3, '3ghi', mock_frame_ia[2], np.int64(6), mock_input_ab[2])

def test_img_params():
    db = ap.init_recording(':memory:', [])

    with pytest.raises(ap.NotFound):
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

    # Try to record the same image parameters twice: no-op
    ap.record_img_params(db, img_params_in)
    n = db.execute(
            'SELECT COUNT(*) FROM meta WHERE key="img_params"'
    ).fetchone()[0]
    assert n == 1

    # Try to record different image parameters: error
    img_params_err = ImageParams(
            grid=Grid(
                length_voxels=img_params_in.grid.length_voxels,
                resolution_A=img_params_in.grid.resolution_A + 0.01,
                center_A=img_params_in.grid.center_A,
            ),
            channels=img_params_in.channels,
            element_radii_A=img_params_in.element_radii_A,
    )
    with pytest.raises(AssertionError):
        ap.record_img_params(db, img_params_err)

def test_manual_predictions():
    db = ap.init_recording(':memory:', np.zeros((6, 4, 4)))
    ap.init_manual_predictions(db)

    ap.record_training_example(db, 1, '1abc', None, np.int64(0), None)
    ap.record_training_example(db, 2, '2def', None, np.int64(5), None)

    i, j = ap.load_all_training_example_ids(db)

    def expected(*rows):
        return pd.DataFrame(
                rows,
                columns=['example_id', 'num_successes', 'num_failures'],
        )

    pd.testing.assert_frame_equal(
            ap.load_manual_predictions(db),
            expected(),
    )

    ap.record_manual_prediction(db, i, 0)

    pd.testing.assert_frame_equal(
            ap.load_manual_predictions(db),
            expected((i, 1, 0))
    )

    ap.record_manual_prediction(db, i, 5)

    pd.testing.assert_frame_equal(
            ap.load_manual_predictions(db),
            expected((i, 1, 1))
    )

    ap.record_manual_prediction(db, i, None)

    pd.testing.assert_frame_equal(
            ap.load_manual_predictions(db),
            expected((i, 1, 2))
    )

    ap.record_manual_prediction(db, j, 5)

    pd.testing.assert_frame_equal(
            ap.load_manual_predictions(db),
            expected((i, 1, 2), (j, 1, 0))
    )

    ap.record_manual_prediction(db, j, 5)

    pd.testing.assert_frame_equal(
            ap.load_manual_predictions(db),
            expected((i, 1, 2), (j, 2, 0)),
    )

