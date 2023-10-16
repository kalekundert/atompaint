import sqlite3
import numpy as np
import json
import io

from atompaint.datasets.voxelize import ImageParams, Grid
from dataclasses import asdict
from more_itertools import one

class ManuallyVerifiedDataset:
    pass

def init_recording(path, frames_ab):
    db = load_recording(path)
    cur = db.cursor()

    cur.execute('''\
            CREATE TABLE IF NOT EXISTS meta (
                key UNIQUE,
                value
            )
    ''')
    cur.execute('''\
            CREATE TABLE IF NOT EXISTS frames_ab (
                b INTEGER PRIMARY KEY,
                frame_ab ARRAY
            )
    ''')
    cur.execute('''\
            CREATE TABLE IF NOT EXISTS training_examples (
                id INTEGER PRIMARY KEY,
                seed INTEGER,
                tag TEXT,
                frame_ia ARRAY,
                b INTEGER,
                input ARRAY,
                FOREIGN KEY(b) REFERENCES frames_ab(b)
            )
    ''')

    any_ids = cur.execute('SELECT b FROM frames_ab').fetchone()
    if not any_ids:
        cur.executemany(
                'INSERT INTO frames_ab VALUES (?, ?)',
                enumerate(frames_ab),
        )
    else:
        existing_frames_ab = load_frames_ab(db)
        np.testing.assert_allclose(frames_ab, existing_frames_ab, atol=1e-7)

    db.commit()
    return db

def init_manual_predictions(db):
    db.execute('''\
            CREATE TABLE IF NOT EXISTS manual_predictions (
                example_id INTEGER,
                prediction INTEGER,
                FOREIGN KEY(example_id) REFERENCES training_examples(id)
            )
    ''')
    db.commit()

def record_img_params(db, img_params):
    try:
        existing = load_img_params(db)

    except NotFound:
        db.execute('INSERT INTO meta VALUES ("img_params", ?)', (img_params,))
        db.commit()

    else:
        assert existing.grid.length_voxels == img_params.grid.length_voxels
        assert existing.grid.resolution_A == img_params.grid.resolution_A
        assert (existing.grid.center_A == img_params.grid.center_A).all()
        assert existing.channels == img_params.channels
        assert existing.element_radii_A == img_params.element_radii_A

def record_training_example(db, seed, tag, frame_ia, b, input_ab):
    db.execute('''\
            INSERT INTO training_examples (seed, tag, frame_ia, b, input)
            VALUES (?, ?, ?, ?, ?)''',
            # Have to convert *b* from "numpy int" to "python int", otherwise 
            # the foreign key constraint won't work.
            (seed, tag, frame_ia, int(b), input_ab),
    )
    db.commit()

def record_manual_prediction(db, example_id, prediction):
    db.execute(
            'INSERT INTO manual_predictions VALUES (?, ?)',
            (example_id, prediction),
    )
    db.commit()


def load_recording(path):
    sqlite3.register_adapter(np.ndarray, _adapt_np_array)
    sqlite3.register_converter('ARRAY', _convert_np_array)

    sqlite3.register_adapter(ImageParams, _adapt_img_params)
    sqlite3.register_converter('IMG_PARAMS', _convert_img_params)

    db = sqlite3.connect(
            path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    )
    db.execute('PRAGMA foreign_keys = ON')
    return db

def load_frames_ab(db):
    cur = db.execute('SELECT frame_ab FROM frames_ab ORDER BY b')
    cur.row_factory = _scalar_row_factory
    return np.stack(cur.fetchall())

def load_img_params(db):
    cur = db.execute('''\
            SELECT value
                AS "value [IMG_PARAMS]"
                FROM meta
                WHERE key="img_params"
    ''')
    cur.row_factory = _scalar_row_factory

    if img_params := cur.fetchone():
        return img_params
    else:
        raise NotFound("can't find image parameters in recording")

def load_training_example(db, id):
    cur = db.execute('''\
            SELECT id, seed, tag, frame_ia, fr.frame_ab, ex.b, input
            FROM training_examples AS ex
            INNER JOIN frames_ab AS fr ON ex.b == fr.b
            WHERE id=?''',
            (id,),
    )
    cur.row_factory = sqlite3.Row

    if row := cur.fetchone():
        return row
    else:
        raise NotFound(f"can't find training example id={id}")

def load_all_training_example_ids(db):
    cur = db.execute('SELECT id FROM training_examples')
    cur.row_factory = _scalar_row_factory
    return cur.fetchall()

def load_validated_training_example_ids(db, min_predictions):
    cur = db.execute('''\
            SELECT example_id FROM (
                SELECT
                    man.example_id,
                    SUM(man.prediction IS ex.b) AS num_successes,
                    SUM(man.prediction IS NOT ex.b) AS num_failures
                FROM manual_predictions AS man
                INNER JOIN training_examples AS ex ON ex.id == man.example_id
                GROUP BY man.example_id
            )
            WHERE num_successes >= ? AND num_failures == 0
            ''',
            (min_predictions,),
    )
    cur.row_factory = _scalar_row_factory
    return cur.fetchall()

class NotFound(Exception):
    pass

def _adapt_np_array(array):
    out = io.BytesIO()
    np.save(out, array, allow_pickle=False)
    return out.getvalue()

def _convert_np_array(bytes):
    in_ = io.BytesIO(bytes)
    x = np.load(in_)
    return x

def _adapt_img_params(img_params):
    # I considered pickling the `img_params` object, instead of using JSON.  
    # This wouldn't have required manually converting the grid center 
    # coordinate from a numpy array to a list, but I worried that it would've 
    # been more fragile w.r.t. future changes to the `ImageParams` data 
    # structure.

    d = asdict(img_params)
    d['grid']['center_A'] = d['grid']['center_A'].tolist()

    return json.dumps(d)

def _convert_img_params(bytes):
    d = json.loads(bytes)
    d['grid']['center_A'] = np.asarray(d['grid']['center_A'])
    d['grid'] = Grid(**d['grid'])
    return ImageParams(**d)

def _scalar_row_factory(cur, row):
    return one(row)
