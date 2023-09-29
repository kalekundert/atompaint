import sqlite3
import numpy as np
import json
import io

from atompaint.datasets.voxelize import ImageParams, Grid
from dataclasses import asdict

class ManuallyVerifiedDataset:
    pass

def init_recording(path, frames_ab):
    db = load_recording(path)
    cur = db.cursor()

    cur.execute('''\
            CREATE TABLE IF NOT EXISTS meta (key, value)
    ''')
    cur.execute('''\
            CREATE TABLE IF NOT EXISTS frames_ab (
                b INTEGER PRIMARY KEY,
                frame_ab ARRAY
            )
    ''')
    cur.execute('''\
            CREATE TABLE IF NOT EXISTS training_examples (
                seed INTEGER,
                tag TEXT,
                frame_ia ARRAY,
                frame_ab INTEGER,
                input ARRAY,
                FOREIGN KEY(frame_ab) REFERENCES frames_ab(b)
            )
    ''')

    ids = cur.execute('SELECT b FROM frames_ab').fetchall()
    if not ids:
        cur.executemany(
                'INSERT INTO frames_ab VALUES (?, ?)',
                enumerate(frames_ab),
        )
    else:
        existing_frames_ab = load_frames_ab(db)
        np.testing.assert_allclose(frames_ab, existing_frames_ab, atol=1e-7)

    db.commit()
    return db

def record_img_params(db, img_params):
    db.execute('INSERT INTO meta VALUES ("img_params", ?)', (img_params,))
    db.commit()

def record_training_example(db, seed, tag, frame_ia, b, input_ab):
    db.execute(
            'INSERT INTO training_examples VALUES (?, ?, ?, ?, ?)',
            (seed, tag, frame_ia, int(b), input_ab),
    )
    db.commit()

def drop_training_example(db, input_ab):
    db.execute('DELETE FROM training_examples WHERE input=?', (input_ab,))
    db.commit()

def has_training_example(db, input_ab):
    cur = db.execute(
            'SELECT rowid FROM training_examples WHERE input=?',
            (input_ab,),
    )
    return cur.fetchone() is not None


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
    rows = db.execute('SELECT b, frame_ab FROM frames_ab').fetchall()
    frames_ab = [frame_ab for _, frame_ab in sorted(rows)]
    return np.stack(frames_ab)

def load_img_params(db):
    cur = db.execute('''\
            SELECT value
                AS "value [IMG_PARAMS]"
                FROM meta
                WHERE key="img_params"
    ''')

    if row := cur.fetchone():
        return row[0]
    else:
        raise ValueError("can't find image parameters in recording")

def load_training_example(db, seed):
    cur = db.execute('''\
            SELECT tag, frame_ia, frame_ab, input
                FROM training_examples
                WHERE seed=?
    ''', (seed,))

    if row := cur.fetchone():
        return row
    else:
        raise ValueError(f"can't find training example with seed={seed}")

def iter_training_examples(db):
    cur = db.execute('SELECT * FROM training_examples')
    while row := cur.fetchone():
        yield row


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



