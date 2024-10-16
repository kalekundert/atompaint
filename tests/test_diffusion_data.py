import torch
import numpy as np
import atompaint.diffusion.data as _ap

def test_batch_generator_mock():

    class MockGenerator:

        def __init__(self, x):
            self.x = x

        def get(self):
            return self.x

    rngs = _ap.BatchGenerator([
        MockGenerator(0),
        MockGenerator(1),
        MockGenerator(2),
    ])

    torch.testing.assert_close(
            rngs.get(),
            torch.tensor([0, 1, 2]),
    )

def test_batch_generator_uniform():
    rngs = _ap.BatchGenerator([
        np.random.default_rng(0),
        np.random.default_rng(1),
        np.random.default_rng(2),
    ])

    x = rngs.uniform()

    assert x.shape == (3,)
    assert x.dtype is torch.float64
    assert len(set(x)) > 1
    assert torch.all(x >= 0)
    assert torch.all(x < 1)

def test_batch_generator_uniform_3d():
    rngs = _ap.BatchGenerator([
        np.random.default_rng(0),
        np.random.default_rng(1),
        np.random.default_rng(2),
    ])

    x = rngs.uniform(size=(2,4))

    assert x.shape == (3,2,4)
    assert x.dtype is torch.float64
    assert torch.all(x >= 0)
    assert torch.all(x < 1)

def test_batch_generator_integers():
    rngs = _ap.BatchGenerator([
        np.random.default_rng(0),
        np.random.default_rng(1),
        np.random.default_rng(2),
    ])

    x = rngs.integers(0, 10)

    assert x.shape == (3,)
    assert x.dtype is torch.int64
    assert len(set(x)) > 1
    assert torch.all(x >= 0)
    assert torch.all(x < 10)

def test_batch_generator_pickle():
    rngs = _ap.BatchGenerator([
        np.random.default_rng(0),
        np.random.default_rng(1),
        np.random.default_rng(2),
    ])

    import pickle
    rngs_packed = pickle.dumps(rngs)
    rngs_unpacked = pickle.loads(rngs_packed)

    x = rngs_unpacked.uniform()

    assert x.shape == (3,)
    assert x.dtype is torch.float64
    assert len(set(x)) > 1
    assert torch.all(x >= 0)
    assert torch.all(x < 1)

