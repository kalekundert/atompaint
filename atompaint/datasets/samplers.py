import logging
from reprfunc import repr_from_init

log = logging.getLogger(__name__)

class RangeSampler:

    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __iter__(self):
        yield from range(self.start, self.stop)

    def __len__(self):
        return self.stop - self.start

    __repr__ = repr_from_init

class InfiniteSampler:
    """\
    Yield indices that continue incrementing between epochs.

    This sampler is meant to be used with an infinite map-style dataset.  This 
    means a dataset that accepts integer indices of any size.  The data set 
    should also define a length, but this length just defines what is 
    considered an epoch (i.e. how often to run the validation set, to save 
    checkpoints, etc.); it doesn't relate to number of training examples (which 
    should be infinite).
    """

    def __init__(self, epoch_size: int, *, start_index=0, curr_epoch=0):
        self.start_index = start_index
        self.epoch_size = epoch_size
        self.curr_epoch = curr_epoch

    def __iter__(self):
        n = self.epoch_size
        i = self.start_index + n * self.curr_epoch
        yield from range(i, i+n)

    def __len__(self):
        return self.epoch_size

    def set_epoch(self, epoch):
        log.info("starting a new epoch; epoch=%d", epoch)
        self.curr_epoch = epoch

    __repr__ = repr_from_init

