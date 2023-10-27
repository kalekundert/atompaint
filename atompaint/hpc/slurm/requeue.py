import os
import logging
import time

from .utils import get_job_id
from lightning.pytorch.callbacks import Callback
from more_itertools import pairwise
from subprocess import run

log = logging.getLogger(__name__)

class RequeueBeforeTimeLimit(Callback):
    
    def __init__(self, buffer=1.2):
        super().__init__()
        self.epoch_splits = []
        self.buffer = buffer
        self.requeue_needed = False

    def on_train_start(self, *_):
        self.epoch_splits.append(time.time())

    def on_train_epoch_end(self, trainer, *_):
        now = time.time()
        self.epoch_splits.append(now)

        # Only count the first epoch if it's the only one that's happened so 
        # far.  I don't expect the first iteration to be predictive of later 
        # iterations, because it has to warm up a bunch of caches.
        i = len(self.epoch_splits) > 2
        epoch_times = [
                end - start
                for start, end in pairwise(self.epoch_splits[i:])
        ]

        time_limit = float(os.environ['SLURM_JOB_END_TIME'])
        time_remaining = time_limit - now
        time_needed = max(epoch_times) * self.buffer
        self.requeue_needed = time_needed > time_remaining
        trainer.should_stop = self.requeue_needed

        log.info("checking if time for another epoch; last_epoch=%.0fs time_remaining=%.0fs time_needed=%.0fs stop=%r", epoch_times[-1], time_remaining, time_needed, self.requeue_needed)

    def on_train_end(self, *_):
        if self.requeue_needed:
            job_id = get_job_id()
            log.info("requeuing the job; job_id=%s", job_id)
            scontrol = ['scontrol', 'requeue', job_id]
            run(scontrol)

            # Give SLURM time to shut down the job.  I'm not sure if this is 
            # necessary---will the job still get requeued if it exits 
            # successfully?---but I don't think it can hurt.
            time.sleep(60)

            raise SystemExit('requeue')
