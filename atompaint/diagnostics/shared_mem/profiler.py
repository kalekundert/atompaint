import pandas as pd

import os
import time
import sqlite3

from lightning.pytorch.profilers import Profiler
from psutil import Process
from pathlib import Path
from multiprocessing import Process as Worker, Event
from typing import Optional, Union

class SharedMemoryProfiler(Profiler):

    def __init__(
            self,
            dirpath: Optional[Union[str, Path]] = None,
            filename: Optional[str] = None,
            polling_interval_s: int = 10,
            aggregate_file_mmaps: bool = True,
    ):
        super().__init__(dirpath, filename)

        self._db = None
        self._worker = None
        self._t0 = time.monotonic()
        self._start_times = {}
        self._teardown_event = None

        self.polling_interval_s = polling_interval_s
        self.aggregate_file_mmaps = aggregate_file_mmaps

    def setup(
            self,
            stage: str,
            local_rank: Optional[int] = None,
            log_dir: Optional[str] = None,
    ) -> None:
        super().setup(stage, local_rank, log_dir) 

        def log_memory_maps(df):
            df.to_sql('memory_maps', self._db, if_exists='append', index=False)

        db_path = Path(self._prepare_filename(extension='.db'))
        if db_path.exists():
            db_path.unlink()

        self._db = sqlite3.connect(db_path)
        self._teardown_event = Event()
        self._worker = Worker(
                target=poll_memory_maps,
                kwargs=dict(
                    pid=os.getpid(),
                    t0=self._t0,
                    polling_interval_s=self.polling_interval_s,
                    agg_file_mmaps=self.aggregate_file_mmaps,
                    teardown_event=self._teardown_event,
                    on_capture=log_memory_maps,
                ),
                daemon=True,
        )
        self._worker.start()

    def start(self, action_name):
        if self._db is None:
            return

        assert action_name not in self._start_times
        self._start_times[action_name] = get_elapsed_time(self._t0)

    def stop(self, action_name):
        if self._db is None:
            return

        action = {
                'name': action_name,
                'start': self._start_times.pop(action_name),
                'stop': get_elapsed_time(self._t0),
        }
        df = pd.DataFrame([action])

        # Note that this may block until the database is not being written to 
        # by the worker process.
        df.to_sql('actions', self._db, if_exists='append', index=False)

    def summary(self) -> str:
        self._teardown_event.set()
        self._worker.join()

        # Read database
        actions = pd.read_sql('SELECT * FROM actions', self._db)
        mem_maps = mm = pd.read_sql('SELECT * FROM memory_maps', self._db)

        # Make summary
        mm = aggregate_file_mmaps(mm)
        mm = hide_syscall_memory(mm)
        mm = make_bytes_human_readable(mm)
        mm = mm.set_index(['time', 'parent_pid', 'pid'])

        print(actions.to_string())
        print()
        print(mm.to_string())

def poll_memory_maps(
        pid,
        *,
        t0,
        polling_interval_s,
        agg_file_mmaps,
        teardown_event,
        on_capture,
):
    while True:
        pids = find_relevant_pids(pid, recursive=True)
        mm = query_memory_maps(pids)
        mm['time'] = get_elapsed_time(t0)

        if agg_file_mmaps:
            mm = aggregate_file_mmaps(mm)

        on_capture(mm)

        if teardown_event.wait(polling_interval_s):
            return

def find_relevant_pids(pid, *, recursive=False):
    monitor_pid = os.getpid()
    return [pid] + [
            x.pid
            for x in Process(pid).children(recursive=recursive)
            if x.pid != monitor_pid
    ]

def query_memory_maps(pids):
    df = pd.concat(
            [query_memory_maps_for_pid(x) for x in pids],
            ignore_index=True,
    )
    return df

def query_memory_maps_for_pid(pid):
    try:
        p = Process(pid)
        df = pd.DataFrame(p.memory_maps())
        df['pid'] = pid
        df['parent_pid'] = p.ppid()
        return df

    # It's possible that the process in question will have exited in between 
    # now and when we got it's PID, so we have to handle this case gracefully.
    except psutil.NoSuchProcess:
        return pd.DataFrame()

def aggregate_file_mmaps(mm):

    def by_type(i):
        path = mm.at[i, 'path']

        if re.fullmatch(r'\[.*\]', path):
            return path
        else:
            return '[file]'

    groups = ['time', 'parent_pid', 'pid', by_type]
    return mm.groupby(groups)\
            .sum(numeric_only=True)\
            .rename_axis(index={None: 'path'})\
            .reset_index()

def make_bytes_human_readable(mm):
    byte_cols = [
            'rss',
            'size',
            'pss',
            'shared_clean',
            'shared_dirty',
            'private_clean',
            'private_dirty',
            'referenced',
            'anonymous',
            'swap',
    ]
    mm[byte_cols] /= 1e9
    return mm.rename(columns={x: f'{x}_GB' for x in byte_cols})

def hide_syscall_memory(mm):
    mask = ~mm['path'].isin(['[vsyscall]', '[vdso]', '[vvar]'])
    return mm[mask]

def get_elapsed_time(t0):
    return time.monotonic() - t0



