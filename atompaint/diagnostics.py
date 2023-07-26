import pandas as pd
import networkx as nx

import os
import re
import time
import queue
import sqlite3

from lightning.pytorch.profilers import Profiler
from psutil import Process
from pathlib import Path
from more_itertools import one

# Should I use pytorch multiprocessing?
from multiprocessing import Process as Worker, Queue, Event

# Existing code:
#
# - Base class assumes that "summary" will be identical to what's written to 
#   file.  This is incompatible with the idea of writing to SQLite.
#
# - Base class also provides a utility for picking a nice file name.  I should 
#   probably use this.
#
# - Stage: Pretty sure this is menat in the same sense as for DataModule, i.e. 
#   either "fit", "validate", "test", or "predict".  See `setup()` and 
#   `teardown()` methods of `LightningDataModule`.
#
# - Local rank:  Pytorch runs one process per GPU.  This is a number 
#   identifying the process in question.
#   
#
# My concerns:
#
# - Don't want to store profiling info in RAM, because that could affect the 
#   results.
#
# My Plan:
#
# - Always write everything to SQLite
#
# - In summary method, read DB and print resulting DFs
#
#
#
#
#
# - Overwrite describe and 

from typing import Optional, Union

class SmapsProfiler(Profiler):

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


def get_elapsed_time(t0):
    return time.monotonic() - t0


def filter_by_pids(mm, pids=None, parent_pids=None):
    if pids or parent_pids:
        mask = mm['pid'].isin(pids or []) | mm['parent_pid'].isin(parent_pids or [])
        return mm[mask]
    else:
        return mm

def filter_by_time(mm, actions, action_patterns):
    if not action_patterns:
        return mm

    mask = pd.Series(False, index=df.index)

    for pattern in action_patterns:
        mask |= df['name'].str.contains(pattern)

    start = df[mask].start.min()
    stop = df[mask].stop.max()

    return mm.query('@start <= time <= @stop')

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

def aggregate_by_type(mm, type):
    # This function is meant to be used with `groupby.apply()`.  Also note that 
    # the `type` column is created by `aggregate_file_mmaps()`.
    
    if type != 'all':
        mask = mm['path'].isin(type.split('+'))
        mm = mm[mask]

    mem_cols = [
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
    return mm.aggregate({x: 'sum' for x in mem_cols})

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

def parse_pids(pid_str):
    return [int(x) for x in pid_str.split(',')] if pid_str else None

def count_process_ancestors(mm):
    g = nx.DiGraph()
    pids = mm[['pid', 'parent_pid']].drop_duplicates()
    for _, (pid, ppid) in pids.iterrows():
        g.add_edge(ppid, pid)

    root = one(n for n, d in g.in_degree() if d == 0)
    return nx.shortest_path_length(g, root)

def iter_pids_depth_first(mm):
    g = nx.DiGraph()
    pids = mm[['pid', 'parent_pid']].drop_duplicates()
    for _, (pid, ppid) in pids.iterrows():
        g.add_edge(ppid, pid)

    root = one(n for n, d in g.in_degree() if d == 0)
    yield from nx.dfs_edges(g, root)

# Options
# - Color
#   - Each process is nice, but might get busy...
#
#   - Choose colors to minimize timepoint overlaps.
#   - How to handle 16 processes nicely?
#
#   - Color groups based on level in parent hierarchy?
#
# - Max children
#
#   - Too many can clutter the display, and is redundant.
#   - Way to limit colors, as well.
#
#
# - Shared/private split
#   - These add up to RSS, though: not so useful
#   - Maybe useful to see Private_Clean + Private_Dirty, basically RSS that 
#     belongs solely to this process.
#   - And PSS - (Private_Clean + Private_Dirty) is the amount of shared RSS, 
#     without double counting.
#
#   - Include solid lines in legend always...

if __name__ == '__main__':
    import docopt
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    usage = """\
Plot memory benchmarking data collected by SmapsProfiler.

Usage:
    diagnostics <path> [-t <type>]... [-m <metric>] [-p <pids>] [-P <pids>] 
        [-a <action>]... [-srA]

Arguments:
    <path>
        A path to an `*.db` file produced by the smaps profiler.   This is an 
        SQLite database containing information on (i) what the program was 
        doing at different times and (ii) how much memory it was using.
        
Options:
    -s --shared

    -r --private

    -A --all

    -t --type <name or sum of names>                    [default: all]
        Which "type" of memory to show.  The most common types, along with 
        brief descriptions, are given below.  You can specify either an 
        individual type (e.g. 'anon') or a sum of types (e.g. 'anon+heap').  
        You can also specify this option multiple times, in which case a 
        different plot will be made for each.

        all:
            The sum of everything described below.

        anon:
            Anonymous regions of virtual memory.  This includes any large (>128 
            kB) allocations made by `malloc()`, which often account for a 
            substantial fraction of the memory used by the whole process.  This 
            also includes memory meant to be shared between a process and its 
            children.  Torch uses this mechanism to efficiently communicate 
            tensors between processes.

            Note that in `/proc/PID/smaps`, these regions simply not given 
            names.  For convenience, this analysis program combines all these 
            regions and refers to the result as "anon".

        heap:
            Small (<128 kB) allocations made by `malloc()`.  This is usually 
            smaller than [anon], but can still account for a substantial 
            fraction of the memory used by the whole process.

        stack:
            Memory that is automatically allocated and deallocated as functions 
            enter and exit.  No python objects are allocated on the stack, and 
            this will generally not be a large source of memory usage.

        file:
            Memory directly mapped from a file.  Typically this corresponds to 
            executables and shared libraries, but some programs may use this 
            kind of memory to efficiently manage large data structures.

            Note that in `/proc/PID/smaps`, these regions are referred to by 
            their actual file names.  For convenience, this analysis program 
            combines all these regions and refers to the result as "file".

        vsyscall, vdso, vvar:
            Memory segments that are used to accelerate common system calls, 
            e.g. `gettimeofday()`.  These should never be a significant source 
            of memory usage, and are not shown in the summary printed at the 
            end of a profiling run.

    -p --pids <comma separated integers>
        Plot only processes with the specified ids.  By default, all processes 
        are plotted.

    -P --parent-pids <comma separated integers>
        Plot only processes that are direct children of the given process ids.  
        By default, all processes are plotted.  If you specify `-p` and `-P`, 
        processes included by either option will be plotted.

    -a --action <regex>
        Only plot memory usage for timepoints that coincide with the given 
        action.  This option can be specified multiple times to inlcude 
        multiple actions.  Each action is specified as a regular expression.

More information:
    - Link to blog post about shared memory and pytorch loaders
    - Link to PyTorch issue
    - Link to kernel docs
"""

    args = docopt.docopt(usage)

    db = sqlite3.connect(Path(args['<path>']))
    actions = pd.read_sql('SELECT * FROM actions', db)
    mem_maps = mm = pd.read_sql('SELECT * FROM memory_maps', db)

    mm = filter_by_pids(
            mm,
            parse_pids(args['--pids']),
            parse_pids(args['--parent-pids']),
    )
    mm = filter_by_time(mm, actions, args['--action'])
    mm = aggregate_file_mmaps(mm)

    fig, axes = plt.subplots(
            len(args['--type']), 1,
            squeeze=False,
            layout='constrained',
    )

    num_ancestors = count_process_ancestors(mm)

    for i, type in enumerate(args['--type']):
        ax = axes[i, 0]
        df = mm.groupby(['pid', 'parent_pid', 'time'])\
                .apply(aggregate_by_type, type)\
                .reset_index()\
                .set_index('pid')

        # Legend:
        # - False artists
        # - Align with axis -- good as is, but why not perfect?

        legend_artists = []
        legend_labels = []
        
        for parent_pid, pid in iter_pids_depth_first(mm):
            n = num_ancestors[pid]
            if n > 1:
                figure_space = '\u2007'
                m_space = '\u2001'
                prefix = figure_space * (n - 2) + '↳'
                prefix2 = figure_space * (n - 2) + m_space
            else:
                prefix = ''
                prefix2 = ''

            kwargs = {}

            if args['--all'] or (not args['--private'] and not args['--shared']):
                lines = ax.plot(
                        df.loc[pid]['time'],
                        df.loc[pid]['pss'] / 1e9,
                        zorder=2,
                )
                kwargs['color'] = lines[0].get_color()

            x = df.loc[pid]

            private_pss = x['private_clean'] + x['private_dirty']
            shared_pss = x['pss'] - private_pss

            if args['--private']:
                lines = ax.plot(
                        x['time'],
                        private_pss / 1e9,
                        linestyle=(0, (3, 1, 1, 1)),
                        zorder=1,
                        **kwargs,
                )
                kwargs['color'] = lines[0].get_color()

            if args['--shared']:
                lines = ax.plot(
                        x['time'],
                        shared_pss / 1e9,
                        linestyle=(0, (1, 1)),
                        zorder=0,
                        **kwargs,
                )
                kwargs['color'] = lines[0].get_color()

            artist = mlines.Line2D(
                    [], [],
                    label=f'{prefix}{pid}',
                    **kwargs,
            )
            legend_artists.append(artist)


        artist = mlines.Line2D(
                [], [],
                color='white',
        )
        legend_artists.append(artist)

        artist = mlines.Line2D(
                [], [],
                label=f'private + shared',
                color='black',
        )
        legend_artists.append(artist)

        artist = mlines.Line2D(
                [], [],
                label=f'private',
                color='black',
                linestyle=(0, (3, 1, 1, 1)),
        )
        legend_artists.append(artist)

        artist = mlines.Line2D(
                [], [],
                label=f'shared',
                color='black',
                linestyle=(0, (1, 1)),
        )
        legend_artists.append(artist)


        #for (pid, parent_pid), g in df.groupby(['pid', 'parent_pid']):
        #    ax.plot(
        #            g['time'],
        #            g[metric] / 1e9,
        #            #color=f'C{num_ancestors[pid] - 1}',
        #            label=f'{pid}',
        #    )

        if type != 'all':
            ax.set_title(type)
        ax.set_xlabel('time (s)')
        ax.set_ylabel(f'PSS (GB)')

        #fig.legend(loc='outside right upper')
        ax.legend(
                handles=legend_artists,
                bbox_to_anchor=(1.05, 1),
                borderaxespad=0.,
        )

    plt.show()
