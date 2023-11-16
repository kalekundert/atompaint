"""\
Plot memory benchmarking data collected by SharedMemoryProfiler.

Usage:
    diagnostics actions <path>
    diagnostics plot <path> [-m <metric>] [-p <pids>] [-P <ppids>]
        [-l <levels>] [-L] [-M <level>] [-a <action>]... [-n] [-x <paths>]...

Arguments:
    <path>
        A path to an `*.db` file produced by the profiler.   This is an SQLite 
        database containing information on (i) what the program was doing at 
        different times and (ii) how much memory it was using at each time.
    
Options:
    -m --metrics <t|p|s>                            [default: t]
        A string of characters specifying what values to plot for each 
        process/collection of processes.  The following characters are 
        understood:

        t: Loosely, this is the "total" memory used by the process.  More 
           precisely, this is the "proportional set size" (PSS) of the process.  
           The PSS is the amount of memory accessible to the process which is 
           actually resident in RAM, with a correction to avoid double-counting 
           memory that is accessible to other processes as well.  The sum of 
           PSS for every process currently running will equal the amount of 
           physical memory currently in use.

        p: The amount of memory that is "private" to the process in question.  
           In other words, memory locations that can not be accessed by any 
           other process.  This includes memory that other processes would be 
           granted access to if they wanted (e.g. memory mapped files), so long 
           as none have actually done so yet.  Note that even if you've 
           configured the plot to aggregate data from multiple processes (e.g. 
           with the `-L` or `-M` options), memory that is shared entirely 
           within that aggregate will still be counted as shared.

        s: The amount of memory that is "shared" between at least two 
           processes.  This is really the difference between the "total" and 
           "private" values; as such, the amount of shared memory is normalized 
           by the number of processes sharing it.

    -p --pids <comma separated integers>
        Plot only processes with the specified ids.  By default, all processes 
        are plotted.

    -P --parent-pids <comma separated integers>
        Plot only processes that are direct children of the given process ids.  
        By default, all processes are plotted.  If you specify `-p` and `-P`, 
        processes included by either option will be plotted.

    -l --levels <comma separated integers>
        Plot only processes at the given "levels".  For example, the process 
        running the main training loop is level #1.  The children of the 
        process, which include the data loaders, are level #2.  Any processes 
        spawned by the 2nd level processes would be level #3, and so on.  Note 
        that levels count from 1.

    -L --group-by-level
        Don't plot memory usage for individual processes, but instead group 
        together all of the processes at a given "level".  See `--levels` for a 
        description of what exactly a level is.

    -M --merge-subprocs <level>
        Treat any processes beyond the specified level as if they were actually 
        a part of their parent process.  For example, `--merge-subprocs 1` 
        would mean that all the memory usage for every process spawned by the 
        training loop would be combined into a single plot.  See `--levels` for 
        a description of what exactly a level is.

    -a --action <regex>
        Indicate when the specified action was occurring on the plot, by 
        shading the background.  The argument should be a regular expression 
        (python syntax) that may match any number of actions.  Use the 
        `actions` subcommand to see all the actions that occurred during the 
        profiling run.  This argument can be specified multiple times; the 
        first 10 specifications will be shaded with unique colors.

    -n --num-processes
        Instead of plotting memory usage, just plot how many processes are 
        running at each point in time.  This can be helpful to make sure that 
        data loaders are being created/destroyed as expected.

    -x --path <name or sum of names>               [default: all]
        Which "types" of memory to show.  More specifically, each region of 
        memory accessible to a process may be associated with a path.  If a 
        path is given, it could either specify the actual file being mapped to 
        that region or a short-hand name for the region.  Some possible values 
        for this argument are shown below.  Note that you can specify either an 
        individual type (e.g. 'anon') or a sum of types (e.g. 'anon+heap').  
        You can also specify this argument multiple times, to separately plot 
        different paths and/or sums or paths.

        all:
            The sum of everything described below.

        anon:
            Anonymous regions of virtual memory, i.e. those with no path given.  
            This includes any large (>128 kB) allocations made by `malloc()`, 
            which often account for a substantial fraction of the memory used 
            by the whole process.  This also includes memory meant to be shared 
            between a process and its children.  Torch uses this mechanism to 
            efficiently communicate tensors between processes.

            Note that most processes will have multiple, distinct regions of 
            anonymous memory.  This analysis program, however, does not provide 
            the means to distinguish between such regions.  Instead, all the 
            anonymous regions are aggregated and given the name "anon".

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

More information:
- Link to blog post about shared memory and pytorch loaders
  See lab notebook
- Link to PyTorch issue
- Link to kernel docs
"""

import docopt
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import sys

from .analysis import (
        plot_actions, plot_num_processes, plot_processes, UsageError,
        parse_pids, parse_levels, parse_level, parse_metrics,
)
from pathlib import Path

try:
    args = docopt.docopt(__doc__)

    db = sqlite3.connect(Path(args['<path>']))
    actions = pd.read_sql('SELECT * FROM actions', db)
    mem_maps = pd.read_sql('SELECT * FROM memory_maps', db)

    if args['actions']:
        print(actions.to_string())

    if args['plot']:
        ax = plt.subplot()

        plot_actions(ax, actions, args['--action'])

        if args['--num-processes']:
            plot_num_processes(ax, mem_maps)

        else:
            plot_processes(
                    ax,
                    mem_maps,
                    group_by_level=args['--group-by-level'],
                    pids=parse_pids(args['--pids']),
                    ppids=parse_pids(args['--parent-pids']),
                    levels=parse_levels(args['--levels']),
                    merge_after_level=parse_level(args['--merge-subprocs']),
                    paths=args['--path'],
                    metrics=parse_metrics(args['--metrics']),
            )

            #fig.legend(loc='outside right upper')
            # Too busy
            # ax.legend(
            #         handles=legend_artists,
            #         bbox_to_anchor=(1.05, 1),
            #         borderaxespad=0.,
            # )

        plt.tight_layout()
        plt.show()

except UsageError as err:
    print(f"Error: {err}")
    sys.exit(1)
