import numpy as np
import pandas as pd
import networkx as nx
import re

from more_itertools import one

MEM_COLS = [
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
SUM_MEM_COLS = {k: np.sum for k in MEM_COLS}

def plot_num_processes(ax, mm):
    df = mm.groupby('time', as_index=False).agg({'pid': 'nunique'})

    ax.plot(
            df['time'] / 60, df['pid'],
    )
    ax.set_xlabel('time (min)')
    ax.set_ylabel('# processes')

def plot_actions(ax, df, patterns, cmap='Pastel1'):
    import matplotlib as mpl

    if not patterns:
        return
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]

    for _, row in df.iterrows():
        for i, pattern in enumerate(patterns):
            if re.search(pattern, row['name']):
                ax.axvspan(row['start'] / 60, row['stop'] / 60, color=cmap(i))

def plot_processes(
        ax, mm,
        *,
        group_by_level=False,
        pids=None,
        ppids=None,
        levels=None,
        merge_after_level=None,
        paths=None,
        metrics=None,
):
    traces = []

    for style1, df1 in iter_process_groups(
            mm,
            group_by_level=group_by_level,
            pids=pids,
            ppids=ppids,
            levels=levels,
            merge_after_level=merge_after_level,
    ):
        for style2, df2 in iter_path_groups(df1, path_exprs=paths):
            traces.append((style1 | style2, df2))

    if not traces:
        raise UsageError('no data selected')

    color_handles = assign_colors(traces)
    metric_handles = {}

    for style, df in traces:
        metric_handles |= plot_pss_trace(ax, df, style, metrics=metrics)

    make_legend(ax, color_handles, metric_handles)

def plot_pss_trace(ax, mm, style, metrics=None):
    from matplotlib.lines import Line2D

    mm = mm.groupby('time', as_index=False).agg(SUM_MEM_COLS)

    time = mm['time'] / 60
    all_pss = mm['pss'] / 1e9
    private_pss = (mm['private_clean'] + mm['private_dirty']) / 1e9
    shared_pss = all_pss - private_pss

    legend_handles = {}

    if not metrics or 'total' in metrics:
        ax.plot(
                time,
                all_pss,
                color=style['color'],
                zorder=5,
        )
        legend_handles['total'] = Line2D(
                [], [],
                color='black',
                label="Private + Shared",
        )

    if metrics and 'private' in metrics:
        dot_dash = (0, (3, 1, 1, 1))
        ax.plot(
                time,
                private_pss,
                color=style['color'],
                linestyle=dot_dash,
                zorder=4,
        )
        legend_handles['private'] = Line2D(
                [], [],
                color='black',
                linestyle=dot_dash,
                label="Private",
        )

    if metrics and 'shared' in metrics:
        dot_dot = (0, (1, 1))
        ax.plot(
                time,
                shared_pss,
                color=style['color'],
                linestyle=dot_dot,
                zorder=3,
        )
        legend_handles['shared'] = Line2D(
                [], [],
                color='black',
                linestyle=dot_dot,
                label="Shared",
        )

    ax.set_xlabel('time (min)')
    ax.set_ylabel('PSS (GB)')
    ax.set_ylim(bottom=0)

    return legend_handles

def assign_colors(traces, cmap='tab10'):
    import matplotlib as mpl
    from matplotlib.lines import Line2D

    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]

    if not traces:
        return []

    # Count how many traces overlap at each time point.  This information will 
    # be used to avoid assigning the same color to two processes that are 
    # running at the same time.

    times = pd.concat(df['time'] for _, df in traces).sort_values().unique()
    counts = pd.Series(0, index=times)
    max_rank = 0

    for style, df in traces:
        t0 = df['time'].iloc[0]
        t1 = df['time'].iloc[-1]

        rank = style['rank'] = counts.loc[t0:t1].max()
        max_rank = max(max_rank, rank)

        counts.loc[t0:t1] += 1

    # Choose a color for each trace.  If there are too many overlapping traces, 
    # color based on level instead of process id.

    legend_meta = {}

    for style, df in traces:
        if max_rank > len(cmap.colors):
            level = df['level'].min()
            style['color'] = cmap(level - 1)
            legend_meta.setdefault(level - 1, {'type': 'level'})
        else:
            style['color'] = cmap(style['rank'])
            if 'pid' in style:
                legend_meta\
                        .setdefault(style['rank'], {'type': 'pid', 'pids': []})['pids']\
                        .append(style['pid'])
            else:
                legend_meta.setdefault(style['level'] - 1, {'type': 'level'})

    # Work out an appropriate label for each color.

    legend_handles = []

    for k, meta in legend_meta.items():
        if meta['type'] == 'level':
            legend_handles.append(
                    Line2D(
                        [], [],
                        color=cmap(k),
                        label="Main process" if k == 0 else f"Child processes (level {k})",
                    ),
            )
        else:
            pid_strs = [str(x) for x in meta['pids']]
            if len(pid_strs) > 3:
                pid_strs = [*pid_strs[:3], '…']

            legend_handles.append(
                    Line2D(
                        [], [],
                        color=cmap(k),
                        label='PID=' + ','.join(pid_strs),
                    ),
            )

    return legend_handles

def make_legend(ax, color_handles, metric_handles):
    from matplotlib.lines import Line2D

    if not color_handles:
        return

    handles = color_handles.copy()

    if set(metric_handles) != {'total'}:
        handles += [
                Line2D([], [], linestyle='none', label=''),
                *(metric_handles[k] for k in ['total', 'private', 'shared'] if k in metric_handles),
        ]

    ax.legend(
            handles=handles,
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0.,
    )

def iter_process_groups(
        mm,
        group_by_level=False,
        pids=None,
        ppids=None,
        levels=None,
        merge_after_level=None,
):

    g = make_process_graph(mm)
    ancestor_counts = count_process_ancestors(g)
    mm['level'] = mm['pid'].map(ancestor_counts)
    mm0 = mm

    if merge_after_level:
        merged_pids = merge_pids_after_level(g, ancestor_counts, merge_after_level)
        mm['pid'] = mm['pid'].map(lambda x: merged_pids.get(x, x))
        mm['parent_pid'] = mm['pid'].map(lambda x: one(g.predecessors(x)))

    if pids is not None:
        mm = mm[mm['pid'].isin(pids)]
        if mm.empty:
            raise UsageError(f"""\
can't find the requested processes:
- Requested PIDs: {','.join(map(str, pids))}
- Known PIDs: {','.join(map(str, mm0['pid'].drop_duplicates()))}""")

    if ppids is not None:
        mm = mm[mm['parent_pid'].isin(ppids)]
        if mm.empty:
            raise UsageError(f"""\
can't find the requested parent processes:
- Requested parent PIDs: {','.join(map(str, ppids))}
- Known parent PIDs: {','.join(map(str, mm0['parent_pid'].drop_duplicates()))}""")

    if levels is not None:
        mm = mm[mm['level'].isin(levels)]
        if mm.empty:
            raise UsageError(f"""\
can't find any processes at the requested level:
- Requested levels: {','.join(map(str, levels))}
- Maximum level: {mm0['level'].max()}""")

    if group_by_level:
        for level, df in mm.groupby('level'):
            yield dict(level=level, merged=merge_after_level), df
    else:
        for (ppid, pid), df in mm.groupby(['parent_pid', 'pid']):
            yield dict(ppid=ppid, pid=pid, merged=merge_after_level), df

def iter_path_groups(df, path_exprs):
    for path_expr in path_exprs:
        if path_expr in {None, 'all', '[all]'}:
            yield {}, df
        else:
            keys = {
                    normalize_path(x, df['path'])
                    for x in path_expr.split('+')
            }
            mask = df['path'].isin(keys)
            yield dict(paths=path_expr), df[mask]

def normalize_path(path, candidates):
    bracket_path = f'[{path}]'

    if (path not in candidates) and (bracket_path in candidates):
        return bracket_path
    else:
        return path

def make_process_graph(mm):
    g = nx.DiGraph()
    pids = mm[['parent_pid', 'pid']].drop_duplicates()
    for _, (ppid, pid) in pids.iterrows():
        g.add_edge(ppid, pid)
    return g

def count_process_ancestors(g):
    root = one(n for n, d in g.in_degree() if d == 0)
    return nx.shortest_path_length(g, root)

def merge_pids_after_level(g, ancestor_counts, level_threshold):
    merged_pids = {}

    for pid, level in ancestor_counts.items():
        if level < level_threshold:
            merged_pids[pid] = pid
        if level == level_threshold:
            for _, child_pid in nx.bfs_edges(g, pid):
                merged_pids[child_pid] = pid

    return merged_pids

def parse_metrics(metrics_str):
    metrics = {x[0]: x for x in ['total', 'private', 'shared']}
    return {metrics[x] for x in metrics_str}

def parse_pids(pids_str):
    return [int(x) for x in pids_str.split(',')] if pids_str else None

def parse_levels(levels_str):
    return [int(x) for x in levels_str.split(',')] if levels_str else None

def parse_level(level_str):
    return int(level_str) if level_str else None

class UsageError(Exception):
    pass
