"""\
Compare metrics between different training runs.

Usage:
    analyze.py <logs>... [-m <metrics>] [-p <hparams>] [-o <path>] [-tAT]

Arguments:
    <logs>
        Paths to the log files containing the data to plot.  Path to 
        directories can also be specified, in which case the directory will be 
        searched recursively for log files.

Options:
    -m --metrics <strs>
        A comma separated list of the metrics to plot.  By default, the 
        following metrics will be displayed if present: loss, MAE (for 
        regression tasks), accuracy (for classification tasks).

    -p --hparams <path>
        A path to a file containing a regular expression that will be used to 
        extract hyperparameters from the name of each log directory.  A 
        separate plot will be made for each named capturing group in the 
        regular expression, with different values of those groups being plotted 
        in different colors.

        If a regular expression is not expressive enough to specify the 
        hyperparameter groups you want to see, use the Python API instead of 
        this command-line program.

    -t --elapsed-time
        Plot elapsed time on the x-axis, instead of step number.

    -o --output <path>
        Write the resulting plot to the given path.  If not specified, the plot 
        will be displayed in a GUI instead.

    -A --hide-raw
        Don't plot raw data points; only plot smoothed curves.

    -T --hide-train
        Only plot the validation metrics, not the epoch-level training metrics.
        Note that this option is ignored if `--metrics` is specified.
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import docopt

from tbparse import SummaryReader
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.signal import savgol_filter
from pathlib import Path
from itertools import product
from more_itertools import one, unique_everseen as unique
from operator import itemgetter

def main():
    args = docopt.docopt(__doc__)
    df = load_tensorboard_logs(map(Path, args['<logs>']))

    if args['--metrics']:
        metrics = ','.split(args['--metrics'])
    else:
        metrics = pick_metrics(df, not args['--hide-train'])

    if args['--hparams']:
        non_hparam_cols = df.columns
        df = extract_hparams(df, Path(args['--hparams']).read_text())
        hparams = [x for x in df.columns if x not in non_hparam_cols]
    else:
        hparams = ['model']

    if args['--elapsed-time']:
        x = 'elapsed_time'
    else:
        x = 'step'

    plot_training_metrics(
            df, metrics, hparams,
            x=x,
            show_raw=not args['--hide-raw'],
    )

    if out_path := args['--output']:
        plt.savefig(out_path)
    else:
        plt.show()

def load_tensorboard_logs(log_paths, cache=True, refresh=False):
    dfs = []
    for log_path in log_paths:
        dfs.append(load_tensorboard_log(log_path))

    return pl.concat(dfs)

def load_tensorboard_log(log_path, cache=True, refresh=False):
    if not refresh:
        if log_path.suffix == '.feather':
            return pl.read_ipc(log_path, memory_map=False)
        if (p := log_path / 'cache.feather').exists():
            return pl.read_ipc(p, memory_map=False)

    reader = SummaryReader(
            log_path,
            extra_columns={'dir_name', 'wall_time'},
    )
    
    df = pl.from_pandas(reader.scalars)\
           .lazy()\
           .rename({
               'dir_name': 'model',
               'tag': 'metric',
           })\
           .filter(
               pl.col('step') > 0,
               pl.col('metric').str.ends_with('_step') == False,
           )\
           .filter(
               pl.col('step').len().over('model', 'metric') > 1,
           )\
           .with_columns(
               pl.when(pl.col('model') == '')\
                 .then(pl.lit(log_path.stem))
                 .otherwise(pl.col('model'))
                 .alias('model')
           )\
           .with_columns(
               pl.col('wall_time')\
                 .map_elements(lambda t: pl.Series(infer_elapsed_time(t.to_numpy())))\
                 .over(['model', 'metric'])\
                 .alias('elapsed_time')
           )\
           .collect()

    if cache:
        if log_path.is_dir():
            cache_path = log_path / 'cache.feather'
        else:
            cache_path = log_path.with_suffix('.feather')

        df.write_ipc(cache_path)

    return df

def plot_training_metrics(df, metrics, hparams, *, x='step', show_raw=True):
    ncols = len(metrics) + 1
    nrows = len(hparams)

    fig, axes = plt.subplots(
            figsize=(ncols * 3, nrows * 3),
            ncols=ncols,
            nrows=nrows,
            constrained_layout=True,
            squeeze=False,
            sharex=True,
    )

    x_labels = {
            'elapsed_time': 'elapsed time (h)',
            'step': 'steps (×1000)',
    }
    x_getters = {
            'elapsed_time': lambda x: x / 3600,
            'step': lambda x: (x + 1) / 1000,
    }
    df_by_metric = df.partition_by('metric', as_dict=True)
    hparam_colors = _pick_hparam_colors(df, hparams)

    for i, metric in enumerate(metrics):
        t_raw = []
        y_raw = []
        color_raw = []

        for df_i in df_by_metric[metric].partition_by('model'):
            t = x_getters[x](df_i[x].to_numpy())
            y = df_i['value'].to_numpy()

            t_smooth, y_smooth = _apply_smoothing(t, y)

            t_raw.append(t)
            y_raw.append(y)
            color_raw.append([])

            for j, hparam in enumerate(hparams):
                ax = axes[j,i]

                hparam_value = one(df_i[hparam].unique())
                color = hparam_colors[hparam, hparam_value]
                color_raw[-1].append((j, color))

                ax.plot(
                        t_smooth, y_smooth,
                        label=f'{hparam_value}',
                        color=color,
                )

                if j == 0:
                    ax.set_title(metric)
                if j == len(hparams) - 1:
                    ax.set_xlabel(x_labels[x])

        if show_raw:
            ylim = axes[0,i].get_ylim()

            for t, y, colors in zip(t_raw, y_raw, color_raw):
                for j, color in colors:
                    axes[j,i].plot(t, y, color=color, alpha=0.2)
                    axes[j,i].set_ylim(*ylim)

    for i, ax_row in enumerate(axes):
        h, l = zip(
                *unique(
                    zip(*ax_row[0].get_legend_handles_labels()),
                    key=itemgetter(1),
                )
        )

        ax_row[-1].legend(
                h, l,
                borderaxespad=0,
                title=hparams[i],
                alignment='left',
                frameon=False,
                loc='center left',
        )
        ax_row[-1].axis('off')

def infer_elapsed_time(t):
    # - The wall time data includes both the time it takes to process an 
    #   example and the time spent waiting between job requeues.  We only care 
    #   about the former.  So the purpose of this function is to detect the 
    #   latter, and to replace those data points with the average of the 
    #   former.  Note that this only works if all the jobs run on the same GPU.
    #
    # - I compared a number of different outlier detection algorithms to 
    #   distinguish these two time steps.  I found that isolation forests 
    #   performed the best; classifying the data points exactly as I would on 
    #   the datasets I was experimenting with.  The local outlier factor 
    #   algorithm also performed well, but classified some true time steps as 
    #   outliers.

    dt = np.diff(t)

    outlier_detector = IsolationForest(random_state=0)
    labels = outlier_detector.fit_predict(dt.reshape(-1, 1))

    inlier_mask = (labels == 1)
    outlier_mask = (labels == -1)

    dt_mean = np.mean(dt[inlier_mask])
    dt[outlier_mask] = dt_mean

    return _cumsum0(dt)

def pick_metrics(df, include_train=False):
    known_metrics = set(df['metric'])

    stages = ['val/{}']

    if include_train:
        stages += ['train/{}_epoch']

    metrics = ['loss']

    if 'val/mae' in known_metrics:
        metrics.append('mae')
    if 'val/accuracy' in known_metrics:
        metrics.append('accuracy')

    return [
            stage.format(metric)
            for stage, metric in product(stages, metrics)
    ]

def extract_hparams(df, hparam_pattern):
    return df.with_columns(
            hparams=pl.col('model').str.extract_groups(hparam_pattern),
    ).unnest('hparams')

def _pick_hparam_colors(df, hparams):
    hparam_colors = {}

    for hparam in hparams:
        hparam_values = df[hparam].unique(maintain_order=True)
        for i, value in enumerate(hparam_values):
            hparam_colors[hparam, value] = f'C{i}'

    return hparam_colors

def _apply_smoothing(x, y):
    window_length = max(len(x) // 10, 15)

    lof = LocalOutlierFactor(2)
    labels = lof.fit_predict(y.reshape(-1, 1))
    inlier_mask = (labels == 1)

    x_inlier = x[inlier_mask]
    y_inlier = y[inlier_mask]

    if len(x_inlier) < 2 * window_length:
        x_inlier = x
        y_inlier = y

    y_smooth = savgol_filter(
            y_inlier,
            window_length=window_length,
            polyorder=2,
    )

    return x_inlier, y_smooth

def _cumsum0(x):
    # https://stackoverflow.com/questions/27258693/how-to-make-numpy-cumsum-start-after-the-first-value
    y = np.empty(len(x) + 1, dtype=x.dtype)
    y[0] = 0
    np.cumsum(x, out=y[1:])
    return y


if __name__ == '__main__':
    main()

