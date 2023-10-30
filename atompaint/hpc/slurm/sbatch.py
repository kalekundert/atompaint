"""\
Submit training jobs on a SLURM cluster, accounting for the following:

- Jobs need to run on nodes with A100 GPUs; older GPUs are too slow.
- Jobs need to configure themselves to requeue when they hit their time limits.
- Different models need different amounts of memory.

Usage:
    ap_sbatch <config>... [-d]

Arguments:
    <config>
        A config file specifying the model to train.  This config file should 
        have a "compute" section where the following options can be specified:

            train_cmd (required):
                The command to run to train the model.  This command should 
                take a single argument, which is the path to the same config 
                file provided here.

            num_cpus (optional, default: 16):
                How many CPUs to allocate for each job.

            time_h (optional, default: 2):
                How long each job can run for (in hours) before being requeued.  
                I think that shorter times get scheduled faster, but too short 
                and you'll waste time on overhead.  Checkpoints are only saved 
                at the end of each epoch, so this has to be at least long 
                enough to complete one epoch.
                
            memory_gb (optional, default: 16):
                How much memory to allocate for each job, in gigabytes (GB).

Options:
    -d --dry-run
        Print the sbatch command that would be used, but don't run it.

Environment Variables:
    The following environment variables must be defined:

        AP_SLURM_GRES:
            The `--gres` option for `sbatch`.
        
        AP_SLURM_PARTITION:
            The `--partition` option for `sbatch`.

        AP_SLURM_QOS:
            The `--qos` option for `sbatch`.

        AP_SLURM_SETUP_ENV:
            A script that will be sourced before the training command is run.  
            This script should make sure that all the relevant modules/virtual 
            environments are setup.
"""

import os
import docopt

from atompaint.config import load_compute_config, require_env
from subprocess import run
from pathlib import Path

def main():
    args = docopt.docopt(__doc__)
    config_paths = [Path(x) for x in args['<config>']]

    for config_path in config_paths:
        c = load_compute_config(config_path)

        require_env('AP_SLURM_GRES')
        require_env('AP_SLURM_PARTITION')
        require_env('AP_SLURM_QOS')
        require_env('AP_SLURM_SETUP_ENV')

        sbatch = [
                'sbatch',
                '--gres', os.environ['AP_SLURM_GRES'],
                '--partition', os.environ['AP_SLURM_PARTITION'],
                '--qos', os.environ['AP_SLURM_QOS'],
                '--job-name', c.train_command,
                '--cpus-per-task', str(c.num_cpus),
                '--time', f'0-{c.time_h}:0',  # {days}-{hours}:{minutes}
                '--mem', f'{c.memory_gb}G',
                '--signal', 'B:USR1',
                '--requeue',
                '--open-mode=append',
                '-o', '%x_%j.out',
                '-e', '%x_%j.err',
                Path(__file__).parent / 'train.sbatch',
                c.train_command,
                config_path,
        ]
        if args['--dry-run']:
            sbatch = ['echo'] + sbatch

        run(sbatch)


if __name__ == '__main__':
    main()
