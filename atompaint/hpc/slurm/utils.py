import os

def is_slurm():
    return 'SLURM_JOB_ID' in os.environ

def get_job_id():
    array_job_id = os.getenv('SLURM_ARRAY_JOB_ID')
    if array_job_id is not None:
        array_task_id = os.environ['SLURM_ARRAY_TASK_ID']
        return f'{array_job_id}_{array_task_id}'
    else:
        return os.environ['SLURM_JOB_ID']

