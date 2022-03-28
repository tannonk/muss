import time
import torch
from tqdm import tqdm
from muss.utils.submitit import get_executor

cluster = 'slurm'
slurm_partition = 'generic,volta'
slurm_gres="gpu:1"
slurm_array_parallelism = 1

def check_for_cuda():
    return str(torch.cuda.is_available())

# Split CCNet shards into subshards
print('Testing access to cpu partition')
executor = get_executor(cluster=cluster, slurm_partition='generic,hpc', timeout_min=1, slurm_array_parallelism=1, slurm_gres=None, cpus_per_task=1, mem_gb=1)
# print(dir(executor))
jobs = []
with executor.batch():
    # for i in range(2):
    job = executor.submit(check_for_cuda)
    jobs.append(job)
print([job.job_id for job in jobs])
print('Cuda available for jobs', [job.result() for job in tqdm(jobs)])  # Wait for the jobs to finish


print('Testing access to gpu partititon')
executor = get_executor(cluster=cluster, slurm_partition='volta,vesta', timeout_min=1, slurm_array_parallelism=1, slurm_gres=slurm_gres, cpus_per_task=1, mem_gb=1)
# print(executor)
jobs = []
with executor.batch():
    # for i in range(2):
    job = executor.submit(check_for_cuda)
    jobs.append(job)
print([job.job_id for job in jobs])
print('Cuda available for jobs', [job.result() for job in tqdm(jobs)])  # Wait for the jobs to finish

print('done')
