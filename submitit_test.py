import submitit
from main_s1s2_unet import run_app

def add(a, b):
    for i in range(int(1e4)):
        c = a + b
        print(c)
    return c

# executor is the submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder="log_test")
# set timeout in min, and partition for running the job
executor.update_parameters(timeout_min=1, slurm_partition="geoinfo")
job = executor.submit(run_app)  # will compute add(5, 7)
print(job.job_id)  # ID of your job

output = job.result()  # waits for completion and returns output
# assert output == 12  # 5 + 7 = 12...  your addition was computed in the cluster