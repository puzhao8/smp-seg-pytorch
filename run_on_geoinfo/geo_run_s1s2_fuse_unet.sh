#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 100GB
#SBATCH --cpus-per-task 8
#SBATCH -t 7-00:00:00
#SBATCH --job-name fuse-unet

echo "start"
echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"

nvidia-smi
. /geoinfo_vol1/puzhao/miniforge3/etc/profile.d/conda.sh

CDC=(0 0.01 0.1 1)
CFG=${CDC[$SLURM_ARRAY_TASK_ID]}
echo "Running simulation $CFG"
# echo "singularity exec --nv /cephyr/users/puzhao/Alvis/PyTorch_v1.7.0-py3.sif python main_s1s2_fuse_unet_V1.py model.cross_domain_coef=$CFG"
echo "---------------------------------------------------------------------------------------------------------------"

conda activate pytorch
PYTHONUNBUFFERED=1; python3 main_s1s2_cdc_unet.py model.cross_domain_coef=$CFG model.batch_size=16

# rsync -a $TMPDIR/temporal-consistency/outputs/* $exp_dir
# kill $LOOPPID

rm -rf $SLURM_SUBMIT_DIR/*.log
# rm -rf $SLURM_SUBMIT_DIR/*.out

echo "finish"

# # run
# sbatch --array=0-2 run_s1s2_fuse_unet.sh