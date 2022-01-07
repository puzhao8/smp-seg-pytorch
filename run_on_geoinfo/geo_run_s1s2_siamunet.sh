#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 36GB
#SBATCH --cpus-per-task 8
#SBATCH -t 7-00:00:00
#SBATCH --job-name fcnn4cd


echo "start"
echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
# SBATCH --output=/home/p/u/puzhao/smp-seg-pytorch/logs/$JOB_ID

nvidia-smi
. /geoinfo_vol1/puzhao/miniforge3/etc/profile.d/conda.sh

CDC=("Vanilla_unet" "SiamUnet_conc" "SiamUnet_diff")
CFG=${CDC[$SLURM_ARRAY_TASK_ID]}
echo "Running simulation $CFG"
echo "python3 main_s1s2_fcnn4cd.py model.ARCH=$CFG"
echo "---------------------------------------------------------------------------------------------------------------"

conda activate pytorch
PYTHONUNBUFFERED=1; python3 main_s1s2_siamunet.py model.ARCH=$CFG model.max_epoch=20

# singularity exec --nv /cephyr/users/puzhao/Alvis/PyTorch_v1.7.0-py3.sif python3 main_s1s2_fcnn4cd.py data.satellites=['S1'] model.ARCH=$CFG 

# singularity exec --nv /cephyr/users/puzhao/Alvis/PyTorch_v1.7.0-py3.sif python3 s1s2_evaluator.py

# rsync -a $TMPDIR/temporal-consistency/outputs/* $exp_dir
# kill $LOOPPID

rm -rf $SLURM_SUBMIT_DIR/*.log
# rm -rf $SLURM_SUBMIT_DIR/*.out

echo "finish"

# # run
# sbatch --array=0-2 run_s1s2_fcnn4cd.sh