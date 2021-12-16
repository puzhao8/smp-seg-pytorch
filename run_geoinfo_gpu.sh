#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:5
#SBATCH --mem 36GB
#SBATCH --cpus-per-task 1
#SBATCH -t 7-00:00:00

echo "start"
nvidia-smi
. /geoinfo_vol1/puzhao/miniforge3/etc/profile.d/conda.sh
conda activate pytorch
PYTHONUNBUFFERED=1; python3 test_geoinfo_gpu.py
echo "finish"
