#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 36GB
#SBATCH --cpus-per-task 8
#SBATCH -t 7-00:00:00
#SBATCH --job-name cfg-delay
#SBATCH --output /home/p/u/puzhao/run_logs/%x-%A_%a.out

echo "start"
echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo
nvidia-smi
. /geoinfo_vol1/puzhao/miniforge3/etc/profile.d/conda.sh

conda activate pytorch
PYTHONUNBUFFERED=1; 


#rm -rf $SLURM_SUBMIT_DIR/*.log
# rm -rf $SLURM_SUBMIT_DIR/*.out

# sbatch run_on_geoinfo/run_cfg_delay.sh
python main_cfg_delay.py

echo "finish"

## run
# sbatch --array=1-2 geo_run_s1s2_unet.sh
