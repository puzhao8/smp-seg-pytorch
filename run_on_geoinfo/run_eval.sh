#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 36GB
#SBATCH --cpus-per-task 8
#SBATCH -t 7-00:00:00
#SBATCH --job-name eval
#SBATCH --output /home/p/u/puzhao/smp-seg-pytorch/run_logs/%x-%A_%a.out


echo "start"
echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo
nvidia-smi
. /geoinfo_vol1/puzhao/miniforge3/etc/profile.d/conda.sh

# module --ignore-cache load "intel"
# PROJECT_DIR=/cephyr/NOBACKUP/groups/snic2021-7-104/puzhao-snic-500G/smp-seg-pytorch


# # Choose different config files
# # DIRS=($(find /cephyr/NOBACKUP/groups/snic2021-7-104/puzhao-snic-500G/smp-seg-pytorch/config/s1s2_cfg/))
# DIRS=($(find $PROJECT_DIR/config/s1s2_unet/))
# DIRS=${DIRS[@]:1}
# CFG=${DIRS[$SLURM_ARRAY_TASK_ID]}
# echo "Running simulation $CFG"

# # Choose different sensors
# SAT=('S1' 'S2' 'ALOS')
# SAT=('alos' 's1' 's2')
# CFG=${SAT[$SLURM_ARRAY_TASK_ID]}
# echo "Running simulation $CFG"
# echo "python3 main_s1s2_unet.py s1s2_unet=$CFG model.batch_size=32"
# echo "---------------------------------------------------------------------------------------------------------------"

# singularity exec --nv /cephyr/users/puzhao/Alvis/PyTorch_v1.7.0-py3.sif python main_s1s2_unet.py s1s2_unet=$CFG
conda activate pytorch
PYTHONUNBUFFERED=1; 

# sbatch run_on_geoinfo/run_eval.sh
# python3 s1s2_evaluator.py \
#     --config-name=distill_unet.yaml \
#     model.ARCH=distill_unet \
#     experiment.note=eval

# sbatch run_on_geoinfo/run_eval.sh
python3 s1s2_evaluator_prg.py \
            --config-name=s1s2_cfg_prg.yaml \
            DATA.SATELLITES=['S1'] \
            DATA.INPUT_BANDS.S2=['B4','B8','B12'] \


#rm -rf $SLURM_SUBMIT_DIR/*.log
# rm -rf $SLURM_SUBMIT_DIR/*.out

echo "finish"

## run
# sbatch --array=1-2 geo_run_s1s2_unet.sh
