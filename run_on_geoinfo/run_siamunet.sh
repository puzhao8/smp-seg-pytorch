#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 36GB
#SBATCH --cpus-per-task 8
#SBATCH -t 7-00:00:00
#SBATCH --job-name test-xy
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
SAT=('SiamUnet_conc' 'SiamUnet_diff')
CFG=${SAT[$SLURM_ARRAY_TASK_ID]}
echo "Running simulation $CFG"
# echo "python3 main_s1s2_unet.py model.batch_size=32"
echo "---------------------------------------------------------------------------------------------------------------"

# singularity exec --nv /cephyr/users/puzhao/Alvis/PyTorch_v1.7.0-py3.sif python main_s1s2_unet.py s1s2_unet=$CFG
conda activate pytorch
PYTHONUNBUFFERED=1; 

# sbatch --array=0-1 run_on_geoinfo/run_siamunet.sh

# # SiamUnet-S2
python3 main_s1s2_siamunet_bitemporal.py \
            data.satellites=['S2'] \
            model.ARCH=$CFG \
            model.SHARE_ENCODER=True \
            model.batch_size=16 \
            model.max_epoch=100 \
            experiment.note=allBands

# # SiamUnet-S1
python3 main_s1s2_siamunet_bitemporal.py \
            data.satellites=['S1'] \
            data.INPUT_BANDS.S1=['ND','VH','VV'] \
            model.ARCH=$CFG \
            model.SHARE_ENCODER=True \
            model.batch_size=16 \
            model.max_epoch=5 \
            experiment.note=allBands

# # SiamUnet-S1S2
# python3 main_s1s2_siamunet_multisensor.py \
#             data.satellites=['S1','S2'] \
#             data.stacking=True \
#             model.ARCH=$CFG \
#             model.batch_size=16 \
#             model.max_epoch=5 \
#             experiment.note=test

#rm -rf $SLURM_SUBMIT_DIR/*.log
# rm -rf $SLURM_SUBMIT_DIR/*.out

echo "finish"

## run
# sbatch --array=1-2 geo_run_s1s2_unet.sh
