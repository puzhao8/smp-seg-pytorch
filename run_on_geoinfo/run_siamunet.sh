#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 36GB
#SBATCH --cpus-per-task 8
#SBATCH -t 7-00:00:00
#SBATCH --job-name dual-unet
#SBATCH --output /home/p/u/puzhao/run_logs/%x-%A_%a.out

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

# # # Choose different sensors
# SAT=('DualUnet_LF' 'SiamUnet_conc' 'SiamUnet_diff')
# CFG=${SAT[$SLURM_ARRAY_TASK_ID]}
# echo "Running simulation $CFG"
# # echo "python3 main_s1s2_unet.py model.batch_size=32"
# echo "---------------------------------------------------------------------------------------------------------------"

# singularity exec --nv /cephyr/users/puzhao/Alvis/PyTorch_v1.7.0-py3.sif python main_s1s2_unet.py s1s2_unet=$CFG
conda activate pytorch
PYTHONUNBUFFERED=1; 

##########################################################
## ---- Same Siam-UNet for Single-Sensor Data ----
##########################################################
# sbatch --array=0-2 run_on_geoinfo/run_siamunet.sh

# # Choose different sensors
SAT=('S1' 'S2' 'ALOS')
CFG=${SAT[$SLURM_ARRAY_TASK_ID]}
echo "Running simulation $CFG"
# echo "python3 main_s1s2_unet.py model.batch_size=32"
echo "---------------------------------------------------------------------------------------------------------------"

python3 main_s1s2_unet.py \
            --config-name=siam_unet.yaml \
            RAND.SEED=0 \
            RAND.DETERMIN=False \
            DATA.TRAIN_MASK=poly \
            DATA.SATELLITES=[$CFG] \
            DATA.STACKING=False \
            DATA.INPUT_BANDS.S1=['ND','VH','VV'] \
            DATA.INPUT_BANDS.S2=['B4','B8','B12'] \
            MODEL.ARCH='DualUnet_LF' \
            MODEL.SHARE_ENCODER=True \
            MODEL.ENCODER=resnet18 \
            MODEL.ENCODER_WEIGHTS=imagenet \
            MODEL.USE_DECONV=False \
            MODEL.WEIGHT_DECAY=0.01 \
            MODEL.NUM_CLASSES=1 \
            MODEL.LOSS_TYPE=DiceLoss \
            MODEL.LR_SCHEDULER=cosine \
            MODEL.ACTIVATION=sigmoid \
            MODEL.BATCH_SIZE=16 \
            MODEL.MAX_EPOCH=100 \
            EXP.NOTE=LF

##########################################################
## ---- Different Architecture for Multi-Sensor Data ----
##########################################################
# sbatch --array=0-1 run_on_geoinfo/run_siamunet.sh

# # Choose different sensors
# SAT=('DualUnet_LF' 'SiamUnet_conc' 'SiamUnet_diff')
# CFG=${SAT[$SLURM_ARRAY_TASK_ID]}
# echo "Running simulation $CFG"
# # echo "python3 main_s1s2_unet.py model.batch_size=32"
# echo "---------------------------------------------------------------------------------------------------------------"

# python3 main_s1s2_unet.py \
#             --config-name=siam_unet.yaml \
#             RAND.SEED=0 \
#             RAND.DETERMIN=False \
#             DATA.TRAIN_MASK=poly \
#             DATA.SATELLITES=['S1','S2'] \
#             DATA.STACKING=True \
#             DATA.INPUT_BANDS.S1=['ND','VH','VV'] \
#             DATA.INPUT_BANDS.S2=['B4','B8','B12'] \
#             MODEL.ARCH=$CFG \
#             MODEL.SHARE_ENCODER=True \
#             MODEL.ENCODER=resnet18 \
#             MODEL.ENCODER_WEIGHTS=imagenet \
#             MODEL.USE_DECONV=False \
#             MODEL.WEIGHT_DECAY=0.01 \
#             MODEL.NUM_CLASSES=1 \
#             MODEL.LOSS_TYPE=DiceLoss \
#             MODEL.LR_SCHEDULER=cosine \
#             MODEL.ACTIVATION=sigmoid \
#             MODEL.BATCH_SIZE=16 \
#             MODEL.MAX_EPOCH=100 \
#             EXP.NOTE=Feb26


# python3 main_s1s2_unet.py \
#             --config-name=siam_unet.yaml \
#             RAND.SEED=0 \
#             RAND.DETERMIN=False \
#             DATA.SATELLITES=['S1','S2'] \
#             DATA.STACKING=True \
#             DATA.INPUT_BANDS.S1=['ND','VH','VV'] \
#             DATA.INPUT_BANDS.S2=['B4','B8','B12'] \
#             MODEL.ARCH=$CFG \
#             MODEL.SHARE_ENCODER=True \
#             MODEL.ENCODER=resnet18 \
#             MODEL.ENCODER_WEIGHTS=imagenet \
#             MODEL.USE_DECONV=False \
#             MODEL.WEIGHT_DECAY=0.01 \
#             MODEL.NUM_CLASSES=1 \
#             MODEL.LOSS_TYPE=DiceLoss \
#             MODEL.LR_SCHEDULER=cosine \
#             MODEL.ACTIVATION=sigmoid \
#             MODEL.BATCH_SIZE=16 \
#             MODEL.MAX_EPOCH=100 \
#             EXP.NOTE=Jan15



#rm -rf $SLURM_SUBMIT_DIR/*.log
# rm -rf $SLURM_SUBMIT_DIR/*.out

echo "finish"

## run
# sbatch --array=1-2 geo_run_s1s2_unet.sh
