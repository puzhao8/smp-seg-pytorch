#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 36GB
#SBATCH --cpus-per-task 1
#SBATCH -t 7-00:00:00
#SBATCH --job-name train
#SBATCH --output /home/p/u/puzhao/run_logs/%x-%A_%a.out

echo "start"
echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo
nvidia-smi
. /geoinfo_vol1/puzhao/miniforge3/etc/profile.d/conda.sh

conda activate pytorch
PYTHONUNBUFFERED=1; 

# module --ignore-cache load "intel"
# PROJECT_DIR=/cephyr/NOBACKUP/groups/snic2021-7-104/puzhao-snic-500G/smp-seg-pytorch

# # Choose different config files
# # DIRS=($(find /cephyr/NOBACKUP/groups/snic2021-7-104/puzhao-snic-500G/smp-seg-pytorch/config/s1s2_cfg/))
# DIRS=($(find $PROJECT_DIR/config/s1s2_unet/))
# DIRS=${DIRS[@]:1}
# CFG=${DIRS[$SLURM_ARRAY_TASK_ID]}
# echo "Running simulation $CFG"


## sbatch --array=0-2 run_on_geoinfo/run_unet.sh
# SAT=('S1' 'S2', 'ALOS')
# CFG=${SAT[$SLURM_ARRAY_TASK_ID]}
# echo "Running simulation $CFG"
# echo "---------------------------------------------------------------------------------------------------------------"


##########################################################
## ---- U-Net MTBS ----
##########################################################

# sbatch --array=0-4 run_on_geoinfo/run_mtbs.sh
CFG=$SLURM_ARRAY_TASK_ID
echo "Running simulation $CFG"
echo "---------------------------------------------------------------------------------------------------------------"

python3 main_s1s2_unet_mtbs.py \
            --config-name=mtbs.yaml \
            RAND.SEED=0 \
            RAND.DETERMIN=False \
            DATA.AUGMENT=True \
            DATA.TRAIN_MASK=mtbs \
            DATA.TEST_MASK=mtbs \
            DATA.SATELLITES=['S2'] \
            DATA.PREPOST=['pre','post'] \
            DATA.STACKING=True \
            DATA.INPUT_BANDS.S2=['B4','B8','B12'] \
            MODEL.ARCH=UNet_dualHeads \
            MODEL.CLASS_WEIGHTS=[0.5,0.5,0,0] \
            MODEL.USE_DECONV=False \
            MODEL.WEIGHT_DECAY=0.01 \
            MODEL.LR_SCHEDULER=cosine \
            MODEL.BATCH_SIZE=32 \
            MODEL.MAX_EPOCH=5 \
            MODEL.STEP_WISE_LOG=False \
            EXP.NOTE=debug

##########################################################
## ---- U-Net MTBS 2 Classes ----
##########################################################
# sbatch run_on_geoinfo/run_unet.sh
# python3 main_s1s2_unet_mtbs.py \
#             --config-name=mtbs.yaml \
#             RAND.SEED=0 \
#             RAND.DETERMIN=False \
#             DATA.TRAIN_MASK=poly \
#             DATA.TRAIN_MASK=poly \
#             DATA.SATELLITES=['S2'] \
#             DATA.PREPOST=['pre','post'] \
#             DATA.STACKING=True \
#             DATA.INPUT_BANDS.S2=['B4','B8','B12'] \
#             MODEL.ARCH=UNet \
#             MODEL.NUM_CLASS=2 \
#             MODEL.CLASS_NAMES=['unburn','low'] \
#             MODEL.CLASS_WEIGHTS=[0.5,0.5] \
#             MODEL.USE_DECONV=True \
#             MODEL.WEIGHT_DECAY=0.01 \
#             MODEL.LR_SCHEDULER=cosine \
#             MODEL.BATCH_SIZE=16 \
#             MODEL.MAX_EPOCH=100 \
#             EXP.NOTE=poly-ep100
            




##########################################################
## ---- U-Net with Data Augmentation ----
##########################################################
# sbatch run_on_geoinfo/run_unet.sh
# python3 main_s1s2_unet_aug.py \
#             --config-name=unet.yaml \
#             RAND.SEED=0 \
#             RAND.DETERMIN=False \
#             DATA.SATELLITES=['ALOS'] \
#             DATA.TRAIN_MASK=poly \
#             DATA.AUGMENT=False \
#             MODEL.ARCH=UNet \
#             MODEL.ENCODER=resnet18 \
#             MODEL.ENCODER_WEIGHTS=imagenet \
#             MODEL.USE_DECONV=False \
#             MODEL.LEARNING_RATE=0.0001 \
#             MODEL.WEIGHT_DECAY=0.01 \
#             MODEL.NUM_CLASS=1 \
#             MODEL.LOSS_TYPE=DiceLoss \
#             MODEL.LR_SCHEDULER=cosine \
#             MODEL.ACTIVATION=sigmoid \
#             MODEL.BATCH_SIZE=16 \
#             MODEL.MAX_EPOCH=100 \
#             EXP.NOTE=EF-aug


##########################################################
## ---- Weakly Supervised Learning ----
##########################################################
# sbatch run_on_geoinfo/run_unet.sh
# python3 main_s1s2_unet_wsl.py \
#             --config-name=unet.yaml \
#             RAND.SEED=0 \
#             RAND.DETERMIN=False \
#             DATA.SATELLITES=['ALOS'] \
#             DATA.INPUT_BANDS.S2=['B8','B11','B12'] \
#             DATA.TRAIN_MASK=modis \
#             DATA.AUGMENT=True \
#             MODEL.ARCH=UNet \
#             MODEL.ENCODER=resnet18 \
#             MODEL.ENCODER_WEIGHTS=imagenet \
#             MODEL.USE_DECONV=False \
#             MODEL.LEARNING_RATE=0.0001 \
#             MODEL.WEIGHT_DECAY=0.01 \
#             MODEL.NUM_CLASS=1 \
#             MODEL.LOSS_TYPE=DiceLoss \
#             MODEL.LR_SCHEDULER=cosine \
#             MODEL.ACTIVATION=sigmoid \
#             MODEL.BATCH_SIZE=16 \
#             MODEL.MAX_EPOCH=100 \
#             EXP.NOTE=EF-aug


##########################################################
## ---- Early Fusion for Single Bands for S1 and ALOS ----
##########################################################          
# # sbatch --array=0-2 run_on_geoinfo/run_unet.sh
# SAT=('ND' 'VH' 'VV')
# CFG=${SAT[$SLURM_ARRAY_TASK_ID]}
# echo "Running simulation $CFG"
# echo "---------------------------------------------------------------------------------------------------------------"

# python3 main_s1s2_unet.py \
#             --config-name=unet.yaml \
#             RAND.SEED=0 \
#             RAND.DETERMIN=False \
#             DATA.TRAIN_MASK=poly \
#             DATA.SATELLITES=['ALOS'] \
#             DATA.INPUT_BANDS.ALOS=[$CFG] \
#             MODEL.ARCH=UNet \
#             MODEL.ENCODER=resnet18 \
#             MODEL.ENCODER_WEIGHTS=imagenet \
#             MODEL.USE_DECONV=True \
#             MODEL.WEIGHT_DECAY=0.01 \
#             MODEL.NUM_CLASS=1 \
#             MODEL.LOSS_TYPE=DiceLoss \
#             MODEL.LR_SCHEDULER=cosine \
#             MODEL.ACTIVATION=sigmoid \
#             MODEL.BATCH_SIZE=16 \
#             MODEL.MAX_EPOCH=100 \
#             EXP.NOTE=$CFG



##########################################################
## ---- Early Fusion of Different Sensors ----
##########################################################
## sbatch --array=0-2 run_on_geoinfo/run_unet.sh
# SAT=('S1' 'S2')
# CFG=${SAT[$SLURM_ARRAY_TASK_ID]}
# echo "Running simulation $CFG"
# echo "---------------------------------------------------------------------------------------------------------------"

# python3 main_s1s2_unet.py \
#             --config-name=unet.yaml \
#             RAND.SEED=0 \
#             RAND.DETERMIN=False \
#             DATA.TRAIN_MASK=poly \
#             DATA.SATELLITES=[$CFG,'ALOS'] \
#             DATA.PREPOST=['pre','post'] \
#             DATA.STACKING=True \
#             DATA.INPUT_BANDS.S2=['B4','B8','B12'] \
#             MODEL.ARCH=UNet \
#             MODEL.ENCODER=resnet18 \
#             MODEL.ENCODER_WEIGHTS=imagenet \
#             MODEL.USE_DECONV=True \
#             MODEL.WEIGHT_DECAY=0.01 \
#             MODEL.NUM_CLASS=1 \
#             MODEL.LOSS_TYPE=DiceLoss \
#             MODEL.LR_SCHEDULER=cosine \
#             MODEL.ACTIVATION=sigmoid \
#             MODEL.BATCH_SIZE=16 \
#             MODEL.MAX_EPOCH=100 \
#             EXP.NOTE=EF



##########################################################
## --- Use Post-Fire Data alone for single sensor ---
##########################################################
## sbatch --array=0-2 run_on_geoinfo/run_unet.sh
# SAT=('S1' 'S2' 'ALOS')
# CFG=${SAT[$SLURM_ARRAY_TASK_ID]}
# echo "Running simulation $CFG"
# echo "---------------------------------------------------------------------------------------------------------------"

# python3 main_s1s2_unet.py \
#             --config-name=unet.yaml \
#             RAND.SEED=0 \
#             RAND.DETERMIN=False \
#             DATA.TRAIN_MASK=poly \
#             DATA.SATELLITES=[$CFG] \
#             DATA.PREPOST=['post'] \
#             DATA.STACKING=False \
#             DATA.INPUT_BANDS.S2=['B4','B8','B12'] \
#             MODEL.ARCH=UNet \
#             MODEL.ENCODER=resnet18 \
#             MODEL.ENCODER_WEIGHTS=imagenet \
#             MODEL.USE_DECONV=True \
#             MODEL.WEIGHT_DECAY=0.01 \
#             MODEL.NUM_CLASS=1 \
#             MODEL.LOSS_TYPE=DiceLoss \
#             MODEL.LR_SCHEDULER=cosine \
#             MODEL.ACTIVATION=sigmoid \
#             MODEL.BATCH_SIZE=16 \
#             MODEL.MAX_EPOCH=100 \
#             EXP.NOTE=post

#rm -rf $SLURM_SUBMIT_DIR/*.log
# rm -rf $SLURM_SUBMIT_DIR/*.out

echo "finish"

## run
# sbatch --array=1-2 geo_run_s1s2_unet.sh
