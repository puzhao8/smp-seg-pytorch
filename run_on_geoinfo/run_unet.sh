#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 36GB
#SBATCH --cpus-per-task 8
#SBATCH -t 7-00:00:00
#SBATCH --job-name sar-unet
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

# # Choose different sensors
# echo "python3 main_s1s2_unet.py model.batch_size=32"

# singularity exec --nv /cephyr/users/puzhao/Alvis/PyTorch_v1.7.0-py3.sif python main_s1s2_unet.py s1s2_unet=$CFG
conda activate pytorch
PYTHONUNBUFFERED=1; 

# python3 main_s1s2_unet.py \
#             data.satellites=['S2'] \
#             data.INPUT_BANDS.S2=['B4','B8','B12']\
#             model.ARCH=UNet \
#             model.batch_size=16 \
#             model.max_epoch=100 \
#             experiment.note=B4812

# sbatch run_on_geoinfo/run_unet.sh
# python3 main_s1s2_unet.py \
#             --config-name=unet \
#             data.satellites=['S1'] \
#             model.ARCH=UNet \
#             model.batch_size=16 \
#             model.max_epoch=5 \
#             experiment.note=test

# python3 main_s1s2_unet.py \
#             --config-name=unet \
#             data.satellites=['S1','S2'] \
#             data.INPUT_BANDS.S2=['B4','B8','B12'] \
#             model.ARCH=UNet \
#             model.batch_size=16 \
#             model.max_epoch=100 \
#             experiment.note=EF-new



# python3 main_s1s2_unet.py \
#             data.satellites=['S1','S2'] \
#             data.INPUT_BANDS.S2=['B4','B8','B12'] \
#             model.ARCH=distill_unet \
#             model.batch_size=16 \
#             model.max_epoch=100 \
#             experiment.note=S1_TEST

# sbatch run_on_geoinfo/run_unet.sh
python3 main_s1s2_unet.py \
            --config-name=unet.yaml \
            RAND.SEED=0 \
            RAND.DETERMIN=False \
            DATA.SATELLITES=['S1'] \
            MODEL.DEBUG=False \
            MODEL.ARCH=UNet \
            MODEL.ENCODER=resnet18 \
            MODEL.ENCODER_WEIGHTS=imagenet \
            MODEL.USE_DECONV=True \
            MODEL.WEIGHT_DECAY=0.01 \
            MODEL.NUM_CLASSES=1 \
            MODEL.LOSS_TYPE=DiceLoss \
            MODEL.LR_SCHEDULER=cosine \
            MODEL.ACTIVATION=sigmoid \
            MODEL.BATCH_SIZE=16 \
            MODEL.MAX_EPOCH=100 \
            EXP.NOTE=EF

# sbatch --array=0-2 run_on_geoinfo/run_unet.sh
# SAT=('ND' 'VH' 'VV')
# CFG=${SAT[$SLURM_ARRAY_TASK_ID]}
# echo "Running simulation $CFG"
# echo "---------------------------------------------------------------------------------------------------------------"

# python3 main_s1s2_unet.py \
#             --config-name=unet.yaml \
#             RAND.SEED=0 \
#             RAND.DETERMIN=False \
#             DATA.SATELLITES=['S1'] \
#             DATA.INPUT_BANDS.S1=[$CFG] \
#             MODEL.DEBUG=False \
#             MODEL.ARCH=UNet \
            # MODEL.ENCODER=resnet18 \
            # MODEL.ENCODER_WEIGHTS=imagenet \
            # MODEL.USE_DECONV=True \
            # MODEL.WEIGHT_DECAY=0.01 \
            # MODEL.NUM_CLASSES=1 \
            # MODEL.LOSS_TYPE=DiceLoss \
            # MODEL.LR_SCHEDULER=cosine \
            # MODEL.ACTIVATION=sigmoid \
            # MODEL.BATCH_SIZE=16 \
            # MODEL.MAX_EPOCH=100 \
            # EXP.NOTE=$CFG

#rm -rf $SLURM_SUBMIT_DIR/*.log
# rm -rf $SLURM_SUBMIT_DIR/*.out

echo "finish"

## run
# sbatch --array=1-2 geo_run_s1s2_unet.sh
