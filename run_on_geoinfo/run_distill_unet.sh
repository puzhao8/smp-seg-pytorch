#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 36GB
#SBATCH --cpus-per-task 8
#SBATCH -t 7-00:00:00
#SBATCH --job-name distill-unet
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
SAT=('ND' 'VH' 'VV')
CFG=${SAT[$SLURM_ARRAY_TASK_ID]}
echo "Running simulation $CFG"
# echo "python3 main_s1s2_unet.py model.batch_size=32"
echo "---------------------------------------------------------------------------------------------------------------"

# singularity exec --nv /cephyr/users/puzhao/Alvis/PyTorch_v1.7.0-py3.sif python main_s1s2_unet.py s1s2_unet=$CFG
conda activate pytorch
PYTHONUNBUFFERED=1; 


# sbatch run_on_geoinfo/run_distill_unet.sh
# python3 main_s1s2_unet.py \
#             --config-name=distill_unet.yaml \
#             data.satellites=['S1'] \
#             model.LOSS_COEF=[0,0] \
#             model.ARCH=distill_unet \
#             model.DISTILL=True \
#             model.batch_size=16 \
#             model.max_epoch=100 \
#             experiment.note=S1_pretrain

# echo "-------------------- distill-unet: PRETRAIN ------------------------"
# python3 main_s1s2_distill_unet.py \
#             --config-name=distill_unet.yaml \
#             data.satellites=['S1','S2'] \
#             data.INPUT_BANDS.S2=['B4','B8','B12']\
#             model.LOSS_COEF=[0,0] \
#             model.ARCH=distill_unet \
#             model.DISTILL=False \
#             model.batch_size=16 \
#             model.max_epoch=100 \
#             experiment.note=S2_pretrain_B4812

# # sbatch run_on_geoinfo/run_distill_unet.sh
# echo "-------------------- distill-unet: DISTILL ------------------------"
python3 main_s1s2_distill_unet.py \
            --config-name=distill_unet.yaml \
            RAND.SEED=0 \
            RAND.DETERMIN=False \
            DATA.SATELLITES=['S1','S2'] \
            DATA.STACKING=True \
            DATA.INPUT_BANDS.S1=['ND','VH','VV'] \
            DATA.INPUT_BANDS.S2=['B4','B8','B12'] \
            MODEL.ARCH=distill_unet \
            MODEL.L2_NORM=False \
            MODEL.USE_DECONV=True \
            MODEL.WEIGHT_DECAY=0.01 \
            MODEL.NUM_CLASS=1 \
            MODEL.LOSS_TYPE=DiceLoss \
            MODEL.LOSS_COEF=[1,0,0] \
            MODEL.LR_SCHEDULER=cosine \
            MODEL.ACTIVATION=sigmoid \
            MODEL.BATCH_SIZE=16 \
            MODEL.MAX_EPOCH=100 \
            EXP.NOTE=1_0_0_Jan15

#rm -rf $SLURM_SUBMIT_DIR/*.log
# rm -rf $SLURM_SUBMIT_DIR/*.out

echo "finish"

## run
# sbatch --array=1-2 geo_run_s1s2_unet.sh
