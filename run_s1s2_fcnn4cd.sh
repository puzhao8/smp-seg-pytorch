#!/bin/bash
#SBATCH -A SNIC2021-7-104
#SBATCH -N 1
#SBATCH --gpus-per-node=T4:1
#SBATCH -t 7-00:00:00
#SBATCH --job-name fcnn4cd

echo "start"
echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi

# git clone -b multi-res-label https://github.com/puzhao89/temporal-consistency.git $TMPDIR/temporal-consistency

# cd $TMPDIR/temporal-consistency
# pwd

# rsync -a $SLURM_SUBMIT_DIR/data_for_snic/data $TMPDIR/temporal-consistency/

# ls $TMPDIR/temporal-consistency/data/

# exp_dir=$SLURM_SUBMIT_DIR/tc4wildfire_outputs
# mkdir $exp_dir

# while sleep 20m
# do
#     rsync -a $TMPDIR/temporal-consistency/outputs/* $exp_dir
# done &
# LOOPPID=$!

CDC=("Paddle_unet" "Vanilla_unet" "SiamUnet_conc" "SiamUnet_diff")
CFG=${CDC[$SLURM_ARRAY_TASK_ID]}
echo "Running simulation $CFG"
echo "singularity exec --nv /cephyr/users/puzhao/Alvis/PyTorch_v1.7.0-py3.sif python main_s1s2_fcnn4cd.py model.ARCH=$CFG"
echo "---------------------------------------------------------------------------------------------------------------"

# singularity exec --nv /cephyr/users/puzhao/Alvis/PyTorch_v1.7.0-py3.sif python3 main_s1s2_fcnn4cd.py data.satellites=['S2'] model.ARCH=$CFG model.max_epoch=20
singularity exec --nv /cephyr/users/puzhao/Alvis/PyTorch_v1.7.0-py3.sif python3 puzhao-snic-500G/smp-seg-pytorch/fcnn4cd/unet_paddle.py

# singularity exec --nv /cephyr/users/puzhao/Alvis/PyTorch_v1.7.0-py3.sif python3 s1s2_evaluator.py

# rsync -a $TMPDIR/temporal-consistency/outputs/* $exp_dir
# kill $LOOPPID

rm -rf $SLURM_SUBMIT_DIR/*.log
# rm -rf $SLURM_SUBMIT_DIR/*.out

echo "finish"

# # run
# sbatch --array=0-2 run_s1s2_fcnn4cd.sh