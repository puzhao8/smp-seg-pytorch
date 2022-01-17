#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 36GB
#SBATCH --cpus-per-task 8
#SBATCH -t 7-00:00:00
#SBATCH --job-name segformer
#SBATCH --output /home/p/u/puzhao/smp-seg-pytorch/run_logs/%x-%A_%a.out

# sbatch run_on_geoinfo/run_segformer.sh
python3 main_s1s2_unet.py \
            --config-name=segformer.yaml \
            data.satellites=['S1'] \
            data.INPUT_BANDS.S2=['B4','B8','B12'] \
            model.ARCH=SegFormer_B0 \
            model.LOSS_TYPE=BCEWithLogitsLoss \
            model.ACTIVATION=argmax2d \
            model.DEBUG=False \
            model.batch_size=16 \
            model.max_epoch=100 \
            experiment.note=BS16-BCE