#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 36GB
#SBATCH --cpus-per-task 8
#SBATCH -t 7-00:00:00
#SBATCH --job-name pdunet
#SBATCH --output=slurm-%x-%A_%a.out

echo "start"
echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
echo
nvidia-smi
. /geoinfo_vol1/puzhao/miniforge3/etc/profile.d/conda.sh

conda activate paddle2
PYTHONUNBUFFERED=1; 

#!/usr/bin/env bash
# set -xe

# Test training benchmark for a model.

# Usage：CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh ${run_mode} ${bs_item} ${fp_item} ${max_iter} ${model_name} ${num_workers}

function _set_params(){
    run_mode=${1:-"sp"}         # sp or mp
    batch_size=${2:-"16"}
    # fp_item=${3:-"fp32"}        # fp32 or fp16
    satellites=${3:-"S1"}
    max_epochs=${4:-"100"}
    model_name=${5:-"UNet"}
    num_workers=${6:-"4"}
    weight_decay=${7:-"0.01"}
    run_log_path=${TRAIN_LOG_DIR:-$(pwd)}

    device=${CUDA_VISIBLE_DEVICES//,/ }
    arr=(${device})
    num_gpu_devices=${#arr[*]}
    log_file=${run_log_path}/${model_name}_${satellites}_${run_mode}_bs${batch_size}_${num_gpu_devices}
}

function _train(){
    echo "Train on ${num_gpu_devices} GPUs"
    echo "current CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, gpus=$num_gpu_devices, batch_size=$batch_size"

    # train_cmd="--config=configs/${model_name}.yml \
    #            --batch_size=${batch_size} \
    #            --iters=${max_epochs} \
    #            --num_workers=${num_workers}"

    # python3 main_s1s2_unet.py \
    train_cmd="--config-name=unet.yaml \
            RAND.SEED=0 \
            RAND.DETERMIN=True \
            DATA.SATELLITES=[$satellites] \
            DATA.INPUT_BANDS.S2=['B4','B8','B12'] \
            MODEL.ARCH=$model_name \
            MODEL.WEIGHT_DECAY=$weight_decay \
            MODEL.NUM_CLASSES=1 \
            MODEL.LOSS_TYPE=DiceLoss \
            MODEL.LR_SCHEDULER=cosine \
            MODEL.ACTIVATION=sigmoid \
            MODEL.BATCH_SIZE=${batch_size} \
            MODEL.MAX_EPOCH=${max_epochs}"

    case ${run_mode} in
    sp) train_cmd="python -u train.py ${train_cmd}" ;;
    mp)
        train_cmd="python -m torch.distributed.launch --log_dir=./mylog --gpus=$CUDA_VISIBLE_DEVICES \
                  train.py ${train_cmd}" ;;
    *) echo "choose run_mode(sp or mp)"; exit 1;
    esac

    timeout 15m ${train_cmd} > ${log_file} 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model_name}, FAIL"
        export job_fail_flag=1
    else
        echo -e "${model_name}, SUCCESS"
        export job_fail_flag=0
    fi
    kill -9 `ps -ef|grep 'python'|awk '{print $2}'`

    if [ $run_mode = "mp" -a -d mylog ]; then
        rm ${log_file}
        cp mylog/workerlog.0 ${log_file}
    fi
}

_set_params $@
_train
