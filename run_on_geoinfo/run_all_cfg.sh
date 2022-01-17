
# Test training benchmark for several models.

# Use docker： paddlepaddle/paddle:latest-gpu-cuda10.1-cudnn7  paddle=2.1.2  py=37

# Usage:
#   git clone git clone https://github.com/PaddlePaddle/PaddleSeg.git
#   cd PaddleSeg
#   bash benchmark/run_all.sh

# pip install -r requirements.txt

# # Download test dataset and save it to PaddleSeg/data
# # It automatic downloads the pretrained models saved in ~/.paddleseg
# mkdir -p data
# wget https://paddleseg.bj.bcebos.com/dataset/cityscapes_30imgs.tar.gz \
#     -O data/cityscapes_30imgs.tar.gz
# tar -zxf data/cityscapes_30imgs.tar.gz -C data/

# bash run_on_geoinfo/run_all_cfg.sh
model_name_list=(UNet UNet_resnet18 SegFormer_B2)
# fp_item_list=(fp32)     # set fp32 or fp16, segformer_b0 doesn't support fp16 with Paddle2.1.2
bs_list=(16)
max_iters=100           # control the test time
num_workers=4           # num_workers for dataloader
weight_decay_list=(0.05 0.01)

for model_name in ${model_name_list[@]}; do
      for fp_item in ${fp_item_list[@]}; do
          for bs_item in ${bs_list[@]}; do
            for weight_decay in ${weight_decay_list[@]}; do
                echo "index is speed, 1gpus, begin, ${model_name}"
                run_mode=sp
                CUDA_VISIBLE_DEVICES=0 
                sbatch run_on_geoinfo/run_benchmark.sh ${run_mode} ${bs_item} \
                    ${max_epochs} ${model_name} ${num_workers} ${weight_decay}
                sleep 60

                # echo "index is speed, 8gpus, run_mode is multi_process, begin, ${model_name}"
                # run_mode=mp
                # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash benchmark/run_benchmarksh ${run_mode} ${bs_item} ${fp_item} \
                #     ${max_iters} ${model_name} ${num_workers}
                # sleep 60
                done
            done
      done
done

# rm -rf data/*
# rm -rf ~/.paddleseg
