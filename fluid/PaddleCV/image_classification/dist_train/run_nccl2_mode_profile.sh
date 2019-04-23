#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/paddle/zcd_Paddle/build_fast/python
#export PYTHONPATH=/ssd2/zhaochengduo/zcd_Paddle/build/python
#export FLAGS_reader_queue_speed_test_mode=1

export FLAGS_fuse_parameter_memory_size=1310720
export FLAGS_fuse_parameter_groups_size=3

export MODEL="DistResNet"
export PADDLE_TRAINER_ENDPOINTS="127.0.0.1:7160,127.0.0.1:7161"
# PADDLE_TRAINERS_NUM is used only for reader when nccl2 mode
export PADDLE_TRAINERS_NUM="2"

echo "CUDA_VISIBLE_DEVICES: " $CUDA_VISIBLE_DEVICES
echo "FLAGS_enable_parallel_graph: " $FLAGS_enable_parallel_graph
mkdir -p logs

# NOTE: set NCCL_P2P_DISABLE so that can run nccl2 distribute train on one node.

python dist_train_profile.py --num_threads 32  --model $MODEL --update_method local --batch_size 32 # &> logs/tr0.log &

