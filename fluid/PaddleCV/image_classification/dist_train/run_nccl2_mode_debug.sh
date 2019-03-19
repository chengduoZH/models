#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/ssd2/zhaochengduo/dev_Paddle/build/python
#export FLAGS_reader_queue_speed_test_mode=1

export GLOG_vmodule=build_strategy=2
#export FLAGS_sync_nccl_allreduce=1

#export FLAGS_enable_parallel_graph=1
#export FLAGS_sync_nccl_allreduce=1

export MODEL="DistResNet"
export PADDLE_TRAINER_ENDPOINTS="127.0.0.1:7160,127.0.0.1:7161"
# PADDLE_TRAINERS_NUM is used only for reader when nccl2 mode
export PADDLE_TRAINERS_NUM="2"

mkdir -p logs

# NOTE: set NCCL_P2P_DISABLE so that can run nccl2 distribute train on one node.

python dist_train.py --model $MODEL --update_method local --batch_size 32 # &> logs/tr0.log &

