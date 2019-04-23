#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export MODEL="DistResNet"
export PADDLE_TRAINER_ENDPOINTS="127.0.0.1:7160,127.0.0.1:7161"
# PADDLE_TRAINERS_NUM is used only for reader when nccl2 mode
export PADDLE_TRAINERS_NUM="2"

mkdir -p logs


export PYTHONPATH=/paddle/zcd_Paddle/build_fast/python
export BATCH_SIZE=32

export CUDA_VISIBLE_DEVICES=4
export GLOG_vmodule=build_strategy=2
export FLAGS_sync_nccl_allreduce=1
export EXECUTOR=0  # default executor
python -u dist_train.py --model $MODEL --num_threads=4 --update_method local --use_default_executor=True --batch_size ${BATCH_SIZE} \
   1>./logs/origin_thread4_${MODEL}_gpu_${CUDA_VISIBLE_DEVICES}_use_default_executor_${EXECUTOR}_b_${BATCH_SIZE}.log
export GLOG_vmodule=build_strategy=0
export FLAGS_sync_nccl_allreduce=0

export GLOG_vmodule=build_strategy=2
export FLAGS_sync_nccl_allreduce=1
export EXECUTOR=1  # fast threaded
python -u dist_train.py --model $MODEL  --num_threads=4 --update_method local --use_default_executor=False --batch_size ${BATCH_SIZE} \
   1>./logs/origin_thread4_${MODEL}_gpu_${CUDA_VISIBLE_DEVICES}_use_default_executor_${EXECUTOR}_b_${BATCH_SIZE}.log
export GLOG_vmodule=build_strategy=0
export FLAGS_sync_nccl_allreduce=0

export EXECUTOR=2  # parallel graph
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
python -u dist_train.py --model $MODEL  --num_threads=4 --update_method local  --batch_size ${BATCH_SIZE} \
   1>./logs/origin_thread4_${MODEL}_gpu_${CUDA_VISIBLE_DEVICES}_use_default_executor_${EXECUTOR}_b_${BATCH_SIZE}.log
export FLAGS_enable_parallel_graph=0
export FLAGS_sync_nccl_allreduce=0




exit(-1)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export GLOG_vmodule=build_strategy=2
export FLAGS_sync_nccl_allreduce=1
export EXECUTOR=0  # default executor
python -u dist_train.py --model $MODEL --update_method local --use_default_executor=True --batch_size ${BATCH_SIZE} \
   1>./logs/origin_${MODEL}_gpu_${CUDA_VISIBLE_DEVICES}_use_default_executor_${EXECUTOR}_b_${BATCH_SIZE}.log
export GLOG_vmodule=build_strategy=0
export FLAGS_sync_nccl_allreduce=0

export GLOG_vmodule=build_strategy=2
export FLAGS_sync_nccl_allreduce=1
export EXECUTOR=1  # fast threaded
python -u dist_train.py --model $MODEL --update_method local --use_default_executor=False --batch_size ${BATCH_SIZE} \
   1>./logs/origin_${MODEL}_gpu_${CUDA_VISIBLE_DEVICES}_use_default_executor_${EXECUTOR}_b_${BATCH_SIZE}.log
export GLOG_vmodule=build_strategy=0
export FLAGS_sync_nccl_allreduce=0

export EXECUTOR=2  # parallel graph
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
python -u dist_train.py --model $MODEL --update_method local  --batch_size ${BATCH_SIZE} \
   1>./logs/origin_${MODEL}_gpu_${CUDA_VISIBLE_DEVICES}_use_default_executor_${EXECUTOR}_b_${BATCH_SIZE}.log
export FLAGS_enable_parallel_graph=0
export FLAGS_sync_nccl_allreduce=0

export CUDA_VISIBLE_DEVICES=4,5,6,7

export GLOG_vmodule=build_strategy=2
export FLAGS_sync_nccl_allreduce=1
export EXECUTOR=0  # default executor
python -u dist_train.py --model $MODEL --update_method local --use_default_executor=True --batch_size ${BATCH_SIZE} \
   1>./logs/origin_${MODEL}_gpu_${CUDA_VISIBLE_DEVICES}_use_default_executor_${EXECUTOR}_b_${BATCH_SIZE}.log
export GLOG_vmodule=build_strategy=0
export FLAGS_sync_nccl_allreduce=0

export GLOG_vmodule=build_strategy=2
export FLAGS_sync_nccl_allreduce=1
export EXECUTOR=1  # fast threaded
python -u dist_train.py --model $MODEL --update_method local --use_default_executor=False --batch_size ${BATCH_SIZE} \
   1>./logs/origin_${MODEL}_gpu_${CUDA_VISIBLE_DEVICES}_use_default_executor_${EXECUTOR}_b_${BATCH_SIZE}.log
export GLOG_vmodule=build_strategy=0
export FLAGS_sync_nccl_allreduce=0

export EXECUTOR=2  # parallel graph
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
python -u dist_train.py --model $MODEL --update_method local  --batch_size ${BATCH_SIZE} \
   1>./logs/origin_${MODEL}_gpu_${CUDA_VISIBLE_DEVICES}_use_default_executor_${EXECUTOR}_b_${BATCH_SIZE}.log
export FLAGS_enable_parallel_graph=0
export FLAGS_sync_nccl_allreduce=0

