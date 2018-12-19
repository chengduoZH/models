
export GLOG_v=4
export GLOG_logtostderr=0
export GLOG_log_dir="./"

export CUDA_VISIBLE_DEVICES=3
export DATA_ROOT=/paddle/TransformerData/
#export FLAGS_benchmark=True
#export DATA_ROOT=/home/chengduo/TransformerData/
python -u train.py \
    --src_vocab_fpath $DATA_ROOT/data/wmt_bpe/vocab_all.bpe.32000 \
    --trg_vocab_fpath $DATA_ROOT/data/wmt_bpe/vocab_all.bpe.32000 \
    --special_token '<s>' '<e>' '<unk>' \
    --train_file_pattern $DATA_ROOT/data/wmt_bpe/train.tok.clean.bpe.32000.en-de_40000 \
    --use_token_batch True \
    --batch_size 2048 \
    --sort_type pool \
    --pool_size 10000 \
    --enable_ce True \
    --use_py_reader False \
    weight_sharing False \
    pass_num 20 \
    dropout_seed 10



export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_ROOT=/paddle/TransformerData/
#export DATA_ROOT=/home/chengduo/TransformerData/
python -u train.py \
    --src_vocab_fpath $DATA_ROOT/data/wmt_bpe/vocab_all.bpe.32000 \
    --trg_vocab_fpath $DATA_ROOT/data/wmt_bpe/vocab_all.bpe.32000 \
    --special_token '<s>' '<e>' '<unk>' \
    --train_file_pattern $DATA_ROOT/data/wmt_bpe/train.tok.clean.bpe.32000.en-de_40000 \
    --use_token_batch True \
    --batch_size 2048 \
    --sort_type pool \
    --pool_size 10000 \
    --enable_ce True \
    weight_sharing False \
    pass_num 20 \
    dropout_seed 10

