
export CUDA_VISIBLE_DEVICES=3
export DATA_ROOT=/paddle/TransformerData/
#export DATA_ROOT=/home/chengduo/TransformerData/
python  train_tmp.py \
      --src_vocab_fpath $DATA_ROOT/data/wmt_bpe/vocab_all.bpe.32000 \
      --trg_vocab_fpath $DATA_ROOT/data/wmt_bpe/vocab_all.bpe.32000 \
      --special_token '<s>' '<e>' '<unk>' \
      --train_file_pattern $DATA_ROOT/data/wmt_bpe/train.tok.clean.bpe.32000.en-de_40000 \
      --token_delimiter ' ' \
      --use_token_batch True \
      --batch_size 2048 \
      --sort_type pool \
      --pool_size 200000 \
      --enable_ce True
