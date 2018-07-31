

python  train_pe.py \
     --src_vocab_fpath test_data/vocab.bpe.32000 \
     --trg_vocab_fpath test_data/vocab.bpe.32000 \
     --special_token '<s>' '<e>' '<unk>' \
     --train_file_pattern test_data/train.tok.clean.bpe.32000.en-de.train_1 \
     --val_file_pattern test_data/newstest2013.tok.bpe.32000.en-de \
     --use_token_batch True \
     --batch_size 20 \
     --sort_type none \
     --pool_size 2000 \
     --shuffle False \
     --shuffle_batch False \
     --use_token_batch False \
     --token_delimiter  ' '
