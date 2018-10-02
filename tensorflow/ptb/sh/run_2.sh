#
gpu=0
data_path=/system/user/bioinf04/IW_Seq2SeqData/ptb/simple-examples/data/
model=large
out_file=tmp_iw_mn0.01.txt

CUDA_VISIBLE_DEVICES=$gpu \
python3 ptb_word_lm.py \
 --data_path=$data_path \
 --model=$model \
 2>&1 | tee $out_file

