#
gpu=2
data_path=/system/user/bioinf04/IW_Seq2SeqData/ptb/simple-examples/data/
model=iw
out_file=tmp_3.txt

CUDA_VISIBLE_DEVICES=$gpu \
python3 ptb_word_lm_IW.py \
 --data_path=$data_path \
 --model=$model \
 2>&1 | tee $out_file

