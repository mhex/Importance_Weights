#

export MKL_NUM_THREADS=8
export MKL_THREADING_LAYER="INTEL"

#gpu=1
data_path=/publicdata/nlp/ptb/simple-examples/data/
model=medium
save_dir="logs"
out_file=tmp.txt

dwt=`date "+%m%d_%H%M%S"`
run_id=${dwt}_6w_medium

echo $run_id

#CUDA_VISIBLE_DEVICES=$gpu \
python3 ptb_word_lm.py \
 --data_path=$data_path \
 --model=$model \
 --save_path=$save_dir/${run_id}
# 2>&1 | tee $out_file

