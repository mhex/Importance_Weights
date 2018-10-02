#

export MKL_NUM_THREADS=8
export MKL_THREADING_LAYER="INTEL"

gpu=0
data_dir=/system/user/bioinf04/IW_Seq2SeqData
train_dir=/system/user/bioinf04/IW_Seq2SeqData/checkpoints
#train_dir=checkpoints
batch_size=100
size=256
num_layers=2
steps_per_checkpoint=1
learning_rate=1.0
learning_rate_decay_factor=0.99
C=10.0
p=1.0
max_gradient_norm=0.1
#max_train_data_size=5000
max_train_data_size=500000
#max_train_data_size=0
#out_file=iw_2x256_C3.0_p3.0.txt
num_threads=8
out_file=tmp5.txt

CUDA_VISIBLE_DEVICES=$gpu \
python3 translateIW.py \
 --data_dir=$data_dir \
 --train_dir=$train_dir \
 --batch_size=$batch_size \
 --size=$size \
 --num_layers=$num_layers \
 --steps_per_checkpoint=$steps_per_checkpoint \
 --learning_rate=$learning_rate \
 --learning_rate_decay_factor=$learning_rate_decay_factor \
 --C=$C \
 --p=$p \
 --max_gradient_norm=$max_gradient_norm \
 --max_train_data_size=$max_train_data_size \
 --num_threads=$num_threads \
 2>&1 | tee $out_file

