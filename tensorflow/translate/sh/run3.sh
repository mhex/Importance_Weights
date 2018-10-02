#
gpu=2
data_dir=/system/user/bioinf04/IW_Seq2SeqData
train_dir=/system/user/bioinf04/IW_Seq2SeqData/checkpoints2
#train_dir=checkpoints
batch_size=100
size=256
num_layers=2
steps_per_checkpoint=1
learning_rate=1.0
learning_rate_decay_factor=1.0
C=3.0
p=1.0
max_gradient_norm=5.0
#max_train_data_size=5000
max_train_data_size=0
#out_file=iw_2x256_C3.0_p3.0.txt
out_file=tmp2.txt

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
 2>&1 | tee -a $out_file

