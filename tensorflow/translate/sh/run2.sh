#
data_dir=/publicdata/nlp/wmt15
train_dir=logs
batch_size=64
size=256
num_layers=2
steps_per_checkpoint=200
learning_rate=0.5
learning_rate_decay_factor=0.99
max_gradient_norm=5.0
max_train_data_size=0
out_file=tmp_adam_2048.txt

#CUDA_VISIBLE_DEVICES=$gpu \
python3 translate.py \
 --data_dir=$data_dir \
 --train_dir=$train_dir \
 --batch_size=$batch_size \
 --size=$size \
 --num_layers=$num_layers \
 --steps_per_checkpoint=$steps_per_checkpoint \
 --learning_rate=$learning_rate \
 --learning_rate_decay_factor=$learning_rate_decay_factor \
 --max_gradient_norm=$max_gradient_norm \
 --max_train_data_size=$max_train_data_size # \
 #2>&1 | tee $out_file

