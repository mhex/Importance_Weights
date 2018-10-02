#
gpu=1
train_dir=/system/user/bioinf04/IW_Seq2SeqData/checkpoints2
#train_dir=checkpoints
task="mul"
batch_size=32
max_length=41
steps_per_checkpoint=200
lr=0.001
max_grad_norm=1.0
#max_train_data_size=5000
train_data_size=5000
#out_file=iw_2x256_C3.0_p3.0.txt
out_file=sgd_mul_gd.txt


CUDA_VISIBLE_DEVICES=$gpu \
python3 neural_gpu_trainer.py \
 --task=$task \
 --train_dir=$train_dir \
 --batch_size=$batch_size \
 --max_length=$max_length \
 --steps_per_checkpoint=$steps_per_checkpoint \
 --lr=$lr \
 --max_grad_norm=$max_grad_norm \
 --train_data_size=$train_data_size \
 2>&1 | tee $out_file

