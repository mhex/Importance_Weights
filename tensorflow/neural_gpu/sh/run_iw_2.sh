#
gpu=1
train_dir=/system/user/bioinf04/IW_Seq2SeqData/checkpoints2
#train_dir=checkpoints
task="mul"
batch_size=5
low_batch_size=2
iw_batches=20
pull_incr=1.2
max_length=41
steps_per_checkpoint=1000
max_grad_norm=2.0
lr=0.001
C=2.0
p=2.0
train_data_size=5000
#out_file=iw_2x256_C3.0_p3.0.txt
out_file=tmp_iw_2.txt


CUDA_VISIBLE_DEVICES=$gpu \
python3 neural_gpu_trainer_iw.py \
 --task=$task \
 --train_dir=$train_dir \
 --batch_size=$batch_size \
 --low_batch_size=$low_batch_size \
 --pull_incr=$pull_incr \
 --max_length=$max_length \
 --steps_per_checkpoint=$steps_per_checkpoint \
 --lr=$lr \
 --p=$p \
 --C=$C \
 --iw_batches=$iw_batches \
 --max_grad_norm=$max_grad_norm \
 --train_data_size=$train_data_size \
 2>&1 | tee $out_file

