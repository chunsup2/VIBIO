#!/bin/bash
# cd /home/cl140/research/v-info/mri
# export CUDA_VISIBLE_DEVICES="3"
# source ~/anaconda3/bin/activate base
# cd /home/cl140/research/vib

# data="sks_3.0_0.1_35.0_c2_num_signals_diffusion_n"
#data="ske_3.0_0.05_35.0_c2_num_signals_diffusion_n"
data="ske_3.0_0.05_35.0"
#data="sks_3.0_0.1_25.0"

gpus=0
#cls_type1="CNN"
#cls_type2="VIBCNN"
#depth=4
#ioloss=1.0
#proporation=1.0
#lr=5e-3
#z_dim=128
#kl=0.01

ckpt_dir="/home/chunsup2/PycharmProjects/VIBIO/checkpoints/${data}/171.0/CNNIO/2026-02-28/"
#ckpt_dir="/home/chunsup2/PycharmProjects/VIBIO/checkpoints/${data}/171.0/VIBIO/2026-03-01/"

python test_new_all.py $gpus --data "$data" --ckpt_dir "$ckpt_dir"
#python test_new_ema_all.py $gpus --data "$data" --ckpt_dir "$ckpt_dir"



# mri cnn-IO   proporation 1.0 0.5 0.005
#python train_cls.py 7 --cls_type "$cls_type1" --data "$data" --proporation 0.5 --train_type "$image" --lr 0.005 --depth 3
# mri VIB-CE
#python test_new_ema.py $gpus --cls_type "$cls_type2" --data "$data" --proporation $proporation \
#        --train_type "$image" --lr $lr --depth $depth --z_dim $z_dim --kl $kl \
#        --checkpoint_path "$checkpoint_path"

## mri VIB-IO
#python test_new_ema.py $gpus --cls_type "$cls_type2" --data "$data" --proporation $proporation \
#      --train_type "$image" --lr $lr --depth $depth --z_dim $z_dim --kl $kl --ioloss $ioloss \
#      --checkpoint_path "$checkpoint_path"


