#!/bin/bash
# cd /home/cl140/research/v-info/mri
# export CUDA_VISIBLE_DEVICES="3"
# source ~/anaconda3/bin/activate base
# cd /home/cl140/research/vib

# data="sks_3.0_0.1_35.0_c2_num_signals_diffusion_n"
#data=ske_3.0_0.05_35.0_c2_num_signals_diffusion_n

data=sks_3.0_0.2_25.0_c2_num_signals_diffusion_n
image="measure"
cls_type1="CNN"
cls_type2="VIBCNN"
depth=2

ioloss=1.0
proporation=1,0
patience=20
batch_size=256
val_interval=50
lr=1e-3  # 1e-4~1e-5

gpus=0
z_dim=8
kl=1e-5

python train_cls_new.py $gpus --cls_type "$cls_type1" --data "$data" --proporation $proporation --train_type "$image" --lr $lr \
       --depth $depth --z_dim $z_dim --kl $kl --patience $patience --batch_size $batch_size --val_interval $val_interval

#python train_cls_gaussianIO_EMA_debug.py $gpus --cls_type "$cls_type2" --data "$data" --proporation $proporation --train_type "$image" --lr $lr \
#       --depth $depth --z_dim $z_dim --kl $kl --ioloss $ioloss --patience $patience --batch_size $batch_size --val_interval $val_interval

#python train_cls_debug.py $gpus --cls_type "$cls_type2" --data "$data" --proporation $proporation --train_type "$image" --lr $lr \
#       --depth $depth --z_dim $z_dim --kl $kl

# mri cnn-IO   proporation 1.0 0.5 0.005
#python train_cls.py 7 --cls_type "$cls_type1" --data "$data" --proporation 0.5 --train_type "$image" --lr 0.005 --depth 3
# mri VIB-CE
#python train_cls.py $gpus --cls_type "$cls_type2" --data "$data" --proporation $proporation --train_type "$image" --lr $lr \
#       --depth $depth --z_dim $z_dim --kl $kl
## mri VIB-IO
#python train_cls_gaussianIO_EMA.py $gpus --cls_type "$cls_type2" --data "$data" --proporation $proporation --train_type "$image" --lr $lr \
#       --depth $depth --z_dim $z_dim --kl $kl --ioloss $ioloss --patience $patience --batch_size $batch_size --val_interval $val_interval
## lumpy cnn-IO
#python train_cls_lumpy.py 7 --cls_type "$cls_type1" --data "$data" --proporation 0.5 --train_type "$image" --lr 0.005 --depth 3
## lumpy VIB-CE
#python train_cls_lumpy.py 7 --cls_type "$cls_type2" --data "$data" --proporation 0.5 --train_type "$image" --lr 0.005 --depth 3 --z_dim 2 --kl 10
## lumpy VIB-IO
#python train_cls_lumpy_gaussianIO_EMA.py 7 --cls_type "$cls_type2" --data "$data" --proporation 0.5 --train_type "$image" --lr 0.005 --depth 3 --z_dim 2 --kl 10 --ioloss 1.0



#python train_cls_gaussianIO_EMA.py 7 --cls_type "$cls_type1" --data "$data" --proporation 0.5 --train_type "$image" --lr 0.005 --depth 3 --z_dim 2 --kl 10 --ioloss 1.0
#python train_cls_gaussianIO_EMA.py 7 --cls_type "$cls_type1" --data "$data" --proporation 0.5 --train_type "$image" --lr 0.005 --depth 3 --z_dim 2 --kl 1 --ioloss 1.0
#python train_cls_gaussianIO_EMA.py 7 --cls_type "$cls_type1" --data "$data" --proporation 0.5 --train_type "$image" --lr 0.005 --depth 3 --z_dim 2 --kl 0.1 --ioloss 1.0
#python train_cls_gaussianIO_EMA.py 7 --cls_type "$cls_type1" --data "$data" --proporation 0.5 --train_type "$image" --lr 0.005 --depth 3 --z_dim 2 --kl 0.01 --ioloss 1.0
#python train_cls_gaussianIO_EMA.py 7 --cls_type "$cls_type1" --data "$data" --proporation 0.5 --train_type "$image" --lr 0.0005 --depth 3 --z_dim 4 --kl 0.00000001 --ioloss 1.0


# python train_cls_gaussianIO_EMA.py 2 --cls_type "$cls_type1" --data "$data" --proporation 1.0 --train_type "$image" --lr 0.00005 --depth 3 --z_dim 2 --kl 10 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 2 --cls_type "$cls_type1" --data "$data" --proporation 1.0 --train_type "$image" --lr 0.00005 --depth 3 --z_dim 2 --kl 1 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 2 --cls_type "$cls_type1" --data "$data" --proporation 1.0 --train_type "$image" --lr 0.00005 --depth 3 --z_dim 2 --kl 0.0001 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 2 --cls_type "$cls_type1" --data "$data" --proporation 1.0 --train_type "$image" --lr 0.00005 --depth 3 --z_dim 2 --kl 0.00001 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 2 --cls_type "$cls_type1" --data "$data" --proporation 0.5 --train_type "$image" --lr 0.00005 --depth 3 --z_dim 2 --kl 10 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 2 --cls_type "$cls_type1" --data "$data" --proporation 0.5 --train_type "$image" --lr 0.00005 --depth 3 --z_dim 2 --kl 1 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 2 --cls_type "$cls_type1" --data "$data" --proporation 0.5 --train_type "$image" --lr 0.00005 --depth 3 --z_dim 2 --kl 0.0001 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 2 --cls_type "$cls_type1" --data "$data" --proporation 0.5 --train_type "$image" --lr 0.00005 --depth 3 --z_dim 2 --kl 0.00001 --ioloss 1.0


# python train_cls_gaussianIO_EMA.py 2 --cls_type "$cls_type1" --data "$data" --proporation 1.0 --train_type "$image" --lr 0.00005 --depth 3 --z_dim 4 --kl 10 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 2 --cls_type "$cls_type1" --data "$data" --proporation 1.0 --train_type "$image" --lr 0.00005 --depth 3 --z_dim 4 --kl 1 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 2 --cls_type "$cls_type1" --data "$data" --proporation 1.0 --train_type "$image" --lr 0.00005 --depth 3 --z_dim 4 --kl 0.0001 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 2 --cls_type "$cls_type1" --data "$data" --proporation 1.0 --train_type "$image" --lr 0.00005 --depth 3 --z_dim 4 --kl 0.00001 --ioloss 1.0



# python train_cls_gaussianIO_EMA.py 2 --cls_type "$cls_type1" --data "$data" --proporation 1.0 --train_type "$image" --lr 0.005 --depth 3 --z_dim 128 --kl 50 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 2 --cls_type "$cls_type1" --data "$data" --proporation 1.0 --train_type "$image" --lr 0.005 --depth 3 --z_dim 128 --kl 100 --ioloss 1.0




# python train_cls_gaussianIO_EMA.py 0 --cls_type "$cls_type1" --data "$data" --proporation 0.05 --train_type "$image" --lr 0.00001 --depth 3 --z_dim 8 --kl 0.1 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 0 --cls_type "$cls_type1" --data "$data" --proporation 0.05 --train_type "$image" --lr 0.00001 --depth 3 --z_dim 8 --kl 0.5 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 0 --cls_type "$cls_type1" --data "$data" --proporation 0.05 --train_type "$image" --lr 0.00001 --depth 3 --z_dim 8 --kl 0.01 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 0 --cls_type "$cls_type1" --data "$data" --proporation 0.05 --train_type "$image" --lr 0.00001 --depth 3 --z_dim 8 --kl 0.05 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 0 --cls_type "$cls_type1" --data "$data" --proporation 0.05 --train_type "$image" --lr 0.00001 --depth 3 --z_dim 8 --kl 0.001 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 0 --cls_type "$cls_type1" --data "$data" --proporation 0.05 --train_type "$image" --lr 0.00001 --depth 3 --z_dim 8 --kl 0.005 --ioloss 1.0

# python train_cls_gaussianIO_EMA.py 0 --cls_type "$cls_type1" --data "$data" --proporation 0.05 --train_type "$image" --lr 0.00001 --depth 3 --z_dim 10 --kl 0.1 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 0 --cls_type "$cls_type1" --data "$data" --proporation 0.05 --train_type "$image" --lr 0.00001 --depth 3 --z_dim 10 --kl 0.5 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 0 --cls_type "$cls_type1" --data "$data" --proporation 0.05 --train_type "$image" --lr 0.00001 --depth 3 --z_dim 10 --kl 0.01 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 0 --cls_type "$cls_type1" --data "$data" --proporation 0.05 --train_type "$image" --lr 0.00001 --depth 3 --z_dim 10 --kl 0.05 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 0 --cls_type "$cls_type1" --data "$data" --proporation 0.05 --train_type "$image" --lr 0.00001 --depth 3 --z_dim 10 --kl 0.001 --ioloss 1.0
# python train_cls_gaussianIO_EMA.py 0 --cls_type "$cls_type1" --data "$data" --proporation 0.05 --train_type "$image" --lr 0.00001 --depth 3 --z_dim 10 --kl 0.005 --ioloss 1.0



