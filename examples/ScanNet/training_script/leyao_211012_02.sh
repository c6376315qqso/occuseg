#!/bin/bash

#block(name=occuseg_training_0, threads=10, memory=100000, subtasks=1, gpu=true, hours=200)
   source activate p1
   TASK_NAME=leyao_211012_02
   python -u train_instance.py  \
   --taskname $TASK_NAME \
   --batch_size 2\
   --loss cross_entropy \
   --optim Adam \
   --lr 1e-3 \
   --regress_sigma 0.2 \
   --regress_weight 10 \
   --displacement_weight 1 \
   --gamma 1e-2 \
   --step_size 200 \
   --checkpoints_dir ./ckpts/$TASK_NAME/ \
   --checkpoint 0 \
   --snapshot 10 \
   --m 32 \
   --block_reps 1 \
   --scale 50 \
   --residual_blocks \
   --kernel_size 3 \
   --use_rotation_noise \
   --val_reps 1 \
   --use_feature c \
   --use_dense_model \
   --use_elastic \
   --model_type uncertain \
   --gpu 0 \
   --dataset scenenn \
   --pretrain ./ckpts/leyao_211012_01/Epoch250.pth \
   --mask_name m33_66.pth \
   --uncertain_weight 20 \
   --uncertain_task_weight 0.2 \
   --consistency_weight 1.0 \
   --uncertain_st_epoch 0 \
   --bceloss weighted_bce \
   --restore
#    --simple_train
#   --checkpoint_file ckpts/lhanaf_instance_s50_val_rep1_withElastic/Epoch250.pth
#    --all_to_train
#   --checkpoint_file ckpts/lhanaf_dense_m32r1b4_instance_validation_l2/Epoch400.pth
#   --all_to_train \
#   --simple_train \

    echo "Done"

# if you want to schedule multiple gpu jobs on a server, better to use this tool.
# run: `bash ./qsub-SurfaceNet_inference.sh`
# for installation & usage, please refer to the author's github: https://github.com/alexanderrichard/queueing-tool
