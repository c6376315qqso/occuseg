#!/bin/bash

#block(name=occuseg_training_0, threads=10, memory=100000, subtasks=1, gpu=true, hours=200)
   source activate p1
   TASK_NAME=leyao_210824_02
   python -u train_instance.py  \
   --taskname $TASK_NAME \
   --dataset scannet\
   --batch_size 3\
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
   --m 16 \
   --block_reps 1 \
   --scale 20 \
   --residual_blocks \
   --kernel_size 3 \
   --use_rotation_noise \
   --val_reps 1 \
   --use_feature c \
   --use_dense_model \
   --use_elastic \
   --model_type uncertain \
   --gpu 2\
   --mask_name m33_66.pth \
   --bceloss weighted_bce \
   --pretrain ./ckpts/test_train/Epoch260.pth\
   --uncertain_task_weight 0.5 \
   --uncertain_weight 15.0 \
   --consistency_weight 20.0\
   --uncertain_st_epoch 40\
   --alpha 5.0\
   --beta 1.0\
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
