#!/bin/bash

#block(name=zt_B001_eval, threads=10, memory=100000, subtasks=1, gpu=true, hours=200)
   TASK_NAME=zt_B001_eval
   python -u train_instance.py  \
   --taskname $TASK_NAME \
   --dataset scannet\
   --batch_size 12 \
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
   --val_reps 3 \
   --use_feature c \
   --evaluate \
   --use_dense_model \
   --checkpoint_file ckpts/zt_B001/Epoch280.pth 
#   --all_to_train  
#   --use_elastic \
#   --simple_train	\


   python evaluate_instance.py  \
   --taskname $TASK_NAME \
   --dataset scannet
#   --evaluate
    echo "Done"

# if you want to schedule multiple gpu jobs on a server, better to use this tool.
# run: `bash ./qsub-SurfaceNet_inference.sh`
# for installation & usage, please refer to the author's github: https://github.com/alexanderrichard/queueing-tool

