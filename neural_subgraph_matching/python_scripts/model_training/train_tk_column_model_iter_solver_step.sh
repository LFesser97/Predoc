#!/bin/bash


CUDA_VISIBLE_DEVICES=0 \
python3 train_net_valloss.py --dataset_name tk-columns, --json_annotation_train /mnt/data01/Japan/teikoku/japan-digitization-pipeline/teikoku_column_model/retrain_column_model/jsons/annotation-a.json \
--image_path_train /mnt/data01/Japan/teikoku/japan-digitization-pipeline/teikoku_column_model/retrain_column_model/images, \
--json_annotation_val /mnt/data01/Japan/teikoku/japan-digitization-pipeline/teikoku_column_model/retrain_column_model/jsons/annotation-b.json, \
--image_path_val /mnt/data01/Japan/teikoku/japan-digitization-pipeline/teikoku_column_model/retrain_column_model/images, \
--config-file /mnt/data01/Japan/teikoku/japan-digitization-pipeline/teikoku_column_model/retrain_column_model/config/config.yaml \
--resume \
	
OUTPUT_DIR /mnt/data01/Japan/teikoku/japan-digitization-pipeline/teikoku_column_model/retrain_column_model/output \
	
	# check whether these settings are still correct
    SOLVER.IMS_PER_BATCH 4, \
    SOLVER.MAX_ITER 80000, \
    INPUT.MIN_SIZE_TEST 1250 \
    SOLVER.BASE_LR 0.0035 \
    SOLVER.GAMMA 0.3 \
    SOLVER.STEPS 5000,10000,20000,40000,60000 \
    INPUT.MIN_SIZE_TRAIN 704,736,768,800,900,1000 \
    MODEL.RPN.POST_NMS_TOPK_TEST 1750 \
    MODEL.RPN.POST_NMS_TOPK_TRAIN 2250 \
    MODEL.WEIGHTS "detectron2://ImageNetPretrained/FAIR/X-101-32x8d.pkl" \
    MODEL.RPN.PRE_NMS_TOPK_TEST 1750 \
    MODEL.RPN.PRE_NMS_TOPK_TRAIN 2250 \
    MODEL.ROI_HEADS.NMS_THRESH_TEST 0.35 \
    TEST.EVAL_PERIOD 100 \
    SOLVER.CHECKPOINT_PERIOD 60000

#The new dataset
#    --json_annotation_train /mnt/data01/yxm/train0901.json \
#    --image_path_train      /mnt/data01/yxm/train0901 \
#    --json_annotation_val   /mnt/data01/yxm/test0901.json \
#    --image_path_val        /mnt/data01/yxm/test0901 \
#The changed default model weights: 
#config.yaml
#   #MODEL.WEIGHTS /mnt/data01/yxm/model_final.pth \
#default, no new labels: outputv1
#0.0025 56000 outputv2
#
#First: 0.0035 40000 Step: 30000 is the end benchmark using new labels output v3
#First: 0.0035 40000 Step: 30000 is the end benchmark using new labels output v4 0.25 the thres+test
#Second: 0.0025 56000 Step: 42000 is the end output v5 0.15
#Change the head together with the max iteration step 0.25
#The change the head to 0.15 outputv7 not change the steps
#0.25
#0.3 output v8 not yet out

# --json_annotation_val   /mnt/data01/yxm/test0907v2.json \
# --image_path_val        /mnt/data01/yxm/test0907faster \