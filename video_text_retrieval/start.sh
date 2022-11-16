#!/bin/bash
sleep 200
export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=1

DATA_PATH=[Your path to json files and MSRVTT videos]
OUT_PATH=[Your path to store checkpoint and log files]
Pretrained_PATH=[Your path to the pre-trained model]

python3 -u -m light.pytorch.launch \
main_task_retrieval_msrvtt.py --do_train --num_thread_reader=4 \
--epochs=5 --batch_size=32 --n_display=50 \
--cross_config_path ../CLIP-modules \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/MSRVTT_data.json \
--features_path ${DATA_PATH}/MSRVTT_Videos \
--warmup_proportion 0.1 \
--output_dir ${OUT_PATH} \
--lr 5e-5 --max_words 32 --max_frames 12 --batch_size_val 32 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 2 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--init_model ${Pretrained_PATH}