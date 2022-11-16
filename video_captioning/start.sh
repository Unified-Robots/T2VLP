#!/bin/bash
sleep 200
export NCCL_DEBUG=INFO
pip3 install pycocoevalcap
#export NCCL_IB_DISABLE=1

DATA_PATH=[Your path to json files and MSRVTT videos]
Pretrained_Model=[Your path to the pretrained model]
OUTPUT_PATH=[Your path to store checkpoint and log files]
python3 -u -m light.pytorch.launch \
msvd_captioning.py --do_train --num_thread_reader=4 \
--cross_config_path ../CLIP-modules \
--epochs=10 --batch_size=32 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/MSVD_Videos \
--warmup_proportion 0.2 \
--output_dir ${OUTPUT_PATH} \
--lr 5e-5 --max_words 32 --max_frames 12 --batch_size_val 32 \
--datatype msvd --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--init_model ${Pretrained_Model}