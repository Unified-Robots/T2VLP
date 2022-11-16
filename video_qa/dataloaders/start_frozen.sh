#!/bin/bash
sleep 200
export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=1

DATA_PATH=/apdcephfs/share_1324356/zifengchai/smart/data/data_CLIP4clip
python3 -u -m light.pytorch.launch \
main_task_retrieval.py --do_train --num_thread_reader=2 \
--epochs=10 --batch_size=128 --n_display=50 \
--train_csv ${DATA_PATH}/frozen_train_labels.json \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/frozen_train_labels.json \
--output_dir ${DATA_PATH}/baby_blue7/clip4clip_webvid_128_gpus/ \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 32 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP