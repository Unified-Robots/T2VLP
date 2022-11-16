#!/bin/bash
sleep 200
export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=1

DATA_PATH=[Your path to json files and videos]
OUTPUT_PATH=[Your path to store checkpoint and log files]
INIT_MODEL=[Your path to the pre-trained model]
python3 -u -m light.pytorch.launch \
main_task_qa_msrvtt.py --do_train --num_thread_reader=4 \
--epochs=5 --batch_size=32 --n_display=50 \
--cross_config_path ../CLIP-modules \
--msrvtt_train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--msrvtt_val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--msrvtt_train_json ${DATA_PATH}/MSRVTT_data.json \
--msrvtt_qa_train_json ${DATA_PATH}/train.jsonl \
--msrvtt_qa_val_json ${DATA_PATH}/val.jsonl \
--msrvtt_qa_test_json ${DATA_PATH}/test.jsonl \
--msrvtt_qa_anslabel_json ${DATA_PATH}/train_ans2label.json \
--msrvtt_features_path ${DATA_PATH}/MSRVTT_Videos \
--webvid_train_json ${DATA_PATH}/frozen_train.json \
--webvid_tfrecord ${DATA_PATH}/WebVid_TFRecord \
--warmup_proportion 0.1 \
--output_dir ${OUTPUT_PATH} \
--lr 2e-4 --max_words 32 --max_frames 12 --batch_size_val 32 \
--expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--init_model ${INIT_MODEL}