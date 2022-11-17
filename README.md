# Focus and Align: Learning Tube Tokens for Video-Language Pre-training

The implementation of paper [**Focus and Align: Learning Tube Tokens for Video-Language Pre-training**].

Tube tokens are introduced to exploit the spatio-temporal information of frame patches for video-language pre-training. In order to leverage the complementary information within each frame, the learned semantic centers are utilized to integrate the patch representations within each frame. To enhance the alignments between videos and the paired descriptions, a local alignment mechanism is introduced.


## Requirement
```
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip3 install ftfy regex tqdm
pip3 install opencv-python boto3 requests pandas
pip3 install tensorflow==2.4.0
pip3 install java java-1.8.0-openjdk-devel
```

## Trained Models

|          Model name          |                             link                             |
| :--------------------------: | :----------------------------------------------------------: |
|   WebVid 2.5M Pre-training   |                       [Pre-training]()                       |
| MSR-VTT Video-text Retrieval | [MSR-VTT retrieval](https://drive.google.com/file/d/1sfIDLUGHfGtMS5ttCFfo-ecE71hgHBm1/view?usp=sharing) |
|   MSVD Video-text Retrieval  | [MSVD retrieval]() |
|  LSMDC Video-text Retrieval  | [LSMDC_retrieval](https://drive.google.com/file/d/1PXV4I6nD0JBRmMSUHyH0_1sVRYy7wkO-/view?usp=sharing) |
| DiDeMo Video-text Retrieval  | [DiDeMo retrieval](https://drive.google.com/file/d/16Ai3ksp5kLkcdwIjSNARMIB9qABZQEnQ/view?usp=sharing) |
|      MSR-VTT Captioning      | [MSR-VTT caption](https://drive.google.com/file/d/10yMgLXgDvnit9un77Zkpv3jWoaKTRnD7/view?usp=sharing) |
|       MSVD Captioning        | [MSVD caption](https://drive.google.com/file/d/1UZUoq5SVrsXid1X1PfHYFJZ-jFwSWlL9/view?usp=sharing) |
|          MSR-VTT QA          | [MSR-VTT qa](https://drive.google.com/file/d/1GyFO0HYUm3GfwYpKdEyJVpBkdvKs7st9/view?usp=sharing) |
|           MSVD QA            |                         [MSVD qa]()                          |

## Data processing for WebVid-2.5M

The official data and video links can be found [here](https://github.com/m-bain/webvid).
We have prepared the captions of this dataset and the corresponding data can be found in [here](https://github.com/m-bain/webvid).
For the corresponding videos, you can download them via the given [script](https://github.com/m-bain/webvid/blob/main/download.py). After that, you need to uniformly extract 12 frames for each video clip and run the script `python3 frozen_tfrecord.py` to process the extracted frames. It should be mentioned that the variables including "input_dir_path" and "output_dir" should be modified as needed.

## Pre-training based on WebVid-2.5M

It should be noted that the communication between gpus depends on tencent platform. You need to modify the code contains "light" to achieve the communication between gpus as needed. To obtain the pre-trained model, you need to firstly run `sh start_clip.sh` to fine-tune the parameters of CLIP model. After that, run `sh start_fusion.sh` to train the cross-modality encoder. The pre-trained model can be found [here](https://github.com/m-bain/webvid). To run the above scripts, you need to modify the path to the processed data. For convenience, you can put the processed videos in folder `WebVid_TFRecord`. The contents in `CLIP-modules` can be found [here](https://github.com/m-bain/webvid).

```
DATA_PATH=[Your path to json files and MSRVTT videos]
OUTPUT_PATH=[Your path to store checkpoint and log files]

python3 -u -m light.pytorch.launch \
main_task_clip.py --do_train --num_thread_reader=2 \
--visual_pretrain_path ./CLIP-modules/ViT-B-16.pt \
--cross_config_path ./CLIP-modules \
--epochs=5 --batch_size=32 --n_display=50 \
--train_csv ${DATA_PATH}/frozen_train_labels.json \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/frozen_train_labels.json \
--features_path ${DATA_PATH}/MSRVTT_Videos \
--warmup_proportion 0.1 \
--output_dir ${OUTPUT_PATH} \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 32 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
```

```
DATA_PATH=[Your path to json files and MSRVTT videos]
INIT_MODEL=[Your path to the best checkpoint of start_clip.sh]
OUTPUT_PATH=[Your path to store checkpoint and log files]

python3 -u -m light.pytorch.launch \
main_task_fusion.py --do_train --num_thread_reader=4 \
--cross_config_path ./CLIP-modules \
--epochs=10 --batch_size=32 --n_display=50 \
--train_csv ${DATA_PATH}/frozen_train_labels.json \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/frozen_train_labels.json \
--features_path ${DATA_PATH}/MSRVTT_Videos \
--warmup_proportion 0.2 \
--output_dir ${OUTPUT_PATH} \
--lr 1e-5 --max_words 32 --max_frames 12 --batch_size_val 32 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 2 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header meanP \
--init_model ${INIT_MODEL}
```


## Data Preparing for DOWNSTREAM TASKS

For the following four data splits, the processed sentences and splits can be found [here](http://ms-multimedia-challenge.com/2017/dataset). You need to download the raw videos following the instructions below:

**For MSRVTT**

The official data and video links can be found in [link](http://ms-multimedia-challenge.com/2017/dataset). 

**For MSVD**

Raw videos can be download from [link](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/). 

The splits and `raw_captions` can be found in the wonderful job [collaborative-experts](https://github.com/albanie/collaborative-experts/blob/master/misc/datasets/msvd/README.md).

**For LSMDC**

You must obtain permission from MPII to download and use the data. The download link is [here](https://sites.google.com/site/describingmovies/download).
The 1000 test clips data is [link](http://www.google.com/url?q=http%3A%2F%2Fdatasets.d2.mpi-inf.mpg.de%2FmovieDescription%2Fprotected%2Flsmdc2016%2FLSMDC16_challenge_1000_publictect.csv&sa=D&sntz=1&usg=AFQjCNGIaGVhCeb6zNfUs2UL1zNzoEtaSg). 

**For DiDeMo**

Raw videos can be download from [link](https://github.com/LisaAnne/LocalizingMoments).



## Adaption to DOWNSTREAM TASKS

It should be noted that the code except the dataloader is the same for each downstream task. For each downstream task, please choose the corresponding dataloader for different datasets. The code for downstream tasks can be found in folders `video_text_retrieval`, `video_captioning` and `video_qa` respectively. The trained models for downstream tasks can be download from [link](https://github.com/LisaAnne/LocalizingMoments).

### Video-text Retrieval (e.g. MSRVTT)

```
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
```

### Video Captioning (e.g. MSVD)
```
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
```

### Video Question Answering (e.g. MSRVTT-QA)
```
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
```


# Acknowledgments
Our code is based on [CLIP4clip](https://github.com/ArrowLuo/CLIP4Clip), [CLIP (ViT-B/16)](https://github.com/openai/CLIP) and [UniVL](https://github.com/microsoft/UniVL).
