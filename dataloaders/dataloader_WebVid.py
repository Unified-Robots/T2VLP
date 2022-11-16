from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
from dataloaders.rawvideo_util import RawVideoExtractor
from copy import deepcopy
import torch
import json
import os
import sys
import time
import cv2

import tensorflow as tf
from collections import defaultdict
from datetime import datetime as dt
import struct
from torchkit.data import example_pb2
from PIL import Image
import torch
from tqdm import tqdm


class Formatter:
    def __init__(self,
                 file_dir,
                 file_name,
                 examples_per_shard=1000):
        #self.input_path = input_path
        self.file_dir = file_dir
        self.file_name = file_name
        self.index_path = os.path.join(self.file_dir,
                                       self.file_name + '.index')
        self.examples_per_shard = examples_per_shard
        self.vid2rs = self._read_index_file(self.index_path)

   

    def _read_index_file(self, index_file):
        vid2rs = defaultdict(list)
        with open(index_file, 'r') as ifs:
            for line in ifs:
                vid, record_name, tf_record_offset = line.rstrip().split('\t')
                vid2rs[vid].append(os.path.join(record_name))
                vid2rs[vid].append(int(tf_record_offset))
            return vid2rs

    def _parser(self, feature_list):
        for key, feature in feature_list:
            if key == 'image':
                image_raw = feature.bytes_list.value[0]
                image = np.fromstring(image_raw, dtype=np.uint8)
                image = image.reshape(-1, 256, 256, 3)
                #image = Image.fromarray(np.uint8(image[1])).convert('RGB')
        return image

    def read_video(self,sid):
        if len(self.vid2rs[sid]) != 2:
            print(self.vid2rs[sid])
            record, offset = self.vid2rs[sid][:2]
        else:
            record, offset = self.vid2rs[sid]
        with open(record, 'rb') as ifs:
            ifs.seek(offset)
            byte_len_crc = ifs.read(12)
            proto_len = struct.unpack('Q', byte_len_crc[:8])[0]
            pb_data = ifs.read(proto_len)
            if len(pb_data) < proto_len:
                print(f'### Read pb_data error, '
                        f'proto_len: {proto_len}, '
                        f'pb_data len: {len(pb_data)}')
            example = example_pb2.Example()
            example.ParseFromString(pb_data)
            # keep key value in order
            feature = sorted(example.features.feature.items())
            record = self._parser(feature)
            return record

    def has_video(self,sid):
        if sid in self.vid2rs.keys():
            return True
        else:
            return False


class MSRVTT_DataLoader(Dataset):
    """MSRVTT dataset loader."""
    def __init__(
            self,
            csv_path,
            features_path,
            dir_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        file_name = 'path'
        self.tfrecord = Formatter(dir_path,file_name)
        self.data = json.load(open(csv_path))
        # file = open(csv_path, 'r')
        # for item in file.readlines():
        #     tmp_dict = json.loads(item)
        #     self.data.append(deepcopy(tmp_dict))
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.data)

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            if 0 in input_ids:
                input_ids = [item for item in input_ids if item > 0]
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            #video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            #if os.path.exists(video_path) is False:
               # video_path = video_path.replace(".mp4", ".webm")
            
            raw_video_data = self.tfrecord.read_video(video_id)
            images = []
            for frame in raw_video_data:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(self.rawVideoExtractor.transform(Image.fromarray(frame_rgb).convert("RGB")))
            raw_video_data = torch.tensor(np.stack(images))
            # print(raw_video_data.shape)
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        video_id = self.data[idx]['video_id']
        sentence = self.data[idx]['caption']

        #print(video_id, sentence)

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        
        return pairs_text, pairs_mask, pairs_segment, video, video_mask


class MSRVTT_TrainDataLoader(Dataset):
    """MSRVTT train dataset loader."""
    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            dir_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            unfold_sentences=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        file_name = 'path'
        self.tfrecord = Formatter(dir_path,file_name)
        # self.data = []
        print(json_path)
        self.data = json.load(open(json_path))
        # file = open(csv_path, 'r')

        # for item in file.readlines():
        #     tmp_dict = json.loads(item)
        #     self.data.append(deepcopy(tmp_dict))

        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        
        self.sentences_dict = {}
        for itm in self.data:
            # print(itm)
            self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
        self.sample_len = len(self.sentences_dict)
        

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

        

    def __len__(self):
        return self.sample_len

    def get_pretrain_mask(self, pairs_text, pairs_mask, pairs_segment, video, video_mask):
        max_video_len = video_mask.shape[-1]
        max_text_len = pairs_mask.shape[-1]

        fusion_labels = np.concatenate((video_mask[:, 0:1], video_mask, pairs_mask), axis=-1)[0]
        sep_idx = np.expand_dims(np.concatenate((video_mask.sum(axis=-1), max_video_len+pairs_mask.sum(axis=-1)), axis=-1), axis=0)
        # print(video_mask.sum(axis=-1))
        # print(max_video_len+pairs_mask.sum(axis=-1))
        # print(sep_idx)
        mlm_mask = np.array([i for i in range(max_video_len+2, sep_idx[0][1])])
        mlm_idx = np.random.binomial(1, 0.15, len(mlm_mask))

        mask = mlm_mask[mlm_idx==1]

        try:
            if len(mask) == 0:
                mask = sample(mlm_mask.tolist(), 1)
            fusion_labels[mask] = -1
        except:
            fusion_labels[max_video_len+2] = -1

        return fusion_labels


    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        # print(len(choice_video_ids))
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            # print(words)

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            if 0 in input_ids:
                input_ids = [item for item in input_ids if item > 0]
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        # 1 x 12 x 1 x 3 x 256 x 256
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            #video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            #if os.path.exists(video_path) is False:
               # video_path = video_path.replace(".mp4", ".webm")
            
            raw_video_data = self.tfrecord.read_video(video_id)
            #raw_video_data[:,[0,1,2],:,:] = raw_video_data[:,[2,1,0],:,:]
            images = []
            for frame in raw_video_data:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(self.rawVideoExtractor.transform(Image.fromarray(frame_rgb).convert("RGB")))
            raw_video_data = torch.tensor(np.stack(images))
            # print(raw_video_data.shape)
            
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_framepos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_framepos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        # print(choice_video_ids)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        fusion_labels = self.get_pretrain_mask(pairs_text, pairs_mask, pairs_segment, video, video_mask)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, fusion_labels



