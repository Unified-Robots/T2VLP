import os
import cv2
import json
import torch
import random
import struct
import numpy as np
from PIL import Image
from collections import defaultdict
from torchkit.data import example_pb2
from dataloaders.rawvideo_util import RawVideoExtractor


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


class WEBVID_TrainDataLoader(torch.utils.data.Dataset):
    """WebVid train dataset loader."""
    def __init__(
            self,
            json_path,
            tfrecord_dir_path,
            tfrecord_file_name,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            unfold_sentences=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):

        self.tfrecord = Formatter(tfrecord_dir_path,tfrecord_file_name)
        
        self.sentences_dict = {}
        with open(json_path, 'r') as f:
            for items in f:
                item = json.loads(items)
                target_dict = {
                    'video_id': item['video_id'],
                    'caption': item['caption']
                }
                self.sentences_dict[len(self.sentences_dict)] = target_dict

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
        
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.sentences_dict)

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
        # 1 x 12 x 1 x 3 x 224 x 224
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):

            raw_video_data = self.tfrecord.read_video(str(video_id))

            images = []
            for frame in raw_video_data:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(self.rawVideoExtractor.transform(Image.fromarray(frame_rgb).convert("RGB")))
            raw_video_data = torch.tensor(np.stack(images))
            
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                
                sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                video_slice = raw_video_slice[sample_indx, ...]

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=0)

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
    
    def get_pretrain_mask(self, pairs_text, pairs_mask, pairs_segment, video, video_mask):
        max_video_len = video_mask.shape[-1]  # print(video_mask.shape) [1,12]
        max_text_len = pairs_mask.shape[-1]  # print(pairs_mask.shape) [1,32]

        fusion_labels = np.concatenate((video_mask[:,0:1], video_mask[:,0:1], video_mask, pairs_mask), axis=-1)[0]  # [1,12+32] -> [12+32]
        sep_idx = np.expand_dims(np.concatenate((video_mask.sum(axis=-1) + 1, max_video_len + pairs_mask.sum(axis=-1) + 1), axis=-1), axis=0)  # [1,2]
        
        task_idx = random.random()  # [0,1) random

        if task_idx < 0.25:  # task: 'VMM'
            fusion_labels[1:sep_idx[0][0]] = -1

        elif task_idx > 0.75:  # task: 'TMM'
            fusion_labels[max_video_len+3:sep_idx[0][1]] = -1

        else:  # task: 'MFM-MLM'
            mfm_mask = np.array([i for i in range(1, sep_idx[0][0])])
            mlm_mask = np.array([i for i in range(max_video_len + 3, sep_idx[0][1])])
            mfm_mlm_mask = np.concatenate((mfm_mask, mlm_mask), -1)
            mfm_mlm_idx = np.random.binomial(1, 0.15, len(mfm_mlm_mask))
            
            mask = mfm_mlm_mask[mfm_mlm_idx == 1]

            if len(mask) == 0:
                mask = random.sample(mfm_mlm_mask.tolist(),1)
            try:
                fusion_labels[mask] = -1
            except:
                pass

        # original: 0/1, mask: -1 
        return fusion_labels


    def __getitem__(self, idx):
        video_id = self.sentences_dict[idx]['video_id']
        caption = self.sentences_dict[idx]['caption']

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask = self._get_rawvideo(choice_video_ids)

        fusion_labels = self.get_pretrain_mask(pairs_text, pairs_mask, pairs_segment, video, video_mask)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, fusion_labels
