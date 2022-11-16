#!/usr/bin/env python
# -*- coding: utf-8 -*-
#==============================
# File: data_formatter.py
# Date: 2021-06-28
# Desc: 
# Version: 0.1
#==============================
import os
import sys
import time
import cv2
import numpy as np
import tensorflow as tf
from collections import defaultdict
from datetime import datetime as dt

import struct
from torchkit.data import example_pb2
from PIL import Image
from tqdm import tqdm
import glob
import os
import csv

headers = ['videoid','name','page_idx','page_dir','duration','contentUrl']
count = 0
vids=[]
with open('results_2M_train.csv')as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        if count != 0:
           vids.append(row[0])
        count += 1
        if count % 1000 == 0:
            print(count,'id has loaded...')

with open('results_2M_val.csv')as f:
    f_csv = csv.reader(f)
    for row in f_csv:
        if count != 0:
           vids.append(row[0])
        count += 1
        if count % 100 == 0:
            print(count,'id has loaded...')

class Formatter:
    def __init__(self,
                 input_path,
                 output_dir,
                 output_name,
                 examples_per_shard=1000):
        self.input_path = input_path
        self.output_dir = output_dir
        self.output_name = output_name
        self.index_path = os.path.join(self.output_dir,
                                       self.output_name + '.index')
        self.examples_per_shard = examples_per_shard

        self.input_dir_list = self._load_image_dir_list(
            self.input_path)

    def _load_image_dir_list(self, input_path):
        #path_list = open(input_path, encoding='utf-8').readlines()
        path_list = [input_path+'/'+vid for vid in vids]
        return path_list

    def write(self, transform=None):
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        count = 0 
        shard_size = 0 
        shard_idx = -1
        shard_writer = None
        shard_path = None
        shard_offset = None
        idx_writer = open(self.index_path, 'w')

        b_time = time.time()
        for imgs_path in tqdm(self.input_dir_list):
            vid = imgs_path.strip().split('/')[-1]
            imgs = []
            for img_name in sorted(os.listdir(imgs_path)):
                img_name = os.path.join(imgs_path, img_name)
                try:
                    img = cv2.imread(img_name)
                    # if transform is not None:
                    #     img = transform(img)
                    img = cv2.resize(img, (256, 256))
                    imgs.append(img)
                except Exception as e:
                    print(f'### Read imgs failed: {vid}')
                    continue
            try:
                imgs = np.stack(imgs)
            except Exception as e:
                print(f'### Empty: {vid}, info: {e}')
                continue
            # print(imgs.shape)
            img_bytes = imgs.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))}))
            if shard_size == 0:
                print("{}: {} processed".format(dt.now(), count))
                shard_idx += 1
                record_filename = '{0}-{1:05}.tfrecord'.format(self.output_name, shard_idx)
                if shard_writer is not None:
                    shard_writer.close()
                shard_path = os.path.join(self.output_dir, record_filename) 
                shard_writer = tf.io.TFRecordWriter(shard_path)
                shard_offset = 0
        
            example_bytes = example.SerializeToString()
            shard_writer.write(example_bytes)
            shard_writer.flush()
            idx_writer.write(f'{vid}\t{shard_path}\t{shard_offset}\n')
            shard_offset += (len(example_bytes) + 16)

            count += 1
            if count % 1000 == 0:
                avg_time = (time.time() - b_time) * 1000 / count
                print(''.format(count))
            shard_size = (shard_size + 1) % self.examples_per_shard

        if shard_writer is not None:
            shard_writer.close()
        idx_writer.close()

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

    def read(self):
        vid2rs = self._read_index_file(self.index_path)
        for sid in list(vid2rs.keys()):
            record, offset = vid2rs[sid]
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
                print(record.shape)
                print(f'### Reading {sid}, shape: {record.shape}')


def main():
    input_dir_path = './extracted_frames'
    output_dir = './WebVid_Test'
    output_name = 'path'

    formatter = Formatter(input_dir_path,
                          output_dir,
                          output_name)
    formatter.write()


if __name__ == '__main__':
    main()


