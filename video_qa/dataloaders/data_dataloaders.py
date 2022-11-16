import torch
import numpy as np

from dataloaders.dataloader_webvid_retrieval import WEBVID_TrainDataLoader
from dataloaders.dataloader_msrvtt_retrieval import MSRVTT_DataLoader, MSRVTT_TrainDataLoader
from dataloaders.dataloader_msvd_retrieval import MSVD_DataLoader
from dataloaders.dataloader_lsmdc_retrieval import LSMDC_DataLoader
from dataloaders.dataloader_msrvtt_qa import MSRVTT_QA_DataLoader


def dataloader_msrvtt_qa_train(args, tokenizer):
    msrvtt_trainset = MSRVTT_QA_DataLoader(
        json_path=args.msrvtt_qa_train_json,
        features_path=args.msrvtt_features_path,
        qa_anslabel_json_path=args.msrvtt_qa_anslabel_json,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_trainset)
    dataloader = torch.utils.data.DataLoader(
        msrvtt_trainset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_trainset), train_sampler


def dataloader_msrvtt_qa_val(args, tokenizer):
    msrvtt_valset = MSRVTT_QA_DataLoader(
        json_path=args.msrvtt_qa_val_json,
        features_path=args.msrvtt_features_path,
        qa_anslabel_json_path=args.msrvtt_qa_anslabel_json,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_valset)
    dataloader_msrvtt = torch.utils.data.DataLoader(
        msrvtt_valset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
        sampler=val_sampler,
    )
    return dataloader_msrvtt, len(msrvtt_valset), val_sampler


def dataloader_msrvtt_qa_test(args, tokenizer):
    msrvtt_testset = MSRVTT_QA_DataLoader(
        json_path=args.msrvtt_qa_test_json,
        features_path=args.msrvtt_features_path,
        qa_anslabel_json_path=args.msrvtt_qa_anslabel_json,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_testset)
    dataloader_msrvtt = torch.utils.data.DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
        sampler=test_sampler,
    )
    return dataloader_msrvtt, len(msrvtt_testset), test_sampler


def dataloader_webvid_train(args, tokenizer):
    webvid_dataset = WEBVID_TrainDataLoader(
        json_path=args.webvid_train_json,
        tfrecord_dir_path=args.webvid_tfrecord,
        tfrecord_file_name=args.webvid_tfrecord_filename,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(webvid_dataset)
    dataloader = torch.utils.data.DataLoader(
        webvid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(webvid_dataset), train_sampler


def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_TrainDataLoader(
        csv_path=args.msrvtt_train_csv,
        json_path=args.msrvtt_train_json,
        features_path=args.msrvtt_features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_msrvtt_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = torch.utils.data.DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler


def dataloader_msrvtt_test(args, tokenizer):
    msrvtt_testset = MSRVTT_DataLoader(
        csv_path=args.msrvtt_val_csv,
        features_path=args.msrvtt_features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = torch.utils.data.DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


def dataloader_msvd_train(args, tokenizer):
    msvd_dataset = MSVD_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)
    dataloader = torch.utils.data.DataLoader(
        msvd_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msvd_dataset), train_sampler


def dataloader_msvd_test(args, tokenizer, subset="test"):
    msvd_testset = MSVD_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = torch.utils.data.DataLoader(
        msvd_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msvd_testset)


def dataloader_lsmdc_train(args, tokenizer):
    lsmdc_dataset = LSMDC_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(lsmdc_dataset)
    dataloader = torch.utils.data.DataLoader(
        lsmdc_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(lsmdc_dataset), train_sampler


def dataloader_lsmdc_test(args, tokenizer, subset="test"):
    lsmdc_testset = LSMDC_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
    )
    dataloader_msrvtt = torch.utils.data.DataLoader(
        lsmdc_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(lsmdc_testset)


DATALOADER_DICT = {
    "msrvtt": {"train": dataloader_msrvtt_train, "val": None, "test": dataloader_msrvtt_test},
    "msvd": {"train": dataloader_msvd_train, "val": dataloader_msvd_test, "test": dataloader_msvd_test},
    "lsmdc": {"train": dataloader_lsmdc_train, "val": dataloader_lsmdc_test, "test": dataloader_lsmdc_test},
    "webvid": {"train": dataloader_webvid_train, "val": None, "test": dataloader_msrvtt_test},
    "msrvtt_qa": {"train": dataloader_msrvtt_qa_train, "val": dataloader_msrvtt_qa_val, "test": dataloader_msrvtt_qa_test}
    }
