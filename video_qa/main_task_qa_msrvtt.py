import os
import time
import json
import torch
import random
import argparse
import numpy as np

from util import parallel_apply, get_logger
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.optimization import BertAdam
from dataloaders.data_dataloaders import DATALOADER_DICT

# torch.distributed.init_process_group(backend="nccl")
from light import light_init
global logger

def get_args(description='Video Task Retrieval'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_train",    action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval",     action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--resume",      action='store_true', help=" ")
    parser.add_argument('--loose_type',  action='store_true', help="Default using tight type for retrieval.")

    parser.add_argument("--datatype", default="msrvtt_qa", type=str, help="Point the dataset to finetune.")
    parser.add_argument('--batch_size',         default=256,    type=int, help='batch size')
    parser.add_argument('--batch_size_val',     default=1000,   type=int, help='batch size')
    parser.add_argument('--max_words',          default=32,     type=int, help='')
    parser.add_argument('--max_frames',         default=12,     type=int, help='')
    parser.add_argument('--feature_framerate',  default=1,      type=int, help='')
    parser.add_argument('--num_thread_reader',  default=1,      type=int, help='')
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--n_display',          default=100,    type=int, help='Information display frequence')

    parser.add_argument('--train_frame_order',  default=0,      type=int, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument('--eval_frame_order',   default=0,      type=int, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--slice_framepos',     default=2,      type=int, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")

    parser.add_argument('--seed',       default=2021, type=int, help='random seed')
    parser.add_argument("--world_size", default=0, type=int, help="distributed training")
    parser.add_argument("--local_rank", default=0, type=int, help="distributed training")
    parser.add_argument("--rank",       default=0, type=int, help="distributed training")
    parser.add_argument('--n_gpu',      default=1, type=int, help="Changed in the execute process.")

    parser.add_argument('--epochs',     default=20,     type=int,   help='upper epoch limit')
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='mask ratio')
    parser.add_argument('--lr',         default=5e-5,   type=float, help='initial learning rate')
    parser.add_argument('--coef_lr',    default=1e-3,     type=float, help='coefficient for bert branch.')
    parser.add_argument("--warmup_proportion", default=0.6, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument('--msrvtt_train_csv', type=str, default='data/.train.csv', help='')
    parser.add_argument('--msrvtt_val_csv', type=str, default='data/.val.csv', help='')
    parser.add_argument('--msrvtt_train_json', type=str, default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--msrvtt_features_path', type=str, default='data/videos_feature.pickle', help='feature path')

    parser.add_argument('--msrvtt_qa_train_json', type=str, default='train.jsonl', help='')
    parser.add_argument('--msrvtt_qa_val_json', type=str, default='val.jsonl', help='')
    parser.add_argument('--msrvtt_qa_test_json', type=str, default='test.jsonl', help='')
    parser.add_argument('--msrvtt_qa_anslabel_json', type=str, default='', help='train_ans2label.json')

    parser.add_argument('--webvid_train_json', type=str, default='frozen_train.json', help='')
    parser.add_argument('--webvid_tfrecord', type=str, default='WebVid_TFRecord', help='')
    parser.add_argument('--webvid_tfrecord_filename', type=str, default='path', help='')

    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--cross_config_path", default="cross-base", type=str, required=False, help="   ")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")  
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    parser.add_argument("--cache_dir",  default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    args = parser.parse_args()

    return args


def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = torch.distributed.get_rank()
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(os.path.join(args.output_dir, "ckpt")):
        os.makedirs(os.path.join(args.output_dir, "ckpt"), exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):
    
    global_step = 0
    restore_path = os.path.join(args.output_dir, 'restore.bin')
    if args.init_model and not os.path.exists(restore_path):
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        if 'model_state_dict' in model_state_dict:
            print(f'loading from existing model {args.init_model}')
            print('loading model from its model_state_dict')
            model_state_dict = model_state_dict['model_state_dict']
    elif os.path.exists(restore_path):
        print(f'find previous checkpoint, try to resume training from {restore_path}')
        t = torch.load(restore_path, map_location='cpu')
        model_state_dict = t['model_state_dict']
        global_step = t['global_step']
        assert model_state_dict is not None, "the model is not correctly loaded"
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args, restore_path=restore_path)

    model.to(device)
    

    return model, global_step

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    added_params = ['fusion_positional_embedding', 'fusion_proj_embedding', 'fusion_logit_bias',
                    'fusion_match_matrix', 'fusion_unmatch_matrix', 'fusion_transformer']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if ("clip." in n and not any(name in n for name in added_params))]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]
    decay_clip_param_tp_fuse = [(n, p) for n, p in decay_param_tp if ("clip." in n and any(name in n for name in added_params))]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if ("clip." in n and not any(name in n for name in added_params))]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]
    no_decay_clip_param_tp_fuse = [(n, p) for n, p in no_decay_param_tp if ("clip." in n and any(name in n for name in added_params))]
    
    # print('------------------------------------------')
    # for (n, p) in decay_clip_param_tp_fuse:
    # 	print(n)
    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in decay_clip_param_tp_fuse], 'weight_decay': weight_decay, 'lr': 5e-6},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0},
        {'params': [p for n, p in no_decay_clip_param_tp_fuse], 'weight_decay': 0.0, 'lr': 5e-6}
    ]
    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)
    resume = os.path.join(args.output_dir, "restore.bin")
    if os.path.exists(resume):
        print(f'prepare optimizer from {resume}')
        t = torch.load(resume, map_location=device)
        assert t['optim_state_dict'] is not None, "reloading optimizer is not correct"
        optimizer.load_state_dict(t['optim_state_dict'])

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)

    return optimizer, scheduler, model

def save_model(epoch, args, model, type_name="", optimizer=None, step=None):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "ckpt", "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", step))
    
    resume = os.path.join(args.output_dir, "restore.bin")
    if os.path.exists(resume):
        os.remove(resume)
    assert optimizer is not None, 'optimizer is invalid'
    assert step is not None, 'step must be not None'
    checkpoint = {'global_step': step,
                  'model_state_dict': model_to_save.state_dict(),
                  'optim_state_dict': optimizer.state_dict()}

    torch.save(checkpoint, output_model_file)
    # torch.save(model_to_save.state_dict(), output_model_file)
    torch.save(checkpoint, resume)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

        model.to(device)
    else:
        model = None
    return model

def eval_epoch(args, model, test_dataloader, device, n_gpu):

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################

    model.eval()

    n_correct = 0
    n_sample = 0
    with torch.no_grad():
        for bid, batch in enumerate(test_dataloader):
            # if bid > 5:
            #     break
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
            question_ids, question_mask, segment_ids, video, video_mask, answer_ids, align_text, align_mask = batch
            
            num_correct, num_total = model(question_ids, question_mask, video, video_mask, answer_ids, align_text, align_mask)

            n_correct += num_correct
            n_sample += num_total

            if args.local_rank == 0:
                print("{}/{}\r".format(bid, len(test_dataloader)))
        n_correct = torch.tensor([ n_correct ]).to('cuda')
        n_sample = torch.tensor([ n_sample ]).to('cuda')
        
        correct_gather = [torch.tensor([ n_correct]).to('cuda') for _ in range(args.world_size)]
        sample_gather = [torch.tensor([ n_sample ]).to('cuda') for _ in range(args.world_size)]
        
        torch.distributed.all_gather(correct_gather, n_correct, async_op=False)
        torch.distributed.all_gather(sample_gather, n_sample, async_op=False)
        torch.distributed.barrier()
        
        correct_gather = torch.tensor(correct_gather)
        sample_gather = torch.tensor(sample_gather)

        print('correct_gather', correct_gather.sum(), 'sample_gather', sample_gather.sum())
        total_acc = correct_gather.sum() / sample_gather.sum() * 100
    
    if args.local_rank == 0:
        logger.info("VideoQA-Metric")
        logger.info('\t>>>  Acc: {:.2f} - data_len: {:.1f} - loader_len: {:.1f}'.format(total_acc, sample_gather.sum(), len(test_dataloader)))

    model.train()
    return total_acc

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    tasks = ['cap', 'vsm', 'mlm', 'mfm', 'mpm']
    prob = [0, 1, 0, 0, 0]
    task = 'vsm'

    for step, batch in enumerate(train_dataloader):
        # model.half()
        # task = taskscheduler.getTask()
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        question_ids, question_mask, segment_ids, video, video_mask, answer_ids, align_text, align_mask = batch
        # print(sent_scores)
        # break
        loss = model(question_ids, question_mask, video, video_mask, answer_ids, align_text, align_mask)
        # break
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        
        if args.rank == 0:
            print(f'step is {step}, loss is ', loss.item())
        
        loss.backward()

        total_loss += float(loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if scheduler is not None:
            scheduler.step()  # Update learning rate schedule

        optimizer.step()
        optimizer.zero_grad()

        # https://github.com/openai/CLIP/issues/46
        if hasattr(model, 'module'):
            torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
        else:
            torch.clamp_(model.clip.logit_scale.data, max=np.log(100))

        global_step += 1

        if global_step % log_step == 0 and local_rank == 0:
            logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", 
                        epoch + 1, args.epochs, 
                        step + 1, len(train_dataloader), 
                        "-".join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                        float(loss),
                        (time.time() - start_time) / log_step )
            start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step
        
@light_init(params={"training_framework": "pytorch_ddp"})
def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = ClipTokenizer()

    model, global_step = init_model(args, device, n_gpu, args.local_rank)

    ## ####################################
    # freeze testing
    ## ####################################
    assert args.freeze_layer_num <= 12 and args.freeze_layer_num >= -1
    if hasattr(model, "clip") and args.freeze_layer_num > -1:
        for name, param in model.clip.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue    # need t o train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= args.freeze_layer_num:
                    continue    # need to train

            if args.linear_patch == "3d" and name.find("conv2."):
                continue
            else:
                if 'fusion' in name or 'local' in name:
                    continue
                print(name, ' is freezed')
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    assert args.datatype in DATALOADER_DICT
    test_dataloader, test_length, test_sampler = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)
    val_dataloader, val_length, val_sampler = DATALOADER_DICT[args.datatype]["val"](args, tokenizer)

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(val_dataloader))

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = len(train_dataloader) * args.epochs
        
        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)

        best_score = 0.00001
        best_output_model_file = "None"
        # global_step = 0
        # acc_test = eval_epoch(args, model, test_dataloader, device, n_gpu)
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader ,device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank)

            # break
            acc_test = eval_epoch(args, model, test_dataloader, device, n_gpu)
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

                output_model_file = None
                ## Uncomment if want to save checkpoint
                if args.rank == 0:
                    output_model_file = save_model(epoch, args, model, type_name="", optimizer=optimizer, step=global_step)
                
                if best_score <= acc_test:
                    best_score = acc_test
                    best_output_model_file = output_model_file
                
                logger.info("The best model is: {}, the Acc is: {:.4f}".format(best_output_model_file, best_score))
    elif args.do_eval:
        #acc_val = eval_epoch(args, model, val_dataloader, device, n_gpu)
        acc_test = eval_epoch(args, model, test_dataloader, device, n_gpu)                

        ## Uncomment if want to test on the best checkpoint
        # if args.local_rank == 0:
        #     model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
        #     eval_epoch(args, model, test_dataloader, device, n_gpu)

    # elif args.do_eval:
    #     if args.local_rank == 0:
    #         eval_epoch(args, model, test_dataloader, device, n_gpu)

if __name__ == "__main__":
    main()