from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
from torch import nn

from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import random
import copy
import os
from torch.nn import functional as F

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, restore_path=None, *inputs, **kwargs):
        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        clip_state_dict = CLIP.get_config(pretrained_clip_name="ViT-B/16")
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

            if not os.path.exists(restore_path):
                if key.startswith('transformer'):
                    copyed_key = "clip." + key.replace("transformer", "fusion_transformer")
                    if copyed_key not in state_dict.keys():
                        state_dict[copyed_key] = val.clone()
                if key.startswith('positional_embedding'):
                    copyed_key = "clip." + key.replace("positional_embedding", "fusion_positional_embedding")
                    if copyed_key not in state_dict.keys():
                        state_dict[copyed_key] = val.clone()
        print('-----------------------------------------------------------')
        for key in list(state_dict.keys()):
            # print('-----------------------------------')
            # print(key)
            if key.startswith('clip.visual') and not key.startswith('clip.visual.transformer'):

                new_key = key.replace('visual', 'local_visual')
                # print(key, ' ', new_key)
                if new_key not in state_dict.keys():
                    state_dict[new_key] = state_dict[key].clone()
            elif key.startswith('clip.visual.transformer.resblocks.10'):
                new_key = key.replace('clip.visual.transformer.resblocks.10', 'clip.local_visual.transformer.resblocks.0')
                # print(key, ' ', new_key)
                if new_key not in state_dict.keys():
                    state_dict[new_key] = state_dict[key].clone()
            elif key.startswith('clip.visual.transformer.resblocks.11'):
                new_key = key.replace('clip.visual.transformer.resblocks.11', 'clip.local_visual.transformer.resblocks.1')
                # print(key, ' ', new_key)
                if new_key not in state_dict.keys():
                    state_dict[new_key] = state_dict[key].clone()
         

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1
        self.mask_ratio = self.task_config.mask_ratio
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        self.dropout = nn.Dropout(0.2)

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()
        self.visual_linear = nn.Linear(embed_dim, embed_dim)
        self.embed_dim = embed_dim

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)

        self.loss_fct = CrossEn()
        self.simloss = nn.MSELoss(reduction='mean')

        self.apply(self.init_weights)
        self.soft = nn.Softmax(dim=-1)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None, task="vsm", scores=None, fusion_labels=None):
        assert task in ['vsm', 'mlm', 'mfm', 'cap', 'mpm'], "task is not defined"
        flag_training = True
        if task == "vsm":
            max_video_len = video_mask.shape[-1]
            max_text_len = attention_mask.shape[-1]

            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts
            logit_weight = torch.matmul(self.clip.fusion_proj_embedding, self.clip.token_embedding.weight.T)

            mlm_ids = input_ids.clone()
            # print(mlm_ids)
            sequence_output, visual_output, text_tokens, tube_token = self.get_sequence_visual_output(mlm_ids, token_type_ids, attention_mask,
                                                                                          video, video_mask,shaped=True, video_frame=video_frame)
            tube_num = tube_token.shape[1]
            
            video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
            video_tokens = visual_output * video_mask_un
            video_mask_un_sum = torch.sum(video_mask_un, dim =1, dtype=torch.float)
            video_mask_un_sum[video_mask_un_sum == 0.] = 1.
            v_sep_token = (torch.sum(video_tokens, dim=1) / video_mask_un_sum).unsqueeze(1)

            concated_visual = torch.cat([v_sep_token, visual_output], dim=1)
            concated_visual_len = concated_visual.shape[1]

            #--------------------align_loss--------------------------------------------------

            align_center = self.clip.fusion_center.squeeze().clone()
            align_center = self.clip.fusion_proj_center(align_center.half())
            align_center = align_center.unsqueeze(0).repeat(attention_mask.shape[0], 1, 1)
            align_weight =torch.matmul(align_center.float(), text_tokens.float().permute(0, 2, 1))
            align_mask = attention_mask.clone().float()
            align_mask[align_mask < 0.5] = float('-inf')
            align_mask = align_mask.unsqueeze(1).repeat(1, align_center.shape[1], 1)
            align_weight = align_weight + align_mask
            align_weight_soft = F.softmax(align_weight, dim=-1)
            # print(align_weight_soft)
            text_center = torch.matmul(align_weight_soft, text_tokens.float())
            align_loss = self.simloss(text_center, tube_token)

            #------------------------------VTM-----------------------------------------
            sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,
                                                           shaped=True, loose_type=self.loose_type)
            sim_loss = 0
            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss = (sim_loss1 + sim_loss2) / 2

            vtm_attention_mask = attention_mask.clone().float()
            vtm_attention_mask[vtm_attention_mask==0] = float('-inf')
            vtm_attention_mask[vtm_attention_mask==1] = 0
            
            vtm_tube_mask = torch.ones(video_mask.shape[0], tube_num).long().to(video_mask.device)
            vtm_video_mask = torch.cat((video_mask[:, 0:1], video_mask, vtm_tube_mask), dim=-1).float()
            vtm_video_mask[vtm_video_mask==0] = float('-inf')
            vtm_video_mask[vtm_video_mask==1] = 0
            # print(vtm_video_mask)
            # print(vtm_attention_mask.shape)
            concated_visual = concated_visual.detach()
            tube_token = tube_token.detach()
            text_tokens = text_tokens.detach()
            vtm_attention_mask = vtm_attention_mask.detach()
            vtm_video_mask = vtm_video_mask.detach()
            
            all_video_tokens = allgather(torch.cat((concated_visual, tube_token), dim=1), self.task_config)
            all_text_tokens = allgather(text_tokens, self.task_config)
            all_text_mask = allgather(vtm_attention_mask, self.task_config)
            all_video_mask = allgather(vtm_video_mask, self.task_config)
            torch.distributed.barrier()
            # print(all_text_mask.shape)
            
            diag_mask = torch.ones(sim_matrix.shape[0])
            diag_mask = torch.diag(diag_mask)

            t2v_scores = F.softmax(sim_matrix, dim=-1)
            v2t_scores = F.softmax(sim_matrix.T, dim=-1)
            t2v_scores[diag_mask==1] = 0.
            v2t_scores[diag_mask==1] = 0.

            t2v_hard_idx = torch.multinomial(t2v_scores, 1, replacement=False)
            v2t_hard_idx = torch.multinomial(v2t_scores, 1, replacement=False)
            # print(t2v_hard_idx.shape)

            t2v_hard_neg = all_video_tokens[t2v_hard_idx][self.task_config.rank*b:self.task_config.rank*b+b]
            v2t_hard_neg = all_text_tokens[v2t_hard_idx][self.task_config.rank*b:self.task_config.rank*b+b]
            t2v_hard_mask = all_video_mask[t2v_hard_idx][self.task_config.rank*b:self.task_config.rank*b+b]
            v2t_hard_mask = all_text_mask[v2t_hard_idx][self.task_config.rank*b:self.task_config.rank*b+b]

            t2v_hard_neg = torch.cat((t2v_hard_neg, text_tokens.unsqueeze(1).repeat(1, t2v_hard_neg.shape[1], 1, 1)), dim=-2)
            v2t_hard_neg = torch.cat((torch.cat((concated_visual, tube_token), dim=1).unsqueeze(1).repeat(1, v2t_hard_neg.shape[1], 1, 1), v2t_hard_neg), dim=-2)

            t2v_hard_mask = torch.cat((t2v_hard_mask, vtm_attention_mask.unsqueeze(1).repeat(1, t2v_hard_mask.shape[1], 1)),dim=-1)
            t2v_hard_mask = t2v_hard_mask.unsqueeze(2).repeat(1, 1, t2v_hard_mask.shape[-1], 1)

            v2t_hard_mask = torch.cat((vtm_video_mask.unsqueeze(1).repeat(1, v2t_hard_mask.shape[1], 1), v2t_hard_mask), dim=-1)
            v2t_hard_mask = v2t_hard_mask.unsqueeze(2).repeat(1, 1, v2t_hard_mask.shape[-1], 1)

            pos_input = torch.cat((concated_visual, tube_token, text_tokens), dim=1).unsqueeze(1)
            pos_mask = torch.cat((vtm_video_mask, vtm_attention_mask), dim=-1).unsqueeze(1)
            pos_mask = pos_mask.repeat(1, pos_mask.shape[-1], 1).unsqueeze(1)

            vtm_input = torch.cat((pos_input, t2v_hard_neg, v2t_hard_neg), dim=1)
            vtm_mask = torch.cat((pos_mask, t2v_hard_mask, v2t_hard_mask), dim=1)
            
            vtm_input = vtm_input.view(-1, vtm_input.shape[-2], vtm_input.shape[-1])
            vtm_mask = vtm_mask.view(-1, vtm_mask.shape[-2], vtm_mask.shape[-1])

            targets = torch.cat((torch.ones(pos_input.shape[0],pos_input.shape[1]),
                                 torch.zeros(t2v_hard_neg.shape[0],t2v_hard_neg.shape[1]),
                                 torch.zeros(v2t_hard_neg.shape[0],v2t_hard_neg.shape[1])),dim=1).long().to(pos_input.device)
            # print(targets)
            targets = targets.view(-1)
            pos_emd = self.clip.fusion_positional_embedding[:vtm_input.size(1), :]
            vtm_input = vtm_input + pos_emd

            vtm_input = vtm_input.permute(1, 0, 2)
            vtm_output = self.clip.fusion_transformer(vtm_input.half(), mask=vtm_mask.half(), task='fusion')
            vtm_output = vtm_output.permute(1, 0, 2)     
            
            vtm_v_sep = vtm_output[:, 0, :]
            vtm_t_sep = torch.cat([vtm_output[i][vtm_mask[i][0]==0][-1:] for i in range(vtm_output.shape[0])],dim=0)

            match_scores = self.dropout(vtm_v_sep/vtm_v_sep.norm(dim=-1, keepdim=True)) @ self.clip.fusion_match_matrix.half() @ self.dropout(vtm_t_sep.T/vtm_t_sep.T.norm(dim=-1, keepdim=True))
            unmatch_scores = self.dropout(vtm_v_sep/vtm_v_sep.norm(dim=-1, keepdim=True)) @ self.clip.fusion_unmatch_matrix.half() @ self.dropout(vtm_t_sep.T/vtm_t_sep.T.norm(dim=-1, keepdim=True))

            diag_mask = torch.ones(match_scores.shape[0])
            diag_mask = torch.diag(diag_mask)
            # print(diag_mask)
            match = match_scores[diag_mask==1].view(-1,1)
            unmatch = unmatch_scores[diag_mask==1].view(-1,1)
            vtm_predicted = torch.cat((unmatch, match),dim=-1)
            vtm_loss = self.CrossEntropyLoss(vtm_predicted, targets)

            return vtm_loss, sim_loss, align_loss

    
    def mlm(self, input_ids, token_type_ids, masks):
        bz, sent_len = input_ids.shape
        lengths = masks.squeeze().sum(1)
        mask = torch.zeros(bz, sent_len, sent_len)

        output_filter = torch.zeros(bz, sent_len)
        output_filter.fill_(float('-inf'))
        masked_input_ids = copy.deepcopy(input_ids)

        for i in range(bz):
            flag = True
            for j in range(sent_len):
                if j == 0:
                    output_filter[i, j] = 0.0
                elif j < lengths[i].data:
                    prob = random.random()
                    if prob < self.mask_ratio:
                        output_filter[i, j] = 1.0
                        masked_input_ids[i, j] = 0
                        mask[i, :, j].data.fill_(float("-inf"))
                        flag = False
                else:
                    mask[i, :, j].data.fill_(float("-inf"))
            if flag:
                output_filter[i, 1] = 1.0
                masked_input_ids[i, 1] = 0
                mask[i, :, 1].data.fill_(float("-inf"))
        
        hidden_masked, ground_truth = self.clip.forward_mlm(input_ids, mask, masked_input_ids)
        output_filter = output_filter.unsqueeze(-1).expand_as(hidden_masked)
        predicted_output = hidden_masked[output_filter == 1.0].view(-1, hidden_masked.shape[-1])
        target_output = ground_truth[output_filter == 1.0].view(-1, hidden_masked.shape[-1])
        neg_output = ground_truth[output_filter == 0.0].view(-1, hidden_masked.shape[-1])
        loss = self.nce_loss(predicted_output, target_output, neg_output)
        loss = loss.mean()
        return loss
    
    def get_mlm_metrics(self, input_ids, token_type_ids, masks, video, video_mask=None):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        
        bz, sent_len = input_ids.shape
        lengths = masks.squeeze().sum(1)
        mask = torch.zeros(bz, sent_len, sent_len)

        output_filter = torch.zeros(bz, sent_len)
        output_filter.fill_(float('-inf'))
        masked_input_ids = copy.deepcopy(input_ids)

        for i in range(bz):
            flag = True
            for j in range(sent_len):
                if j == 0:
                    output_filter[i, j] = 0.0
                elif j < lengths[i].data:
                    prob = random.random()
                    if prob < self.mask_ratio:
                        output_filter[i, j] = 1.0
                        masked_input_ids[i, j] = 0
                        mask[i, :, j].data.fill_(float("-inf"))
                        flag = False
                else:
                    mask[i, :, j].data.fill_(float("-inf"))
            if flag:
                output_filter[i, 1] = 1.0
                masked_input_ids[i, 1] = 0
                mask[i, :, 1].data.fill_(float("-inf"))
        
        hidden_masked, ground_truth = self.clip.forward_mlm(input_ids, mask, masked_input_ids)
        output_filter = output_filter.unsqueeze(-1).expand_as(hidden_masked)
        predicted_output = hidden_masked[output_filter == 1.0].view(-1, hidden_masked.shape[-1])
        target_output = ground_truth[output_filter == 1.0].view(-1, hidden_masked.shape[-1])
        neg_output = ground_truth[output_filter == 0.0].view(-1, hidden_masked.shape[-1])
        loss = self.nce_loss(predicted_output, target_output, neg_output)
        loss = loss.mean()
        return 1, 1, loss

    
    def nce_loss(self, masked_output, target_output, neg_output):
        # dot product of ground truth feature
        masked_score = masked_output.matmul(target_output.t())
        # dot product of neative samples
        neg_score = masked_output.matmul(neg_output.t())

        logits = torch.cat([masked_score, neg_score], dim=1).float()
        targets = torch.arange(0, masked_output.size(0),
                                   dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, targets,
                                   reduction='none')
        return loss  

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden, text = self.clip.encode_text(input_ids)
        sequence_hidden = sequence_hidden.float()
        text = text.float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))
        text = text.view(bs_pair, -1, text.size(-1))

        return sequence_hidden, text

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1, train_flag=None):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        # visual_hidden = self.clip.encode_image(video, video_frame=video_frame, flag=train_flag).float()
        visual_hidden, tube_token = self.clip.encode_image(video, video_frame=video_frame, flag=train_flag)
        visual_hidden = visual_hidden.float()
        tube_token = tube_token.float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        return visual_hidden, tube_token

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1, flag=None, fusion_text=None):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        # print(input_ids.shape)
        # print(fusion_text.shape)
        # print(input_ids)
        # print('----------------')
        # print(fusion_text)

        sequence_output, text = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output, tube_token = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)

        return sequence_output, visual_output, text, tube_token

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _loose_similarity(self, sequence_output, visual_output, attention_mask, video_mask, sim_header="meanP", weight=None):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        if sim_header == "meanP":
            # Default: Parameter-free type
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original
        elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output
            seq_length = visual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_output = visual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original

        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            # if weight is not None:
            #     weight = allgather(weight, self.task_config)
            torch.distributed.barrier()
        # print(video_mask.shape)
        # print(weight.dtype)
        # print(visual_output.dtype)

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        # weight = weight.unsqueeze(1).repeat(1, retrieve_logits.shape[-1])
        # print(retrieve_logits.shape)
        # retrieve_logits = retrieve_logits * weight
        # print(retrieve_logits)
        return retrieve_logits, weight

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text      # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1)\
            .to(device=attention_mask.device, dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False, loose_type=False, additional_score=None):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()
        if loose_type:
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf"]
            retrieve_logits, weight = self._loose_similarity(sequence_output, visual_output, attention_mask, video_mask, sim_header=self.sim_header, weight=additional_score)
        else:
            assert self.sim_header in ["tightTransf"]
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask, )

        return retrieve_logits, contrastive_direction, weight
