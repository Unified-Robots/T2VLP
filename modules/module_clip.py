import os
import torch
import torch.nn.functional as F
from torch import nn

from typing import Tuple, Union
from collections import OrderedDict
import math

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
}





class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.heads = heads
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, video_frame=-1, mask=None, task=None):
        if mask is not None:
            bz, length, dim = mask.shape
            mask = mask.unsqueeze(1).expand(bz, self.heads, length, dim)
            mask = mask.reshape(-1, length, dim)
            mask = mask.contiguous()
        if task is not None:
            patch_num, frame_num, dim = x.shape
            middle_out = None
            for i in range(len(self.resblocks)):
                if i == 10:
                    x = x.permute(1, 0, 2)
                    middle_out = x.clone()
                    x = x.view(-1, patch_num*12, dim)
                    x = x.permute(1, 0, 2)

                (x, video_frame, mask) = self.resblocks[i]((x, video_frame, mask))
            x = x.permute(1, 0, 2)
            x = x.view(-1, patch_num, dim)
            x = x.permute(1, 0, 2)
            return x, middle_out

        return self.resblocks((x, video_frame, mask))[0]


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, attn_mask=None):
        # attn_mask_ = self.attn_mask
        # if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
        #     attn_mask_ = self.attn_mask(x.size(0))   # LND
        #     #print(attn_mask_)
        attn_mask_ = attn_mask if attn_mask is not None else None
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__') and attn_mask_ is None:
            attn_mask_ = self.attn_mask(x.size(0))
        
        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, x_tuple:tuple):
        x, video_frame, attn_mask = x_tuple
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return (x, video_frame, attn_mask)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 linear_patch: str = '2d',):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # For 3D
        assert linear_patch in ['2d', '3d']
        self.linear_patch = linear_patch
        if self.linear_patch == '3d':
            self.conv2 = nn.Conv3d(in_channels=3, out_channels=width, kernel_size=(3, patch_size, patch_size),
                                   stride=(1, patch_size, patch_size), padding=(1, 0, 0), bias=False)

    def forward(self, x: torch.Tensor, video_frame=-1):

        if self.linear_patch == '3d':
            assert video_frame != -1
            x_3d = x.reshape(-1, video_frame, x.shape[-3], x.shape[-2], x.shape[-1])
            x_3d = x_3d.permute(0, 2, 1, 3, 4)
            x_3d = self.conv2(x_3d)     # shape = [*, width, frame, grid, grid]
            x_3d = x_3d.permute(0, 2, 1, 3, 4)      # shape = [*, frame, width, grid, grid]
            x = x_3d.reshape(-1, x_3d.shape[-3], x_3d.shape[-2], x_3d.shape[-1]).contiguous() # shape = [*, width, grid, grid]
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, middle_out = self.transformer(x, video_frame=video_frame, task='vision')
        x = x.permute(1, 0, 2)  # LND -> NLD


        return x, middle_out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class CLIP(torch.nn.Module):
    def __init__(self,
                 embed_dim,
                 # vision
                 image_resolution, 
                 vision_layers: Union[Tuple[int, int, int, int], int], 
                 vision_width, 
                 vision_patch_size,
                 # text
                 context_length: int,
                 vocab_size: int, 
                 transformer_width: int, 
                 transformer_heads: int, 
                 transformer_layers: int,
                 # vision linear of patch
                 linear_patch: str = '2d'):
        super(CLIP, self).__init__()
        
        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                linear_patch=linear_patch
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask
        )

        vision_heads = vision_width // 64
        self.local_visual = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=2,
            heads=vision_heads,
            output_dim=embed_dim,
            linear_patch=linear_patch)

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        
        self.center_num = 8
        self.fusion_center = nn.Parameter(vision_width ** -0.5 * torch.randn(1, 1, self.center_num, vision_width))
        # torch.nn.init.orthogonal_(self.fusion_center[0][0], gain=1)
        self.fusion_proj_center = nn.Linear(vision_width, embed_dim)

        self.ln_final = LayerNorm(transformer_width)
        self.lm_head = nn.Linear(transformer_width, embed_dim)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.dropout = nn.Dropout(0.2)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @staticmethod
    def get_config(model_path):
        if os.path.exists(model_path):
            try:  # loading JIT archive
                model = torch.jit.load(model_path, map_location="cpu").eval()
                state_dict = model.state_dict()
            except RuntimeError:
                state_dict = torch.load(model_path, map_location="cpu")
        else:
            raise RuntimeError(f"Model not found; model path = {model_path}")

        return state_dict

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_text(self, text, return_hidden=False):
        x = self.token_embedding(text).type(self.dtype)
        pos_emd = self.positional_embedding[:x.size(1), :].type(self.dtype)  # [77,512]->[32,512]
        x = x + pos_emd

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        hidden = self.ln_final(x).type(self.dtype) @ self.text_projection  # Matrix multiplication

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)]
        # [bs,512]
        
        if return_hidden:
            return x, hidden

        return x, hidden
    
    def encode_image(self, image, return_hidden=False, video_frame=-1):
        hidden, middle_out = self.visual(image.type(self.dtype), video_frame=video_frame)
        hidden = self.visual.ln_post(hidden) @ self.visual.proj
        
        x = hidden[:, 0, :]
        _, patch_num, patch_dim = middle_out.shape
        middle_out = middle_out.view(-1, video_frame, patch_num, patch_dim)
        head = torch.mean(middle_out[:, :, 0, :], dim=1)

        regions = middle_out[:, :, 1:patch_num, :].contiguous()
        centers = self.fusion_center.repeat(regions.shape[0], regions.shape[1], 1, 1)

        weight = torch.matmul(centers.float(), regions.permute(0, 1, 3, 2).float())
        weight = F.softmax(weight, -1)
        generate_tubes = torch.matmul(weight, regions.float())
        generate_tubes = generate_tubes.permute(0, 2, 1, 3)
        head = head.unsqueeze(1).repeat(1, generate_tubes.shape[1], 1).unsqueeze(2)

        final_tubes = torch.cat((head, generate_tubes), dim=2)
        # print(final_tubes.shape)
        bz, tube_num, frame_num, dim = final_tubes.shape
        final_tubes = final_tubes.view(-1, frame_num, dim)

        final_tubes = final_tubes.permute(1, 0, 2)
        encoded_tube = self.local_visual.transformer(final_tubes.half())
        encoded_tube = encoded_tube.permute(1, 0, 2)

        encoded_tube = encoded_tube.view(bz, tube_num, frame_num, dim)
        # print(encoded_tube.shape)

        tube_token = encoded_tube[:, :, 0, :]
        tube_token = self.local_visual.ln_post(tube_token) @ self.local_visual.proj
        
        if return_hidden:
            return x, hidden
        return x, tube_token

    def build_attention_mask(self, context_length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.zeros(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask


def convert_weights(model: torch.nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, torch.nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)
