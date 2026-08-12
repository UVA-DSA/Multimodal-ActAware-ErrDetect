# implementation adapted from:
# https://github.com/yabufarha/ms-tcn/blob/master/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.models.layers import trunc_normal_
except ImportError:
    from torch.nn.init import trunc_normal_
import copy
import numpy as np
from scipy import interpolate
import pdb
from decoder import TransformerDecoderLayer, TransformerDecoder
from transformer_cot import MyTransformer, Transformer_cot, MyTransformer_video
import os

try:
    import clip
except ImportError:
    clip = None


def _build_hf_clip_fallback():
    """HuggingFace CLIP text-only fallback. Lazy so `import models` works without OpenAI clip."""
    try:
        from transformers import CLIPModel, CLIPTokenizer
    except Exception as e:
        raise ImportError(
            "Could not load CLIP. Install OpenAI CLIP (recommended): "
            "pip install git+https://github.com/openai/CLIP.git"
        ) from e

    class _HFClipTextEncoder:
        def __init__(self, model_name, device='cpu'):
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self.model.eval()

        def encode_text(self, tokenized_inputs):
            if isinstance(tokenized_inputs, dict):
                tokenized_inputs = {
                    key: value.to(next(self.model.parameters()).device)
                    for key, value in tokenized_inputs.items()
                }
            with torch.no_grad():
                outputs = self.model.get_text_features(**tokenized_inputs)
            return outputs

    class _FallbackClipModule:
        _MODEL_MAP = {
            'ViT-B/32': 'openai/clip-vit-base-patch32',
            'ViT-L/14': 'openai/clip-vit-large-patch14',
        }

        def __init__(self):
            self._tokenizers = {}

        def _resolve_model_name(self, model_name):
            return self._MODEL_MAP.get(model_name, model_name)

        def load(self, model_name, device='cpu'):
            hf_model_name = self._resolve_model_name(model_name)
            return _HFClipTextEncoder(hf_model_name, device=device), None

        def tokenize(self, texts):
            hf_model_name = self._resolve_model_name('ViT-B/32')
            if hf_model_name not in self._tokenizers:
                self._tokenizers[hf_model_name] = CLIPTokenizer.from_pretrained(hf_model_name)
            tokenizer = self._tokenizers[hf_model_name]
            if isinstance(texts, str):
                texts = [texts]
            return tokenizer(list(texts), return_tensors='pt', padding=True)

    return _FallbackClipModule()


class _LazyClipFallback:
    """Defers HF CLIP import until first use (avoids broken torch/torchvision at import time)."""

    _impl = None

    @classmethod
    def _get_impl(cls):
        if cls._impl is None:
            try:
                cls._impl = _build_hf_clip_fallback()
            except Exception as e:
                low = str(e).lower()
                chain = e.__cause__
                chain_s = str(chain).lower() if chain is not None else ''
                if any(
                    x in low or x in chain_s
                    for x in ('torchvision', 'nms', 'clipmodel', 'could not import module')
                ):
                    raise RuntimeError(
                        "CLIP fallback failed (often mismatched torch/torchvision). "
                        "Install OpenAI CLIP: pip install git+https://github.com/openai/CLIP.git "
                        "or reinstall torchvision to match your PyTorch build (see pytorch.org)."
                    ) from e
                raise
        return cls._impl

    def load(self, model_name, device='cpu'):
        return self._get_impl().load(model_name, device=device)

    def tokenize(self, texts):
        return self._get_impl().tokenize(texts)


if clip is None:
    clip = _LazyClipFallback()

class FPN(nn.Module):
    def __init__(self,num_f_maps):
        super(FPN, self).__init__()
        self.latlayer1 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv1d( num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)

        self.latlayer3 = nn.Conv1d( num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,W = y.size()
        return F.interpolate(x, size=W, mode='linear') + y

    def forward(self,out_list):
        #p3 = out_list[2]
        #c2 = out_list[1]
        #c1 = out_list[0]
        #p2 = self._upsample_add(p3, self.latlayer1(c2))
        #p1 = self._upsample_add(p2, self.latlayer1(c1))
        #return [p1,p2,p3]
        
        p4 = out_list[3]
        c3 = out_list[2]
        c2 = out_list[1]
        c1 = out_list[0]
        p3 = self._upsample_add(p4, self.latlayer1(c3))
        p2 = self._upsample_add(p3, self.latlayer1(c2))
        p1 = self._upsample_add(p2, self.latlayer1(c1))
        return [p1,p2,p3,p4]
    
        '''
        p2 = out_list[1]
        c1 = out_list[0]
        
        p1 = self._upsample_add(p2, self.latlayer1(c1))
        return [p1,p2]
        '''
        

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim )
       

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            # print(self.position_ids)
            position_ids = self.position_ids[:, : x.size(2)]
          
        # print(self.pe(position_ids).size(), x.size())
         
        position_embeddings = self.pe(position_ids).transpose(1,2) + x
        return  position_embeddings


class SAHC_atten(nn.Module):
    ## SAHC from X. Ding et al., “Exploring segment-level semantics for online phase recognition from surgical videos,” IEEE Trans. Med. Imag., vol. 41, no. 11, pp. 3309–3319, 2022.
    def __init__(self, args, num_layers_Basic, num_layers_R, num_R, num_f_maps, num_f_dim, num_classes, causal_conv):
        super(SAHC_atten, self).__init__()
        
        self.num_layers_Basic = num_layers_Basic
        self.num_R = num_R  # 3
        self.num_layers_R = num_layers_R  # 8
        self.num_f_maps = num_f_maps  # 64
        self.dim = num_f_dim  #2048
        self.num_classes = num_classes  # 2
        self.causal_conv =causal_conv

        self.TCN = SingleStageModel1(args, num_layers_Basic, num_f_maps, num_f_dim, num_classes, kernel_size=1, causal_conv = True, dropout = True, hier = False, use_output= True)
        self.Rs = nn.ModuleList([copy.deepcopy(SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes, kernel_size=args.train, causal_conv = True, dropout = False, hier = True)) for _ in range(num_R)])
        #self.conv_out_list = [nn.Conv1d(num_f_maps, num_classes, 1) for s in range(num_R)]
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.fpn = FPN(num_f_maps)

        self.query = nn.Embedding(num_classes, num_f_maps)
        self.position_encoding = LearnedPositionalEncoding(
            19971, num_f_maps
        )
        decoder_layer = TransformerDecoderLayer(num_f_maps, args.head_num, args.embed_num,
                                            0.1, 'relu',normalize_before=True)
        decoder_norm = nn.LayerNorm(num_f_maps)
        self.decoder = TransformerDecoder(decoder_layer, args.block_num, decoder_norm,
                                return_intermediate=False)
        
        #print(
         #   f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
          #  f" {self.num_f_maps}, dim: {self.dim}")
    def forward(self, x):
        out_list = []
        f_list = []
        f, out1 = self.TCN(x)

        f_list.append(f)

        for R in self.Rs:
            f, out1 = R(f)
            f_list.append(f)

        f_list = self.fpn(f_list)
        for f in f_list:
            out_list.append(self.conv_out(f))
        for i in range(len(f_list)):
            if self.position_encoding == None:
                f_list[i] =  f_list[i]
            else:
                # print(f_list[i].size())
                f_list[i] = self.position_encoding(f_list[i]) #[batch_size, num_f_maps, T]
            # query_embed = self.query.weight.unsqueeze(1).repeat( 1, batch_size, 1)
            
            # first_feature = f_list[0]
        first_feature_list= []
        first_feature_list.append(f_list[0])
        first_feature = f_list[0].permute(2,0,1) ##[batch_size, num_f_maps, T] -> [T, batch_size, num_f_maps]
        # print(len(f_list))
            # sss
        for i in range(1, len(f_list)):
            middle_feature = f_list[i]

            first_feature = self.decoder(first_feature, middle_feature, 
                memory_key_padding_mask=None, pos=None, query_pos=None)
        reduced_first_feature = first_feature.permute(1,2,0)
        out_list[0] = self.conv_out(reduced_first_feature)
        #for f, conv_out in zip(f_list, self.conv_out_list):
         #   out_list.append(conv_out(f))
        return out_list, f_list

class COG_video(nn.Module):
    def __init__(self, args, num_layers_Basic, num_layers_R, num_R, num_f_maps, num_f_dim, num_classes, causal_conv, d_model, d_q, len_q, device, gest_template = 'A surgeon is', gest_prompt: str = './utils/gest_prompt_B32.pt'):
        super(COG_video, self).__init__()
        
        self.num_layers_Basic = num_layers_Basic # 11
        self.num_R = num_R  # 2
        self.num_layers_R = num_layers_R  # 10
        self.num_f_maps = num_f_maps  # 64
        self.dim = num_f_dim  #2048
        self.num_classes = num_classes  # 2
        self.causal_conv = causal_conv
        

        self.d_model = d_model #64 or 128
        self.d_q = d_q #8 or 16
        self.len_q = len_q # 30
        self.device = device
        self.gest_prompt = gest_prompt

        num_gest_f = 512 # 768
        
        self.gest_list = ['reaching for needle with right hand',
                    'positioning needle',
                    'pushing needle through tissue',
                    'transferring needle from left to right',
                    'moving to center with needle in grip',
                    'pulling suture with left hand',
                    'pulling suture with right hand',
                    'orienting needle',
                    'using right hand to help tighten suture',
                    'loosening more suture',
                    'dropping suture at end and movign to end points',
                    'reaching for needle with left hand',
                    'making C loop around right hand',
                    'reaching for suture with right hand',
                    'pulling suture with both hands'
        ]
        
        text_model, text_preprocess = clip.load('ViT-B/32', device='cpu') #'ViT-L/14'
        num_gest = len(self.gest_list)  # 15
        self.num_gestures = num_gest
        self.all_gest_fea = torch.zeros(1, num_gest_f) 
        #if not os.path.exists(self.gest_prompt):
        for i in range(num_gest):
            gest_prompt = text_model.encode_text(clip.tokenize(f'{gest_template} {self.gest_list[i]} ...')).float()
            self.all_gest_fea = torch.cat((self.all_gest_fea, gest_prompt), dim = 0)
        self.all_gest_fea = self.all_gest_fea[1:]
        torch.save(self.all_gest_fea, self.gest_prompt)
        self.all_action_fea = nn.Parameter(torch.load(self.gest_prompt), requires_grad=False)
        #else:
         #   self.all_action_fea = nn.Parameter(torch.load(self.gest_prompt), requires_grad=False)
        # print("all_action_fea: ", self.all_action_fea.shape)  # torch.Size([15, 768])
        #self.linear = nn.Linear(in_features = 768, out_features = self.num_f_maps) 

        self.cot = MyTransformer_video(self.dim, num_gest_f, d_model, d_q, len_q, device)
        ##slow path
        self.TCN = SingleStageModel1(args, num_layers_Basic, num_f_maps, d_model, num_classes, kernel_size=1, causal_conv = True, dropout = True, hier = False, use_output= True)
        self.Rs = nn.ModuleList([copy.deepcopy(SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes,  kernel_size=args.train, causal_conv = True, dropout = False, hier = True)) for _ in range(num_R)])
        ##fast path
        self.pool = nn.AvgPool1d(kernel_size=args.k, stride=args.k)
        self.fast_stage1 = SingleStageModel1(args, num_layers_Basic, num_f_maps, d_model, num_classes, kernel_size=1, causal_conv = True, dropout = True, hier = False, use_output= True)
        self.fast_Rs = nn.ModuleList([copy.deepcopy(SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes,  kernel_size=1, causal_conv = True, dropout = False, hier =False, use_output= True)) for _ in range(num_R)])
        

        #self.sf = SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes, kernel_size=16, causal_conv = True, dropout = False, hier = True)
        #self.conv_out_list = [nn.Conv1d(num_f_maps, num_classes, 1) for s in range(num_R)]
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.fpn = FPN(num_f_maps)
        
        #print(
         #   f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
          #  f" {self.num_f_maps}, dim: {self.dim}")
    def forward(self, x):
        # x: visual feature [1, 345, 2048]

        
        all_action_fea = self.all_action_fea.unsqueeze(0) #[1, 15, 768]
        #xx = self.cot(all_action_fea, x) #[1, 345, num_gest * d_model]
        xx = self.cot(all_action_fea, x)
        out_list = []
        f_list = []
        
        xx = xx.permute(0, 2, 1)
        
        ##slow_path
        f, out1 = self.TCN(xx)

        f_list.append(f)

        for R in self.Rs:
            f, out1 = R(f)
            f_list.append(f)

        f_list = self.fpn(f_list)


        for f in f_list:
            out_list.append(self.conv_out(f))
        
        
        ##fast_path
        fast_input = self.pool(xx)
        fast_f, fast_out = self.fast_stage1(fast_input) 
        f_list.append(fast_f)
        out_list.append(fast_out)

        for R in self.fast_Rs:
            fast_f, fast_out = R(F.softmax(fast_out, dim=1))
            f_list.append(fast_f)
            out_list.append(fast_out)

        return out_list, f_list

class COG_GPT(nn.Module):
    def __init__(self, args, num_layers_Basic, num_layers_R, num_R, num_f_maps, num_f_dim, num_classes, causal_conv, d_model, d_q, len_q, device, gest_template = 'A surgeon is', gest_prompt: str = './utils/gest_prompt_B32.pt'):
        super(COG_GPT, self).__init__()
        
        self.num_layers_Basic = num_layers_Basic # 11
        self.num_R = num_R  # 2
        self.num_layers_R = num_layers_R  # 10
        self.num_f_maps = num_f_maps  # 64
        self.dim = num_f_dim  #2048
        self.num_classes = num_classes  # 2
        self.causal_conv = causal_conv
        

        self.d_model = d_model #64 or 128
        self.d_q = d_q #8 or 16
        self.len_q = len_q # 30
        self.device = device
        self.gest_prompt = gest_prompt

        num_gest_f = 512 # 768
        
        self.gest_list = ["The surgeon extends their right hand towards a sterile needle, ensuring a precise and controlled motion to maintain a sterile field",
                          "The surgeon adjusts the needle’s angle and depth, preparing it for insertion into the targeted tissue while maintaining a steady grip",
                          "The surgeon skillfully guides the needle through the layers of tissue, applying consistent pressure to ensure smooth penetration without causing unnecessary damage",
                          "After securing the needle, the surgeon transfers it from their left hand to their right hand, demonstrating dexterity and maintaining focus on the surgical site"
                          "The surgeon aligns the needle centrally over the incision site, ensuring optimal positioning for the next steps in the procedur",
                          "With their left hand, the surgeon pulls the suture taut, ensuring it is appropriately secured to maintain tension in the tissue",
                          "The surgeon uses their right hand to further pull the suture, synchronizing the movements to achieve a balanced tension throughout the suture line",
                          "The surgeon adjusts the orientation of the needle for optimal insertion, considering the angle necessary to navigate through the specific tissue layers",
                          "The surgeon employs their right hand to apply additional force, tightening the suture effectively to secure the tissue and minimize the risk of complications",
                          "The surgeon releases some tension on the suture, allowing for slight adjustments in the tissue positioning to achieve the desired alignment before finalizing the knot",
                          "After completing the suture, the surgeon carefully drops the remaining suture material and shifts focus to the end points, preparing for the final knotting or securing process",
                          "The surgeon reaches out with their left hand to grab a new needle, ensuring their movements are deliberate and within the sterile field",
                          "The surgeon forms a C loop with the suture around their right hand, creating a stable anchor for the next stitching process",
                          "The surgeon extends their right hand to grasp the suture material, ensuring a clean and efficient pick-up for further manipulation",
                          "The surgeon employs both hands to pull the suture tight, achieving the necessary tension to secure the tissue layers while maintaining control over the surgical outcome"
                        ]
        
        text_model, text_preprocess = clip.load('ViT-B/32', device='cpu') #'ViT-L/14'
        num_gest = len(self.gest_list)  # 15
        self.num_gestures = num_gest
        self.all_gest_fea = torch.zeros(1, num_gest_f) 
        #if not os.path.exists(self.gest_prompt):
        for i in range(num_gest):
            gest_prompt = text_model.encode_text(clip.tokenize(f'{self.gest_list[i]} ...')).float()
            self.all_gest_fea = torch.cat((self.all_gest_fea, gest_prompt), dim = 0)
        self.all_gest_fea = self.all_gest_fea[1:]
        torch.save(self.all_gest_fea, self.gest_prompt)
        self.all_action_fea = nn.Parameter(torch.load(self.gest_prompt), requires_grad=False)
        #else:
         #   self.all_action_fea = nn.Parameter(torch.load(self.gest_prompt), requires_grad=False)
        # print("all_action_fea: ", self.all_action_fea.shape)  # torch.Size([15, 768])
        #self.linear = nn.Linear(in_features = 768, out_features = self.num_f_maps) 

        self.cot = MyTransformer(self.dim, num_gest_f, d_model, d_q, len_q, device)
        ##slow path
        self.TCN = SingleStageModel1(args, num_layers_Basic, num_f_maps, num_gest*d_model, num_classes, kernel_size=1, causal_conv = True, dropout = True, hier = False, use_output= True)
        self.Rs = nn.ModuleList([copy.deepcopy(SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes,  kernel_size=args.train, causal_conv = True, dropout = False, hier = True)) for _ in range(num_R)])
        ##fast path
        self.pool = nn.AvgPool1d(kernel_size=args.k, stride=args.k)
        self.fast_stage1 = SingleStageModel1(args, num_layers_Basic, num_f_maps, num_gest*d_model, num_classes, kernel_size=1, causal_conv = True, dropout = True, hier = False, use_output= True)
        self.fast_Rs = nn.ModuleList([copy.deepcopy(SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes,  kernel_size=1, causal_conv = True, dropout = False, hier =False, use_output= True)) for _ in range(num_R)])
        

        #self.sf = SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes, kernel_size=16, causal_conv = True, dropout = False, hier = True)
        #self.conv_out_list = [nn.Conv1d(num_f_maps, num_classes, 1) for s in range(num_R)]
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.fpn = FPN(num_f_maps)
        
        #print(
         #   f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
          #  f" {self.num_f_maps}, dim: {self.dim}")
    def forward(self, x):
        # x: visual feature [1, 345, 2048]

        
        all_action_fea = self.all_action_fea.unsqueeze(0) #[1, 15, 768]
        xx = self.cot(all_action_fea, x) #[1, 345, num_gest * d_model]

        out_list = []
        f_list = []
        
        xx = xx.permute(0, 2, 1)
        
        ##slow_path
        f, out1 = self.TCN(xx)

        f_list.append(f)

        for R in self.Rs:
            f, out1 = R(f)
            f_list.append(f)

        f_list = self.fpn(f_list)


        for f in f_list:
            out_list.append(self.conv_out(f))
        
        
        ##fast_path
        fast_input = self.pool(xx)
        fast_f, fast_out = self.fast_stage1(fast_input) 
        f_list.append(fast_f)
        out_list.append(fast_out)

        for R in self.fast_Rs:
            fast_f, fast_out = R(F.softmax(fast_out, dim=1))
            f_list.append(fast_f)
            out_list.append(fast_out)

        return out_list, f_list

class COG_learnable_token(nn.Module):
    def __init__(self, args, num_layers_Basic, num_layers_R, num_R, num_f_maps, num_f_dim, num_classes, causal_conv, d_model, d_q, len_q, device, gest_template = 'A surgeon is', gest_prompt: str = './utils/gest_prompt_B32.pt'):
        super(COG_learnable_token, self).__init__()
        
        self.num_layers_Basic = num_layers_Basic # 11
        self.num_R = num_R  # 2
        self.num_layers_R = num_layers_R  # 10
        self.num_f_maps = num_f_maps  # 64
        self.dim = num_f_dim  #2048
        self.num_classes = num_classes  # 2
        self.causal_conv = causal_conv
        

        self.d_model = d_model #64 or 128
        self.d_q = d_q #8 or 16
        self.len_q = len_q # 30
        self.device = device
        self.gest_prompt = gest_prompt

        num_gest_f = 512 # 768
        
        self.gest_list = ['reaching for needle with right hand',
                    'positioning needle',
                    'pushing needle through tissue',
                    'transferring needle from left to right',
                    'moving to center with needle in grip',
                    'pulling suture with left hand',
                    'pulling suture with right hand',
                    'orienting needle',
                    'using right hand to help tighten suture',
                    'loosening more suture',
                    'dropping suture at end and movign to end points',
                    'reaching for needle with left hand',
                    'making C loop around right hand',
                    'reaching for suture with right hand',
                    'pulling suture with both hands'
        ]
        
        num_gest = len(self.gest_list)  # 15
        self.num_gestures = num_gest
        self.gest_template = gest_template
        all_gest_fea = []#torch.zeros(1, num_gest_f) 
        for i in range(self.num_gestures):
            text_model, text_preprocess = clip.load('ViT-B/32', device='cpu') #'ViT-L/14'
            gest_prompt = text_model.encode_text(clip.tokenize(f'{self.gest_template} {self.gest_list[i]} ...')).float()
            #self.all_gest_fea = torch.cat((self.all_gest_fea, gest_prompt), dim = 0)
            all_gest_fea.append(gest_prompt)    
        #all_gest_fea = torch.cat(all_gest_fea, dim=0)
        self.all_gest_fea = torch.cat(all_gest_fea, dim=0)
        torch.save(self.all_gest_fea, self.gest_prompt)
        self.all_gest_embeddings = nn.Parameter(torch.load(self.gest_prompt), requires_grad=False)
        
        self.learnable_tokens = nn.Parameter(torch.zeros(num_gest, 1, 512), requires_grad=True)
        
        self.cot = MyTransformer(self.dim, num_gest_f, d_model, d_q, len_q, device)
        ##slow path
        self.TCN = SingleStageModel1(args, num_layers_Basic, num_f_maps, num_gest*d_model, num_classes, kernel_size=1, causal_conv = True, dropout = True, hier = False, use_output= True)
        self.Rs = nn.ModuleList([copy.deepcopy(SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes,  kernel_size=args.train, causal_conv = True, dropout = False, hier = True)) for _ in range(num_R)])
        ##fast path
        self.pool = nn.AvgPool1d(kernel_size=args.k, stride=args.k)
        self.fast_stage1 = SingleStageModel1(args, num_layers_Basic, num_f_maps, num_gest*d_model, num_classes, kernel_size=1, causal_conv = True, dropout = True, hier = False, use_output= True)
        self.fast_Rs = nn.ModuleList([copy.deepcopy(SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes,  kernel_size=1, causal_conv = True, dropout = False, hier =False, use_output= True)) for _ in range(num_R)])
        

        #self.sf = SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes, kernel_size=16, causal_conv = True, dropout = False, hier = True)
        #self.conv_out_list = [nn.Conv1d(num_f_maps, num_classes, 1) for s in range(num_R)]
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.fpn = FPN(num_f_maps)
        
        #print(
         #   f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
          #  f" {self.num_f_maps}, dim: {self.dim}")
    def forward(self, x):
        # x: visual feature [1, 345, 2048]
        all_embedding = []
        for i in range(self.num_gestures):
            gest_embedding = self.all_gest_embeddings[i]
            #self.all_gest_fea = torch.cat((self.all_gest_fea, gest_prompt), dim = 0)
            token_embedding = self.learnable_tokens[i]
            combined_embedding = gest_embedding + token_embedding
            all_embedding.append(combined_embedding)    
        all_embedding = torch.cat(all_embedding, dim=0)
        all_action_fea = all_embedding.unsqueeze(0) #[1, 15, 768]
        xx = self.cot(all_action_fea, x) #[1, 345, num_gest * d_model]

        out_list = []
        f_list = []
        
        xx = xx.permute(0, 2, 1)
        
        ##slow_path
        f, out1 = self.TCN(xx)

        f_list.append(f)

        for R in self.Rs:
            f, out1 = R(f)
            f_list.append(f)

        f_list = self.fpn(f_list)


        for f in f_list:
            out_list.append(self.conv_out(f))
        
        
        ##fast_path
        fast_input = self.pool(xx)
        fast_f, fast_out = self.fast_stage1(fast_input) 
        f_list.append(fast_f)
        out_list.append(fast_out)

        for R in self.fast_Rs:
            fast_f, fast_out = R(F.softmax(fast_out, dim=1))
            f_list.append(fast_f)
            out_list.append(fast_out)

        return out_list, f_list


class COG(nn.Module):
    def __init__(self, args, num_layers_Basic, num_layers_R, num_R, num_f_maps, num_f_dim, num_classes, causal_conv, d_model, d_q, len_q, device, gest_template = 'A surgeon is', gest_prompt: str = './utils/gest_prompt_B32.pt', gesture_prompts=None):
        super(COG, self).__init__()
        
        self.num_layers_Basic = num_layers_Basic # 11
        self.num_R = num_R  # 2
        self.num_layers_R = num_layers_R  # 10
        self.num_f_maps = num_f_maps  # 64
        self.dim = num_f_dim  #2048
        self.num_classes = num_classes  # 2
        self.causal_conv = causal_conv
        

        self.d_model = d_model #64 or 128
        self.d_q = d_q #8 or 16
        self.len_q = len_q # 30
        self.device = device
        self.gest_prompt = gest_prompt

        num_gest_f = 512 # 768
        
        if gesture_prompts is None:
            self.gest_list = ['reaching for needle with right hand',
                        'positioning needle',
                        'pushing needle through tissue',
                        'transferring needle from left to right',
                        'moving to center with needle in grip',
                        'pulling suture with left hand',
                        'pulling suture with right hand',
                        'orienting needle',
                        'using right hand to help tighten suture',
                        'loosening more suture',
                        'dropping suture at end and movign to end points',
                        'reaching for needle with left hand',
                        'making C loop around right hand',
                        'reaching for suture with right hand',
                        'pulling suture with both hands'
            ]
            prompt_texts = [f'{gest_template} {gesture} ...' for gesture in self.gest_list]
        else:
            self.gest_list = list(gesture_prompts)
            prompt_texts = [
                prompt if prompt.endswith('...') else f'{prompt} ...'
                for prompt in self.gest_list
            ]
        
        text_model, text_preprocess = clip.load('ViT-B/32', device='cpu') #'ViT-L/14'
        num_gest = len(self.gest_list)  # 15
        self.num_gestures = num_gest
        self.all_gest_fea = torch.zeros(1, num_gest_f) 
        #if not os.path.exists(self.gest_prompt):
        for prompt_text in prompt_texts:
            gest_prompt = text_model.encode_text(clip.tokenize(prompt_text)).float()
            self.all_gest_fea = torch.cat((self.all_gest_fea, gest_prompt), dim = 0)
        self.all_gest_fea = self.all_gest_fea[1:]
        torch.save(self.all_gest_fea, self.gest_prompt)
        self.all_action_fea = nn.Parameter(torch.load(self.gest_prompt), requires_grad=False)
        #else:
         #   self.all_action_fea = nn.Parameter(torch.load(self.gest_prompt), requires_grad=False)
        # print("all_action_fea: ", self.all_action_fea.shape)  # torch.Size([15, 768])
        #self.linear = nn.Linear(in_features = 768, out_features = self.num_f_maps) 

        self.cot = MyTransformer(self.dim, num_gest_f, d_model, d_q, len_q, device)
        ##slow path
        self.TCN = SingleStageModel1(args, num_layers_Basic, num_f_maps, num_gest*d_model, num_classes, kernel_size=1, causal_conv = True, dropout = True, hier = False, use_output= True)
        self.Rs = nn.ModuleList([copy.deepcopy(SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes,  kernel_size=args.train, causal_conv = True, dropout = False, hier = True)) for _ in range(num_R)])
        ##fast path
        self.pool = nn.AvgPool1d(kernel_size=args.k, stride=args.k)
        self.fast_stage1 = SingleStageModel1(args, num_layers_Basic, num_f_maps, num_gest*d_model, num_classes, kernel_size=1, causal_conv = True, dropout = True, hier = False, use_output= True)
        self.fast_Rs = nn.ModuleList([copy.deepcopy(SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes,  kernel_size=1, causal_conv = True, dropout = False, hier =False, use_output= True)) for _ in range(num_R)])
        

        #self.sf = SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes, kernel_size=16, causal_conv = True, dropout = False, hier = True)
        #self.conv_out_list = [nn.Conv1d(num_f_maps, num_classes, 1) for s in range(num_R)]
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.fpn = FPN(num_f_maps)
        
        #print(
         #   f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
          #  f" {self.num_f_maps}, dim: {self.dim}")
    def forward(self, x):
        # x: visual feature [1, 345, 2048]

        
        all_action_fea = self.all_action_fea.unsqueeze(0) #[1, 15, 768]
        xx = self.cot(all_action_fea, x) #[1, 345, num_gest * d_model]

        out_list = []
        f_list = []
        
        xx = xx.permute(0, 2, 1)
        
        ##slow_path
        f, out1 = self.TCN(xx)

        f_list.append(f)

        for R in self.Rs:
            f, out1 = R(f)
            f_list.append(f)

        f_list = self.fpn(f_list)


        for f in f_list:
            out_list.append(self.conv_out(f))
        
        
        ##fast_path
        fast_input = self.pool(xx)
        fast_f, fast_out = self.fast_stage1(fast_input) 
        f_list.append(fast_f)
        out_list.append(fast_out)

        for R in self.fast_Rs:
            fast_f, fast_out = R(F.softmax(fast_out, dim=1))
            f_list.append(fast_f)
            out_list.append(fast_out)

        return out_list, f_list

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class COG_MLP(nn.Module):
    def __init__(self, args, num_layers_Basic, num_layers_R, num_R, num_f_maps, num_f_dim, num_classes, causal_conv, d_model, d_q, len_q, device, gest_template = 'A surgeon is', gest_prompt: str = './utils/gest_prompt_B32.pt'):
        super(COG_MLP, self).__init__()
        
        self.num_layers_Basic = num_layers_Basic # 11
        self.num_R = num_R  # 2
        self.num_layers_R = num_layers_R  # 10
        self.num_f_maps = num_f_maps  # 64
        self.dim = num_f_dim  #2048
        self.num_classes = num_classes  # 2
        self.causal_conv = causal_conv
        

        self.d_model = d_model #64 or 128
        self.d_q = d_q #8 or 16
        self.len_q = len_q # 30
        self.device = device
        self.gest_prompt = gest_prompt

        num_gest_f = 512 # 768
        
        self.gest_list = ['reaching for needle with right hand',
                    'positioning needle',
                    'pushing needle through tissue',
                    'transferring needle from left to right',
                    'moving to center with needle in grip',
                    'pulling suture with left hand',
                    'pulling suture with right hand',
                    'orienting needle',
                    'using right hand to help tighten suture',
                    'loosening more suture',
                    'dropping suture at end and movign to end points',
                    'reaching for needle with left hand',
                    'making C loop around right hand',
                    'reaching for suture with right hand',
                    'pulling suture with both hands'
        ]
        
        text_model, text_preprocess = clip.load('ViT-B/32', device='cpu') #'ViT-L/14'
        num_gest = len(self.gest_list)  # 15
        self.num_gestures = num_gest
        self.all_gest_fea = torch.zeros(1, num_gest_f) 
        #if not os.path.exists(self.gest_prompt):
        for i in range(num_gest):
            gest_prompt = text_model.encode_text(clip.tokenize(f'{gest_template} {self.gest_list[i]} ...')).float()
            self.all_gest_fea = torch.cat((self.all_gest_fea, gest_prompt), dim = 0)
        self.all_gest_fea = self.all_gest_fea[1:]
        self.MLP = SimpleMLP(num_gest_f, 1024, num_gest_f)
        #torch.save(self.all_gest_fea, self.gest_prompt)
        #self.all_action_fea = nn.Parameter(torch.load(self.gest_prompt), requires_grad=False)
        #else:
         #   self.all_action_fea = nn.Parameter(torch.load(self.gest_prompt), requires_grad=False)
        # print("all_action_fea: ", self.all_action_fea.shape)  # torch.Size([15, 768])
        #self.linear = nn.Linear(in_features = 768, out_features = self.num_f_maps) 

        self.cot = MyTransformer(self.dim, num_gest_f, d_model, d_q, len_q, device)
        ##slow path
        self.TCN = SingleStageModel1(args, num_layers_Basic, num_f_maps, num_gest*d_model, num_classes, kernel_size=1, causal_conv = True, dropout = True, hier = False, use_output= True)
        self.Rs = nn.ModuleList([copy.deepcopy(SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes,  kernel_size=args.train, causal_conv = True, dropout = False, hier = True)) for _ in range(num_R)])
        ##fast path
        self.pool = nn.AvgPool1d(kernel_size=args.k, stride=args.k)
        self.fast_stage1 = SingleStageModel1(args, num_layers_Basic, num_f_maps, num_gest*d_model, num_classes, kernel_size=1, causal_conv = True, dropout = True, hier = False, use_output= True)
        self.fast_Rs = nn.ModuleList([copy.deepcopy(SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes,  kernel_size=1, causal_conv = True, dropout = False, hier =False, use_output= True)) for _ in range(num_R)])
        

        #self.sf = SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes, kernel_size=16, causal_conv = True, dropout = False, hier = True)
        #self.conv_out_list = [nn.Conv1d(num_f_maps, num_classes, 1) for s in range(num_R)]
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.fpn = FPN(num_f_maps)
        
        #print(
         #   f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
          #  f" {self.num_f_maps}, dim: {self.dim}")
    def forward(self, x):
        # x: visual feature [1, 345, 2048]
        all_action_fea = self.MLP(self.all_action_fea)
        
        all_action_fea = self.all_action_fea.unsqueeze(0) #[1, 15, 768]
        xx = self.cot(all_action_fea, x) #[1, 345, num_gest * d_model]

        out_list = []
        f_list = []
        
        xx = xx.permute(0, 2, 1)
        
        ##slow_path
        f, out1 = self.TCN(xx)

        f_list.append(f)

        for R in self.Rs:
            f, out1 = R(f)
            f_list.append(f)

        f_list = self.fpn(f_list)


        for f in f_list:
            out_list.append(self.conv_out(f))
        
        
        ##fast_path
        fast_input = self.pool(xx)
        fast_f, fast_out = self.fast_stage1(fast_input) 
        f_list.append(fast_f)
        out_list.append(fast_out)

        for R in self.fast_Rs:
            fast_f, fast_out = R(F.softmax(fast_out, dim=1))
            f_list.append(fast_f)
            out_list.append(fast_out)

        return out_list, f_list
    
class MSTR(nn.Module):
    ##To generate Delta GVR
    def __init__(self, args, num_layers_Basic, num_layers_R, num_R, num_f_maps, num_f_dim, num_classes, causal_conv, device):
        super(MSTR, self).__init__()
        
        self.num_layers_Basic = num_layers_Basic # 11
        self.num_R = num_R  # 2
        self.num_layers_R = num_layers_R  # 10
        self.num_f_maps = num_f_maps  # 64
        self.dim = num_f_dim  #2048
        self.num_classes = num_classes  # 2
        self.causal_conv = causal_conv
        


        self.device = device
        
        ##slow path
        self.TCN = SingleStageModel1(args, num_layers_Basic, num_f_maps, num_f_dim, num_classes, kernel_size=1, causal_conv = True, dropout = True, hier = False, use_output= True)
        self.Rs = nn.ModuleList([copy.deepcopy(SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes,  kernel_size=args.train, causal_conv = True, dropout = False, hier = True)) for _ in range(num_R)])
        ##fast path
        self.pool = nn.AvgPool1d(kernel_size=16, stride=16)
        self.fast_stage1 = SingleStageModel1(args, num_layers_Basic, num_f_maps, num_f_dim, num_classes, kernel_size=1, causal_conv = True, dropout = True, hier = False, use_output= True)
        self.fast_Rs = nn.ModuleList([copy.deepcopy(SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes,  kernel_size=1, causal_conv = True, dropout = False, hier =False, use_output= True)) for _ in range(num_R)])
        

        #self.sf = SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes, kernel_size=16, causal_conv = True, dropout = False, hier = True)
        #self.conv_out_list = [nn.Conv1d(num_f_maps, num_classes, 1) for s in range(num_R)]
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.fpn = FPN(num_f_maps)
        
        #print(
         #   f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
          #  f" {self.num_f_maps}, dim: {self.dim}")
    def forward(self, x):
        # x: visual feature [1, 345, 2048]
        out_list = []
        f_list = []
        
        xx = x.permute(0, 2, 1)
        
        ##slow_path
        f, out1 = self.TCN(xx)

        f_list.append(f)

        for R in self.Rs:
            f, out1 = R(f)
            f_list.append(f)

        f_list = self.fpn(f_list)


        for f in f_list:
            out_list.append(self.conv_out(f))
        
        
        ##fast_path
        fast_input = self.pool(xx)
        fast_f, fast_out = self.fast_stage1(fast_input) 
        f_list.append(fast_f)
        out_list.append(fast_out)

        for R in self.fast_Rs:
            fast_f, fast_out = R(F.softmax(fast_out, dim=1))
            f_list.append(fast_f)
            out_list.append(fast_out)

        return out_list, f_list
    
class GVR_FastPath(nn.Module):
    # to generate without slow path
    def __init__(self, args, num_layers_Basic, num_layers_R, num_R, num_f_maps, num_f_dim, num_classes, causal_conv, d_model, d_q, len_q, device, gest_prompt: str = './utils/gest_prompt_B32.pt'):
        super(GVR_FastPath, self).__init__()
        
        self.num_layers_Basic = num_layers_Basic # 11
        self.num_R = num_R  # 2
        self.num_layers_R = num_layers_R  # 10
        self.num_f_maps = num_f_maps  # 64
        self.dim = num_f_dim  #2048
        self.num_classes = num_classes  # 2
        self.causal_conv = causal_conv
        

        self.d_model = d_model #64 or 128
        self.d_q = d_q #8 or 16
        self.len_q = len_q # 30
        self.device = device
        self.gest_prompt = gest_prompt
        
        num_gest_f = 512
        
        self.gest_list = ['reaching for needle with right hand',
                    'positioning needle',
                    'pushing needle through tissue',
                    'transferring needle from left to right',
                    'moving to center with needle in grip',
                    'pulling suture with left hand',
                    'pulling suture with right hand',
                    'orienting needle',
                    'using right hand to help tighten suture',
                    'loosening more suture',
                    'dropping suture at end and movign to end points',
                    'reaching for needle with left hand',
                    'making C loop around right hand',
                    'reaching for suture with right hand',
                    'pulling suture with both hands'
        ]
        
        text_model, text_preprocess = clip.load('ViT-B/32', device='cpu')
        num_gest = len(self.gest_list)  # 15
        self.num_gestures = num_gest
        self.all_gest_fea = torch.zeros(1, num_gest_f) 
        if not os.path.exists(self.gest_prompt):
            for i in range(num_gest):
                gest_prompt = text_model.encode_text(clip.tokenize(f'A surgeon is {self.gest_list[i]} ...')).float()
                self.all_gest_fea = torch.cat((self.all_gest_fea, gest_prompt), dim = 0)
            self.all_gest_fea = self.all_gest_fea[1:]
            torch.save(self.all_gest_fea, self.gest_prompt)
            self.all_action_fea = nn.Parameter(torch.load(self.gest_prompt), requires_grad=False)
        else:
            self.all_action_fea = nn.Parameter(torch.load(self.gest_prompt), requires_grad=False)
        # print("all_action_fea: ", self.all_action_fea.shape)  # torch.Size([15, 768])
        #self.linear = nn.Linear(in_features = 768, out_features = self.num_f_maps) 

        self.cot = MyTransformer(self.dim, num_gest_f, d_model, d_q, len_q, device)

        self.fast_stage1 = SingleStageModel1(args, num_layers_Basic, num_f_maps, num_gest*d_model, num_classes, kernel_size=1, causal_conv = True, dropout = False, hier = False, use_output= True)
        

        self.fast_stages = nn.ModuleList([
            copy.deepcopy(
                SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes, kernel_size=1, causal_conv = True, dropout = False, hier = False, use_output= True))
            for _ in range(self.num_R)
        ])

        #self.slow_stage1 = SingleStageModel1(args, num_layers_Basic, num_f_maps, num_gest*d_model, num_classes, kernel_size=1, causal_conv = True, dropout = False, hier = False, use_output= True)
        
        #self.slow_stages = nn.ModuleList([
         #   copy.deepcopy(
          #      SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes, kernel_size=1, causal_conv = True, dropout = False, hier = False, use_output= True))
           # for _ in range(self.num_R)
        #])
        #self.ws = [nn.Parameter(torch.ones(2)) for _ in range(self.num_R + 1)]
        self.pool = nn.AvgPool1d(kernel_size=16, stride=16)


        
    def forward(self, x):
        all_action_fea = self.all_action_fea.unsqueeze(0) #[1, 15, 768]
        xx = self.cot(all_action_fea, x) #[1, 345, num_gest * d_model]

        out_list = []
        f_list = []
        
        xx = xx.permute(0, 2, 1)
        
        length = xx.shape[2]
                
        ##fast_path
        fast_input = self.pool(xx)
        fast_f, fast_out = self.fast_stage1(fast_input)

        up_fast_out = self.Sample(fast_out,  length)#F.interpolate(fast_out, size=length, mode='linear') 
        
        f_list.append(fast_f)
        f_list.append(fast_f)
        out_list.append(up_fast_out)
        out_list.append(fast_out)

        for R in self.fast_stages:
            fast_f, fast_out = R(F.softmax(fast_out, dim=1))
            f_list.append(fast_f)
            out_list.append(fast_out)

        return out_list, f_list
    

        '''
        all_action_fea = self.all_action_fea.unsqueeze(0) #[1, 15, 768]
        xx = self.cot(all_action_fea, x) #[1, 345, num_gest * d_model]
        
        xx = xx.permute(0, 2, 1)

        length = xx.shape[2]
        ##Fast path
        ##### initial stage
        fast_input = self.downsample(xx)
        fast_f, fast_out = self.fast_stage1(fast_input)
        up_fast_out = self.Sample(fast_out,  length)#F.interpolate(fast_out, size=length, mode='linear')#self.Sample(fast_out,  length)
        up_fast_outputs = up_fast_out.unsqueeze(0)
        
        ##### refine stage
        for s in self.fast_stages:
            fast_f, fast_out = s(F.softmax(fast_out, dim=1))
            up_fast_out = self.Sample(fast_out, length)#F.interpolate(fast_out, size=length, mode='linear')#self.Sample(fast_out, length)
            up_fast_outputs = torch.cat((up_fast_outputs, up_fast_out.unsqueeze(0)), dim=0)

        ##Slow path
        ##### initial stage
        slow_f, slow_out = self.slow_stage1(xx)

        out = self.ws[0][0]*slow_out + self.ws[0][1] * up_fast_outputs[0]

        outputs = out.unsqueeze(0)

        ##### refine stage
        i = 1
        for s in self.slow_stages:
            slow_f, slow_out = s(F.softmax(out, dim=1))
            out = self.ws[i][0]*slow_out + self.ws[i][1] * up_fast_outputs[i]
            
            outputs= torch.cat((outputs, out.unsqueeze(0)), dim=0)
            i += 1

        return outputs
        '''
    def Sample(self, input, out_frame):
        ## input: [1, features, in_frame] --> output: [1, features, out_frame]
        in_frame = input.shape[-1]
        inds = torch.linspace(0, in_frame - 1, out_frame)
        rounded_inds = torch.round(inds).to(int)

        return input[:, :, rounded_inds]

class GVR_SlowPath(nn.Module):
    # to generate without fast path
    def __init__(self, args, num_layers_Basic, num_layers_R, num_R, num_f_maps, num_f_dim, num_classes, causal_conv, d_model, d_q, len_q, device,
                 gest_prompt: str = './utils/gest_prompt.pt'):
        super(GVR_SlowPath, self).__init__()
        self.num_layers_Basic = num_layers_Basic
        self.num_R = num_R  # 2
        self.num_layers_R = num_layers_R  # 8
        self.num_f_maps = num_f_maps  # 32
        self.dim = num_f_dim  #2048
        self.num_classes = num_classes  # 7
        self.causal_conv =causal_conv
        

        self.d_model = d_model #64 or 128
        self.d_q = d_q #8 or 16
        self.len_q = len_q # 30
        self.device = device
        self.gest_prompt = gest_prompt

        self.gest_list = ['reaching for needle with right hand',
                    'positioning needle',
                    'pushing needle through tissue',
                    'transferring needle from left to right',
                    'moving to center with needle in grip',
                    'pulling suture with left hand',
                    'pulling suture with right hand',
                    'orienting needle',
                    'using right hand to help tighten suture',
                    'loosening more suture',
                    'dropping suture at end and movign to end points',
                    'reaching for needle with left hand',
                    'making C loop around right hand',
                    'reaching for suture with right hand',
                    'pulling suture with both hands'
        ]
        
        text_model, text_preprocess = clip.load('ViT-L/14', device='cpu')
        num_gest = len(self.gest_list)  # 15
        self.num_gestures = num_gest
        self.all_gest_fea = torch.zeros(1, 768) 
        if not os.path.exists(self.gest_prompt):
            for i in range(num_gest):
                gest_prompt = text_model.encode_text(clip.tokenize(f'A surgeon is {self.gest_list[i]} ...')).float()
                self.all_gest_fea = torch.cat((self.all_gest_fea, gest_prompt), dim = 0)
            self.all_gest_fea = self.all_gest_fea[1:]
            torch.save(self.all_gest_fea, self.gest_prompt)
            self.all_action_fea = nn.Parameter(torch.load(self.gest_prompt), requires_grad=False)
        else:
            self.all_action_fea = nn.Parameter(torch.load(self.gest_prompt), requires_grad=False)

        
        self.TCN = SingleStageModel1(args, num_layers_Basic, num_f_maps, num_f_dim, num_classes, kernel_size=1, causal_conv = True, dropout = True, hier = False, use_output= True)

        self.cot = MyTransformer(num_f_maps, 768, d_model, d_q, len_q, device)
        self.linear = nn.Linear(num_gest*d_model, num_f_maps)
        self.Rs = nn.ModuleList([copy.deepcopy(SingleStageModel1(args, num_layers_R, num_f_maps, num_classes, num_classes, kernel_size=args.train, causal_conv = True, dropout = False, hier = True)) for _ in range(num_R)])
        #self.conv_out_list = [nn.Conv1d(num_f_maps, num_classes, 1) for s in range(num_R)]
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.fpn = FPN(num_f_maps)

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


        
    def forward(self, x):
        out_list = []
        f_list = []
        x = x.permute(0, 2, 1) #[1,345,dim] -> [1, dim, 345]
        f0, out0 = self.TCN(x) #[1, dim, 345] -> [1, num_f_maps, 345]
        
        all_action_fea =self.all_action_fea.unsqueeze(0)
        
        f = self.cot(all_action_fea, f0.permute(0, 2, 1)) #[1, 345, num_gest*d_model]

        f = self.linear(f).permute(0, 2, 1) #[1, 345, num_gest*d_model] ->[1, 345, num_f_maps] -> [1, num_f_maps, 345]
        f_list.append(f)


       
        for R in self.Rs:
            f, out1 = R(f)
            f_list.append(f)

        f_list = self.fpn(f_list)

        for f in f_list:
            out_list.append(self.conv_out(f))
        
        return out_list, f_list
    
class my_GVR(nn.Module):
    #to generate Delta MSTR
    def __init__(self, args, num_f_maps, num_f_dim, num_classes, causal_conv, d_model, d_q, len_q, device, gest_prompt: str = './utils/gest_prompt_B32.pt'):
        super(my_GVR, self).__init__()
        
        self.num_f_maps = num_f_maps  # 64
        self.dim = num_f_dim  #2048
        self.num_classes = num_classes  # 2
        self.causal_conv = causal_conv
        

        self.d_model = d_model #64 or 128
        self.d_q = d_q #8 or 16
        self.len_q = len_q # 30
        self.device = device
        self.gest_prompt = gest_prompt
        
        num_gest_f = 512
        
        self.gest_list = ['reaching for needle with right hand',
                    'positioning needle',
                    'pushing needle through tissue',
                    'transferring needle from left to right',
                    'moving to center with needle in grip',
                    'pulling suture with left hand',
                    'pulling suture with right hand',
                    'orienting needle',
                    'using right hand to help tighten suture',
                    'loosening more suture',
                    'dropping suture at end and movign to end points',
                    'reaching for needle with left hand',
                    'making C loop around right hand',
                    'reaching for suture with right hand',
                    'pulling suture with both hands'
        ]
        
        text_model, text_preprocess = clip.load('ViT-B/32', device='cpu')
        num_gest = len(self.gest_list)  # 15
        self.num_gestures = num_gest
        self.all_gest_fea = torch.zeros(1, num_gest_f) 
        if not os.path.exists(self.gest_prompt):
            for i in range(num_gest):
                gest_prompt = text_model.encode_text(clip.tokenize(f'A surgeon is {self.gest_list[i]} ...')).float()
                self.all_gest_fea = torch.cat((self.all_gest_fea, gest_prompt), dim = 0)
            self.all_gest_fea = self.all_gest_fea[1:]
            torch.save(self.all_gest_fea, self.gest_prompt)
            self.all_action_fea = nn.Parameter(torch.load(self.gest_prompt), requires_grad=False)
        else:
            self.all_action_fea = nn.Parameter(torch.load(self.gest_prompt), requires_grad=False)
        # print("all_action_fea: ", self.all_action_fea.shape)  # torch.Size([15, 768])
        #self.linear = nn.Linear(in_features = 768, out_features = self.num_f_maps) 

        self.cot = MyTransformer(self.dim, num_gest_f, d_model, d_q, len_q, device)
        self.fc = nn.Sequential(nn.Linear(self.num_gestures * d_model, num_f_maps),
                                nn.ReLU(),
                                nn.Linear(num_f_maps, self.num_classes))


        
    def forward(self, x):
        all_action_fea = self.all_action_fea.unsqueeze(0) #[1, 15, 768]
        xx = self.cot(all_action_fea, x) #[1, 345, num_gest * d_model]
        out = self.fc(xx)
        
        return out
    



def fusion(predicted_list,labels):
    all_out_list = []
    resize_out_list = []
    labels_list = []
    all_out = 0

    for out in predicted_list:
        resize_out =F.interpolate(out,size=labels.size(0),mode='nearest')
        resize_out_list.append(resize_out)
        
        if out.size(2)==labels.size(0):
            resize_label = labels
            labels_list.append(resize_label.squeeze().long())
        else:
            # resize_label = max_pool(labels_list[-1].float().unsqueeze(0).unsqueeze(0))
            resize_label = F.interpolate(labels.float().unsqueeze(0).unsqueeze(0),size=out.size(2),mode='nearest')
            
            labels_list.append(resize_label.squeeze().long())
        
        all_out_list.append(out)
    return all_out, all_out_list, labels_list

def fusion2(predicted_list,labels):
    resize_out_list = []
    labels_list = []

    for out in predicted_list:
        resize_out =F.interpolate(out,size=labels.size(0),mode='nearest')
        resize_out_list.append(resize_out)
        labels_list.append(labels)
        
    return resize_out_list, labels_list

class SlowFastMSTCN(nn.Module):
    def __init__(self, fast_frames, mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features, mstcn_causal_conv ):
        super(SlowFastMSTCN, self).__init__()
        self.fast_frames = fast_frames
        self.num_stages = mstcn_stages  # 2
        self.num_layers = mstcn_layers  # 8
        self.num_f_maps = mstcn_f_maps  # 32
        self.dim = mstcn_f_dim  #2048
        self.num_classes = out_features  # 7
        self.causal_conv = mstcn_causal_conv
        print(
            f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
            f" {self.num_f_maps}, dim: {self.dim}")
        self.fast_stage1 = SingleStageModel(self.num_layers,
                                       self.num_f_maps,
                                       self.dim,
                                       self.num_classes,
                                       causal_conv=self.causal_conv)
        self.fast_stages = nn.ModuleList([
            copy.deepcopy(
                SingleStageModel(self.num_layers,
                                 self.num_f_maps,
                                 self.num_classes,
                                 self.num_classes,
                                 causal_conv=self.causal_conv))
            for _ in range(self.num_stages - 1)
        ])

        self.slow_stage1 = SingleStageModel(self.num_layers,
                                       self.num_f_maps,
                                       self.dim,
                                       self.num_classes,
                                       causal_conv=self.causal_conv)
        self.slow_stages = nn.ModuleList([
            copy.deepcopy(
                SingleStageModel(self.num_layers,
                                 self.num_f_maps,
                                 self.num_classes,
                                 self.num_classes,
                                 causal_conv=self.causal_conv))
            for _ in range(self.num_stages - 1)
        ])
        self.ws = [nn.Parameter(torch.ones(2)) for _ in range(self.num_stages)]
        self.downsample = nn.MaxPool1d(kernel_size=self.fast_frames, stride=self.fast_frames)


        
    def forward(self, input):
        length = input.shape[2]
        ##Fast path
        ##### initial stage
        fast_input = self.downsample(input)
        fast_out = self.fast_stage1(fast_input)
        up_fast_out = self.Sample(fast_out,  length)#F.interpolate(fast_out, size=length, mode='linear')#self.Sample(fast_out,  length)
        up_fast_outputs = up_fast_out.unsqueeze(0)
        
        ##### refine stage
        for s in self.fast_stages:
            fast_out = s(F.softmax(fast_out, dim=1))
            up_fast_out = self.Sample(fast_out, length)#F.interpolate(fast_out, size=length, mode='linear')#self.Sample(fast_out, length)
            up_fast_outputs = torch.cat((up_fast_outputs, up_fast_out.unsqueeze(0)), dim=0)

        ##Slow path
        ##### initial stage
        slow_out = self.slow_stage1(input)

        out = self.ws[0][0]*slow_out + self.ws[0][1] * up_fast_outputs[0]

        outputs = out.unsqueeze(0)

        ##### refine stage
        i = 1
        for s in self.slow_stages:
            slow_out = s(F.softmax(out, dim=1))
            out = self.ws[i][0]*slow_out + self.ws[i][1] * up_fast_outputs[i]
            
            outputs= torch.cat((outputs, out.unsqueeze(0)), dim=0)
            i += 1

        return outputs
    
    def Sample(self, input, out_frame):
        ## input: [1, features, in_frame] --> output: [1, features, out_frame]
        in_frame = input.shape[-1]
        inds = torch.linspace(0, in_frame - 1, out_frame)
        rounded_inds = torch.round(inds).to(int)

        return input[:, :, rounded_inds]


class MultiStageModel(nn.Module):
    def __init__(self, mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features, mstcn_causal_conv):
        self.num_stages = mstcn_stages  # 2
        self.num_layers = mstcn_layers  # 8
        self.num_f_maps = mstcn_f_maps  # 32
        self.dim = mstcn_f_dim  #2048
        self.num_classes = out_features  # 7
        self.causal_conv = mstcn_causal_conv
        print(
            f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
            f" {self.num_f_maps}, dim: {self.dim}")
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(self.num_layers,
                                       self.num_f_maps,
                                       self.dim,
                                       self.num_classes,
                                       causal_conv=self.causal_conv)
        self.stages = nn.ModuleList([
            copy.deepcopy(
                SingleStageModel(self.num_layers,
                                 self.num_f_maps,
                                 self.num_classes,
                                 self.num_classes,
                                 causal_conv=self.causal_conv))
            for s in range(self.num_stages - 1)
        ])
        self.smoothing = False

    def forward(self, x):
        out_classes = self.stage1(x)
        outputs_classes = out_classes.unsqueeze(0)
        for s in self.stages:
            out_classes = s(F.softmax(out_classes, dim=1))
            outputs_classes = torch.cat(
                (outputs_classes, out_classes.unsqueeze(0)), dim=0)
        return outputs_classes

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        mstcn_reg_model_specific_args = parser.add_argument_group(
            title='mstcn reg specific args options')
        mstcn_reg_model_specific_args.add_argument("--mstcn_stages",
                                                   default=4,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_layers",
                                                   default=10,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_f_maps",
                                                   default=64,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_f_dim",
                                                   default=2048,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mstcn_causal_conv",
                                                   action='store_true')
        return parser


class SingleStageModel(nn.Module):
    def __init__(self,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 causal_conv=False):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        self.layers = nn.ModuleList([
            copy.deepcopy(
                DilatedResidualLayer(2**i,
                                     num_f_maps,
                                     num_f_maps,
                                     causal_conv=causal_conv))
            for i in range(num_layers)
        ])
        self.conv_out_classes = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out_classes = self.conv_out_classes(out)
        return out_classes


class DilatedResidualLayer(nn.Module):
    def __init__(self,
                 dilation,
                 in_channels,
                 out_channels,
                 causal_conv=False,
                 kernel_size=3):
        super(DilatedResidualLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation = dilation
        self.kernel_size = kernel_size
        if self.causal_conv:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=(dilation *
                                                   (kernel_size - 1)),
                                          dilation=dilation)
        else:
            self.conv_dilated = nn.Conv1d(in_channels,
                                          out_channels,
                                          kernel_size,
                                          padding=dilation,
                                          dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        if self.causal_conv:
            out = out[:, :, :-(self.dilation * 2)]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)


class SingleStageModel1(nn.Module):
    def __init__(self, args,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 kernel_size,
                 causal_conv = True, dropout = False, hier = False, use_output = False):
        super(SingleStageModel1, self).__init__()
        self.use_output = use_output
        if self.use_output:
            self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        self.layers = nn.ModuleList([
            copy.deepcopy(
                DilatedResidualLayer(2**i,
                                     num_f_maps,
                                     num_f_maps,
                                     causal_conv=causal_conv))
            for i in range(num_layers)
        ])
        self.conv_out_classes = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = dropout
        self.hier = hier
        self.pool_kernel_size = kernel_size if hier else None
        if dropout:
            self.channel_dropout = nn.Dropout2d()
        if hier:
            self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size)#nn.AvgPool1d(kernel_size=args.train, stride=args.train)


    def forward(self, x):
        if self.use_output:
            out = self.conv_1x1(x)
        else:
            out = x

        if self.dropout:
            out = out.unsqueeze(3)
            out = self.channel_dropout(out)
            out = out.squeeze(3)
        
        for layer in self.layers:
            out = layer(out)
        if self.hier:
            if out.size(-1) < self.pool_kernel_size:
                raise ValueError(
                    "Hierarchical pooling kernel {} is larger than the current temporal length {}. "
                    "Lower the refinement pooling kernel (the `--train` argument in COG training scripts).".format(
                        self.pool_kernel_size,
                        out.size(-1),
                    )
                )
            f = self.pool(out)
        else:
            f = out

        out_classes = self.conv_out_classes(f)
        return f, out_classes
    
class Refinement(nn.Module):
    def __init__(self, args, num_layers, num_f_maps, num_dim, num_classes, causal_conv):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(num_dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps, causal_conv = causal_conv)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        # self.conv_out = conv_out
        self.max_pool_1x1 = nn.AvgPoold(kernel_size=7,stride=3)
        self.use_output = args.output
        self.hier = args.hier

    def forward(self, x):
        if self.use_output:
            out = self.conv_1x1(x)
        else:
            out = x
        for layer in self.layers:
            out = layer(out)
        if self.hier:
            f = self.max_pool_1x1(out)
        else:
            f = out
        out = self.conv_out(f)
        
        return f, out
    
class MultiStageModel1(nn.Module):
    def __init__(self, mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features, mstcn_causal_conv):
        self.num_stages = mstcn_stages  # 4 #2
        self.num_layers = mstcn_layers  # 10  #5
        self.num_f_maps = mstcn_f_maps  # 64 #64
        self.dim = mstcn_f_dim  #2048 # 2048
        self.num_classes = out_features  # 7
        self.causal_conv = mstcn_causal_conv
        print(
            f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
            f" {self.num_f_maps}, dim: {self.dim}")
        super(MultiStageModel1, self).__init__()
        self.stage1 = SingleStageModel1(self.num_layers,
                                       self.num_f_maps,
                                       self.dim,
                                       self.num_classes,
                                       causal_conv=self.causal_conv)
        self.stages = nn.ModuleList([
            copy.deepcopy(
                SingleStageModel1(self.num_layers,
                                 self.num_f_maps,
                                 self.num_classes,
                                 self.num_classes,
                                 causal_conv=self.causal_conv))
            for s in range(self.num_stages - 1)
        ])
        self.smoothing = False

    def forward(self, x):
        out_classes, _ = self.stage1(x)
        outputs_classes = out_classes.unsqueeze(0)
        for s in self.stages:
            out_classes, out = s(F.softmax(out_classes, dim=1))
            outputs_classes = torch.cat(
                (outputs_classes, out_classes.unsqueeze(0)), dim=0)
        return out




class COG_GVROnly(nn.Module):
    """
    Chain-of-Gesture with ONLY the Gestural-Visual Reasoning (GVR) module.

    This is the "CoG (GVR only)" baseline: the GVR transformer (`self.cot`,
    identical to the one used by `COG`) contextualises per-frame visual features
    with CLIP-encoded gesture prompts, and a single 1x1 convolution maps the
    resulting per-frame representation to class logits. The multi-scale temporal
    reasoning stack of the full CoG model (slow-path MS-TCN + refinement stages,
    fast-path pooled branch, FPN) is removed.

    `forward` returns `(out_list, f_list)` with exactly one stage each, so the
    existing CoG training/evaluation loops work unchanged.
    """

    def __init__(self, args, num_f_dim, num_classes, d_model, d_q, len_q, device,
                 gest_template='A surgeon is', gest_prompt: str = './utils/gest_prompt_B32.pt',
                 gesture_prompts=None):
        super(COG_GVROnly, self).__init__()

        self.dim = num_f_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.d_q = d_q
        self.len_q = len_q
        self.device = device
        self.gest_prompt = gest_prompt

        num_gest_f = 512

        if gesture_prompts is None:
            self.gest_list = ['reaching for needle with right hand',
                              'positioning needle',
                              'pushing needle through tissue',
                              'transferring needle from left to right',
                              'moving to center with needle in grip',
                              'pulling suture with left hand',
                              'pulling suture with right hand',
                              'orienting needle',
                              'using right hand to help tighten suture',
                              'loosening more suture',
                              'dropping suture at end and movign to end points',
                              'reaching for needle with left hand',
                              'making C loop around right hand',
                              'reaching for suture with right hand',
                              'pulling suture with both hands'
                              ]
            prompt_texts = [f'{gest_template} {gesture} ...' for gesture in self.gest_list]
        else:
            self.gest_list = list(gesture_prompts)
            prompt_texts = [
                prompt if prompt.endswith('...') else f'{prompt} ...'
                for prompt in self.gest_list
            ]

        text_model, text_preprocess = clip.load('ViT-B/32', device='cpu')
        num_gest = len(self.gest_list)
        self.num_gestures = num_gest
        self.all_gest_fea = torch.zeros(1, num_gest_f)
        for prompt_text in prompt_texts:
            gest_prompt_fea = text_model.encode_text(clip.tokenize(prompt_text)).float()
            self.all_gest_fea = torch.cat((self.all_gest_fea, gest_prompt_fea), dim=0)
        self.all_gest_fea = self.all_gest_fea[1:]
        torch.save(self.all_gest_fea, self.gest_prompt)
        self.all_action_fea = nn.Parameter(torch.load(self.gest_prompt), requires_grad=False)

        # The GVR module itself (identical to COG's `self.cot`).
        self.cot = MyTransformer(self.dim, num_gest_f, d_model, d_q, len_q, device)
        # Minimal classification head on top of the GVR representation.
        self.conv_out = nn.Conv1d(num_gest * d_model, num_classes, 1)

    def forward(self, x):
        # x: visual features [1, T, feature_dim]
        all_action_fea = self.all_action_fea.unsqueeze(0)     # [1, num_gest, 512]
        xx = self.cot(all_action_fea, x)                      # [1, T, num_gest * d_model]
        xx = xx.permute(0, 2, 1)                              # [1, num_gest * d_model, T]
        out = self.conv_out(xx)                               # [1, num_classes, T]
        return [out], [xx]
