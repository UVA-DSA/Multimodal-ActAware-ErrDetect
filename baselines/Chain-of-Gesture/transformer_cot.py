import torch
import numpy as np
import torch.nn as nn
import math
import pdb

# some code adapted from https://wmathor.com/index.php/archives/1455/


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, n_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.n_heads = n_heads

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]

        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_heads, len_q, len_q]
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_q, n_heads, gpu_id):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = nn.Linear(d_model, d_q * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_q * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_q * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_q, d_model, bias=False)

        self.d_model = d_model
        self.d_q = d_q
        self.n_heads = n_heads
        self.ScaledDotProductAttention = ScaledDotProductAttention(self.d_q, n_heads)
        self.gpu_id = gpu_id

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_q).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]

        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_q).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]

        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_q).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = self.ScaledDotProductAttention(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_q)  # context: [batch_size, len_q, n_heads * d_v]
        output = context  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).to(self.gpu_id)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, gpu_id):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model*4, bias=False),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model, bias=False)
        )
        self.d_model = d_model
        self.gpu_id = gpu_id

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).to(self.gpu_id)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_q, n_heads, gpu_id):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        

        self.enc_self_attn = MultiHeadAttention(d_model, d_q, n_heads, gpu_id)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, gpu_id)

    def forward(self, Q, K, V):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        Q = self.norm1(Q)
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(Q, K, V)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(self.norm3(enc_outputs))  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn



# d_model,   Embedding Size
# d_ff, FeedForward dimension
# d_k = d_v,   dimension of K(=Q), V
# n_layers,   number of Encoder of Decoder Layer
# n_heads,   number of heads in Multi-Head Attention
class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, n_layers, n_heads, gpu_id):
        super(Encoder, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, d_k, n_heads, gpu_id) for _ in range(n_layers)])

    def forward(self, visual, text):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        visual = self.norm(visual)
        for layer in self.layers:
            text, attn = layer(text, visual, visual)

        return text, attn

    
class Transformer_cot(nn.Module):
    def __init__(self, d_model, d_ff, d_q, n_heads, gpu_id):
        super(Transformer_cot, self).__init__()
        self.layer1 = EncoderLayer(d_model, d_ff, d_q, n_heads, gpu_id).to(gpu_id)
        self.layer2 = ScaledDotProductAttention(d_q, n_heads).to(gpu_id)

    def forward(self, a, f):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(gpu_id)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.layer1(a, f, f) # [batch_size, src_len, d_model]
        dec_outputs, dec_attns = self.layer2(enc_outputs, a, a) #[batch_size, src_len, d_model]
        return dec_outputs




class TransformerCOT(nn.Module):
    def __init__(self, d_model, d_ff, d_q, n_layers, n_heads, gpu_id):
        super(TransformerCOT, self).__init__()
        self.layer1 = Encoder(d_model, d_ff, d_q, n_layers, n_heads, gpu_id).to(gpu_id)
        #self.layer2 = ScaledDotProductAttention(d_q, n_heads).to(gpu_id)
        self.atten = MultiHeadAttention(d_model = d_model, d_q = d_model, n_heads=1, gpu_id=gpu_id)

    def forward(self, visual, text):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(gpu_id)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_cross_attns = self.layer1(visual, text) # [batch_size, src_len, d_model]
        dec_outputs, dec_attns = self.atten(enc_outputs, text, text) #[batch_size, src_len, d_model]
        return dec_outputs

class TransformerCOT_video(nn.Module):
    def __init__(self, d_model, d_ff, d_q, n_layers, n_heads, gpu_id):
        super(TransformerCOT_video, self).__init__()
        self.layer1 = Encoder(d_model, d_ff, d_q, n_layers, n_heads, gpu_id).to(gpu_id)
        #self.layer2 = ScaledDotProductAttention(d_q, n_heads).to(gpu_id)
        self.atten = MultiHeadAttention(d_model = d_model, d_q = d_model, n_heads=1, gpu_id=gpu_id)

    def forward(self, visual, text, q1):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(gpu_id)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_cross_attns = self.layer1(visual, q1) # [batch_size, src_len, d_model]
        dec_outputs, dec_attns = self.atten(enc_outputs, text, text) #[batch_size, src_len, d_model]
        return dec_outputs
    
# d_model,   Embedding Size
# d_ff, FeedForward dimension
# d_q = d_k = d_v,   dimension of K(=Q), V
# n_layers,   number of Encoder of Decoder Layer
# n_heads,   number of heads in Multi-Head Attention
# len_q: length of sequence
    
class MyTransformer(nn.Module):
    def __init__(self, f_dim, gest_f_dim, d_model, d_q, len_q, gpu_id):
        super(MyTransformer, self).__init__()
        
        self.dim = f_dim  # 2048
        self.dim2 = gest_f_dim # 512
        self.len_q = len_q
        self.d_model = d_model
        self.d_q = d_q
        self.gpu_id = gpu_id

        self.linear1 = nn.Linear(f_dim, d_model, bias=False)
        self.linear2 = nn.Linear(gest_f_dim, d_model, bias=False)
        self.transformer = TransformerCOT(d_model = d_model, d_ff= f_dim,  d_q = d_q,
                                        n_heads = 8, n_layers = 2, gpu_id = gpu_id)

        #self.fc1 = nn.Linear(gest_f_dim, out_features, bias = False)
        #self.fc2 = nn.Linear(gest_f_dim, gest_features, bias = False)
        #self.linear = nn.Linear(gest_features*d_model, num_f_maps)


    def forward(self, g, long_feature):
        # g: gesture prompt [15, 768]
        # long_feature: visual feature [1, total_frame, 2048]
        visual = self.linear1(long_feature) # 1, 345, 2048 -> 1, 345, d_model
        text = self.linear2(g) #1, 15, 768 -> 1, 15, d_model

        inputs = []
        frame_length = visual.size(1)
        for i in range(frame_length):
            if i<self.len_q-1:
                input = torch.zeros((1, self.len_q-1-i, self.d_model)).to(self.gpu_id)
                input = torch.cat([input, visual[:, 0:i+1]], dim=1)
            else:
                input = visual[:, i-self.len_q+1:i+1]
            inputs.append(input)
        visual_feas = torch.stack(inputs, dim=0).squeeze(1) # [total_frame, len_q, d_model] = [345, 30, d_model]

        text_feas = [text for _ in range(frame_length)]
        text_feas = torch.stack(text_feas, dim = 0).squeeze(1) # [total_frame, num_gest, num_dim] = [345, 15, d_model]
        

        output = self.transformer(visual_feas, text_feas) #[345,15,d_model]

        outputs = output.reshape(frame_length, -1) 
        #output1 = self.linear(outputs) #[345, num_f_maps]
        #output2 = self.fc22(outputs) #[345,15]
        #output = output.transpose(1,2)
        #output = self.fc(output)
        return outputs.unsqueeze(0)#, output2.unsqueeze(0) [1, 345, num_gest * d_model]

class MyTransformer_video(nn.Module):
    def __init__(self, f_dim, gest_f_dim, d_model, d_q, len_q, gpu_id):
        super(MyTransformer_video, self).__init__()
        
        self.dim = f_dim  # 2048
        self.dim2 = gest_f_dim # 512
        self.len_q = len_q
        self.d_model = d_model
        self.d_q = d_q
        self.gpu_id = gpu_id

        self.linear1 = nn.Linear(f_dim, d_model, bias=False)
        self.linear2 = nn.Linear(gest_f_dim, d_model, bias=False)
        self.transformer = TransformerCOT_video(d_model = d_model, d_ff= f_dim,  d_q = d_q,
                                        n_heads = 8, n_layers = 2, gpu_id = gpu_id)

        #self.fc1 = nn.Linear(gest_f_dim, out_features, bias = False)
        #self.fc2 = nn.Linear(gest_f_dim, gest_features, bias = False)
        #self.linear = nn.Linear(gest_features*d_model, num_f_maps)


    def forward(self, g, long_feature):
        # g: gesture prompt [15, 768]
        # long_feature: visual feature [1, total_frame, 2048]
        visual = self.linear1(long_feature) # 1, 345, 2048 -> 1, 345, d_model
        text = self.linear2(g) #1, 345, 2048 -> 1, 345, d_model

        inputs = []
        frame_length = visual.size(1)
        for i in range(frame_length):
            if i<self.len_q-1:
                input = torch.zeros((1, self.len_q-1-i, self.d_model)).to(self.gpu_id)
                input = torch.cat([input, visual[:, 0:i+1]], dim=1)
            else:
                input = visual[:, i-self.len_q+1:i+1]
            inputs.append(input)
        visual_feas = torch.stack(inputs, dim=0).squeeze(1) # [total_frame, len_q, d_model] = [345, 30, d_model]

        q1 = visual.transpose(0,1) #[345, 1, num_dim]
        #text_feas = text.transpose(0,1) #[345, 1, num_dim]
        text_feas = [text for _ in range(frame_length)]
        text_feas = torch.stack(text_feas, dim = 0).squeeze(1) # [total_frame, num_gest, num_dim] = [345, 15, d_model]
        

        output = self.transformer(visual_feas, text_feas, q1) #[345,15,d_model]

        outputs = output.reshape(frame_length, -1) 
        #output1 = self.linear(outputs) #[345, num_f_maps]
        #output2 = self.fc22(outputs) #[345,15]
        #output = output.transpose(1,2)
        #output = self.fc(output)
        return outputs.unsqueeze(0)#, output2.unsqueeze(0) [1, 345, num_gest * d_model]
