import torch.nn as nn
from model import GPT_conditional
import torch
import math
from torch import Tensor
import torch.nn.functional as F



class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.att_hid_size = 512
        self.alpha_net = nn.Linear(512, 1)
        self.proj = nn.Linear(self.att_hid_size, self.att_hid_size)

    def forward(self, att_feats,  att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        dot = self.proj(att_feats)
        dot = torch.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size

        weight = F.softmax(dot, dim=1)                       # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size
        return att_res

# MLP AGGE
class MLPagg(nn.Module):   
    def __init__(self):
        super(MLPagg, self).__init__()
        self.proj = nn.Linear(512,10)
        self.linear = nn.Linear(10*502,512)  # 501 # maxlength =502
       
    def forward(self,x):
        x = F.relu(self.proj(x))
        x = x.reshape(x.shape[0],-1)
        x = self.linear(x)
        return x


class TransformerEncoder_GPTConditional(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder_GPTConditional, self).__init__()
        self.config = config
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.n_embd, 
                                                   nhead=config.n_head,
                                                   dropout=0.1,
                                                   dim_feedforward= 512)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layer)
        self.decoder =  GPT_conditional(config)  
        
        self.src_tok_emb = TokenEmbedding(config.vocab_size, config.n_embd)
        self.src_positional_encoding = PositionalEncoding(config.n_embd, dropout=0.1)
        # Note here GPT based decoder has its own token embedder and position encoding modules.

        if config.agg_type == 'Attention':
            self.agg = Attention()
        elif config.agg_type == 'MLP':
            self.agg = MLPagg()
        else:
            pass
        
        
    def create_src_mask(self,src, PAD_IDX, device):
        src_seq_len = src.shape[0]
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)
        src_padding_mask = (src == PAD_IDX).transpose(0, 1)
        
        return src_mask, src_padding_mask        

    def forward(self, seq, pad_token=-100):
        device = seq.device
        x = seq[:,:-1]
        y = seq[:,1:]
        src = x.t()
        
        src_mask, src_padding_mask = self.create_src_mask(src, pad_token, device)
        src_emb = self.src_positional_encoding(self.src_tok_emb(src))
        memory = self.encoder(src_emb, src_mask, src_padding_mask)
        memory = memory.permute(1,0,2)  # convert into batch first 
        
        if self.config.agg_type ==  'MLP':
            z = self.agg(memory)
        elif self.config.agg_type ==  'Attention':    
             z = self.agg(memory, att_masks = torch.logical_not(src_padding_mask))
        elif self.config.agg_type == 'AveragePool':
            z = torch.sum(memory,dim=1) / memory.shape[1]
        elif self.config.agg_type == 'FirstOut':
            z = memory[:,0,:]
        else:
            print (f'Aggregation type {self.config.agg_type} Not Implemented')
            
        logits, loss = self.decoder(x, z, target=y, pad_token=pad_token)
        return logits, loss ,z

    def forward_sample(self, seq, seq_all, pad_token=-100):
        device = seq.device
        assert(pad_token != -100)   
        x = seq
        src = seq_all[:,:-1].t()
        
        
        src_mask, src_padding_mask = self.create_src_mask(src, pad_token, device)
        src_emb = self.src_positional_encoding(self.src_tok_emb(src))
        memory = self.encoder(src_emb, src_mask, src_padding_mask)
        memory = memory.permute(1,0,2)  # convert into batch first 
        
        if self.config.agg_type ==  'MLP':
            z = self.agg(memory)
        elif self.config.agg_type ==  'Attention':    
             z = self.agg(memory, att_masks = torch.logical_not(src_padding_mask))
        elif self.config.agg_type == 'AveragePool':
            z = torch.sum(memory,dim=1) / memory.shape[1]
        elif self.config.agg_type == 'FirstOut':
            z = memory[:,0,:]
        else:
            print (f'Aggregation type {self.config.agg_type} Not Implemented')
            
            
        
        logits, loss = self.decoder(x, z, target=None, pad_token=pad_token)
        return logits, loss, z

