#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 00:39:01 2022

@author: dipu
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


# encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
# transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
# src = torch.rand(502, 32, 512)
# out = transformer_encoder(src)

# out = out.permute(1,0,2)

# #
# sum_pool = torch.sum(out,dim=1) # 32 x 512

# #
# avg_pool = torch.sum(out,dim=1) / out.shape[1]

# # first out
# first_out = out[:,0,:]

# # MLP AGGE
# class MLPagg(nn.Module):   
#     def __init__(self):
#         super(MLPagg, self).__init__()
#         self.proj = nn.Linear(512,10)
#         self.linear = nn.Linear(10*502,512)
       
#     def forward(self,x):
#         x = F.relu(self.proj(x))
#         x = x.reshape(x.shape[0],-1)
#         x = self.linear(x)
#         return x
    
# mlpagg = MLPagg()
# mlpout = mlpagg(out)

# ##

# class Attention(nn.Module):
#     def __init__(self):
#         super(Attention, self).__init__()
#         self.alpha_net = nn.Linear(512, 1)
#         self.proj = nn.Linear(512,512)
        
        
#     def forward(self, att_feats,  att_masks=None):   # att_masks --> BxN
#         # The p_att_feats here is already projected
#         att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
#         dot = att_feats                                     # B X N x hidden_size
#         dot = self.proj(dot)
#         dot = torch.tanh(dot)                                # B X N x hidden_size
#         dot = dot.view(-1, 512)               #  (B*N) x hidden_size
#         dot = self.alpha_net(dot)                           #  (B*N) x 1
#         dot = dot.view(-1, 512)                        # B x N

#         weight = F.softmax(dot, dim=1)                       # B X N
#         if att_masks is not None:
#             weight = weight * att_masks.view(-1, att_size).float()
#             weight = weight / weight.sum(1, keepdim=True)                #  BxN 
#         att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))    # B x N x hidden_size
#         att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # (B x1xN)  , (BxN,hidden_size)
#         return att_res

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


aa = torch.rand(32,502,512)
agg = Attention()
out = agg(aa, (32,502))   