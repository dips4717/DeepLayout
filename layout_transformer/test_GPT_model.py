#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 12:12:28 2022

@author: dipu
"""

from model import GPT, GPTConfig

mconf = GPTConfig(283, 100,
                  n_layer=2, n_head=8, n_embd=512)  # a GPT-1
model = GPT(mconf)


#%%
from gpt2_encoder import  GPT2Encoder

model2 = gpt2_encoder()

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)