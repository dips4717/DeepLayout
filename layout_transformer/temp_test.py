#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:20:24 2022

@author: dipu
"""

import torch
pad_token = 99
seq= torch.randint(0, 53, (5,15,), dtype=torch.long, device='cuda:1')

b = seq.shape[0]
t = seq.shape[1]
drop_rate = torch.rand(b)
drop_index = (drop_rate*t).type(torch.long)
drop_index = torch.clip(drop_index, min=1)  # prevent dropping bos  

for ii, tt in enumerate(drop_index):
    seq[ii,tt:-1] = torch.tensor(pad_token, dtype=torch.long, device=seq.device)
