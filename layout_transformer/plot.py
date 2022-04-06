#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:45:16 2022

@author: dipu
"""

from matplotlib import pyplot as plt
from utils_dips import pickle_load
import torch

file = 'runs/ConditionalLayoutTransformer/Condlayoutransformer_bs40_lr0.001_lrdecay10_FzSMFalse_SL_True/results/result_seq_acc.pkl'
result = pickle_load(file)

test_acc = result['1']
tesr_masks = result['2']

import numpy as np

test_acc = torch.stack(test_acc)
test_acc = test_acc.astype(torch.float32)
mean_test_acc = test_acc.sum(dim=0)

