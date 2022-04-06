#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:30:48 2022

@author: dipu
"""

from utils_dips import pickle_load, pickle_save

model_path = 'runs/ConditionalLayoutTransformer/Condlayoutransformer_bs40_lr0.001_lrdecay10_FzSMFalse_SL_True'
result_file = f'{model_path}/results/gen_result_pairdifferent_splittest.pkl'

results = pickle_load(result_file)

uxids = list(results.keys())

result = results[uxids[0]]

boxes = result['bbb']
boxes = boxes/256

xl = boxes[:,0]           
yl = boxes[:,1]  
xr = xl+boxes[:,2] 
yr = yl + boxes[:,3] 
xc = (xl + xr)/2
yc = (yl + yr)/2

all_aligns = [xl , yl , xc, yc, xr, yr]


