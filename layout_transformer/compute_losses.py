#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:04:57 2022

@author: dipu
"""
from cmath import log
from distutils.command.build_scripts import first_line_re
import os
import argparse
import torch
from dataset import MNISTLayout, JSONLayout, RICO_Seq_Box_RandRef, RICOLayout, RICOLayout_withImage, RICO_Seq_Box, RICO_Seq_Box_RandRef
#from layout_transformer.dataset import RICOLayout
from model import GPT_conditional, GPTConfig
from trainer_conditional import Trainer, TrainerConfig
from utils import set_seed, sample
import sys
import pickle
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F


sys.path.append('/home/dipu/dipu_ps/codes/UIGeneration/VTN')
from model_MLP_AE2_old import Simple_MLPAE

def pickle_save(fn,obj):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)
        print(f'obj saved to {fn}')

def pickle_load(fn):
    with open(fn,'rb') as f:
        obj = pickle.load(f)
    print (f'Object loaded from {fn}')    
    return obj

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count         
        
        

#%%
def argument_parser():
    parser = argparse.ArgumentParser('Layout Transformer')
    parser.add_argument("--exp", default="layout", help="mnist_threshold_1")
    parser.add_argument("--log_dir", default="./runs", help="/path/to/logs/dir")
    parser.add_argument("--device", type=str, default='cuda:1')
    
    
    parser.add_argument("--dataset_name", type=str, default='RICO_Seq_Box', choices=['RICO', 'MNIST', 'COCO', 'PubLayNet'])
    # MNIST options
    parser.add_argument("--data_dir", default=None, help="")
    parser.add_argument("--threshold", type=int, default=1, help="threshold for grayscale values")
    
    # COCO/PubLayNet options
    parser.add_argument("--train_json", default="data/coco/instances_train2014.json", help="/path/to/train/json")
    parser.add_argument("--val_json", default="data/coco/instances_val2014.json", help="/path/to/val/json")
    
    # RICO Options
    parser.add_argument("--rico_info", default='data/rico.pkl')
    parser.add_argument('--train_ann', type=str, default='/home/dipu/dipu_ps/codes/UIGeneration/VTN/data/mtn50_ECCV_train_data.pkl')
    parser.add_argument('--gallery_ann', type=str, default='/home/dipu/dipu_ps/codes/UIGeneration/VTN/data/mtn50_ECCV_gallery_data.pkl')
    parser.add_argument('--query_ann', type=str, default='/home/dipu/dipu_ps/codes/UIGeneration/VTN/data/mtn50_ECCV_query_data.pkl')
   
    # Layout options
    parser.add_argument("--max_length", type=int, default=128, help="batch size")
    parser.add_argument('--precision', default=8, type=int)
    parser.add_argument('--element_order', default='raster')
    parser.add_argument('--attribute_order', default='cxywh')
    
    # Architecture/training options
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=4.5e-06, help="learning rate")
    parser.add_argument('--n_layer', default=6, type=int)
    parser.add_argument('--n_embd', default=512, type=int)
    parser.add_argument('--n_head', default=8, type=int)
    # parser.add_argument('--evaluate', action='store_true', help="evaluate only")
    parser.add_argument('--lr_decay', action='store_true', help="use learning rate decay")
    parser.add_argument('--warmup_iters', type=int, default=0, help="linear lr warmup iters")
    parser.add_argument('--final_iters', type=int, default=0, help="cosine lr final iters")
    parser.add_argument('--sample_every', type=int, default=1, help="sample every epoch")
    
    args = parser.parse_args()
    return args

#%%
args = argument_parser()
set_seed(args.seed)
device = args.device

#Dataset
train_dataset = RICO_Seq_Box(args.rico_info, args.train_ann,  max_length=None, precision=args.precision)
valid_dataset = RICO_Seq_Box(args.rico_info, args.gallery_ann,  max_length=train_dataset.max_length, precision=args.precision)
query_dataset = RICO_Seq_Box(args.rico_info, args.query_ann,  max_length=train_dataset.max_length, precision=args.precision)
    
# MLPAE model
trained_path = '/home/dipu/dipu_ps/codes/UIGeneration/VTN/runs/SimpleMLPAE/attention_aggregator/SimpleMLPAE_25DecAtLast0_lr0.01_bs256_tanD0_HM_1_UBL_0_SW1.0_BW50.0_EW1.0_RW10.0_AttEnc_attention_BoxLMSE_Actrelu_dim512'
config_fn = f'{trained_path}/config.pkl'
config_mlp = pickle_load(config_fn)
search_model = Simple_MLPAE(config_mlp)


#Paths
log_dir = os.path.join(args.log_dir, args.dataset_name)
samples_dir = os.path.join(log_dir, "samples")
ckpt_dir = os.path.join(log_dir, "checkpoints")


#Seq Model
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_length,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      conditional=True, is_causal=True, activation_function= 'gelu')  # a GPT-1
model = GPT_conditional(mconf)
model.to(device)


#%%
model_name = 'model_bs16_lr0.001_evry7_fzsearch_False_searchLoss_False'
ep=13

pt_model_path = f'{log_dir}/{model_name}/checkpoints/checkpoint_{ep}.pth'
pt_model = torch.load(pt_model_path, map_location=device)
pt_syn_model = pt_model['model_synthesis'] # sythesis model
pt_search_model = pt_model['model_search'] # search model   
model.load_state_dict(pt_syn_model)
search_model.load_state_dict(pt_search_model)
search_model = search_model.to(device)

search_model.eval()
model.eval()


def compute_loss(loader):
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        pbar = enumerate(loader)
        for it, data_loaded in pbar:
            x, y, uxid, boxes, box_exists, n_boxes, label_ids = data_loaded
        
            boxes, box_exists  = boxes.to(device), box_exists.to(device)
            label_ids = label_ids.to(device)
            x = x.to(device)
            y = y.to(device)    
            
            # Get latent vector for condition
            _ , z = search_model.encoder(boxes, box_exists, n_boxes, label_ids)
            _, loss = model(x, z, y, pad_token=pad_token)
            loss_meter.update(loss.item())
    
    return loss_meter.avg


pad_token = train_dataset.vocab_size - 1
# Dataloader on validation set
valloader = DataLoader(valid_dataset, shuffle=False, pin_memory=True,
                drop_last=False, batch_size=args.batch_size, num_workers=4)
trainloader = DataLoader(train_dataset, shuffle=False, pin_memory=True,
                drop_last=False, batch_size=args.batch_size, num_workers=4)


avg_trainloss = compute_loss(trainloader)
avg_valloss = compute_loss(valloader)

result_file = f'{log_dir}/{model_name}/result.txt'
with open(result_file, 'a') as f:
    f.write (f'\n\n{model_name}\n')
    f.write(f'Average Training Loss:   {avg_trainloss:.3f}\t num: {len(train_dataset)}\n')
    f.write(f'AVerage Validation Loss: {avg_valloss:.3f}\t num:{len(valid_dataset)}\n')
