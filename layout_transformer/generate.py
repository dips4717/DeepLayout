#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 11:59:49 2021

@author: dipu
"""

import os
import argparse
import torch
from dataset import MNISTLayout, JSONLayout, RICOLayout
#from layout_transformer.dataset import RICOLayout
from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from utils import set_seed
from torch.utils.data.dataloader import DataLoader

from utils import sample
import torch.nn.functional as F

def argument_parser():
    parser = argparse.ArgumentParser('Layout Transformer')
    parser.add_argument("--exp", default="layout", help="mnist_threshold_1")
    parser.add_argument("--log_dir", default="./runs", help="/path/to/logs/dir")
    
    
    parser.add_argument("--dataset_name", type=str, default='RICO', choices=['RICO', 'MNIST', 'COCO', 'PubLayNet'])
    # MNIST options
    parser.add_argument("--data_dir", default=None, help="")
    parser.add_argument("--threshold", type=int, default=1, help="threshold for grayscale values")
    
    # COCO/PubLayNet options
    parser.add_argument("--train_json", default="data/coco/instances_train2014.json", help="/path/to/train/json")
    parser.add_argument("--val_json", default="data/coco/instances_val2014.json", help="/path/to/val/json")
    
    # RICO Options
    parser.add_argument("--rico_info", default='data/rico.pkl')
        
    # Layout options
    parser.add_argument("--max_length", type=int, default=128, help="batch size")
    parser.add_argument('--precision', default=8, type=int)
    parser.add_argument('--element_order', default='raster')
    parser.add_argument('--attribute_order', default='cxywh')
    
    # Architecture/training options
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
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


args = argument_parser()
set_seed(args.seed)
device = 'cuda:0'
args.exp_name = 'runs/LayoutTransformer/layoutransformer_bs40_lr0.001_lrdecay10'
split = 'test' # 'test'

save_dir_gen = f'{args.exp_name}/{split}_gen_images'
if not os.path.exists(save_dir_gen):
    os.makedirs(save_dir_gen, exist_ok=True)



pt_model_path = f'{args.exp_name}/checkpoints/checkpoint_best.pth'
pt_model = torch.load(pt_model_path, map_location=device)
epoch = pt_model['epoch']
test_loss = pt_model['epoch']
print(f'Epoch: {epoch} \ntest_loss: {test_loss}')


log_dir = os.path.join(args.log_dir, 'LayoutTransformer', args.exp_name )
samples_dir = os.path.join(log_dir, "samples")
ckpt_dir = os.path.join(log_dir, "checkpoints")

train_dataset = RICOLayout(args.rico_info, split='train', precision=args.precision)
valid_dataset = RICOLayout(args.rico_info, split='test', max_length=train_dataset.max_length, precision=args.precision)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_length,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)  # a GPT-1
model = GPT(mconf)
model.to(device)
tconf = TrainerConfig(max_epochs=args.epochs,
                      batch_size=args.batch_size,
                      lr_decay=args.lr_decay,
                      learning_rate=args.lr * args.batch_size,
                      warmup_iters=args.warmup_iters,
                      final_iters=args.final_iters,
                      ckpt_dir=ckpt_dir,
                      samples_dir=samples_dir,
                      sample_every=args.sample_every)


model.load_state_dict(pt_model['state_dict'])

dataset_ = train_dataset if split =='train' else valid_dataset
loader = DataLoader(dataset_, shuffle=True, pin_memory=True,
                batch_size=tconf.batch_size,
                num_workers=tconf.num_workers)

pbar = enumerate(loader)
for it, (x, y) in pbar:
    if it == 100:
        break
    
    x_cond = x.to(device) 
      
    gt_layouts= x.detach().cpu().numpy()
    gt_layouts = [train_dataset.render(layout) for layout in gt_layouts]
    
    # Reconstruction
    logits, _ = model(x_cond)
    probs = F.softmax(logits, dim=-1)
    _, y = torch.topk(probs, k=1, dim=-1)
    layouts_recon = torch.cat((x_cond[:, :1], y[:, :, 0]), dim=1).detach().cpu().numpy()
    recon_layouts = [train_dataset.render(layout) for layout in layouts_recon]

    # Generation with First element
    layouts_firstel = sample(model, x_cond[:, :6], steps=train_dataset.max_length,                                temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
    layouts_firstel = [train_dataset.render(layout) for layout in layouts_firstel] 
    

    # Generation with bos only
    layouts_bos = sample(model, x_cond[:, :1], steps=train_dataset.max_length,
                                temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
    layouts_bos = [train_dataset.render(layout) for layout in layouts_bos] 

    # Generation with partial
    layouts_partial = sample(model, x_cond[:, :26], steps=train_dataset.max_length, 
                                temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
    layouts_partial = [train_dataset.render(layout) for layout in layouts_partial] 


    #Plot in a simple figure
    from matplotlib import pyplot as plt
    
    fig, ax = plt.subplots(1,5, figsize=(25, 12), constrained_layout=True)
    plt.setp(ax,  xticklabels=[],  yticklabels=[])
    ax[0].imshow(gt_layouts[0])
    ax[0].set_title('GroundTruth')

    ax[1].imshow(recon_layouts[0])
    ax[1].set_title('Reconstruction')

    ax[2].imshow(layouts_bos[0])
    ax[2].set_title('Gen-BoS')

    ax[3].imshow(layouts_firstel[0])
    ax[3].set_title('Gen-FirstEle.')


    ax[4].imshow(layouts_partial[0])
    ax[4].set_title('Gen-Partial-(5 Elem.)')

    plt.savefig(f'{save_dir_gen}/{str(it)}_plot.png')
    
   










