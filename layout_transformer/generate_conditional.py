#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 11:04:57 2022

@author: dipu
"""
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

#%%
def argument_parser():
    parser = argparse.ArgumentParser('Layout Transformer')
    
    parser.add_argument("--log_dir", default="./runs", help="/path/to/logs/dir")   
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

#%%
args = argument_parser()
set_seed(args.seed)
device = 'cuda:0'

#Dataset
train_dataset = RICO_Seq_Box(args.rico_info, args.train_ann,  max_length=None, precision=args.precision)
valid_dataset = RICO_Seq_Box(args.rico_info, args.gallery_ann,  max_length=train_dataset.max_length, precision=args.precision)
query_dataset = RICO_Seq_Box(args.rico_info, args.query_ann,  max_length=train_dataset.max_length, precision=args.precision)
    
train_dataset2 = RICO_Seq_Box_RandRef(args.rico_info, args.train_ann,  max_length=train_dataset.max_length, precision=args.precision)
valid_dataset2 = RICO_Seq_Box_RandRef(args.rico_info, args.gallery_ann,  max_length=train_dataset.max_length, precision=args.precision)


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
tconf = TrainerConfig(max_epochs=args.epochs,
                      batch_size=args.batch_size,
                      lr_decay=args.lr_decay,
                      learning_rate=args.lr * args.batch_size,
                      warmup_iters=args.warmup_iters,
                      final_iters=args.final_iters,
                      ckpt_dir=ckpt_dir,
                      samples_dir=samples_dir,
                      sample_every=args.sample_every)

# model_name = 'model_bs16_lr4.5e-06_evry7_fzsearch_True_searchLoss_False'
# # model_name = 'model_bs16_lr4.5e-06_evry7_fzsearch_False_searchLoss_False'
#pt_model_path = f'{log_dir}/{model_name}/checkpoints/checkpoint_{ep}.pth'

pt_model_path = 'runs/ConditionalLayoutTransformer/Condlayoutransformer_bs40_lr0.001_lrdecay10_FzSMFalse_SL_False'

split = 'train' # 'test'
pair = 'different'
save_dir_gen = f'{pt_model_path}/{split}_gen_images_pair_{pair}'
if not os.path.exists(save_dir_gen):
    os.makedirs(save_dir_gen, exist_ok=True)



pt_model = torch.load(f'{pt_model_path}/checkpoints/checkpoint_best.pth', map_location=device)
pt_syn_model = pt_model['model_synthesis']
pt_search_model = pt_model['model_search']
#print(pt_model.keys())
epoch = pt_model['epoch']
train_loss = pt_model['train_loss']
test_loss = pt_model['test_loss']
print(f'TrainLoss: {train_loss}\nTestLoss: {test_loss}\nEpcoh: {epoch}  ')


model.load_state_dict(pt_syn_model)
search_model.load_state_dict(pt_search_model)
search_model = search_model.to(device)


# Dataloader on validation set
if pair == 'same':
    dataset_ = train_dataset if split =='train' else valid_dataset
    loader = DataLoader(dataset_, shuffle=False, pin_memory=True,
                    batch_size=tconf.batch_size,
                    num_workers=tconf.num_workers)
else:
    dataset_ = train_dataset2 if split =='train' else valid_dataset2
    loader = DataLoader(dataset_, shuffle=False, pin_memory=True,
                    batch_size=tconf.batch_size,
                    num_workers=tconf.num_workers)



# #%%
with torch.no_grad():
    pbar = enumerate(loader)
    for it, data_loaded in pbar:
        if pair == 'same':
            x, y, uxid, boxes, box_exists, n_boxes, label_ids = data_loaded
        else:
            x, y, uxid, ref_uxid, ref_x, boxes, box_exists, n_boxes, label_ids = data_loaded
        
        if it == 100:
            break
        boxes, box_exists  = boxes.to(device), box_exists.to(device)
        label_ids = label_ids.to(device)    
        
        # Get latent vector for condition
        _ , z = search_model.encoder(boxes, box_exists, n_boxes, label_ids)
        print(it)
   
        x_cond = x.to(device) 
      
        gt_layouts= x.detach().cpu().numpy()
        gt_layouts = [train_dataset.render(layout) for layout in gt_layouts]

        if pair == 'different':
            ref_layouts= ref_x.detach().cpu().numpy()
            ref_layouts = [train_dataset.render(layout) for layout in ref_layouts]
        else:
            ref_layouts = gt_layouts


        # Reconstructed layouts
        logits, _ = model(x_cond, z)
        probs = F.softmax(logits, dim=-1)
        _, y = torch.topk(probs, k=1, dim=-1)
        layouts_recon = torch.cat((x_cond[:, :1], y[:, :, 0]), dim=1).detach().cpu().numpy()
        recon_layouts = [train_dataset.render(layout) for layout in layouts_recon]

        # Generation with First element
        layouts_firstel = sample(model, x_cond[:, :6], steps=train_dataset.max_length, z=z,
                                    temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
        layouts_firstel = [train_dataset.render(layout) for layout in layouts_firstel] 

        # Generation with bos only
        layouts_bos = sample(model, x_cond[:, :1], steps=train_dataset.max_length, z=z,
                                    temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
        layouts_bos = [train_dataset.render(layout) for layout in layouts_bos] 

        # Generation with partial
        layouts_partial = sample(model, x_cond[:, :26], steps=train_dataset.max_length, z=z,
                                    temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
        layouts_partial = [train_dataset.render(layout) for layout in layouts_partial] 


        #Plot in a simple figure
        from matplotlib import pyplot as plt
        
        fig, ax = plt.subplots(1,6, figsize=(30, 12), constrained_layout=True)
        plt.setp(ax,  xticklabels=[],  yticklabels=[])
        ax[0].imshow(gt_layouts[0])
        ax[0].set_title('GroundTruth')

        ax[1].imshow(ref_layouts[0])
        ax[1].set_title('Condition/Reference')

        ax[2].imshow(recon_layouts[0])
        ax[2].set_title('Reconstruction')

        ax[3].imshow(layouts_bos[0])
        ax[3].set_title('Gen-BoS')

        ax[4].imshow(layouts_firstel[0])
        ax[4].set_title('Gen-FirstEle.')


        ax[5].imshow(layouts_partial[0])
        ax[5].set_title('Gen-Partial-(5 Elem.)')

        plt.savefig(f'{save_dir_gen}/{str(uxid[0].numpy())}_plot.png')
