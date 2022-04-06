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
from dataset import RICO_Seq_Box_SimilarRef, RICOLayout,  RICO_Seq_Box, RICO_Seq_Box_RandRef
#from layout_transformer.dataset import RICOLayout
from model import GPT_conditional, GPTConfig
from trainer_conditional import Trainer, TrainerConfig
from utils import set_seed, sample
import sys
import pickle
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from collections import defaultdict
from matplotlib import pyplot as plt
from utils_dips import pickle_load, pickle_save

sys.path.append('/home/dipu/dipu_ps/codes/UIGeneration/VTN')
from model_MLP_AE2_old import Simple_MLPAE

#%%
def argument_parser():
    parser = argparse.ArgumentParser('Layout Transformer')
    parser.add_argument("--device", default='cuda:0', type=str)
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

    # Evaluation args
    parser.add_argument('--pair', type=str, default='different', choices=['different', 'same'])
    parser.add_argument('--pt_model_path', type=str, default='runs/ConditionalLayoutTransformer/Condlayoutransformer_bs40_lr0.001_lrdecay10_FzSMFalse_SL_True')
    parser.add_argument('--reference_num', type=int, default=2)
    args = parser.parse_args()
    return args

#%%
args = argument_parser()
set_seed(args.seed)
device = args.device

#Dataset
train_dataset = RICO_Seq_Box(args.rico_info, args.train_ann,  max_length=None, precision=args.precision)
# valid_dataset = RICO_Seq_Box(args.rico_info, args.gallery_ann,  max_length=train_dataset.max_length, precision=args.precision)
# query_dataset = RICO_Seq_Box(args.rico_info, args.query_ann,  max_length=train_dataset.max_length, precision=args.precision)
    
#train_dataset2 = RICO_Seq_Box_SimilarRef(args.rico_info, args.train_ann,  max_length=train_dataset.max_length, precision=args.precision)
valid_dataset2 = RICO_Seq_Box_SimilarRef(args.rico_info, args.gallery_ann,  
                                         max_length=train_dataset.max_length, 
                                         precision=args.precision,
                                         ref_index=args.reference_num)


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

pt_model_path = args.pt_model_path # 'runs/ConditionalLayoutTransformer/Condlayoutransformer_bs40_lr0.001_lrdecay10_FzSMFalse_SL_False'

split = 'test' # 'test'
pair = args.pair
save_dir_gen = f'{pt_model_path}/{split}_gen_images_pair_{pair}_multiple'
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
                    num_workers=0)
else:
    dataset_ = train_dataset2 if split =='train' else valid_dataset2
    loader = DataLoader(dataset_, shuffle=False, pin_memory=True,
                    batch_size=tconf.batch_size,
                    num_workers=0)



# #%%
ncompletions = 3
reference_num = args.reference_num

all_results = defaultdict(dict)
with torch.no_grad():
    pbar = enumerate(loader)
    for it, data_loaded in pbar:
        if it == 20:
            break
        if it%2 ==0:
            print(f'Done {it} iterations ')
        if pair == 'same':
            x, y, uxid, boxes, box_exists, n_boxes, label_ids = data_loaded
        else:
            x, y, uxid, ref_uxid, ref_x, boxes, box_exists, n_boxes, label_ids = data_loaded
        
        boxes, box_exists  = boxes.to(device), box_exists.to(device)
        label_ids = label_ids.to(device)    
        uxid = str(uxid[0])

        # Get latent vector for condition
        _ , z = search_model.encoder(boxes, box_exists, n_boxes, label_ids)
        print(it)
   
        x_cond = x.to(device) 
      
        gt_layouts= x.detach().cpu().numpy()
        gt_layouts, gt_boxes, gt_catnames = train_dataset.render(gt_layouts, return_bbox=True)

        if pair == 'different':
            ref_layouts= ref_x.detach().cpu().numpy()
            ref_layouts, ref_boxes, ref_catnames = train_dataset.render(ref_layouts,return_bbox=True)
        else:
            ref_layouts = gt_layouts
            ref_boxes = gt_boxes
            ref_catnames = gt_catnames


        for nii in range(ncompletions):
            # Reconstructed layouts
            logits, _ = model(x_cond, z)
            probs = F.softmax(logits, dim=-1)
            _, y = torch.topk(probs, k=1, dim=-1)
            layouts_recon = torch.cat((x_cond[:, :1], y[:, :, 0]), dim=1).detach().cpu().numpy()
            recon_layouts, recon_boxes, recon_cat_names = train_dataset.render(layouts_recon, return_bbox=True) 

            # Generation with First element
            layouts_firstel = sample(model, x_cond[:, :6], steps=train_dataset.max_length, z=z,
                                        temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
            layouts_firstel, firstel_boxes, firstel_catnames = train_dataset.render(layouts_firstel, return_bbox=True)

            # Generation with bos only
            layouts_bos = sample(model, x_cond[:, :1], steps=train_dataset.max_length, z=z,
                                        temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
            layouts_bos, bos_boxes, bos_catnames = train_dataset.render(layouts_bos,return_bbox=True)  

            # Generation with partial
            layouts_partial = sample(model, x_cond[:, :26], steps=train_dataset.max_length, z=z,
                                        temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
            layouts_partial, partial_boxes, partial_catnames = train_dataset.render(layouts_partial, return_bbox=True)


            cats_and_boxes = {'gbb': gt_boxes, 'gc': gt_catnames,
                        'cbb':ref_boxes, 'cc':ref_catnames,    
                        'rbb' : recon_boxes, 'rc': recon_cat_names,
                        'fbb': firstel_boxes, 'fc':firstel_catnames,
                        'bbb': bos_boxes, 'bc':bos_catnames,
                        'pbb': partial_boxes, 'pc':partial_catnames}
            
            all_results[uxid] = cats_and_boxes


            #Plot in a simple figure
            if it <=100:
                fig, ax = plt.subplots(1,6, figsize=(30, 12), constrained_layout=True)
                plt.setp(ax,  xticklabels=[],  yticklabels=[])
                ax[0].imshow(gt_layouts)
                ax[0].set_title('GroundTruth')

                ax[1].imshow(ref_layouts)
                ax[1].set_title('Condition/Reference')

                ax[2].imshow(recon_layouts)
                ax[2].set_title('Reconstruction')

                ax[3].imshow(layouts_bos)
                ax[3].set_title('Gen-BoS')

                ax[4].imshow(layouts_firstel)
                ax[4].set_title('Gen-FirstEle.')


                ax[5].imshow(layouts_partial)
                ax[5].set_title('Gen-Partial-(5 Elem.)')

                plt.savefig(f'{save_dir_gen}/{uxid}_ref{reference_num}_gen{nii+1}plot.png')
                plt.close()

# result_dir = pt_model_path + '/results'
# os.makedirs(result_dir, exist_ok=True)
# pickle_save(result_dir+f'/gen_result_pair{args.pair}_split{split}.pkl',     all_results)    
   
