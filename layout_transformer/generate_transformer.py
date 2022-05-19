import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import argparse
from standard_transformer import Seq2SeqTransformer
from dataset import RICOLayout
import os
import wandb
from torch.utils.data.dataloader import DataLoader
import numpy as np
from utils_dips import save_checkpoint, create_mask, pickle_save
from standard_transformer import sample_standard_trans as sample
import torch.nn.functional as F
from collections import defaultdict
from matplotlib import pyplot as plt
import torch.nn.functional as F


parser = argparse.ArgumentParser('Conditional Layout Transformer')
parser.add_argument("--exp", default="layout", help="mnist_threshold_1")
parser.add_argument("--log_dir", default="./runs", help="/path/to/logs/dir")
parser.add_argument("--dataset_name", type=str, default='RICO', choices=['RICO', 'MNIST', 'COCO', 'PubLayNet', 'RICO_Image'])
parser.add_argument("--device", type=str, default='cuda:1')


# MNIST options
parser.add_argument("--data_dir", default=None, help="")
parser.add_argument("--threshold", type=int, default=1, help="threshold for grayscale values")

# COCO/PubLayNet options
parser.add_argument("--train_json", default="data/coco/instances_train2014.json", help="/path/to/train/json")
parser.add_argument("--val_json", default="data/coco/instances_val2014.json", help="/path/to/val/json")

# RICO Options
parser.add_argument("--rico_info", default='/vol/research/projectSpaceDipu/codes/UIGeneration/DeepLayout/layout_transformer/data/rico.pkl')
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
parser.add_argument("--epochs", type=int, default=15, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=3, help="batch size")
parser.add_argument("--lr", type=float, default=4.5e-06, help="learning rate")
parser.add_argument("--lr_decay_every", type=int, default=7, help="learning decay every x epochs")
parser.add_argument("--num_workers", type=int, default=8)


parser.add_argument('--n_layer', default=6, type=int)
parser.add_argument('--n_embd', default=512, type=int)
parser.add_argument('--n_head', default=8, type=int)
# parser.add_argument('--evaluate', action='store_true', help="evaluate only")
parser.add_argument('--lr_decay', action='store_false', help="use learning rate decay")
parser.add_argument('--warmup_iters', type=int, default=0, help="linear lr warmup iters")
parser.add_argument('--final_iters', type=int, default=0, help="cosine lr final iters")
parser.add_argument('--sample_every', type=int, default=1, help="sample every epoch")

# Evaluation args
parser.add_argument('--pair', type=str, default='same', choices=['different', 'same'])
parser.add_argument('--pt_model_path', type=str, default='runs/TransEnc_TransDec/TransformerEncDec_bs40_lr4.5e-06_lrdecay7' )
parser.add_argument('--split', default='test', type=str)


args = parser.parse_args()

args.rico_info = 'data/rico.pkl'
args.log_dir = args.pt_model_path
samples_dir = os.path.join(args.log_dir, "samples")
args.ckpt_dir = os.path.join(args.log_dir, "checkpoints")
args.result_dir = os.path.join(args.log_dir, "results")
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(args.ckpt_dir, exist_ok=True)
os.makedirs(args.result_dir, exist_ok=True)
print('Initialized log directories')

split = args.split
pair = args.pair
save_dir_gen = f'{args.pt_model_path}/{split}_gen_images_pair_{pair}'
if not os.path.exists(save_dir_gen):
    os.makedirs(save_dir_gen, exist_ok=True)


train_dataset = RICOLayout(args.rico_info, split='train', precision=args.precision, pad='None', inference=True)
valid_dataset = RICOLayout(args.rico_info, split='gallery', max_length=train_dataset.max_length, precision=args.precision, pad='None',inference=True)
query_dataset = RICOLayout(args.rico_info, split='query', max_length=train_dataset.max_length, precision=args.precision, pad='None',inference=True)

if pair == 'same':
    dataset_ = train_dataset if split =='train' else valid_dataset
    loader = DataLoader(dataset_, shuffle=False, pin_memory=True,
                    batch_size=1,
                    num_workers=4)
else:
    pass

max_length = train_dataset.max_length
pad_token = train_dataset.vocab_size - 1

model = Seq2SeqTransformer(args.n_layer, args.n_layer, args.n_embd, 
                                    args.n_head, train_dataset.vocab_size, train_dataset.vocab_size, args.n_embd)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
model = model.to(args.device)

pt_model = torch.load(f'{args.pt_model_path}/checkpoints/checkpoint_best.pth', map_location=args.device)
state_dict = pt_model['state_dict']

#print(pt_model.keys())
epoch = pt_model['epoch']
train_loss = pt_model['train_loss']
test_loss = pt_model['test_loss']
model.load_state_dict(state_dict)

print(f'TrainLoss: {train_loss}\nTestLoss: {test_loss}\nEpcoh: {epoch}  ')


#%%
all_results = defaultdict(dict)
with torch.no_grad():
    pbar = enumerate(loader)
    for it, data_loaded in pbar:
        if it == 5:
            break
        if it%50 ==0:
            print(f'Done {it} iterations ')
        
        if pair == 'same':
            x , y , uxid = data_loaded
        else:
            pass
        
        uxid = str(uxid[0])
        x = x.to(args.device)
        y=  y.to(args.device)


        # Groundtruth Layouts
        gt_layouts= x.detach().cpu().numpy()
        gt_layouts, gt_boxes, gt_catnames = train_dataset.render(gt_layouts, return_bbox=True)

        if pair == 'different':
            pass
        else:
            ref_layouts = gt_layouts
            ref_boxes = gt_boxes
            ref_catnames = gt_catnames


        # Reconstruction
        src = x.t()
        tgt = y.t()
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_token, args.device)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        probs = F.softmax(logits, dim=-1)
        _, y = torch.topk(probs, k=1, dim=-1)
        layouts_recon = torch.cat((x[:, :1].cpu(), y.permute(1,0,2)[:, :, 0].cpu()), dim=1).detach().numpy()
        recon_layouts, recon_boxes, recon_cat_names = train_dataset.render(layouts_recon, return_bbox=True) 
        
        
        # Generation with bos only
        src = src 
        tgt_input = tgt[:-1, :]
        tgt_input = tgt_input[:1,:]
        layouts_bos = sample(model, src, tgt_input, pad_token, steps=train_dataset.max_length,
                                 temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
        layouts_bos = layouts_bos.transpose()  # Convert into batch first 
        layouts_bos, bos_boxes, bos_catnames = train_dataset.render(layouts_bos,return_bbox=True) 
        

        # Generation with First element
        src = src 
        tgt_input = tgt[:-1, :]
        tgt_input = tgt_input[:6,:]
        layouts_firstele = sample(model, src, tgt_input, pad_token, steps=train_dataset.max_length,
                                 temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
        layouts_firstele = layouts_firstele.transpose()  # Convert into batch first 
        layouts_firstel, firstel_boxes, firstel_catnames = train_dataset.render(layouts_firstele, return_bbox=True)
        
        
        
       


        # Generation with partial
        src = src 
        tgt_input = tgt[:-1, :]
        tgt_input = tgt_input[:26,:]
        layouts_partial = sample(model, src, tgt_input, pad_token, steps=train_dataset.max_length,
                                 temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
        layouts_partial = layouts_partial.transpose()  # Convert into batch first 
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

            plt.savefig(f'{save_dir_gen}/{uxid}_plot.png')
            plt.close()
#%%
result_dir = args.pt_model_path + '/results'
os.makedirs(result_dir, exist_ok=True)
pickle_save(result_dir+f'/gen_result_pair{args.pair}_split{split}_iter{it}.pkl',     all_results)    
   
