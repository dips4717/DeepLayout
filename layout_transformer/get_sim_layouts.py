import os
import argparse
import torch
from dataset import MNISTLayout, JSONLayout, RICOLayout, RICOLayout_withImage, RICO_Seq_Box
#from layout_transformer.dataset import RICOLayout
from model import GPT_conditional, GPTConfig
from trainer_conditional import Trainer, TrainerConfig
from utils import set_seed
from utils_dips import pickle_load, pickle_save, str2bool
import sys
import pickle
from torch.utils.data.dataloader import DataLoader
from scipy.spatial.distance import cdist
from model_MLP_AE2_old import Simple_MLPAE
import numpy as np

from utils_dips import plot_retrieved_images_and_uis
gallery_ann = 'data/mtn50_ECCV_gallery_data.pkl'

parser = argparse.ArgumentParser('Conditional Layout Transformer')

parser.add_argument("--device", type=str, default='cuda:3')
parser.add_argument("--log_dir", default="runs", help="/path/to/logs/dir")
parser.add_argument("--evaluate", type=str2bool, default=True)
parser.add_argument("--mode", type=str, default='conditional', choices=['conditional', 'unconditional'])
parser.add_argument("--server", type=str, default='aineko', choices=['condor', 'aineko'])
    
# Model architecture parameters    
parser.add_argument("--freeze_search_model", type=str2bool, default=False)
parser.add_argument("--search_loss", type=str2bool, default=True)

# Layout options
parser.add_argument("--max_length", type=int, default=128, help="batch size")
parser.add_argument('--precision', default=8, type=int)
parser.add_argument('--element_order', default='raster')
parser.add_argument('--attribute_order', default='cxywh')

# Architecture/training options
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=40, help="batch size")
parser.add_argument("--batch_size_eval", type=int, default=8, help="batch size")
# parser.add_argument("--lr", type=float, default=4.5e-06, help="learning rate")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--lr_decay_every", type=int, default=10, help="learning decay every x epochs")
parser.add_argument('--lr_decay_rate', type=float, default=0.1)

parser.add_argument('--n_layer', default=6, type=int)
parser.add_argument('--n_embd', default=512, type=int)
parser.add_argument('--n_head', default=8, type=int)
parser.add_argument('--lr_decay', action='store_false', help="use learning rate decay")
parser.add_argument('--warmup_iters', type=int, default=0, help="linear lr warmup iters")
parser.add_argument('--final_iters', type=int, default=0, help="cosine lr final iters")
parser.add_argument('--sample_every', type=int, default=1, help="sample every epoch")

args = parser.parse_args()

# Base working directory
if args.server == 'condor':
    args.base_wdir = '/vol/research/projectSpaceDipu/codes/UIGeneration/DeepLayout/layout_transformer/'
else:
    args.base_wdir = './'
args.rico_info = args.base_wdir + 'data/rico.pkl'

# Set some args based on input args

args.dataset_name = 'RICO_Seq_Box'
args.train_ann = args.base_wdir + 'data/mtn50_ECCV_train_data.pkl'
args.gallery_ann = args.base_wdir + 'data/mtn50_ECCV_gallery_data.pkl'
args.query_ann = args.base_wdir+ 'data/mtn50_ECCV_query_data.pkl'
train_dataset = RICO_Seq_Box(args.rico_info, args.train_ann,  max_length=None, precision=args.precision)
valid_dataset = RICO_Seq_Box(args.rico_info, args.gallery_ann,  max_length=train_dataset.max_length, precision=args.precision)
query_dataset = RICO_Seq_Box(args.rico_info, args.query_ann,  max_length=train_dataset.max_length, precision=args.precision)

# MLPAE model
trained_path = args.base_wdir + 'trained_mlpae/SimpleMLPAE_25DecAtLast0_lr0.01_bs256_tanD0_HM_1_UBL_0_SW1.0_BW50.0_EW1.0_RW10.0_AttEnc_attention_BoxLMSE_Actrelu_dim512'
config_fn = f'{trained_path}/config.pkl'
config_mlp = pickle_load(config_fn)
model_mlp = Simple_MLPAE(config_mlp)
config_mlp.embLossOnly = False
config_mlp.probabilistic= False

pt_model = f'{trained_path}/ckpt_ep39.pth.tar'
checkpoint = torch.load(pt_model,map_location=args.device)
model_mlp.load_state_dict(checkpoint['state_dict'])
print(f'Loaded search encoder model from {trained_path}')
model_mlp.to(args.device)
model_mlp.eval()

##
loader = DataLoader(valid_dataset, shuffle=False, pin_memory=True,
                                batch_size=40,
                                num_workers=4,
                                drop_last=False)

features = []
uxids = []
with torch.no_grad():
    for it, input_data in enumerate(loader):
        x, y, uxid, boxes, box_exists, n_boxes, label_ids = input_data
        boxes, box_exists  = boxes.to(args.device), box_exists.to(args.device)
        label_ids = label_ids.to(args.device)
        _ , z = model_mlp.encoder(boxes, box_exists, n_boxes, label_ids)
        features.append(z.detach().cpu().numpy())
        uxids.append(uxid)


uxids = [x for y in uxids for x in y ]
uxids = [str(x.item()) for x in uxids]
features = np.concatenate(features)
distances = cdist(features, features, metric= 'euclidean')
sort_inds = np.argsort(distances)

plot_retrieved_images_and_uis(sort_inds, uxids, uxids, avgIouArray=None, avgPixAccArray=None)

from collections import defaultdict
retrieval_results = defaultdict()

for i in range((sort_inds.shape[0])):
    top5 = [uxids[sort_inds[i,x]] for x in range(6) ]
    retrieval_results[uxids[i]] = top5

pickle_save('data/mlpae_results.pkl', retrieval_results)



