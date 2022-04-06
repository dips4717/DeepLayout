import os
import argparse
import torch
from torch.utils.data.dataloader import DataLoader

from dataset import RICOLayout
from model import  GPTConfig
from model_transEnc_GPTdecoder import TransformerEncoder_GPTConditional
from trainer_transEnc_conditional import Trainer, TrainerConfig
from utils import set_seed
from utils_dips import pickle_load, pickle_save, str2bool
from utils_dips import sample_transEnc_conditional
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict
from matplotlib import pyplot as plt
import torch.nn.functional as F



parser = argparse.ArgumentParser('Conditional Layout Transformer')

parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--log_dir", default="runs", help="/path/to/logs/dir")
parser.add_argument("--evaluate", type=str2bool, default=False)
parser.add_argument("--mode", type=str, default='conditional', choices=['conditional', 'unconditional'])
parser.add_argument("--server", type=str, default='aineko', choices=['condor', 'aineko'])
    
   
# Layout / Dataset options
parser.add_argument("--max_length", type=int, default=128, help="batch size")
parser.add_argument('--precision', default=8, type=int)
parser.add_argument('--element_order', default='raster')
parser.add_argument('--attribute_order', default='cxywh')

#training options
parser.add_argument("--seed", type=int, default=95, help="random seed")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=40, help="batch size")
# parser.add_argument("--lr", type=float, default=4.5e-06, help="learning rate")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--lr_decay_every", type=int, default=15, help="learning decay every x epochs")
parser.add_argument('--lr_decay_rate', type=float, default=0.1)
parser.add_argument('--lr_decay', action='store_false', help="use learning rate decay")
parser.add_argument('--warmup_iters', type=int, default=0, help="linear lr warmup iters")
parser.add_argument('--final_iters', type=int, default=0, help="cosine lr final iters")
parser.add_argument('--sample_every', type=int, default=1, help="sample every epoch")

# Architecture/
parser.add_argument('--n_layer', default=2, type=int) # 6
parser.add_argument('--n_embd', default=512, type=int)
parser.add_argument('--n_head', default=8, type=int)
parser.add_argument('--agg_type', default='Attention', type=str, 
                    choices=['Attention', 'MLP', 'AveragePool', 'FirstOut' ])

# Evaluation args
parser.add_argument('--pair', type=str, default='same', choices=['different', 'same'])
parser.add_argument('--pt_model_path', type=str, default='runs/TransEnc_ConditionalDec/TransEnc_ConditionalDec_bs40_lr0.001_lrdecay15_aggtypeAttention' )
parser.add_argument('--split', default='test', type=str)
args = parser.parse_args()



 
args.rico_info = 'data/rico.pkl'
args.log_dir = args.pt_model_path

split = args.split
pair = args.pair
save_dir_gen = f'{args.pt_model_path}/{split}_gen_images_pair_{pair}'
if not os.path.exists(save_dir_gen):
    os.makedirs(save_dir_gen, exist_ok=True)

train_dataset = RICOLayout(args.rico_info, split='train', precision=args.precision, pad='Padding', inference=True)
valid_dataset = RICOLayout(args.rico_info, split='gallery', max_length=train_dataset.max_length, precision=args.precision, pad='Padding',inference=True)
query_dataset = RICOLayout(args.rico_info, split='query', max_length=train_dataset.max_length, precision=args.precision, pad='Padding',inference=True)

mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_length,
                  n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                  conditional=True, is_causal=True, activation_function= 'gelu',
                  agg_type= args.agg_type) # a GPT-1

model = TransformerEncoder_GPTConditional(mconf)
pt_model = torch.load(f'{args.pt_model_path}/checkpoints/checkpoint_best.pth', map_location=args.device)
epoch = pt_model['epoch']
train_loss = pt_model['train_loss']
test_loss = pt_model['test_loss']
print(f'TrainLoss: {train_loss}\nTestLoss: {test_loss}\nEpcoh: {epoch}  ')
model.load_state_dict(pt_model['model'])


#TODO   
train_dataset2 = None
valid_dataset2 = None 

samples_dir = os.path.join(args.log_dir, "samples")
args.ckpt_dir = os.path.join(args.log_dir, "checkpoints")
args.result_dir = os.path.join(args.log_dir, "results")
os.makedirs(samples_dir, exist_ok=True)
os.makedirs(args.ckpt_dir, exist_ok=True)
os.makedirs(args.result_dir, exist_ok=True)
print('Initialized log directories')

# Seq2Seq Model
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_length,
                  n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                  conditional=True, is_causal=True, activation_function= 'gelu',
                  agg_type= args.agg_type) # a GPT-1

model = TransformerEncoder_GPTConditional(mconf)
model = model.to(args.device)
model.eval()


# Dataloader on validation set
if pair == 'same':
    dataset_ = train_dataset if split =='train' else valid_dataset
    loader = DataLoader(dataset_, shuffle=False, pin_memory=True,
                    batch_size=1,
                    num_workers=4)
else:
    dataset_ = train_dataset2 if split =='train' else valid_dataset2
    loader = DataLoader(dataset_, shuffle=False, pin_memory=True,
                    batch_size=1,
                    num_workers=4)


## Gen iteration
pad_token = train_dataset.vocab_size - 1
all_results = defaultdict(dict)
with torch.no_grad():
    pbar = enumerate(loader)
    for it, data_loaded in pbar:
        if it == 5:
            break
        if it%50 ==0:
            print(f'Done {it} iterations ')
        
        if pair == 'same':
            x , _, uxid = data_loaded
        else:
            pass
        uxid = str(uxid[0])
        x = x.to(args.device)
        
        gt_layouts= x.detach().cpu().numpy()
        gt_layouts, gt_boxes, gt_catnames = train_dataset.render(gt_layouts, return_bbox=True)

        if pair == 'different':
            ref_layouts= None
            ref_layouts, ref_boxes, ref_catnames = train_dataset.render(ref_layouts,return_bbox=True)
        else:
            ref_layouts = gt_layouts
            ref_boxes = gt_boxes
            ref_catnames = gt_catnames

        # Reconstructed layouts
        logits, _,_ = model(x, pad_token=pad_token)
        probs = F.softmax(logits, dim=-1)
        _, y = torch.topk(probs, k=1, dim=-1)
        layouts_recon = torch.cat((x[:, :1], y[:, :, 0]), dim=1).detach().cpu().numpy()
        recon_layouts, recon_boxes, recon_cat_names = train_dataset.render(layouts_recon, return_bbox=True) 

        # Generation with First element
        layouts_firstel = sample_transEnc_conditional(model, x[:, :6], 
                                                      seq_all=x, 
                                                      inference=True, 
                                                      steps=train_dataset.max_length,
                                                      temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
        layouts_firstel, firstel_boxes, firstel_catnames = train_dataset.render(layouts_firstel, return_bbox=True)

        # Generation with bos only
        layouts_bos = sample_transEnc_conditional(model, x[:, :1],
                                                  seq_all=x, 
                                                  inference=True,
                                                  steps=train_dataset.max_length, 
                                                  temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
        layouts_bos, bos_boxes, bos_catnames = train_dataset.render(layouts_bos,return_bbox=True)  

        # Generation with partial
        layouts_partial = sample_transEnc_conditional(model, x[:, :26], 
                                                      seq_all=x, 
                                                      steps=train_dataset.max_length, 
                                                      inference=True,
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

            plt.savefig(f'{save_dir_gen}/{uxid}_plot.png')
            plt.close()

result_dir = args.pt_model_path + '/results'
os.makedirs(result_dir, exist_ok=True)
pickle_save(result_dir+f'/gen_result_pair{args.pair}_split{split}.pkl',     all_results)    
   
