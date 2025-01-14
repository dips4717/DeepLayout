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

from model_MLP_AE2_old import Simple_MLPAE


if __name__ == "__main__":
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
    if args.mode == 'conditional':
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

    elif args.mode == 'unconditional':
        args.dataset_name = 'RICO'
        train_dataset = RICOLayout(args.rico_info, split='train', precision=args.precision)
        valid_dataset = RICOLayout(args.rico_info, split='gallery', max_length=train_dataset.max_length, precision=args.precision)
        query_dataset = RICOLayout(args.rico_info, split='query', max_length=train_dataset.max_length, precision=args.precision)
        model_mlp = None
        config_mlp=None
    else:
        raise('Not Implemented')

    args.log_dir = args.base_wdir + args.log_dir
    exp_prefix = 'Un' if args.mode =='unconditional' else ''
    args.exp_name = f'{exp_prefix}Condlayoutransformer_bs{args.batch_size}_lr{args.lr}_lrdecay{args.lr_decay_every}_FzSM{args.freeze_search_model}_SL_{args.search_loss}'
    
    args.log_dir = os.path.join(args.log_dir, 'ConditionalLayoutTransformer', args.exp_name )
    samples_dir = os.path.join(args.log_dir, "samples")
    args.ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    args.result_dir = os.path.join(args.log_dir, "results")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    set_seed(args.seed)


    # Seq2Seq Model
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_length,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      conditional=True, is_causal=True, activation_function= 'gelu') # a GPT-1
    model = GPT_conditional(mconf)
    
    if args.evaluate:
        args.batch_size = args.batch_size_eval
    tconf = TrainerConfig(max_epochs=args.epochs,
                          batch_size=args.batch_size,
                          lr_decay=args.lr_decay,
                          learning_rate = args.lr * args.batch_size,
                          warmup_iters=args.warmup_iters,
                          final_iters=args.final_iters,
                          ckpt_dir=args.ckpt_dir,
                          samples_dir=samples_dir,
                          sample_every=args.sample_every)
    
    trainer = Trainer(model, model_mlp, train_dataset, valid_dataset, tconf, args, config_mlp)
    
    if args.evaluate:
        trainer.evaluate()
        print('\nEvalution...')
    else:
        trainer.train()
        print('\nTraining...')
    


# nohup python main_conditional.py --device 'cuda:0' --freeze_search_model True --search_loss False &  --> thorin cuda0
# nohup python main_conditional.py --device 'cuda:0' --freeze_search_model False --search_loss False  & --> thorin cuda 0
# nohup python main_conditional.py --device 'cuda:1' --freeze_search_model False --search_loss True &  --> thorin cuda1

# Lr decay
# nohup python main_conditional.py --device  'cuda:2' --lr 0.001 --lr_decay_every 10 --freeze_search_model True --search_loss False &
# nohup python main_conditional.py --device  'cuda:2'  --lr 0.001 --lr_decay_every 10  --freeze_search_model False --search_loss False &
# nohup python main_conditional.py --device  'cuda:3'  --lr 0.001 --lr_decay_every 10  --freeze_search_model False --search_loss True &



