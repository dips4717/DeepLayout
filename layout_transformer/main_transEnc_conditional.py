import os
import argparse
import torch
from dataset import RICOLayout
from model import  GPTConfig
from model_transEnc_GPTdecoder import TransformerEncoder_GPTConditional
from trainer_transEnc_conditional import Trainer, TrainerConfig
from utils import set_seed
from utils_dips import pickle_load, pickle_save, str2bool
import sys
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Conditional Layout Transformer')

    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--log_dir", default="runs", help="/path/to/logs/dir")
    parser.add_argument("--evaluate", type=str2bool, default=False)
    parser.add_argument("--mode", type=str, default='conditional', choices=['conditional', 'unconditional'])
    parser.add_argument("--server", type=str, default='aineko', choices=['condor', 'aineko'])
    parser.add_argument("--repeat_exp", type=str2bool, default=False)
   
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
    parser.add_argument('--agg_type', default='MLP', type=str, 
                        choices=['Attention', 'MLP', 'AveragePool', 'FirstOut' ])
    
    args = parser.parse_args()
    
    # Base working directory
    if args.server == 'condor':
        args.base_wdir = '/vol/research/projectSpaceDipu/codes/UIGeneration/DeepLayout/layout_transformer/'
    else:
        args.base_wdir = './'
    args.rico_info = args.base_wdir + 'data/rico.pkl'
    args.log_dir = args.base_wdir + args.log_dir

    args.dataset_name = 'RICO'
    train_dataset = RICOLayout(args.rico_info, split='train', precision=args.precision, pad=None)
    valid_dataset = RICOLayout(args.rico_info, split='gallery', max_length=train_dataset.max_length, precision=args.precision, pad=None)
    query_dataset = RICOLayout(args.rico_info, split='query', max_length=train_dataset.max_length, precision=args.precision, pad=None)
       
    exp_prefix = 'Un' if args.mode =='unconditional' else ''
    repeat_prefix = '_Repeat' if args.repeat_exp else ''
    args.exp_name = f'TransEnc_ConditionalDec_bs{args.batch_size}_lr{args.lr}_lrdecay{args.lr_decay_every}_aggtype{args.agg_type}_seed{args.seed}{repeat_prefix}'
    args.log_dir = os.path.join(args.log_dir, 'TransEnc_ConditionalDec', args.exp_name)
    samples_dir = os.path.join(args.log_dir, "samples")
    args.ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    args.result_dir = os.path.join(args.log_dir, "results")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    print('Initialized log directories')
    
    print(args.ckpt_dir)
    print(args.log_dir)
    print(args.result_dir)

    set_seed(args.seed)

    # Seq2Seq Model
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_length,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      conditional=True, is_causal=True, activation_function= 'gelu',
                      agg_type= args.agg_type) # a GPT-1
    
    model = TransformerEncoder_GPTConditional(mconf)
    
    tconf = TrainerConfig(max_epochs=args.epochs,
                          batch_size=args.batch_size,
                          lr_decay=args.lr_decay,
                          learning_rate = args.lr * args.batch_size,
                          warmup_iters=args.warmup_iters,
                          final_iters=args.final_iters,
                          ckpt_dir=args.ckpt_dir,
                          samples_dir=samples_dir,
                          sample_every=args.sample_every)
    
    trainer = Trainer(model, train_dataset, valid_dataset, tconf, args)
    
    #trainer.save_checkpoint(5)

    if args.evaluate:
        print('\nEvalution...')
        trainer.evaluate()
    else:
        print('\nTraining...')
        trainer.train()
    
        
# nohup python main_conditional.py --device 'cuda:0' --freeze_search_model True --search_loss False &  --> thorin cuda0
# nohup python main_conditional.py --device 'cuda:0' --freeze_search_model False --search_loss False  & --> thorin cuda 0
# nohup python main_conditional.py --device 'cuda:1' --freeze_search_model False --search_loss True &  --> thorin cuda1

# Lr decay
# nohup python main_conditional.py --device  'cuda:2' --lr 0.001 --lr_decay_every 10 --freeze_search_model True --search_loss False &
# nohup python main_conditional.py --device  'cuda:2'  --lr 0.001 --lr_decay_every 10  --freeze_search_model False --search_loss False &
# nohup python main_conditional.py --device  'cuda:3'  --lr 0.001 --lr_decay_every 10  --freeze_search_model False --search_loss True &



