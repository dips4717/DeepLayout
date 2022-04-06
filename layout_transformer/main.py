import os
import argparse
import torch
from dataset import MNISTLayout, JSONLayout, RICOLayout
#from layout_transformer.dataset import RICOLayout
from model import GPT, GPTConfig
from trainer import Trainer, TrainerConfig
from utils import set_seed
from utils_dips import pickle_load, pickle_save, str2bool

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Layout Transformer')
    parser.add_argument("--log_dir", default="runs", help="/path/to/logs/dir")
    parser.add_argument("--dataset_name", type=str, default='RICO', choices=['RICO', 'MNIST', 'COCO', 'PubLayNet'])
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--evaluate", type=str2bool, default=False)
    parser.add_argument("--mode", type=str, default='conditional', choices=['conditional', 'conditional'])
    parser.add_argument("--server", type=str, default='aineko', choices=['condor', 'aineko'])
    
    # RICO Options
    #parser.add_argument("--rico_info", default='/vol/research/projectSpaceDipu/codes/UIGeneration/DeepLayout/layout_transformer/data/rico.pkl')
    parser.add_argument("--rico_info", default='data/rico.pkl')

    # Layout options
    parser.add_argument("--max_length", type=int, default=128, help="batch size")
    parser.add_argument('--precision', default=8, type=int)
    parser.add_argument('--element_order', default='raster')
    parser.add_argument('--attribute_order', default='cxywh')

    # Architecture/training options
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=40, help="batch size")
    #parser.add_argument("--lr", type=float, default=4.5e-06, help="learning rate")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument('--lr_decay', action='store_true', help="use learning rate decay")
    parser.add_argument('--lr_decay_every', type=int, default=15)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)

    parser.add_argument('--n_layer', default=6, type=int)
    parser.add_argument('--n_embd', default=512, type=int)
    parser.add_argument('--n_head', default=8, type=int)
    parser.add_argument('--ffnet', type=str, default='linear')
    
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

    args.log_dir = args.base_wdir + args.log_dir
    args.exp_name = f'layoutransformer_FF_{args.ffnet}_bs{args.batch_size}_lr{args.lr}_lrdecay{args.lr_decay_every}'
    args.log_dir = os.path.join(args.log_dir, 'LayoutTransformer', args.exp_name )
    args.samples_dir = os.path.join(args.log_dir, "samples")
    args.ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    os.makedirs(args.samples_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    

    set_seed(args.seed)

    train_dataset = RICOLayout(args.rico_info, split='train', precision=args.precision)
    valid_dataset = RICOLayout(args.rico_info, split='test', max_length=train_dataset.max_length, precision=args.precision)
    

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_length,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, 
                      ffnet=args.ffnet, is_causal=True, activation_function= 'gelu')  # a GPT-1

    
    model = GPT(mconf)
    tconf = TrainerConfig(max_epochs=args.epochs,
                          batch_size=args.batch_size,
                          lr_decay=args.lr_decay,
                          learning_rate= args.lr * args.batch_size,
                          warmup_iters=args.warmup_iters,
                          final_iters=args.final_iters,
                          ckpt_dir=args.ckpt_dir,
                          samples_dir=args.samples_dir,
                          sample_every=args.sample_every)
    trainer = Trainer(model, train_dataset, valid_dataset, tconf, args)
    
    if args.evaluate:
        trainer.evaluate()
        print('\nEvalution...')
    else:
        trainer.train()
        print('\nTraining...')


# nohup python main.py --device 'cuda:1' 
# nohup python main.py --device 'cuda:2' --lr 0.001 --lr_decay_every 10 > layoutransformer_lrdecay.out &

