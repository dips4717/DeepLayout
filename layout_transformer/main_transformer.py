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
from utils_dips import save_checkpoint, create_mask
from standard_transformer import sample_standard_trans as sample
import torch.nn.functional as F

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Conditional Layout Transformer')
    parser.add_argument("--exp", default="layout", help="mnist_threshold_1")
    parser.add_argument("--log_dir", default="./runs", help="/path/to/logs/dir")
    parser.add_argument("--dataset_name", type=str, default='RICO', choices=['RICO', 'MNIST', 'COCO', 'PubLayNet', 'RICO_Image'])
    parser.add_argument("--device", type=str, default='cuda:0')
    
    
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



args = parser.parse_args()
args.exp_name = f'TransformerEncDec_bs{args.batch_size}_lr{args.lr}_lrdecay{args.lr_decay_every}'
args.log_dir = os.path.join(args.log_dir, 'ConditionalLayoutTransformer', args.exp_name )
args.samples_dir = os.path.join(args.log_dir, "samples")
args.ckpt_dir = os.path.join(args.log_dir, "checkpoints")
args.result_dir = os.path.join(args.log_dir, "results")
os.makedirs(args.samples_dir, exist_ok=True)
os.makedirs(args.ckpt_dir, exist_ok=True)
os.makedirs(args.result_dir, exist_ok=True)


train_dataset = RICOLayout(args.rico_info, split='train', precision=args.precision, pad=None)
test_dataset = RICOLayout(args.rico_info, split='gallery', max_length=train_dataset.max_length, precision=args.precision, pad=None)
query_dataset = RICOLayout(args.rico_info, split='query', max_length=train_dataset.max_length, precision=args.precision, pad=None)


def main(args):
    torch.manual_seed(0)
    wandb.init(project='LayoutGeneration', name=args.exp_name)
    wandb.config.update(args)
    global iters
    iters = 0
    device = args.device
    max_length = train_dataset.max_length
    pad_token = train_dataset.vocab_size - 1

    model = Seq2SeqTransformer(args.n_layer, args.n_layer, args.n_embd, 
                                    args.n_head, train_dataset.vocab_size, train_dataset.vocab_size, args.n_embd)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model = model.to(args.device)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    def run_epoch(split):
        global iters
        is_train = split == 'train'
        model.train(is_train)
        data = train_dataset if is_train else test_dataset
        loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers, drop_last=False)
        losses = []
        for it, (x, y) in enumerate(loader):
            if epoch == 0 and not is_train:
                    args.fixed_x = x[:min(4, len(x))]
                    args.fixed_y = y[:min(4, len(y))]
            # place data on the correct device
            src = x.to(device)
            tgt = y.to(device)
            src = src.t()
            tgt = tgt.t()
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_token, args.device)
            # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            tgt_out = tgt[1:, :]
            
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses.append(loss.item())
            
            if is_train:
                # optimizer.zero_grad()    
                model.zero_grad()            
                loss.backward()
                optimizer.step()
                iters+=1
                wandb.log({
                            'train loss': loss.item(),
                            'lr': optimizer.param_groups[0]['lr'], 'epoch': epoch+1
                        }, step= iters)



        if not is_train:
                test_loss = float(np.mean(losses))
                wandb.log({'test loss': test_loss}, step=iters)
                return test_loss
        else:
            train_loss = float(np.mean(losses))
            wandb.log({'Avg Train Loss': train_loss}, step=iters)
            return train_loss

    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = run_epoch('train')
        
        with torch.no_grad():
            test_loss = run_epoch('test')

        # supports early stopping based on the test loss, or just save always if no test set is provided
        good_model = test_dataset is None or test_loss < best_loss
        if args.ckpt_dir is not None and good_model:
            best_loss = test_loss
            save_checkpoint(args, model, epoch, test_loss=test_loss, train_loss=train_loss)

        # sample from the model
        if (epoch+1) % args.sample_every == 0:
            # import ipdb; ipdb.set_trace()
            # inputs
            layouts = args.fixed_x.detach().cpu().numpy()
            input_layouts = [train_dataset.render(layout) for layout in layouts]
            
            # reconstruction
            src = args.fixed_x.to(args.device)
            tgt = args.fixed_x.to(args.device)
            src = src.t()
            tgt = tgt.t()
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_token, args.device)
            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            probs = F.softmax(logits, dim=-1)
            _, y = torch.topk(probs, k=1, dim=-1)
            layouts = torch.cat((args.fixed_x[:, :1], y.permute(1,0,2)[:, :, 0].cpu()), dim=1).detach().numpy()
            recon_layouts = [train_dataset.render(layout) for layout in layouts]
            
            # samples - random
            src = src 
            tgt_input = tgt[:-1, :]
            tgt_input = tgt_input[:6,:]
            layouts = sample(model, src, tgt_input, pad_token, steps=train_dataset.max_length,
                                 temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
            layouts = layouts.transpose()  # Convert into batch first 
            sample_random_layouts = [train_dataset.render(layout) for layout in layouts]
            
            # samples - deterministic
            layouts = sample(model,src, tgt_input, pad_token, steps=train_dataset.max_length,
                                 temperature=1.0, sample=False, top_k=None).detach().cpu().numpy()
            layouts = layouts.transpose()  # Convert into batch first 
            sample_det_layouts = [train_dataset.render(layout) for layout in layouts]

            wandb.log({
                "input_layouts": [wandb.Image(pil, caption=f'input_{epoch:02d}_{i:02d}.png')
                                    for i, pil in enumerate(input_layouts)],
                "recon_layouts": [wandb.Image(pil, caption=f'recon_{epoch:02d}_{i:02d}.png')
                                    for i, pil in enumerate(recon_layouts)],
                "sample_random_layouts": [wandb.Image(pil, caption=f'sample_random_{epoch:02d}_{i:02d}.png')
                                            for i, pil in enumerate(sample_random_layouts)],
                "sample_det_layouts": [wandb.Image(pil, caption=f'sample_det_{epoch:02d}_{i:02d}.png')
                                        for i, pil in enumerate(sample_det_layouts)],
            }, step=iters)    


if __name__ == '__main__':
    main(args)





























