"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import os
import math
import logging
import wandb

from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from utils import sample
from utils_dips import AverageMeter

logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 32 #64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_iters = 0
    final_iters = 0  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_dir = None
    samples_dir = None
    sample_every = 1
    num_workers =4 # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, args):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.args = args
        self.iters = 0
        self.fixed_x = None
        self.fixed_y = None
        print("Using wandb")
        wandb.init(project='LayoutGeneration', name=args.exp_name)
        wandb.config.update(args)

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device =  args.device #torch.cuda.current_device()
            # self.model = torch.nn.DataParallel(self.model).to(self.device)
            print(self.device)
            self.model = self.model.to(self.device)    
            
            
    def save_checkpoint(self, epoch, test_loss=None, train_loss=None):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt_path = os.path.join(self.config.ckpt_dir, f'checkpoint_best.pth')
        logger.info("saving %s", ckpt_path)
        torch.save({'state_dict': raw_model.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'test_loss': test_loss},  ckpt_path)


    def compute_loss(self):        
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        pad_token = self.train_dataset.vocab_size - 1
        trainloader = DataLoader(self.train_dataset, shuffle=False, pin_memory=True,
                                drop_last=False, batch_size=config.batch_size, num_workers=config.num_workers)
        valloader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                                drop_last=False, batch_size=config.batch_size, num_workers=config.num_workers)
        model.eval()

        def run_one_epoch(loader):
            with torch.no_grad():
                loss_meter = AverageMeter()
                pbar = tqdm(enumerate(loader), total=len(loader)) 
                for it, (x, y) in pbar:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    logits, loss = model(x, y, pad_token=pad_token)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    loss_meter.update(loss.item())

            return loss_meter.avg

        train_loss = run_one_epoch(trainloader)
        val_loss = run_one_epoch(valloader)

        result_file = 'runs/RICO/result.txt'
        with open(result_file, 'a') as f:
            f.write(f'Average Training Loss:   {train_loss}\t num: {len(self.train_dataset)}\n')
            f.write(f'AVerage Validation Loss: {val_loss}\t num:{len(self.test_dataset)}\n')



    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        pad_token = self.train_dataset.vocab_size - 1

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:

                if epoch == 0 and not is_train:
                    self.fixed_x = x[:min(4, len(x))]
                    self.fixed_y = y[:min(4, len(y))]

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    # import ipdb; ipdb.set_trace()
                    logits, loss = model(x, y, pad_token=pad_token)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    self.iters += 1
                    # decay the learning rate based on our progress
                    # if config.lr_decay:
                    #     # self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                    #     if self.iters < config.warmup_iters:
                    #         # linear warmup
                    #         lr_mult = float(self.iters) / float(max(1, config.warmup_iters))
                    #     else:
                    #         # cosine learning rate decay
                    #         progress = float(self.iters - config.warmup_iters) / float(max(1, config.final_iters - config.warmup_iters))
                    #         lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        
                    #     lr = config.learning_rate * lr_mult
                    #     for param_group in optimizer.param_groups:
                    #         param_group['lr'] = lr
                    # else:
                    
                    # StepLR 
                    if self.args.lr_decay_every is not None:
                        lr_mult = math.floor(math.ceil(self.iters / len(loader)) / int(self.args.lr_decay_every) )
                        lr = self.args.lr  * math.pow(self.args.lr_decay_rate, lr_mult)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                    else:
                        lr = self.config.learning_rate 

                    # report progress
                    wandb.log({
                        'train loss': loss.item(),
                        'lr': lr, 'epoch': epoch+1
                    }, step=self.iters)
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                wandb.log({'test loss': test_loss}, step=self.iters)
                return test_loss
            else:
                train_loss = float(np.mean(losses))
                logger.info("Avg Train Loss: %f", train_loss)
                wandb.log({'Avg Train Loss': train_loss}, step=self.iters)
                return train_loss

        best_loss = float('inf')
        for epoch in range(config.max_epochs):
            train_loss = run_epoch('train')
            if self.test_dataset is not None:
                with torch.no_grad():
                    test_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_dir is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint(epoch, test_loss=test_loss, train_loss=train_loss)
                print(f'Saved checkpoint at epoch {epoch}')
            # sample from the model
            if self.config.samples_dir is not None and (epoch+1) % self.config.sample_every == 0:
                # import ipdb; ipdb.set_trace()
                # inputs
                layouts = self.fixed_x.detach().cpu().numpy()
                input_layouts = [self.train_dataset.render(layout) for layout in layouts]
               
                # reconstruction
                x_cond = self.fixed_x.to(self.device)
                logits, _ = model(x_cond)
                probs = F.softmax(logits, dim=-1)
                _, y = torch.topk(probs, k=1, dim=-1)
                layouts = torch.cat((x_cond[:, :1], y[:, :, 0]), dim=1).detach().cpu().numpy()
                recon_layouts = [self.train_dataset.render(layout) for layout in layouts]
                
                # samples - random
                layouts = sample(model, x_cond[:, :6], steps=self.train_dataset.max_length,
                                 temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
                sample_random_layouts = [self.train_dataset.render(layout) for layout in layouts]
                
                # samples - deterministic
                layouts = sample(model, x_cond[:, :6], steps=self.train_dataset.max_length,
                                 temperature=1.0, sample=False, top_k=None).detach().cpu().numpy()
                sample_det_layouts = [self.train_dataset.render(layout) for layout in layouts]

                wandb.log({
                    "input_layouts": [wandb.Image(pil, caption=f'input_{epoch:02d}_{i:02d}.png')
                                      for i, pil in enumerate(input_layouts)],
                    "recon_layouts": [wandb.Image(pil, caption=f'recon_{epoch:02d}_{i:02d}.png')
                                      for i, pil in enumerate(recon_layouts)],
                    "sample_random_layouts": [wandb.Image(pil, caption=f'sample_random_{epoch:02d}_{i:02d}.png')
                                              for i, pil in enumerate(sample_random_layouts)],
                    "sample_det_layouts": [wandb.Image(pil, caption=f'sample_det_{epoch:02d}_{i:02d}.png')
                                           for i, pil in enumerate(sample_det_layouts)],
                }, step=self.iters)


    def save_checkpoint(self, epoch, test_loss=None, train_loss=None):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt_path = os.path.join(self.config.ckpt_dir, f'checkpoint_best.pth')
        logger.info("saving %s", ckpt_path)
        torch.save({'state_dict': raw_model.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'test_loss': test_loss},  ckpt_path)
    def evaluate(self):        
        config = self.config
        model =  self.model
        
        ckpt_path = os.path.join(self.config.ckpt_dir, 'checkpoint_best.pth')
        pt_model = torch.load(ckpt_path)

        model.load_state_dict(pt_model['state_dict'])
        epoch = pt_model['epoch']
        pt_train_loss = pt_model['train_loss']
        pt_test_loss = pt_model['test_loss']
        print(f'Loaded pretrained model\n Epoch: {epoch} \nTrain_Loss:{pt_train_loss} \nTest_Loss: {pt_test_loss}')
        
        pad_token = self.train_dataset.vocab_size - 1
        trainloader = DataLoader(self.train_dataset, shuffle=False, pin_memory=True,
                                drop_last=False, batch_size=config.batch_size, num_workers=config.num_workers)
        valloader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                                drop_last=False, batch_size=config.batch_size, num_workers=config.num_workers)
        
        eos_token = self.train_dataset.eos_token
        pad_token = self.train_dataset.pad_token
        
        eos_meter = AverageMeter()
        x_meter =AverageMeter()
        y_meter = AverageMeter()
        w_meter = AverageMeter()
        h_meter = AverageMeter()
        cat_meter = AverageMeter()
        
        model.eval()
        def run_one_epoch(loader):
            with torch.no_grad():
                loss_meter = AverageMeter()
                pbar = tqdm(enumerate(loader), total=len(loader)) 
                for it, (x, y) in pbar:
                    
                    logits, loss = model(x, y, pad_token=pad_token)
                    probs = F.softmax(logits, dim=-1)
                    _, ix = torch.topk(probs, k=1, dim=-1)
                    
                    for jj in range(x.shape[0]):
                        current_o = ix[jj,:,0]
                        current_y = y[jj,:]
                        eos_ind = torch.where(current_y==eos_token)
                        eos_acc = int(current_y[eos_ind] == current_o[eos_ind])
                        
                        box_cat_acc = current_y[0:eos_ind[0].item()] == current_o[0:eos_ind[0].item()]
                        box_cat_acc = box_cat_acc.view(-1,5)
                        n_boxs_gt = box_cat_acc.shape[0]
                        
                        x_acc = torch.sum(box_cat_acc[:,0]).item() / n_boxs_gt 
                        y_acc = torch.sum(box_cat_acc[:,1]).item() / n_boxs_gt
                        w_acc = torch.sum(box_cat_acc[:,2]).item() / n_boxs_gt
                        h_acc = torch.sum(box_cat_acc[:,3]).item() / n_boxs_gt
                        cat_acc = torch.sum(box_cat_acc[:,4]).item() / n_boxs_gt
                        
                        eos_meter.update(eos_acc)
                        x_meter.update(x_acc)
                        y_meter.update(y_acc)
                        w_meter.update(w_acc)
                        h_meter.update(h_acc)
                        cat_meter.update(cat_acc)
                        
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    loss_meter.update(loss.item())

            return loss_meter.avg

        train_loss = run_one_epoch(trainloader)
        val_loss = run_one_epoch(valloader)

        result_file = f'{self.args.result_dir}/result.txt'
        with open(result_file, 'a') as f:
            f.write(f'Average Training Loss:   {train_loss}\t num: {len(self.train_dataset)}\n')
            f.write(f'AVerage Validation Loss: {val_loss}\t num:{len(self.test_dataset)}\n')
            f.write(f'Eos Acc : {eos_meter.avg:4f}\n')
            f.write(f'X Acc : {x_meter.avg:4f}\n')
            f.write(f'Y Acc : {y_meter.avg:4f}\n')
            f.write(f'W Acc : {w_meter.avg:4f}\n')
            f.write(f'H Acc : {h_meter.avg:4f}\n')
            f.write(f'Cat Acc : {cat_meter.avg:4f}\n')
