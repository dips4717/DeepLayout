"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import os
import math
import logging
from xml.sax.handler import property_dom_node
import wandb

from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from gpt2 import Conv1D
from utils import sample
from utils_dips import AverageMeter
logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 8 #64
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

    def __init__(self, s2s_model, search_model, train_dataset, test_dataset, config, args, config_search_model):
        search_model = search_model
        self.model = s2s_model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.args = args
        self.config_search_model = config_search_model

        self.iters = 0
        self.fixed_x = None
        self.fixed_y = None
        self.fixed_z = None
        
        if not args.evaluate:
            print("Using wandb")
            wandb.init(project='LayoutGeneration', name=args.exp_name)
            wandb.config.update(args)
        
        if args.freeze_search_model:
            for p in search_model.parameters():
                p.requires_grad=False
            print('Search Model Parameters are frozen')
        else:
            print('Search Model Parameters are TRAINABLE')
        self.search_model = search_model

        
        self.device = args.device
        if torch.cuda.is_available():
            #torch.cuda.current_device()
            # self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.model = self.model.to(self.device) 
            self.search_model = self.search_model.to(self.device) 
         

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, Conv1D )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            {"params": self.search_model.parameters()}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        print(optimizer)
        return optimizer

            
    def save_checkpoint(self, epoch, test_loss=None, train_loss=None):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt_path = os.path.join(self.config.ckpt_dir, f'checkpoint_best.pth')
        logger.info("saving %s", ckpt_path)
        save_dict = {
                     'model_search': self.search_model.state_dict(),
                     'model_synthesis': raw_model.state_dict(),
                     'epoch': epoch,
                     'train_loss': train_loss,
                     'test_loss': test_loss }
        torch.save(save_dict, ckpt_path)

    def train(self):
        #model, config = self.model, self.config
        #raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = self.configure_optimizers(self.config)
        pad_token = self.train_dataset.vocab_size - 1

        def run_epoch(split):
            is_train = split == 'train'
            self.model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=self.config.batch_size,
                                num_workers=self.config.num_workers)

            losses = []
            if self.args.search_loss:
                search_losses_list = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, uxid, boxes, box_exists, n_boxes, label_ids) in pbar:

                # if epoch == 0 and not is_train:
                #     self.fixed_x = x[:min(4, len(x))]
                #     self.fixed_y = y[:min(4, len(y))]

                # place data on the correct device
                # print (box_exists.shape, type(box_exists))
                # print(boxes.shape, type(boxes))
                # print(label_ids.shape, type(label_ids))
                boxes, box_exists  = boxes.to(self.device), box_exists.to(self.device)
                label_ids = label_ids.to(self.device)    
                x = x.to(self.device)
                y = y.to(self.device)


                # forward the model
                with torch.set_grad_enabled(is_train):
                    # import ipdb; ipdb.set_trace()

                    if not self.args.search_loss:
                        _ , z = self.search_model.encoder(boxes, box_exists, n_boxes, label_ids)
                        logits, loss = self.model(x, z, y, pad_token=pad_token) # x input seq, z latent vector acting as condition, y target seq
                        final_loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                        losses.append(loss.item())
                    else:
                        z_i, z =  self.search_model.encoder(boxes, box_exists, n_boxes, label_ids)
                        search_losses  = self.search_model.compute_loss(z_i, z, boxes, box_exists, n_boxes, label_ids, raster=None)
                        search_losses['box_loss'] *= self.config_search_model.loss_weight_box
                        search_losses['class_loss'] *= self.config_search_model.loss_weight_semantic 
                        search_losses['unused_box_loss'] *= self.config_search_model.loss_weight_box
                        search_losses['exists_loss'] *= self.config_search_model.loss_weight_exists  
                        search_losses['raster_loss'] *= self.config_search_model.loss_weight_raster
                        if self.config_search_model.probabilistic:
                            search_losses['kldloss'] =   - (losses['kldloss'].mean())
                            search_losses['kldloss'] *= self.config_search_model.loss_weight_kld
    
                        else:
                            search_losses['kldloss'] = torch.tensor(0, dtype=torch.float32, device=self.device)
                        total_search_loss = torch.tensor(0.0, device=self.device)
                        for value in search_losses.values():
                            total_search_loss += value

                        logits, syn_loss = self.model(x, z, y, pad_token=pad_token) # x input seq, z latent vector acting as condition, y target seq
                        syn_loss = syn_loss.mean()
                        losses.append(syn_loss.item())
                        search_losses_list.append(total_search_loss.item())
                        final_loss = total_search_loss + syn_loss 


                if epoch == 0 and not is_train:
                    self.fixed_x = x[:min(4, len(x))]
                    self.fixed_y = y[:min(4, len(y))]
                    self.fixed_z = z[:4,:]   # We also save the latent vectors for test set and regular plotting 

                if is_train:

                    # backprop and update the parameters
                    self.model.zero_grad()
                    if self.args.search_loss:
                        self.search_model.zero_grad()
                    final_loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                    optimizer.step()
                    self.iters += 1
                    
                    # # decay the learning rate based on our progress
                    # if self.config.lr_decay:
                    #     # self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                    #     if self.iters < self.config.warmup_iters:
                    #         # linear warmup
                    #         lr_mult = float(self.iters) / float(max(1, self.config.warmup_iters))
                    #     else:
                    #         # cosine learning rate decay
                    #         progress = float(self.iters - self.config.warmup_iters) / float(max(1, self.config.final_iters - self.config.warmup_iters))
                    #         lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    #     lr = self.config.learning_rate * lr_mult
                    #     for param_group in optimizer.param_groups:
                    #         param_group['lr'] = lr
                    # else:
                    #     lr = self.config.learning_rate

                    # StepLR 
                    if self.args.lr_decay_every is not None:
                        lr_mult = math.floor(math.ceil(self.iters / len(loader)) / int(self.args.lr_decay_every) )
                        lr = self.args.lr  * math.pow(self.args.lr_decay_rate, lr_mult)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                    else:
                        lr = self.config.learning_rate 

                    # report progress
                    if not self.args.search_loss:
                        wandb.log({
                            'train loss': final_loss.item(),  # this is indeed syn_loss
                            'lr': lr, 'epoch': epoch+1
                        }, step=self.iters)
                    else:
                        wandb.log({
                            'train loss': syn_loss.item(),
                            'search loss': total_search_loss.item(),
                            'lr': lr, 'epoch': epoch+1
                        }, step=self.iters)
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {final_loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                wandb.log({'test loss': test_loss}, step=self.iters)
                return test_loss
            else:
                avg_train_loss = float(np.mean(losses))
                logger.info("Avg Train Loss: %f", avg_train_loss)
                wandb.log({'Avg Train Loss': avg_train_loss}, step=self.iters)
                if self.args.search_loss:
                    avg_search_loss = float(np.mean(search_losses_list))
                    logger.info("Avg Search Loss: %f ", avg_search_loss)
                    wandb.log({'Avg Search Loss': avg_search_loss}, step=self.iters)
                
                return avg_train_loss

        best_loss = float('inf')
        for epoch in range(self.config.max_epochs):
            
            train_loss = run_epoch('train')  # synthesis loss
            if self.test_dataset is not None:
                with torch.no_grad():
                    test_loss = run_epoch('test')

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_dir is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint(epoch, test_loss=test_loss, train_loss=train_loss)

            # sample from the model
            if self.config.samples_dir is not None and (epoch+1) % self.config.sample_every == 0:
                # import ipdb; ipdb.set_trace()
                # inputs
                layouts = self.fixed_x.detach().cpu().numpy()
                input_layouts = [self.train_dataset.render(layout) for layout in layouts]
        
                # reconstruction
                x_cond = self.fixed_x.to(self.device)
                z_cond = self.fixed_z
                logits, _ = self.model(x_cond, z_cond)
                probs = F.softmax(logits, dim=-1)
                _, y = torch.topk(probs, k=1, dim=-1)
                layouts = torch.cat((x_cond[:, :1], y[:, :, 0]), dim=1).detach().cpu().numpy()
                recon_layouts = [self.train_dataset.render(layout) for layout in layouts]

                # samples - random
                layouts = sample(self.model, x_cond[:, :6], steps=self.train_dataset.max_length,
                                 z=self.fixed_z, temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
                
                sample_random_layouts = [self.train_dataset.render(layout) for layout in layouts]
      
                # samples - deterministic
                layouts = sample(self.model, x_cond[:, :6], steps=self.train_dataset.max_length,
                                 z=self.fixed_z, temperature=1.0, sample=False, top_k=None).detach().cpu().numpy()
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


    def evaluate(self):        
        config = self.config
        model =  self.model
        model_search = self.search_model
        ckpt_path = os.path.join(self.config.ckpt_dir, 'checkpoint_best.pth')
        pt_model = torch.load(ckpt_path)

        model.load_state_dict(pt_model['model_synthesis'])
        model_search.load_state_dict(pt_model['model_search'])
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
        model_search.eval()    
        def run_one_epoch(loader):
            with torch.no_grad():
                loss_meter = AverageMeter()
                pbar = tqdm(enumerate(loader), total=len(loader)) 
                for it, (x, y, uxid, boxes, box_exists, n_boxes, label_ids) in pbar:
                    
                    boxes, box_exists  = boxes.to(self.device), box_exists.to(self.device)
                    label_ids = label_ids.to(self.device)    
                    x = x.to(self.device)
                    y = y.to(self.device)

                    _,z = model_search.encoder(boxes, box_exists, n_boxes, label_ids)
                    logits, loss = model(x, z, y, pad_token=pad_token)

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
