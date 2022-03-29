"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
from email.policy import strict
import os
import sys
import math
import logging
import wandb
import torch.nn as nn

from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from utils import sample

from model import Search_Synthesis

logger = logging.getLogger(__name__)
from eval_metrics.get_overall_Classwise_IOU import get_overall_Classwise_IOU
from eval_metrics.get_overall_pix_acc import get_overall_pix_acc
from scipy.spatial.distance import cdist

sys.path.append('bbox_lib')
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
import pickle

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
    num_workers =8 # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, base_model, train_dataset, test_dataset, query_dataset, config, args):
        self.device =  args.device
        self.args = args
        s2s_model = base_model
        PT_model = 'runs/RICO/checkpoints/checkpoint.pth'
        s2s_model.load_state_dict(torch.load(PT_model, map_location=self.device))
        s2s_model.ln_f = nn.Identity()
        s2s_model.head = nn.Identity()
        
        #Freeze the Seq2seq model
        for param in s2s_model.parameters():
            param.requires_grad = False
        
        self.model = Search_Synthesis(s2s_model)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.query_dataset = query_dataset
        self.config = config
        self.iters = 0
        self.fixed_x = None
        self.fixed_y = None
        self.criterion = nn.MSELoss()
        print("Using wandb")
        wandb.init(project='LayoutTransformer', name=args.dataset_name + args.exp)
        wandb.config.update(args)


        self.model = self.model.to(self.device)    
            
            
    def save_checkpoint(self, model, epoch=None):
        # DataParallel wrappers keep raw model object in .module attribute
        #raw_model = self.model.module if hasattr(self.model, "module") else self.model
        ckpt_path = os.path.join(self.config.ckpt_dir, 'checkpoint.pth') if epoch==None else os.path.join(self.config.ckpt_dir, f'checkpoint_ep{epoch}.pth')
        logger.info("saving %s", ckpt_path)
        torch.save(model.state_dict(), ckpt_path)
    

    def ret_eval(self, model=None, epoch=None):
        config = self.config
        if model==None:
            trained_path =  os.path.join(self.config.ckpt_dir, f'checkpoint_ep{epoch}.pth')
            model.load_state_dict(torch.load(trained_path, map_location=self.device))
        
        save_file = os.path.join(self.args.result_dir, 'result.txt')
        model.eval()
        pad_token = self.train_dataset.vocab_size - 1
        gallery_dataset = self.test_dataset
        query_dataset = self.query_dataset
        gallery_loader = DataLoader(gallery_dataset, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
        query_loader = DataLoader(query_dataset, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)
        boundingBoxes = getBoundingBoxes_from_info()

        def extract_features(model, loader, split='gallery'):
            feat = []
            fnames = []             
            pbar = tqdm(enumerate(loader), total=len(loader))
            for it, (x, y, img, img_id) in pbar:             
                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                    
                # forward the model
                with torch.set_grad_enabled(False):
                    xlatent = model.forward_latent(x, y, pad_token=pad_token)
                    xlatent = xlatent.detach().cpu()
                    feat.append(xlatent)
                    fnames+=img_id
            print (f'Extracted features from {len(fnames)} uis')
            return feat, fnames

        q_feat, q_fnames = extract_features(model, query_loader, split='gallery')
        g_feat, g_fnames = extract_features(model, gallery_loader, split='query')
         
        q_feat = np.concatenate(q_feat)
        g_feat = np.concatenate(g_feat)

        feat_dict = {'q_feat': q_feat, 'g_feat':g_feat, 'q_fnames': q_fnames, 'g_fnames': g_fnames}
        with open(self.args.result_dir+'/feat_dict.pkl', 'wb') as f:
            pickle.dump(feat_dict, f)
        print(f'Obj saved to {self.args.result_dir}')

        
        distances = cdist(q_feat, g_feat, metric= 'euclidean')
        sort_inds = np.argsort(distances)
        
        overallMeanClassIou, overallMeanWeightedClassIou, classwiseClassIoU = get_overall_Classwise_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10])
        overallMeanAvgPixAcc, overallMeanWeightedPixAcc, classPixAcc = get_overall_pix_acc(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10])     
          
        print('\n\nep:%s'%(epoch))
        print('overallMeanClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanClassIou]) + '\n')        
        print('overallMeanAvgPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanAvgPixAcc]) + '\n')
        
            
        with open(save_file, 'a') as f:
            f.write('\n\ep: {}\n'.format(epoch))
            f.write('overallMeanClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanClassIou]) + '\n')
            f.write('overallMeanAvgPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanAvgPixAcc]) + '\n')

           

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        pad_token = self.train_dataset.vocab_size - 1

        def run_epoch(split, model, epoch=None):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, img, img_id) in pbar:
                
                img = img.to(self.device)
                img = F.interpolate(img, size= [239,111])
                
                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                    
                # forward the model
                with torch.set_grad_enabled(is_train):
                    out = model(x, y, pad_token=pad_token) 
                    loss = self.criterion(out, img) 
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
                    if config.lr_decay:
                        # self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.iters < config.warmup_iters:
                            # linear warmup
                            lr_mult = float(self.iters) / float(max(1, config.warmup_iters))
                        else:
                            # cosine learning rate decay
                            progress = float(self.iters - config.warmup_iters) / float(max(1, config.final_iters - config.warmup_iters))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        if epoch%self.args.lr_decay_every==0 and epoch>1  :
                            lr = config.learning_rate/10
                        else:
                            lr = config.learning_rate
                        
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                    # report progress
                    wandb.log({
                        'train loss': loss.item(),
                        'lr': lr, 'epoch': epoch+1
                    }, step=self.iters)
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")
                
            loss = float(np.mean(losses))
            return model, loss 


        best_loss = float('inf')
        for epoch in range(config.max_epochs):
            
            model, train_loss = run_epoch('train', model=model, epoch=epoch)
            if epoch+1%5 == 0:
                self.ret_eval(model=model, epoch=None)
            model.train()

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = train_loss < best_loss
            if self.config.ckpt_dir is not None and good_model:
                best_loss = train_loss
                self.save_checkpoint(model, epoch=epoch)

            # sample from the model
            # if self.config.samples_dir is not None and (epoch+1) % self.config.sample_every == 0:
            #     # import ipdb; ipdb.set_trace()
            #     # inputs
            #     layouts = self.fixed_x.detach().cpu().numpy()
            #     input_layouts = [self.train_dataset.render(layout) for layout in layouts]
            #     # for i, layout in enumerate(layouts):
            #     #     layout = self.train_dataset.render(layout)
            #     #     layout.save(os.path.join(self.config.samples_dir, f'input_{epoch:02d}_{i:02d}.png'))

            #     # reconstruction
            #     x_cond = self.fixed_x.to(self.device)
            #     logits, _ = model(x_cond)
            #     probs = F.softmax(logits, dim=-1)
            #     _, y = torch.topk(probs, k=1, dim=-1)
            #     layouts = torch.cat((x_cond[:, :1], y[:, :, 0]), dim=1).detach().cpu().numpy()
            #     recon_layouts = [self.train_dataset.render(layout) for layout in layouts]
            #     # for i, layout in enumerate(layouts):
            #     #     layout = self.train_dataset.render(layout)
            #     #     layout.save(os.path.join(self.config.samples_dir, f'recon_{epoch:02d}_{i:02d}.png'))

            #     # samples - random
            #     layouts = sample(model, x_cond[:, :6], steps=self.train_dataset.max_length,
            #                      temperature=1.0, sample=True, top_k=5).detach().cpu().numpy()
                
            #     sample_random_layouts = [self.train_dataset.render(layout) for layout in layouts]
            #     # for i, layout in enumerate(layouts):
            #     #     layout = self.train_dataset.render(layout)
            #     #     layout.save(os.path.join(self.config.samples_dir, f'sample_random_{epoch:02d}_{i:02d}.png'))

            #     # samples - deterministic
            #     layouts = sample(model, x_cond[:, :6], steps=self.train_dataset.max_length,
            #                      temperature=1.0, sample=False, top_k=None).detach().cpu().numpy()
            #     sample_det_layouts = [self.train_dataset.render(layout) for layout in layouts]
            #     # for i, layout in enumerate(layouts):
            #     #     layout = self.train_dataset.render(layout)
            #     #     layout.save(os.path.join(self.config.samples_dir, f'sample_det_{epoch:02d}_{i:02d}.png'))

            #     wandb.log({
            #         "input_layouts": [wandb.Image(pil, caption=f'input_{epoch:02d}_{i:02d}.png')
            #                           for i, pil in enumerate(input_layouts)],
            #         "recon_layouts": [wandb.Image(pil, caption=f'recon_{epoch:02d}_{i:02d}.png')
            #                           for i, pil in enumerate(recon_layouts)],
            #         "sample_random_layouts": [wandb.Image(pil, caption=f'sample_random_{epoch:02d}_{i:02d}.png')
            #                                   for i, pil in enumerate(sample_random_layouts)],
            #         "sample_det_layouts": [wandb.Image(pil, caption=f'sample_det_{epoch:02d}_{i:02d}.png')
            #                                for i, pil in enumerate(sample_det_layouts)],
            #     }, step=self.iters)


def getBoundingBoxes_from_info(info_file = 'data/rico_box_info.pkl'):
    allBoundingBoxes = BoundingBoxes()
    info = pickle.load(open(info_file, 'rb'))
    #files = glob.glob(data_dir+ "*.json")
    for imageName in info.keys():
        count = info[imageName]['nComponent']
        for i in range(count):
            box = info[imageName]['xywh'][i]
            bb = BoundingBox(
                imageName,
                info[imageName]['componentLabel'][i],
                box[0],
                box[1],
                box[2],
                box[3],
                iconClass=info[imageName]['iconClass'],
                textButtonClass=info[imageName]['textButtonClass'])
            allBoundingBoxes.addBoundingBox(bb) 
    print('Collected {} bounding boxes from {} images'. format(allBoundingBoxes.count(), len(info) ))         
#    testBoundingBoxes(allBoundingBoxes)
    return allBoundingBoxes