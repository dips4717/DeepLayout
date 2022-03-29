"""
Similar to model_ML_AE, which was much completed with activations.
We simplify the model here observing the experiment from model_box_AE 
where we are able to reconstruct close to 100%  removing all the activations 
in the encoder, and only keeping activation on class and exist branch in decoder. 

Update: Copied from VTN codebase, but removed decode_structure function and gen function to bypass some dependencises 
RICO hierarchy...

author: d.manandhar@surrey.ac.uk
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from utils_dips import linear_assignment, bbox_iou, centerscale2box

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.config = config
        self.alpha_net = nn.Linear(self.config.box_emb_size, 1)


    def forward(self, att_feats,  att_masks=None):
        # The p_att_feats here is already projected
        #att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        B,N,_ = att_feats.shape
        dot = att_feats
        dot = torch.tanh(dot)                 # batch * att_size * att_hid_size
        dot = dot.view(B*N, -1)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)             # (batch * att_size) * 1
        dot = dot.view(-1, N)                 # batch * N

        weight = F.softmax(dot, dim=1)        # batch * att_size
        if att_masks is not None:           
            weight = weight * att_masks.view(-1, N).float()
            weight = weight / weight.sum(1, keepdim=True)               # normalize to 1
        att_feats_ = att_feats.view(-1, N, att_feats.size(-1))          # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size
       
        return att_res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 


class Sampler(nn.Module): 
    def __init__(self, config, probabilistic=True):

        super(Sampler, self).__init__()
        self.config = config
        self.probabilistic = probabilistic
        
        self.mlp1 = nn.Linear(config.box_emb_size*50,  self.config.emb_size)
        self.mlp2mu = nn.Linear( self.config.emb_size, self.config.emb_size)
        self.mlp2var = nn.Linear( self.config.emb_size, self.config.emb_size)

    def forward(self, x):
        # encode = torch.relu(self.mlp1(x))
        encode = self.mlp1(x)

        if self.probabilistic:
            # mu = self.mlp2mu(encode)
            # logvar = self.mlp2var(encode)
            # std = logvar.mul(0.5).exp_()
            # eps = torch.randn_like(std)
            # kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            # return eps.mul(std).add_(mu), kld
            
            mu = encode
            logvar = self.mlp2var(encode)
            var = torch.exp(0.5 * logvar)
            xi = torch.randn_like(var)
            z_sample = xi * var + mu 
            kld = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return z_sample, kld


        else:
            return encode


class Simple_MLPAE(nn.Module):
    
    def __init__(self, config):
        super(Simple_MLPAE, self).__init__()
        
        # General propoerties
        self.config = config
        self.max_box = 50
        
        if self.config.activation == 'relu':
            activation_layer = nn.ReLU()
        elif self.config.activation == 'elu':
            activation_layer = nn.ELU()
        elif self.config.activation == 'prelu':
            activation_layer = nn.PReLU()
        elif self.config.activation == 'none':
            activation_layer= nn.Identity()

        # Encoder attributes    
        self.box_encoder = nn.Linear(4,10)
        self.sem_embedding = nn.Embedding(26,10)
        self.concat_mlp = nn.Linear(20,20)                                    
        self.mlp_encoder = nn.Linear(20, config.box_emb_size)  # 20 * 50

        
        # Aggregation layer / Readout layer in GCN-CNN paper
        if self.config.aggregator == 'attention':
            self.agg_layer = Attention(self.config)  
        

        elif self.config.aggregator == 'mlp':
            # self.agg_layer = nn.Sequential(nn.Linear(config.box_emb_size*self.max_box, self.config.box_emb_size*10),
            #                                nn.Linear(self.config.box_emb_size*10,self.config.box_emb_size ))
            # self.agg_layer = nn.Linear(config.box_emb_size*self.max_box, self.config.emb_size)
            self.agg_layer = Sampler(config, probabilistic=config.probabilistic)


                                           
        # Decoder attributes:
        self.mlp_decoder = nn.Linear(config.box_emb_size, 1600)                                  

        #Decoder Braches: bbox, isexists, label    
        self.box_decoder = nn.Linear(32,4)                                   
        self.exist_decoder = nn.Sequential( nn.Linear(32,32),
                                            activation_layer,
                                            nn.Linear(32,1))
        self.label_decoder = nn.Sequential (nn.Linear(32,32),
                                           activation_layer,
                                            nn.Linear(32,26))

        if self.config.use_CNN_decoder_at_last:
            self.raster_decoder = Strided25ChannelCNNDecoder(config)


        # Losses
        self.bceloss = nn.BCEWithLogitsLoss(reduction='mean')
        self.semCELoss = nn.CrossEntropyLoss(reduction='mean')
        
        self.recon_class_acc = AverageMeter()
        self.recon_iou = AverageMeter()
        self.recon_exist_acc = AverageMeter()
        self.recon_edit_distance= AverageMeter()
        
    def box_loss(self, pred_boxes, gt_boxes, reduction='mean'):
        if self.config.box_loss == 'MSE':
            return F.mse_loss(pred_boxes, gt_boxes, reduction=reduction)
        elif self.config.box_loss == 'SmoothL1':
            return torch.nn.SmoothL1Loss(reduction='mean')(pred_boxes, gt_boxes) 


    def encoder(self, gtbox, box_exists, n_boxes, gtclass):
        B,N,_ = gtbox.shape                
        #gtbox = gtbox.view(B*N,-1)         #(B*N) x 4
        #gtclass = gtclass.view(B*N,-1).squeeze()   # (B*N)
        
        box = self.box_encoder(gtbox)               # 0
        semclass = self.sem_embedding (gtclass)     
        x = torch.cat((box, semclass), dim=-1)
        x = self.concat_mlp(x)
        z_i = self.mlp_encoder(x)  # B x N x 256
        
        # Zero out non existing features
        z_i = z_i*box_exists
        if self.config.aggregator == 'mlp':
            z = z_i.view(B,-1)
            z = self.agg_layer(z)
        
        elif self.config.aggregator == 'attention': 
            #Get masks
            # att_masks = torch.zeros((B,N), dtype=torch.float32, device= gtbox.device)
            # for ii, n in enumerate(n_boxes):
            #     att_masks[ii,:n]=1                    
            z = self.agg_layer(z_i, att_masks=None)
        
        return z_i, z

    def compute_loss(self, z_i, z, gtbox, box_exists, n_boxes, gtclass, raster=None):
        
        B,N,_ = gtbox.shape                

        if self.config.probabilistic:
            z, kld_loss = z
        else:
            kld_loss=0.0
        device = z.device
        x = self.mlp_decoder(z)
        x_i = x.view(B,self.max_box,-1)
        box_logits = self.box_decoder(x_i)            # 32x50x4
        exists_logits = self.exist_decoder(x_i)    # torch.Size([32, 50, 1])
        label_logits = self.label_decoder(x_i)     # torch.Size([32, 50, 26])

        if not self.config.use_hungarian_matching:

            if self.config.embLossOnly:
                # embloss = F.mse_loss(x_i,z_i)
                # embloss = F.l1_loss(x_i,z_i)
                embloss = F.smooth_l1_loss(x_i, z_i, reduction='mean')
                
                losses = {'class_loss': torch.tensor(0, dtype=torch.float32, device=x.device), 
                        'exists_loss': torch.tensor(0, dtype=torch.float32, device=x.device),
                        'box_loss': torch.tensor(0, dtype=torch.float32, device=x.device),
                        'unused_box_loss':torch.tensor(0, dtype=torch.float32, device=x.device),
                        'raster_loss':torch.tensor(0, dtype=torch.float32, device=x.device),
                        'embloss': embloss,
                        'kldloss': kld_loss}  
            else:
                label_logits = label_logits.view(-1, label_logits.shape[-1])
                gtclass = gtclass.view(-1)
                semantic_loss = self.semCELoss(label_logits, gtclass)
                exists_loss = F.binary_cross_entropy_with_logits(input=exists_logits, target=box_exists)
                
                if self.config.use_unusedbox_loss:
                    box_loss = self.box_loss(gtbox, box_logits, reduction='mean')
                else:
                    tmp_box_loss = torch.tensor(0, dtype=torch.float32, device=x.device)
                    for ii, n_box in enumerate(n_boxes):
                        box_logit_tmp = box_logits[ii,:n_box, :]
                        gtbox_tmp = gtbox[ii,:n_box, :]
                        tmp_box_loss +=  self.box_loss(gtbox_tmp, box_logit_tmp, reduction='mean')
                    box_loss = tmp_box_loss.mean()

                losses = {'class_loss': semantic_loss, 
                        'exists_loss': exists_loss,
                        'box_loss': box_loss,
                        'unused_box_loss':  torch.tensor(0, dtype=torch.float32, device=x.device),
                        'raster_loss': torch.tensor(0, dtype=torch.float32, device=x.device),
                        'kldloss': kld_loss}  

        else:

            losses = {'class_loss': torch.tensor(0, dtype=torch.float32, device=x.device),
                        'box_loss': torch.tensor(0, dtype=torch.float32, device=x.device),
                        'exists_loss': torch.tensor(0, dtype=torch.float32, device=x.device),
                        'unused_box_loss': torch.tensor(0, dtype=torch.float32, device=x.device),
                        'raster_loss': torch.tensor(0, dtype=torch.float32, device=x.device),
                        'kldloss': kld_loss}
            
            
            for i in range(x.shape[0]):
                pred_boxes = box_logits[i] # 50 x4
                # print(pred_boxes.shape) 
                gt_boxes = gtbox[i]         # 50x4 
                num_gt = n_boxes[i]         #
                exists_logits_ = exists_logits[i]  # 50 x1
                gtclass_ = gtclass[i]           # 50
                label_logits_ = label_logits[i]  # 50x26
                
                # Perform hungarian matching between predicted
                with torch.no_grad():
                    gt_boxes =  gt_boxes[:num_gt,:]
                    
                    num_pred = pred_boxes.shape[0]
                    pred_boxes_tiled = pred_boxes.unsqueeze(dim=0).repeat(num_gt, 1, 1) # num_gt x 10 x4
                    gt_boxes_tiled = gt_boxes.unsqueeze(dim=1).repeat(1, num_pred, 1) # num_gt x 10 x 2

                    dist_mat = self.box_loss(gt_boxes_tiled.view(-1, 4), pred_boxes_tiled.view(-1, 4), reduction = 'none' )
                    dist_mat = dist_mat.mean(dim=1)
                    dist_mat = dist_mat.view(-1, num_gt, num_pred) # [1, 3, 10]
                    _, matched_gt_idx, matched_pred_idx = linear_assignment(dist_mat) 

                # gather information based on matching
                sem_gt_labels = []
                sem_pred_logits = []
                box_gt = []
                box_pred = []
                exists_gt = torch.zeros_like(exists_logits_)
                for j in range(len(matched_gt_idx)):
                    sem_gt_labels.append(gtclass_[matched_gt_idx[j]])
                    sem_pred_logits.append(label_logits_[matched_pred_idx[j], :].view(1, -1))  
                    box_gt.append(gt_boxes[matched_gt_idx[j]].view(1, -1))
                    box_pred.append(pred_boxes[matched_pred_idx[j], :].view(1, -1))
                    exists_gt[matched_pred_idx[j], :] = 1
                
                # Compute losses
                # Semantic loss - class labels
                sem_pred_logits = torch.cat(sem_pred_logits, dim=0) # 9x26
                sem_gt_labels = torch.stack(sem_gt_labels,dim=0)
                semantic_loss = self.semCELoss(sem_pred_logits, sem_gt_labels)
                semantic_loss = semantic_loss.sum()
                
                # Exist loss 
                exists_loss = F.binary_cross_entropy_with_logits(\
                    input=exists_logits_, target=exists_gt, reduction='mean')
                exists_loss = exists_loss.sum()
                
                # Bounding box loss
                box_gt = torch.cat(box_gt, dim=0)
                box_pred = torch.cat(box_pred, dim=0)
                box_loss =self.box_loss(box_gt, box_pred)
                
                # train unused boxes to zeros
                if self.config.use_unusedbox_loss:
                     unused_box_loss = 0.0
                else:
                    unmatched_boxes = []
                    for i in range(num_pred):
                        if i not in matched_pred_idx:
                            unmatched_boxes.append(pred_boxes[i, 2:].view(1, -1))
                    if len(unmatched_boxes) > 0:
                        unmatched_boxes = torch.cat(unmatched_boxes, dim=0)
                        unused_box_loss = unmatched_boxes.pow(2).sum() * 0.01
                    else:
                        unused_box_loss = 0.0

                losses['class_loss'] += semantic_loss
                losses['box_loss'] += box_loss
                losses['exists_loss'] += exists_loss
                losses['unused_box_loss'] += unused_box_loss

            for loss_name in losses.keys():
                losses[loss_name] /= x.shape[0]  # Divide by batch size

        if raster is not None:
            raster_out = self.raster_decoder(x)     
            mse_loss = F.mse_loss(raster, raster_out)
            losses['raster_loss'] = mse_loss

        return losses 
            
            
    def forward(self, x): 
        pass

    def decode_structure(self,z_i, z, gt=None):
        if gt is not None:
            gt_boxes, gt_box_exists, gt_n_boxes, gt_label_ids =gt
        
        if self.config.probabilistic:
            z, kld_loss = z
        else:
            kld_loss=0.0

        x_i = self.mlp_decoder(z)
        x2 = x_i.view(z.shape[0],self.max_box,-1)
        
        box_logits = self.box_decoder(x2)      #torch.Size([1, 50, 4])
        exists_logits = self.exist_decoder(x2)
        label_logits = self.label_decoder(x2)   
        
        #Since only one data at a time.
        box_logits, exists_logits, label_logits = box_logits.squeeze(0), exists_logits.squeeze(0), label_logits.squeeze(0) 
        
        child_nodes = [] 
        n_pred_box = torch.sum(torch.sigmoid(exists_logits) > 0.5)
        self.recon_exist_acc.update(int(gt_n_boxes==n_pred_box))
               
        # print(f'isexist logits sigmoid: {(torch.sigmoid(exists_logits)>0.5).squeeze()}')
        # print (f'Predicted isexist sum {n_pred_box}')
        # print(f'n_box gt: {gt_n_boxes}')
        
        
        
        #Iou computation
        A=[]
        B=[]
        for ii in range(gt_n_boxes):
            A.append(centerscale2box(gt_boxes[ii,:]))
            B.append(centerscale2box(box_logits[ii,:]))
        A = np.array(A)
        B = np.array(B)
        
        ious = bbox_iou(A, B)
        current_iou = ious.diagonal().mean()
        self.recon_iou.update(current_iou)   
        
        K = min(n_pred_box, gt_n_boxes)
        edit_distance = max(n_pred_box, gt_n_boxes) - K

        for ci in range(self.max_box):
            if torch.sigmoid(exists_logits[ci, :]).item() > 0.5:
                
                idx = np.argmax(label_logits[ci, :].cpu().numpy())
                _, pred_idx = torch.max(label_logits[ci,:], dim=0)
                _, sort_inds = torch.sort(label_logits[ci, :], descending=True)
                
                full_label = Hierarchy.ID2SEM[idx]
                gt_label = Hierarchy.ID2SEM[gt_label_ids[ci].item()]
                box = box_logits[ci,:]
                
                # print (f'GT box: {gt_boxes[ci,:]}')
                # print(f'Pred box: {box}')
                # print(f'GT idx: {gt_label_ids[ci]}   GT label: { Hierarchy.ID2SEM[gt_label_ids[ci].item()]}')
                # print (f'Pred idx: {idx}   Pred label: {full_label}  ')
                # print (f'Pred label sort indices: {sort_inds} ')
                self.recon_class_acc.update(int(full_label==gt_label))
                if full_label!=gt_label:
                    edit_distance+=1
                    
                node = Node(label = full_label)
                if self.config.bbox_format == 'center_scale':
                    node.set_center_scale(box.view(-1))
                elif self.config.bbox_format == 'XYAbsolute':
                    node.box = np.array(box.view(-1).detach().cpu())
                elif self.config.bbox_format == 'XYRelative':
                    box = torch.mul(box.detach().cpu(),torch.tensor([1440.0 , 2560.0, 1440.0, 2560.0], dtype=torch.float32) )
                    node.box = np.array(box.view(-1).detach().cpu())
            
                child_nodes.append(node)

        self.recon_edit_distance.update(edit_distance)
        
        node = Node(label= '[ROOT]', children=child_nodes)
        #node.set_center_scale(torch.tensor([0,0,1,1]))
        node.set_xywh(torch.tensor([-1,-1,1,1]))
        obj = RicoHierarchy(node)

        return obj
    
    
    
    def decode_structure_gen(self, z, gt=None):
        x = self.mlp_decoder(z)
        x2 = x.view(z.shape[0],self.max_box,-1)
        
                
        box_logits = self.box_decoder(x2)      #torch.Size([1, 50, 4])
        exists_logits = self.exist_decoder(x2)
        label_logits = self.label_decoder(x2)   
        
        #Since only one data at a time.
        box_logits, exists_logits, label_logits = box_logits.squeeze(0), exists_logits.squeeze(0), label_logits.squeeze(0) 
        
        child_nodes = [] 
        n_pred_box = torch.sum(torch.sigmoid(exists_logits) > 0.5)        
        
        for ci in range(50):
            if torch.sigmoid(exists_logits[ci, :]).item() > self.config.isexist_thres:
                
                idx = np.argmax(label_logits[ci, :].cpu().numpy())
                _, pred_idx = torch.max(label_logits[ci,:], dim=0)
                _, sort_inds = torch.sort(label_logits[ci, :], descending=True)
                
                full_label = Hierarchy.ID2SEM[idx]
                box = box_logits[ci,:]
                node = Node(label = full_label)
                if self.config.bbox_format == 'center_scale':
                    node.set_center_scale(box.view(-1))
                elif self.config.bbox_format == 'XYAbsolute':
                    node.box = np.array(box.view(-1).detach().cpu())
                elif self.config.bbox_format == 'XYRelative':
                    box = torch.mul(box.detach().cpu(),torch.tensor([1440.0 , 2560.0, 1440.0, 2560.0], dtype=torch.float32) )
                    node.box = np.array(box.view(-1).detach().cpu())
            
                child_nodes.append(node)

        node = Node(label= '[ROOT]', children=child_nodes)
        #node.set_center_scale(torch.tensor([0,0,1,1]))
        node.set_xywh(torch.tensor([-1,-1,1,1]))
        obj = RicoHierarchy(node)

        return obj



class Strided25ChannelCNNDecoder(nn.Module):
    def __init__(self,config):
        super(Strided25ChannelCNNDecoder, self).__init__()
        self.conf = config   
        self.decoder_FC = nn.Linear(1600, 32*14*6)
        self.decoder_raster = nn.Sequential(
             nn.ConvTranspose2d(32,25,3,  stride=2),
             nn.ReLU(),
             nn.ConvTranspose2d(25,25,3, stride=2),
             nn.ReLU(),
             nn.ConvTranspose2d(25,25,3, stride=2),
             nn.ReLU(),
             nn.ConvTranspose2d(25,25,3, stride=2),
             nn.ReLU()
             )
    
    def forward(self,x):
        x = F.relu(self.decoder_FC(x))
        x = x.reshape(x.size(0),32,14,6)
        x = self.decoder_raster(x)
        return x    
