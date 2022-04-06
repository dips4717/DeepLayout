import pickle
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import os
import torch.nn.functional as F
from matplotlib import pyplot as plt 
from PIL import Image

def pickle_save(fn,obj):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)
        print(f'obj saved to {fn}')

def pickle_load(fn):
    with open(fn,'rb') as f:
        obj = pickle.load(f)
    print (f'Object loaded from {fn}')    
    return obj

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception("value not allowed")

def generate_square_subsequent_mask(sz,device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample_transEnc_conditional(model, x, steps, seq_all=None, temperature=1.0, sample=False, top_k=None, inference=False):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    # block_size = model.module.get_block_size() if hasattr(model, "module") else model.getcond_block_size()
    block_size = model.decoder.module.get_block_size() if hasattr(model, "module") else model.decoder.get_block_size()

    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits, _,_ = model(x_cond, seq_all=seq_all, inference=inference)
        
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x


def create_mask(src, tgt, PAD_IDX, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def save_checkpoint(args, model, epoch, test_loss=None, train_loss=None):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = model.module if hasattr(model, "module") else model
        ckpt_path = os.path.join(args.ckpt_dir, 'checkpoint_best.pth')
       
        save_dict = {
                     'state_dict': raw_model.state_dict(),
                     'epoch': epoch,
                     'train_loss': train_loss,
                     'test_loss': test_loss }
        torch.save(save_dict, ckpt_path)


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
        

# row_counts, col_counts: row and column counts of each distance matrix (assumed to be full if given)
def linear_assignment(distance_mat, row_counts=None, col_counts=None):
    batch_ind = []
    row_ind = []
    col_ind = []
    for i in range(distance_mat.shape[0]):
        # print(f'{i} / {distance_mat.shape[0]}')

        dmat = distance_mat[i, :, :]
        if row_counts is not None:
            dmat = dmat[:row_counts[i], :]
        if col_counts is not None:
            dmat = dmat[:, :col_counts[i]]

        rind, cind = linear_sum_assignment(dmat.to('cpu').numpy())
        rind = list(rind)
        cind = list(cind)

        if len(rind) > 0:
            rind, cind = zip(*sorted(zip(rind, cind)))
            rind = list(rind)
            cind = list(cind)

        # complete the assignemnt for any remaining non-active elements (in case row_count or col_count was given),
        # by assigning them randomly
        #if len(rind) < distance_mat.shape[1]:
        #    rind.extend(set(range(distance_mat.shape[1])).difference(rind))
        #    cind.extend(set(range(distance_mat.shape[1])).difference(cind))

        batch_ind += [i]*len(rind)
        row_ind += rind
        col_ind += cind    

    return batch_ind, row_ind, col_ind

def normalize(value, value_min, value_max):
    """Map value from [value_min, value_max] to [-1, 1]"""
    return 2 * ((value - value_min) / (value_max - value_min)) - 1

def unnormalize(value, value_min, value_max):
    """Map value from [-1, 1] to [value_min, value_max]"""
    return ((value + 1) / 2.0 * (value_max - value_min) + value_min)



def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.
    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.
    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)

def centerscale2box(center_scale):
    center_x, center_y, scale_x, scale_y = center_scale.cpu().numpy().squeeze()
    center_x = unnormalize(center_x, 0, 1440)
    center_y = unnormalize(center_y, 0, 2560)
    scale_x = scale_x * 1440
    scale_y = scale_y * 2560
    x1 = center_x - scale_x / 2; x2 = center_x + scale_x / 2
    y1 = center_y - scale_y / 2; y2 = center_y + scale_y / 2
    box = np.array([x1, y1, x2, y2])
    return box 

def plot_retrieved_images_and_uis(sort_inds, q_fnames, g_fnames, avgIouArray=None, avgPixAccArray=None):
    
    base_im_path = '/mnt/amber/scratch/Dipu/RICO/combined/'
    base_ui_path = '/mnt/amber/scratch/Dipu/RICO/semantic_annotations/'
    
    for i in range((sort_inds.shape[0])): #range(1): 
#    for i in  range(2):    
        if i == 10:
            break
        q_path = base_im_path + q_fnames[i] + '.jpg'
        q_img  =  Image.open(q_path).convert('RGB')
        q_ui_path = base_ui_path + q_fnames[i] + '.png'
        q_ui = Image.open(q_ui_path).convert('RGB')
        
        fig, ax = plt.subplots(2,6, figsize=(30, 12), constrained_layout=True)
        plt.setp(ax,  xticklabels=[],  yticklabels=[])
        fig.suptitle('Query-%s)'%(i), fontsize=20)
        fig = plt.figure(1)
#        fig.set_size_inches(30, 12)
#        plt.subplots_adjust(bottom = 0.1, top=10)
        #f1 = fig.add_subplot(2,6,1)
        
        ax[0,0].imshow(q_ui)
        ax[0,0].axis('off')
        ax[0,0].set_title('Query: %s '%(i) + q_fnames[i] + '.png')
        ax[1,0].imshow(q_img)
        ax[1,0].axis('off') 
        ax[1,0].set_title('Query: %s '%(i) + q_fnames[i] + '.jpg')
        #plt.pause(0.1)
     
        for j in range(5):
            path = base_im_path + g_fnames[sort_inds[i][j]] + '.jpg'
           # print(g_fnames[sort_inds[i][j]] )
            im = Image.open(path).convert('RGB')
            ui_path = base_ui_path + g_fnames[sort_inds[i][j]] + '.png'
            #print(g_fnames[sort_inds[i][j]]) 
            ui = Image.open(ui_path).convert('RGB')
            
            ax[0,j+1].imshow(ui)
            ax[0,j+1].axis('off')
            if avgIouArray is None:
                ax[0,j+1].set_title('Rank: %s  '%(j+1)  + g_fnames[sort_inds[i][j]])            
            else: 
                ax[0,j+1].set_title('Rank: %s  '%(j+1)  + g_fnames[sort_inds[i][j]] \
                                     + '.png\nAvg IoU: %.3f'%(avgIouArray[i][j])
                                     + '\nAvg PixAcc: %.3f'%(avgPixAccArray[i][j]))
           
            
            ax[1,j+1].imshow(im)
            ax[1,j+1].axis('off')
            ax[1,j+1].set_title('Rank: %s  '%(j+1) + g_fnames[sort_inds[i][j]] + '.jpg')
            
#        directory =  'Retrieval_Results_Iou_PixAcc/{}/Gallery_Only/'.format(model_name)
        directory =  'runs/RICO_Image/retrievals/'
        if not os.path.exists(directory):
            os.makedirs(directory)  
            
        plt.savefig( directory + str(i) + '.png')
       # plt.pause(0.1)
        plt.close()
        #print('Wait')
        print('Plotting the retrieved images: {}'.format(i))