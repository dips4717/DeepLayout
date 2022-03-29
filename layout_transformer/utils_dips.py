import pickle
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import os

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