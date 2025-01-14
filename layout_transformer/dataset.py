import numpy as np
import torch
from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageDraw, ImageOps
import json
import pickle
import os
import random

from utils import trim_tokens, gen_colors


class Padding(object):
    def __init__(self, max_length, vocab_size):
        self.max_length = max_length
        self.bos_token = vocab_size - 3
        self.eos_token = vocab_size - 2
        self.pad_token = vocab_size - 1

    def __call__(self, layout):
        # grab a chunk of (max_length + 1) from the layout

        chunk = torch.zeros(self.max_length+1, dtype=torch.long) + self.pad_token  # 468
        # Assume len(item) will always be <= self.max_length:
        chunk[0] = self.bos_token
        chunk[1:len(layout)+1] = layout
        chunk[len(layout)+1] = self.eos_token

        x = chunk[:-1]
        y = chunk[1:]
        return {'x': x, 'y': y}

class Padding2(object):
    def __init__(self, max_length, vocab_size):
        self.max_length = max_length
        self.bos_token = vocab_size - 3
        self.eos_token = vocab_size - 2
        self.pad_token = vocab_size - 1

    def __call__(self, layout):
        # grab a chunk of (max_length + 1) from the layout

        chunk = torch.zeros(self.max_length+1, dtype=torch.long) + self.pad_token  # 468
        # Assume len(item) will always be <= self.max_length:
        chunk[0] = self.bos_token
        chunk[1:len(layout)+1] = layout
        chunk[len(layout)+1] = self.eos_token
       
        return {'x': chunk, 'y': chunk}


class MNISTLayout(MNIST):

    def __init__(self, root, train=True, download=True, threshold=32, max_length=None):
        super().__init__(root, train=train, download=download)
        self.vocab_size = 784 + 3  # bos, eos, pad tokens
        self.bos_token = self.vocab_size - 3
        self.eos_token = self.vocab_size - 2
        self.pad_token = self.vocab_size - 1

        self.threshold = threshold
        self.data = [self.img_to_set(img) for img in self.data]
        self.max_length = max_length
        if self.max_length is None:
            self.max_length = max([len(x) for x in self.data]) + 2  # bos, eos tokens
        self.transform = Padding(self.max_length, self.vocab_size)

    def __len__(self):
        return len(self.data)

    def img_to_set(self, img):
        fg_mask = img >= self.threshold
        fg_idx = fg_mask.nonzero(as_tuple=False)
        fg_idx = fg_idx[:, 0] * 28 + fg_idx[:, 1]
        return fg_idx

    def render(self, layout):
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        x_coords = layout % 28
        y_coords = layout // 28
        # valid_idx = torch.where((y_coords < 28) & (y_coords >= 0))[0]
        img = np.zeros((28, 28, 3)).astype(np.uint8)
        img[y_coords, x_coords] = 255
        return Image.fromarray(img, 'RGB')

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        layout = self.transform(self.data[idx])
        return layout['x'], layout['y']


class JSONLayout(Dataset):
    def __init__(self, json_path, max_length=None, precision=8):
        with open(json_path, "r") as f:
            data = json.loads(f.read())

        images, annotations, categories = data['images'], data['annotations'], data['categories']
        self.size = pow(2, precision)

        self.categories = {c["id"]: c for c in categories}
        self.colors = gen_colors(len(self.categories))

        self.json_category_id_to_contiguous_id = {
            v: i + self.size for i, v in enumerate([c["id"] for c in self.categories.values()])
        }

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.vocab_size = self.size + len(self.categories) + 3  # bos, eos, pad tokens # 256+80+3 = 339
        self.bos_token = self.vocab_size - 3      #336
        self.eos_token = self.vocab_size - 2      #337  
        self.pad_token = self.vocab_size - 1      # 338 

        image_to_annotations = {}
        for annotation in annotations:
            image_id = annotation["image_id"]

            if not (image_id in image_to_annotations):
                image_to_annotations[image_id] = []

            image_to_annotations[image_id].append(annotation)

        self.data = []
        for image in images:
            image_id = image["id"]
            height, width = float(image["height"]), float(image["width"])

            if image_id not in image_to_annotations:
                continue

            ann_box = []
            ann_cat = []
            for ann in image_to_annotations[image_id]:
                x, y, w, h = ann["bbox"]
                ann_box.append([x, y, w, h])
                ann_cat.append(self.json_category_id_to_contiguous_id[ann["category_id"]])

            # Sort boxes
            ann_box = np.array(ann_box)
            ind = np.lexsort((ann_box[:, 0], ann_box[:, 1]))
            ann_box = ann_box[ind]

            # Discretize boxes
            ann_box = self.quantize_box(ann_box, width, height)

            # Append the categories
            ann_cat = np.array(ann_cat)
            layout = np.concatenate([ann_cat.reshape(-1, 1), ann_box], axis=1)

            # Flatten and add to the dataset
            self.data.append(layout.reshape(-1))

        self.max_length = max_length
        if self.max_length is None:
            self.max_length = max([len(x) for x in self.data]) + 2  # bos, eos tokens
        self.transform = Padding(self.max_length, self.vocab_size)

    def quantize_box(self, boxes, width, height):

        # range of xy is [0, large_side-1]
        # range of wh is [1, large_side]
        # bring xywh to [0, 1]
        boxes[:, [2, 3]] = boxes[:, [2, 3]] - 1
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / (width - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / (height - 1)
        boxes = np.clip(boxes, 0, 1)

        # next take xywh to [0, size-1]
        boxes = (boxes * (self.size - 1)).round()

        return boxes.astype(np.int32)

    def __len__(self):
        return len(self.data)

    def render(self, layout):
        img = Image.new('RGB', (256, 256), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        layout = layout.reshape(-1)
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        layout = layout[: len(layout) // 5 * 5].reshape(-1, 5)
        box = layout[:, 1:].astype(np.float32)
        box[:, [0, 1]] = box[:, [0, 1]] / (self.size - 1) * 255
        box[:, [2, 3]] = box[:, [2, 3]] / self.size * 256
        box[:, [2, 3]] = box[:, [0, 1]] + box[:, [2, 3]]

        for i in range(len(layout)):
            x1, y1, x2, y2 = box[i]
            cat = layout[i][0]
            col = self.colors[cat-self.size] if 0 <= cat-self.size < len(self.colors) else [0, 0, 0]
            draw.rectangle([x1, y1, x2, y2],
                           outline=tuple(col) + (200,),
                           fill=tuple(col) + (64,),
                           width=2)

        # Add border around image
        img = ImageOps.expand(img, border=2)
        return img

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        layout = torch.tensor(self.data[idx], dtype=torch.long)
        layout = self.transform(layout)
        return layout['x'], layout['y']



class RICOLayout(Dataset):
    def __init__(self, rico_info_fn, split='train', max_length=None, precision=8, pad='Padding',
                inference=False):
        with open(rico_info_fn, "rb") as f:
            rico = pickle.load(f)
        self.info = rico['info']
        self.size = pow(2, precision)
        nclass = len(rico['classname2id'])
        self.rico_classid2name = {v:k for k,v in rico['classname2id'].items()}

        self.rico_category_id_to_contiguous_id = {
                v:i+self.size for i,v in enumerate(rico['classname2id'].values()) }
        self.contiguous_category_id_to_rico_id = {v:k for k,v in self.rico_category_id_to_contiguous_id.items()}
        
        self.colors = gen_colors(nclass)
        self.vocab_size = self.size + nclass + 3  # bos, eos, pad tokens # 256+80+3 = 339
        self.bos_token = self.vocab_size - 3      #336
        self.eos_token = self.vocab_size - 2      #337  
        self.pad_token = self.vocab_size - 1      # 338 
        self.pad = pad
        self.inference = inference
        if split=='train':
            ids = rico['train_ids']
        else:
            ids = rico['gallery_uis']

        self.data = []
        self.uxids = []

        for id in ids:
            self.uxids.append(id)
            width, height = self.info[id]['img_size']
            ann_box = self.info[id]['xywh']
            ann_cat = [self.rico_category_id_to_contiguous_id[cid]  for cid in self.info[id]['class_id'] ]

            # Sort boxes
            ann_box = np.array(ann_box).astype(float)
            ind = np.lexsort((ann_box[:, 0], ann_box[:, 1]))
            ann_box = ann_box[ind]

            # Discretize boxes
            ann_box = self.quantize_box(ann_box, width, height)

            # Append the categories
            ann_cat = np.array(ann_cat)
            layout = np.concatenate([ann_cat.reshape(-1, 1), ann_box], axis=1)

            # Flatten and add to the dataset
            self.data.append(layout.reshape(-1))
        
        self.max_length = max_length
        
        if self.max_length is None:
            self.max_length = max([len(x) for x in self.data]) + 2  # bos, eos tokens
        
        if self.pad == 'Padding':
            self.transform = Padding(self.max_length, self.vocab_size)
        else:
            self.transform = Padding2(self.max_length, self.vocab_size)

        
            

    def quantize_box(self, boxes, width, height):

        # range of xy is [0, large_side-1]
        # range of wh is [1, large_side]
        # bring xywh to [0, 1]
        boxes[:, [2, 3]] = boxes[:, [2, 3]] - 1
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / (width - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / (height - 1)
        boxes = np.clip(boxes, 0, 1)

        # next take xywh to [0, size-1]
        boxes = (boxes * (self.size - 1)).round()

        return boxes.astype(np.int32)

    def __len__(self):
        return len(self.data)
    
    def get_label_name(self,index):
        if 0 <= index-self.size < len(self.colors):
            label = self.rico_classid2name[self.contiguous_category_id_to_rico_id[index]]
        else:
            label = 'background'
        return label

    def render(self, layout, return_bbox=False):
        img = Image.new('RGB', (256, 256), color=(255, 255, 255))
        draw = ImageDraw.Draw(img, 'RGBA')
        layout = layout.reshape(-1)
        layout = trim_tokens(layout, self.bos_token, self.eos_token, self.pad_token)
        layout = layout[: len(layout) // 5 * 5].reshape(-1, 5)
        box = layout[:, 1:].astype(np.float32)
        box[:, [0, 1]] = box[:, [0, 1]] / (self.size - 1) * 255
        box[:, [2, 3]] = box[:, [2, 3]] / self.size * 256
        box[:, [2, 3]] = box[:, [0, 1]] + box[:, [2, 3]]

        for i in range(len(layout)):
            x1, y1, x2, y2 = box[i]
            cat = layout[i][0]
            col = self.colors[cat-self.size] if 0 <= cat-self.size < len(self.colors) else [0, 0, 0]
            draw.rectangle([x1, y1, x2, y2],
                           outline=tuple(col) + (200,),
                           fill=tuple(col) + (64,),
                           width=2)

        # Add border around image
        img = ImageOps.expand(img, border=2)
        
        if return_bbox:
            cat = layout[:,0]
            cat_names = [self.get_label_name(x) for x in cat]
            return img, box, cat_names
        return img
    
    

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) tokens from the data
        layout = torch.tensor(self.data[idx], dtype=torch.long)
        layout = self.transform(layout)
        #print(self.inference)
        if self.inference:
            uxid = self.uxids[idx]
            return layout['x'], layout['y'], uxid
        else:
            return layout['x'], layout['y']


class RICOLayout_withImage(RICOLayout):
    def __init__(self, rico_info_fn, split='train', max_length=None, precision=8):
        
        self.Channel_img_dir = '/mnt/amber/scratch/Dipu/RICO/25ChannelImages/'
        
        with open(rico_info_fn, "rb") as f:
            rico = pickle.load(f)
        self.info = rico['info']
        self.size = pow(2, precision)
        nclass = len(rico['classname2id'])

        self.rico_category_id_to_contiguous_id = {
                v:i+self.size for i,v in enumerate(rico['classname2id'].values()) }
        self.contiguous_category_id_to_rico_id = {v:k for k,v in self.rico_category_id_to_contiguous_id.items()}
        
        self.colors = gen_colors(nclass)
        self.vocab_size = self.size + nclass + 3  # bos, eos, pad tokens # 256+80+3 = 339
        self.bos_token = self.vocab_size - 3      #336
        self.eos_token = self.vocab_size - 2      #337  
        self.pad_token = self.vocab_size - 1      # 338 

        if split=='train':
            ids = rico['train_ids']
        elif split == 'gallery':
            ids = rico['gallery_uis']
        elif split == 'query':
            ids = rico['query_uis']
            
        self.data = []
        self.img_ids = [] 

        for id in ids:
            self.img_ids.append(id)
            width, height = self.info[id]['img_size']
            ann_box = self.info[id]['xywh']
            ann_cat = [self.rico_category_id_to_contiguous_id[cid]  for cid in self.info[id]['class_id'] ]

            # Sort boxes
            ann_box = np.array(ann_box).astype(float)
            ind = np.lexsort((ann_box[:, 0], ann_box[:, 1]))
            ann_box = ann_box[ind]

            # Discretize boxes
            ann_box = self.quantize_box(ann_box, width, height)

            # Append the categories
            ann_cat = np.array(ann_cat)
            layout = np.concatenate([ann_cat.reshape(-1, 1), ann_box], axis=1)

            # Flatten and add to the dataset
            self.data.append(layout.reshape(-1))
        
        self.max_length = max_length
        if self.max_length is None:
            self.max_length = max([len(x) for x in self.data]) + 2  # bos, eos tokens
        self.transform = Padding(self.max_length, self.vocab_size)
            
        
    def __getitem__(self, idx):
    # grab a chunk of (block_size + 1) tokens from the data
        layout = torch.tensor(self.data[idx], dtype=torch.long)
        layout = self.transform(layout)
        img_id = self.img_ids[idx]

        channel25_path = os.path.join(self.Channel_img_dir, img_id + '.npy' )
        img = np.load(channel25_path)    
        img = torch.tensor(img.astype(np.float32))

        return layout['x'], layout['y'], img, img_id


class RICO_Seq_Box(RICOLayout):
    """
        This dataset loads both RICO layout as
            1. sequences for Transformer model and 
            2. stacked box for MLPAE model
    
    """

    def __init__(self, rico_info_fn, ann_file, max_length=None, precision=8):
        with open(ann_file, 'rb') as f:
            data_box = pickle.load(f)
        
        with open(rico_info_fn, "rb") as f:
            rico = pickle.load(f)

        self.data_box = data_box
        self.ids = list(data_box.keys())        
        self.info = rico['info']
        self.size = pow(2, precision)
        nclass = len(rico['classname2id'])
        self.rico_classid2name = {v:k for k,v in rico['classname2id'].items()}
        self.rico_category_id_to_contiguous_id = {
                v:i+self.size for i,v in enumerate(rico['classname2id'].values()) }
        self.contiguous_category_id_to_rico_id = {v:k for k,v in self.rico_category_id_to_contiguous_id.items()}
        
        self.colors = gen_colors(nclass)
        self.vocab_size = self.size + nclass + 3  # bos, eos, pad tokens # 256+80+3 = 339
        self.bos_token = self.vocab_size - 3      #336
        self.eos_token = self.vocab_size - 2      #337  
        self.pad_token = self.vocab_size - 1      # 338 

        self.data = []
        self.uxids = [] 
        

        
        
        for id in self.ids:
            self.uxids.append(id)
            width, height = self.info[str(id)]['img_size']
            ann_box = self.info[str(id)]['xywh']
            ann_cat = [self.rico_category_id_to_contiguous_id[cid]  for cid in self.info[str(id)]['class_id'] ]

            # Sort boxes
            ann_box = np.array(ann_box).astype(float)
            ind = np.lexsort((ann_box[:, 0], ann_box[:, 1]))
            ann_box = ann_box[ind]

            # Discretize boxes
            ann_box = self.quantize_box(ann_box, width, height)

            # Append the categories
            ann_cat = np.array(ann_cat)
            layout = np.concatenate([ann_cat.reshape(-1, 1), ann_box], axis=1)

            # Flatten and add to the dataset
            self.data.append(layout.reshape(-1))
        
        self.max_length = max_length
        if self.max_length is None:
            self.max_length = max([len(x) for x in self.data]) + 2  # bos, eos tokens
        self.transform = Padding(self.max_length, self.vocab_size)
            
        
    def __getitem__(self, idx):
          
         # Data for Seq2Seq model
        layout = torch.tensor(self.data[idx], dtype=torch.long)
        layout = self.transform(layout)
        uxid = self.uxids[idx]
       
        # Data for MLP-AE
        boxes = self.data_box[uxid] ['boxes_cs']    
        box_exists = self.data_box[uxid]['box_exists']
        n_boxes = self.data_box[uxid]['n_boxes'] 
        label_ids = torch.LongTensor(self.data_box[uxid]['label_ids'])

        return layout['x'], layout['y'], uxid, boxes, box_exists, n_boxes, label_ids



class RICO_Seq_Box_RandRef(RICOLayout):
    """
        Same as RICO_Seq_Box but sample a random UX for search model.
    
    """

    def __init__(self, rico_info_fn, ann_file, max_length=None, precision=8):
        with open(ann_file, 'rb') as f:
            data_box = pickle.load(f)
        
        with open(rico_info_fn, "rb") as f:
            rico = pickle.load(f)

        self.data_box = data_box
        self.ids = list(data_box.keys())        
        self.info = rico['info']
        self.size = pow(2, precision)
        nclass = len(rico['classname2id'])
        self.rico_classid2name = {v:k for k,v in rico['classname2id'].items()}
        self.rico_category_id_to_contiguous_id = {
                v:i+self.size for i,v in enumerate(rico['classname2id'].values()) }
        self.contiguous_category_id_to_rico_id = {v:k for k,v in self.rico_category_id_to_contiguous_id.items()}
        
        self.colors = gen_colors(nclass)
        self.vocab_size = self.size + nclass + 3  # bos, eos, pad tokens # 256+80+3 = 339
        self.bos_token = self.vocab_size - 3      #336
        self.eos_token = self.vocab_size - 2      #337  
        self.pad_token = self.vocab_size - 1      # 338 

        self.data = []
        self.uxids = [] 

        for id in self.ids:
            self.uxids.append(id)
            width, height = self.info[str(id)]['img_size']
            ann_box = self.info[str(id)]['xywh']
            ann_cat = [self.rico_category_id_to_contiguous_id[cid]  for cid in self.info[str(id)]['class_id'] ]

            # Sort boxes
            ann_box = np.array(ann_box).astype(float)
            ind = np.lexsort((ann_box[:, 0], ann_box[:, 1]))
            ann_box = ann_box[ind]

            # Discretize boxes
            ann_box = self.quantize_box(ann_box, width, height)

            # Append the categories
            ann_cat = np.array(ann_cat)
            layout = np.concatenate([ann_cat.reshape(-1, 1), ann_box], axis=1)

            # Flatten and add to the dataset
            self.data.append(layout.reshape(-1))
        
        self.max_length = max_length
        if self.max_length is None:
            self.max_length = max([len(x) for x in self.data]) + 2  # bos, eos tokens
        self.transform = Padding(self.max_length, self.vocab_size)
            
        
    def __getitem__(self, idx):
          
         # Data for Seq2Seq model
        layout = torch.tensor(self.data[idx], dtype=torch.long)
        layout = self.transform(layout)
        uxid = self.uxids[idx]
       
        ref_idx = random.choice(list(range(len(self.data))))
        ref_uxid = self.uxids[ref_idx]
        ref_layout = torch.tensor(self.data[ref_idx], dtype=torch.long)
        ref_layout = self.transform(ref_layout)


        # Data for MLP-AE
        boxes = self.data_box[ref_uxid] ['boxes_cs']    
        box_exists = self.data_box[ref_uxid]['box_exists']
        n_boxes = self.data_box[ref_uxid]['n_boxes'] 
        label_ids = torch.LongTensor(self.data_box[ref_uxid]['label_ids'])

        return layout['x'], layout['y'], uxid, ref_uxid, ref_layout['x'], boxes, box_exists, n_boxes, label_ids



class RICO_Seq_Box_SimilarRef(RICOLayout):
    """
        Same as RICO_Seq_Box but sample a random UX for search model.
    
    """

    def __init__(self, rico_info_fn, ann_file, max_length=None, precision=8, ref_index=None):
        with open(ann_file, 'rb') as f:
            data_box = pickle.load(f)
        
        with open('data/mlpae_results.pkl', 'rb') as f:
            self.retrieval = pickle.load(f)

        with open(rico_info_fn, "rb") as f:
            rico = pickle.load(f)

        self.data_box = data_box
        self.ids = list(data_box.keys())        
        self.info = rico['info']
        self.size = pow(2, precision)
        nclass = len(rico['classname2id'])
        self.rico_classid2name = {v:k for k,v in rico['classname2id'].items()}
        self.rico_category_id_to_contiguous_id = {
                v:i+self.size for i,v in enumerate(rico['classname2id'].values()) }
        self.contiguous_category_id_to_rico_id = {v:k for k,v in self.rico_category_id_to_contiguous_id.items()}
        
        self.colors = gen_colors(nclass)
        self.vocab_size = self.size + nclass + 3  # bos, eos, pad tokens # 256+80+3 = 339
        self.bos_token = self.vocab_size - 3      #336
        self.eos_token = self.vocab_size - 2      #337  
        self.pad_token = self.vocab_size - 1      # 338 

        self.data = []
        self.uxids = [] 
        self.ref_index = ref_index
        
        #for id in self.ids:
        for id in self.retrieval.keys():
            self.uxids.append(id)
            width, height = self.info[str(id)]['img_size']
            ann_box = self.info[str(id)]['xywh']
            ann_cat = [self.rico_category_id_to_contiguous_id[cid]  for cid in self.info[str(id)]['class_id'] ]

            # Sort boxes
            ann_box = np.array(ann_box).astype(float)
            ind = np.lexsort((ann_box[:, 0], ann_box[:, 1]))
            ann_box = ann_box[ind]

            # Discretize boxes
            ann_box = self.quantize_box(ann_box, width, height)

            # Append the categories
            ann_cat = np.array(ann_cat)
            layout = np.concatenate([ann_cat.reshape(-1, 1), ann_box], axis=1)

            # Flatten and add to the dataset
            self.data.append(layout.reshape(-1))
        
        self.max_length = max_length
        if self.max_length is None:
            self.max_length = max([len(x) for x in self.data]) + 2  # bos, eos tokens
        self.transform = Padding(self.max_length, self.vocab_size)

    def __len__(self):
        return len( self.retrieval.keys())        
        
    def __getitem__(self, idx):
          
         # Data for Seq2Seq model
        layout = torch.tensor(self.data[idx], dtype=torch.long)
        layout = self.transform(layout)
        uxid = self.uxids[idx]

        
        ref_uxid_pool = self.retrieval[uxid]
        if self.ref_index is not None:
            ref_uxid = ref_uxid_pool[self.ref_index]
        else:
            ref_uxid = random.choice(ref_uxid_pool)
        ref_idx = self.uxids.index(ref_uxid)

        # print(type(ref_uxid))
        # print(ref_idx)
        # print(type(uxid))
        

        ref_layout = torch.tensor(self.data[ref_idx], dtype=torch.long)
        ref_layout = self.transform(ref_layout)


        # Data for MLP-AE
        boxes = self.data_box[int(ref_uxid)] ['boxes_cs']    
        box_exists = self.data_box[int(ref_uxid)]['box_exists']
        n_boxes = self.data_box[int(ref_uxid)]['n_boxes'] 
        label_ids = torch.LongTensor(self.data_box[int(ref_uxid)]['label_ids'])

        return layout['x'], layout['y'], uxid, ref_uxid, ref_layout['x'], boxes, box_exists, n_boxes, label_ids