import pickle
from tqdm import tqdm
from PIL import Image

# Separate out indexes for the train and test 
info = pickle.load(open('/home/dipu/codes/GraphEncoding-RICO/data/rico_box_info_list.pkl', 'rb'))
info_dict = pickle.load(open('/home/dipu/codes/GraphEncoding-RICO/data/rico_box_info.pkl', 'rb'))

UI_data = pickle.load(open("/mnt/amber/scratch/Dipu/RICO/UI_data.p", "rb"))
UI_test_data = pickle.load(open("/mnt/amber/scratch/Dipu/RICO/UI_test_data.p", "rb"))
uis_ncomponent_g100 = pickle.load(open('/home/dipu/codes/GraphEncoding-RICO/data/ncomponents_g100_imglist.pkl', 'rb'))

train_uis = UI_data['train_uis']
query_uis = UI_test_data['query_uis']
gallery_uis = UI_test_data['gallery_uis']

# Remove '.png' extension for ease
train_uis = [x.replace('.png', '') for x in train_uis]
query_uis = [x.replace('.png', '') for x in query_uis]
gallery_uis = [x.replace('.png', '') for x in gallery_uis]

# Donot use the images with large number of components. 
train_uis = list(set(train_uis) & set([x['id'] for x in info]))  #some img (e.g. img with no comp are removed in info)
train_uis = list(set(train_uis) - set(uis_ncomponent_g100))
gallery_uis = list(set(gallery_uis) & set([x['id'] for x in info]))  #some img (e.g. img with no comp are removed in info)
gallery_uis = list(set(gallery_uis) - set(uis_ncomponent_g100))

com2index = {
            'Toolbar':          1,
            'Image':            2,
            'Icon':             3,
            'Web View':         4,
            'Text Button':      5,
            'Text':             6,
            'Multi-Tab':        7,
            'Card':             8,
            'List Item':        9,
            'Advertisement':    10,
            'Background Image': 11,
            'Drawer':           12,
            'Input':            13,
            'Bottom Navigation':14,
            'Modal':            15,
            'Button Bar':       16,
            'Pager Indicator':  17,
            'On/Off Switch':    18,
            'Checkbox':         19,
            'Map View':         20,
            'Radio Button':     21,
            'Slider':           22,
            'Number Stepper':   23,
            'Video':            24,
            'Date Picker':      25,
            }

base_sui_path = '/mnt/scratch/Dipu/RICO/semantic_annotations/'

for id in tqdm(info_dict.keys()):
    sui = base_sui_path + id + '.png'
    s_img = Image.open(sui).convert('RGB')
    size = s_img.size
    info_dict[id]['img_size'] = size

rico = {'info': info_dict,
        'classname2id': com2index,
        'train_ids': train_uis,
        'gallery_uis': gallery_uis,
        'query_uis': query_uis}

with open ('../data/rico.pkl', 'wb') as f:
    pickle.dump(rico,f)
    
