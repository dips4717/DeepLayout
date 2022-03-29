

from trainer_ss import getBoundingBoxes_from_info
import pickle
import numpy as np
from eval_metrics.get_overall_Classwise_IOU import get_overall_Classwise_IOU
from eval_metrics.get_overall_pix_acc import get_overall_pix_acc
from scipy.spatial.distance import cdist


feat_dict = pickle.load(open('runs/RICO_Image/results/feat_dict.pkl', 'rb'))
q_feat = feat_dict['q_feat']
g_feat = feat_dict['g_feat']
q_fnames = feat_dict['q_fnames']
g_fnames = feat_dict['g_fnames']

boundingBoxes = getBoundingBoxes_from_info()

distances = cdist(q_feat, g_feat, metric= 'euclidean')
sort_inds = np.argsort(distances)

overallMeanClassIou, overallMeanWeightedClassIou, classwiseClassIoU = get_overall_Classwise_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10])
overallMeanAvgPixAcc, overallMeanWeightedPixAcc, classPixAcc = get_overall_pix_acc(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10])     
    
# print('\n\nep:%s'%(epoch))
print('overallMeanClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanClassIou]) + '\n')        
print('overallMeanAvgPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanAvgPixAcc]) + '\n')

    
# with open(save_file, 'a') as f:
#     f.write('\n\ep: {}\n'.format(epoch))
#     f.write('overallMeanClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanClassIou]) + '\n')
#     f.write('overallMeanAvgPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanAvgPixAcc]) + '\n')

from matplotlib import pyplot as plt
from PIL import Image
import os 

def plot_retrieved_images_and_uis(sort_inds, q_fnames, g_fnames, avgIouArray=None, avgPixAccArray=None):
    
    base_im_path = '/mnt/scratch/Dipu/RICO/combined/'
    base_ui_path = '/mnt/scratch/Dipu/RICO/semantic_annotations/'
    
    for i in range((sort_inds.shape[0])): #range(1): 
#    for i in  range(2):    
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
        
        
        
plot_retrieved_images_and_uis(sort_inds, q_fnames, g_fnames)