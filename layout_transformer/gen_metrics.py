import numpy as np
import random

result = np.random.rand(1000,20,5)

def overlapping_loss(result):
    losses=np.zeros(len(result))
    idx=0
    for i in result:
        over=0
        for j in range(len(i)):
            A=float(i[j][3]*i[j][4])
            if A==0:
                continue
            for k in range(len(i)):
                if j==k:
                    continue
                x1=i[j][1]
                x2=i[j][1]+i[j][3]
                y1=i[j][2]
                y2=i[j][2]+i[j][4]
                x3=i[k][1]
                x4=i[k][1]+i[k][3]
                y3=i[k][2]
                y4=i[k][2]+i[k][4]
                x_over=max(min(x2,x4)-max(x1,x3),0)
                y_over=max(min(y2,y4)-max(y1,y3),0)
                over+=x_over*y_over/A
        losses[idx]=over
        idx+=1
    return np.mean(losses)*100



def alignment_loss(result):
    xl =result[...,1]           
    yl = result[...,2]
    
    xr = xl+result[...,3]
    yr = yl + result[...,4]

    xc = (xl + xr)/2
    yc = (yl + yr)/2

    ele = [xl , yl , xc, yc, xr, yr]
    ele1 = []
    epsilon = 0
    for element in ele:
        min_xl = np.ones(shape = element.shape)
        for i in range(len(element)):
            for j in range(len(element[i])):
                for k in range(len(element[i])): 
                    if j != k :
                        min_xl[i][j] = min(min_xl[i][j],abs(element[i][j]-element[i][k]))        
        min_xl = -np.log(1.0-min_xl + epsilon)
        ele1.append(min_xl)
    ele1 = np.min(np.array(ele1), axis = 0)
    ele1 = np.mean(np.sum(ele1 , axis  = 1))
    return ele1*100



# def calculate_iou(result):
#     losses=np.zeros(len(result))
#     idx=0
#     for i in result:
#         iou=0
#         for j in range(len(i)):
#             for k in range(j+1,len(i)):
#                 x1=i[j][1]
#                 x2=i[j][1]+i[j][3]
#                 y1=i[j][2]
#                 y2=i[j][2]+i[j][4]
#                 x3=i[k][1]
#                 x4=i[k][1]+i[k][3]
#                 y3=i[k][2]
#                 y4=i[k][2]+i[k][4]

#                 box_1 = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
#                 box_2 = [[x3, y3], [x4, y3], [x4, y4], [x3, y4]]

#                 poly_1 = Polygon(box_1)
#                 poly_2 = Polygon(box_2)

#                 if poly_1.union(poly_2).area!=0:
#                     iou += poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
#         losses[idx]=iou
#         idx+=1
#     return np.mean(losses)*100#

output = alignment_loss(result)