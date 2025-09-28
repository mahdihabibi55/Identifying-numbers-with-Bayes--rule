import numpy as np
import pandas as pd
from scipy import ndimage
import ast
import os
from os.path import join


feature_name = ['zeros.csv','one.csv','tow.csv','three.csv','four.csv','five.csv','six.csv','seven.csv','eight.csv','nine.csv']
image_num = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
list_feat = [[list(range(0,49))],[list(range(0,49))],[list(range(0,49))],[list(range(0,49))],[list(range(0,49))]
             ,[list(range(0,49))],[list(range(0,49))],[list(range(0,49))],[list(range(0,49))],[list(range(0,49))]]
path_file = 'Feature_number'
image = pd.read_csv('trans_csv_data/train.csv')

def Binery_Image(images):
    for i in range(0,28*28):
        if images[i]>127:
            images[i]=1
        else :
            images[i]=0
    return np.sum(images)
def block_cont(images):
    images = images.reshape(28,28)
    lise_cons = []
    for i in range(0,21,7):
        for j in range(0,21,7):
            lise_cons.append(images[i:i+7,j:j+7].mean())
    return lise_cons
def row_col_mean(images):
    list_mean = list()
    images = images.reshape(28,28)
    for i in range(0,28):
        list_mean.append(((images[i,0:27].sum()+images[0:28,i].sum())/2))
    return list_mean
def hist_attr(images):
    hist = np.histogram(images,bins=8,range=(0,255))
    return hist[0]
def memnt_img(images):
    images = images.reshape(28,28)
    list_memnt = ndimage.center_of_mass(images)
    return list_memnt
for j in range(0,10):
    image_num[j]=image[image['lable'] == j]
    image_num[j].reset_index(drop=True,inplace=True)
for j in range(0,10):
    for i in range(0,len(image_num[j])):
        os.system('cls')
        print(f"number of {j} : " , i/len(image_num[j])*100,'%')
        list_temp = []
        image = np.array(ast.literal_eval(image_num[j]['image_data'][i]),np.uint8)
        featur_0 = Binery_Image(image)
        featur_1 = block_cont(image)
        featur_2 = row_col_mean(image)
        featur_3 = hist_attr(image)
        featur_4 = memnt_img(image)
        list_temp.append(featur_0)
        list_temp.extend(featur_1)
        list_temp.extend(featur_2)
        list_temp.extend(featur_3)
        list_temp.extend(featur_4)
        list_temp.append(j)
        list_feat[j].append(list_temp)
for i in range(0,10):
    data = pd.DataFrame(data=list_feat[i],index=range(0,len(list_feat[i])),columns=range(0,49))
    name_path = join(path_file,feature_name[i])
    data.to_csv(name_path)

    


    
