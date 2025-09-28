import pandas as pd
from scipy import ndimage
from os.path import join
import numpy as np
import ast
import os

feature_name = ['zeros.csv','one.csv','tow.csv','three.csv','four.csv','five.csv','six.csv','seven.csv','eight.csv','nine.csv']
path_file = 'Feature_number'
image = pd.read_csv('trans_csv_data/test.csv')


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
def diff_bit(list_1,list_2):
    s = 0
    for i in range(0,len(list_1)):
        if (list_1[i]<(list_2[i]+2)) and (list_1[i]>(list_2[i]-2)):
            s = s+1
    return s/len(list_1)

list_temp = []
list_feat = []
image_0 = np.array(ast.literal_eval(image['image_data'][8]),np.uint8)
print(image['lable'][4])
input()
featur_0 = Binery_Image(image_0)
featur_1 = block_cont(image_0)
featur_2 = row_col_mean(image_0)
featur_3 = hist_attr(image_0)
featur_4 = memnt_img(image_0)
list_temp.append(featur_0)
list_temp.extend(featur_1)
list_temp.extend(featur_2)
list_temp.extend(featur_3)
list_temp.extend(featur_4)
list_feat.extend(list_temp)
list_prop = []
for i in range(0,10):
    path_name = join(path_file,feature_name[i])
    df=pd.read_csv(path_name)
    s = 0
    for j in range(0,df.shape[0]):
        print(f"total off {i}: ", j/df.shape[0]*100 ,'%')
        list_num = list(df.loc[j])
        p = diff_bit(list_feat,list_num)
        s = s+p/(df.shape[0])
        os.system('cls')
    list_prop.append(s*100)
prior_num=0
number = 0
j = 0
for i in list_prop:
    if i > prior_num:
        prior_num = i
        number = j
    j = j+1
print(number)
    

