import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randrange
import ast

def  show_photo(list_photo,list_lable,colm,row):
    for num in range(0,colm*row):
        plt.subplot(row,colm,num+1)
        plt.title(list_lable[num])
        plt.imshow(list_photo[num],cmap='gray')
    plt.show()


df = pd.read_csv('trans_csv_data/train.csv')
list_img=[]
list_lable = []
for num in range(0,20):
    irand = randrange(0, len(df["image_data"]))
    img_data = np.array(ast.literal_eval(df['image_data'][irand]))
    img = img_data.reshape(28,28)
    list_img.append(img)
    list_lable.append(df['lable'][irand])
image = list_img[0].astype(np.uint8)
show_photo(list_img,list_lable,4,5)
