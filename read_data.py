from struct import unpack
import numpy as np
from array import array
import csv
from os.path import join


class read_file_data():
    def __init__(self,path_train_image,path_train_lable,path_test_image,path_test_lable):
        self.path_train_image = path_train_image
        self.path_train_lable = path_train_lable
        self.path_test_image = path_test_image
        self.path_test_lable = path_test_lable
    def read_data(self,path_image,path_label):
        with open(path_label,'rb') as file:
            magic , size = unpack('>II',file.read(8))
            if magic != 2049:
                raise " valu is not avalable"
            lable = array('B',file.read())
        with open(path_image,'rb') as file:
            magic, size , row , colm = unpack('>IIII',file.read(16))
            if magic != 2051:
                raise "is not avalable"
            images = []
            for num in range(size):
                images.append([0]*colm*row)
            for num in range(size):
                img=array('B',file.read(colm*row))
                images[num][:]=img
        return images,lable
    def write_data_csv(self,path):
        train_image , train_lable = self.read_data(self.path_train_image,self.path_train_lable)
        test_image , test_lable = self.read_data(self.path_test_image,self.path_test_lable)
        combin_train = [['image_data','lable']]
        combin_train.extend(list([a,b] for a,b in zip(train_image,train_lable)))
        combin_test = [['image_data','lable']]
        combin_test.extend(list([a,b] for a,b in zip(test_image,test_lable)))
        path_train = join(path,'train.csv')
        path_test = join(path ,'test.csv')
        with open(path_train,'w',newline='') as file:
            writer = csv.writer(file)
            writer.writerows(combin_train)
        with open(path_test,'w',newline='') as file:
            writer = csv.writer(file)
            writer.writerows(combin_test)
input_path = 'data_set'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
image_nist = read_file_data(training_images_filepath,training_labels_filepath,test_images_filepath,test_labels_filepath)
image_nist.write_data_csv('trans_csv_data')


