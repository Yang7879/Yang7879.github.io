# -*- coding: utf-8 -*-
#用来读取cifar10的自制data读取器，只做了train。test没有做，也类似。效果可以。注意在train上加入target = target.long()，这是为了自己读取数据集时候用的一句话，否则类型不匹配。
from torch.utils.data import Dataset
import torch
import numpy as np
from PIL import Image

class MyDataset(Dataset):
    def __init__(self,txt_path,transform = None,target_transform = None):
        fh = open(txt_path,'r')
        init_ = True
        for line in fh:
            line = line.strip()
            dict_ = self.unpickle(line)
            label = dict_[b'labels']
            data = dict_[b'data']
            
            data = data.reshape(10000,3,32,32)
            
            label = np.array(label)
            label.reshape(10000,1)
            
            if init_:
                self.data = data
                self.label = label
                init_ = False
            else:
                self.data = np.concatenate((self.data,data))
                self.label = np.concatenate((self.label,label))
            
        self.transform = transform
        self.target_transform = target_transform
            
            
    def __getitem__(self,index):
        data = self.data[index]
        label = self.label[index]
        # data = data.reshape(1,3*32*32)
        data = data.transpose(1,2,0)
        data = Image.fromarray(np.uint8(data))        
        if self.transform is not None:
            data = self.transform(data) 
        return data, label
    
    def __len__(self):
        return np.size(self.data,0)
    
    def unpickle(self,file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

if __name__ == '__main__':
    # dict_ = unpickle('cifar-10-batches-py/data_batch_1')
    # print('都有什么Key',dict_.keys())
    # print('有',len(dict_[b'labels']),'个样本')
    # print('样本的大小',dict_[b'data'].shape)
    
    temp = MyDataset('cifar-10-batches-py/path.txt')