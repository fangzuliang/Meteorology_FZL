# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:21:59 2020
Dataloader
@author: fzl
"""
import os
import torch.utils.data as data
import numpy as np
import torch
import pandas as pd
import cv2 as cv
from torchvision import transforms
import matplotlib.pyplot as plt

# os.chdir('/home/fzl/data/code/unet_test/')
# device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu:0')

# DEVICE = torch.device('cuda:0,1,2,3' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cuda:0,1,2,3' if torch.cuda.device_count() > 1 else 'cpu')


#%%
class Radar_Dataset_Train_Valid(data.Dataset):
    
    '''
    Dataloader of Radar Dataset for train or val
    Parameters
    ----------
    csv_path: str
        Radar Dataset info composed by csv, 
        including time, index, pathfile of station radar observation.
    flag: str
        'train' or 'valid', 
        Indicates whether the data state of the output is 'train' or 'validation'
    train_test_split_size: float 
        default 0.3, which means 0.3 proportion of all radar dataset was spilted to test
    x_length : int
        default 2, The number of radar images entered by the model,
        represented by the number of channels in a two-dimensional image   
    y_length : int
        default 4, The number of radar images that the model outputs,
        represented by the number of channels in a two-dimensional image
    stride : int
        The sliding step size in the radar sequence when constructing the sequence sample
        generally >= x_length
    aim_size: tutle
        aim size of X, Y, default = (256, 256)
        Must be an integral multiple of 256
    min_threshold: int
        default 50000;
        To determine whether there is precipitation process in the sample sequence. 
        If the accumulative value of all radar echoes in the sample sequence is less than 
        the threshold value, the precipitation process is not found in the sample sequence,
        and the sample is discarded by default
    index: None or int or list
        default None.
        when None: 默认输出的Y的帧数为y_length，即全部输出
        when int: 默认输出y_lenghth帧中的第index帧，index从0开始计数，< y_length
        when list: 由整数组成的list, eg: [0,1,4], 则表示输出Y的第[0,1,4]帧, max(index) < y_length
    Returns
    -------
    out 
    '''
    def __init__(self,
                 csv_path, 
                 flag = 'train',
                 train_test_split_size = 0.3,
                 x_length = 2,
                 y_length = 4,
                 stride = 2,
                 aim_size = (256, 256), 
                 min_threshold = 50000,
                 transform = None, 
                 index = None
                 ):
        
        self.csv_path = csv_path
        self.train_test_split_size = train_test_split_size
        
        self.x_length = x_length
        self.y_length = y_length
        self.stride = stride
        self.index = index
        
        self.aim_size = aim_size
        self.transform = transform
        
        self.min_threshold = min_threshold
        
        self.all_sample_filelist = self.get_sample(self)
        length = len(self.all_sample_filelist)
        
        assert flag in ['train','valid']
        
        if flag == 'train':
            self.flag_sample_filelist = self.all_sample_filelist[0:int(length*(1 - self.train_test_split_size))]
            
        if flag == 'valid':
            self.flag_sample_filelist = self.all_sample_filelist[int(length*(1 - self.train_test_split_size)):]
    
    def __getitem__(self, index):
        
        sample_filelist = self.flag_sample_filelist[index]
        img_x_y = []
        # image_x, image_y = [], []
        
        for file in sample_filelist:
            
            img = cv.imread(file,0)
            img = cv.resize(img,self.aim_size, interpolation = cv.INTER_CUBIC)
            # img = cv.resize(img,self.aim_size, interpolation = cv.INTER_AREA)
            
            #size = (256,256)
            img = np.array(img)
            img = torch.tensor(img.astype(np.float32))
                                          
            if self.transform:
                img = self.transform(img)
            
            img_x_y.append(img)
            
        #沿着新维度进行叠加, size = (self.sample_lenght,256, 256)
        img_x_y = torch.stack(img_x_y, dim=0) 
            
        img_x = img_x_y[0:self.x_length,:,:] / 255
        img_y = img_x_y[self.x_length:,:,:] / 255
        
        #如果self.index = None, 则默认输出所有y_lenght
        #如果self.index为int, 则表示img_y中的第几帧
        #如果self.index为list，则表示img_y中的list中指定的帧数
        if self.index != None:
            if isinstance(self.index,int):
                assert self.index < self.y_length
                img_y = img_y[self.index:self.index + 1,:]
            elif isinstance(self.index, list):
                assert max(self.index) < self.y_length
                img_y = img_y[self.index,:]
                
        return img_x, img_y
            
            
    def __len__(self):
        return len(self.flag_sample_filelist)
        
        
    @staticmethod
    def get_sample(self):
        '''
        构建样本序列列表，每个子列表为一个样本序列
        '''
        pd_file = pd.read_csv(self.csv_path)
        days_list = pd_file['day']
        index_list = pd_file['index']
        radar_file_list = pd_file['radar_file']
        
        #样本序列长度
        sample_length = self.x_length + self.y_length
        
        #每个样本序列为一个子list,所有子list构建成all_sample_filelist
        all_sample_filelist = []
        
        i = 0
        # index = index_list[i]
        while i < len(radar_file_list) - sample_length + 1:
            
            start_day = days_list[i]
            start_index = index_list[i] #样本序列第一帧对应的index，eg: 1
            
            
            end_day = days_list[i + sample_length - 1]
            end_index = index_list[i + sample_length - 1]
            
            index_gap = (end_day - start_day)*240 + end_index - start_index
            
            #如果这几帧是连续的，则头尾两帧对应的索引差 应该与 (样本序列长度 - 1) 一致 
            if index_gap == sample_length - 1:
                sample_filelist = []
                
                all_img = 0
                for j in range(sample_length):
                    radar_file = radar_file_list[i + j]
                    img = cv.imread(radar_file,0)
                    sample_filelist.append(radar_file)

                    all_img += img #把所有像素加在一起
                
                if np.sum(all_img) >= self.min_threshold:
                    all_sample_filelist.append(sample_filelist)
                
                #向前滑动 stride帧 后开始构建样本序列
                i = i + self.stride
            
            #如果不连续，则过渡到下一帧, 重新构建样本序列
            else:
                i = i + 1
              
        return all_sample_filelist

#%%
#Dataloader test

# csv_path = '/home/fzl/data/code/unet_test/train_data04_unet_file_index.csv'
# # # csv_path = 'F:/caiyun/unet_test/train_data04_unet_file_index.csv'
# # # csv_path = 'F:/caiyun/unet_test/train_fzl_unet_file_index.csv'

# radar_dataset = Radar_Dataset_Train_Valid(csv_path = csv_path, 
#                                     flag = 'train',
#                                     train_test_split_size = 0.2,
#                                     x_length = 3,
#                                     y_length = 5,
#                                     stride = 3,
#                                     )                
                              
# # # # # all_sample_filelist = radar_dataset.all_sample_filelist 
               
# dataloader = data.DataLoader(radar_dataset, batch_size = 1, shuffle = False, num_workers=0)

# print(dataloader.__len__())

#%%
# import Radar_utils

# mask = Radar_utils.get_circle_mask(256)

# i = 1
# k = 1000
# for input_x,target in dataloader:
    
#     # print(i)
#     # print(input_x.shape)
#     # print(target.shape)
#     # print()
    
#     i = i + 1
    
#     if i > k:
#         break
                
# for i in range(5):
    
#     plt.imshow(target[0,i,:,:] )
#     plt.show()                
            
            
    
#%%
class Radar_Dataset_Test(data.Dataset):
    
    '''
    Dataloader of Radar Dataset 
    Parameters
    ----------
    csv_path: str
        Radar Dataset info composed by csv, 
        including time, index, pathfile of station radar observation.
    x_length : int
        default 2, The number of radar images entered by the model,
        represented by the number of channels in a two-dimensional image   
    y_length : int
        default 4, The number of radar images that the model outputs,
        represented by the number of channels in a two-dimensional image
    stride : int
        The sliding step size in the radar sequence when constructing the sequence sample
        generally >= x_length
    aim_size: tutle
        aim size of X, Y, default = (256, 256)
        Must be an integral multiple of 256
    min_threshold: int
        default 50000;
        To determine whether there is precipitation process in the sample sequence. 
        If the accumulative value of all radar echoes in the sample sequence is less than 
        the threshold value, the precipitation process is not found in the sample sequence,
        and the sample is discarded by default
    index: None or int or list
        default None.
        when None: 默认输出的Y的帧数为y_length，即全部输出
        when int: 默认输出y_lenghth帧中的第index帧，index从0开始计数，< y_length
        when list: 由整数组成的list, eg: [0,1,4], 则表示输出Y的第[0,1,4]帧, max(index) < y_length
    Returns
    -------
    out
        
    '''

    def __init__(self,
                 csv_path, 
                 x_length = 2,
                 y_length = 4,
                 stride = 2,
                 aim_size = (256, 256), 
                 min_threshold = 50000,
                 transform = None, 
                 index = None,
                 ):
        
        self.csv_path = csv_path
        
        self.x_length = x_length
        self.y_length = y_length
        self.stride = stride
        self.index = index
        
        self.aim_size = aim_size
        self.transform = transform
        
        self.min_threshold = min_threshold
        
        self.all_sample_filelist = self.get_sample(self)
        
        
    def __getitem__(self, index):
        
        sample_filelist = self.all_sample_filelist[index]
        img_x_y = []
        # image_x, image_y = [], []
        
        for file in sample_filelist:
            
            img = cv.imread(file,0)
            img = cv.resize(img,self.aim_size, interpolation = cv.INTER_CUBIC)
            # img = cv.resize(img,self.aim_size, interpolation = cv.INTER_AREA)
            
            #size = (256,256)
            img = np.array(img)
            img = torch.tensor(img.astype(np.float32))
                                          
            if self.transform:
                img = self.transform(img)
            
            img_x_y.append(img)
            
        #沿着新维度进行叠加, size = (self.sample_lenght,256, 256)
        img_x_y = torch.stack(img_x_y, dim=0) 
            
        img_x = img_x_y[0:self.x_length,:,:] / 255
        img_y = img_x_y[self.x_length:,:,:] / 255
        
        #如果self.index = None, 则默认输出所有y_lenght
        #如果self.index为int, 则表示img_y中的第几帧
        #如果self.index为list，则表示img_y中的list中指定的帧数
        if self.index != None:
            if isinstance(self.index,int):
                assert self.index < self.y_length
                img_y = img_y[self.index:self.index + 1,:]
            elif isinstance(self.index, list):
                assert max(self.index) < self.y_length
                img_y = img_y[self.index,:]
            
        return img_x, img_y
            
            
    def __len__(self):
        return len(self.all_sample_filelist)
        
    @staticmethod
    def get_sample(self):
        '''
        构建样本序列列表，每个子列表为一个样本序列
        '''
        pd_file = pd.read_csv(self.csv_path)
        days_list = pd_file['day']
        index_list = pd_file['index']
        radar_file_list = pd_file['radar_file']
        
        #样本序列长度
        sample_length = self.x_length + self.y_length
        
        #每个样本序列为一个子list,所有子list构建成all_sample_filelist
        all_sample_filelist = []
        
        i = 0
        # index = index_list[i]
        while i < len(radar_file_list) - sample_length + 1:
            
            start_day = days_list[i]
            start_index = index_list[i] #样本序列第一帧对应的index，eg: 1
            
            
            end_day = days_list[i + sample_length - 1]
            end_index = index_list[i + sample_length - 1]
            
            index_gap = (end_day - start_day)*240 + end_index - start_index
            
            #如果这几帧是连续的，则头尾两帧对应的索引差 应该与 (样本序列长度 - 1) 一致 
            if index_gap == sample_length - 1:
                sample_filelist = []
                
                all_img = 0
                for j in range(sample_length):
                    radar_file = radar_file_list[i + j]
                    img = cv.imread(radar_file,0)
                    sample_filelist.append(radar_file)

                    all_img += img #把所有像素加在一起
                
                if np.sum(all_img) >= self.min_threshold:
                    all_sample_filelist.append(sample_filelist)
                
                #向前滑动 stride帧 后开始构建样本序列
                i = i + self.stride
            
            #如果不连续，则过渡到下一帧, 重新构建样本序列
            else:
                i = i + 1
              
        return all_sample_filelist

      
#%%
#Dataloader test

# csv_path = '/home/fzl/data/code/unet_test/test_fzl_unet_file_index.csv'
# # csv_path = 'F:/caiyun/test_fzl_unet_file_index.csv'

# radar_dataset = Radar_Dataset_Test(csv_path = csv_path, 
#                                     x_length = 2,
#                                     y_length = 4,
#                                     stride = 3,
#                                     )                
                              
# all_sample_filelist = radar_dataset.all_sample_filelist 
               

# dataloader = data.DataLoader(radar_dataset, batch_size=1, shuffle=False, num_workers=0)

# i = 1
# for input,target in dataloader:
#     print(i)
#     print(input.shape)
#     print(target.shape)
#     print()
    
#     i = i + 1
    
#     if i > 3:
#         break
                
# for i in range(4):
#     plt.imshow(target[0,i,:,:])
#     plt.show()                
            
            
            

#%%
# path = 'F:/caiyun/radar_images'
# filelist = os.listdir(path)

# all_image = []
# for file in filelist[1:3]:
#     abs_file = os.path.join(path, file)
    
#     print(abs_file)
#     img = cv.imread(abs_file, 0)
    
#     plt.imshow(img)
#     plt.show()
    
#     # img1 = cv.resize(img,(256,256),interpolation = cv.INTER_AREA)
#     img1 = cv.resize(img,(256,256),interpolation = cv.INTER_CUBIC)
#     # img1 = cv.resize(img,(256,256))
#     plt.imshow(img1)
#     plt.show()
    
#     image = np.array(img1).reshape((1, 256, 256))
#     image = torch.tensor(image.astype(np.float32))
    
#     all_image.append(image)            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
             
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    