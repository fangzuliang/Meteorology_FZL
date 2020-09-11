# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 13:55:28 2020

@author: fzl
"""

#%%
###导入相应的包
import os
# os.chdir('/home/fzl/data/code/')

import numpy as np
import matplotlib.pyplot as plt
import logging
import arrow

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader_fzl import Radar_Dataset_Train_Valid, Radar_Dataset_Test
from Train_Val_Test_Model import Model
from Model_Evaluation_with_Multi_Metrics import Model_Evaluate

import BLoss

#%%
#模型训练结果保存路径
save_path = '/home/fzl/data/code/leibniz/model_train_results/All_12_2_Basic_model/'

#%%
######################## 参数配置 ##########

#优化器参数配置
opt_configs = {}
opt_configs['opt'] = 'Adam'
opt_configs['lr'] = 0.001
opt_configs['weight_decay'] = 0.0

##学习率调整策略参数配置
lr_scheduler_configs = {}
lr_scheduler_configs['lr_scheduler'] = 'ReduceLROnPlateau' 
lr_scheduler_configs['mode'] = 'min'
lr_scheduler_configs['factor'] = 0.5
lr_scheduler_configs['patience'] = 3
lr_scheduler_configs['verbose'] = True
lr_scheduler_configs['min_lr'] = 1e-8

#%%
##Dataset_train, Dataset_val, Dataset_test参数配置
##Dataset_train, Dataset_val, Dataset_test参数配置
dataset_configs = {}
dataset_configs['train_valid_csv_path'] = '/home/fzl/data/code/fzl_convttlstm/train_fzl_unet_file_index.csv'
dataset_configs['test_csv_path'] = '/home/fzl/data/code/fzl_convttlstm/test_fzl_unet_file_index.csv'
dataset_configs['train_test_split_size'] = 0.3
dataset_configs['input_frames'] = 10
dataset_configs['output_frames'] = 12
dataset_configs['stride'] = 3
dataset_configs['aim_size'] = (256,256)
dataset_configs['min_threshold'] = 80000
dataset_configs['index'] = None
dataset_configs['train_batch_size'] = 16
dataset_configs['test_batch_size'] = 1 #必须为1
dataset_configs['n_cpu'] = 8
#assert dataset_configs['stride'] >= dataset_configs['input_frames']

#%%
data_train = Radar_Dataset_Train_Valid(csv_path = dataset_configs['train_valid_csv_path'],
                                        flag='train',
                                        train_test_split_size = dataset_configs['train_test_split_size'],
                                        x_length = dataset_configs['input_frames'],
                                        y_length = dataset_configs['output_frames'],
                                        stride = dataset_configs['stride'],
                                        aim_size= dataset_configs['aim_size'],
                                        min_threshold = dataset_configs['min_threshold'],
                                        transform = None,
                                        index = dataset_configs['index'],
                                            )
                                        
data_val = Radar_Dataset_Train_Valid(csv_path = dataset_configs['train_valid_csv_path'],
                                        flag='valid',
                                        train_test_split_size = dataset_configs['train_test_split_size'],
                                        x_length = dataset_configs['input_frames'],
                                        y_length = dataset_configs['output_frames'],
                                        stride = dataset_configs['stride'],
                                        aim_size= dataset_configs['aim_size'],
                                        min_threshold = dataset_configs['min_threshold'],
                                        transform = None,
                                        index = dataset_configs['index'],
                                            )

data_test = Radar_Dataset_Test(csv_path = dataset_configs['test_csv_path'],
                                x_length = dataset_configs['input_frames'],
                                y_length = dataset_configs['output_frames'],
                                stride = dataset_configs['stride'],
                                aim_size= dataset_configs['aim_size'],
                                min_threshold = dataset_configs['min_threshold'],
                                transform = None,
                                index = dataset_configs['index'],
                                            )

dataloader_train = DataLoader(data_train, batch_size = dataset_configs['train_batch_size'], shuffle=True, num_workers=dataset_configs['n_cpu'])
dataloader_eval = DataLoader(data_val, batch_size=dataset_configs['train_batch_size'], shuffle = True, num_workers=dataset_configs['n_cpu'])
dataloader_test = DataLoader(data_test, batch_size=dataset_configs['test_batch_size'], shuffle = False, num_workers=dataset_configs['n_cpu'])

#%%
from leibniz.unet.base import UNet
from leibniz.unet.hyperbolic import HyperBottleneck
from leibniz.unet.residual import Basic
from leibniz.nn.activation import CappingRelu
import leibniz.unet.warp as wp

model_configs = {}
model_configs['in_channels'] = dataset_configs['input_frames']

if dataset_configs['index'] != None:
    if isinstance(dataset_configs['index'], int):
        model_configs['out_channels'] = 1
    elif isinstance(dataset_configs['index'], list):
        model_configs['out_channels'] = len(dataset_configs['index'])
else:        
    model_configs['out_channels'] = dataset_configs['output_frames']

model_configs['normalizor'] = 'batch'
model_configs['spatial'] = dataset_configs['aim_size']
model_configs['layers'] = 4
model_configs['ratio'] = 0
model_configs['vblks'] = [2] * model_configs['layers']
model_configs['hblks'] = [0] * model_configs['layers']
model_configs['scales'] = [-1] * model_configs['layers']
model_configs['factors'] = [1] * model_configs['layers']
# model_configs['block'] = wp.WarpBottleneck
model_configs['block'] = Basic
# model_configs['block'] = HyperBottleneck

model_configs['relu'] = CappingRelu()
model_configs['final_normalized'] = False

network = UNet(**model_configs)
               
#%%
##损失函数及其参数配置
loss_configs = {}
loss_configs['a'] = 6
loss_configs['b'] = 0.8
loss_configs['mse_w'] = 1
loss_configs['mae_w'] = 1

loss_fn = BLoss.BEXPMSAELoss(**loss_configs)

loss_configs['loss_name'] = loss_fn.__str__()[:-2]
#%%
##训练参数配置
train_configs = {}
train_configs['CUDA_VISIBLE_DEVICES'] = "4"
train_configs['device'] = torch.device('cuda')
train_configs['num_epochs'] = 200
train_configs['patience'] = 10
train_configs['display_interval'] = 25
train_configs['mask'] = False
train_configs['gradient_clipping'] = False
train_configs['clipping_threshold'] = 3
train_configs['loss_fn'] = loss_fn

#%%
if __name__ == '__main__':
    
    time_str = arrow.now().format('YYYYMMDD_HHmmss')
    
    if dataset_configs['index'] != None:
        if isinstance(dataset_configs['index'],int):
            save_file = '_' + loss_configs['loss_name'] + '_'+ str(model_configs['block'].__name__) + \
                                str(dataset_configs['index'])
        elif isinstance(dataset_configs['index'],list):
            save_file = '_' + loss_configs['loss_name'] + '_'+ str(model_configs['block'].__name__) + \
                                str(dataset_configs['index'][0]) + '_' + str(len(dataset_configs['index']))
    else:
        save_file = '_' + loss_configs['loss_name'] + '_'+ str(model_configs['block'].__name__)
                                
                                                           
    log_path = os.path.join(save_path,
                            time_str + save_file) 
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    log_file = os.path.join(log_path,'train_log.txt')
    
    logging.basicConfig(level=logging.INFO,
                        filename = log_file, 
                        filemode='w', 
                        format = '%(asctime)s - %(message)s',
    #                   format='%(message)s'
                       )
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info('Start!')
    logger.info('save_path: {}'.format(save_path))
    
    #记录和保存这些信息
    logger.info('------------------ model_configs ---------------------')
    logger.info(model_configs)
    f = open(os.path.join(log_path,'model_configs.txt'),'w')
    f.write(str(model_configs))
    f.close()
    
    logger.info('------------------ opt_configs ---------------------')
    logger.info(opt_configs)
    f = open(os.path.join(log_path,'opt_configs.txt'),'w')
    f.write(str(opt_configs))
    f.close()
    
    logger.info('------------------ lr_scheduler_configs ---------------------')
    logger.info(lr_scheduler_configs)
    f = open(os.path.join(log_path,'lr_scheduler_configs.txt'),'w')
    f.write(str(lr_scheduler_configs))
    f.close()
    
    logger.info('------------------ dataset_configs ---------------------')
    logger.info(dataset_configs)
    f = open(os.path.join(log_path,'dataset_configs.txt'),'w')
    f.write(str(dataset_configs))
    f.close()
    
    logger.info('------------------ loss_configs ---------------------')
    logger.info(loss_configs)
    f = open(os.path.join(log_path,'loss_configs.txt'),'w')
    f.write(str(loss_configs))
    f.close()
    
    logger.info('------------------ train_configs ---------------------')
    logger.info(train_configs)
    f = open(os.path.join(log_path,'train_configs.txt'),'w')
    f.write(str(train_configs))
    f.close()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = train_configs['CUDA_VISIBLE_DEVICES']
    model = Model(device = train_configs['device'],
                  model = network,
                  save_path = log_path,
                  opt_configs = opt_configs,
                  lr_scheduler_configs = lr_scheduler_configs,
                  logger = logger,
                  loss_fn = train_configs['loss_fn'],
                  num_epochs = train_configs['num_epochs'],
                  patience = train_configs['patience'],
                  display_interval = train_configs['display_interval'],
                  mask = train_configs['mask'],
                  aim_size = dataset_configs['aim_size'],
                  gradient_clipping = train_configs['gradient_clipping'],
                  clipping_threshold = train_configs['clipping_threshold']
                )
    
    logger.info('loading train dataloader')
    logger.info('loading eval dataloader')
    logger.info('loading test dataloader')
    
    model.train(dataloader_train, dataloader_eval)
    logger.info('\n######training finished!########\n')
    model.load_model(os.path.join(log_path,'checkpoint.chk'))
    loss_test = model.test(dataloader_test,index = 999, plot_img = True)
    logger.info("test loss: {0}".format(loss_test))
    
    logger.info("-----------------metric model--------------------")
    model.metric_model(dataloader_test, 
                       metrics = ['TS','MAR','FAR','POD',
                                'BIAS','F1','precision',
                                'Single_HSS','MSE','MAE',
                                'Multi_HSS'])
    
    

    
  









