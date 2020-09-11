# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 14:51:35 2020
加载12个单帧预测模型，和dataloader_test，进行评分输出
@author: fzl 
"""
###导入相应的包
import os
# os.chdir('/home/fzl/data/code/')

import numpy as np
import matplotlib.pyplot as plt
import Radar_utils

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader_fzl import Radar_Dataset_Train_Valid, Radar_Dataset_Test
from Train_Val_Test_Model import Model
from Model_Evaluation_with_Multi_Metrics import Model_Evaluate

#%%
from leibniz.unet.base import UNet
from leibniz.unet.residual import Basic
from leibniz.nn.activation import CappingRelu

#%%
# save_path = '/home/fzl/data/code/leibniz/model_train_results/All_12_Basic_model'

#%%
class ComposeMoreSingleModel():
    '''
    funcs:
    Parameter
    ---------
    device: torch.device
        cpu or cuda
    model_save_path: str
        多个单帧模型模型保存的路径
    path1: str
        每个单帧预测模型所在的文件夹名称
        eg: model_save_path = '/home/All_12_Basic_model'
            path1 = 'Basic', 第i个模型的完整路径为:'/home/All_12_Basic_model/Basic+str(i)/checkpoint.chk'    
    in_dim: int
        模型的输入帧数, default 10
    out_dim: int
        集合模型的输出帧数, default 12
    single_out_dim: int
        单个模型的输出帧数, default 1
    '''
    def __init__(self,
                 device, 
                 model_save_path, 
                 path1 = 'Basic',
                 in_dim = 10, 
                 out_dim = 12,
                 single_out_dim = 1
                 ):
        
        self.model_save_path = model_save_path
        self.path1 = path1
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.single_out_dim = single_out_dim
        
        self.device = device
    
    
    def forward(self, test_x, test_y, 
                model_list = None,
                is_save_file = False,
                fig_save_path = None,
                ):
        '''
        func:
        Parameter
        --------
        test_x: 4D-Tensor ---> [batch, in_dim,height, width]
            模型的输入数据
        test_y: 4D-Tensor ---> [batch, out_dim, height, width]
            模型的输出数据
        model_list: list or None or single model
            多个单帧预测模型组成的list
            when None: 则由self.load_more_model()去调用多个模型
            when not list: 则表示是一个单独的模型
        is_save_file: bool
            是否保存图片路径，默认False
        fig_save_path: str
            图片保存路径
        '''
        
        if model_list == None:
            model_list = self.load_more_model()
        
        #当传入的model_list不是多个模型，而是一个模型时，则调用单个模型预测函数
        if not isinstance(model_list,list):
            model = model_list
            pre = self.single_model_predict(test_x, model)
            
        else:
            pre = self.more_model_predict(test_x,model_list = model_list)
            
        if is_save_file:
            if not os.path.exists(fig_save_path):
                os.makedirs(fig_save_path)
            
        self.plot_compare_obs_pre(test_y,pre,
                                  is_save_file = is_save_file,
                                  fig_save_path = fig_save_path)
        
        return [test_y,pre]  
    
    def one_batch_metric(self, obs,pre, metric = 'TS',
                         threshold = 30,scale = 80, 
                         if_plot = False,
                         save_path = None):
                        
        '''
        func: 对一个batch内obs和pre进行评价，默认输入的batch维为1
        Parameter
        ---------
        obs: tensor
            真实值
        pre: tensor
            预测值
        metric: str
            评价指标,可选用['Multi_HSS','Single_HSS','MSE','MAE','TS','MAR','FAR','POD','BIAS','F1','precision']
            当metric in ['MSE','MAE']时，scale必须等于1, 其他情况下为80
            其中Multi_HSS:分为多个DBZ阈值综合考虑obs 和 pre之间的相似度;
                Single_HSS:只考虑单个阈值下，obs 和 pre的相似度
        scale: int
            由于模型的输出为[0,1]之间,为方便比对需要 *scale, 数值扩充到[0,scale]范围
        threshold: int
            当metric in ['TS','MAR','FAR','POD','BIAS','F1','precision','Single_HSS'] 时,threshold才其作用
            当metric in ['MSE','MAE']时，threshold不起作用
            数值范围在（0,80）之间，通常取值[0.1,1,5,10,20,30,40,50]
        Returns
        -------
        scores: list
            分数列表
        '''
        
        obs = obs.cpu().detach().numpy() 
        pre = pre.cpu().detach().numpy() 
        
        assert obs.shape == pre.shape
        assert len(obs.shape) in [4,5]
        assert obs.shape[0] == 1
        
        #如果维度为5，则保证channel为1
        if len(obs.shape) == 5:
            assert obs.shape[2] == 1 #channel通道为1
            obs = obs[:,:,0,:,:]
            pre = pre[:,:,0,:,:]
            
        batch, seq, height,width = obs.shape
        scores = [] #得分列表
        if metric == 'Multi_HSS':
            assert scale == 80
            for i in range(seq):
                obs_img = obs[0,i,:,:]* scale
                pre_img = pre[0,i,:,:]* scale
                hss_score = Radar_utils.HSS(obs_img,pre_img)           
                scores.append(hss_score)
                
        if metric in ['MSE','MAE']:
            assert scale == 1
            for i in range(seq):
                obs_img = obs[0,i,:,:]* scale
                pre_img = pre[0,i,:,:]* scale    
                score = Radar_utils.MSE(obs_img,pre_img) if metric == 'MSE' else Radar_utils.MAE(obs_img,pre_img)
                scores.append(score)
            
        if metric in ['TS','MAR','FAR','POD','BIAS','F1','precision','Single_HSS']:
            assert scale == 80
            for i in range(seq):
                obs_img = obs[0,i,:,:]* scale
                pre_img = pre[0,i,:,:]* scale                
                # hits, misses, falsealarms, correctnegatives = Radar_utils.prep_clf(obs_img,pre_img,
                #                                                                     threshold = threshold)
                
                if metric == 'TS':
                    score = Radar_utils.TS(obs_img,pre_img,threshold = threshold)
                elif metric == 'MAR':
                    score = Radar_utils.MAR(obs_img,pre_img,threshold = threshold)
                elif metric == 'FAR':
                    score = Radar_utils.FAR(obs_img,pre_img,threshold = threshold)
                elif metric == 'POD':
                    score = Radar_utils.POD(obs_img,pre_img,threshold = threshold)
                elif metric == 'BIAS':
                    score = Radar_utils.BIAS(obs_img,pre_img,threshold = threshold)
                elif metric == 'F1':
                    score = Radar_utils.FSC(obs_img,pre_img,threshold = threshold)
                elif metric == 'precision':
                    score = Radar_utils.precision(obs_img,pre_img,threshold = threshold)
                elif metric == 'Single_HSS':
                    score = Radar_utils.HSS_one_threshold(obs_img,pre_img,threshold = threshold)
                else: 
                    print('No such metric!')
                scores.append(score)
                
        if if_plot:
            self.plot_scores_curve(scores,metric,threshold,save_path = save_path)
            
        return scores
    
    def plot_scores_curve(self,scores, metric = 'TS', threshold = 30, save_path = None):
        '''
        func: 画出scores变化
        Parameter
        ---------
        scores: list or array 
            if array: 一维数组
        metric: str
            评价的名称，eg: 'TS'
        threshold: int
            default 30
        save_filepath: str or None
            图片的保存路径, eg: 'D:/metrics'
            when None, 则不保存
        '''
        f1 = plt.figure(figsize = (10,7))
        plt.plot(scores)
        
        plt.xlabel('Time', fontsize = 20)
        plt.ylabel('Score', fontsize = 20)
        
        xticks = np.arange(len(scores))
        xtickslabel = np.arange(len(scores))
        plt.xticks(xticks,xtickslabel,fontsize = 20)
        plt.yticks(fontsize = 20)
        
        if metric not in ['Multi_HSS','MSE','MAE']:
            plt.title(metric + '—' + str(threshold),fontsize = 20)
        else:
            plt.title(metric,fontsize = 20)
            
        if metric not in ['MSE','MAE']:
            plt.ylim((0,1))
        
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_filepath = os.path.join(save_path,metric + '_' + str(threshold) + '.jpg')
            plt.savefig(save_filepath,bbox_inches = 'tight')
        
        plt.show()
    
        
    def plot_compare_obs_pre(self,obs, pre,scale = 80, 
                             is_save_file = False,
                             fig_save_path = 'D:/'
                             ):
        '''
        func: 对比 obs 和 模型对应预测的pre，图形展示
        parameters
        ----------
        obs: Tensor , 4D or 5D
            真实值 
        pre: Tensor
            预测值
        scale: int
            default = 255
            由于模型的输出为[0,1]之间,为方便比对需要 *scale, 数值扩充到[0,scale]范围
        is_save_file: bool
            是否保存图片路径，默认False
        fig_save_path: str
            图片保存路径
        '''
        obs = obs.cpu().detach().numpy() * scale
        pre = pre.cpu().detach().numpy() * scale
        
        assert len(obs.shape) == 4

        batch,channel,height,width = obs.shape
        index = np.random.randint(0,batch)
        for k in range(channel):
            obs_img = obs[index,k,:,:]
            pre_img = pre[index,k,:,:]
            Radar_utils.compare_obs_pre_img(obs_img, pre_img, index = k, 
                                            is_save_file = is_save_file,
                                            save_path = os.path.join(fig_save_path,str(k) + '.png'))
            plt.show()
            
        return None
    
    
    def more_model_predict(self,test_x, model_list = None):
        '''
        func: 得到给定模型和输入下的模型输出
        parameters
        ----------
        model_list: list
            由多个实例化后的单帧预测模型组成的list
            当为None时，则调用load_more_model()
        test_x: np.array
            输入的测试的样本
        ---------------
        return
            tensor. 模型对test_x 的预测结果
        '''
        if model_list == None:
            model_list = self.load_more_model()
            
        all_target = []
        for model in model_list:
            target = self.single_model_predict(test_x,model)
            all_target.append(target)
        
        #将多个模型的预测按照channels维度拼接起来
        all_target = torch.cat(all_target,dim = 1)
        
        #如果模型个数和最终输出个数一致，且每个模型输出多帧，则只取单模型预测的第一帧
        if len(model_list) == self.out_dim:
            if self.single_out_dim > 1:
                all_target = all_target[:,0::self.single_out_dim]
        
        return all_target
            
    def single_model_predict(self, test_x, model):
        '''
        func: 得到给定模型和输入下的模型输出
        parameters
        ----------
        test_x: np.array
            输入的测试的样本
        model: instance of model
            实例化后的模型
        ---------------
        return
            tensor. 模型对test_x 的预测结果
        '''
        model.eval()
        test_x = test_x.float().to(self.device)
        target = None
        with torch.no_grad():
            target = model(test_x)
        return target
    

    def load_more_model(self,path1 = None):
        '''
        func: 加载多个单帧预测模型
        Return
        -----
        all_model: list
            self.out_dim个单帧模型(实例化后)组成的list
        '''
        if path1 == None:
            path1 = self.path1
        
        all_model = []
        for i in range(self.out_dim):
            trained_model_filepath = os.path.join(self.model_save_path, path1 + str(i),'checkpoint.chk')
            single_model = self.load_single_model(trained_model_filepath)
            all_model.append(single_model)
            
        return all_model
        
    def load_single_model(self, trained_model_filepath):
        '''
        func: 加载并实例化trained_model_filepath这个路径下的模型
        Parameter
        ---------
        trained_model_filepath: str
            训练好的单帧预测模型的保存路径，
            eg: '/home/All_12_Basic_model/Basic0/checkpoint.chk'
        Returns:
        --------
        network: model
            已经加载并实例化并后的模型
        '''
        from leibniz.unet.base import UNet
        from leibniz.unet.residual import Basic
        from leibniz.nn.activation import CappingRelu
        
        network = UNet(in_channels = self.in_dim,
                       out_channels = self.single_out_dim,
                       normalizor= 'batch',
                       spatial = (256,256), 
                       layers = 4, 
                       ratio = 0,
                       vblks= [2,2,2,2],
                       hblks= [0,0,0,0],
                       scales= [-1,-1,-1,-1],
                       factors= [1,1,1,1],
                        block = Basic, 
                       relu = CappingRelu(),
                       final_normalized= False
                       )
        
        checkpoint = torch.load(trained_model_filepath)
        network.load_state_dict(checkpoint['net'])
                
        return network.to(self.device)
        
        
#%%
# save_path = '/home/fzl/data/code/leibniz/model_train_results/All_12_Basic_model'        
        
# save_path = 'F:/caiyun/code/leibniz/model_train_results/All_12_Basic_model'   

# device = torch.device('cpu:0')       
# composeMore = ComposeMoreSingleModel(device = device, model_save_path=save_path)
# all_model = composeMore.load_more_model()       
        
        
        
        
        
        
        
        
        
        
        
        
    

#%%


