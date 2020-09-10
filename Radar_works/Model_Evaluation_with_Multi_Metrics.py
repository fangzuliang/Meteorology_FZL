# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:11:52 2020
'''
输入测试数据的Dataloader + 加载了参数的模型 ：输出在某些特定评价指标下的模型得分
'''
@author: fzl
"""

import os
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import torch

import Radar_utils

class Model_Evaluate():
    
    '''
    func: 在指定dataloader_test上对模型适用某种评价指标进行评估.
    Parameters
    ----------
    model: instance of class
        加载好的模型
    dataloader_test: instance of class
        加载dataloader,其batch_size 必须为1
    device: torch.device
        cuda 或者 cpu
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
    '''
    
    def __init__(self,model, dataloader_test, 
                 device, metric='TS',
                 scale =80,threshold=10):
        
        self.device = device
        
        self.model = model
        self.dataloader_test = dataloader_test
        
        self.metric = metric
        self.scale = scale
        self.threshold = threshold
        

    def plot_compare_obs_pre(self,obs, pre,scale = 255):
        '''
        func: 对比Dataloader中输出的 obs 和 模型对应预测的pre，图形展示
        parameters
        ################
        obs: Tensor , 4D or 5D
            真实值 
        pre: Tensor
            预测值
        scale: int
            default = 255
            由于模型的输出为[0,1]之间,为方便比对需要 *scale, 数值扩充到[0,scale]范围
        '''
        obs = obs.cpu().detach().numpy() * scale
        pre = pre.cpu().detach().numpy() * scale
        
        assert len(obs.shape) in [4,5]
    
        if len(obs.shape) == 5:
            batch,seq, channel, height,width = obs.shape
            index = np.random.randint(0,batch)
            for k in range(seq):
                obs_img = obs[index,k,0,:,:]
                pre_img = pre[index,k,0,:,:]
                Radar_utils.compare_obs_pre_img(obs_img, pre_img, index = k)
                plt.show()
                
        if len(obs.shape) == 4:
            batch,channel,height,width = obs.shape
            index = np.random.randint(0,batch)
            for k in range(channel):
                obs_img = obs[index,k,:,:]
                pre_img = pre[index,k,:,:]
                Radar_utils.compare_obs_pre_img(obs_img, pre_img, index = k)
                plt.show()
            
        return None
    
    def plot_scores_curve(self,scores):
        '''
        func: 画出scores变化
        Parameter
        ---------
        scores: list or array 
            if array: 一维数组
        '''
        f1 = plt.figure(figsize = (10,7))
        plt.plot(scores)
        
        plt.xlabel('Time', fontsize = 20)
        plt.ylabel('Score', fontsize = 20)
        
        xticks = np.arange(len(scores))
        xtickslabel = np.arange(len(scores))
        plt.xticks(xticks,xtickslabel,fontsize = 20)
        plt.yticks(fontsize = 20)
        
        if self.metric not in ['Multi_HSS','MSE','MAE']:
            plt.title(self.metric + ':' + str(self.threshold),fontsize = 20)
        else:
            plt.title(self.metric,fontsize = 20)
            
        if self.metric not in ['MSE','MAE']:
            plt.ylim((0,1))
        
        plt.show()
    
    def predict(self, test_x):
        '''
        func: 只迭代预测一次，不做参数更新
        parameters
        ---------------
        test_x: np.array
            输入的测试的样本
        ---------------
        return
            tensor. 模型对test_x 的预测结果
        '''
        self.model.eval()
        test_x = test_x.float().to(self.device)
        target = None
        with torch.no_grad():
            target = self.model(test_x)
        return target
    
    
    def one_batch_metric(self, obs,pre):
                        
        '''
        func: 对一个batch内obs和pre进行评价，默认输入的batch维为1
        Parameter
        ---------
        obs: tensor
            真实值
        pre: tensor
            预测值
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
        if self.metric == 'Multi_HSS':
            assert self.scale == 80
            for i in range(seq):
                obs_img = obs[0,i,:,:]*self.scale
                pre_img = pre[0,i,:,:]*self.scale
                hss_score = Radar_utils.HSS(obs_img,pre_img)           
                scores.append(hss_score)
                
        if self.metric in ['MSE','MAE']:
            assert self.scale == 1
            for i in range(seq):
                obs_img = obs[0,i,:,:]*self.scale
                pre_img = pre[0,i,:,:]*self.scale    
                score = Radar_utils.MSE(obs_img,pre_img) if self.metric == 'MSE' else Radar_utils.MAE(obs_img,pre_img)
                scores.append(score)
            
        if self.metric in ['TS','MAR','FAR','POD','BIAS','F1','precision','Single_HSS']:
            assert self.scale == 80
            for i in range(seq):
                obs_img = obs[0,i,:,:]*self.scale
                pre_img = pre[0,i,:,:]*self.scale                
                # hits, misses, falsealarms, correctnegatives = Radar_utils.prep_clf(obs_img,pre_img,
                #                                                                     threshold = self.threshold)
                if self.metric == 'TS':
                    score = Radar_utils.TS(obs_img,pre_img,threshold = self.threshold)
                elif self.metric == 'MAR':
                    score = Radar_utils.MAR(obs_img,pre_img,threshold = self.threshold)
                elif self.metric == 'FAR':
                    score = Radar_utils.FAR(obs_img,pre_img,threshold = self.threshold)
                elif self.metric == 'POD':
                    score = Radar_utils.POD(obs_img,pre_img,threshold = self.threshold)
                elif self.metric == 'BIAS':
                    score = Radar_utils.BIAS(obs_img,pre_img,threshold = self.threshold)
                elif self.metric == 'F1':
                    score = Radar_utils.FSC(obs_img,pre_img,threshold = self.threshold)
                elif self.metric == 'precision':
                    score = Radar_utils.precision(obs_img,pre_img,threshold = self.threshold)
                elif self.metric == 'Single_HSS':
                    score = Radar_utils.HSS_one_threshold(obs_img,pre_img,threshold = self.threshold)
                else: 
                    print('No such metric!')
                scores.append(score)
               
        return scores
    
    
    def plot_which_index(self,which_batch_plot=[0,1,2], show_metric=False):
        '''
        func:
        Parameter
        ---------
        which_batch_plot: list
            default [0,1,2],Dataloader中的所有iterter中，哪些iter的输入和预测是需要被可视化对比的
        show_metric: bool
            是否在输出对比图的同时，也输出metric，即对应的评价信息
            
        '''
        max_batch_index = max(which_batch_plot)
        
        for i, (input_x,input_y) in enumerate(self.dataloader_test):
            input_y = input_y.to(self.device)
            
            if i in which_batch_plot:
                pre_y = self.predict(input_x)
                self.plot_compare_obs_pre(input_y,pre_y,scale = 80)
                if show_metric:
                    score = self.one_batch_metric(input_y,pre_y)
                    print('index:{}-- metric:{}, threshold:{} ----- score:{}'.format(i,self.metric, self.threshold,score))
                print()
                
            if i>= max_batch_index:
                break
                

    def dataloader_metric(self):
        '''
        func: 对整个Dataloader进行综合评分
        '''
        all_scores = []
        assert self.dataloader_test.batch_size == 1
        all_scores = []
        for i, (input_x,input_y) in enumerate(self.dataloader_test):
            input_y = input_y.to(self.device)
            pre_y = self.predict(input_x)
            scores = self.one_batch_metric(input_y, pre_y)
            
            all_scores.append(scores)   
                
        return all_scores
   
    def score_mean_with_frame(self, plot = True):
        '''
        func: 得到预测的n_frames帧的逐帧的评分
        Parameter:
        ----------
        plot: boole
            是否画出得分情况，默认画出
        '''
        scores = self.dataloader_metric()
        
        pd_scores = pd.DataFrame(scores).dropna() #去掉存在nan的行
        
        #获取全为0的行所在的index，去掉这这些行
        zero_index = pd_scores[(pd_scores == 0).all(axis=1)].index
        pd_scores = pd_scores.drop(index = list(zero_index))
        
        score_mean = pd_scores.mean(axis = 0)
        
        if plot:
            self.plot_scores_curve(score_mean)
        
        return scores,score_mean


    def save_all_metric(self,save_path, 
                        all_metrics = ['TS','MAR','FAR','POD',
                                       'BIAS','F1','precision',
                                       'Single_HSS','MSE','MAE',
                                       'Multi_HSS']):
        '''
        func: 将模型在指定评价指标上的评分和图保存到save_path + '/metric'路径下
            文件名用str(metric)表示
        Parameter
        ---------
        save_path: str
            评估结果保存路径
        all_metrics: list
            default ['TS','MAR','FAR','POD','BIAS','F1','precision', 'Single_HSS','MSE','MAE', 'Multi_HSS']
            即使用哪些评价指标进行评估                          
        '''
        metric_save_path = save_path + '/metric'
        if not os.path.exists(metric_save_path):
            os.makedirs(metric_save_path) 
        
        metrics_1 = ['TS','MAR','FAR','POD','BIAS','F1','precision','Single_HSS']
        metrics_2 = ['MSE','MAE','Multi_HSS']
        
        for metric in all_metrics[0:]:
            if metric in metrics_1:
                self.metric = metric
                print(self.metric)
                f2 = plt.figure(figsize=(10,6))
                thresholds = [0.1,1,5,10,20,30,40,50]
                all_scores = []
                for threshold in thresholds[0:]:
                    self.threshold = threshold
                    _, mean_scores = self.score_mean_with_frame(plot=False)
                    all_scores.append(mean_scores)
                    plt.plot(mean_scores,'-*',label = str(threshold))
                    
                plt.xlabel('Time', fontsize = 20)
                plt.ylabel('Score', fontsize = 20)
                xticks = np.arange(len(mean_scores))
                xtickslabel = np.arange(len(mean_scores))
                plt.xticks(xticks,xtickslabel,fontsize = 20)
                plt.yticks(fontsize = 20)
                if self.metric != 'BIAS':
                    plt.ylim((0,1))
                plt.legend()
            
                plt.title(self.metric,fontsize = 16)
                plt.savefig(os.path.join(metric_save_path,metric + '.png'),dpi = 200)
                plt.show()
                
                pd_scores = pd.DataFrame(all_scores,index = thresholds)
                pd_scores.to_csv(os.path.join(metric_save_path,metric + '.csv'))
                    
                f2.clf()
                
            elif metric in metrics_2:
                self.metric = metric
                print(self.metric)
                if metric in ['MSE','MAE']:
                    self.scale = 1
                else:
                    self.scale = 80
                _,mean_scores = self.score_mean_with_frame(plot = False)
                
                f2 = plt.figure(figsize=(10,6))
                plt.plot(mean_scores,'-*',label = self.metric)
                plt.xlabel('Time', fontsize = 20)
                plt.ylabel('Score', fontsize = 20)
                xticks = np.arange(len(mean_scores))
                xtickslabel = np.arange(len(mean_scores))
                plt.xticks(xticks,xtickslabel,fontsize = 20)
                plt.yticks(fontsize = 20)
                plt.legend()
                
                if self.metric == 'Multi_HSS':
                    plt.ylim((0,1))
                
                plt.title(self.metric,fontsize = 16)
                plt.savefig(os.path.join(metric_save_path,metric + '.png'),dpi = 200)
                plt.show()
                
                pd_scores = pd.DataFrame(mean_scores)
                pd_scores.to_csv(os.path.join(metric_save_path, self.metric + '.csv'))   
            
                f2.clf()
        
        return None
                

        
 
        
        
    
        
        







    
    
