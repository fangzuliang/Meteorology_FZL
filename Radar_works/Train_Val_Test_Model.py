# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:02:15 2020
func: 编写属于自己的模型训练、验证、预测范式
@author: fzl
"""

#%%
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging

# import BLoss
# from BLoss import BMAELoss, BMSELoss, BMSAELoss, BMSAELoss
import Radar_utils
from Model_Evaluation_with_Multi_Metrics import Model_Evaluate


#%%
class Model():
    def __init__(self, 
                 device,
                 model, 
                 save_path,
                 opt_configs,
                 lr_scheduler_configs,
                 loss_fn = nn.MSELoss(),
                 logger = None, 
                 num_epochs = 200,
                 patience = 5,
                 display_interval = 25,
                 mask = False,
                 aim_size = (256,256),
                 gradient_clipping = False,
                 clipping_threshold = 3, 
                 ):
        
        '''
        func:
        parameters:
        ###########################
        device: torch.device
            CPU或者CUDA
        model: 
            实例化后的模型
        save_path: str
            模型的训练过程的一些输出结果的保存路径
            eg: /home/fzl/data/code/fzl_convttlstm/20200721_17
        opt_configs: dict
            优化器名称及其参数字典
            eg: opt_configs = {'opt': 'Adam', 'lr':0.001, 'weight_decay': 0} 
        lr_scheduler_configs: dict
            学习率调整策略名称及其参数字典
            eg: lr_scheduler_configs = {'lr_scheduler': 'ReduceLROnPlateau',
                                         'mode': 'min',  'factor': 0.5, 'patience': 3,
                                         'verbose': True, 'min_lr': 1e-07}
        logger: logging
            使用logging记录模型训练信息
            default None.则表示没有传入已经实例化的logger,需要在class内重新定义logger
        loss_fn: instance of loss function
            default: nn.MSELoss(), 传入已经实例化后的损失函数
        num_epochs: int
            default 200,训练的最大轮数
        patience: int
            default 5. 
            验证集上的损失有超过patience个epoch不下降时，则中止训练，并保存验证集上损失最低时的模型
        display_interval: int
            default 25. 
            在每个epoch的训练、验证和测试过程中，每间隔display_interval个 iteration保存和打印一次损失信息
        mask: bool
            default False. 
            是否在二维图片中，只选择部分区域(mask用1表示)参与损失计算。
            考虑到雷达图输入为正方形，而雷达观测为圆盘形状，因此4个角一般是不参与损失计算
        aim_size: tuple or list
            default (256,256). 模型输入和输出的图片宽和高
        gradient_clipping: bool
            default False. 在训练过程中是否使用梯度裁剪
        clipping_threshold: 3.
            default 3. 梯度裁剪阈值
            只有当gradient_clipping为True时，才起作用
        '''
        
        if not logger:
            import arrow
            time_str = arrow.now().format('YYYYMMDD_HHmmss')
            log_path = os.path.join(save_path,time_str)
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            
            log_file = os.path.join(log_path,'train_log.txt')
            
            logging.basicConfig(level=logging.INFO,
                                filename = log_file, 
                                filemode='w', 
                                format = '%(asctime)s - %(message)s',
                               )
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            
            self.logger.info('--------------------------opt_info----------------------------')
            self.logger.info(opt_configs)
            self.logger.info('----------------------lr_scheduler_info----------------------')
            self.logger.info(lr_scheduler_configs)
            

        self.logger = logger
        
        self.device = device
        self.save_path = save_path
        
        self.network = model
        self.network.to(self.device)
        self.opt_configs = opt_configs
        self.lr_scheduler_configs = lr_scheduler_configs
        self.loss_fn = loss_fn
        
        self.num_epochs = num_epochs
        self.patience = patience
        self.display_interval = display_interval
        self.mask = mask
        self.aim_size = aim_size
        self.gradient_clipping = gradient_clipping
        self.clipping_threshold = clipping_threshold
        
        #如果不存在则创建保存路径
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)  
        
        if self.opt_configs['opt'] == 'Adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(), 
                                              lr=opt_configs['lr'],
                                              weight_decay=opt_configs['weight_decay'])
            print(opt_configs)
        
        if self.opt_configs['opt'] == 'AdamW':
            self.optimizer = torch.optim.Adam(self.network.parameters(), 
                                  lr=opt_configs['lr'],
                                  weight_decay=opt_configs['weight_decay'])

    
            print(opt_configs)

        if lr_scheduler_configs['lr_scheduler'] == 'ReduceLROnPlateau':
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer,
                                                  mode = self.lr_scheduler_configs['mode'],
                                                  factor = self.lr_scheduler_configs['factor'],
                                                  patience = self.lr_scheduler_configs['patience'],
                                                  verbose = self.lr_scheduler_configs['verbose'], 
                                                  min_lr= self.lr_scheduler_configs['min_lr'],
                                                  )
            
            print(lr_scheduler_configs)

        
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 4, gamma = 0.5) #每4个epoch学习率降低50%
        
        #判断所使用的学习率调整方案 里面是否存在 get_lr 方法
        # ReduceLROnPlateau不存在，其他的几种方案存在
        #如果lr_flag 为True,则每个epoch结束后，输出学习率信息,训练结束后，保存lr随epoch的变化
        self.lr_flag = 'get_lr' in self.lr_scheduler.__dir__()
        
        #设置掩码区域
        if self.mask:
            mask_info = 'Just circle pixel are involved in the loss operation'
            self.logger.info(mask_info)
            mask = Radar_utils.get_circle_mask(self.aim_size[0])
            self.mask = torch.tensor(mask)
        else:
            mask_info = 'All pixel are involved in the loss operation '
            self.mask = torch.tensor(1)
            
        self.mask = self.mask.to(self.device)
        
        print(mask_info)
        self.logger.info(mask_info)
        print()
        
    
    def plot_scores_fig(self, scores, x_label = None,y_label = None ,title = None, save_file = None):
        '''
        func: 画出分数变化并保存
        parameters:
        ###################
        scores: list or np.array
            损失list或者一维数组
        x_label: str
            x轴的label
        y_label: str
            y轴的label
        title: str
            图的title
        save_file: str
            default None. 默认不保存
            该图保存名称  
        '''
        scores = list(scores)
        
        f1 = plt.figure(figsize = (10,6))
        plt.plot(scores)
        
        if x_label:
            plt.xlabel(x_label,fontsize= 20)
        if y_label:
            plt.ylabel(y_label, fontsize = 20)
        plt.xticks(fontsize = 16)
        plt.xticks(fontsize = 16)
        
        if title:
            plt.title(title, fontsize = 20)
        
        if save_file:
            save_filepath = os.path.join(self.save_path,save_file)
            plt.savefig(save_filepath,dpi = 200, bbox_inches = 'tight')
            
        plt.show()
        f1.clf()
        
        return None
    
    
    def plot_compare_obs_pre(self,obs, pre,epoch = 1 ,scale = 255):
        '''
        func: 对比Dataloader中输出的 obs 和 模型对应预测的pre， 图形展示
        parameters
        ################
        obs: Tensor , 4D or 5D
            真实值 
        pre: Tensor
            预测值
        scale: int
            由于模型的输出为[0,1]之间,为方便比对需要 *scale, 数值扩充到[0,scale]范围
        '''
        
        obs = obs.cpu().detach().numpy() * scale
        pre = pre.cpu().detach().numpy() * scale
        
        assert len(obs.shape) in [4,5]
        
        fig_save_path = os.path.join(self.save_path, str(epoch))
        
        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)
        
        if len(obs.shape) == 5:
            batch,seq, channel, height,width = obs.shape
            index = np.random.randint(0,batch)
            for k in range(seq):
                obs_img = obs[index,k,0,:,:]
                pre_img = pre[index,k,0,:,:]
                
                Radar_utils.compare_obs_pre_img(obs_img, pre_img, index = k, 
                                                is_save_file = True,
                                                save_path = os.path.join(fig_save_path,str(k) + '.png'))
                
                plt.show()
                
        if len(obs.shape) == 4:
            batch,channel,height,width = obs.shape
            index = np.random.randint(0,batch)
            for k in range(channel):
                obs_img = obs[index,k,:,:]
                pre_img = pre[index,k,:,:]
                Radar_utils.compare_obs_pre_img(obs_img, pre_img, index = k,
                                                is_save_file = True,
                                                save_path = os.path.join(fig_save_path,str(k) + '.png')
                                                )
                
                plt.show()
            
        return None
    
    def initial(self):
        '''
        func:对网络的卷积层和线形层的参数进行初始化，初始化方式使用kaiming_normal_
        '''
        for m in self.network.modules():
            if isinstance(m, nn.Conv2d):
                #权重初始化
                # nn.init.normal(m.weight.data)
                #  nn.init.xavier_normal(m.weight.data)
                # nn.init.constant_(m.weight.data, val = 1)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                # m.bias.data.fill_(1)
            elif isinstance(m, nn.Linear):
                # nn.init.constant_(m.weight.data, val= 1)
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')

    def train_once(self, input_x, input_y,j):
        '''
        func: 只迭代训练一次，并损失后传，参数更新
        parameters
        --------------
        input_x: np.array
        input_y: np.array
            Dataloader中迭代出来的一组 input_x, input_y
        --------------
        return:
            损失值
        '''
        
#         print('input_x size:',input_x.size())
#         print('input_y size:',input_y.size())
        self.network.train()
        input_x = input_x.float().to(self.device)
        input_y = input_y.float().to(self.device)
        self.optimizer.zero_grad()
   
        output_seq = self.network(input_x)
        if j % self.display_interval == 0:
        
            print()
            print('output_seq.max:', output_seq.max())
            print('output_seq.min:', output_seq.min())
            print('output_seq.mean:', output_seq.mean())
            print('target_seq.max:', input_y.max())
            print('target_seq.min:', input_y.min())
            print('target_seq.mean:', input_y.mean())
            print()
        
        loss = self.loss_fn(output_seq * self.mask, input_y * self.mask)
        loss.backward()
        if self.gradient_clipping:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.clipping_threshold)
        self.optimizer.step()
        return loss.item()
    
    
    def train(self, dataloader_train, dataloader_eval):
        '''
        func: 完整的训练，每个epoch训练结束后，就做一次dataloader_eval上做一次验证,并记录和保存每个epoch内的损失的变化
        parameters
        ----------
        dataloader_train: Dataloader实例
            训练集的Dataloader
        dataloader_eval: Dataloader实例
            验证集的Dataloader
        return
        '''
        
        #多少次iteration
        train_batch_number = dataloader_train.__len__() 
        
        count = 0
        best = 1000000000
        
        all_train_epoch_loss = []
        all_test_epoch_loss = []
        all_epoch_lr = []
        
        #参数初始化
        self.initial()
        
        for i in range(self.num_epochs):
            
            one_epoch_all_loss = []
            
            self.logger.info('\nepoch: {0}'.format(i+1))
            for j, (input_x, input_y) in enumerate(dataloader_train):
                loss_train = self.train_once(input_x, input_y,j)
                one_epoch_all_loss.append(loss_train)
                if j % self.display_interval == 0:
                    
                    train_info = 'Train: epoch {}: {}/{} -----------------loss:{}'.format(i + 1,j,train_batch_number,loss_train)
                    self.logger.info(train_info)
                    print(train_info)
            
            ######画出某个epoch内的损失变化图并保存
            title = 'Train_' + 'Epoch_' + str(i + 1)
            save_file = title + '.png'  
            self.plot_scores_fig(one_epoch_all_loss,x_label='iterations', y_label = 'Train_loss' , title = title,save_file = save_file)
            
            #得到每个epoch的Train的平均损失，并做记录
            avg_train_loss = torch.mean(torch.tensor(one_epoch_all_loss))
            all_train_epoch_loss.append(avg_train_loss)
            train_epoch_info = 'Train: Epoch_{} avg loss: {}'.format(i + 1,avg_train_loss)
            self.logger.info(train_epoch_info)
            print()
            print(train_epoch_info)
            print()
            
            #得到每个epoch Test的损失,并做记录
            loss_eval = self.test(dataloader_eval,index = i + 1,plot_img = True)
            all_test_epoch_loss.append(loss_eval)
            test_epoch_info = 'Test: Epoch_{} avg loss: {}'.format(i,loss_eval) 
            self.logger.info(test_epoch_info)
            print()
            print(test_epoch_info)
            print()
            
            #如果可以获取每个epoch的lr信息, 则对应输出和记录每个epoch的lr信息
            if self.lr_flag:           
                lr = self.lr_scheduler.get_lr()           
                all_epoch_lr.append(lr)
                lr_info = 'Epoch_{} lr:{}'.format(i+1,lr)
                logging.info(lr_info)
                print(lr_info)
                self.lr_scheduler.step() 
            else:
                self.lr_scheduler.step(loss_eval)
            
            if loss_eval >= best:
                count += 1
                self.logger.info('eval loss is not improved for {0} epoch'.format(count))
            else:
                count = 0
                self.logger.info('eval loss is improved from {:.5f} to {:.5f}, saving model'.format(best, loss_eval))
                self.save_model()
                best = loss_eval

            if count == self.patience: 
                self.logger.info('early stopping reached, best loss is {:5f}'.format(best))
                print('early stopping reached, best loss is {:5f}'.format(best))
                break
                
        #保存所有损失随着epoch变化的图
        train_title = 'Train_Loss-Epoch'
        test_title = 'Test_Loss-Epoch'

        self.plot_scores_fig(all_train_epoch_loss,x_label='Epoch', y_label='Train loss', title = train_title,save_file = train_title + '.png')
        self.plot_scores_fig(all_test_epoch_loss, x_label='Epoch', y_label='Test loss', title = test_title,save_file = test_title + '.png')
    
        if self.lr_flag:
            lr_title = 'lr-Epoch'
            self.plot_scores_fig(all_epoch_lr,x_label='Epoch',y_label='Learning rate',  title = lr_title, save_file = lr_title + '.png')

        return self.network, self.logger
    
    def test(self, dataloader_test,index = 1, plot_img = False):
        '''
        func:
        parameters
        -----------
        dataloader_test: Dadaloader
            用来测试的数据集
        index: int
            default 1, 表示第几个epoch(从1开始计数)
        plot_img: bool
            default False
        '''
        test_batch_number = dataloader_test.__len__()
        
        self.network.eval()
        one_epoch_all_loss = []
        with torch.no_grad():
            for j, (test_x, test_y) in enumerate(dataloader_test):
                pre_y = self.predict(test_x)
                test_y = test_y.to(self.device)
                loss_test = self.loss_fn(pre_y * self.mask, test_y * self.mask)
                one_epoch_all_loss.append(loss_test.item())
                
                if index:
                    if j % self.display_interval == 0:
                                            
                        test_info = 'Test: epoch {}: {}/{} -----------------loss:{}'.format(index,j,test_batch_number,loss_test)
                        self.logger.info(test_info)
                        print(test_info)    
        if index:           
            title = 'Test_' + 'Epoch_' + str(index)
            save_file = title + '.png'  
            self.plot_scores_fig(one_epoch_all_loss,x_label='iteration',y_label='Test_loss', title = title,save_file = save_file)
            
        if plot_img:
            self.plot_compare_obs_pre(test_y,pre_y,scale = 255,epoch = index)
            
        test_loss_avg = torch.mean(torch.tensor(one_epoch_all_loss))
        
        return test_loss_avg
    

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
        self.network.eval()
        test_x = test_x.float().to(self.device)
        pre_y = None
        with torch.no_grad():
            pre_y = self.network(test_x)
                        
        return pre_y
        

    def save_model(self):
        '''
        func: 保存模型，同时保存模型主体参数+优化器, 保存路径为 self.save_path，格式为.chk
        '''
        torch.save({'net': self.network.state_dict(), 
                    'optimizer':self.optimizer.state_dict()}, 
                       os.path.join(self.save_path, 'checkpoint.chk'))
    
    def load_model(self, chk_path):
        '''
        func: 加载模型    
        Parameters
        ----------
        chk_path : str
            模型保存路径
        '''
        checkpoint = torch.load(chk_path)
        self.network.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        return self.network
    
    def metric_model(self, dataloader_test, 
                     metrics = ['TS','MAR','FAR','POD',
                                'BIAS','F1','precision',
                                'Single_HSS','MSE','MAE',
                                'Multi_HSS']):
        '''
        func: 将模型在指定评价指标上的评分和图保存到save_path + '/metric'路径下
                文件名用str(metric)表示
        Parameters
        ----------
        dataloader_test: Dadaloader
            用来测试的数据集
        metrics: 评估指标
            default ['TS','MAR','FAR','POD','BIAS','F1','precision', 
                     'Single_HSS','MSE','MAE', 'Multi_HSS']
            即使用哪些评价指标进行评估.          
        '''
        
        model_evaluate = Model_Evaluate(self.network, 
                                        dataloader_test = dataloader_test,
                                        device = self.device)
        
        model_evaluate.save_all_metric(save_path = self.save_path,
                                       all_metrics = metrics)
        
        return None
                                        
        
        
        
        
    
 


    
    