# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:04:06 2020

@author: fangzuliang
"""
import torch
import torch.nn as nn

#《Tensor Low-Rank Reconstruction for Semantic Segmentation》
#论文地址：https://arxiv.org/pdf/2008.00490.pdf
class RecoNet(torch.nn.Module):
    '''
    func:
    Parameter
    ---------
    r: int
        重复多少次TGM: 低阶张量生成模块
    '''
    def __init__(self, r = 64):
        super(RecoNet, self).__init__()

        self.r = r
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.conv = None
        self.sigmoid = torch.nn.Sigmoid()
        
        self.parameter_r = torch.nn.Parameter(torch.ones(r), requires_grad = True)
     
    def forward(self, X):
        '''
        func: TRM模块,多个r的低阶张量组合
        '''
        assert len(X.size()) == 4
        batch, channel, height, width = X.size()
        
        for i in torch.arange(self.r):
            if i == 0:
                y = self.TGM_All(X) * self.parameter_r[i]
            else:
                y += self.TGM_All(X)* self.parameter_r[i]
                
        return y * X
    
    def TGM_All(self, X):
        '''
        func: 分别以 C、H 和W为通道进行通道注意力计算
        Parameter
        ---------
        X: 4D-Tensor
            X.size ---> [batch, channel, height, width]
        '''

        assert len(X.size()) == 4
        batch, channel, height, width = X.size()   
                
        C_weight = self.TGM_C(self, X)
        H_weight = self.TGM_C(self, X.permute(0,2,1,3)).permute(0,2,1,3)
        W_weight = self.TGM_C(self, X.permute(0,3,2,1)).permute(0,3,2,1)
        
        A = C_weight * H_weight * W_weight  
        
        return A

    @staticmethod
    def TGM_C(self, X):
        '''
        func: 通道注意力
        Parameter
        ---------
        X: 4D-Tensor
            X.size ---> [batch, channel, height, width]
        '''
                
        assert len(X.size()) == 4
        batch, channel, height, width = X.size()
        
        self.conv = torch.nn.Conv2d(channel, channel, kernel_size = 1)
        
        y = self.avg_pool(X)
        y = self.conv(y)
        y = self.sigmoid(y)
        
        return y


