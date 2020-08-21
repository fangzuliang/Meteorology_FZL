# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:04:06 2020

@author: fangzuliang
"""
import torch
import torch.nn as nn

#ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
#论文地址: https://arxiv.org/abs/1910.03151
#代码地址：https://github.com/BangguWu/ECANet

class ECA(nn.Module):
    '''
    func: 实现ECANet的通道注意力 
    Parameter
    ---------
    channel: int
        输入的特征图的通道数
    k_size: int
        一维通道卷积中的kernel_size, 默认3
    '''
    def __init__(self, channel, k_size=3):
        
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        x: 4D-Tensor
            size = [batch,channel,height,width]
        '''
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        
        #对通道做一维卷积
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
    
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

x = torch.randn((2,8,20,20))
eca = ECA(8)
y = eca(x)
print(y.size()) 

'''
y.size: torch.Size([2, 8, 20, 20])
'''
