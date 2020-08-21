# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:04:06 2020

@author: fangzuliang
"""
import torch
import torch.nn as nn

#论文地址：https://arxiv.org/abs/1711.07971
#代码地址：https://github.com/pprp/SimpleCVReproduction/tree/master/attention/Non-local/Non-Local_pytorch_0.4.1_to_1.1.0/lib

import torch
from torch import nn
from torch.nn import functional as F

class NonLocalBlockND(nn.Module):
    """
    func: 非局部信息统计的注意力机制
    Parameter
    ---------
    in_channels: int
    	输入的通道数
    inter_channels: int
    	生成attention时Conv的输出通道数，一般为in_channels//2.
                        如果为None, 则自动为in_channels//2
    dimension: int
    	默认2.可选为[1,2,3]，
        1：输入为size = [batch,in_channels, width]或者[batch,time_steps,seq_length]，可表示时序数据
        2: 输入size = [batch, in_channels, width,height], 即图片数据
        3: 输入size = [batch, time_steps, in_channels, width,height]，即视频数据                
    sub_sample: bool
    	默认True,是否在Attention过程中对input进行size降低，即w,h = w//2, h//2               
    bn_layer: bool
    	默认True
    
    """
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=2,
                 sub_sample=True,
                 bn_layer=True):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            # 进行压缩得到channel个数
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)
            
            
    def forward(self, x):
        '''
        #if dimension == 3 , N = w*h*t ; if sub_sample: N1 = (w//2) * (h//2) * t ,else: N1 = N
        #if dimension == 2 , N = w*h  
        #if dimension == 1 , N = w 
        #C0 = in_channels;   C1 = inter_channels
		''' 
        batch_size = x.size(0) 

        g_x = self.g(x).view(batch_size, self.inter_channels, -1) #[B, C1, N1]
        g_x = g_x.permute(0, 2, 1) #[B, N1, C1]

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1) #[B, C1, N]
        theta_x = theta_x.permute(0, 2, 1) #[B, N, C1]

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1) #[B, C1, N1]
        
        f = torch.matmul(theta_x, phi_x) #[B,N,N1]

        # print(f.shape) 

        f_div_C = F.softmax(f, dim=-1) 

        y = torch.matmul(f_div_C, g_x) #[B,N,N1] *[B, N1, C1] = [B,N,C1] 
        y = y.permute(0, 2, 1).contiguous() #[B,C1,N] 

        size = [batch_size, self.inter_channels] + list(x.size()[2:])
        y = y.view(size)  #size = [B,N,C1,x.size()[2:]] 
        
        W_y = self.W(y)  #1 × 1 卷积 size = x.size()
        z = W_y + x  #残差连接
        return z 

x = torch.randn(size = (4,16,20,20))  
non_local = NonLocalBlockND(16,inter_channels = 8,dimension = 2)
y = non_local(x)
print('y.size:',y.size())

'''
output:
y.size: torch.Size([4, 16, 20, 20])
'''


