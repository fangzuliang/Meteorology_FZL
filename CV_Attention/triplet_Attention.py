# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:38:06 2020
Triplet Attention

@author: fzl
"""

'''
https://mp.weixin.qq.com/s/WYBswnB9duO-DLSwG_F6QQ
https://github.com/LandskapeAI/triplet-attention
https://arxiv.org/abs/2010.03045
'''

import torch
import torch.nn as nn

class BasicConv(nn.Module):
    '''
    func: 卷积模块，流程为: conv + bn + relu
    Parameter
    ---------
    in_dim: int 
        特征图的输入通道数
    out_dim: int
        输出通道数
    kernel_size: int
        卷积核尺寸，default 3
    stride: int
        卷积滑动步长，可选 1,2 . default 2
        when 1: 保持原尺寸
        when 2: 尺寸减半
    relu: bool
        default: True. 是否添加激活函数, 默认激活函数为relu
    bn: bool
        default: True. 是否添加BN层
    Returns
    -------
    '''
    def __init__(self,
                 in_dim, out_dim, 
                 kernel_size = 3, stride=1, 
                 relu=True, bn=True, 
                 bias=False):
                 
        
        super(BasicConv, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(self.in_dim, self.out_dim,
                              kernel_size=kernel_size, stride=stride, 
                              padding=padding, bias=bias)
                        
        self.bn = nn.BatchNorm2d(self.out_dim,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        '''
        x: 4D-Tensor ---> [batch, in_dim, height, width]
        '''
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    '''
    func: 对输入的特征图在通道维上进行最大池化和平均池化
    '''
    def forward(self, x):
        '''
        x: 4D-Tensor ---> [batch, in_dim, height, width]
        '''
        
        max_pool_x = torch.max(x,dim = 1)[0].unsqueeze(1) #size = (batch,1,height,width)
        mean_pool_x = torch.mean(x,dim = 1).unsqueeze(1)
        
        y = torch.cat([max_pool_x,mean_pool_x],dim = 1) #size = (batch,2,height,width)
        
        return y
    
  
class SpatialGate(nn.Module):
    '''
    func: triplet_attention的范式
    '''
    def __init__(self):
        
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, relu=False)
        
    def forward(self, x):
        '''
        x: 4D-Tensor ---> [batch, in_dim, height, width]
        ''' 
        x_compress = self.compress(x) #size = (batch,2,height,width)
        x_out = self.spatial(x_compress) #size = (batch,1,heigth,width)
        scale = torch.sigmoid_(x_out) 
        
        return x * scale

class TripletAttention(nn.Module):
    '''
    func:
    Parameter
    ---------
    no_spatial: bool
        是否不添加空间注意力，default False.即添加空间注意力
    '''
    def __init__(self, 
                 pool_types=['avg', 'max'], 
                 no_spatial=False):
        
        super(TripletAttention, self).__init__()
        
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial=no_spatial
        
        if not no_spatial:
            self.SpatialGate = SpatialGate()
            
    def forward(self, x):
        '''
        x: 4D-Tensor ---> [batch, in_dim, height, width]
        '''     
        
        #分支一：对H维进行 通道+空间注意力
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        
        #分支二：对W维进行 通道+空间注意力
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        
        if not self.no_spatial:
            #分支三： 对C维进行空间注意力
            x_out = self.SpatialGate(x)
            x_out = (1/3)*(x_out + x_out11 + x_out21)
        else:
            x_out = (1/2)*(x_out11 + x_out21)
            
        return x_out
    
    
# device = torch.device('cpu:0')
# device = torch.device('cuda:0')
# x = torch.randn((2,1,20,20))

# atten = TripletAttention()
# y = atten(x)
# print(y.size())
# print(y.sum())




