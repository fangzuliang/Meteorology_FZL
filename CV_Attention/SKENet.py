# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:04:06 2020

@author: fangzuliang
"""
import torch
import torch.nn as nn

#SKENet: Selective Kernel Networks
# 论文地址：https://arxiv.org/abs/1903.06586
# 代码地址：https://github.com/implus/SKNet

class SKEConv(nn.Module):
    '''
    func: 实现Selective Kernel Networks(SKE) Attention机制。主要由Spit + Fuse + Select 三个模块组成 
    parameters
    ----------
    in_channels: int
    	input的通道数;
    M: int
    	Split阶段. 使用不同大小的卷积核(M个)对input进行卷积，得到M个分支，默认2;
    G: int 
    	在卷积过程中使用分组卷积，分组个数为G, 默认为2.可以减小参数量;
    stride: int 
    	默认1. split卷积过程中的stride,也可以选2，降低输入输出的w,h;
    L: int
    	默认32; 
    reduction: int 
    	默认2，压缩因子; 在线性部分压缩部分，输出特征d = max(L, in_channels / reduction);    
    batch_first: bool 
    	默认True;  
    '''
    def __init__(self,in_channels, M = 2, G = 2, stride = 1, L = 32, reduction = 2, batch_first = True):
        
        super(SKEConv,self).__init__()
        
        self.M = 2
        self.in_channels = in_channels
        self.batch_first = batch_first
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 
                              kernel_size = 3 + i*2,
                              stride = stride,
                              padding = 1 + i,
                              groups = G),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace = True)
                    ))
        
        self.d = max(int(in_channels / reduction), L)
        self.fc = nn.Linear(in_channels, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(self.d,in_channels))
            
        self.softmax = nn.Softmax(dim = 1)
        
        
    def forward(self, x):
        
        if not self.batch_first:
            x = x.permutation(1,0,2,3)
            
        for i ,conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim = 1)  #size = (batch,1,in_channels,w,h)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas,fea],dim = 1) #size = (batch,M,in_channels,w,h)
        
        fea_U = torch.sum(feas,dim = 1) #size = (batch,in_channels,w,h)
        fea_s = fea_U.mean(-1).mean(-1) #size = (batch,in_channels)
        fea_z = self.fc(fea_s)  #size = (batch,d)
        
        for i,fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1) #size = (batch,1,in_channels)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors,vector],
                                              dim = 1)  #size = (batch,M,in_channels)
                
        attention_vectors = self.softmax(attention_vectors) #size = (batch,M,in_channels)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1) #size = (batch,M,in_channels,w,h) 
        fea_v = (feas * attention_vectors).sum(dim=1) #size = (batch,in_channels,w,h)
        
        if not self.batch_first:
            fea_v = fea_v.permute(1,0,2,3)
                    
        return fea_v
    
#%%
x = torch.randn(size = (4,8,20,20))  
ske = SKEConv(8,stride = 2)
y = ske(x)
print('y.size:',y.size())   

'''
y.size: torch.Size([4, 16, 10, 10])
'''
