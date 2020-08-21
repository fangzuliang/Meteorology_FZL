# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:04:06 2020

@author: fangzuliang
"""
import torch
import torch.nn as nn

#《Dual Attention Network for Scene Segmentation》
#论文地址 https://arxiv.org/pdf/1809.02983.pdf
#github地址：https://github.com/junfu1115/DANet/    

#空间自注意力的实现参考4.1小节

class CAM_Module(torch.nn.Module):
    
    '''
    func: 通道自注意力
    Parameter
    ---------
    in_dim: int
        输入的特征图的通道数
    '''
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        
        self.in_dim = in_dim
 
        self.gamma = torch.nn.Parameter(torch.zeros(1))  # β尺度系数初始化为0，并逐渐地学习分配到更大的权重
        self.softmax  = torch.nn.Softmax(dim=-1)  # 对每一行进行softmax
        
    def forward(self,x):
        """
        inputs: 4D Tensor
            x : input feature maps( B × C × H × W)
        returns: 4D Tensor
            out : attention value + input feature
            attention: B × C × C
        """
        m_batchsize, C, height, width = x.size()
        # A -> (N,C,HW)
        proj_query = x.view(m_batchsize, C, -1)
        # A -> (N,HW,C)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        
        # 矩阵乘积，通道注意图：X -> (N,C,C)
        energy = torch.bmm(proj_query, proj_key)
        
        # 这里实现了softmax用最后一维的最大值减去了原始数据，获得了一个不是太大的值
        # 沿着最后一维的C选择最大值，keepdim保证输出和输入形状一致，除了指定的dim维度大小为1
        # expand_as表示以复制的形式扩展到energy的尺寸
        energy = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        
        attention = self.softmax(energy)
        # A -> (N,C,HW)
        proj_value = x.view(m_batchsize, C, -1)
        # XA -> （N,C,HW）
        out = torch.bmm(attention, proj_value)
        # output -> (N,C,H,W)
        out = out.view(m_batchsize, C, height, width)
        
        out = self.gamma*out + x
        return out

