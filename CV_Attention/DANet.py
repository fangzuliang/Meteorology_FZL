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

#空间自注意力的实现参考
class Self_Attn_Spatial(nn.Module):
    """ 
    func: Self attention Spatial Layer 自注意力机制.通过类似Transformer中的Q K V来实现
    Parameter
    ---------
    in_dim: int
    	输入的通道数	
    out_dim: int
    	在进行self attention时生成Q,K矩阵的列数, 一般默认为in_dim//8
    """
    def __init__(self,in_dim,out_dim):
        super(Self_Attn_Spatial,self).__init__()
        self.chanel_in = in_dim
        
        self.out_dim = out_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = out_dim , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = out_dim , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad = True)
 
        self.softmax  = nn.Softmax(dim=-1)
        
    def forward(self,x):
        """
        Parameter
        ---------
        X: int
        	input feature maps( B X C X W X H) 
        returns
        -------
        out : self attention value + input feature
              attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        
        #proj_query中的第i行表示第i个像素位置上所有通道的值。size = B X N × C1
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) 
        
        #proj_key中的第j行表示第j个像素位置上所有通道的值，size = B X C1 x N
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) 
        
        #Energy中的第(i,j)是将proj_query中的第i行与proj_key中的第j行点乘得到
        #energy中第(i,j)位置的元素是指输入特征图第j个元素对第i个元素的影响，
        #从而实现全局上下文任意两个元素的依赖关系
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        
        #对行的归一化，对于(i,j)位置即可理解为第j位置对i位置的权重，所有的j对i位置的权重之和为1
        attention = self.softmax(energy) # B X N X N
        
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        out = torch.bmm(proj_value,attention.permute(0,2,1)) #B X C X N
        out = out.view(m_batchsize,C,width,height) #B X C X W X H
        
        #跨连，Gamma是需要学习的参数
        out = self.gamma*out + x #B X C X W X H
        
        return out,attention

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

class DANet(nn.Module):
    
    def __init__(self,in_dim):
        
        super(DANet, self).__init__()
        
        self.in_dim = in_dim
        self.ca = CAM_Module(self.in_dim)
        self.sa = Self_Attn_Spatial(self.in_dim, self.in_dim // 4)
        
    def forward(self, x):
        
        x1 = self.ca(x)
        x2 = self.sa(x)[0]
        
        y = x1 + x2
        
        return y
        
        
# x = torch.randn(size = (4,16,20,20))  
# Danet = DANet(16)
# y = Danet(x)
# print('y.size:',y.size())       