# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:04:06 2020

@author: fangzuliang
"""
import torch
import torch.nn as nn

#视觉应用中的Self-Attention-Spatial机制 

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

x = torch.randn(size = (4,16,20,20))  
self_atten_spatial = Self_Attn_Spatial(16,4)
y = self_atten_spatial(x)
print('y.size:',y[0].size())   

'''
y.size: torch.Size([4, 16, 20, 20])
'''

#视觉应用中的Self-Attention-Channel机制 

class Self_Attn_Channel(nn.Module):
    """ 
    func: Self attention Channel Layer 自注意力机制.通过类似Transformer中的Q K V来实现
    Parameter
    ---------
    in_dim: int
    	输入的通道数
    out_dim: int 
    	在进行self attention时生成Q,K矩阵的列数, 默认可选取为：in_dim
    """
    def __init__(self,in_dim,out_dim ):
        super(Self_Attn_Channel,self).__init__()
        self.chanel_in = in_dim
        self.out_dim = out_dim
 
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = out_dim , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = out_dim , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = out_dim , kernel_size= 1)
        self.x_conv = nn.Conv2d(in_channels = in_dim , out_channels = out_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))
 
        self.softmax  = nn.Softmax(dim=-1)
        
    def forward(self,x):
        """
        x : input feature maps( B X C0 X W X H)
        """
        #C0 = in_dim; C1 = out_dim
        
        m_batchsize,C0,width ,height = x.size() 
        
        #proj_query中的第i行表示第i个通道位置上所有像素的值: size = B X C1 × N
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height) 
        
        #proj_key中的第j行表示第j个通道位置上所有像素的值，size = B X N x C1
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) 
        
        #Energy中的第(i,j)是将proj_query中的第i行与proj_key中的第j行点乘得到
        #energy中第(i,j)位置的元素是指输入特征图第j个通道对第i个通道的影响，
        #从而实现全局上下文任意两个通道的依赖关系. size = B X C1 X C1
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        
        #对于(i,j)位置即可理解为第j通道对i通道的权重，所有的j对i通道的权重之和为1
        #对行进行归一化，即每行的所有列加起来为1
        attention = self.softmax(energy) # B X C1 X C1
        
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C1 X N
        out = torch.bmm(attention, proj_value) #B X C1 X N
        out = out.view(m_batchsize,self.out_dim, width,height) #B X C1 X W X H
        
        #跨连，Gamma是需要学习的参数
        out = self.gamma*out + self.x_conv(x) #B X C1 X W X H
        
        return out,attention

x = torch.randn(size = (4,8,20,20))  
self_atten_channel = Self_Attn_Channel(8, 8)
y = self_atten_channel(x)
print('y.size:',y[0].size()) 

'''
output:
y.size: torch.Size([4, 8, 20, 20])
''' 

