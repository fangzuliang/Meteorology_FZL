# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:04:06 2020

@author: fangzuliang
"""
import torch
import torch.nn as nn

#CBAM：Convolutional Block Attention Module（CBAM）
#论文地址：https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf

class ChannelAttention(nn.Module):
    '''
    func: 实现通道Attention. 
    parameters
    ----------
    in_channels: int
    	input的通道数, input.size = (batch,channel,w,h) if batch_first else (channel,batch,,w,h)    
    reduction: int
    	默认4. 即在FC的时,存在in_channels --> in_channels//reduction --> in_channels的转换
    batch_first: bool
    	默认True.如input为channel_first，则batch_first = False
    '''
    def __init__(self,in_channels, reduction = 4, batch_first = True):
        
        super(ChannelAttention,self).__init__()
        
        self.batch_first = batch_first
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size = 1, bias = False),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size = 1, bias = False),
            )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        if not self.batch_first: 
            x = x.permute(1,0,2,3) 
        
        avgout = self.sharedMLP(self.avg_pool(x)) #size = (batch,in_channels,1,1)
        maxout = self.sharedMLP(self.max_pool(x)) #size = (batch,in_channels,1,1)
        
        w = self.sigmoid(avgout + maxout) #通道权重  size = (batch,in_channels,1,1)
        out = x * w.expand_as(x) #返回通道注意力后的值 size = (batch,in_channels,w,h)
        
        if not self.batch_first:
            out = out.permute(1,0,2,3) #size = (channel,batch,w,h)

        return out
    
class SpatialAttention(nn.Module):
    '''
    func: 实现空间Attention. 
    parameters
    ----------
    kernel_size: int
    	卷积核大小, 可选3,5,7,
    batch_first: bool
    	默认True.如input为channel_first，则batch_first = False
    '''
    def __init__(self, kernel_size = 3, batch_first = True):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,5,7), "kernel size must be 3 or 7"
        padding = kernel_size // 2
        
        self.batch_first = batch_first
        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        if not self.batch_first:
            x = x.permute(1,0,2,3)  #size = (batch,channels,w,h)
        
        avgout = torch.mean(x, dim=1, keepdim=True) #size = (batch,1,w,h)
        maxout,_ = torch.max(x, dim=1, keepdim=True)  #size = (batch,1,w,h)
        x1 = torch.cat([avgout, maxout], dim=1)    #size = (batch,2,w,h)
        x1 = self.conv(x1)    #size = (batch,1,w,h)
        w = self.sigmoid(x1)   #size = (batch,1,w,h)  
        out = x * w            #size = (batch,channels,w,h)

        if not self.batch_first:
            out = out.permute(1,0,2,3) #size = (channels,batch,w,h)

        return  out
    

class CBAtten_Res(nn.Module):
    '''
    func: channel-attention + spatial-attention + resnet
    parameters
    ----------
    in_channels: int
    	input的通道数, input.size = (batch,in_channels,w,h) if batch_first else (in_channels,batch,,w,h);
    out_channels: int
    	输出的通道数
    kernel_size: int
    	默认3, 可选[3,5,7]
    stride: int
    	默认2, 即改变out.size --> (batch,out_channels,w/stride, h/stride).
                一般情况下，out_channels = in_channels * stride
    reduction: int
    	默认4. 即在通道atten的FC的时,存在in_channels --> in_channels//reduction --> in_channels的转换
    batch_first: bool
    	默认True.如input为channel_first，则batch_first = False
    '''
    def __init__(self,in_channels,out_channels,kernel_size = 3, 
                 stride = 2, reduction = 4,batch_first = True):
        
        super(CBAtten_Res,self).__init__()
        
        self.batch_first = batch_first
        self.reduction = reduction
        self.padding = kernel_size // 2
        
        
        #h/2, w/2
        self.max_pool = nn.MaxPool2d(3, stride = stride, padding = self.padding)
        self.conv_res = nn.Conv2d(in_channels, out_channels,
                               kernel_size = 1,
                               stride = 1,
                               bias = True)
        
        
        #h/2, w/2
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size = kernel_size,
                               stride = stride, 
                               padding = self.padding,
                               bias = True)
        self.bn1 = nn.BatchNorm2d(out_channels) 
        self.relu = nn.ReLU(inplace = True)
        self.ca = ChannelAttention(out_channels, reduction = self.reduction,
                                   batch_first = self.batch_first)
        
        self.sa = SpatialAttention(kernel_size = kernel_size,
                                   batch_first = self.batch_first)
        
        
    def forward(self,x):
        
        if not self.batch_first:
            x = x.permute(1,0,2,3)  #size = (batch,in_channels,w,h)
        residual = x 
        
        out = self.conv1(x)   #size = (batch,out_channels,w/stride,h/stride)
        out = self.bn1(out) 
        out = self.relu(out) 
        out = self.ca(out)
        out = self.sa(out)  #size = (batch,out_channels,w/stride,h/stride)
        
        residual = self.max_pool(residual)  #size = (batch,in_channels,w/stride,h/stride)
        residual = self.conv_res(residual)  #size = (batch,out_channels,w/stride,h/stride)
        
        out += residual #残差
        out = self.relu(out)  #size = (batch,out_channels,w/stride,h/stride)
        
        if not self.batch_first:
            out = out.permute(1,0,2,3) #size = (out_channels,batch,w/stride,h/stride) 
            
        return out
    
    
x = torch.randn(size = (4,8,20,20))  
cba = CBAtten_Res(8,16,reduction = 2,stride = 1) 
y = cba(x)
print('y.size:',y.size())   

'''
y.size: torch.Size([4, 16, 20, 20])
'''

