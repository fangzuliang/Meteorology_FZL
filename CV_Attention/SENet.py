# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:04:06 2020

@author: fangzuliang
"""
import torch
import torch.nn as nn

#%%
#SENet:Squeeze-and-Excitation Networks  
#通道注意力
#论文地址：https://arxiv.org/abs/1709.01507
#代码地址：https://github.com/hujie-frank/SENet

class SELayer(nn.Module):
    '''
    func: 实现通道Attention. 
    parameters
    ----------
    channel: int
    	input的通道数, input.size = (batch,channel,w,h) if batch_first else (channel,batch,,w,h)
    reduction: int
    	默认4. 即在FC的时,存在channel --> channel//reduction --> channel的转换
    batch_first: bool
    	默认True.如input为channel_first，则batch_first = False
    '''
    def __init__(self, channel,reduction = 2, batch_first = True):
        super(SELayer, self).__init__()
        
        self.batch_first = batch_first
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
            nn.Linear(channel,channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid()
            )
        
    def forward(self, x):
        '''
        Parameter
        --------
        X: 4D-Tensor
        	输入的feature map
        '''
        if not self.batch_first:
            x = x.permute(1,0,2,3)  
            
        b, c, _, _ = x.size() 
        y = self.avg_pool(x).view(b,c) #size = (batch,channel)
                
        y = self.fc(y).view(b,c,1,1)  #size = (batch,channel,1,1)
        out = x * y.expand_as(x) #size = (batch,channel,w,h)
        
        if not self.batch_first: 
            out = out.permute(1,0,2,3) #size = (channel,batch,w,h)

        return out 
    
x = torch.randn(size = (4,8,20,20))        
selayer = SELayer(channel = 8, reduction = 2, batch_first = True)
out = selayer(x)    
print(out.size()) 

'''
output: 
torch.Size([4, 8, 20, 20])
'''   

