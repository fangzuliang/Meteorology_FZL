# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:18:53 2020
常见Attention模块
@author: fzl
"""
#%%
'''
1. SELayer()  #SENet
2. CBAtten_Res()  #CBAM
3. SKEConv()  #SKENet
4. ECA()      #ECANet
5. RecoNet    #《Tensor Low-Rank Reconstruction for Semantic Segmentation》
6. Self_Atten_Spatial()  #self-attention
7. Self_Atten_Channel()  #self-attention
8. NonLocalBlockND()  #Non-local
9. DANet()   #DANet
10. GCNet_Atten()  #GCNet
11. CC_module()   #CCNet 
12. RowAttention() #Axial Attention 的横向注意力
13. ColAttention() #Axial Attention 的纵向注意力 
14. AxialAttention() #来自开源的github库
'''

#%%

import torch
import torch.nn as nn
from torch.nn import functional as F
from axial_attention import AxialAttention

#%%

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
    


#%%
####CBAM

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
    

#%%
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
    
    
#%%
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

#%%

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

# x = torch.randn(size = (4,16,20,20))  
# self_atten_spatial = Self_Attn_Spatial(16,4)
# y = self_atten_spatial(x)
# print('y.size:',y[0].size())   

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

# x = torch.randn(size = (4,8,20,20))  
# self_atten_channel = Self_Attn_Channel(8, 8)
# y = self_atten_channel(x)
# print('y.size:',y[0].size()) 

# '''
# output:
# y.size: torch.Size([4, 8, 20, 20])
# ''' 


#%%

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

# x = torch.randn(size = (4,16,20,20))  
# non_local = NonLocalBlockND(16,inter_channels = 8,dimension = 2)
# y = non_local(x)
# print('y.size:',y.size())

# '''
# output:
# y.size: torch.Size([4, 16, 20, 20])
# '''

#%%

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
        
#%%       
        
class GCNet_Atten(torch.nn.Module):
    '''
    func:
    Parameter
    ---------
    in_dim: int
        输入的特征图的通道数
    reduction: int
        default 4. 通道权重融合过程中的通道减少倍数
    fusion_type: str
        ['add','mul']. 默认'mul'
        输入的特征图和self-attention权重的结合方式     
        'mul': 输出为权重与X相乘 
        'add': 输出为权重 + X
    '''
    def __init__(self, in_dim, 
                 reduction = 4,
                 fusion_type = 'mul'
                 ):
    
        self.in_dim = in_dim
        self.fusion_type = fusion_type
        
        self.out_dim = self.in_dim // reduction if self.in_dim >= reduction else 1 
    
        super(GCNet_Atten, self).__init__()
        
        assert self.fusion_type in ['add','mul']

        self.conv_mask = torch.nn.Conv2d(self.in_dim, 1, kernel_size=1)
        self.softmax = torch.nn.Softmax(dim=2)

            
        self.channel_conv = torch.nn.Sequential(
                            torch.nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1),
                            torch.nn.LayerNorm([self.out_dim, 1, 1]),
                            torch.nn.ReLU(inplace=True),  
                            torch.nn.Conv2d(self.out_dim, self.in_dim, kernel_size=1))
        
      
    def forward(self, X):
        '''
        X: 4D-Tensor
        '''
        batch, channel, height, width = X.size()

        ##Query
        input_x = X
        input_x = input_x.view(batch, channel, height * width)
        input_x = input_x.unsqueeze(1) #size = [B,1,C,H*W]
        
        ##Key
        context_mask = self.conv_mask(X)
        context_mask = context_mask.view(batch, 1, height * width)  # [B, 1, H * W]
        context_mask = self.softmax(context_mask) # [B, 1, H * W]
        context_mask = context_mask.unsqueeze(-1)     # [B, 1, H*W, 1]
        
        ##weight
        context = torch.matmul(input_x, context_mask) #[B,1,C,1] 
        context = context.permute(0,2,1,3)   #[B, C, 1, 1]
        
        context = self.channel_conv(context) #[B, C, 1, 1]
        if self.fusion_type == 'add':
            y = X + context
        else: 
            y = X * torch.sigmoid(context)
        
        return y         

#%%

def INF(B,H,W):
    '''
    生成(B*W,H,H)大小的对角线为inf的三维矩阵
    Parameters
    ----------
    B: batch
    H: height
    W: width
    '''
    return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class CC_module(nn.Module):
    
    def __init__(self,in_dim):
        '''
        Parameters
        ----------
        in_dim : int
            channels of input
     	device: torch.device
        '''
        super(CC_module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
          
    def forward(self, x):
        
        self.gamma = self.gamma.to(x.device)

        m_batchsize, _, height, width = x.size()
        
        proj_query = self.query_conv(x) #size = (b,c2,h,w), c1 = in_dim, c2 = c1 // 8
        
        #size = (b*w, h, c2)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1) 
        
        #size = (b*h, w, c2)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        
        proj_key = self.key_conv(x) #size = (b,c2,h,w)
        
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height) 
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width) #size = (b*w,c2,h)
        
        proj_value = self.value_conv(x) #size = (b,c1,h,w)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height) #size = (b*w,c1,h)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width) #size = (b*h,c1,w)
        
        #size = (b*w, h,h) ,其中[:,i,j]表示Q所有W的第Hi行的所有通道值与K上所有W的第Hj列的所有通道值的向量乘积 
        energy_H = torch.bmm(proj_query_H, proj_key_H) 
        
        #size = (b,h,w,h) #这里为什么加 INF并没有理解
        energy_H = (energy_H + self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        
        #size = (b*h,w,w),其中[:,i,j]表示Q所有H的第Wi行的所有通道值与K上所有H的第Wj列的所有通道值的向量乘积
        energy_W = torch.bmm(proj_query_W, proj_key_W)
        energy_W = energy_W.view(m_batchsize,height,width,width) #size = (b,h,w,w)
        
        concate = self.softmax(torch.cat([energy_H, energy_W], 3)) #size = (b,h,w,h+w) #softmax归一化
        #concate = concate * (concate>torch.mean(concate,dim=3,keepdim=True)).float()
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height) #size = (b*w,h,h)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width) #size = (b*h,w,w)
        
        #size = (b*w,c1,h) #[:,i,j]表示V所有W的第Ci行通道上的所有H 与att_H的所有W的第Hj列的h权重的乘积  
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1))
        out_H = out_H.view(m_batchsize,width,-1,height).permute(0,2,3,1)  #size = (b,c1,h,w)
        
        #size = (b*h,c1,w) #[:,i,j]表示V所有H的第Ci行通道上的所有W 与att_W的所有H的第Wj列的W权重的乘积  
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1))
        out_W = out_W.view(m_batchsize,height,-1,width).permute(0,2,1,3) #size = (b,c1,h,w)
        #print(out_H.size(),out_W.size())
        
        return self.gamma*(out_H + out_W) + x 
    
# model = CC_module(8)
# x = torch.randn(4, 8, 20, 20)
# out = model(x)
# print(out.shape)   

#%%
