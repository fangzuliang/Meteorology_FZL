# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:41:36 2020
损失函数集成
@author: fzl
"""
#%%
"""
1. BMAELoss() ;MAE损失中给强回波处的误差更大的权重，权重固定
2. BMSELoss() ;MSE损失中给强回波处的误差更大的权重，权重固定
3. BMSAELoss() ; MSE和MAE损失中给强回波处的误差更大的权重，同时将BMSE 和 BMAE按照不同权重累加起来,权重固定
4. STBMSELoss() ;MSE损失在空间中给强回波处的误差更大的权重,在时间序列上给时间靠后帧更多的权重, 权重固定
5. STBMAELoss() ;MSE损失在空间中给强回波处的误差更大的权重,在时间序列上给时间靠后帧更多的权重, 权重固定
6. STBMSAELoss() ; MSE和MAE损失中给强回波处的误差更大的权重，在时间序列上给时间靠后帧更多的权重,
                    同时将BMSE 和 BMAE按照不同权重累加起来, 权重固定
7. MultiMSELoss() ;图像金字塔(多尺寸)MSE损失
8. MultiMAELoss() ;图像金字塔(多尺寸)MAE损失
9. MultiBMSAELoss() ;图像金字塔(多尺寸)MSE or MAE or MSE+MAE损失 ,损失中给强回波处的误差更大的权重
10.IOULoss() ;在回归中使用IOU,即1-TS评分作为损失
11. BEXPRMSELoss() ;RMSE损失在空间中给强回波处的误差更大的权重,权重与真实回波呈指数关系
12. BEXPMSELoss() ;MSE损失在空间中给强回波处的误差更大的权重, 其中权重与真实回波值呈指数正比
13. BEXPMAELoss() ;MAE损失在空间中给强回波处的误差更大的权重, 其中权重与真实回波值呈指数正比
14. BEXPMSAELoss() ;MSE + MAE损失在空间中给强回波处的误差更大的权重, 其中权重与真实回波值呈指数正比
                  即将MSE + MAE结合使用;
15. BEXPRMSEIOULoss() ;RMSELoss + IOU损失
16. SSIM() ; 图像结构相似度，数值在 0-1之间；一般用 1 -SSIM()作为损失 ;structural similarity index
17. BMSAE_SSIM_Loss() ; BMSEA + SSIM损失的结合
18. BEXPMSAE_SSIM_Loss() ;BEXPMSAE + SSIM损失的组合
"""
#%%
import torch
#%%

class BMAELoss(torch.nn.Module):
    '''
    func: MAE损失中给强回波处的误差更大的权重
    Parameter
    ---------
    weights: list
        default [1,2,5,10,30].权重列表,给不同的回波强度处对应的像素点的损失不同的权重
    thresholds: list
        阈值列表，即将回波强度按照范围分为若干段，不同段给与不同的损失权重
        default [20,30,40,50,80].对应0~1之间的输入为: [0.25, 0.375, 0.5, 0.625, 1.0]
    '''
    def __init__(self, weights = [1,2,5,10,30], thresholds = [20,30,40,50,80]):
        super(BMAELoss,self).__init__()
        
        assert len(weights) == len(thresholds)
        scale = max(thresholds)
        self.weights = weights
        self.thresholds = [threshold/scale for threshold in thresholds] 
        #[0.25, 0.375, 0.5, 0.625, 1.0]
        
    def forward(self,y_pre,y_true):
        
        assert y_true.min() >= 0
        assert y_true.max() <= 1
        
        w_true = y_true.clone()
        
        for i in range(len(self.weights)):
            w_true[w_true < self.thresholds[i]] = self.weights[i]
            
        return torch.mean(w_true * (abs(y_pre - y_true)))
   
class BMSELoss(torch.nn.Module):
    '''
    func: MSE损失中给强回波处的误差更大的权重
    Parameter
    ---------
    weights: list
        default [1,2,5,10,30].权重列表,给不同的回波强度处对应的像素点的损失不同的权重
    thresholds: list
        阈值列表，即将回波强度按照范围分为若干段，不同段给与不同的损失权重
        default [20,30,40,50,80].对应0~1之间的输入为: [0.25, 0.375, 0.5, 0.625, 1.0]
    '''
    def __init__(self, weights = [1,2,5,10,30], thresholds = [20,30,40,50,80]):
        super(BMSELoss,self).__init__()
        
        assert len(weights) == len(thresholds)
        scale = max(thresholds)
        self.weights = weights
        self.thresholds = [threshold/scale for threshold in thresholds] 
        #[0.25, 0.375, 0.5, 0.625, 1.0]
        
    def forward(self,y_pre,y_true):
        
        assert y_true.min() >= 0
        assert y_true.max() <= 1
        
        w_true = y_true.clone()
        for i in range(len(self.weights)):
            w_true[w_true < self.thresholds[i]] = self.weights[i] #获取权重矩阵
            
        return torch.mean(w_true * (y_pre - y_true)**2)
    
  
class BMSAELoss(torch.nn.Module):
    '''
    func: MSE和MAE损失中给强回波处的误差更大的权重，同时将BMSE 和 BMAE按照不同权重累加起来
    Parameter
    ---------
    weights: list
        default [1,2,5,10,30].权重列表,给不同的回波强度处对应的像素点的损失不同的权重
    thresholds: list
        阈值列表，即将回波强度按照范围分为若干段，不同段给与不同的损失权重
        default [20,30,40,50,80].对应0~1之间的输入为: [0.25, 0.375, 0.5, 0.625, 1.0]
    mse_w: float
        mse权重, default 1
    mae_w: float
        mae权重, default 1
    '''
    def __init__(self, weights = [1,2,5,10,30], 
                 thresholds = [20,30,40,50,80],
                 mse_w = 1,mae_w = 1):
        super(BMSAELoss,self).__init__()
        
        assert len(weights) == len(thresholds)
        scale = max(thresholds)
        self.weights = weights
        self.thresholds = [threshold/scale for threshold in thresholds] 
        #[0.25, 0.375, 0.5, 0.625, 1.0]
        self.mse_w = mse_w
        self.mae_w = mae_w
        
    def forward(self,y_pre,y_true):
        
        assert y_true.min() >= 0
        assert y_true.max() <= 1
        
        w_true = y_true.clone()
        for i in range(len(self.weights)):
            w_true[w_true < self.thresholds[i]] = self.weights[i] #获取权重矩阵
            
        return self.mse_w*torch.mean(w_true * (y_pre - y_true)**2) + self.mae_w*torch.mean(w_true * (abs(y_pre - y_true)))
    
#%%
class STBMSELoss(torch.nn.Module):
    '''
    func: MSE损失在空间中给强回波处的误差更大的权重,在时间序列上给时间靠后帧更多的权重  
    Parameter
    ---------
    spatial_weights: list
        default [1,2,5,10,30].权重列表,给不同的回波强度处对应的像素点的损失不同的权重
    thresholds: list
        阈值列表，即将回波强度按照范围分为若干段，不同段给与不同的损失权重
        default [20,30,40,50,80].对应0~1之间的输入为: [0.25, 0.375, 0.5, 0.625, 1.0]
    time_weight_gap: int
        给不同时间雷达帧损失不同的权重，默认weight(t+1) - weight(t) = time_weight_gap
        default 1.如果为0，则表示所有时间帧权重一致
    '''
    def __init__(self, spatial_weights = [1,2,5,10,30],
                 thresholds = [20,30,40,50,80],time_weight_gap = 1):
        super(STBMSELoss,self).__init__()
        
        assert len(spatial_weights) == len(thresholds)
        scale = max(thresholds)
        self.spatial_weights = spatial_weights
        self.thresholds = [threshold/scale for threshold in thresholds] 
        #[0.25, 0.375, 0.5, 0.625, 1.0]
        
        self.time_weight_gap = time_weight_gap
        
    def forward(self,y_pre, y_true):
        '''
        Parameter
        ---------
        y_pre: 4D or 5D Tensor
            predict by model.
        y_true: 4D or 5D Tensor
            real value.
        '''
        
        assert y_true.min() >= 0
        assert y_true.max() <= 1
        
        w_true = y_true.clone()
        for i in range(len(self.spatial_weights)):
            w_true[w_true < self.thresholds[i]] = self.spatial_weights[i] #获取权重矩阵
        
        
        if len(y_true.size()) == 4:
            batch, seq, height, width = y_true.shape
            # y_true = np.expand_dims(y_true,axis = 2)
            # y_pre = np.expand_dims(y_pre,axis = 2)
            # w_true = np.expand_dims(w_true,axis = 2)
        
        if len(y_true.size()) == 5:
            batch,seq, channel,height,width = y_true.shape
            assert channel == 1
            
        time_weight = torch.arange(0,seq)*self.time_weight_gap + 1 
        time_weight = time_weight.to(y_pre.device)
        
        all_loss = 0
        for i in range(seq):
            loss = torch.mean(w_true[:,i]*(y_pre[:,i] - y_true[:,i])**2)
            all_loss += time_weight[i]*loss
        
        return all_loss
       
class STBMAELoss(torch.nn.Module):
    '''
    func: MAE损失在空间中给强回波处的误差更大的权重,在时间序列上给时间靠后帧更多的权重  
    Parameter
    ---------
    spatial_weights: list
        default [1,2,5,10,30].权重列表,给不同的回波强度处对应的像素点的损失不同的权重
    thresholds: list
        阈值列表，即将回波强度按照范围分为若干段，不同段给与不同的损失权重
        default [20,30,40,50,80].对应0~1之间的输入为: [0.25, 0.375, 0.5, 0.625, 1.0]
    time_weight_gap: int
        给不同时间雷达帧损失不同的权重，默认weight(t+1) - weight(t) = time_weight_gap
        default 1.如果为0，则表示所有时间帧权重一致
    '''
    def __init__(self, spatial_weights = [1,2,5,10,30],
                 thresholds = [20,30,40,50,80],time_weight_gap = 1):
        super(STBMAELoss,self).__init__()
        
        assert len(spatial_weights) == len(thresholds)
        scale = max(thresholds)
        self.spatial_weights = spatial_weights
        self.thresholds = [threshold/scale for threshold in thresholds] 
        #[0.25, 0.375, 0.5, 0.625, 1.0]
        
        self.time_weight_gap = time_weight_gap
        
    def forward(self,y_pre, y_true):
        '''
        Parameter
        ---------
        y_pre: 4D or 5D Tensor
            predict by model.
        y_true: 4D or 5D Tensor
            real value.
        '''
        assert y_true.min() >= 0
        assert y_true.max() <= 1
        
        w_true = y_true.clone()
        for i in range(len(self.spatial_weights)):
            w_true[w_true < self.thresholds[i]] = self.spatial_weights[i] #获取权重矩阵
        
        
        if len(y_true.size()) == 4:
            batch, seq, height, width = y_true.shape
            # y_true = np.expand_dims(y_true,axis = 2)
            # y_pre = np.expand_dims(y_pre,axis = 2)
            # w_true = np.expand_dims(w_true,axis = 2)
        
        if len(y_true.size()) == 5:
            batch,seq, channel,height,width = y_true.shape
            assert channel == 1
            
        time_weight = torch.arange(0,seq)*self.time_weight_gap + 1 
        time_weight = time_weight.to(y_pre.device)
        
        all_loss = 0
        for i in range(seq):
            loss = torch.mean(w_true[:,i]*abs(y_pre[:,i] - y_true[:,i]))
            all_loss += time_weight[i]*loss
        
        return all_loss
    
class STBMSAELoss(torch.nn.Module):
    '''
    func: MSE和MAE损失中给强回波处的误差更大的权重，同时将BMSE 和 BMAE按照不同权重累加起来
    Parameter
    ---------
    weights: list
        default [1,2,5,10,30].权重列表,给不同的回波强度处对应的像素点的损失不同的权重
    thresholds: list
        阈值列表，即将回波强度按照范围分为若干段，不同段给与不同的损失权重
        default [20,30,40,50,80].对应0~1之间的输入为: [0.25, 0.375, 0.5, 0.625, 1.0]
    mse_w: float
        mse权重, default 1
    mae_w: float
        mae权重, default 1
    '''    
    def __init__(self, spatial_weights = [1,2,5,10,30],
                 thresholds = [20,30,40,50,80],time_weight_gap = 1,
                 mse_w = 1,mae_w = 1):
        super(STBMSAELoss,self).__init__()
        
        assert len(spatial_weights) == len(thresholds)
        scale = max(thresholds)
        self.spatial_weights = spatial_weights
        self.thresholds = [threshold/scale for threshold in thresholds] 
        #[0.25, 0.375, 0.5, 0.625, 1.0]
        
        self.time_weight_gap = time_weight_gap
        self.mse_w = mse_w
        self.mae_w = mae_w
        
    def forward(self,y_pre, y_true):
        '''
        Parameter
        ---------
        y_pre: 4D or 5D Tensor
            predict by model.
        y_true: 4D or 5D Tensor
            real value.
        '''
        assert y_true.min() >= 0
        assert y_true.max() <= 1
        
        mse_loss = STBMSELoss(self.spatial_weights,
                              self.thresholds,
                              self.time_weight_gap).forward(y_pre, y_true)
        
        mae_loss = STBMAELoss(self.spatial_weights,
                              self.thresholds,
                              self.time_weight_gap).forward(y_pre, y_true)
        
        loss = self.mse_w * mse_loss + self.mae_w * mae_loss
        
        return loss
    
#%%
class MultiMSELoss(torch.nn.Module):
    '''
    func: 图像金字塔(多尺寸)MSE损失
    Parameter
    ---------
    cascades: int
        级联数量,即图像金字塔的层数，每一层都计入损失，图像尺寸越小，损失权重越大
        default 3
    '''
    
    def __init__(self,cascades = 3):
        super(MultiMSELoss,self).__init__()
        self.cascades = cascades
        self.MaxPool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2)
                                            
        
    def forward(self,y_pre,y_true):
        '''
        Parameter
        ---------
        y_pre: 4D or 5D Tensor
            predict by model.
        y_true: 4D or 5D Tensor
            real value.
        ---------
        '''
        
        assert len(y_pre.size()) in [4,5]
        
        if len(y_pre.size()) == 5:
            batch,seq,channel,height,width = y_pre.size()
            y_pre = y_pre.view(batch,seq*channel,height,width)
            y_true = y_true.view(batch,seq*channel,height,width)
            
        loss_fn = torch.nn.MSELoss(reduction='sum')
        self.MaxPool2d.to(y_pre.device)
        
        all_loss = loss_fn(y_pre,y_true)
        # all_loss = 0
        for i in range(self.cascades):
            y_pre, y_true = self.MaxPool2d(y_pre),self.MaxPool2d(y_true)
            index = 2**(i+1)
            all_loss += loss_fn(y_pre,y_true)*(2**index)
        
        return all_loss
    
    
class MultiMAELoss(torch.nn.Module):
    '''
    func: 图像金字塔(多尺寸)MAE损失
    Parameter
    ---------
    cascades: int
        级联数量,即图像金字塔的层数，每一层都计入损失，图像尺寸越小，损失权重越大
        default 3
    '''
    
    def __init__(self,cascades = 3):
        super(MultiMAELoss,self).__init__()
        self.cascades = cascades
        self.MaxPool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2)
                                            
        
    def forward(self,y_pre,y_true):
        '''
        Parameter
        ---------
        y_pre: 4D or 5D Tensor
            predict by model.
        y_true: 4D or 5D Tensor
            real value.
        ---------
        '''
        assert len(y_pre.size()) in [4,5]
        
        if len(y_pre.size()) == 5:
            batch,seq,channel,height,width = y_pre.size()
            y_pre = y_pre.view(batch,seq*channel,height,width)
            y_true = y_true.view(batch,seq*channel,height,width)
            
        loss_fn = torch.nn.L1Loss()
        self.MaxPool2d.to(y_pre.device)
        
        all_loss = loss_fn(y_pre,y_true)
        # all_loss = 0
        for i in range(self.cascades):
            y_pre, y_true = self.MaxPool2d(y_pre),self.MaxPool2d(y_true)
            index = 2**(i+1) 
            all_loss += loss_fn(y_pre,y_true)*(2**index)
        
        return all_loss
        

class MultiBMSAELoss(torch.nn.Module):
    '''
    func: 图像金字塔(多尺寸)MSE or MAE or MSE+MAE损失 ,损失中给强回波处的误差更大的权重
    Parameter
    ---------
    cascades: int
        级联数量,即图像金字塔的层数，每一层都计入损失，图像尺寸越小，损失权重越大
        default 3
    flag: int
        每层金字塔的损失函数，可选为BMSE,BMAE,BMSAE
        flag == 0: loss_fn = BMSELoss()  
        flag == 1: loss_fn = BMAELoss()
        flag == 2: loss_fn = BMSAELoss()
    '''
    
    def __init__(self,cascades = 3,flag=2):
        super(MultiBMSAELoss,self).__init__()
        self.cascades = cascades
        self.flag = flag
        self.MaxPool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2)
                                            
        
    def forward(self,y_pre,y_true):
        '''
        Parameter
        ---------
        y_pre: 4D or 5D Tensor
            predict by model.
        y_true: 4D or 5D Tensor
            real value.
        ---------
        '''
        assert y_true.min() >= 0
        assert y_true.max() <= 1
        
        assert len(y_pre.size()) in [4,5]
        
        if len(y_pre.size()) == 5:
            batch,seq,channel,height,width = y_pre.size()
            y_pre = y_pre.view(batch,seq*channel,height,width)
            y_true = y_true.view(batch,seq*channel,height,width)
        
        if self.flag == 0:
            loss_fn = BMSELoss()
        elif self.flag == 1:
            loss_fn = BMAELoss()
        elif self.flag == 2:
            loss_fn = BMSAELoss()
        
        self.MaxPool2d.to(y_pre.device)
        
        all_loss = loss_fn(y_pre,y_true)
        # all_loss = 0
        for i in range(self.cascades):
            y_pre, y_true = self.MaxPool2d(y_pre),self.MaxPool2d(y_true)
            index = 2**(i+1)
            all_loss += loss_fn(y_pre,y_true)*(2**index)
        
        return all_loss


#%%
class IOUloss(torch.autograd.Function):
    '''
    func: 按照相关教程自定义梯度后传。
          在回归中使用IOU(TS)损失没办法进行梯度更新(由于离散化),因此需要自定义损失后传方法
          https://pytorch.org/docs/0.3.1/notes/extending.html?highlight=apply
    '''
    
    @staticmethod
    def forward(ctx,y_pre, y_true,threshold):
        '''
        func: 
        Parameter
        ---------
        ctx: 固定参数
        -----------
        input parameter:
        ----------------
        y_pre: 3D,4D or 5D Tensor
            predict by model.
        y_true: 3D,4D or 5D Tensor
            real value.
        threshold: float or tensor.float
            阈值,选取该阈值进行[0,1]二分类
        ---------------
        return
            IOU_loss,为 1 - IOU
        '''
        
        #Tensor化
        if not isinstance(threshold,torch.Tensor):
            threshold = torch.tensor(threshold)
        
        zero = torch.tensor(0).to(y_pre.device)
        one = torch.tensor(1).to(y_pre.device)
        
        #根据阈值二值化
        pre_binary = torch.where(y_pre >= threshold,one,zero)
        true_binary = torch.where(y_true >= threshold,one,zero)
        
        #计算IOU(TS)评分，注意需要转成float型
        hits = torch.sum(pre_binary*true_binary).float()
        IOU_Score = hits/(torch.sum(pre_binary) + torch.sum(true_binary) - hits)
        
        #将1-IOU_Score定位为IOU损失
        IOU_loss = 1 - IOU_Score
        
        #保存想应参数，传递到backward()中使用
        ctx.save_for_backward(y_pre,y_true,threshold, 
                              pre_binary, true_binary, IOU_loss) 

        return IOU_loss
        
    @staticmethod
    def backward(ctx,grad_output):
        '''
        func:
        Parameter
        ---------
        grad_output: 
            为反向传播上一级计算得到的梯度值
        ---------
        return:
            返回forward对应输入参数的梯度
        '''
        
        y_pre,y_true,threshold,pre_binary, true_binary, IOU_loss = ctx.saved_tensors
        
        #计算漏报区域和空报区域
        misses = (true_binary == 1) & (pre_binary == 0) 
        falsealarms = (true_binary == 0) & (pre_binary == 1)
        
        #在y_pre的空报区域正梯度后传(降低数值)，漏报区域负梯度后传（增大该区域数值）
        #乘以系数IOU_loss系数
        loss = falsealarms*y_pre - misses*y_pre
        loss = grad_output*IOU_loss*loss   
    
        return loss, None,None
         

def IOULoss(y_pre,y_true,threshold):
    return IOUloss.apply(y_pre,y_true,threshold)


#%%
class BEXPRMSELoss(torch.nn.Module):
    '''
    func: RMSE损失在空间中给强回波处的误差更大的权重
    Parameter
    '''
    def __init__(self):
        super(BEXPRMSELoss, self).__init__()
        
    def forward(self,y_pre, y_true):
        '''
        Parameter
        ---------
        y_pre: 4D or 5D Tensor
            predict by model.
        y_true: 4D or 5D Tensor
            real value.
        ---------
        '''    
        assert y_true.min() >= 0
        assert y_true.max() <= 1
        
        weight = y_true.clone()
        weight = torch.exp(weight*6) - 0.8
        
        loss = torch.sum(((y_pre - y_true)**2)*(weight) /len(y_true.view(-1,1)))
        
        return torch.sqrt(loss)
    
class BEXPMSELoss(torch.nn.Module):
    '''
    func: MSE损失在空间中给强回波处的误差更大的权重, 其中权重与真实回波值呈指数正比
    Parameter
    ---------
    a: float or int
        default 6, 即权重是真实回波的指数系数
    b: float
        default 0.8, 即weight = exp(y_true * a) - b
    '''
    def __init__(self, a = 6,b = 0.8):
        super(BEXPMSELoss, self).__init__()
        self.a = a
        self.b = b
        
    def forward(self,y_pre, y_true):
        '''
        Parameter
        ---------
        y_pre: 4D or 5D Tensor
            predict by model.
        y_true: 4D or 5D Tensor
            real value.
        ---------
        '''  
        #确保真实值的范围在 0-1之间
        assert y_true.min() >= 0
        assert y_true.max() <= 1
        
        weight = y_true.clone()
        weight = torch.exp(weight* self.a) - self.b
        
        loss = torch.mean(((y_pre - y_true)**2)*(weight))
        
        return loss
    
class BEXPMAELoss(torch.nn.Module):
    '''
    func: MAE损失在空间中给强回波处的误差更大的权重, 其中权重与真实回波值呈指数正比
    Parameter
    ---------
    a: float or int
        default 6, 即权重是真实回波的指数系数
    b: float
        default 0.8, 即weight = exp(y_true * a) - b
    '''
    def __init__(self, a = 6,b = 0.8):
        super(BEXPMAELoss, self).__init__()
        self.a = a
        self.b = b
        
    def forward(self,y_pre, y_true):
        '''
        Parameter
        ---------
        y_pre: 4D or 5D Tensor
            predict by model.
        y_true: 4D or 5D Tensor
            real value.
        ---------
        '''  
        #确保真实值的范围在 0-1之间
        assert y_true.min() >= 0
        assert y_true.max() <= 1
        
        weight = y_true.clone()
        weight = torch.exp(weight* self.a) - self.b
        
        loss = torch.mean(abs((y_pre - y_true))*(weight))
        
        return loss
    
#%%
class BEXPMSAELoss(torch.nn.Module):
    '''
    func: MSE + MAE损失在空间中给强回波处的误差更大的权重, 其中权重与真实回波值呈指数正比
          即将MSE + MAE结合使用
    Parameter
    ---------
    a: float or int
        default 6, 即权重是真实回波的指数系数
    b: float
        default 0.8, 即weight = exp(y_true * a) - b
    '''
    def __init__(self,
                 a = 6,b = 0.8,
                 mse_w = 1,mae_w = 1):
        super(BEXPMSAELoss, self).__init__()
        self.a = a
        self.b = b
        
        self.mse_w = mse_w
        self.mae_w = mae_w
        
    def forward(self,y_pre, y_true):
        '''
        Parameter
        ---------
        y_pre: 4D or 5D Tensor
            predict by model.
        y_true: 4D or 5D Tensor
            real value.
        ---------
        '''  
        #确保真实值的范围在 0-1之间
        assert y_true.min() >= 0
        assert y_true.max() <= 1
        
        weight = y_true.clone()
        weight = torch.exp(weight* self.a) - self.b
        
        loss_mse = torch.mean(((y_pre - y_true)**2)*(weight))
        loss_mae = torch.mean(abs((y_pre - y_true))*(weight))
        
        return loss_mse * self.mse_w + loss_mae * self.mae_w

#%%
# y_true = torch.randint(0,10,(1,2,40,40))/10.0    
# y_pre = torch.randint(0,10,(1,2,40,40))/10.0

# # loss_fn = BEXPMSELoss()
# # loss_fn = BEXPMAELoss()
# # loss_fn = BEXPMSAELoss()
# loss = loss_fn(y_pre, y_true)

# print(loss.max())
#%%
class BEXPRMSEIOULoss(torch.nn.Module):
    '''
    func: RMSELoss + IOU损失
    摘自：Deep Learning Prediction of Incoming Rainfalls: An Operational Service for the City of Beijing China
    Parameter
    ---------
    IOU_threshold: list 
        default [15,35],  获取哪些雷达回波阈值计算IOU   
    max_value: int
        default 80. 雷达回波范围最大值，一般为80
    
    '''
    def __init__(self,
                  IOU_threshold = [15,35],
                  max_value = 80):
        super(BEXPRMSEIOULoss, self).__init__()
        
        self.IOU_threshold = [threshold / max_value for threshold in IOU_threshold]
        
    def forward(self, y_pre, y_true):
        '''
        Parameter
        ---------
        y_pre: 4D or 5D Tensor
            predict by model.
        y_true: 4D or 5D Tensor
            real value.
        ---------
        '''
        
        loss_fn = BEXPRMSELoss()
        loss1 = loss_fn(y_pre, y_true)
        
        iou_loss = 0
        for threshold in self.IOU_threshold:
            loss = IOULoss(y_pre, y_true, threshold)
            iou_loss += loss * 0.005
        
        return loss1 + iou_loss
            
        

#%%
import torch.nn.functional as F
from math import exp
import numpy as np
 
# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
 
        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = self.create_window(window_size)
 
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
 
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
            # print(1)
        else:
            window = self.create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
            # print(2)
 
        return self.ssim(img1, img2, window= window, window_size=self.window_size, size_average=self.size_average)   

    # 计算一维的高斯分布向量
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    
    # 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
    # 可以设定channel参数拓展为3通道
    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window 
    
    
    def ssim(self, img1, img2, window_size=11, window=None, 
             size_average=True, full=False, val_range=None):
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1
     
            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = val_range
     
        padd = 0
        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = self.create_window(real_size, channel=channel).to(img1.device)
     
        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
     
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
     
        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
     
        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2
     
        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity
     
        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
     
        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)
     
        if full:
            return ret, cs
        return ret

# device = torch.device('cuda:0')
# x1 = torch.randn((2,10,100,100)).to(device)
# x2 = x1 + torch.randn((2,10,100,100)).to(device)

# ssim_1 = SSIM()
# loss = 1 - ssim_1(x1,x1 + 0.001)
# print(loss)

#%%
class BMSAE_SSIM_Loss(torch.nn.Module):
    '''
    func: MSE和MAE损失中给强回波处的误差更大的权重，同时加入结构相似度损失1-ssim, 三者按照一定权重结合起来
    Parameter
    ---------
    weights: list
        default [1,2,5,10,30].权重列表,给不同的回波强度处对应的像素点的损失不同的权重
    thresholds: list
        阈值列表，即将回波强度按照范围分为若干段，不同段给与不同的损失权重
        default [20,30,40,50,80].对应0~1之间的输入为: [0.25, 0.375, 0.5, 0.625, 1.0]
    mse_w: float
        mse权重, default 1
    mae_w: float
        mae权重, default 1
    ssim_w: float
        ssim损失的权重，default 1
    '''
    def __init__(self, weights = [1,2,5,10,30], 
                 thresholds = [20,30,40,50,80],
                 mse_w = 1,mae_w = 1, ssim_w = 1):
        super(BMSAE_SSIM_Loss,self).__init__()
        
        assert len(weights) == len(thresholds)
        scale = max(thresholds)
        self.weights = weights
        self.thresholds = [threshold/scale for threshold in thresholds] 
        #[0.25, 0.375, 0.5, 0.625, 1.0]
        self.mse_w = mse_w
        self.mae_w = mae_w
        self.ssim_w = ssim_w
        
    def forward(self,y_pre,y_true):
        
        assert y_true.min() >= 0
        assert y_true.max() <= 1
        
        w_true = y_true.clone()
        for i in range(len(self.weights)):
            w_true[w_true < self.thresholds[i]] = self.weights[i] #获取权重矩阵
            
        mse_loss = torch.mean(w_true * (y_pre - y_true)**2)
        mae_loss = torch.mean(w_true * (abs(y_pre - y_true)))
        
        ssim = SSIM()
        ssim_loss = 1 - ssim(y_pre, y_true)
            
        return  self.mse_w *mse_loss + self.mae_w * mae_loss + self.ssim_w * ssim_loss

#%%
class BEXPMSAE_SSIM_Loss(torch.nn.Module):
    '''
    func: MSE + MAE损失在空间中给强回波处的误差更大的权重, 其中权重与真实回波值呈指数正比, 同时加上 1 - SSIM损失
    Parameter
    ---------
    a: float or int
        default 6, 即权重是真实回波的指数系数
    b: float
        default 0.8, 即weight = exp(y_true * a) - b
    mse_w: float
        mse权重, default 1
    mae_w: float
        mae权重, default 1
    ssim_w: float
        ssim损失的权重，default 1
    '''
    def __init__(self,
                 a = 6,b = 0.8,
                 mse_w = 1,mae_w = 1, ssim_w = 1):
        super(BEXPMSAE_SSIM_Loss, self).__init__()
        self.a = a
        self.b = b
        
        self.mse_w = mse_w
        self.mae_w = mae_w
        self.ssim_w = ssim_w
        
    def forward(self,y_pre, y_true):
        '''
        Parameter
        ---------
        y_pre: 4D or 5D Tensor
            predict by model.
        y_true: 4D or 5D Tensor
            real value.
        ---------
        '''  
        #确保真实值的范围在 0-1之间
        assert y_true.min() >= 0
        assert y_true.max() <= 1
        
        weight = y_true.clone()
        weight = torch.exp(weight* self.a) - self.b
        
        loss_mse = torch.mean(((y_pre - y_true)**2)*(weight))
        loss_mae = torch.mean(abs((y_pre - y_true))*(weight))
        
        ssim = SSIM()
        ssim_loss = 1 - ssim(y_pre, y_true)
        
        return loss_mse * self.mse_w + loss_mae * self.mae_w + self.ssim_w * ssim_loss



#%%

# y_true = torch.randint(0,10,(1,2,40,40))/10.0    
# y_pre = torch.randint(0,10,(1,2,40,40))/10.0
# # loss_fn = BEXPMSELoss()
# # loss_fn = BEXPMAELoss()
# # loss_fn = BEXPMSAELoss()
# # loss_fn = BMSAE_SSIM_Loss()
# # loss_fn = BEXPMSAE_SSIM_Loss()
# loss = loss_fn(y_pre, y_true)

# print(loss.max())













