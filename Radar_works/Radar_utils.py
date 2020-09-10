# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:28:52 2020
'''
一些常常用到的函数集合:
Part1: dbz --- 反射率 --- 降水量 之间的转化;
Part2: 图片读取，画图, 图片--> 视频。 
        image_read_cv(); 读取单张图片
        plot_img();
        compare_obs_pre_img();
        compose2gif(); 将图片组成成 .gif文件
        compose2gif_path() ;将图片组成 .gif文件，输入图片列表所在路径即可
        compose_photo2video(); 
        get_sample_data();

Part3: 评分函数, 衡量两帧雷达图的差异
       get_hits()
       HSS()  即HSS评分
       MSE()
       MAE()
       plot_scores()

Part4: 雷达样本质量检测；
        check_radar_quality(); get_error_photo_index()
        
Part4: EarlyStopping()
Part5: 损失函数
        BMSE()
        BMAE()
        BMSAE() , 即BMSE + BMAE 的加权和
Part6: 气象上常用的评分函数:
       prep_clf() 计算指定阈值下的 hits/ misses/falsealarms等
       precision() ; recall() ;ACC() ; FSC(),即F1; TS() ;
       MAR() 即漏报率
       FAR() 即空报率
       POD()  命中率
       BIAS() 偏差评分
       HSS_one_threshold() 单个阈值下二分类的HSS评分
       multil_scores() 
       plot_multi_scores() 即画出不同阈值情况下的 模型预测对比实况的 Acc/Recall/Precision/F1/TS/MAR/FAR 的评分情况


Part7: 
       get_circle_mask(), 获取正方形雷达图上的圆盘mask
       
'''
@author: fzl
"""

import numpy as np
import numpy.ma as ma
import os
import matplotlib.pyplot as plt
import sklearn 
import cv2
import torch
import matplotlib.animation as animation
import matplotlib
# import imageio

#%%
#part1: dbz --- 反射率 --- 降水量 之间的转化
def dbz2rfl(d):
    '''
    func: dBZ ---> 反射率rf(Z)
    based on wradlib.trafo.idecibel function   
    '''
    return 10. ** (d / 10.)

def rfl2dbz(z):
    '''
    func: 反射率rf(Z) ---> dBZ
    based on wradlib.trafo.decibel function
    '''
    return 10. * np.log10(z)


def rfl2mmh(z, a=256., b=1.42):
    '''
    func: 反射率 ---> 小时雨强intensity(mm/h)
    based on wradlib.zr.z2r function
    '''
    return (z / a) ** (1. / b)


def mmh2rfl(r, a=256., b=1.42):
    '''
    func: 小时雨强intensity(mm/h) ---> 反射rf(Z)
    based on wradlib.zr.r2z function

    .. r --> z
    '''
    return a * r ** b

def intensity2depth(intensity, interval = 360):
    """
    func: 小时雨强度intensity(mm/h) --->  6分钟间隔降水量(mm)
    Function for convertion rainfall intensity (mm/h) to
    rainfall depth (in mm)

    Args:
        intensity: float
        float or array of float
        rainfall intensity (mm/h)

        interval : number
        time interval (in sec) which is correspondend to depth values

    Returns:
        depth: float
        float or array of float
        rainfall depth (mm)
    """
    return intensity * interval / 3600


def depth2intensity(depth, interval = 360):
    
    """
    func: 6分钟间隔降水量(mm) ---> 小时雨强度intensity(mm/h)
    Function for convertion rainfall depth (in mm) to
    rainfall intensity (mm/h)

    Args:
        depth: float
        float or array of float
        rainfall depth (mm)

        interval : number
        time interval (in sec) which is correspondend to depth values

    Returns:
        intensity: float
        float or array of float
        rainfall intensity (mm/h)
    """
    return depth * 3600 / interval

def mm2dBZ(X_mm, interval = 360):
    '''
    func: 6分钟间隔降水量(mm)  ---> dBZ
    '''
    # mm to mm/h
    X_mmh = depth2intensity(X_mm, interval = interval)
    # mm/h to reflectivity
    X_rfl = mmh2rfl(X_mmh)
    # remove zero reflectivity
    # then log10(0.1) = -1 not inf (numpy warning arised)
    X_rfl[X_rfl == 0] = 0.1
    # reflectivity to dBz
    X_dbz = rfl2dbz(X_rfl)
    # remove all -inf
    X_dbz[X_dbz < 0] = 0
    
    return X_dbz

def dbZ2mm(d, interval = 360):
    '''
    func:  dBZ  ---> 6分钟间隔降水量(mm)

    '''
    # decibels to reflectivity
    X_rfl = dbz2rfl(d)
    # 0 dBz are 0 reflectivity, not 1
    X_rfl[X_rfl == 1] = 0
    # reflectivity to rainfall in mm/h
    X_mmh = rfl2mmh(X_rfl)
    # intensity in mm/h to depth in mm
    X_mm = intensity2depth(X_mmh, interval = interval)

    return X_mm

#%%
#Part2: 读取图片、画图等
def image_read_cv(file_path, dtype = 'uint8'):
    
    #默认读取结果为: uint8
    image = cv2.imread(file_path,0)
    
    if dtype != 'unit8':
        
        image = np.array(image, dtype = np.float32)
    
    return image


def plot_img(img, title = None):
    '''
    func: 画一帧图
    inputs: 
        img: 图片对应的二维数组， shape = (width,height)
        title: 图片标题
    '''
    fig,ax = plt.subplots(1,1)
        
    h = ax.imshow(img)
    fig.colorbar(h)
    if title:
        ax.set_title(title)
    plt.show()
    
    return None

def compare_obs_pre_img(obs, pre, index = 0, is_save_file = False, save_path = None):
    '''
    func: 画图比较obs 和 pre
    inputs:
        obs: 观测真值
        pre: 预测值
        index: 表示第几个时刻，默认0,当作图片的 title
        is_save_file: 是否保存图片路径，默认False
        save_path: 图片保存路径 + 文件名
    
    '''
    
    #设置colorbar范围一致
    vmin = min(obs.min(),pre.min())
    vmax = max(obs.max(),pre.max())
    norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax)
    
    fig, axs = plt.subplots(1,2,figsize = (16,10))
    h0 = axs[0].imshow(obs, norm = norm)
    fig.colorbar(h0, ax = axs[0], shrink = 0.6)
    axs[0].set_title(str(index) + ':obs')
    
    h1 = axs[1].imshow(pre,norm = norm)
    fig.colorbar(h1, ax = axs[1], shrink = 0.6)
    axs[1].set_title(str(index) + ':pre')
    
    if is_save_file:
        
        fig.savefig(save_path, 
                    # dpi = 200,
                    bbox_inches = 'tight')
    
    # plt.show()
    
    return fig

#将图片组成成 .gif文件
def compose2gif(img_filelist, save_filepath = None, fps = 1):
    '''
    func: 将图片组成成 .gif文件
    Parameter
    --------
    img_filelist: list of str
        图片文件路径+文件名组成的 list
    save_filepath: str
        生成的.gif文件的保存路径+文件名
    fps: float
        帧率，即每秒闪过几帧图片，默认1
    Return
    '''

    gif_images = []
    for file in img_filelist:
        gif_images.append(imageio.imread(file))
        
    imageio.mimsave(save_filepath,gif_images, fps = 2)
    
    return None

def compose2gif_path(img_path,save_filepath = None, img_type = '.png', fps = 1):
    '''
    func: 输入图片列表的路径,将图片列表组成成 .gif文件
    Parameter
    --------
    img_path: list of str
        图片所在路径, eg: F:/caiyun/img_path
    img_type: str
        图片格式, 默认'.png'
    save_filepath: str
        生成的.gif文件的保存路径+文件名
    fps: float
        帧率，即每秒闪过几帧图片，默认2
    Return
    '''
    
    img_filelist = os.listdir(img_path)    
    
    img_filelist = [file for file in img_filelist if img_type in file] #保留所有 '.png' 文件

    #保证文件名格式为：index.png， 其中int(index) = 整数 
    img_filelist.sort(key=lambda x: int(x.split('.')[0]))  
    img_filelist = [os.path.join(img_path,file) for file in img_filelist]
    
    # print(img_filelist)
    
    gif_images = []
    for file in img_filelist:
        gif_images.append(imageio.imread(file))
        
    imageio.mimsave(save_filepath,gif_images, fps = 2)
    
    return None


##读取零散图片(上面分解的图片)，并将其合成视频
def compose_photo2video(filepath, save_path_name, 
                  video_type = '.mp4', 
                  nframes_per_sec = 1):
    '''
    func: 将某个文件夹内的图片组成成为视频
    input:
        filepath: 存储系列图片的文件夹名称，一般为绝对路径
                    且该系列图片width,height应该一致
        save_path_name: 转化成视频后的存储位置及其文件名,eg: D:/video/pre
        video_type: 视频类型，default '.mp4' , 可选: '.mp4','.avi'
        nframes_per_sec: 每秒多少帧, 默认1
        
    return:
        None
    '''

    os.chdir(filepath)
    file_list = os.listdir()
    file0 = file_list[0]
    
    img = cv2.imread(file0)
    imginfo = img.shape
    
    #与默认不同，opencv使用 height在前，width在后，所以需要自己重新排序
    size = (imginfo[1],imginfo[0])  
    print(size)
    
    
    #创建写入对象，包括 新建视频名称，每秒钟多少帧图片(10张) ,size大小
    #一般人眼最低分辨率为19帧/秒
    videoWrite = cv2.VideoWriter(save_path_name + video_type,-1,
                                 nframes_per_sec,size) 
    
    for file in file_list:
    
        img = cv2.imread(file,1)  #1 表示彩图，0表示灰度图  
        
        #直接写入图片对应的数据
        videoWrite.write(img)  
    
    videoWrite.release() #关闭写入对象
    print('end')

    return None


def get_sample_data(abs_path, channel_first = True):
    '''
    file: 获取每个雷达样本文件中的41张图片
    inputs:
        abs_path: 输入文件名,绝对路径
        channel_first: 把通道数放首位，默认True
        eg: D:\Train_data\RAD_185497421184001
        
    return:
        samples, shape: (41,256, 256) or (256, 256, 41)
        
    '''

    samples = []
    
    file_list = os.listdir(abs_path) #eg:RAD_185497421184001_000.png  
    
    radar_name = file_list[0].split('.')[0][0:-3] #eg: RAD_185497421184001_
    #默认按照时间顺序存储的，以防万一，使用显示文件名排序
    for i in range(len(file_list)): 
        
        #eg: '012'
        radar_time = '0' + str(i)  if i > 9 else '00' + str(i)  
        #获取文件名称
        abs_file = os.path.join(abs_path ,radar_name + radar_time + '.png')
        img = image_read_cv(abs_file)
        img = np.expand_dims(img, axis = 0)
        samples.append(img)
        i = i + 1
    
    samples = np.concatenate(samples, axis = 0)
    
    if not channel_first:
        samples = np.transpose(samples,[1,2,0])
        
    return samples


def Scale_Predict(src_path, dst_path):
    '''
    func: 对预测的4张图片进行scale操作
    inputs:
        src_path: predict样本所在路径，eg: D:/Predict/RAD_185099420483101
        dst_path: 改变原始样本数值大小后，样本保存的位置,
                eg: eg: D:/Scale_Predict/RAD_185099420483101
    '''

    file_list = ['30.png','60.png','90.png','120.png']
    
    for file in file_list:
        
        src_filename = os.path.join(src_path, file)
        dst_filename = os.path.join(dst_path, file)
        
        src_img = cv2.imread(src_filename, 0) #单通道读取
        
        if file == '30.png':
            max0 = src_img.max() #获取第30分钟时刻图片的最大数值
            dst_img = src_img
            
        else:
            #获取后续每张图片的最大值, 如果max1远小于max0,则进行尺寸变大。
            max1 = src_img.max()  
            if max1 <= (max0 - 2): 
                scale = max0/max1
                dst_img = src_img * scale
            else:
                dst_img = src_img
        
                
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
            
        cv2.imwrite(dst_filename,dst_img)
           
    return None

def Scale_Predict_1(max0, src_path, dst_path):
    '''
    func:
    inputs:
        max0: TEST样本中的最后3张图片的回波最大值的平均
        src_path: predict样本所在路径，eg: D:/Predict/RAD_185099420483101
        dst_path: 改变原始样本数值大小后，样本保存的位置,
                eg: eg: D:/Scale_Predict/RAD_185099420483101
    '''

    file_list = ['30.png','60.png','90.png','120.png']
    
    for file in file_list:
        
        src_filename = os.path.join(src_path, file)
        dst_filename = os.path.join(dst_path, file)
        
        src_img = cv2.imread(src_filename, 0) #单通道读取
        
        max1 = src_img.max()  
        if max1 <= (max0 - 2): 
            scale = max0/max1
            dst_img = src_img * scale
        else:
            dst_img = src_img
        
                
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
            
        cv2.imwrite(dst_filename,dst_img)
           
    return None

def create_gif(gif_name, path, duration = 0.3):
    '''
    生成gif文件，原始图片仅支持png格式
    gif_name ： 字符串，所生成的 gif 文件名，带 .gif 后缀
    path :      需要合成为 gif 的图片所在路径
    duration :  gif 图像时间间隔
    '''

    frames = []
    pngFiles = os.listdir(path)
    image_list = [os.path.join(path, f) for f in pngFiles]
    for image_name in image_list:
        # 读取 png 图像文件
        frames.append(imageio.imread(image_name))
    # 保存为 gif 
    imageio.mimsave(gif_name, frames, 'GIF', duration = duration)

    return


def cv_morphologyEx(img,erode_kernel_size = 10, dilate_kernel_size = 5):
    '''
    func: 形态学开运算，即先腐蚀，后扩张。对消除离散杂波和边缘杂波很有效
    Parameters:
        img: 原始的图像
        erode_kernel_size: 腐蚀时使用的kernel_size， 默认是10
        dilate_kernel_size: 膨胀时使用的kernel_size， 默认是5
    '''
    import cv2 as cv
    
    # step1: 先获取图片的mask
    img_mask = np.where(img > 0, 1, 0)
    img_mask = np.array(img_mask,dtype = float)
    
    #对mask进行腐蚀
    kernel = np.ones((erode_kernel_size,erode_kernel_size),np.uint8)
    img1_mask = cv.erode(img_mask,kernel,iterations = 1)
    
    #对腐蚀后的mask进行膨胀
    kernel = np.ones((dilate_kernel_size,dilate_kernel_size),np.uint8)
    img2_mask = cv.dilate(img1_mask,kernel,iterations = 1)
    
    img2 = img2_mask * img
    
    return img2


#%%
#Part3:评分函数
#HSS 评分函数
def get_hits(obs, pre, thresholds = [0,20], fill_value = 255):
    
    '''
    func: 计算预测pre 和 观测obs 有多少格点在thresholds范围内，和两者共同有多少在该范围内
          get_hits(obs,pre, thresholds = [1,10], fill_value = 255) 
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        thresholds: [left_threshold, right_threshold],阈值范围
        fill_value: 雷达图上无效值默认用255表示，该值位置用mask表示，不参与后续运算
    
    returns:
        hits: obs 和 pre 都在thresholds范围内的格点数
        obs_1: obs在thresholds范围内的格点数
        pre_1：pre在thresholds范围内的格点数 
    '''
    
    #将obs 和 pre中的缺测值mask掉，不参与评分计算
    obs = ma.array(obs, mask = obs == fill_value)
    pre = ma.array(pre, mask = pre == fill_value)
    
    #获取左右阈值
    left_threshold = thresholds[0]
    right_threshold = thresholds[1]
    
    #[0,20]为闭区间，之后的都是(] 开闭区间
    if left_threshold == 0: 

        #根据阈值分类为 0, 1
        obs = np.where( (obs >= left_threshold) & (obs <= right_threshold) , 1, 0)
        pre = np.where( (pre >= left_threshold) & (pre <= right_threshold) , 1, 0)
    
    else:
        
        #根据阈值分类为 0, 1
        obs = np.where( (obs > left_threshold) & (obs <= right_threshold) , 1, 0)
        pre = np.where( (pre > left_threshold) & (pre <= right_threshold) , 1, 0)
    
    
    # True positive (TP)
    #预测类别为1的像素点中真实类别为1的像素点总数, 用 hits表示
    hits = np.sum((obs == 1) & (pre == 1))
    
    #N(Fi)表示预测为类别i的像素点总数 , 用pre_1表示
    #N(Oj)表示真实类别为j的像素点总数,  用obs_1表示  
    pre_1 = np.sum(pre == 1)
    obs_1 = np.sum(obs == 1)
    
    #将整数转为 float32类型，避免后续obs_1 和 pre_1大数相乘出现负值
    obs_1 = np.float64(obs_1)
    pre_1 = np.float64(pre_1)

    return hits, obs_1, pre_1


def HSS(obs,pre,class_nums = 5, weights = [1,1,1,1,1]):
    '''
    func: 计算obs 和 pre的多类别带权重HSS评分;
          HSS(obs,pre, class_nums = 5, weights = [1,1,1,1,1])
    
    inputs: 
        obs: 观测值，即真实值；
        pre: 预测值
        class_nums: 类别数目，为5
        weights: 每个类别的权重，为list列表。默认[1,1,1,1,1]
                只有权重都为1时，输入obs,pre为同一幅图片，HSS才为1.
            
    return:
        HSS评分
        
    '''
    #判定类别数目 和 权重个数一致
    assert class_nums == len(weights)
    
    #各个类别的阈值范围
    thresholds_list = [[0,20],
                       [20,30],
                       [30,40],
                       [40,50],
                       [50,80]]
    
    #总格点数
    N = len(obs.ravel())
    
    #两个分子，一个分母
    num1 = np.float64(0)
    num2 = np.float64(0)
    
    for weight, thresholds in zip(weights, thresholds_list):
        
        hits, obs_1, pre_1 = get_hits(obs, pre, thresholds = thresholds)
         
        # print('hits {}, obs_1 {} , pre_1 {}'.format(hits, obs_1,pre_1))
        
        num1 = num1 + hits*weight
        num2 = num2 + obs_1*pre_1*weight

    num1 = num1/N
    num2 = num2/N/N
    den = 1 - num2
    
    #如果分母为0, 即obs 和 pre的元素都只在一个类别里，返回1. 
    # print('num1',num1)
    # print('num2',num2)
    # print('den',den)
    
    #如果分母小于 10-3次方，则说明两幅图已经基本一致，返回1.
    if abs(den) <= 1e-3:
        return 1
    
    #如果分子为0，则返回1
    elif num1 == num2:  
        return 1
    else:
        return (num1 - num2)/den
    
    
def MAE(obs,pre,weight = 1):
    ''' 
    func: 计算obs 和 pre之间的 MAE
    '''
    
    return np.mean(abs(obs - pre)) * weight

def MSE(obs,pre,weight = 1):
    ''' 
    func: 计算obs 和 pre之间的 MSE
    '''
    return np.mean(np.square(obs - pre)) * weight    


def plot_scores(scores,title = None):
    '''
    func: 画出多帧评分变化
    inputs:
        scores: 多个时刻的pre 和 obs 的评分，为list
        title: 图title, 默认为：HSS
    '''
    fig = plt.figure(figsize = (10,6))
    
    x_ticks = np.arange(len(scores))
    
    plt.plot(x_ticks,scores,'-r',linewidth = 2)
     
    plt.xticks(x_ticks,x_ticks, fontsize = 16)
    plt.xlabel('T',fontsize = 20)
    
    plt.yticks(fontsize = 16)
    plt.ylabel('score',fontsize = 20)
    
    if title:
        if title == 'HSS':    
            plt.ylim(0,1)
        plt.title(title,fontsize = 20)
        
    plt.show()
    return None

#%%
#Part4: 雷达数据质量检测
def check_radar_quality(left, middle, right, 
                       score_threshold = 0.35,
                       num_threshold = 4000 ):
    
    '''
    func: 对比middle图与左右两张图，确定middle图有缺陷
    inputs:
        left: 时序中前一帧图(t-1)，二维数组
        middle: t时刻的radar图，二维数组
        right: 时序中后一帧图(t+1)，二维数组
        score_threshold: HSS评分低于score_threshold 表示可能存在缺陷
        num_threshold: 某个时刻雷达数组中大于0的元素个数 vs num_threshold 
    return:
        quality: 布尔类型，False/True ，表示质量好坏，True表示存在质量问题.
    
    '''
    
    #确定雷达数组中有效元素个数
    left_nums = np.sum(left > 0)
    middle_nums = np.sum(middle > 0)
    right_nums = np.sum(right > 0)
    
    ##确定middle与左右图像的HSS评分
    left_score = HSS(middle,left, weights = [1,1,1,1,1])
    right_score = HSS(middle,right, weights = [1,1,1,1,1])
    # print('left score: {}, right score: {}'.format(left_score,right_score))
    
    flag = False
    
    if (left_score <= score_threshold)&(right_score <= score_threshold):
    
        #如果左右图片像素点> 0 的数量都大于阈值，
        #则说明HSS低 不是由于middle图片有效像素太少造成, middle存在缺陷
        if (left_nums > num_threshold)&(right_nums > num_threshold):
            flag = True
    
        #而如果middle有效像素多，左右有效像素少，
        #则说明middle上有许多杂波            
        if middle_nums > num_threshold:
            if (left_nums < num_threshold)&(right_nums < num_threshold):
                flag = True
    
    
    if flag:
        #如果存在缺陷，则画出该图
        print('left score: {}, right score: {}'.format(left_score,right_score))
        
        fig, axes = plt.subplots(1,3, figsize = (15,5))
        
        for i, img in enumerate([left, middle, right]):
            ax = axes[i]
            h = ax.imshow(img)
            fig.colorbar(h,ax = ax, shrink = 0.7)
            
        plt.show()
            
    return flag

def get_error_photo_index(abs_path): 
    '''
    file: 获取每个雷达样本文件中 雷达图质量不佳的图对应的index
    inputs:
        abs_path: 输入文件名,绝对路径
        eg: D:\Train_data\RAD_185497421184001
        
    return:
        error_photo_index: dtype为list, 质量存在缺陷的雷达序列号
    '''
    
    #stpe1: 先读取该文件夹下的所有img, 添加到X中
    # X = []
    # file_list = os.listdir(abs_path) #eg:RAD_185497421184001_000.png  
    
    # radar_name = file_list[0].split('.')[0][0:-3] #eg: RAD_185497421184001_
    # #默认按照时间顺序存储的，以防万一，使用显示文件名排序
    # for i in range(len(file_list)): 
    
    #     #eg: '012'
    #     radar_time = '0' + str(i)  if i > 9 else '00' + str(i)  
    #     #获取文件名称
    #     abs_file = os.path.join(abs_path ,radar_name + radar_time + '.png')
    #     X.append(image_read_cv(abs_file))
    #     i = i + 1
        # print('radar_time:',radar_time)
             
    X = get_sample_data(abs_path, channel_first = True)
    
    n_frams = X.shape[0] 
        
    #step2: 判断异常雷达图像
    error_photo_index = []
    for i in range(0,n_frams):

        # print(i)
        # print()
        
        #检查第0帧, 与1,2帧比较
        if i == 0:
            flag = check_radar_quality(X[i+1,:,:],X[i,:,:],X[i+2,:,:])
        
        #检查最后一帧，与T-1,T-2帧比较
        elif i == n_frams-1:
            flag = check_radar_quality(X[i-1,:,:],X[i,:,:],X[i-2,:,:])
        
        else:
            #有缺陷的雷达图的index
            flag = check_radar_quality(X[i-1,:,:],X[i,:,:],X[i+1,:,:])
            
        if flag:
            error_photo_index.append(i)
            print(i)
            
    return error_photo_index

#%%
# import numpy as np
# import torch

# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, patience=7, verbose=False, delta=0):
#         """
#         Args:
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 7
#             verbose (bool): If True, prints a message for each validation loss improvement. 
#                             Default: False
#             delta (float): Minimum change in the monitored quantity to qualify as an improvement.
#                             Default: 0
#         """
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta

#     def __call__(self, val_loss, model):

#         score = -val_loss

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#             self.counter = 0

#     def save_checkpoint(self, val_loss, model):
#         '''Saves model when validation loss decrease.'''
#         if self.verbose:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         torch.save(model.state_dict(), 'checkpoint.pt')	# 这里会存储迄今最优模型的参数
#         self.val_loss_min = val_loss

#参考链接： https://github.com/Bjarten/early-stopping-pytorch
         # https://blog.csdn.net/qq_37430422/article/details/103638681

#%%
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0


#%%
#
class BMSELoss(torch.nn.Module):

    def __init__(self):
        super(BMSELoss, self).__init__()
        self.w_l = [1, 2, 5, 10, 30]
        self.y_l = [0.283, 0.353, 0.424, 0.565, 1]

    def forward(self, y_true, y_pre):
        w = y_pre.clone()
        for i in range(len(self.w_l)):
            w[w < self.y_l[i]] = self.w_l[i]
        return torch.mean(w * ((y_pre - y_true)** 2) )

    
class BMAELoss(torch.nn.Module):

    def __init__(self):
        super(BMAELoss,self).__init__()
        
        self.w_l = [1,2,5,10,30]
        self.y_l = [0.283,0.353,0.424,0.565,1]
        
    def forward(self,y_true, y_pre):
        w = y_pre.clone()
        for i in range(len(self.w_l)):
            w[w < self.y_l[i]] = self.w_l[i]
        return torch.mean(w * (abs(y_pre - y_true)))
    
    
class BMSAELoss(torch.nn.Module):
    '''
    func: BMSE + BMAE 损失权重相加
    input:
        w1: BMSE的权重, 默认1
        w2: BMAE的权重, 默认1
    '''
    def __init__(self, w1 = 1, w2 = 1):
        super(BMSAELoss,self).__init__()
        
        self.w1 = w1
        self.w2 = w2
        self.w_l = [1,2,5,10,30]
        self.y_l = [0.283,0.353,0.424,0.565,1]
        
    def forward(self,y_true, y_pre):
        w = y_pre.clone()
        for i in range(len(self.w_l)):
            w[w < self.y_l[i]] = self.w_l[i]
            
        loss = self.w1 * torch.mean(w * ((y_pre - y_true)** 2)) + self.w2 * torch.mean(w * (abs(y_pre - y_true))) 
        return loss


#%%
'''
from torch import nn
import torch
from nowcasting.config import cfg
from nowcasting.utils import rainfall_to_pixel, dBZ_to_pixel
import torch.nn.functional as F

class Weighted_mse_mae(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0, NORMAL_LOSS_GLOBAL_SCALE=0.00005, LAMBDA=None):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self._lambda = LAMBDA

    def forward(self, input, target, mask):
        balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS
        weights = torch.ones_like(input) * balancing_weights[0]
        thresholds = [rainfall_to_pixel(ele) for ele in cfg.HKO.EVALUATION.THRESHOLDS]
        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (target >= threshold).float()
        weights = weights * mask.float()
        # input: S*B*1*H*W
        # error: S*B
        mse = torch.sum(weights * ((input-target)**2), (2, 3, 4))
        mae = torch.sum(weights * (torch.abs((input-target))), (2, 3, 4))
        if self._lambda is not None:
            S, B = mse.size()
            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w.to(mse.get_device())
            mse = (w * mse.permute(1, 0)).permute(1, 0)
            mae = (w * mae.permute(1, 0)).permute(1, 0)
        return self.NORMAL_LOSS_GLOBAL_SCALE * (self.mse_weight*torch.mean(mse) + self.mae_weight*torch.mean(mae))


class WeightedCrossEntropyLoss(nn.Module):

    # weight should be a 1D Tensor assigning weight to each of the classes.
    def __init__(self, thresholds, weight=None, LAMBDA=None):
        super().__init__()
        # 每个类别的权重，使用原文章的权重。
        self._weight = weight
        # 每一帧 Loss 递进参数
        self._lambda = LAMBDA
        # thresholds: 雷达反射率
        self._thresholds = thresholds

    # input: output prob, S*B*C*H*W
    # target: S*B*1*H*W, original data, range [0, 1]
    # mask: S*B*1*H*W
    def forward(self, input, target, mask):
        assert input.size(0) == cfg.HKO.BENCHMARK.OUT_LEN
        # F.cross_entropy should be B*C*S*H*W
        input = input.permute((1, 2, 0, 3, 4))
        # B*S*H*W
        target = target.permute((1, 2, 0, 3, 4)).squeeze(1)
        class_index = torch.zeros_like(target).long()
        thresholds = [0.0] + rainfall_to_pixel(self._thresholds).tolist()
        # print(thresholds)
        for i, threshold in enumerate(thresholds):
            class_index[target >= threshold] = i
        error = F.cross_entropy(input, class_index, self._weight, reduction='none')
        if self._lambda is not None:
            B, S, H, W = error.size()

            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w.to(error.get_device())
                # B, H, W, S
            error = (w * error.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # S*B*1*H*W
        error = error.permute(1, 0, 2, 3).unsqueeze(2)
        return torch.mean(error*mask.float())

'''


#%%
#判定EC和SMS的累计6小时降水资料的准确率 + recall + precision + accuracy + TS评分 + POD + MAR + FAR
#下面为给出的一些评分函数
def prep_clf(obs,pre, threshold=0.1):
    '''
    func: 计算二分类结果-混淆矩阵的四个元素
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    
    returns:
        hits, misses, falsealarms, correctnegatives
        #aliases: TP, FN, FP, TN 
    '''
    #根据阈值分类为 0, 1
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))
    hits = np.float32(hits)
    misses = np.float32(misses)
    falsealarms = np.float32(falsealarms)
    correctnegatives = np.float32(correctnegatives)

    return hits, misses, falsealarms, correctnegatives


def precision(obs, pre, threshold=0.1):
    '''
    func: 计算精确度precision: TP / (TP + FP)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    
    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre = pre, threshold=threshold)

    return TP / (TP + FP)


def recall(obs, pre, threshold=0.1):
    '''
    func: 计算召回率recall: TP / (TP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    
    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre = pre, threshold=threshold)

    return TP / (TP + FN)


def ACC(obs, pre, threshold=0.1):
    '''
    func: 计算准确度Accuracy: (TP + TN) / (TP + TN + FP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    
    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre = pre, threshold=threshold)

    return (TP + TN) / (TP + TN + FP + FN)

def FSC(obs, pre, threshold=0.1):
    '''
    func:计算f1 score = 2 * ((precision * recall) / (precision + recall))
    '''
    precision_socre = precision(obs, pre, threshold=threshold)
    recall_score = recall(obs, pre, threshold=threshold)

    return 2 * ((precision_socre * recall_score) / (precision_socre + recall_score))


def TS(obs, pre, threshold=0.1):
    
    '''
    func: 计算TS评分: TS = hits/(hits + falsealarms + misses) 
    	  alias: TP/(TP+FP+FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre, threshold=threshold)
    
    
    return hits/(hits + falsealarms + misses) 

def POD(obs, pre, threshold=0.1):
    '''
    func : 计算命中率 hits / (hits + misses)
    pod - Probability of Detection
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: PDO value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return hits / (hits + misses)

def BIAS(obs, pre, threshold = 0.1):
    '''
    func: 计算Bias评分: Bias =  (hits + falsealarms)/(hits + misses) 
    	  alias: (TP + FP)/(TP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''    
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return (hits + falsealarms) / (hits + misses)


def MAR(obs, pre, threshold=0.1):
    '''
    func : 计算漏报率 misses / (hits + misses)
    MAR - Missing Alarm Rate
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: MAR value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return misses / (hits + misses)

def FAR(obs, pre, threshold=0.1):
    '''
    func: 计算误警率。falsealarms / (hits + falsealarms) 
    FAR - false alarm rate
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: FAR value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return falsealarms / (hits + falsealarms)


def HSS_one_threshold(obs, pre, threshold=0.1):
    '''
    HSS - Heidke skill score
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): pre
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: HSS value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    HSS_num = 2 * (hits * correctnegatives - misses * falsealarms)
    HSS_den = (misses**2 + falsealarms**2 + 2*hits*correctnegatives +
               (misses + falsealarms)*(hits + correctnegatives))

    return HSS_num / HSS_den


def multil_scores(obs, pre, threshold):
    '''
    func: 输入某个阈值下，输出该阈值下预报相对观测的评分（以直接打印的方式） 
    '''
    print('Threshold:', threshold)
    print('Acc:',ACC(obs,pre,threshold = threshold))
    print('Recall score:',recall(obs,pre,threshold = threshold))
    print('Precision score:',precision(obs,pre,threshold = threshold))
    print('F1 score:',FSC(obs,pre,threshold = threshold))
    print('TS评分:',TS(obs,pre,threshold = threshold))
    print('漏报率(MAR)评分:',MAR(obs,pre,threshold = threshold))
    print('误报率(FAR)评分:',FAR(obs,pre,threshold = threshold))
    

def plot_multi_scores(obs, pre, thresholds = [0.1, 5,10,15,20,25,30,35,40], title = None):

    '''
    画出不同阈值情况下的 模型预测对比实况的 Acc/Recall/Precision/F1/TS/MAR/FAR 的评分情况
    Parameters
    ----------
    obs : np.array
        数组
    pre : np.array
        数组.
    thresholds : list
        The default is [0.1, 5,10,15,20,25,30,35,40]. 进行评分判断的阈值
    title : str or None
        The default is None, 图片的 title
    Returns
    -------
    None.

    '''
    
    Acc_scores = []
    Recall_scores = []
    Precision_scores = []
    F1_scores = []
    TS_scores = []
    MAR_scores = []
    FAR_scores = []
    
    # thresholds = [0.1,5,10, 15,20,25,30]
    
    for threshold in thresholds:
        Acc_scores.append(ACC(obs,pre, threshold = threshold))
        Recall_scores.append(recall(obs,pre, threshold = threshold))
        Precision_scores.append(precision(obs,pre, threshold = threshold))
        F1_scores.append(FSC(obs,pre, threshold = threshold))
        TS_scores.append(TS(obs,pre, threshold = threshold))
        MAR_scores.append(MAR(obs,pre, threshold = threshold))
        FAR_scores.append(FAR(obs,pre, threshold = threshold))
        
         
    plt.figure(figsize = (12,8))
    plt.plot(thresholds, Acc_scores, linewidth = 2,label = 'Acc')
    plt.plot(thresholds, Recall_scores,linewidth = 2, label = 'Recall')
    plt.plot(thresholds, Precision_scores,linewidth = 2, label = 'Precision')
    plt.plot(thresholds, F1_scores, linewidth = 2,label = 'F1')
    plt.plot(thresholds, TS_scores,linewidth = 2, label = 'TS')
    plt.plot(thresholds, MAR_scores, linewidth = 2,label = 'MAR')
    plt.plot(thresholds, FAR_scores, linewidth = 2, label = 'FAR')
    plt.plot([20,20],[0,1],linewidth = 2)
    
    plt.ylabel('scores',fontsize = 20)
    plt.yticks(fontsize = 18)
    plt.ylim([0,1])
    
    plt.xlabel('Thresholds',fontsize = 20)
    plt.xticks(thresholds,thresholds, fontsize = 18)
    
    plt.legend(fontsize = 12)
    
    if title:
        plt.title(title, fontsize = 20)

    plt.show()
    
    return None


#%%
def get_circle_mask(diameter):
    '''
    func: 获取正方形雷达图中真实雷达圆盘所在的位置，即mask
    Parameters
    ----------
    diameter: int
        圆盘的直径，也即雷达图的宽(高)
        
    return:
        mask, 即圆盘位置对应的索引为1，圆盘外的位置对应的索引为0 
    '''
    
    r = diameter // 2
    
    if diameter % 2 == 1:
        x = np.arange(-r,r + 1)
    else:
        x = np.arange(-r,r)
        
    x = abs(x)
    y = x
    
    xx ,yy = np.meshgrid(x,y)
    z = xx**2 + yy**2

    mask = np.where(z <= r**2, 1,0)
    
    return mask


#%%

# mask = get_circle_mask(461)

# import os
# path1 = 'F:/caiyun/radar_images'

# filelist = os.listdir(path1)

# for file in filelist[0:]:
#     filepath = os.path.join(path1,file)
    
#     if os.path.isdir(filepath):
#         continue
    
#     print(filepath)
    
#     img = image_read_cv(filepath) 
#     plot_img(img,title = filepath)
    
    

    

    
    
    
    
    
    
    
    
    
    
    
    





