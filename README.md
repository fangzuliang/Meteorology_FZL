## Meteorology_FZL

目前工作中常用的一些函数集成：



### 1 Radar_works 雷达工作

- --- Model_Evaluation_with_Multi_Metrics.py 模型评估
- --- Radar_utils.py   常用函数集合
- ----Loss_Assemble.py 雷达回波外推中常见损失函数集合
  - BMAELoss() ;MAE损失中给强回波处的误差更大的权重，权重固定
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



### 2 CV_Attention 视觉注意力代码

- Axial_Attention.py

- CBAM.py

- CCNet.py

- DANet.py

- ECANet.py

- GCNet.py

- Non_local.py

- RecoNet.py

- Self_Attention.py

- SKENet.py

- SENet.py

  

  

  

  

  

  

  

  

  ​	