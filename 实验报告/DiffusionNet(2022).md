![[diffusionnet.png]]
---
# Human
![[human.png]]
## Reproducing Settings
Face-level segmentation(8 classes)
使用hks热核签名特征
数据集为
## Result
![[Training.png]]
**TEST ACC：97.77%**
**Ground Truth  | Prediction**
![[007.png]]

![[040.png]]

边界瑕疵较为明显

--- 
# Coseg
## Chairs（96.16%）
![[chairs0.png]]
## aliens
### HKS（93.97%）
Cases:
![[实验报告/img/alien0.png]]alien0(hks)分不清尾巴和腿（80.2%）
![[alien1.png]]alien1，有较大瑕疵（98.6%） ^31d880

### XYZ
![[xyz_alien0.png]]
**alien0(96.1%)，这说明了一个自然的问题，HKS难以区分两个几何特征相似的形体（尾巴与腿），这时坐标特征带来了更充分的信息**（请对比[[#^31d880]]）