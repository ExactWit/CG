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
![[alien0.png]]alien0分不清尾巴和腿（80.2%）
![[alien1.png]]alien1，有较大瑕疵（98.6%）