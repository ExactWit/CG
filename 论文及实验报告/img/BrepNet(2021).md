---
date: 2026-06-27
---
http://arxiv.org/abs/2104.00706
---
# 方法
使用的特征仅从step中求值、采样，不要求精确分解

| Content              | Shape       | Remark                                                    |
| -------------------- | ----------- | --------------------------------------------------------- |
| face_feat            | (F,7)       | 面类型one-hot（plane,cylinder,cone,sphere,torus）+面积+有理NURBS标志 |
| face_point_grids     | (F,7,10,10) | UV网格采样:xyz，法向，trim mask                                   |
| edge_feat            | (E,10)      | 凹/凸/光滑、长度、圆、椭圆、直线、B-spline类型等                             |
| coedge_feat          | (C,1)       | 是否反向                                                      |
| coedge_points_grids  | (C,12,10)   | 边上采样点：xyz，切向，左/右面法向                                       |
| coedge_scale_factors | (C,)        | 尺度归一化因子                                                   |
| next                 | index       | wire内coedge环                                              |
| mate                 | index       | 对偶coedge                                                  |
| face                 | index       | coedge->face                                              |
| edge                 | index       | coedge->edge                                              |


---
# 实验
![[BRepNet训练结果.png]]
所得mean_iou(per face)低于论文的$0.771\pm 0.005$
可能原因：
a.数据集差异，论文的S2.0.0（旧）和本次使用的S2.0.1（新）有细微区别，S2.0.1仅有35,680个样本。
![[讨论纪要/周报/img/s2.0.0.png]]
b.超参数问题（epoch200 v.s. 50)、s2.0.0可能简单样本更多
已下载旧版本重新训练（6.25）
s2.0.0
![[s2.0.0 1.png]]
论文结果可信！
![[s201case546.png]]
![[s200case546.png]]
仅更换数据集(s200比s201多178个样本)，模型训练效果差别明显，case546的切除侧面（CutSide）被Model_s201错误识别为拉伸侧面。
也有case（#999）反映s200的效果更差，但是错误的**一致性**更强，一致性高的错误应该会比层次不齐的正确要好:

| Dataset  | IoU  | Acc  |
| -------- | ---- | ---- |
| S200(上图) | 63.2 | 77.6 |
| S201（下图） | 59   | 63.8 |
![[操作图例.png]]
![[999s201.png]]
![[999s200.png]]
