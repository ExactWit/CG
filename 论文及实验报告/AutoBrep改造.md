---
date: 2026-07-22
---
关于AutoBrep详见笔记：[[AutoBrep]]

---
# ECCV-26-CAD Challenge
## 三视图不拆分
模型概览
![[autobrep_eccv_base.png]]
### Case Study
169
gt
![[autobrep_eccv_3view_169.png]]
![[autobrep_eccv_169.png]]
pred
![[autobrep_eccv_169pred.png]]
6670
gt
![[006670.svg]]
![[autobrep_eccv_6670.png]]
pred
![[autobrep_eccv_6670pred.png]]
大体形态可以对得上，但是细节啥的有点多。目前三视图未拆分处理，复杂线元关系被压缩为单一向量，同时整体模型参数量大约1B，情理之中。
## 三视图拆分