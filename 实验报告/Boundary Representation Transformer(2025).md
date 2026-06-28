---
date: 2026-06-28
---
[Bringing Attention to CAD: Boundary Representation Learning via Transformer](10.1016/j.cad.2025.10394)
# 概览
## 问题（Problem）
现有几何深度学习方法在实体建模中的应用通常依赖离散化，这引入了采样误差，一定程度上牺牲了原生连续表示的形状信息...(原文给出的problem是processing boundary representation (B-rep) models via transformer remains largely unexplored)
主流方法仍是GNN，难以跟上NLP(Transformer)潮流...
## 动机 （Motivation）
利用B-Rep进行直接实体建模（Direct Solid Modeling），避免转换为点云、网格、体素带来的精度损失。
## 方法（Solution）
![[BRTMethod.png]]
将B-Rep视为一个多级有向图（Vertex$\rightarrow$Egde$\rightarrow$Loop$\rightarrow$Face$\rightarrow$Shell），逐级编码边界元素：
### **点表示**：
直接使用MLP编码**三维坐标**
### **边表示**：
[[B-Spline]]控制点数量是不定的，传统方法通过采样固定数量的点或折线，这样的方式依赖具体的离散化尺度校准。本文将B样条曲线分解为连通的[[Bezier Curve]]序列，利用Boehm算法进行节点插入。
使用MLP将贝塞尔控制点与中点切向量进行编码得到隐表示，再使用Transformer对隐表示序列进行编码（平均池化）
### **面表示**：
使用贝塞尔三角来表示B样条曲面：对于非剪裁B样条曲面，先进行结点插入得到贝塞尔矩形表示，然后转换为贝塞尔三角表示；对于剪裁B样条曲面，则进行一系列**规则操作**（见原文3.2.3）


1.使用MLP编码点坐标，得到点嵌入表示
2.使用MLP+Transformer编码参数曲线对应的贝塞尔控制点与切向量，得到边嵌入表示
3.使用1、2结果，利用RNN编码Loop，得到环嵌入表示
4.使用MLP+Transformer编码将参数曲面对应的贝塞尔三角控制点和面法向量，得到面嵌入表示
5.
6.
## 贡献 （Contribution）
1.一种B-Rep的分词方法（序列化）
2.一种连续域上的几何编码方法，无需离散化
3.一种更全面的拓扑编码方法

# Notes：代数拓扑与微分几何视角下的边界表示建模（B-Rep Modeling）
接下来笔者将引入代数拓扑与微分几何的内容来重新认识Boundary Representation及其建模方法，主要动机在于：B-Rep表示反映的拓扑信息也许能用同调论中的链群进行系统描述，而微分几何则为曲线曲面的内蕴表示提供了有力工具。
## B-Rep
![[BRep.png]]
一个实体对象在边界表示下的结构是层次化的，从宏观到微观依次是Solids-Shells-Faces-Loops-Edges-Vertices，这种结构恰好对应了三维实体作为带边流形(Manifold with boundary)的胞腔分解（Cell Decomposition）：

| B-Rep元素 | 拓扑维度 | 代数拓扑对应                        | 微分几何对应             |
| ------- | ---- | ----------------------------- | ------------------ |
| Solid   | 3    | 3维带边流形$\mathcal{M}$           | $\mathbb{R}^3$有界区域 |
| Shell   | 2    | 边界$\partial \mathcal{M}$的连通分支 | 闭曲面或带边曲面的并         |
| Face    | 2    | 2-胞腔（2-cell）                  | 带边参数曲面片            |
| Loop    | 1    | 1-闭链（1-cycle），面边界$\partial F$ | 分段光滑闭曲线            |
| Edge    | 1    | 1-cell                        | 参数曲线段              |
| Vertex  | 0    | 0-cell                        | 点                  |
>[!remark]
> "A solid is mathematically defined as a **bounded, closed, regular, and semi-analytic** of $\mathbb{R}^3$" 
> 三维实体是一个$\mathbb{R}^{3}$中的正则集（r-set）
****

##