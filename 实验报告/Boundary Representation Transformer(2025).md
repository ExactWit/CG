---
date: 2026-06-28
---
[Bringing Attention to CAD: Boundary Representation Learning via Transformer](10.1016/j.cad.2025.10394)（CAD 2025, Zou et.al,ZJU CAD/CG）
# 概览
## 问题
>[!problem]
>现有几何深度学习方法在实体建模中的应用通常依赖离散化，这引入了采样误差，一定程度上牺牲了原生连续表示的形状信息...
>(原文给出的problem是processing boundary representation (B-rep) models via transformer remains largely unexplored)
>主流方法仍是GNN，难以跟上NLP(Transformer)潮流...
## 动机 
>[!motivation]
>利用B-Rep进行直接实体建模（Direct Solid Modeling），避免转换为点云、网格、体素带来的精度损失。
## 方法
![[BRTMethod.png]]
>[!solution]
>将B-Rep视为一个多级有向图（Vertex$\rightarrow$Egde$\rightarrow$Loop$\rightarrow$Face$\rightarrow$Shell），逐级编码边界元素，
>首先进行**几何信息嵌入**：
>### **点表示**：
>直接使用MLP编码**三维坐标**
>
>### **边表示**：
>[[B-Spline]]控制点数量是不定的，传统方法通过采样固定数量的点或折线，这样的方式依赖具体的离散化尺度校准。本文将B样条曲线分解为连通的[[Bezier Curve]]序列，利用Boehm算法进行节点插入。
>使用MLP将贝塞尔控制点与中点切向量进行编码得到隐表示，再使用Transformer对隐表示序列进行编码（平均池化）
>
>### **面表示**：
>使用贝塞尔三角来表示B样条曲面：对于非剪裁B样条曲面，先进行结点插入得到贝塞尔矩>形表示，然后转换为贝塞尔三角表示；对于剪裁B样条曲面，则进行一系列**规则操作>**（见原文3.2.3，本文的重点）
>使用MLP+Transformer对面法向量和控制点进行编码。
>
>其次进行**拓扑信息聚合**：
>### **环表示**：
>利用RNN对环上的点、边表示进行编码
>
>### **壳表示**：
利用MLP+Transformer对环表示和面表示进行编码，得到用于下游任务的表示。

## 贡献 
>[!contribution]
>1.一种B-Rep的分词方法（序列化）
>2.一种连续域上的几何编码方法，无需离散化
>3.一种更全面的拓扑编码方法
>4.开展了零件**形状分类**（TMCAD、FebWave、SolidLetters）、**操作分割**（Fusion360Seg）、**机加工特征识别**（MFCAD++）上四个模型（BrepNet、UV-Net、AAGNet、BRT）的实验（似乎BrepNet的指标偏低，笔者在s201上的结果是IoU70，S200上IoU79）

---
# 复现结果
## Fusion 360 Gallery Segmentation S2.0.0
BRT数据划分与BrepNet对不上因此可比性不强，但仍和论文差距较大（使用默认划分，应该和论文对的上），现已使用与BRepNet相同的划分重新训练，但仍有**预处理问题**导致部分样本（2601个，占7.3%）无法为BRT使用：
![[BRT数据预处理问题.png]]
公平起见，需要在BRepNet上排除缺失样本重新看看BRepNet的性能，作为BRT的参照；其次，这个三角化问题说明BRT对于某些奇异情况不能很好处理，这是一个未来可以改进的点。

| Model            | IoU       | Acc       |
| ---------------- | --------- | --------- |
| BrepNet          | **79.3**  | **93.6**  |
| BRT              |           |           |
| BRT（划分有问题）       | 71.8      | 92.9      |
| BrepNet(BRT论文披露) | 68.92     | 90.91     |
| BRT(论文披露)        | **79.23** | **94.48** |


--- 
# Notes：代数拓扑与微分几何视角下的边界表示建模（B-Rep Modeling）
接下来笔者将引入代数拓扑与微分几何的内容来重新认识Boundary Representation及其建模方法，主要动机在于：B-Rep表示反映的拓扑信息也许能用同调论中的链群进行系统描述，而微分几何则为曲线曲面的内蕴表示提供了有力工具。
## 胞腔分解
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

更正式地，我们针对**流形B-Rep**给出以下定义：
>[!definition]
>一个**流形B-Rep实体模型**$\mathcal{B}$ 由以下资料组成：
>**（1）承载空间（Underlying Space）** 一个**三维带边流形$\mathcal{M}$** (3-manifold with boundary) 及其其边界$\partial\mathcal{M}$，记为$|\mathcal{B}|$。通常$\mathcal{M}$是$\mathbb{R}^3$中的紧致子集。
>**（2）胞腔复形（Cell Complex）**边界$\partial\mathcal{M}$上的一个**有限正则胞腔分解**（finite regular  cell decomposition） ：
>$$
>\mathcal{C}=C_0\bigsqcup C_1\bigsqcup C_2,
>$$
>其中
>$$
>\begin{gather}
>C_0=\set{v_1,\dots,v_{n_0}},\text{i.e. vertexs}\\
>C_0=\set{e_1,\dots,e_{n_1}},\text{i.e. edges}\\
>C_0=\set{f_1,\dots,f_{n_2}},\text{i.e. faces}
>\end{gather}
>$$
>**（3）关联数与边界算子（Incidence Numbers & Boundary Operators）**
>对每个$k=1,2$定义$\mathbb{Z}$-线性边界算子
>$$
>\partial_k：C_k\rightarrow C_{k-1},\partial_k(c)=\sum_{c^\prime\in C_{k-1}}[c:c^\prime]\cdot c^\prime
>$$
>其中关联数$[c:c^\prime]=\set{-1,0,1}$，当$c^\prime\in \partial c$时根据定向相容性取$\pm 1$
>
>
>

>[!remark]
>一般情况，我们处理单连通实体$s$，其满足$C_3=\set{s}$，有$\partial_3(s)=\sum_{f\in C_2}f$（全体Face的代数和）。
>在边界表示中，$s$由$\partial \mathcal{M}$隐式界定。

流形B-Rep满足以下代数拓扑公理：
>[!property]
>**Axiom.1 Nilpotency**
>$$
>\partial_k\circ\partial_{k+1}=0,k=1,2
>$$
>**Axiom.2 Regularity**
>$$
>	\forall c\in C_k，\exists \Phi_c:{D}^k\rightarrow |\mathcal{B}|,
>$$
>满足:
>$\Phi_c(D^k)=\bar c$(像集是$c$的闭包)，
>$\Phi_c|_{\mathring{D}^k}:{\mathring{D}^k}\rightarrow c$是同胚（内部无折叠、重叠），
>$\Phi_c|_{\partial{D}^k}:\partial{D}^k\rightarrow c\setminus c$是嵌入（边界无自粘）
>
>**Axiom.3 Manifold Condition**
>$\partial\mathcal{M}$是闭2-流形，也即：
>对于每个边$e\in C_1$定义面度$deg(e):=\#\set{f\in C_2|[f:e]\ne 0}$，要求：
>$\forall e\in C_1,deg(e)=2$


>[!remark]
>公理1保证$\mathcal{B}$是水密的（边界的边界为空）；
>公理2要求k-胞腔内部是无折叠、重叠的（同胚于k-开圆盘内部），边界是无粘合的；
>公理3要求局部像二维平面，每条边恰好连接两个面。
>然而实际工业界情况存在诸多情况使得流形B-Rep模型无法合理容纳（需要额外的异常处理）：
>**1.零厚度壁（违反公理2）**
>钣金件、垫片、电路板铜层在CAD中都是"零厚度"的，两个不同的Face占用了空间的同一位置（某精度下），整体嵌入的单射性被破坏了。如果强制要求两个Face不能重合，这些零件就无法表示。
>**2.T型接头（违反公理3）**
>注塑件、焊接件、建筑结构中的梁交接处，普遍存在三个或更多表面交汇于一条边，使得多于两个面交汇于同一条边。
>**3.奇异点（违反公理3）**
>复杂零件的尖角、多特征交汇处（如五个倒角交于一点）、3D打印中的支撑连接点...
>
>于是，我们不妨考虑通过弱化正则性（允许不同胞腔几何重叠）和流形性（允许 $\deg(e)\ge 3$ 及奇异链接来容纳这些真实实体，建立**广义B-Rep框架**。

## 胞腔复形上的离散微分形式

>[!definition]
>**链群（Chain Groups）**
>令$C_k()$

---

