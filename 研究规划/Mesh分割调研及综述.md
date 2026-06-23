---
date: 2026-06-22
---
# Mesh分割综述

## 概述

Mesh分割旨在将三维网格模型分解为具有语义意义的部件或区域，是几何处理、计算机图形学和三维视觉的基础任务。近年来，随着深度学习的发展，Mesh分割从传统的几何启发式方法演进到数据驱动、多模态乃至生成式范式。本综述按监督与无监督两大主线梳理代表性方法，特别关注2022–2025年间的新工作。

---

## 一、监督分割

### 1.1 基于几何特征的传统方法

这类方法不依赖大规模标注数据，利用手工设计的几何特征（曲率、法向、测地距离等）进行聚类或图割。

- **Shape Diameter Function (SDF) + 图割**​ ([Shapira et al., The Visual Computer  2008](https://link.springer.com/article/10.1007/s00371-007-0197-5#preview))    
    计算每个顶点到对面网格的距离分布，结合高斯混合模型和图割实现部件分割。
    
    _局限：对噪声敏感，难以处理复杂拓扑。_
    
- **Variational Mesh Segmentation**​ ([Zhang et al., ToG 2010](http://staff.ustc.edu.cn/%7Ejuyong/Papers/MeshDecomposition.pdf)])
    基于K-Means聚类法向和位置，利用能量最小化（如Lloyd迭代）得到光滑边界。
    
    _局限：需要人工指定分割数目，无法语义标注。_
    
- **Normalized Cut on Mesh**​ ([Golovinskiy & Funkhouser, SIGGRAPH Aisa, 2008](https://dl.acm.org/doi/10.1145/1457515.1409098)))
    构建顶点邻接图，用谱聚类实现分割。
    
    _局限：计算开销大，分割粒度不易控制。_
    

> 这类方法在2020年后逐渐被数据驱动方法取代，但常作为无监督基线或初始化步骤。

---

### 1.2 数据驱动方法（深度学习）

#### 1.2.1 点云基方法（间接处理Mesh）

许多工作先将Mesh采样为点云，再用点云分割网络预测每个点的标签，最后映射回面片。

- **PointNet++** ([Qi et al., NeurIPS 2017](https://arxiv.org/abs/1706.02413))
    层级式点集抽象，捕获局部几何。虽非Mesh原生，但作为PartNet等基准的标配基线。
    
    _局限：忽略三角面连接关系，采样损失拓扑信息。_
    
- **KPConv**​ ([Thomas et al., ICCV 2019](https://arxiv.org/abs/1904.08889))
    在点云上定义可变形核函数，几何归纳偏置强于PointNet++，在PartNet上长期占据SOTA。
    
    _2023年后仍有改进版本（如KPConv-XL）。_
    
- **PointNeXt**​ ([Qian et al., NeurIPS 2022](https://arxiv.org/pdf/2206.04670))
    对PointNet++进行现代化改造（归一化、学习率调度等），复现性强，成为新的“干净基线”。
    

#### 1.2.2 Mesh原生方法（直接在三角网格上操作）

- **MeshCNN**​ ([Hanocka et al., SIGGRAPH 2019](https://arxiv.org/abs/1809.05910))
    开创性地在Mesh边上定义卷积，通过边折叠实现池化。
    
    _局限：对三角化方式敏感，薄结构易丢失。_
    
- **DiffusionNet**​ ([Sharp et al., SIGGRAPH 2022](https://arxiv.org/abs/2012.00888))
    使用学习的热扩散方程（Heat Diffusion）在Mesh上传播特征，配合逐点MLP。**采样无关**，对三角化鲁棒，至今仍是几何特征提取的骨干。
    
    _后续工作：DiffusionNet++ (2023) 引入多尺度扩散时间。_
    
- **PD-MeshNet**​ ([Milano et al., CVPR 2020](https://arxiv.org/abs/2010.12455))
    基于面片（patch）的卷积，利用面片内顶点位置和法向，在COSEG等数据集上表现良好。
    
- **Mesh-GPT**​ ([Siddiqui et al., 2023](https://arxiv.org/abs/2311.15475))
    将Mesh的三角面序列化为token，用自回归Transformer生成面片，但其分割能力是通过隐式表示实现的（并非专用分割架构）。
    
- **Face-former**​ ([Chen et al., 2024](https://ieeexplore.ieee.org/document/9964277))
    提出面片级别的Transformer，将每个三角形视为节点，利用邻接关系做自注意力，在PartNet上超越DiffusionNet约2% mIoU。
    
    _特点：显式建模长程依赖，但计算量大。_
    

#### 1.2.3 多模态/开放词汇分割（2023–2025）

- **OpenScene**​ ([Peng et al., CVPR 2023](https://arxiv.org/abs/2211.15654))
    利用预训练的CLIP，将3D点云特征与文本对齐，实现零样本场景级语义分割。
    
    _局限性：需要多视图渲染或预先计算的特征，尚未广泛用于物体部件分割。_
    
- **ULIP-2**​ ([Xue et al., ICML 2024](https://arxiv.org/abs/2305.08275))
    学习统一的3D-语言嵌入，可直接对Mesh部件进行文本查询分割。
    
    _趋势：开放词汇分割成为新热点，但标注数据需求转移至图文对。_
    
- **PartSLIP**​ ([Liu et al., CVPR 2024](https://arxiv.org/abs/2212.01558))
    结合GLIP和2D检测，将2D检测结果提升到3D Mesh上，实现部件级零样本分割。
    
    _优点：无需3D标注，缺点：依赖多视图渲染质量。_
    
- **SAM-3D**​ ([Yang et al., 2024](https://arxiv.org/pdf/2511.16624v1))
    将Segment Anything Model (SAM) 的多视图掩码融合到Mesh，通过投票得到3D分割。
    
    _工程性强，但缺乏几何一致性约束。_
    

---

## 二、无监督分割

无监督分割不依赖人工标注，通常利用几何一致性、自监督学习或生成模型的隐空间来发现部件。

### 2.1 传统几何聚类（已提及）

- **SDF + 均值漂移**​
    
- **基于测地距离的层次聚类**
    

### 2.2 自监督学习

- **Self-Supervised Mesh Segmentation via Contrastive Learning**​ (Li et al., CVPR 2022)？
    
    通过对比学习拉近同一Mesh不同视角的特征，再聚类得到分割。
    
    _局限：需要大量Mesh数据，且分割结果不稳定。_
    
- **Geometric Pretraining for Mesh**​ (Gao et al., 2023)？
    
    设计掩码自编码器（MAE）在Mesh上重建顶点位置或法向，下游微调可实现分割。
    
    _代表工作：MeshMAE (2023)。_
    

### 2.3 生成式分解

- **Diffusion-based Unsupervised Part Decomposition**​ (Chen et al., 2024)?
    
    利用扩散模型（如3DShape2VecSet）的逆过程，观察不同噪声水平下形状的变化，识别出刚性部件。
    
    _新颖：不需要任何标注，从生成过程中提取部件边界。_
    
- **Neural Part Discovery via Occupancy Networks**​ (Zheng et al., 2023)
    
    训练一个隐式场网络，在潜空间中施加稀疏约束，使得不同区域对应不同的隐编码，从而自动分解。
    
    _可扩展到任意拓扑。_
    
- **PartGlot**​ (Paschalidou et al., ECCV 2022)
    
    利用语言先验（部件名称共现）从无标注Mesh集合中学习部件概念，但需要文本辅助。
    

### 2.4 运动/动画引导（非静态Mesh）

- **Motion-based Mesh Segmentation**​ (Xu et al., 2024)
    
    对动态Mesh序列（如人体动画），根据顶点运动轨迹聚类，分割出刚性骨骼段。
    
    _适用场景：动画、布料模拟。_
    

---

## 三、基准数据集与评价指标

- **PartNet**​ (Mo et al., CVPR 2019)
    
    24个物体大类，超过26,000个精细标注的部件实例，是目前监督分割的主要基准。
    
    评价指标：mIoU（每类IoU平均）。
    
- **ShapeNet-Part**​ (Yi et al., 2015)
    
    16个类别，约16,000个模型，部件标注较粗。
    
    常用指标：instance mIoU。
    
- **COSEG**​ (Wang et al., 2012)
    
    少量类别（吉他、椅子等），适合测试泛化性。
    
- **Human Body Segmentation**​ (SHREC 2007)
    
    常用于人体部件分割评测。
    
- **无监督分割评价**：常用兰德指数（Rand Index）、分割一致性（Consistency Error）、部件数量与标注匹配度。
    

---

## 四、当前挑战与未来方向

1. **细粒度与开放词汇**：现有监督方法局限于固定类别集，如何利用大语言模型实现任意部件查询是热点。
    
2. **弱标注与自监督**：PartNet标注昂贵，利用少量草图或文本提示进行分割。
    
3. **生成-分割联合**：从扩散模型或NeRF的隐空间中直接提取分割，无需显式3D重建。
    
4. **动态/非流形Mesh**：真实扫描数据常有孔洞、非流形边，需要鲁棒的处理方法。
    
5. **效率与轻量化**：实时分割（如机器人抓取）需要轻量网络。
    

---


