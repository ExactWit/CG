论文：[BRepCLIP: Contrastive Multimodal Pretraining on BRep Primitives for CAD Understanding，ArXiv 2026](https://arxiv.org/pdf/2606.05515)
代码：Coming Soon...
权重：Coming Soon...
资源：4卡A100（Stage 1）,1卡A100(Stage 2)

---
# 问题
>[!problem]
>现有3D多模态方法通常使用**点云表示**，目前没有多模态方法能有效处理CAD模型的**原生**BRep结构：
>- 基于**点云表示**的方法（ULIP-2 CVPR 2024）将精确的边界表示退化为无序坐标集，丢失解析曲面类型、曲线基元、拓扑邻接关系；
>- **网格表示**、**体素表示**因其固有缺陷，在3D多模态方法中较少出现；
>- 现有BRep学习方法（如UV-Net、BRepNet）只关注CAD数据本身的预训练，未对齐语言/图像模态，无法做**开放词汇检索**。

>[!remark]
>在**3D多模态学习**的工作中**点云表示**是主流。
>**体素表示**因其内存占用高、分辨率受限的特性，直接出局；而主流工作较少出现**网格表示**的主要原因是缺乏**典范网格化方法**，这导致了Mesh表示具有以下缺陷：
>- 不同样本的边数、面数差异极大，批次化计算性能浪费大（padding、masking）；
>- 不同复杂度模型的恰当Mesh化面数难以被同一固定数目满足；
>- 作为Graph的连通信息无法与几何特征一一对应（同一个圆柱可以对应无数种图结构）

# 动机
>[!motivation]
>