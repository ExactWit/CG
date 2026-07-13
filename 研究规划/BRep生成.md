---
date: 2026-07-12
---
# 最新工作

| Name                                                                                                                       | Repo                                                  | Publication               | Affiliation      | Remark                                        |
| -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------- | ---------------- | --------------------------------------------- |
| [ B-repLer: Language-guided Editing of CAD Models](https://arxiv.org/abs/2508.10201)                                       | [Brepler](https://github.com/yilinliu77/Brepler)      | SIGGRAPH 2026             | AutoDesk         | BRep+text$\rightarrow$BRep；Diffusion Model    |
| [ AutoBrep: Autoregressive B-Rep Generation with Unified Topology and Geometry](https://arxiv.org/abs/2512.03018)          | [AutoBrep](https://github.com/AutodeskAILab/AutoBrep) | SIGGRAPH Asia 2025        | AutoDesk         | Discrete coordinate；Transformer；Serialization |
| [ HoLa: B-Rep Generation using a Holistic Latent Representation](https://arxiv.org/abs/2504.14257)                         | Patent Protected                                      | SIGGRAPH 2025<br>ToG 2025 | 深圳大学             | Latent; Holistic                              |
| [DualBrep: A Dual-Field Continuous Representation for B-rep Modelling](https://arxiv.org/pdf/2606.31579)                   | [DualBrep](https://github.com/AutodeskAILab/DualBrep) | SIGGRAPH 2026             | AutoDesk         | SDF+Parameterization                          |
| [ Text2CAD: Generating Sequential CAD Models from Beginner-to-Expert Level Text Prompts](https://arxiv.org/abs/2409.17106) | [Text2CAD](https://github.com/SadilKhan/Text2CAD)     | NIPS 2024                 | 德国人工智能研究中心（DFKI） | Text$\rightarrow$Construction Sequence；       |
| [Sketch2CAD: Sequential CAD Modeling by Sketching in Context](https://arxiv.org/pdf/2009.04927)                            | [Sketch2CAD](https://github.com/Enigma-li/Sketch2CAD) | SIGGRAPH Asia 2020        | 伦敦大学, Adobe, 微软  |                                               |
注：上述工作几乎都是无条件生成或文本条件生成，**没有一个是拿工程视图（3D渲染+SVG三视图）来做逆向工程的**（ECCV CAD Challenge）。工程图转mesh、voxel的工作老旧成熟，但工程图转BRep的工作尚且空白。

---

# 最新活动
1.[ECCV CAD Challenge（HKU）](https://huggingface.co/datasets/jingwei-xu-00/eccv2026-cad-challenge-data)：输入CAD模型的**3D渲染**和**TechDraw视图**，生成相应的标准**STEP BRep**（Due date: Oct.15）

# Notes
对于BRep生成任务，从无到有得到一个实体的边界表示，让我们尝试想象我们需要什么。
容易想到有两种生成模式：
## 两种生成 B-Rep 的底层逻辑

### 路径 A：构造历史（操作序列）
**“创建一个草图 → 拉伸 → 打孔 → 倒角 → …”**  
这本质是在**模仿人类 CAD 工程师的设计过程**。模型输出一串宏指令，最终的 B-Rep 是通过执行这些指令（用 CAD 内核）得到的。

**代表工作**：Text2CAD, DeepCAD, ZoneGraph 等。

**优点**：
- 输出高度结构化、可编辑（改一个参数就变了整个零件）。
- 序列长度较短（几十步操作即可生成复杂形状）。
- 自然适合语言/草图条件，因为工程师就是看着图纸这么干的。

**致命弱点**：
- **必须依赖 CAD 内核**，端到端不可微，训练只能靠 RL 或预定义顺序的监督，极难优化。
- **覆盖面有限**：只能表达能用该宏语言（如 sketch+extrude+boolean）描述的 B-Rep，遇到复杂自由曲面（如叶片、汽车覆盖件）直接歇菜。
- **无法利用连续几何表示**：操作指令里的草图还是离散线段，拉伸出来的面也只是自然的平面/圆柱。

### 路径 B：直接边界表示（拓扑+几何）
**“这个实体有 N 个面，面 1 是 Bézier 片控制点为…，它与面 2 以边 E₁ 相连…”**  
模型直接输出 B-Rep 的**拓扑结构**（面-边-环关系图）和**几何载体**（每个面的曲面参数，如控制点）。最后通过缝合算法得到有效实体。

**代表工作**：AutoBrep, HoLa, DualBrep等。

**优点**：
- **端到端可微**（通过可微缝合或直接监督控制点+拓扑关系），训练稳定。
- **可以表示任意复杂曲面**：理论上能逼近任何光滑面。
- **天然适合你的连续 tokenizer**：每个面就是一个 latent，解码出控制点，拓扑分组保留结构

**弱点**：
- 拓扑生成是一个难啃的骨头——如何保证生成的面-边-环关系是一个闭合、无冗余、非自交的合法实体？这需要强大的结构约束或后处理。
- 输出序列可能很长（几十个面，每个面很多坐标），对自回归模型压力大。

---

## 输入数据告诉我们什么？

现在我们看看输入：**3D 渲染图 + TechDraw 三视图**。这些数据是“结果导向”的，不是“过程导向”的。

- 一张渲染图展示的是**最终实体的外观**：曲面的明暗、遮挡关系、阴影。
- 一张 SVG 三视图展示的是**最终实体的精确投影轮廓**。

它们共同给出了关于**“这个零件长什么样”**的完备信息，但几乎没有给出**“它是怎么画出来的”**的信息。人类专家能从三视图反推建模步骤，但这需要极高智能，远非当前数据驱动模型所能稳定习得。

因此，**输入信号本质上是在描述“最终实体的几何与拓扑”，而不是“构造历史”**。强行让模型从结果图反推操作序列，等于是让它解一个欠定的逆问题（同一个 B-Rep 可以有无数种构造步骤），白白增加学习负担。

**结论：输入数据强烈倾向路径 B（直接生成拓扑+几何）。**

---

## 那么，我们具体需要什么？

如果走路径 B，从无到有生成一个实体的 B-Rep，我们需要一个生成系统，它能够输出两部分信息，并且这两部分要一致：

### 1. 拓扑骨架 (Topological Skeleton)
- **面的集合**：多少个面，每个面的类型（平面/圆柱/Bézier/…）。
- **边与环的集合**：哪些边属于哪个面，边由哪些顶点界定，面的边界环由哪些边构成。
- **邻接关系**：面与面之间共享哪些边，这些边方向如何，保证整体构成一个封闭的二维流形。

这部分可以用一个**图结构**（面-边-顶点 bipartite 图）或一个**精心设计的序列**来表示。AutoBrep 证明自回归序列可以生成合法拓扑，但需要大量的拓扑规则约束。你也可以考虑用 **Set Prediction + 匹配** 的方式，同时预测所有面及其拓扑关系，然后用 Hungarian 匹配来对应真值。

### 2. 几何载体 (Geometry Carrier)
- **每个面的数学表示**：对于每个面，需要输出其曲面方程的**系数**（如 Bézier 控制点、NURBS 控制点和节点向量）。
- **每条边的几何**：边的曲线方程（通常是两个面的交线，也可以是独立的 3D 曲线）。在简单方案中，边可以由两个相邻面的控制点边界隐式定义（即两个面在边界上共享控制点），这样就不需要单独生成边几何，拓扑约束自动保证几何连续性。

**连续神经 B-Rep 表示在这里大放异彩**：每个面的几何不是一个离散点云，而是一个 **latent vector**，它通过 tokenizer 映射到控制点环。这个 latent 可以捕捉该面内蕴的几何（与仿射无关的曲率特征），使得生成过程是在一个流形空间中操作，更平滑、更可泛化。

### 3. 缝合与验证
- 需要有一个**可微或至少可评价的缝合模块**，把预测的面片和拓扑关系组装成 Solid，并在训练时给出反馈（如面片间缝隙大小、自交检测）。官方评估器已经提供了不可微的指标，我们可以把它作为奖励信号或验证指标，训练时则用更松弛的损失函数（如对应面片的 Chamfer 距离、边重合度损失）来引导。

---

## 生成架构蓝图（核心创新点）

结合上面分析，我建议你的生成系统采用 **“拓扑图+几何latent”的双流架构**：

1. **多模态编码器**输出两个条件向量：  
   - \(c_{topo}\)：偏向结构信息（从隐藏线视图、SVG 轮廓中提取）。  
   - \(c_{geo}\)：偏向几何信息（从渲染图的光影中提取）。

2. **拓扑解码器**：基于 \(c_{topo}\) 生成一个**面-边-环图**（可用 Graph Transformer 或 Pointer Network）。它输出面的数量、类型，以及它们之间的邻接矩阵。

3. **几何解码器**：对每一个预测出的面，以 \(c_{geo}\) 和该面的拓扑上下文为条件，生成一个 **geometry latent \(z_i\)**。

4. **你的连续表示模块（核心）**：这是一个**预训练好、冻结的 B-Rep tokenizer**，它能够把 \(z_i\) 解码成该面的控制点环（或等价的多项式系数）。因为是冻结的，它保证了生成的面一定位于真实 B-Rep 面的流形上，极大降低了学习难度。

5. **缝合模块**：将控制点环和拓扑图喂入 PythonOCC 缝合，输出 STEP，并计算损失（端到端梯度可通过控制点回传，因为缝合前的控制点是可微的）。

---

## 为什么这个架构很可能赢

- **拓扑和几何解耦**：符合 B-Rep 的本源，各自的学习目标更清晰。
- **你的连续 tokenizer 成为核心壁垒**：它把几何生成变成了“在已知流形上插值”，而不是“从零学曲面函数”，这将让几何精度飞升，且训练极稳。
- **多模态信息各司其职**：SVG 管拓扑边界，渲染图管曲面起伏，完美分工。
- **完全可微**：从像素到控制点，无需任何不可微的 CAD 操作介入训练循环（只有最终缝合是不在训练中的后处理，但损失在缝合前的控制点和拓扑上就已计算）。

---

**总结**：我们不需要生成操作指令，我们要**直接生成 B-Rep 的拓扑骨架和每个面的内蕴几何 latent**，然后用你的 tokenizer 把这些 latent 变为精确的控制点环，最后缝合。输入的多模态数据恰好能分别为拓扑和几何提供监督信号。这就是你论文的骨架。

