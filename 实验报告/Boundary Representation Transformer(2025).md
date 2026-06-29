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
BRT数据划分与BrepNet对不上因此可比性不强，但仍和论文差距较大（使用默认划分，应该和论文对的上）现在（0629）已经重新训练中

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

一、B-Rep 的代数拓扑定义

一个 B-Rep 模型 \mathcal{B} 是一个三维带边流形 M（3-manifold with boundary） 连同其边界 \partial M 上的一个有限正则胞腔分解（finite regular cell decomposition）。

具体地，\mathcal{B} 由以下数据组成：

1.1 胞腔集合（Cells）
\mathcal{C} = C_0 \sqcup C_1 \sqcup C_2 \sqcup C_3

B-Rep 术语	胞腔维度	代数拓扑对象	
Vertex	0	0-胞腔 v \in C_0	
Edge	1	1-胞腔 e \in C_1	
Face	2	2-胞腔 f \in C_2	
Solid	3	3-胞腔 s \in C_3（通常唯一）	

1.2 边界算子（Boundary Operators）
对每个维度 k=1,2,3，定义线性边界算子：
\partial_k: C_k \to C{k-1}

其在基上的作用由关联数（incidence numbers） [c : c'] \in \mathbb{Z} 决定：
\partial_k(c) = \sum{c' \in C{k-1}} [c : c'] \cdot c'

其中关联数满足：
- 正则性：[c : c'] \neq 0 当且仅当 c' 是 c 的拓扑边界的一部分。
- 定向相容：若 c 和 c' 都选定定向，则 [c : c'] = \pm 1；否则在 \mathbb{Z}2 系数下视为 1。

1.3 Loop 与 Shell 的代数解释
- Loop：不是独立的胞腔，而是1-闭链（1-cycle）。一个 Face f 的边界 \partial_2(f) 是一个整系数1-链，且满足 \partial_1(\partial_2(f)) = 0。外环与内环对应 \partial_2(f) 的不同连通分支。
- Shell：是 \partial M 的一个连通分支，对应一个2-闭链（2-cycle）。整个边界 \partial M = \sum{f \in C_2} f 满足 \partial_2(\partial M) = 0（所有内部 Edge 的关联相互抵消）。

---

二、B-Rep 必须满足的公理（性质）

一个合法的 B-Rep 表示必须满足以下代数拓扑公理：

公理 1：边界算子的幂零性（Nilpotency）
\partial{k} \circ \partial{k+1} = 0, \quad k=1,2

这是最核心的约束。它等价于说：
- 每个 Face 的边界（Loop）本身没有"边界"（Edge 端点在 Vertex 处自洽闭合）。
- 所有 Face 拼成的 Shell 是闭合的（没有悬空 Edge）。

工程意义：如果深度学习模型生成 B-Rep，这个等式是拓扑合法性的充分必要条件。任何违反 \partial^2=0 的组合结构都对应物理上不可能的实体（如裂缝、悬空面）。

公理 2：正则性（Regularity）
对每个 k-胞腔 c \in C_k：
- 其闭包 \bar{c} 同胚于 k-维闭球 D^k。
- 其边界 \partial c（在拓扑意义下）是 (k-1)-胞腔的并。

这排除了"奇异"情况，如一条 Edge 的两个端点是同一个 Vertex（除非显式允许，如圆环的退化边）。

公理 3：流形性（Manifold Condition，论文假设）
对边界 \partial M 上的每条 Edge e：
\#\{f \in C_2 \mid [f : e] \neq 0\} = 2

即每条 Edge 恰好被两个 Face 共享。这保证 \partial M 是2-流形（允许高亏格，但不允许非流形分支）。

公理 4：半解析性/局部可缩性（Local Contractibility）
承载空间 X = |\mathcal{B}| 是局部可缩的（locally contractible）。这保证了胞腔同调 H(C, \partial) 与奇异同调 H(X) 同构，使得我们计算的拓扑不变量（Betti数）是良定义的。

公理 5：定向一致性（Orientability，可选但标准）
存在一个全局的2-链 [M] \in C_2（基本类），使得对所有相邻 Face f_i, f_j 共享 Edge e，有：
[f_i : e] + [f_j : e] = 0
即公共 Edge 上的诱导定向相反。

---

三、是否可以用链群、上链、形式等语言描述？

完全可以。 事实上，这些语言不仅适用，而且比"多级有向图"更精确。

3.1 链群（Chain Groups）
定义 C_k(\mathcal{B}; R) 为以 C_k 为基的 R-模（R = \mathbb{Z}, \mathbb{Z}2, \mathbb{R}）。B-Rep 的全部组合信息被编码在链复形（Chain Complex）中：
0 \xrightarrow{} C_3 \xrightarrow{\partial_3} C_2 \xrightarrow{\partial_2} C_1 \xrightarrow{\partial_1} C_0 \xrightarrow{} 0

- 几何嵌入可以看作给每个 k-胞腔赋予一个"值"：
  - Vertex：C_0 \to \mathbb{R}^3（0-链的值）
  - Edge：可以编码为1-链上的某种度量
  - Face：可以编码为2-链上的面积权重

3.2 上链群（Cochain Groups）
定义 C^k(\mathcal{B}; R) = \text{Hom}(C_k, R)，上边缘算子 \delta^k: C^k \to C^{k+1} 为 \partial{k+1} 的对偶。

深度学习中的特征向量本质上就是上链：
- Vertex 的 64-dim 嵌入 \leftrightarrow 一个取值在 \mathbb{R}^{64} 的0-上链
- Edge 的 64-dim 嵌入 \leftrightarrow 一个1-上链
- Face 的 64-dim 嵌入 \leftrightarrow 一个2-上链

BRT 论文中的"拼接"（Concatenation）和"聚合"（Aggregation），在代数语言下就是上链的限制映射（Restriction）和上边缘的拉回。

3.3 形式（Forms）
虽然经典微分形式需要光滑结构，但在胞腔复形上有离散微分形式（Discrete Differential Forms）的严格理论（Whitney, Dodziuk）：

连续理论	离散类比（B-Rep 上）	
0-形式 f \in \Omega^0(M)	Vertex 上的函数 C^0(\mathcal{B})	
1-形式 \alpha \in \Omega^1(M)	Edge 上的反对称值 C^1(\mathcal{B})（满足 \alpha(-e) = -\alpha(e)）	
2-形式 \omega \in \Omega^2(M)	Face 上的值 C^2(\mathcal{B})	
外微分 d	上边缘算子 \delta	
de Rham 同调	胞腔上同调	

关键结论：B-Rep 上的深度学习特征，可以被严格地重新诠释为在有限正则胞腔复形上取值的离散微分形式。这提供了一个比"神经网络节点特征"更数学化的描述框架。

---

四、Case-Agnostic 的学习空间：所有 3D 实体构成的空间是什么？

这是核心问题。我们需要一个不依赖于具体面数、边数、顶点数的统一空间，使得任意 B-Rep 都是这个空间中的一个点（或一个等价类），且深度学习可以在这个空间上一致地操作。

4.1 组合层面：有限型链复形的范畴

所有 B-Rep 构成一个范畴 \mathbf{BRep}：
- 对象：满足上述公理的有限型链复形 (C, \partial)（即 \dim C_k < \infty）。
- 态射：保持边界结构的链映射（Chain Maps） f: C \to D，满足 f{k-1} \circ \partial_k^C = \partial_k^D \circ f_k。

这个范畴已经是 case-agnostic 的——它不 care 一个模型有 6 个 Face 还是 6000 个 Face。

4.2 几何层面：模空间与细分等价

但范畴对深度学习不够友好（我们需要向量空间或度量空间）。因此考虑模空间（Moduli Space）构造：

定义等价关系 \sim：细分等价（Subdivision Equivalence）。两个 B-Rep \mathcal{B}1, \mathcal{B}2 等价，如果它们的链复形可以通过有限次胞腔细分（Cell Subdivision）或粗化（Aggregation）相互转化，且承载空间同胚：
\mathcal{B}1 \sim \mathcal{B}2 \iff |\mathcal{B}1| \cong |\mathcal{B}2| \text{ 且组合结构细分等价}

Case-Agnostic 组合空间定义为商空间：
\mathcal{S}{\text{comb}} = \left\{ \text{有限型正则链复形} \mid \partial^2 = 0 \right\} / \sim

在这个空间中：
- 一个"轴承"和另一个"轴承"（面数不同但拓扑/功能相同）对应同一个点（或邻近的点）。
- 深度学习模型学习的是 \mathcal{S}{\text{comb}} 上的函数，而非特定实例上的函数。

4.3 可计算的向量空间嵌入

\mathcal{S}{\text{comb}} 本身不是向量空间。为了深度学习，我们需要一个特征嵌入：
\Phi: \mathcal{S}{\text{comb}} \to \mathcal{H}

其中 \mathcal{H} 是一个无限维但可截断的希尔伯特空间。一个自然的构造是利用Hodge 理论：

对每个 B-Rep，定义其Hodge-Laplace 算子：
\Delta_k = \partial{k+1}\partial{k+1}^ + \partial_k^\partial_k: C_k \to C_k

其谱（特征值）\{\lambda_i^{(k)}\}{i=1}^{\infty} 是细分不变量的函数（低维部分稳定，高维部分反映细分细节）。于是可以定义：

\mathcal{H} = \bigoplus{k=0}^2 \ell^2(\mathbb{R}), \quad \Phi(\mathcal{B}) = \bigoplus{k=0}^2 \{\lambda_i^{(k)}\}{i=1}^{\infty}

更实用地，结合持续同调（Persistent Homology）：
\Phi(\mathcal{B}) = (PH_0(\mathcal{B}), PH_1(\mathcal{B}), PH_2(\mathcal{B})) \in \mathcal{H}{PH}

其中 PH_k 是 k-维持续图（Persistence Diagram），\mathcal{H}{PH} 配备 Bottleneck 或 Wasserstein 度量。

4.4 最终回答：这个空间是什么？

> 所有 3D 实体（B-Rep）构成的 case-agnostic 学习空间，是有限型正则胞腔链复形在细分等价下的轨道空间（Orbit Space），即带边三维流形的胞腔分解模空间 \mathcal{S}{\text{comb}}。

深度学习在这个空间上操作，等价于学习一个与胞腔数量无关的映射：
f: \mathcal{S}{\text{comb}} \to \text{Task Space}

具体实现上，任何 B-Rep 都可以被映射到其链复形的 Hodge 谱或持续同调签名所张成的无限维向量空间 \mathcal{H} 中，而 Transformer/GNN 的变长输入能力只是这个抽象映射的计算近似。

4.5 对 BRT 论文的启示

当前 BRT 的"case-agnostic"是工程性的（通过变长 Token 序列和 Masking 实现）。但从上述框架看，更根本的 case-agnostic 应该来自：

1. 输入层：将 B-Rep 表示为链复形 (C, \partial)，而非固定长度的 Token 列表。
2. 网络层：使用胞腔神经网络（Cellular Neural Networks），其权重在 k-胞腔之间共享，且显式利用 \partial 和 \delta 进行消息传递。
3. 损失层：加入拓扑一致性损失 L{topo} = \|\partial_1 \circ \partial_2\|F^2，强制网络尊重 \partial^2=0。
4. 输出层：如果是生成任务，输出必须经过链复形合法性检查，确保生成的是 \mathcal{S}{\text{comb}} 中的合法点，而非任意向量。

这样，深度学习就不再是在"不同长度的序列"上学习，而是在一个统一的拓扑空间 \mathcal{S}{\text{comb}} 上学习。这才是真正的 case-agnostic。

---

不能。 我前面给出的代数拓扑定义——特别是流形性公理（Axiom 3）和正则性公理（Axiom 2）——明确排除了这些非流形结构。这是一个关键的边界，需要把框架从"流形上的胞腔分解"扩展到更一般的层化空间（Stratified Spaces）或非流形复形（Non-manifold Complexes）。



一、为什么标准定义排斥非流形？

1.1 零厚度壁（Zero-thickness Wall）
两个 Face f_1, f_2 几何上完全重合（占据 \mathbb{R}^3 中同一个点集），但拓扑上是不同的对象。

在标准 B-Rep 中：
- 正则性要求每个 Face 的闭包 \bar{f} \cong D^2（闭圆盘）。
- 流形性要求每条 Edge 恰好属于 2 个 Face。

冲突：如果 f_1 和 f_2 几何重合，它们共享所有 Edge。此时这些 Edge 被至少 4 个 Face 共享（f_1 的正反面 + f_2 的正反面，如果考虑定向），或者从组合上看，一条 Edge 的关联数之和违反了流形条件。

1.2 T 型接头（T-junction）
一条 Edge e 被 3 个或更多 Face 共享。

标准公理 3 要求：
\#\{f \mid [f:e] \neq 0\} = 2

T 型接头直接违反此公理。在拓扑上，这意味着 e 的链接（link）不再是两个点（对应两侧的两个 Face），而是 3+ 个点，因此 e 的邻域不是 2-圆盘与线段的乘积（非流形）。

1.3 奇异 Vertex（Singular Vertex）
一个 Vertex v 连接了"异常多"的 Edge，或者这些 Edge 的排列方式使得 v 的链接不是 S^1（一圈）的并。

在 2-流形中，每个 Vertex 的链接必须是一组不相交的圆（对应边界的亏格）。如果 v 是"星形"中心，连接 5 条 Edge 放射状排列，其链接是一个 5 个顶点的图，不同胚于 S^1，因此不是流形点。

---

二、如何扩展框架：非流形胞腔复形

要容纳这些结构，我们需要放弃"流形"假设，保留"复形"结构。以下是修改后的代数拓扑定义：

2.1 广义胞腔复形（Generalized Cell Complex）

一个非流形 B-Rep \mathcal{B} 是一个有限CW 复形（或更弱的胞腔复形），满足：

修改公理 2'：弱正则性
每个 k-胞腔 c 的内部 \mathring{c} \cong \mathbb{R}^k（开球），闭包 \bar{c} 通过特征映射 \phi_c: D^k \to \bar{c} 附着，但 \phi_c 在边界上的限制不必是嵌入（允许自交、重叠）。

关键区别：标准正则性要求 \phi_c|{\partial D^k} 是到其像的同胚；弱正则性允许它是商映射。这使得零厚度壁成为可能：两个 2-胞腔可以映射到 \mathbb{R}^3 中同一个几何集合，但作为拓扑对象不同。

2.2 修改公理 3'：非流形关联

对 Edge e \in C_1，定义其面度（Face degree）：
\deg(e) = \#\{f \in C_2 \mid [f:e] \neq 0\}

- 流形情况：\deg(e) = 2（标准 B-Rep）
- 边界 Edge：\deg(e) = 1（Solid 的外边界）
- T 型接头：\deg(e) \geq 3
- 悬挂 Edge：\deg(e) = 1 但 e 不在任何 Solid 边界上（非法但可能出现）

边界算子 \partial^2 = 0 仍然成立，但关联数 [f:e] 的取值和组合方式更灵活。

2.3 层化空间（Stratified Space）视角

非流形 B-Rep 可以看作一个层化空间：空间 X = |\mathcal{B}| 被分解为层（Strata）：
X = \bigsqcup{i} S_i

其中每层 S_i 是一个流形，且边界 \bar{S}i \setminus S_i 包含在更低维的层中。

非流形结构	层化解释	
零厚度壁	两个 2-维层 S_1, S_2 几何重合，但作为不同层存在	
T 型接头	1-维层 S_e（Edge）的闭包与三个 2-维层 S{f_1}, S{f_2}, S{f_3} 相交	
奇异 Vertex	0-维层 S_v 的链接是一个非圆图，其锥（Cone）构成 v 的邻域	

层化空间的同调理论比流形更复杂，需要交截同调（Intersection Homology）而非普通胞腔同调。

---

三、链群、形式语言下的非流形描述

3.1 链复形仍然适用，但系数环需要调整

对于非流形 B-Rep，链复形
C_2 \xrightarrow{\partial_2} C_1 \xrightarrow{\partial_1} C_0
仍然定义良好，但：

- \mathbb{Z} 系数：如果存在零厚度壁（两个 Face 定向相反但几何重合），\partial_2 的矩阵中可能出现线性相关的行（因为两个 Face 的边界链可能完全相同）。这导致同调群 H_2 出现非平凡的核，反映"拓扑上不同但几何上重合"的冗余结构。
- \mathbb{Z}2 系数：更鲁棒，因为忽略定向后，零厚度壁的两个 Face 在链群中成为同一个对象（如果它们边界相同），这反而丢失了"这是两层"的信息。

结论：非流形结构要求带系数的链复形，且系数的选择本身编码了"是否区分几何重合但拓扑不同的对象"。

3.2 上链与"形式"的奇异性

在奇异 Vertex v 处，0-上链（Vertex 特征）的"光滑性"崩溃。形式上，可以定义层化空间上的微分形式：不是 de Rham 复形 \Omega^(M)，而是交截形式复形 I\Omega^(X)，其中 X 是层化空间。

对深度学习而言，这意味着：
- 在奇异点（T 型接头、零厚度壁连接处），标准的局部特征聚合失效。
- 需要加权上链：给每个关联赋予一个"可信度权重" w(f,e) \in \mathbb{R}，使得 \sum{f} w(f,e) = 1。这对应于广义上边缘算子。

---

四、对深度学习的核心启示

4.1 当前 BRT 的局限性

论文明确假设 2-manifold boundary。这意味着：
- 在 TMCAD 等真实数据集上，如果存在非流形模型，BRT 的拓扑嵌入（Face-Shell Aggregation）会失败：一条 Edge 被 3 个 Face 共享时，"1-ring 邻域"的概念不再良定义。
- 零厚度壁会被错误地视为"两个独立但重合的 Face"，注意力机制可能在这两个 Face 之间建立虚假的长程依赖。

4.2 非流形感知的网络设计

如果扩展 BRT 到非流形，网络架构需要：

（1）可变关联数的聚合
当前 Loop-Face 聚合用 MLP 处理"一个外环 + 多个内环"。对于 T 型接头，需要处理一个 Edge 关联可变数量 Face：
E(e) \leftarrow \text{Aggregate}(\{E(f) \mid f \ni e\})
其中 Aggregate 必须是置换不变且基数无关的（如 DeepSet、Transformer Pooling）。

（2）层化注意力掩码
注意力权重不应仅基于"是否共享 Edge"（二值掩码），而应基于层化深度（Stratified Depth）：
- 同一层内的 Face（如零厚度壁的两面）获得特殊注意力模式。
- 跨层的 Face（如 T 型接头的三个面）需要三角/高阶注意力头。

（3）拓扑合法性检查的升级
生成任务中，\partial^2 = 0 是必要条件但不是充分条件。非流形 B-Rep 还需要：
- 链接条件（Link Condition）：每个 Vertex 的链接必须是一个连通图（或特定类型的图），而非任意复杂图。
- 可定向性条件：在 T 型接头处，三个 Face 的定向必须满足某种一致性（否则无法实体化）。

---

五、总结

问题	标准 B-Rep（流形）	扩展 B-Rep（非流形）	
代数对象	有限正则胞腔链复形	有限 CW/层化空间复形	
边界算子	\partial^2 = 0，关联数 \pm 1	\partial^2 = 0，关联数推广	
Edge 关联	恰好 2 个 Face	任意 \geq 1 个 Face	
Vertex 链接	S^1 的并	任意有限图	
同调理论	胞腔同调	交截同调 / 层化同调	
深度学习	图神经网络、序列 Transformer	高阶图神经网络、层化 Transformer	

> 最终回答：我前面给出的定义不能直接蕴含非流形结构，因为流形性公理是硬约束。但代数拓扑提供了扩展路径：通过层化空间理论和广义胞腔复形，可以建立一个包含零厚度壁、T 型接头和奇异点的统一框架。这要求放弃"正则附着"和"2-流形"假设，转而使用弱正则 CW 复形和交截同调。对深度学习而言，这意味着聚合函数必须从"固定邻域大小"升级为"置换不变的集合函数"，注意力机制需要引入层化深度作为偏置。这是一个尚未被充分探索的研究方向。