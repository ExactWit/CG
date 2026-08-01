---
date: 2026-07-22
---
关于AutoBrep详见笔记：[[AutoBrep]]

---
# 进度总览
base test
![[autobrep_eccv_base_test.png]]

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
![[006670.svg|637]]
![[autobrep_eccv_6670.png]]
pred
![[autobrep_eccv_6670pred.png]]
大体形态可以对得上，但是细节啥的有点多。目前三视图未拆分处理，复杂线元关系被压缩为单一向量，同时整体模型参数量大约1B，情理之中。
## 三视图拆分

pending中

---

# AutoBrep ECCV：训练/推理显存与耗时精算

> 口径：RTX 4090 D 24GB；方案 `eccv-base`（flat DXF）；`batch_size=2`，`accumulate_grad_batches=2`，`max_seq=3000`。  
> 数据来源：代码配置、`train`/`test` 日志、parquet tokenize 实测（2026-07-22）。

## 1. 精度

| 项 | 值 |
|----|-----|
| Lightning | `precision: bf16-mixed`（`configs/autobrep_eccv.yaml`） |
| 计算 | **bf16** |
| ckpt 主权重 | **fp32**（Lightning master weights） |
| Infer sampler | AR 另转 **fp16**（`transformer.to(..., dtype=float16)`） |
| 条件张量 | `images` / `prim_geom` / `face_ncs` / `edge_ncs` → `bfloat16` |

**结论：混合精度，不是全程 fp32。**

---

## 2. 模型与静态显存

| 项 | 数量 |
|----|------|
| 总参数 | ≈ **1.035B**（`dim=2048`，`depth=16`，`heads=32`） |
| 可训参数 | ViewEncoder ≈ **17.6M**（AR/FSQ 冻结） |
| 全量 bf16 权重 | ≈ **2.07 GB** |
| 全量 fp32 权重 | ≈ **4.14 GB** |

粗分解：

$$
\begin{aligned}
M_{\mathrm{frozen}} &\approx (1.035\mathrm{B}-17.6\mathrm{M})\times 2\,\mathrm{B}
  \approx 2.04\,\mathrm{GB}\quad(\mathrm{bf16}) \\
M_{\mathrm{train+Adam}} &\approx 17.6\mathrm{M}\times 4\,\mathrm{B}\times(1+2)
  \approx 0.21\,\mathrm{GB}\quad(\mathrm{fp32\ param}+m+v) \\
M_{\mathrm{static}} &\sim 2.2\text{–}4.5\,\mathrm{GB}
\end{aligned}
$$

训练日志稳态：`alloc ≈ 4.5 GB`，`peak ≈ 16.1 GB`（含碎片/工作区）。

---

## 3. Token 序列长度分布（train）

对 `processed/autobrep/train` 做 AR tokenize（跳过 RGB/DXF I/O），`pre_filter` 后 **n = 5174 / 6110**。

### 3.1 有效长度 $L_{\mathrm{eff}}$（pad 前；超长裁到 3000）

| 分位 | 0% | 5% | 10% | 25% | **50%** | 75% | 90% | 95% | 99% | 100% |
|------|----|----|-----|-----|---------|-----|-----|-----|-----|------|
| \(L_{\mathrm{eff}}\) | 149 | 384 | 494 | 759 | **1192** | 1865 | 2755 | 3000 | 3000 | 3000 |

| 统计              | 值              |
| --------------- | -------------- |
| mean / std      | **1381** / 787 |
| $\ge 500$       | 89.8%          |
| $\geq 1000$     | 59.9%          |
| $\geq 2000$     | 21.9%          |
| $\geq 3000$（顶满） | **7.9%**       |

### 3.2 几何规模（parquet 列）

| | mean | med | p90 | max |
|--|------|-----|-----|-----|
| `num_faces` | 52.6 | 43 | 101 | 200 |
| `num_edges` | 139.6 | 114 | 271 | 614 |

### 3.3 Padding

- 配置 **`max_seq = 3000`**
- 训练：**一律 `pad_neg` 到 \(L_{\mathrm{pad}}=3000\)**；若原始更长则随机裁窗再 pad
- 因此 attention/激活按 **满长 3000** 付费，尽管中位有效长仅 ~1192

Infer：逐步生成至 EOS，步数 \(\approx L_{\mathrm{gen}}\)（样本相关，上限 3000）；view 路径 **`cache_kv=False`**。

---

## 4. 显存公式（测量标定）

激活峰值（B=2，L=3000）：

$$
A_{\mathrm{peak}} \approx 16.1 - 4.5 = 11.6\,\mathrm{GB}
$$

结构标定：

$$
\mathrm{VRAM}(B,L) \approx M_0 + \kappa\cdot B\cdot L\cdot d\cdot n_{\mathrm{layer}}
$$

其中 $d=2048$，$n_{\mathrm{layer}}=16$，

$$
\kappa \approx \frac{11.6\times 10^9}{2\times 3000\times 2048\times 16} \approx 59\ \mathrm{bytes/token/layer}
$$

（与无 activation checkpoint 的 GPT 经验系数 ~48–64 一致。）

**实用式（本机标定）：**

$$
\boxed{
\mathrm{VRAM}(B,L)\approx 4.5 + 11.6\cdot\frac{B}{2}\cdot\frac{L}{3000}\ \mathrm{GB}
}
$$

| B | L | 预测 peak | 结论 |
|---|----|-----------|------|
| 1 | 3000 | ≈10.3 GB | 安全 |
| 2 | 3000 | ≈16.1 GB | 当前，接近红线 |
| 4 | 3000 | ≈27.7 GB | **>24 GB → OOM** |

**结论：batch 开不大，主因是 \(B\times L_{\mathrm{pad}}\) 激活，不是 1B 权重本身。**

---

## 5. 训练迭代与时间

| 量                      | 值                                    |
| ---------------------- | ------------------------------------ |
| parquet train          | 6110                                 |
| microbatch / epoch     | \(\lceil 6110/2\rceil =\) **3055**   |
| optimizer step / epoch | \(3055/2 \approx\) **1528**（accum=2） |
| 吞吐（日志）                 | ≈ **2.3–2.5 it/s**                   |
| **1 microbatch 墙钟**    | ≈ **0.40 s**（B=2，含 fwd+bwd）          |
| **1 epoch**            | ≈ **20–22 min**                      |
| **50 epoch CE**        | ≈ **17–18 h**（不含官方 STEP val）         |

关系：

$$
\text{microbatches/epoch}=\lceil N_{\mathrm{train}}/B\rceil,\quad
T_{\mathrm{epoch}}\approx \frac{\lceil N/B\rceil}{\mathrm{it/s}}
$$

### Forward / Backward（无单独计时，按 bwd≈2×fwd 拆）

| | 每 microbatch (B=2) | 摊到每条样本 |
|--|---------------------|--------------|
| Forward（估） | ≈ 0.13 s | ≈ 0.07 s |
| Backward（估） | ≈ 0.27 s | ≈ 0.13 s |
| **合计（测）** | **0.40 s** | **0.20 s** |

---

## 6. Infer / Official Test 耗时

流水线（小批交错，非「先全量 GPU 再全量 CPU」）：

```
for chunk in batches(gen_batch=2):
  ① GPU: 条件编码 + AR 自回归 token（主耗时；cache_kv=False）
  ② 逐样本: FSQ decode → joint_optimize → OCC 拟 BSpline → 写 STEP
全部写完后:
  ③ min_eval(pred vs gt) → summary / surf / edge / vert / topo
```

### 6.1 实测（`eccv-base-resume` official test 日志）

| 指标 | 值 |
|------|-----|
| 目标样本 | 348（test ∩ parquet ∩ GT） |
| 相邻成功 STEP 间隔 | mean **595 s**，median **254 s** |
| 吞吐 | ≈ **6 STEP/h**（~10 min/样本均墙钟） |
| `Write Done` / `decode_failed`（阶段性） | ~39 / ~30 |
| CUDACachingAllocator OOM | 大量告警；进程未崩，会重试变慢 |

同批两个 STEP 写完间隔常 **~1 s** → 单样本 **OCC+写盘多为秒～几十秒**；分钟级空白主要是 **AR ± OOM 重试**。

### 6.2 为何 infer ≫ train

| | Train CE | Infer → STEP |
|--|----------|--------------|
| Transformer | **1×** 长序列 fwd | **\(L_{\mathrm{gen}}\)×** 逐步 fwd（可至 \(10^3\)） |
| 另加 | — | FSQ + joint_opt + OCC 串行 |
| 每样本 | **~0.2 s** | **~10²–10³ s** |

粗算：\(L_{\mathrm{gen}}\sim 10^3\)、单步 fwd ~0.05–0.1 s（无高效 KV）→ 仅 AR 就 1–2 min，再加 OCC/失败/OOM → 与 ~10 min/样本一致。

---

## 7. OCC 角色（重建路径）

- 输入：FSQ 解码的面/边 **点网格**（+ 可选 `joint_optimize`）
- OCC：`GeomAPI_PointsToBSpline` / `PointsToBSplineSurface` **直接拟 BSpline**
- **无** plane/cylinder 等解析类型分类；基本按 BSpline 处理
- 与 AR **按小批交替**；重建在批内 **串行**

---

## 附录：关键路径

| 项                     | 路径                                                                                               |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| 精度/模型配置               | `configs/autobrep_eccv.yaml`                                                                     |
| 训练日志（it/s、alloc）      | `/data/hdd/exps/logs/AutoBrep_*_eccv-base-resume_*_train.log`                                    |
| Official test 日志      | `/data/hdd/exps/logs/AutoBrep_*_eccv-base-resume_260722-132547_test.log`                         |
| Mid-24 参考指标           | `.../eccv-base-resume__train/metrics/official_val_mid_best_repaired/metrics.json`（summary≈0.077） |
| $L_{\mathrm{eff}}$ 数组 | `/tmp/seq_lengths_train.npy`（本次统计生成）                                                             |


# ECCV 管线完整性对照（base / 3view / pc-cond）

更新时间：2026-07-27

图例：✅ 可用 · ⚠️ 有产物但不可信/不完整 · ❌ 缺失 · 🔄 进行中

数据与实验根目录：

| 域 | 路径 |
|----|------|
| 数据集 | `/data/hdd/datasets/eccv2026ws-cad-data` |
| 实验 runs | `/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep` |
| 产出 outputs | `/data/hdd/outputs/eccv2026ws-cad-data/gen/AutoBrep` |
| 预训练权重 | `/data/hdd/outputs/AutoBrep` |
| 代码仓库 | `/home/divisor/workspace/repo/AutoBrep` |

---

## 五阶段对照表

| 阶段 | 含义 | **base**（flat TechDraw） | **3view-geom** | **pc-cond** |
|------|------|---------------------------|----------------|-------------|
| **训练** | 50 epoch CE，有 ckpt | ✅ | ✅ | ❌ 无 ECCV run（仅有 ABC-1M 旧训） |
| **val** | 训练中 CE `val_loss` | ✅ 有 best/last | ✅ 有 best/last | — |
| **official val** | 训中/训后 STEP + 官方分 | ⚠️ mid/full 多为 0（旧 KV bug）；仅 mid-24 repaired 可用 | ❌ mid×3 + full 全 gen=0（同 KV bug，无 repaired） | — |
| **official test** | datasplit `test`，有 GT | ✅ | ✅ | — |
| **public infer** | `public_test`，无 GT | ✅ 有 submission.zip | 🔄 全量重跑中 | — |

---

## 各模型关键路径与数字

### base（entry `eccv-base`，tag `eccv-base-resume`）

| 阶段            | 状态  | Run 目录                                                                                              | 关键指标                                                                                                                                         |
| ------------- | --- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 训练            | ✅   | `/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/260721-194515/eccv-base-resume__train`        | best val_loss ≈ **0.2253**（step 18336）；`last.ckpt` + best ckpt                                                                               |
| val（CE）       | ✅   | 同上 `metrics/train_summary.json`、`tensorboard/`                                                      | 与 best 监控一致                                                                                                                                  |
| official val  | ⚠️  | 同上 `metrics/official_val_*`                                                                         | mid/full 定时评多为 **gen=0**；可用：`official_val_mid_best_repaired` → gen **13/24 (54.2%)**，summary **≈0.077**                                      |
| official test | ✅   | `/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/260722-132547/eccv-base-resume__test`         | 348：gen **107 (30.7%)**，summary **0.0506**，surf F1 0.0676；见 `metrics/test.json`                                                              |
| public infer  | ✅   | `/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/260724-191452/eccv-base-resume__public_infer` | 927：gen **285 (30.7%)**；zip：`/data/hdd/outputs/eccv2026ws-cad-data/gen/AutoBrep/260724-191452/eccv-base-resume__public_infer/submission.zip` |

### 3view-geom（entry `eccv-3view-geom`，tag `eccv-3view-geom-resume`）

| 阶段            | 状态  | Run 目录                                                                                                    | 关键指标                                                                            |
| ------------- | --- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 训练            | ✅   | `/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/260723-162838/eccv-3view-geom-resume__train`        | best val_loss ≈ **0.2253**（step 22920）；`last.ckpt` + best ckpt                  |
| val（CE）       | ✅   | 同上 `metrics/train_summary.json`、`tensorboard/`                                                            | 与 base 同量级                                                                      |
| official val  | ❌   | 同上 `metrics/official_val_mid_*`、`official_val_full_*`                                                     | mid×3 + full：**gen=0**（训中 KV bug）；**无** mid-24 repaired                         |
| official test | ✅   | `/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/260725-002218/eccv-3view-geom-resume__test`         | 348：gen **123 (35.3%)**，summary **0.0460**，surf F1 0.0645；见 `metrics/test.json` |
| public infer  | 🔄  | `/data/hdd/exps/runs/eccv2026ws-cad-data/gen/AutoBrep/260727-172702/eccv-3view-geom-resume__public_infer` | 全量重跑中（PATH 修复后）；历史：`260725-014123` 曾到 ~418/927 被主机挂死；尚无完整 submission            |

### pc-cond

| 说明 | 路径 |
|------|------|
| 不在本 ECCV 数据闭环 | 无 `/data/hdd/exps/runs/eccv2026ws-cad-data/...` 下的 train/test/public |
| 旧 ABC-1M 训练（参考） | `/data/hdd/exps/runs/abc-1m/gen/AutoBrep/260719-183113/pc-cond__train` |

若要对齐五阶段，需在 `eccv2026ws-cad-data` 上单独开训评。

---

## GT test 对比（可写报告）

| 模型 | Run | gen 成功率 | summary | surf F1 |
|------|-----|-----------|---------|---------|
| base | `.../260722-132547/eccv-base-resume__test` | 30.7% (107/348) | **0.0506** | 0.0676 |
| 3view | `.../260725-002218/eccv-3view-geom-resume__test` | **35.3%** (123/348) | 0.0460 | 0.0645 |

3view 合法率略高，官方 summary 略低；同量级。

---

## 缺口与补齐优先级

1. **等 3view public 跑完** → 两模型 public 才对称（当前 run：`260727-172702`）。
2. **（可选）KV 修复后补 official val**：base / 3view 各跑 mid-24（及可选 full），否则报告只能写 CE val + official test，没有可信的训中 STEP 曲线。
3. **3view 可补 mid-24 repaired**（对标 base `official_val_mid_best_repaired`）。
4. **pc-cond**：仅当需要第三条件分支时再在 ECCV 数据上立项。

---

## 工程修复备忘（影响评测可信度）

| 问题 | 影响 | 状态 |
|------|------|------|
| `gen_batch=2` OOM thrash | public/test 极慢或挂 | 已改默认 `gen_batch=1` |
| KV cache rotary / prepend | official val / public STEP 全 0 | 已修（`input_not_include_cache`） |
| `conda activate` 弄丢 `exp_launcher` | 空日志、~6s 假完成 | 已修（绝对路径 + filter 在 activate 外） |
| 主机硬挂 / CUDA 804 | 3view public 多次中断 | 已重提；需盯当前 run |

---

## 一句话

对「能交报告的主结果」：base 齐；3view 差完整 public，且两边都缺可信 official val STEP；CE 训练与 official test 两边都齐。
