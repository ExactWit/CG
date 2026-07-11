 **BRep 几何诊断 + 网格验证 + 参数化分析**​ 的工具
## 一、3D Viewer 基础快捷键

|快捷键|功能|备注|
|---|---|---|
|`S`|Shaded 着色模式|默认，看光顺面|
|`W`|Wireframe 线框|**看三角 facets 必切**​|
|`H`|HLR 隐藏线|看轮廓|
|`B`|Black-box HLR|投影用|
|`F`|Fit All|重置视角|
|`Shift + 中键`|Pan||
|`滚轮`|Zoom||
|`Ctrl + 点击`|多选 sub-shape|选 Face / Edge / Vertex|

> 💡 排查三角化问题 **必用 `W`**，配合放大三圆柱交汇区看黑洞/针状三角。

---

## 二、Sub-shape 信息浏览（BRep 体检）

### 选一个 Face / Edge / Vertex 后看属性

- 右侧 **Property Panel**​ 自动刷新
    
- 关键信息：
    
    - **Surface 类型**：`Geom_BSplineSurface`/ `Geom_CylindricalSurface`/ `Geom_Plane`
        
    - **Natural bounds**：`true / false`（是否用了 Surface 原生参数域）
        
    - **U/V 范围**：`Min/Max U, Min/Max V`
        
    - **Degree / Num poles**：判断退化（如 `Degree V=1, Poles V=2`= 条带面）
        
    - **Address**：OCC TopoDS_Face 内存地址，比对是否同面复用
        
    

### Console 等价命令

```
# 打印整个 part 的 BRep 摘要
describe

# 打印某个 sub-shape 详情（你贴日志那种输出）
describe -shape 15

# 只看面
explode-faces
```

---

## 三、UV / Domain 视图（诊断参数化）

### 打开 Domain 视图

- 选中 Face → 右键 **Show Domain**（或工具栏图标）
    
- 画的是：**Surface 的 UV 方框 + 该 Face 上所有 Wire 的 PCurve 投影**
    

### Domain 里你能读到什么

- **外环 / 内环**：外环 CCW，内环 CW（理论规则）
    
- **Edge 朝向**：Forward（PCurve 顺 t）→ Reversed（PCurve 逆 t），AS 用箭头/颜色区分
    
- **跨 u=0 接缝**：圆柱面常见，Domain 里 PCurve 从 0 跳 2π
    
- **多叉交汇**：三圆柱交汇处三条 PCurve 挤在同一 UV 顶点 → 反向 Edge 多
    

### 常用联动

- `Show PCurve`：单独看某条 Edge 在该 Face 上的 2D 曲线
    
- `Show Curve`：看 Edge 的 3D 几何曲线（与 PCurve 对照，查参数化不连续）
    

---

## 四、三角化（Tessellation）相关

### 主动重剖：`Ctrl + T`

弹出对话框，两个核心参数：

|参数|含义|调参建议|
|---|---|---|
|**Linear deflection**​|网格 ↔ 原始曲面最大距离|三圆柱交汇区可降到 `5e-4 ~ 1e-4`，默认一般 `1e-3`|
|**Angular deflection**​|相邻三角法向最大夹角|条带面（Face 15 那种）可加大，减少针状|

底层调用的是 OCC 的 `BRepMesh_IncrementalMesh`。

### Inspect Mesh 面板

选中 Face → **Inspect → Mesh**​ 标签：

- `NbTriangles`：三角数
    
- `MinEdge / MaxEdge`：边长比，**> 50 基本就是 needle**
    
- `MinAngle`：最小内角，趋近 0° = 退化
    
- `MinArea`：最小三角面积，趋零就是"被丢弃的极小 patch"前身
    

### 退化检查命令

```
# 全 part 扫退化
check-degenerate

# 单 face
check-degenerate -face 15
```

会报：needle triangles / null-area / collapsed edges。

---

## 五、Explode（拆拓扑）与测量

|操作|用途|
|---|---|
|右键 Face → `Explode → Edges`|拆出该面所有边，查哪条是 Trim 边界|
|右键 Face → `Explode → Wires`|拆出 Outer/Inner Loop|
|右键 Edge → `Explode → Vertices`|看两端 Vertex，查"交点分裂"|
|`Measure`工具|测两 Vertex 距离，> 几何容差（~1e-7）就是撕裂|

> 三圆柱"交点数量不对"的验证路径：`Explode Edges`→ 交汇处数 Vertex 个数 → `Measure`相邻 Vertex 距离。

---

## 六、修复类功能（Repair）

|功能|场景|
|---|---|
|`Merge Vertices`|UV / 3D 里近似重合的顶点合并，容差可控|
|`Merge Edges`|共线的短 Edge 合并，减少 trim 边界段数|
|`Stitch Faces`|相邻面按边缝合，修三圆柱接缝微缝|
|`Fix Orientation`|Wire 方向乱了（你那个"方向不闭合"怀疑对象）|
|`ShapeFix_Wire`|强制修复 Wire 的 PCurve 连续性|

---

## 七、Tcl 控制台常用命令速查

```
# 加载 STEP
load-step model.stp

# 保存
save-step model_fixed.stp

# 浏览
describe
describe -shape 15
explode-faces

# 网格
mesh                          # 全 part 剖分（用 Ctrl+T 的当前参数）
mesh -face 14 -deflection 1e-4
describe -mesh                # 看 mesh 统计

# 退化
check-degenerate
check-degenerate -face 15

# 显示
show-domain 15                # 显示 Face 15 的 Domain
show-pcurve -edge 42 -face 15 # 显示 Edge 42 在 Face 15 上的 PCurve
```

> AS 的 Tcl 外壳很全，输入命令时按 `Tab`能补。

---

## 八、三圆柱相交场景的排查 SOP

按这个顺序走基本能定位根因：

1. **`describe`**​ 看所有 Face → 锁定交汇区的 ID（如 14/15）
    
2. **`show-domain <ID>`**​ → 看 UV 里是不是多叉交汇 + 反向 Edge 爆
    
3. **`explode-edges`**​ 交汇 Face → `measure`查顶点是否分裂（交点数量不对）
    
4. **`Ctrl+T`**​ 调小 linear deflection → `W`模式看黑洞补没补上
    
5. **Inspect Mesh**​ → 看 `MinEdge/MinAngle/MinArea`，对 `~12k 丢弃、~5k 退化`
    
6. **`check-degenerate -face <ID>`**​ → 量化 needle / null-area
    
7. **`Merge Vertices / Stitch Faces`**​ → 修复后重剖对比
    

