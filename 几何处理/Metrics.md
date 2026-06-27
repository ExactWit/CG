# IoU
交并比是图像分割、目标检测任务的常见指标：
$$IoU=\frac{\text{intersection}}{union}$$
像素IoU的运算对象是预测区域与真实区域，我们不妨回归本原地去理解这个指标，考虑任意元素集合的IoU。
给定一个类别集合$C$（$|C|\ge 2$），$|C|$-类别集合$S$含$|S|$个元素，记各类别元素数目为$N_c$，有$$\sum_{c\in C}N_c=|S|$$容易导出$S$上类别分布$P_c=\frac{N_c}{|S|}$
集合$S$中各类别的预测结果记为：
预测为$c$的$c$类元素数量：$TP_{c,S}$ ,
预测为$c$的非$c$类元素数量：$FN_{c,S}$ ，
预测为非$c$的$c$类元素数量：$FP_{c,S}$ ，
预测为非$c$的非$c$类元素数量：$TN_{c,S}$ ，
其中
$$
\begin{gather}
N_c=TP_{c,S}+FP_{c,S},\\
|S|=TP_{c,S}+FP_{c,S}+FN_{c,S}+TN_{c,S}
\end{gather}
$$
此时集合$S$上的$c$类预测表现为：$$Acc_{c,S}=\frac{TP_{c,S}}{N_c}$$
集合$S$上全类别情况为：$$Acc_{S}=\frac{\sum_{c\in C}TP_{c,S}}{|S|}=\sum_{c\in C}P_c\cdot Acc_{c,S}$$
而对于$IoU$则有：$$IoU_{c,S}=\frac{TP_{c,S}}{TP_{c,S}+FN_{c,S}+FP_{c,S}}$$
集合$S$上全类别$IoU$情况为：
$$
IoU_{S}=\frac{\sum_{c\in C}IoU_{c,S}}{|C|}
$$
>[!remark]
>由于$Acc$...显然判别器可以通过提高多数类的准确性来提升整体预测准确性，我们无法通过它看出


**假设判别器对任意样本$s$的预测结果都一致为$x$**，则$Acc_S=P_x$,$$IoU_S=\frac{IoU_{x,S}}{|C|}=\frac{N_x}{|C|\cdot|S|}=\frac{P_x}{|C|}$$
我们会发现，对于元素集合$S$，$IoU$和$Acc$都无法处理这种极端情形，
进一步地，我们考虑集族$\mathcal{S}:=\set{S_i}$，记$N_\mathcal{S}=\sum^{|\mathcal{S}|}_{i=1}|S_i|$
重新记$S_i$中$c$类元素数目为$N_{c,i}$，我们考虑$S_i$上类别分布$$P_{c,i}=\frac{N_{c,i}}{|S_i|}$$与总类别分布$$P_{c}=\frac{\sum_{i}N_{c,i}}{N_{\mathcal{S}}}$$