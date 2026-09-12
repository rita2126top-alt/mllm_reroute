本文从 **mllm-reroute 的 skipped token 为什么会出现状态过时** 开始，逐步给出 **Reroute-Ghost** 的完整设计。每一个参数第一次出现时，说明：

```math
\boxed{\text{它表示什么}\quad+\quad\text{维度是多少}\quad+\quad\text{是不是可训练的}\quad+\quad\text{为什么需要它}}
```

本文的方法底座固定为上传项目 **Reroute: Recoverable Visual Token Routing**。原有 decision layers、attention scoring、Full-token Top-K、keep ratios、阶段内决策缓存、位置编码、compact attention、评测任务与基线全部保留。唯一新增的模型机制，是把原来的 skipped/deferred visual tokens 划分为 **Ghost / Cold**，对 Ghost 执行原稿中的低成本状态演化。

这里区分两类内容：**“项目原有”**来自上传压缩包中的代码与配置；**“本方案新增”**是本文确定的 Cold/Ghost 扩展，不能当作仓库已经实现或已经验证过的功能。上传项目没有 Ghost 训练脚本、训练清单、Ghost 权重或该扩展的实验结果，本文不虚构这些内容。

项目依据主要是 `models/router.py`、`models/patching.py`、`configs/experiment/`、`configs/model/`、`configs/eval/`、`scripts/` 与 `profiler/`。本文对原稿中不适用于阶段式路由的逐层重选、固定每层 64-token 预算、未来层标签和拼接流程作了相应替换；保留原稿的 Self Evolution、Residual Prototype、Ghost gate、Reactivation Prediction 与 Freshness Loss 主线。

---

# 1. 先不要设计模块，先把问题说透

假设一个 LVLM 的语言模型部分一共有：

```math
\boxed{L=\text{LLM decoder 的总层数}}
```

项目的 `configs/model/` 给定：

```math
L_{\mathrm{LLaVA}}=32,\qquad L_{\mathrm{Qwen}}=28.
```

全文 decoder layer index 使用项目代码的 **0-based 索引**：

```math
\boxed{l\in\{0,\ldots,L-1\}}
```

因此配置中的 `drop_layers: [2,7,15,23]` 就是代码中的索引 2、7、15、23，不能额外减 1。

---

图片经过原有视觉编码器与 multimodal projector 后，得到：

```math
\boxed{N=\text{该样本实际输入 LLM 的 visual token 总数}}
```

以原稿中的 LLaVA 单图设置为例：

```math
N=576.
```

Qwen 的配置保留 `max_pixels: 451584`，但它是图像预处理上限，不意味着每个样本的实际 `N` 都必须等于 576。实际 `N` 使用项目检测到的视觉区间长度。

第 `l` 层输入处，第 `i` 个 visual token 的 hidden state 写成：

```math
\boxed{v_i^l\in\mathbb R^D}
```

其中：

```math
i\in\{1,\ldots,N\},\qquad
\boxed{D=\text{当前骨干的 hidden dimension}}.
```

`D` 从实际模型读取，不为两个骨干共用一个硬编码值。原稿中的 `D=4096` 继续作为 LLaVA 的计算示例。

经过第 `l` 层，状态变成 `v_i^{l+1}`；因此 `v_i^L` 表示全部 decoder layers 之后的状态，而不是编号为 `L` 的层。

最后定义：

```math
\boxed{M=\text{该样本中所有非视觉 token 的数量}}
```

它包括项目始终保留的 system、文本及其他非视觉位置。完整输入长度为 `N+M`，不是只有 `N` 个视觉 token。

---

# 2. Reroute 在第 `l` 层干了什么？

先明确最重要的一点：

```math
\boxed{\text{Reroute 不是每层重新选 Top-K，而是在指定 decision layers 重新选择。}}
```

## 2.1 Decision layers 完全继承项目

定义：

```math
\boxed{\mathcal D=\{d_0,d_1,\ldots,d_{S-1}\}}
```

其中 `d_s` 是第 `s` 次决策所在的 decoder layer index，`S` 是决策次数。项目的 Reroute 配置均有 4 次决策。

| 模型 | 原项目 Reroute 配置族 | `drop_layers` |
| --- | --- | --- |
| LLaVA-1.5-7B | `ours_vs_fastv`，含其 stagewise 版本 | `[3,7,15,23]` |
| LLaVA-1.5-7B | `ours_vs_pdrop`，含其 stagewise 版本 | `[2,7,15,23]` |
| Qwen2.5-VL-7B | `ours_vs_fastv`，含其 stagewise 版本 | `[3,6,13,20]` |
| Qwen2.5-VL-7B | `ours_vs_pdrop`，含其 stagewise 版本 | `[2,6,13,20]` |

注意 `ours_vs_fastv` 虽然名字里有 FastV，项目为它配置的也是 **PDropRouter 的多阶段调度**，不是单次 FastVRouter。本文不更改这种安排。

## 2.2 Importance scoring 完全继承项目

仅在 `l=d_s` 时，使用项目 `_capture_attention_weights` 从该层输入计算 attention weights，再调用原来的 scorer：

```python
attn_avg = attn_weights.mean(dim=1)
scores = attn_avg[:, -1, vis_start:vis_end].float()
```

因此：

```math
\boxed{
s_i^{d_s}=\frac1{n_h}\sum_{h=1}^{n_h}
A^{d_s}_{h,\,\mathrm{last},\,\mathrm{vis}(i)}
}
```

这里 `n_h` 是 attention head 数；`A^{d_s}` 是项目捕获的 attention weights；`last` 是代码实际使用的**最后一个序列位置**，通常是最后的文本位置；`vis(i)` 是视觉 token `i` 的原序列位置。

` s_i^{d_s}` 是动态标量，不是可训练参数。保留项目的 Q/K projection、原始位置编码及 mask 处理，不额外加入 learned Full selector、额外打分层或新 attention aggregation。

## 2.3 Full-token Top-K 完全继承项目

定义：

```math
\boxed{\rho_s=\text{原配置中第 }s\text{ 阶段的 keep ratio}}
```

对于可恢复的 Reroute 路径，完整视觉索引空间一直保留，因此：

```math
\boxed{K_F^{(s)}=\min\!\left(N,\max\!\left(1,\lfloor N\rho_s\rfloor\right)\right)}
```

这里 `K_F^{(s)}` 是本阶段真正执行完整 decoder computation 的视觉 token 数，**不是所有层共享的固定常数**。

项目执行 `scores.topk(K_F)` 后，再按原视觉索引升序排列，恢复原序列顺序：

```math
\boxed{\mathcal A_{d_s}=\operatorname{TopK}(\{s_i^{d_s}\}_{i=1}^N,K_F^{(s)})}
```

对 Full 选择直接复用原 `PDropRouter.compute_scores`，包括其原有并列分数处理；本文不重写一套 Top-K。

保留：

```yaml
routing:
  method: pdrop
  monotonic: false
```

这意味着下一次决策时，候选集合仍然是全部 `N` 个视觉位置，包括之前的 Ghost 和 Cold，而不是只能从此前的 Active 集合中继续删减。

## 2.4 阶段内复用同一个 Active Set

定义当前阶段：

```math
\boxed{s(l)=\max\{s:d_s\le l\}}
```

那么：

```math
\boxed{
\mathcal A_l=\mathcal A_{d_{s(l)}},\qquad
\mathcal U_l=\{1,\ldots,N\}\setminus\mathcal A_l
}
```

`U_l` 是当前阶段被 skip 的视觉集合。非决策层复用已缓存的 `RoutingDecision`，不重新计算 Full importance，不重新选择 Full tokens。

在第一次决策之前：

```math
\boxed{\mathcal A_l=\{1,\ldots,N\},\quad\mathcal U_l=\varnothing,\qquad l<d_0.}
```

这些层继续 dense 计算。

## 2.5 保留全部原始预算

下面的 keep ratios 同时适用于两个模型，stagewise 与对应非 stagewise 配置完全一致。

| 原配置族 | `avg192` | `avg128` | `avg64` |
| --- | --- | --- | --- |
| `ours_vs_fastv` | `[0.2644,0.2644,0.2644,0.2644]` | `[0.1418,0.1418,0.1418,0.1418]` | `[0.01916,0.01916,0.01916,0.01916]` |
| `ours_vs_pdrop` | `[0.4502,0.4502,0.2251,0.1126]` | `[0.2655,0.2655,0.1328,0.0664]` | `[0.1392,0.1392,0.0696,0.0348]` |

例如 `N=576` 时，原配置产生：

```math
\begin{aligned}
\text{ours\_vs\_fastv, avg64}:&\quad[11,11,11,11],\\
\text{ours\_vs\_pdrop, avg64}:&\quad[80,80,40,20],\\
\text{ours\_vs\_fastv, avg128}:&\quad[81,81,81,81],\\
\text{ours\_vs\_pdrop, avg128}:&\quad[152,152,76,38].
\end{aligned}
```

因此：

```math
\boxed{\text{avg64 是原项目预算档位名称，不是“每层固定保留 64 个 token”。}}
```

本文直接保留 YAML 数值与取整方式，不根据档位名称重新反推或调整 Full 预算。

以上依据：`models/router.py`、`models/patching.py`、`configs/experiment/{llava15,qwen25vl}/`。

---

# 3. Active token 和 skipped token 的区别

对于：

```math
i\in\mathcal A_l,
```

Reroute 把它与所有非视觉 tokens 按原位置排序，gather 成 compact sequence：

```math
\boxed{X_{\mathrm{compact}}^l=\operatorname{Gather}(X^l,\mathcal A_l\cup\mathcal T)}
```

其中 `X^l` 是完整逻辑序列，`T` 是所有非视觉位置的索引集合。这里 `T` 不是新增输入，也不是新增可训练模块。

随后调用原 decoder layer：

```math
X_{\mathrm{compact}}^{l+1}=F_l(X_{\mathrm{compact}}^l).
```

`F_l` 包含原有 normalization、attention、MLP/FFN 与 residual connections，参数全部保持不变。

严格说，Full 分支 attention 的序列长度是：

```math
\boxed{M+K_F^{(s(l))}}
```

不是只有 `K_F`，因为文本和 system tokens 始终参加计算。

---

对于：

```math
i\in\mathcal U_l,
```

原项目 `compact_route` 使用输入状态初始化输出，只覆盖 compact 分支对应的位置。因此 skipped token 严格执行：

```math
\boxed{v_i^{l+1}=v_i^l.}
```

`compact_route_stagewise` 把这些 token 存到 `_stage_deferred_hidden`，阶段内不送入原 decoder layer。到下一决策层，再与最新 compact states 一起恢复完整逻辑序列。

这两条路径的关键语义相同：

> 不删除 skipped token；不让它进入当前 Full attention/FFN；保留它在未来决策层重新入选的资格。

本文就在这个 **deferred-state 更新位置** 插入 Ghost。不会把 Ghost 重新塞进 Full attention，也不会让 Ghost 写入该层原有 KV cache。

依据：`models/patching.py::_forward_compact_route`、`_forward_compact_route_stagewise`、`_forward_compact_in_stage`。

---

# 4. 问题到底发生在哪里？

以 LLaVA 的原始决策层 `[3,7,15,23]` 为例。

假设 token `i` 在决策层 7 没有入选：

```math
i\in\mathcal U_7.
```

由于阶段内不重新选择，在层 7 到层 14，它一直被 skip：

```math
\boxed{v_i^{15}=v_i^{14}=\cdots=v_i^7.}
```

与此同时，持续 active 的 token 已经经历了本阶段全部 8 个 decoder layers。

如果原 Reroute scorer 在下一决策层 15 又选中了 `i`：

```math
i\in\mathcal A_{15},
```

它就会携带**层 7 输入处的旧状态**进入层 15 的完整计算。

这就是本文要研究的问题：

```math
\boxed{\text{Skipped-State Staleness：被跳过 token 的状态未随网络深度及时演化。}}
```

但需要区分：代码可以直接确认 identity bypass 导致状态停止更新；它是否造成任务性能损失、误差是否随 skip age 单调增大，仍然需要实验测量，不能预先当作结果。

与原稿的逐层路由例子不同，这里 token 不会在非决策层 8、9、10 自行重激活。重激活只能发生在项目已有的下一次 decision layer。

---

# 5. 我们真正想预测的量是什么？

先考虑完全不压缩的原始模型。

假设第 `i` 个 token 在第 `l` 层真的执行完整 Transformer。

进入层之前：

```math
v_i^l.
```

经过 Transformer 后：

```math
v_i^{l+1}.
```

那么这一层对这个 token 实际施加的变化就是：

```math
\boxed{ \Delta_i^l = v_i^{l+1}-v_i^l }
```

这里：

```math
\boxed{ \Delta_i^l\in\mathbb R^D }
```

叫做：

> **Layer-wise State Evolution Residual**

或者简单理解成：

> “这一层本来应该把这个 token 改多少。”

例如：

```math
v_i^l = \begin{bmatrix} 0.2\\ 0.4\\ -0.1\\ \vdots \end{bmatrix}
```

经过 Transformer 后：

```math
v_i^{l+1} = \begin{bmatrix} 0.25\\ 0.37\\ 0.05\\ \vdots \end{bmatrix}.
```

那么：

```math
\Delta_i^l = \begin{bmatrix} 0.05\\ -0.03\\ 0.15\\ \vdots \end{bmatrix}.
```

---

在 identity bypass 中，Reroute 对 skipped token 施加的实际状态增量为：

```math
\boxed{ \Delta_i^{l,\mathrm{bypass}}=0 }
```

因为：

```math
v_i^{l+1}=v_i^l.
```

而我们的核心假设是：

```math
\boxed{ \text{token 当前不重要} \not\Rightarrow \Delta_i^l=0 }
```

也就是说：

> 一个 token 当前不值得花完整 Transformer 计算，不代表它的 representation 就完全不应该随 depth 演化。

---

# 6. 所以我们的真正目标变得非常明确

对于 skipped token，不重新运行完整 decoder，而是用低成本 predictor 得到：

```math
\boxed{\widehat\Delta_i^l\in\mathbb R^D.}
```

帽子表示预测值。训练时使用同一骨干的未路由 dense teacher，提供：

```math
\boxed{\Delta_i^{l,*}=v_i^{l+1,*}-v_i^{l,*}.}
```

这里 `*` 代表 dense teacher 输出，不代表真实数据集人工标注，也不代表 teacher 具有额外模型参数。

对需要更新的 skipped token，目标是学习：

```math
\widehat\Delta_i^l\approx\Delta_i^{l,*},
```

并执行：

```math
\boxed{v_i^{l+1}=v_i^l+\widehat\Delta_i^l.}
```

但不是给所有 skipped tokens 都做这一步。本文最终固定为：

```math
\boxed{\text{Ghost：cheap residual update；Cold：identity bypass。}}
```

Ghost 和 Cold 都属于 Reroute 的 skipped 集合；它们都没有新增的 Full 计算资格。哪些 token 能重新进入 Full，仍然只由下一次原 Reroute Top-K 决定。

此外，dense teacher 与 compact student 使用的 attention 上下文不同，因此 dense-state 偏差不全是“skip 太久”造成的。单步 residual 拟合与重激活时的 freshness 监督必须结合使用，不能把一个近似更新等同于恢复完整 dense 轨迹。

---

# 7. 现在问题变成：怎样便宜地预测 `\widehat{\Delta}_i^l`？

一个只看 token 自身的 predictor 写成：

```math
\widehat\Delta_i^l=f(v_i^l).
```

但完整 decoder 的状态变化还受 attention 中可见上下文影响：

```math
\boxed{\Delta_i^l=f_l(v_i^l,\mathcal V_{\mathrm{visible}}^l,\mathcal T_{\mathrm{visible}}^l).}
```

这里 `V_visible` 和 `T_visible` 表示原注意力规则下实际可见的视觉与非视觉状态，都是动态输入，不是新参数。

对 Reroute 的 Full 分支，可见视觉上下文又受当前 Active Set 限制。因此只根据 `v_i^l` 预测，并不能利用“当前这一层实际计算出来了哪些变化”。

本文固定保留原稿的两部分设计：token-local Self Evolution，以及利用 active residual 的 Context Evolution。二者组合是待验证的建模假设，不预先宣称一定优于简单 Adapter。

---

# 8. 把 `\widehat{\Delta}` 分成两个来源

先定义未加 gate 的两个增量分量：

```math
\boxed{\widehat\Delta_{i,\mathrm{self}}^l=\text{根据 token 自身预测的变化}}
```

以及：

```math
\boxed{\widehat\Delta_{i,\mathrm{ctx}}^l=\text{根据当前 active residual 预测的变化}.}
```

二者都属于 `R^D`。

核心分解为：

```math
\boxed{\text{状态更新候选}=\text{自身演化}+\text{上下文驱动演化}.}
```

第 37 节给出固定使用的 gate；正式方案始终对两个分量之和施加 gate。

---

# 9. 第一部分：Self Evolution Predictor

先只看 token 本身。

输入：

```math
v_i^l\in\mathbb R^D.
```

因为：

```math
D=4096
```

很大，我们不希望再跑一个 `4096\rightarrow4096` 的大 MLP。

所以先降维。

定义：

```math
\boxed{ W_{\mathrm{down}}^l \in \mathbb R^{d_b\times D} }
```

这里：

```math
\boxed{ d_b=\text{Ghost predictor 的 bottleneck dimension} }
```

固定满足：

```math
d_b\ll D.
```

本文的 LLaVA 维度示例：

```math
D=4096, \qquad d_b=32.
```

---

那么：

```math
W_{\mathrm{down}}^l
```

的作用就是：

```math
\boxed{ 4096\text{ 维} \rightarrow 32\text{ 维} }
```

得到：

```math
\boxed{ z_i^l = W_{\mathrm{down}}^l \operatorname{LN}(v_i^l) }
```

其中：

```math
z_i^l\in\mathbb R^{d_b}.
```

这里：

```math
\operatorname{LN}
```

是本文沿用的归一化简写；实现时直接使用该层原有的 `input_layernorm`，并冻结其参数，不额外插入一个与原模型不同的归一化层。

所有新增 bottleneck 的维度固定为 `d_b=32`，所有大投影均不使用 bias。上标 `l` 表示“第 `l` 层使用的参数”，共享规则在第 45 节统一确定。

为什么先 LN？

因为不同 layer、不同 token 的 hidden-state magnitude 差异可能比较大；先归一化后，小 predictor 更容易学习稳定的 state transition。

---

然后经过一个激活函数：

```math
\phi(\cdot).
```

固定使用：

```math
\phi=\operatorname{SiLU}.
```

于是：

```math
\phi(z_i^l)\in\mathbb R^{d_b}.
```

---

接下来再升维回 `D`：

```math
\boxed{ W_{\mathrm{up}}^l \in \mathbb R^{D\times d_b} }
```

因此：

```math
\boxed{ \widehat{\Delta}_{i,\mathrm{self}}^l = W_{\mathrm{up}}^l \phi(z_i^l) }
```

维度重新变成：

```math
\widehat{\Delta}_{i,\mathrm{self}}^l \in \mathbb R^D.
```

整个过程：

```math
\boxed{ D \rightarrow d_b \rightarrow D }
```

例如：

```math
4096 \rightarrow 32 \rightarrow 4096.
```

---

# 10. 这两个矩阵到底学什么？

```math
W_{\mathrm{down}}^l
```

和：

```math
W_{\mathrm{up}}^l
```

都是：

```math
\boxed{\text{可训练参数}}
```

原始 LVLM 参数全部 frozen；不对视觉编码器、projector 或 LLM 做微调。

它们的目标不是重新模拟完整 Transformer。

它们只学习：

> 如果一个具有这种 hidden representation 的 visual token 在 layer `l` 继续向前演化，它大概率会朝哪个方向发生变化？

所以：

```math
\widehat{\Delta}_{i,\mathrm{self}}^l
```

主要建模的是：

```math
\boxed{ \text{token-local state evolution} }
```

---

# 11. 但到这里仍然不够

对于同一 token-local representation，其他 active tokens 的状态不同，当前层观察到的实际 residual pattern 也可能不同。

所以：

```math
\widehat\Delta_{i,\mathrm{self}}^l
```

只能提供 token-local 的状态演化预测，不能直接读取当前样本这一层的其他实际状态变化。

本文增加：

```math
\boxed{\widehat\Delta_{i,\mathrm{ctx}}^l}
```

来利用本层已经算出的 active residual。

这里必须修正一个容易误解的地方：原 decoder 的 causal attention 不能让前面的视觉 token 任意读取后面的用户问题。因此不能把所有 visual residual 都说成“直接融合了整段问题”。Reroute 的最后位置 attention score 可以依赖问题，进而影响 Active Set；但 active visual state 能读取哪些文本，仍由原模型的序列顺序与 attention mask 决定。

本文不改变原 decoder 的这种可见性，不使用未来生成答案作为 Ghost 输入。

---

# 12. 这里出现我们方案最关键的想法：借用 Active Token 的真实变化

当前层我们虽然跳过很多 token，但是：

```math
K_F^{(s(l))}
```

个 active visual tokens 仍然真的执行了原项目的 compact decoder layer。

对于某个 active token `j`：

```math
j\in\mathcal A_l.
```

它输入：

```math
v_j^l
```

输出：

```math
v_j^{l+1}.
```

因此我们可以直接得到：

```math
\boxed{ d_j^l = v_j^{l+1}-v_j^l }
```

注意这里我使用：

```math
d_j^l
```

而不是：

```math
\Delta_j^l
```

只是为了区分：

```math
d_j^l=\text{当前实际观察到的 active-token residual}
```

而：

```math
\widehat{\Delta}_i^l=\text{我们正在预测的 skipped-token residual}.
```

---

这里最重要的是：

```math
\boxed{ d_j^l \text{几乎不需要额外 Transformer 计算} }
```

因为：

```math
v_j^l
```

和：

```math
v_j^{l+1}
```

本来就已经存在。

只做一次向量减法：

```math
v_j^{l+1}-v_j^l.
```

---

# 13. 为什么 active token 的 `d_j^l` 能帮助 skipped token？

对于本层 active visual token：

```math
\boxed{d_j^l=v_j^{l+1}-v_j^l,\qquad j\in\mathcal A_l,}
```

我们观察到的是当前 compact Reroute 分支的实际变化，而不是额外运行 dense teacher 得到的变化。

它携带的信息包括：

```math
\boxed{\text{当前样本的视觉内容}+\text{当前 Active Set}+\text{当前层的实际变换}.}
```

因此，在 active 与 skipped tokens 的状态演化具有共享结构时，这些 residual 能为 skipped-token predictor 提供样本相关的线索。

这是一项需要检验的假设，不保证一个 token 的 residual 总能从其他 token 的 residual 推断出来。

本文只用 **active visual residuals** 建立 Prototype Bank，不混入文本 residual、dense teacher 的未来特征或答案 token。这样不会把训练时才有的信息引入推理。

---

# 14. 固定使用 Residual Prototype Bank 压缩 active residual

当前层有：

```math
\{d_j^l:j\in\mathcal A_l\},\qquad d_j^l\in\mathbb R^D.
```

`|A_l|` 由原 keep ratios 决定，随阶段与样本变化。本文不要求它等于 64。

本方案固定把这些 residual 聚合成少量低维 prototypes，再由 Ghost tokens 读取：

```math
\boxed{\{d_j^l\}_{j\in\mathcal A_l}\longrightarrow\{p_m^l\}_{m=1}^{M_p}.}
```

这里：

```math
\boxed{M_p=8}
```

是固定超参数。它不增加 Full token 的数量，也不把这些 prototype 插入原 LLM token 序列。

Prototype Bank 仅在本层需要 Ghost update 时构造，随着层和样本重新生成，不跨样本缓存。

---

# 15. 什么是 Residual Prototype？

Residual Prototype 是若干 active-token residual 的**低维、可学习加权汇总**。

例如，同一物体的多个 active patch 在某一层可能呈现相近的状态变化。若这种相关性实际存在，就有机会用 8 个低维汇总描述主要变化模式，而不让每个 Ghost token逐个读取全部高维 residual。

这里：

```math
\boxed{M_p=8,\qquad p_m^l\in\mathbb R^{d_b},\qquad d_b=32.}
```

`M_p`、`d_b` 是固定超参数；`p_m^l` 是本次前向动态生成的 feature。

即使某个输入的 `K_F` 小于 8，仍保留 8 个 learnable aggregation queries；它们可以形成相关甚至重复的汇总，不强制 8 个 prototype 彼此独立。原 Top-K 至少保留一个视觉 token，因此本方案正常路由时不会出现空 Active Set。

“存在低维共享结构”是实验假设，不把这些 prototype 预先命名为人工定义的语义类别。

---

# 16. Prototype 首先也要降维

原来的 active residual：

```math
\boxed{d_j^l\in\mathbb R^D}
```

与 LLM hidden state 一样是高维向量。本文固定先把它压缩到 `d_b=32` 维。

定义可训练矩阵：

```math
\boxed{W_R^l\in\mathbb R^{d_b\times D}.}
```

它的作用是：

```math
\boxed{\text{active token 的真实 compact residual}\quad D\longrightarrow d_b.}
```

得到：

```math
\boxed{c_j^l=W_R^l d_j^l,\qquad c_j^l\in\mathbb R^{d_b}.}
```

这里 `c_j^l` 叫做 **compressed residual code**，是动态 feature；`W_R^l` 是可训练参数。本文统一使用 `c_j^l`，把 `r_i` 专门留给后面的 reactivation probability，避免同一字母表示两种不同含义。

推理时的 `d_j^l` 来自当前 Reroute Full 分支，不调用 teacher，也不额外执行一次完整 decoder。

---

# 17. 怎样把本阶段的 active residual codes 变成 8 个 prototype？

我们给每个 prototype 一个 learnable query。

第 `m` 个 prototype query：

```math
\boxed{ q_m^{P} \in \mathbb R^{d_b} }
```

其中：

```math
m\in\{1,\ldots,M_p\}.
```

这里上标 `P` 表示：

```math
P=\text{Prototype}.
```

这些：

```math
q_1^P,\ldots,q_{M_p}^P
```

都是：

```math
\boxed{\text{可训练参数}}
```

它们可以理解成：

> 第 `m` 个 prototype 在学习“我应该从 active tokens 中收集哪一种变化模式”。

---

同时把 active token 的输入 state 投影成 key：

```math
\boxed{ k_j^P = W_K^P \operatorname{LN}(v_j^l) }
```

其中：

```math
W_K^P \in \mathbb R^{d_b\times D}.
```

作用是：

```math
D\rightarrow d_b.
```

---

然后 prototype `m` 对 active token `j` 的相关性：

```math
e_{mj}^l = \frac{ (q_m^P)^\top k_j^P }{ \sqrt{d_b} }.
```

这里：

```math
e_{mj}^l
```

是一个标量。

含义：

```math
\boxed{ \text{active token }j \text{ 的变化有多适合进入 prototype }m }
```

---

对所有 active token 做 softmax：

```math
\boxed{ \beta_{mj}^l = \frac{ \exp(e_{mj}^l) }{ \sum_{j'\in\mathcal A_l} \exp(e_{mj'}^l) } }
```

所以：

```math
\sum_{j\in\mathcal A_l}\beta_{mj}^l=1.
```

---

最终：

```math
\boxed{ p_m^l = \sum_{j\in\mathcal A_l} \beta_{mj}^l c_j^l }
```

得到：

```math
p_m^l\in\mathbb R^{d_b}.
```

这里：

```math
\boxed{ p_m^l = \text{第 }l\text{ 层第 }m\text{ 个 residual prototype} }
```

它不是可训练参数本身。

它是：

```math
\boxed{ \text{当前输入样本动态生成的 feature} }
```

这一点非常重要。

---

# 18. 现在拥有了什么？

当前阶段原有数量的 active residuals：

```math
\{d_j^l\in\mathbb R^D:j\in\mathcal A_l\}
```

被聚合为：

```math
\boxed{P_l=\{p_1^l,\ldots,p_8^l\},\qquad p_m^l\in\mathbb R^{32}.}
```

`P_l` 是当前层动态 residual prototype bank，不是一个额外的固定模型权重，也不是 8 个进入 Full attention 的新 token。

它描述当前层观察到的若干低维变化模式。具体学到什么，需要通过实验分析；不预先声称某个 prototype 一定对应“物体”“文字”或“空间关系”。

---

# 19. 接下来 skipped token 怎样读取这些 prototypes？

第 33—34 节会确定 Ghost 集合 `G_l`；这里先说明已进入 Ghost 集合的 token 如何读取 prototypes。现在考虑：

```math
i\in\mathcal G_l\subseteq\mathcal U_l.
```

它已经拥有：

```math
v_i^l\in\mathbb R^D.
```

首先生成一个小 query：

```math
\boxed{ q_i^G = W_Q^G \operatorname{LN}(v_i^l) }
```

其中：

```math
W_Q^G \in \mathbb R^{d_b\times D}.
```

上标：

```math
G=\text{Ghost branch}.
```

所以：

```math
q_i^G\in\mathbb R^{d_b}.
```

含义：

> skipped token `i` 当前需要什么类型的 state evolution？

---

每个 prototype 生成 key：

```math
\boxed{ k_m^G = W_K^G p_m^l }
```

其中：

```math
W_K^G \in \mathbb R^{d_b\times d_b}.
```

这里是一个非常小的矩阵。

---

同时生成 value：

```math
\boxed{ u_m^G = W_V^G p_m^l }
```

其中：

```math
W_V^G \in \mathbb R^{d_b\times d_b}.
```

于是：

```math
k_m^G,u_m^G \in \mathbb R^{d_b}.
```

---

# 20. Ghost Attention

第 `i` 个 Ghost token 对第 `m` 个 prototype 的匹配程度：

```math
e_{im}^{G,l} = \frac{ (q_i^G)^\top k_m^G }{ \sqrt{d_b} }.
```

然后：

```math
\boxed{ \alphe_{im}^{G,l} = \operatorname{softmax}_m(e_{im}^{G,l}) }
```

满足：

```math
\sum_{m=1}^{M_p} \alphe_{im}^{G,l} = 1.
```

这里：

```math
\boxed{ \alphe_{im}^{G,l} = \text{token }i\text{ 应该借用 prototype }m \text{ 多大比例} }
```

然后得到 context evolution code：

```math
\boxed{ h_{i,\mathrm{ctx}}^l = \sum_{m=1}^{M_p} \alphe_{im}^{G,l} u_m^G }
```

维度：

```math
h_{i,\mathrm{ctx}}^l \in \mathbb R^{d_b}.
```

---

# 21. 再把 context code 升回原 hidden dimension

定义：

```math
\boxed{ W_{\mathrm{ctx}}^l \in \mathbb R^{D\times d_b} }
```

它是可训练参数。

然后：

```math
\boxed{ \widehat{\Delta}_{i,\mathrm{ctx}}^l = W_{\mathrm{ctx}}^l h_{i,\mathrm{ctx}}^l }
```

于是：

```math
\widehat{\Delta}_{i,\mathrm{ctx}}^l \in \mathbb R^D.
```

这就是：

> 根据当前真正 active token 已经发生的状态变化，估计 skipped token 应该发生的 context-dependent 变化。

---

# 22. 至此我们的核心 Predictor 已经完整了

Self branch：

```math
\boxed{\widehat\Delta_{i,\mathrm{self}}^l
=W_{\mathrm{up}}^l\operatorname{SiLU}\!\left(W_{\mathrm{down}}^l\operatorname{LN}_l(v_i^l)\right).}
```

Context branch：

```math
\boxed{\widehat\Delta_{i,\mathrm{ctx}}^l
=W_{\mathrm{ctx}}^l\sum_{m=1}^{M_p}\alpha_{im}^l u_m^G.}
```

二者均为 `D` 维向量。第 37 节确定的 scalar gate 为：

```math
\boxed{g_i^l\in(0,1).}
```

最终固定采用：

```math
\boxed{\widehat\Delta_i^l
=g_i^l\left(\widehat\Delta_{i,\mathrm{self}}^l+\widehat\Delta_{i,\mathrm{ctx}}^l\right).}
```

仅对 `i∈G_l` 执行：

```math
\boxed{v_i^{l+1}=v_i^l+\widehat\Delta_i^l.}
```

后文会逐一确定：谁进入 Ghost、什么时候决定、如何训练、写回哪个缓存，以及如何保持原实验设置。

---

# 23. Residual Transport 改进的是哪一部分？

Local-only Adapter 的输入是：

```math
\widehat\Delta_i^l=f(v_i^l).
```

本文 predictor 的输入增加了本层实际观察到的 residual samples：

```math
\boxed{\widehat\Delta_i^l
=f\!\left(v_i^l,\{d_j^l\}_{j\in\mathcal A_l}\right).}
```

其中：

```math
d_j^l=v_j^{l+1}-v_j^l.
```

因此它不是重新决定 Full token 的重要性，而是在 **Full 集合已由 Reroute 决定且已完成当前层计算之后**，借用这次计算提供的变化模式，预测 Ghost token 的状态增量。

严格的执行顺序为：

```math
\boxed{\text{原 Reroute 决策}\rightarrow\text{原 Full 分支}\rightarrow\text{真实 active residual}\rightarrow\text{Ghost 更新}.}
```

同一层的 Ghost 输出不会反过来改变已经完成的 Full attention。这保证新增机制集中在 skipped-state evolution，而不是把原路由算法改造成另一种 attention 模型。

---

# 24. 从数学角度，它更接近学习一个 transformation field

把网络深度 `l` 看作离散时间。

Transformer residual network：

```math
v_i^{l+1} = v_i^l + \Delta_i^l.
```

类似：

```math
x(t+\Delta t) = x(t) + f(x,t)\Delta t.
```

因此：

```math
\Delta_i^l
```

可以理解为：

```math
\boxed{ \text{representation 在 depth direction 上的 velocity} }
```

Active token 真正观察到了：

```math
d_j^l.
```

我们利用：

```math
\{d_j^l\}
```

去估计 skipped token 所在位置的：

```math
\widehat{\Delta}_i^l.
```

所以这个方法可以定义为：

```math
\boxed{ \textbf{Residual Field Interpolation} }
```

而不是普通：

```math
\text{feature reconstruction}.
```

这里是对 residual network 的离散深度类比，不构成误差收敛或性能提升的理论保证。

---

# 25. 到这里出现第二个问题：需要给所有 skipped token 补吗？

不需要。Reroute 允许 token 在后续 decision layer 回到 Full，但并非所有 skipped token 都一定会回来。

本文把有限的状态补偿预算集中到：

```math
\boxed{\text{当前被 skip、且在下一次原 decision layer 更可能重新入选的 token。}}
```

因此，保留原稿的 **Reactivation-Aware Ghost Routing**，但把它严格适配到 Reroute 的阶段式决策。

它只决定：

```math
\boxed{\mathcal U_l\longrightarrow\mathcal G_l\cup\mathcal C_l.}
```

它不决定 `A_l`，不增加 Full budget，也不允许任何 token 在非决策层自行变成 Full。

还有一个可以从执行路径直接得到的边界：最后一次决策之后，没有下一次重选；剩余 skipped tokens 又不进入原 attention/KV。因此本方案在最后一个阶段不再做 Ghost update，剩余 skipped tokens 全部为 Cold。

---

# 26. 先定义“重激活”

在 Reroute 中，重激活只在决策层发生。

如果 token `i` 在上一阶段被 skip，但在下一决策层 `d_{s+1}` 被原 Top-K 选中：

```math
\boxed{i\notin\mathcal A_{d_s},\qquad i\in\mathcal A_{d_{s+1}},}
```

就称为：

```math
\boxed{\text{token }i\text{ 在 }d_{s+1}\text{ 层发生 reactivation。}}
```

例如 LLaVA 的 `[3,7,15,23]` 调度中：

```math
i\notin\mathcal A_7,\qquad i\in\mathcal A_{15}.
```

它在 7—14 层连续跳过 Full computation，在 15 层重新进入 Full。

即使某层被 Ghost 更新，它也仍属于 skipped token，因为它没有执行那一层的完整 decoder。因此：

```math
\boxed{\text{Ghost}\ne\text{Full；Ghost update}\ne\text{reactivation。}}
```

---

# 27. 定义 Future Horizon：固定为下一次 decision layer

本文固定：

```math
\boxed{H_{\mathrm{dec}}=1.}
```

它表示预测未来 **1 次决策**，不是未来 1 个 decoder layer，也不是原稿中的固定未来 4 层。

在阶段 `s`，目标层就是：

```math
\boxed{d_{s+1}.}
```

例如 LLaVA 的 `[3,7,15,23]` 对应的有效预测目标依次是：

```math
3\rightarrow7,\qquad7\rightarrow15,\qquad15\rightarrow23.
```

从阶段入口到下一次决策的距离为：

```math
\boxed{\delta_s=d_{s+1}-d_s,\qquad s<S-1.}
```

`delta_s` 是根据原 schedule 直接计算的整数，不可训练。它进入 Ghost predictor 的辅助输入，使共享参数知道“这次补偿要跨越多长的阶段”。

最后一次决策 `d_{S-1}` 没有下一目标层，因此不运行 reactivation predictor、不产生 Ghost tokens、不构造新 Prototype Bank。原 Full 分支仍照常运行到模型末层。

---

# 28. Reactivation label 怎么产生？

标签只用于训练，不在推理时查看未来。

对阶段 `s<S-1` 的 skipped token，固定标签为：

```math
\boxed{y_i^{s,\mathrm{react}}
=\mathbf1[i\in\operatorname{stopgrad}(\mathcal A_{d_{s+1}}^{\mathrm{trace}})],
\qquad i\in\mathcal U_{d_s}.}
```

其中 `1[·]` 是 indicator，条件成立取 1，否则取 0。`trace` 是同一训练样本实际运行得到的 Reroute 路由轨迹，不是随意构造的 dense Top-K oracle。

训练分为两个顺序阶段，标签来源固定如下。

**Warm-up 阶段：**先运行原始 identity-bypass Reroute，记录其原 scorer 和原 schedule 产生的 Active Sets；以该轨迹下一决策层的选择作为标签。

**Rollout 阶段：**运行当前 Reroute-Ghost student，在完整前向结束后读取它实际的下一决策层 Active Set，再构造 detached 标签。不给 student 强制套上 warm-up 的旧 Active Sets。

这不是推理时的未来信息泄漏：

```math
\boxed{\text{未来选择只作为监督标签，不作为当前 Ghost predictor 的输入。}}
```

Dense teacher 负责提供 `v_i^{l,*}` 和 `Delta_i^{l,*}`，不负责替换原 Full router。这避免把“沿用 Reroute 的路由机制”偷换成“用 dense teacher 的新路由方案”。

---

# 29. Predictor 输入哪些信息？

Reactivation predictor 只在原决策层 `l=d_s`、且 `s<S-1` 时运行。

第一个信息是原 Full router 已经产生的 score：

```math
\boxed{s_i^{d_s}\in\mathbb R.}
```

第二个信息是 token 与 Full Top-K 边界的距离：

```math
\boxed{m_i^{d_s}=s_i^{d_s}-\tau_s,\qquad
\tau_s=\min_{j\in\mathcal A_{d_s}}s_j^{d_s}.}
```

`tau_s` 是本次已选中 Full token 的最低分，标量、不可训练；`m_i` 是动态 margin。

例如：

```math
\tau_s=0.12,\quad s_i^{d_s}=0.119
\quad\Longrightarrow\quad m_i^{d_s}=-0.001.
```

表示该 token 很接近 Full 选择边界。但这种接近不赋予它额外的 Full 名额。

为减少不同样本的分数量级差异，仅在新增 reactivation head 内使用：

```math
\boxed{B_s=\sum_{j=1}^N s_j^{d_s}+\epsilon,\qquad
\widetilde s_i^{d_s}=s_i^{d_s}/B_s,\qquad
\widetilde m_i^{d_s}=m_i^{d_s}/B_s.}
```

这里 `epsilon=10^{-6}` 是固定数值稳定常数。此归一化不回写 `RoutingDecision.scores`，不影响原 Full Top-K。

第三个信息是下一节定义的 Skip Age；第四个信息是：

```math
\boxed{\widetilde\delta_s=(d_{s+1}-d_s)/L,}
```

表示当前阶段到下一次决策的归一化距离。全部标量都来自当前及过去信息。

---

# 30. 再引入 Skip Age

定义：

```math
\boxed{a_i^l=\text{进入第 }l\text{ 层之前，token }i\text{ 已连续跳过 Full computation 的层数。}}
```

它是推理时维护的非负整数状态，不是可训练参数。

初始化：

```math
\boxed{a_i^0=0.}
```

在本层结束后更新：

```math
\boxed{a_i^{l+1}=\begin{cases}
0,&i\in\mathcal A_l,\\
a_i^l+1,&i\in\mathcal G_l\cup\mathcal C_l.
\end{cases}}
```

Ghost 也增加 Skip Age，因为它仍然没有做完整 decoder computation；不能把 cheap update 伪装成一次真实 Full refresh。

固定归一化为：

```math
\boxed{\widetilde a_i^l=\frac{\min(a_i^l,A_{\max})}{A_{\max}},\qquad A_{\max}=8.}
```

真实整数 age 不截断，仅提供给 predictor 的数值截断。

若 token 第一次在层 7 被 skip，它在层 7 输入处的 age 仍是 0；处理完层 7 后，`a_i^8=1`。这样不会出现“当前层还没运行，先把 age 加 1”的索引错误。

---

# 31. 为什么 Skip Age 有用？

两个 token 的当前 importance 可以很接近，但它们距离上一次完整更新的时间不同：

```math
s_1^{d_s}\approx s_2^{d_s},\qquad a_1^{d_s}=1,\quad a_2^{d_s}=8.
```

Skip Age 给 predictor 提供的是“多久没有做 Full computation”的历史线索。

它不是 staleness 误差的精确测量：一个 token 可能经过了多次有效 Ghost update，虽然 age 很大，状态仍然较接近 teacher；另一个 token 即使 age 很小，也可能存在较大 compact-context 偏差。

因此本文把 age 作为 reactivation head 与 gate 的辅助输入，而不直接设定：

```math
\text{age 大}\Rightarrow\text{一定进入 Full，或一定发生严重误差。}
```

Full 的选择始终不受 age 规则控制。

---

# 32. Reactivation Predictor 具体怎么写？

在决策层 `d_s`，先对所有当前 skipped tokens 计算低维 token code：

```math
\boxed{z_i^{d_s}=W_{\mathrm{down}}^{d_s}\operatorname{LN}_{d_s}(v_i^{d_s})
\in\mathbb R^{d_b}.}
```

这与 Self branch 使用同一个投影，不再引入一个独立的大矩阵。

拼接输入：

```math
\boxed{x_i^{s,\mathrm{react}}=
[z_i^{d_s};\widetilde s_i^{d_s};\widetilde m_i^{d_s};\widetilde a_i^{d_s};\widetilde\delta_s]
\in\mathbb R^{d_b+4}.}
```

分号表示向量拼接。相比原稿的 `d_b+3`，这里多一个明确的 stage-length 标量，用于适配 Reroute 的不等长阶段。

随后固定使用线性 sigmoid head：

```math
\boxed{r_i^s=\sigma((w_{\mathrm{react}}^{d_s})^\top x_i^{s,\mathrm{react}}
+b_{\mathrm{react}}^{d_s}).}
```

其中：

```math
\boxed{w_{\mathrm{react}}^{d_s}\in\mathbb R^{d_b+4},\qquad
b_{\mathrm{react}}^{d_s}\in\mathbb R}
```

都是可训练参数，按第 45 节的 block-sharing 规则共享。

```math
\sigma(x)=\frac1{1+e^{-x}},\qquad0<r_i^s<1.
```

`r_i^s` 的固定含义是：

> 当前 skipped token 在下一次原 decision layer 重新进入 Full 的预测概率。

阶段内不反复调用 reactivation head，不用过期 score 冒充“当前层新计算的 score”。Ghost 集合确定后，仅对 Ghost tokens 在后续层计算 Self/Context/gate。

因此 Cold 并非完全零计算：它在决策层参与原 scoring，还参与新增的 Ghost 候选打分；但它不做逐层状态更新。

---

# 33. 到这里可以变成三类 token

原 Reroute 的：

```math
\text{Full / Skip}
```

被细化为：

```math
\boxed{\text{Full / Ghost / Cold}.}
```

### Full

```math
\mathcal A_l
```

仍由原 Reroute 决定，进入原 attention、FFN，并按原逻辑写入该层 KV cache。

### Ghost

```math
\mathcal G_l\subseteq\mathcal U_l
```

不进入原 Full 分支，仅在 deferred-state 分支中做 cheap residual update。

### Cold

```math
\mathcal C_l=\mathcal U_l\setminus\mathcal G_l
```

保持 identity bypass，不做状态演化。

三个集合互不相交，并满足：

```math
\boxed{\mathcal A_l\cup\mathcal G_l\cup\mathcal C_l=\{1,\ldots,N\}.}
```

**划分时机固定：**在原 decision layer 完成 Full 选择后，只对其剩余 skipped tokens 划分一次 Ghost/Cold；在该阶段的非决策层，三个集合的成员资格均不重新选择。

下一决策层，先恢复包含最新 Ghost states 的完整逻辑序列，再由原 Full router 重选，最后对新的 skipped 集合重分 Ghost/Cold。

---

# 34. 怎么决定哪些 skipped token 进入 Ghost？

固定 Ghost 预算：

```math
\boxed{K_G=128.}
```

这是新增 skipped-state computation 的预算，不是 Full 预算，也不抵扣原 `K_F`。

对存在下一次决策的阶段：

```math
\boxed{K_G^{(s)}=\min(128,|\mathcal U_{d_s}|),\qquad s<S-1.}
```

选择：

```math
\boxed{\mathcal G_{d_s}
=\operatorname{TopK}(\{r_i^s:i\in\mathcal U_{d_s}\},K_G^{(s)})}
```

以及：

```math
\boxed{\mathcal C_{d_s}=\mathcal U_{d_s}\setminus\mathcal G_{d_s}.}
```

仅对新增 Ghost 排序，固定使用概率降序、原视觉索引升序作为并列分数的次级顺序，保证相同输入下候选划分可复查；原 Full Top-K 的实现不变。

阶段内部直接继承：

```math
\boxed{\mathcal G_l=\mathcal G_{d_s},\quad
\mathcal C_l=\mathcal C_{d_s},\qquad d_s\le l<d_{s+1}.}
```

例如 LLaVA、`N=576`、`ours_vs_pdrop / avg64` 的第一个阶段：

```math
\boxed{80\text{ Full}+128\text{ Ghost}+368\text{ Cold}=576.}
```

后一个 Full budget 为 40 且仍存在下一决策的阶段：

```math
40\text{ Full}+128\text{ Ghost}+408\text{ Cold}=576.
```

最后一次决策后的阶段固定：

```math
\boxed{K_G^{(S-1)}=0,\quad\mathcal G_l=\varnothing,\quad\mathcal C_l=\mathcal U_l.}
```

因此最后阶段的例子是 `20 Full + 0 Ghost + 556 Cold`，不是仍然更新 128 个永远无法再入选的 token。

---

# 35. Ghost priority 固定使用 reactivation probability

正式方案唯一的 Ghost priority 是：

```math
\boxed{\operatorname{priority}_i^s=r_i^s.}
```

不另加 residual norm 排序、risk score、age 强制激活、额外 Full 名额或动态修改 keep ratios。

Skip Age 已经作为 predictor 的输入；gate 负责控制实际更新幅度。二者都不成为另一个能覆盖原 Full Top-K 的规则。

这里的分工明确为：

```math
\boxed{\begin{aligned}
\text{原 Reroute score}&\longrightarrow\text{谁做 Full},\\
\text{新增 reactivation probability}&\longrightarrow\text{谁在 Skip 中做 Ghost},\\
\text{新增 gate}&\longrightarrow\text{Ghost 更新施加多大幅度}.
\end{aligned}}
```

后面的 ablation 用于检验组件必要性，正式方案保持上述分工。

---

# 36. 那么完整 token 更新公式终于可以写出来

为避免把 attention 误写成单 token 独立函数，先定义 Full 输出：

```math
\boxed{\{u_i^l:i\in\mathcal A_l\}\cup T^{l+1}
=\operatorname{Split}\!\left(F_l(X_{\mathrm{compact}}^l)\right).}
```

`u_i^l∈R^D` 是 token `i` 在原 compact decoder 中真正计算得到的本层输出；`T^{l+1}` 是所有非视觉位置的对应输出。

随后视觉状态的统一更新为：

```math
\boxed{
v_i^{l+1}=\begin{cases}
u_i^l,&i\in\mathcal A_l,\\[2mm]
v_i^l+\widehat\Delta_i^l,&i\in\mathcal G_l,\\[2mm]
v_i^l,&i\in\mathcal C_l.
\end{cases}}
```

其中：

```math
\boxed{\widehat\Delta_i^l
=g_i^l(\widehat\Delta_{i,\mathrm{self}}^l+\widehat\Delta_{i,\mathrm{ctx}}^l).}
```

非视觉 tokens 沿用原 Full 输出，不接入 Ghost 分支。原 RoPE/MRoPE 位置、原 compact sequence 顺序与 KV 写入规则全部不改。

在非决策层，Ghost token 的新状态留在 deferred storage；到下一决策层，它才重新参与原 Full scoring。下一次分数可以因状态被更新而改变，但**打分公式、决策层和 Full 数量不变**。

---

# 37. 使用固定的 gate 控制 Ghost 更新幅度

Gate 的输入由 Self code、Context code 和当前 Skip Age 构成：

```math
\boxed{x_i^{G,l}=[z_i^l;h_{i,\mathrm{ctx}}^l;\widetilde a_i^l]
\in\mathbb R^{2d_b+1}.}
```

定义：

```math
\boxed{g_i^l=\sigma((w_g^l)^\top x_i^{G,l}+b_g^l).}
```

其中：

```math
\boxed{w_g^l\in\mathbb R^{2d_b+1},\qquad b_g^l\in\mathbb R}
```

是可训练参数。Gate 是标量，乘在完整 `D` 维 residual 上：

```math
\boxed{\widehat\Delta_i^l=g_i^l
\left(\widehat\Delta_{i,\mathrm{self}}^l+\widehat\Delta_{i,\mathrm{ctx}}^l\right).}
```

当 `g_i^l` 接近 0，更新趋向 identity bypass；当它接近 1，施加较完整的预测 residual。

固定初始化：`W_up` 与 `W_ctx` 初始化为零，`w_g=0`、`b_g=-2`；其余投影采用 Xavier uniform，prototype queries 采用标准差 0.02 的零均值正态初始化，reactivation head 的 weight 与 bias 初始化为零。

这样初始 Ghost residual 为零，初始前向在状态更新意义上退化为原 identity bypass。训练中 gate 是学到的缩放系数，**不是经过校准的不确定性概率，也不保证 residual 范数有绝对上界**。

数值稳定性还由第 39—43 节的归一化监督、方向损失有效项处理及第 59 节的梯度裁剪共同检查。

---

# 38. 现在开始讲训练：Ground Truth 从哪里来？

Teacher 固定为与 student 相同 checkpoint 的**未路由原始 LVLM**，不加载更大的另一个模型。

对相同图片和问题：

```math
(I,Q),
```

使用相同视觉预处理、prompt template 与 tokenization，进行 dense prefill，记录：

```math
\boxed{v_i^{l,*},\qquad v_i^{l+1,*},\qquad
\Delta_i^{l,*}=v_i^{l+1,*}-v_i^{l,*}.}
```

Teacher 全部冻结，并在无梯度模式下运行。训练只使用当前输入 prompt，不把待评测答案拼进 teacher 或 Ghost 输入。

训练输入契约固定为图文清单，最少包含 `sample_id`、`image_id`、`image_path` 与 `question`。独立训练和验证清单路径在第 59 节确定；上传仓库没有这些训练清单，所以不能把本节描述当作项目已有数据。

Dense teacher 的监督信号与 Reroute routing trace 是两回事：前者教状态演化，后者教下一次重激活。不能直接对 dense hidden states 另跑一套选择器，替代 student 的原 Reroute 规则。

Teacher states 按 microbatch 流式产生和释放，不要求把整个训练集的全部层状态预先保存到磁盘。

---

# 39. 第一项 Loss：Residual Reconstruction Loss

对正式 rollout 中实际进行 Ghost update 的位置：

```math
\boxed{\Omega_G=\{(i,l):i\in\mathcal G_l\},\qquad Z=|\Omega_G|,}
```

监督预测 residual 接近 dense teacher 的本层真实增量。

为减少不同 hidden dimension 和不同层幅值的影响，定义 teacher residual 的平均平方尺度：

```math
\boxed{q_i^l=\operatorname{stopgrad}\!\left(\frac1D\|\Delta_i^{l,*}\|_2^2\right)+\epsilon.}
```

`q_i^l` 是监督归一化标量，不是 query，也不是新增推理输入。

固定损失为：

```math
\boxed{\mathcal L_\Delta=
\frac1{\max(1,Z)}\sum_{(i,l)\in\Omega_G}
\frac{\|\widehat\Delta_i^l-\Delta_i^{l,*}\|_2^2}{Dq_i^l}.}
```

这保留原稿“拟合 layer-wise residual”的目标，只明确了平均方式和数值尺度。

Warm-up 阶段只为训练 predictor，在存在下一决策的阶段对全部 skipped candidates 计算监督，令训练集合为 `Omega_W={(i,l):i∈U_l, d_0≤l<d_{S-1}}`。进入正式 rollout 和评测后，只更新选中的 Ghost 集合，不把 warm-up 的候选计算带入推理预算。

如果本批没有有效训练位置，对应 loss 定义为 0，不除以空集合大小。

需要注意：student 当前状态不一定等于 teacher 当前状态，因此学习 teacher 的本层增量不自动消除已积累的误差；重激活时的累计偏差由 Freshness Loss 直接约束。

---

# 40. 第二项 Loss：Direction Loss

除了 residual 数值，还约束其方向。

固定：

```math
\boxed{\epsilon=10^{-6}.}
```

为了避免零初始化和近零 teacher residual 导致无意义的 cosine 监督，仅对双方范数都大于 `epsilon` 的训练位置计入方向项：

```math
\Omega_{\mathrm{dir}}=\{(i,l)\in\Omega_G:
\|\Delta_i^{l,*}\|_2>\epsilon,\ \|\widehat\Delta_i^l\|_2>\epsilon\}.
```

集合判断不反向传播。损失固定为：

```math
\boxed{\mathcal L_{\mathrm{dir}}=
\frac1{\max(1,|\Omega_{\mathrm{dir}}|)}
\sum_{(i,l)\in\Omega_{\mathrm{dir}}}
\left[1-
\frac{(\widehat\Delta_i^l)^\top\Delta_i^{l,*}}
{\|\widehat\Delta_i^l\|_2\|\Delta_i^{l,*}\|_2+\epsilon}\right].}
```

Warm-up 用 `Omega_W` 替代 `Omega_G`，其余公式相同。零初始化的首次更新主要由 residual reconstruction 提供梯度；预测非零之后再纳入 direction supervision。

这些有效项筛选仅用于训练损失，不是另一条 Ghost routing 规则。

---

# 41. 第三项 Loss：Reactivation Prediction Loss

Reactivation probability 只在阶段入口预测一次：

```math
r_i^s,\qquad i\in\mathcal U_{d_s},\quad s<S-1.
```

使用第 28 节定义的下一决策标签：

```math
y_i^{s,\mathrm{react}}.
```

固定使用 binary cross entropy：

```math
\boxed{\mathcal L_{\mathrm{react}}=-\frac1{\max(1,Z_R)}
\sum_{s=0}^{S-2}\sum_{i\in\mathcal U_{d_s}}
\left[y_i^{s,\mathrm{react}}\log r_i^s+
(1-y_i^{s,\mathrm{react}})\log(1-r_i^s)\right],}
```

其中：

```math
\boxed{Z_R=\sum_{s=0}^{S-2}|\mathcal U_{d_s}|.}
```

实现使用 logits 形式的 `binary_cross_entropy_with_logits`，避免直接对接近 0 或 1 的概率取对数。

同一阶段的标签和预测只计入一次，不在每个非决策层重复计算 BCE。最后一个阶段没有预测目标，也不强行补一批全负标签。

原 Full Top-K 和新增 Ghost Top-K 都保持离散选择，不使用 straight-through estimator。Reactivation head 通过该 BCE 训练；被选中 Ghost 的状态演化模块通过 residual 和 freshness 等连续损失训练。

---

# 42. 最重要的 Loss：直接监督“重新回来之前”的状态

我们真正关心的是：token 被重新选入 Full 时，带回来的 state 是否仍然明显过时。

定义发生过 skip→Full 的位置：

```math
\boxed{\mathcal R=\{(i,d_s):1\le s<S,\ i\in\mathcal A_{d_s}\cap\mathcal U_{d_{s-1}}\}.}
```

这里使用 student 实际 rollout 的选择结果。集合本身 detached，不对“是否入选”求梯度。

在重新入选的 `d_s` 层，取 **进入该层 Full computation 之前** 的 student state：

```math
v_i^{d_s}.
```

而不是先运行完整 decoder，把误差修复一部分之后才测量。

对应 teacher state 为：

```math
v_i^{d_s,*}.
```

固定 teacher-state 尺度：

```math
\boxed{q_{i,\mathrm{state}}^{d_s}
=\operatorname{stopgrad}\!\left(\frac1D\|v_i^{d_s,*}\|_2^2\right)+\epsilon.}
```

Freshness Loss：

```math
\boxed{\mathcal L_{\mathrm{fresh}}=
\frac1{\max(1,|\mathcal R|)}
\sum_{(i,d_s)\in\mathcal R}
\frac{\|v_i^{d_s}-v_i^{d_s,*}\|_2^2}{Dq_{i,\mathrm{state}}^{d_s}}.}
```

这直接对应 **Reactivated Token Freshness**。它允许多个层的 Ghost updates 共同减小重激活时的累计误差。

如果某个重激活 token 在此前整个阶段都是 Cold，那么这一段 identity bypass 本身没有可训练参数；该样本仍能提供 freshness 诊断，但不能凭空给一个不存在的 Ghost update 产生梯度。这也是需要 reactivation predictor 尽量覆盖未来重激活 token 的原因。

---

# 43. 总 Loss 怎么组合？

正式 Rollout Training 固定使用原稿的四类监督：

```math
\boxed{\mathcal L_{\mathrm{total}}=
\lambda_\Delta\mathcal L_\Delta+
\lambda_{\mathrm{dir}}\mathcal L_{\mathrm{dir}}+
\lambda_{\mathrm{react}}\mathcal L_{\mathrm{react}}+
\lambda_{\mathrm{fresh}}\mathcal L_{\mathrm{fresh}}.}
```

权重固定为：

```math
\boxed{\lambda_\Delta=1,\quad\lambda_{\mathrm{dir}}=0.1,\quad
\lambda_{\mathrm{react}}=0.1,\quad\lambda_{\mathrm{fresh}}=1.}
```

它们是人为设定的 loss weighting hyperparameters，不是训练参数，也不是理论最优值。

Warm-up 尚未运行自身 Ghost states 的闭环，因此固定使用：

```math
\boxed{\mathcal L_{\mathrm{warm}}=
\mathcal L_\Delta+0.1\mathcal L_{\mathrm{dir}}+0.1\mathcal L_{\mathrm{react}}.}
```

之后进入正式 rollout，才加入 `L_fresh`。这是两个顺序训练阶段，不是两套待选择的方法。

不额外加入 language-model task loss、不用 benchmark 测试标签微调原骨干、不训练 Full router。全部 loss 在 FP32 中归约；空有效集合对应的 loss 为 0。

---

# 44. 哪些参数训练，哪些不训练？

原视觉编码器：

```math
\boxed{\text{Frozen}}
```

原 multimodal projector：

```math
\boxed{\text{Frozen}}
```

原 LLM、原 normalization、Q/K/V/O projections、原 FFN、原 LM head：

```math
\boxed{\text{Frozen}}
```

原 Reroute 的 attention scorer、Top-K、keep ratios 与 stage cache：

```math
\boxed{\text{原算法不变；没有新增的可训练 Full selector。}}
```

不能把原 scorer 描述成“先冻结一个已训练的 selector”，因为上传项目中的 Full selection 本身是 attention scoring 与离散 Top-K 逻辑。

---

新增并训练的参数是：

```math
\boxed{W_{\mathrm{down}},W_{\mathrm{up}}}
```

用于 Self Evolution；

```math
\boxed{W_R}
```

用于 residual compression；

```math
\boxed{q_m^P,W_K^P}
```

用于 residual prototype construction；

```math
\boxed{W_Q^G,W_K^G,W_V^G,W_{\mathrm{ctx}}}
```

用于 Ghost residual retrieval；

```math
\boxed{w_g,b_g}
```

用于 residual gate；

```math
\boxed{w_{\mathrm{react}},b_{\mathrm{react}}}
```

用于预测下一 decision layer 的重激活。

**Frozen weights 不等于整个 student forward 都使用 `no_grad()`。** Rollout Training 必须保留 hidden-state 计算图，使后面 decision layer 的 Freshness Loss 可以反传到前面各层的 Ghost updates。Teacher、离散路由标签和 warm-up 的原始 Reroute reference forward 才使用无梯度路径。

---

# 45. 每层一套参数会不会很多？固定使用 Block-shared Ghost Predictor

固定每 4 个 decoder layers 使用同一组 Ghost 参数：

```math
\boxed{G=4.}
```

这里 `G` 是共享跨度超参数，不是 Ghost 集合 `G_l`。

采用 0-based 分组：

```math
\boxed{b(l)=\lfloor l/4\rfloor.}
```

于是：

```math
\begin{aligned}
l=0,1,2,3&\longrightarrow b=0,\\
l=4,5,6,7&\longrightarrow b=1,\\
l=8,9,10,11&\longrightarrow b=2.
\end{aligned}
```

前文写作 `W_down^l` 的参数，在实现中实际读取 `ghost_blocks[b(l)].W_down`。所有新增矩阵、prototype queries、gate 与 reactivation head 都按同一个规则共享，原 decoder 各层参数不共享、不改变。

原层的 `input_layernorm` 仍使用该层自己的冻结模块，不把它替换成 Ghost block 的共享 normalization。

只为实际可能执行 Ghost 的层创建 block：

```math
\boxed{d_0\le l<d_{S-1}.}
```

因此本项目四次决策的配置下，LLaVA 使用 block 0—5，共 6 组；Qwen 使用 block 0—4，共 5 组。最后决策之后不执行 Ghost，不为永远不用的尾部层额外创建 predictor。

不同骨干不共享参数。第 59 节确定每个“骨干×预算档位×调度族”独立训练一个 checkpoint；其 `compact_route` 与 `compact_route_stagewise` 两条实现使用同一份 Ghost 权重。

---

# 46. 完整训练流程重新从头走一次

### Stage 0：建立严格对应的输入与监督

读入训练清单中的：

```math
(I,Q).
```

使用原模型的视觉预处理与 prompt template，并按样本实际 visual range 记录原 token index。运行 dense teacher 得到：

```math
v_i^{l,*},\qquad\Delta_i^{l,*}.
```

同一 microbatch 内，teacher 与 student 的图片、文本、视觉索引映射必须一致；不能分别 resize 或使用不同问题模板。

所有路由上下文、deferred states、Ghost masks 与 Skip Ages 均按新样本重置，不能把前一个样本的 stage cache 沿用到下一个样本。

### Stage 1：Warm-up，学习单层演化与下一次重激活

使用相同配置，运行**原始 identity-bypass Reroute**，记录原项目 scorer 产生的各阶段 Active Sets、各层 active residuals 与 skipped states。

对存在下一决策的阶段，在其每一层根据该层实际 active residuals 构造 Prototype Bank；对全部 skipped candidates 计算训练用的预测 residual。

在每个阶段入口，利用 reference trace 的下一决策选择产生 reactivation labels。

固定优化：

```math
\mathcal L_{\mathrm{warm}}
=\mathcal L_\Delta+0.1\mathcal L_{\mathrm{dir}}+0.1\mathcal L_{\mathrm{react}}.
```

Warm-up 中不把预测 residual 写回 reference Reroute，不改变 reference 的后续 Full routing。该阶段用于让新增模块获得可用的初始参数，不能作为正式推理路径。

### Stage 2：Rollout Training，使用真正的 Ghost states 闭环运行

切换到正式 Reroute-Ghost：阶段入口由原 router 选择 Full，再选择 Ghost/Cold；每层先执行 Full，再更新当前 Ghost deferred states。

下一 decision layer 读取最新状态并由原 scorer 重新选择，不强制复用 Stage 1 的 Active Sets。完整前向结束后，根据本次 student 实际的下一决策选择构造 detached reactivation labels。

同时计算 Ghost residual reconstruction、direction、reactivation 与重激活前 freshness 四项损失，优化：

```math
\mathcal L_{\mathrm{total}}.
```

更新的始终只有 Ghost 模块参数。训练轮数、优化器、数据清单接口、checkpoint 粒度在第 59 节统一固定。

---

# 47. 为什么还需要 Rollout Training？

Warm-up 中，Ghost predictor 看到的是原 identity-bypass Reroute 的状态；正式推理中，它看到的可能已经包含此前 Ghost updates 的状态：

```math
v_i^l=v_i^{d_s}+\sum_{t=d_s}^{l-1}\widehat\Delta_i^t,\qquad i\in\mathcal G_{d_s}.
```

因此后续层 predictor 的输入分布会改变，误差也可能逐层积累。这是原稿讨论的 **State Evolution Error Accumulation** 在 Reroute stage 中的对应形式。

Rollout 必须真实运行完整的：

```math
\boxed{\text{原决策}\rightarrow\text{阶段内 Ghost 演化}\rightarrow\text{下一原决策}\rightarrow\cdots}
```

才会暴露这种误差。

实现上的固定要求如下：Ghost 写回 deferred buffer 使用保留 autograd 的函数式更新，不在写回时 `.detach()`；teacher states 与未来离散标签保持 detached；原 scorer 与离散 Top-K 不反传梯度。

训练固定 `use_cache=False`，避免把推理 KV cache 当成训练状态；不启用会重复执行有状态 routing hooks 的 gradient checkpointing。评测则恢复原项目的正常 cache 语义。

Freshness Loss 是训练目标，不是“只要回到正确 manifold 就一定成功”的保证。最终是否改善模型答案和效率，仍由同一组评测任务衡量。

---

# 48. 完整 inference 流程

下面是唯一正式推理流程。它只在 **visual prefill** 中启用 Ghost，不在自回归单 token decode 中新增视觉更新。

## 第一步：视觉与文本输入

沿用原模型视觉编码器、projector、tokenizer、prompt template 和 processor，获得完整逻辑序列 `X^0`，检测实际视觉区间和原始位置索引。

初始化：

```math
a_i^0=0,
```

清空原 routing cache、stagewise deferred cache 以及新增 Ghost state。没有视觉区间的输入走原项目普通路径。

## 第二步：第一次决策之前

对于：

```math
l<d_0,
```

原样运行 dense decoder layers，不建立 Ghost/Cold 集合。

## 第三步：到达原 decision layer

如果从上一 compact stage 进入，先将最新 Full states 与 deferred states 按原 token index 恢复完整逻辑序列。

其中 deferred states 必须包含此前逐层更新后的 Ghost states，而不是最初阶段入口时的旧快照。

随后调用原 `_capture_attention_weights` 与原 `PDropRouter.compute_scores`，使用当前层的完整逻辑状态评分，得到：

```math
\mathcal A_{d_s},\qquad\mathcal U_{d_s}.
```

既不添加额外 decision layer，也不在原 scorer 之前插入一次“预刷新后再重选”。

## 第四步：只对 skipped 集合划分 Ghost/Cold

若 `s<S-1`，用阶段入口的 token states、原 score、margin、Skip Age 与下一决策距离预测 `r_i^s`，选择最多 128 个 Ghost tokens。

若已经是最后一次决策，则：

```math
\mathcal G_{d_s}=\varnothing,\qquad\mathcal C_{d_s}=\mathcal U_{d_s}.
```

缓存本阶段 Full、Ghost、Cold 的索引，非决策层不重新划分。

## 第五步：Full branch 先执行

gather 所有非视觉 tokens 与 `A_l`，保留原 token 顺序和 RoPE/MRoPE 位置，调用原 decoder：

```math
X_{\mathrm{compact}}^{l+1}=F_l(X_{\mathrm{compact}}^l).
```

记录 active visual tokens 的输入和输出，计算：

```math
d_j^l=u_j^l-v_j^l,\qquad j\in\mathcal A_l.
```

原 Full 分支中没有 Ghost/Cold 位置，不给它们写 KV。

## 第六步：在本层构造 Residual Prototype Bank

只有 `G_l` 非空时，才执行：

```math
\{d_j^l\}_{j\in\mathcal A_l}\longrightarrow P_l=\{p_1^l,\ldots,p_8^l\}.
```

Bank 每层重建，不沿用上一层的 residual prototypes。

## 第七步：Ghost update

对 `i∈G_l`，使用该 token 当前 deferred state 计算：

```math
z_i^l,\quad\widehat\Delta_{i,\mathrm{self}}^l,\quad
h_{i,\mathrm{ctx}}^l,\quad\widehat\Delta_{i,\mathrm{ctx}}^l,\quad g_i^l.
```

随后：

```math
\boxed{v_i^{l+1}=v_i^l+
 g_i^l(\widehat\Delta_{i,\mathrm{self}}^l+\widehat\Delta_{i,\mathrm{ctx}}^l).}
```

只写回对应 deferred visual rows。Prototype attention 是新增小分支的内部计算，不改变原 LLM attention matrix 的大小。

## 第八步：Cold identity bypass

对 `i∈C_l`：

```math
\boxed{v_i^{l+1}=v_i^l.}
```

Cold 不做 Self/Context/gate，不在非决策层进行 reactivation scoring。

## 第九步：更新 age 并进入下一层

Full 的 age 置 0；Ghost 与 Cold 的 age 加 1。

如果下一层仍在当前阶段，沿用已缓存的三个集合，只执行 Full 与逐层 Ghost 演化。只有到下一原 decision layer 才恢复完整逻辑序列并重新选择。

## 第十步：完成 prefill 与 decode

最后决策之后只保留原 Full 路径与 Cold deferred states，不再执行 Ghost。prefill 结束后，回答生成沿用原模型与原项目 cache 逻辑。

自回归 decode 的 `seq_len=1` 路径不重建 Ghost Bank，不重算 Full/Ghost Top-K，不额外补写此前被跳过层的视觉 K/V。

---

# 49. 最后把三部分重新拼回来：必须匹配原 compact 实现

逻辑上，我们始终可以表示完整视觉状态：

```math
\boxed{V^{l+1}=\operatorname{Scatter}
(V_{\mathrm{Full}}^{l+1},V_{\mathrm{Ghost}}^{l+1},V_{\mathrm{Cold}}^{l+1}).}
```

但“逻辑上有全部 `N` 个 token”不意味着 stagewise 路径每层都要物理构造一个完整 `N+M` 张量。

## 49.1 `compact_route` 的对应修改

保留现有每层 gather → original decoder → scatter 结构。在 `_forward_compact_route` 中，原来只把 Full outputs 写回完整输出，现在再把 Ghost outputs 写入其原视觉位置；Cold 位置保留输入值。

下一层只有在原 decision layer 才重新 scoring。非决策层即使调用 `compute_scores`，也必须沿用原 stage cache，而不是因此重新 Top-K。

## 49.2 `compact_route_stagewise` 的对应修改

保持阶段内 forward tensor 只包含 Full 与非视觉 tokens。

原上下文中的：

```python
_stage_compact_kept_indices
_stage_deferred_indices
_stage_deferred_hidden
```

继续承担同样职责；只把 `_stage_deferred_hidden` 从“整个阶段完全不变的快照”改为“Ghost 行逐层更新、Cold 行保持不变的 deferred buffer”。

新增的状态仅包括：本阶段 Ghost 在 deferred buffer 中的行号、原视觉索引映射、Skip Ages，以及训练时需要的阶段入口 logits。Ghost 参数由模型模块持有，不把训练参数放进动态 routing cache。

在阶段入口 `_forward_compact_route_stagewise` 和阶段内 `_forward_compact_in_stage` 两处，都在拿到当前层 Full output 后计算 active residual，再更新 deferred buffer 中的 Ghost 行。

到下一 decision layer：

```math
\boxed{\text{最新 compact states}+\text{最新 deferred buffer}
\longrightarrow\text{完整逻辑序列}
\longrightarrow\text{原 scorer 与 Top-K}.}
```

**不能只修改阶段入口函数。** 如果阶段内 helper 仍然不更新 deferred buffer，Ghost 在长 stage 内仍会停留在旧状态，无法实现本文逐层演化。

## 49.3 原 attention 与 KV 语义保持不变

在原决策层，scoring 本身仍会访问完整逻辑状态并执行原 Q/K 打分开销；这与真正进入 compact decoder 并写 KV 是不同的计算。

Ghost/Cold 不额外参加原 Full attention，不写该层原 KV cache，不追溯补齐过去层的 KV。Ghost state memory 必须单独计入运行内存，不能称作“零额外内存”。

stagewise 与非 stagewise 是同一个状态更新算法的两条既有执行路径，不是两套待选方法。二者用同一 checkpoint、同一 Full/Ghost 划分规则；数值一致性必须通过测试核对，不能直接沿用 README 对原版的“bit-identical”描述作为新增模块已经通过的证明。

---

# 50. 这时一个 token 的完整 trajectory 可以长这样

仍以 LLaVA 的 `[3,7,15,23]` 为例。

假设 token 271 在决策层 7 被原 Full router 跳过，但被新增 reactivation head 列入 Ghost：

```math
271\notin\mathcal A_7,\qquad271\in\mathcal G_7.
```

整个 7—14 层阶段中，它的身份一直是 Ghost，不会中途变成 Full：

```math
v_{271}^{8}=v_{271}^{7}+\widehat\Delta_{271}^{7},
```

```math
v_{271}^{9}=v_{271}^{8}+\widehat\Delta_{271}^{8},
```

依次到：

```math
\boxed{v_{271}^{15}=v_{271}^{7}+\sum_{l=7}^{14}\widehat\Delta_{271}^l.}
```

在决策层 15，只有原 Reroute Top-K 再次选中它，才有：

```math
271\in\mathcal A_{15}.
```

它随后以已经过 Ghost 演化的状态进入原 Full decoder。

因此同一个阶段式路由框架中的变化是：

```math
\boxed{\text{8 层 identity bypass}\rightarrow\text{原决策层 Full}}
```

变为：

```math
\boxed{\text{8 层 Ghost state evolution}\rightarrow\text{原决策层 Full}.}
```

第 15 层是否真的选中 271，仍由原 score 决定；Ghost predictor 的高概率不是强制重激活命令。

---

# 51. 与普通 Adapter 相比，本文多用了什么信息？

Local-only Adapter 根据：

```math
\widehat\Delta_i^l=f(v_i^l)
```

预测状态变化。

本文额外使用：

```math
\boxed{\widehat\Delta_i^l=f(v_i^l,\{d_j^l\}_{j\in\mathcal A_l}).}
```

也就是当前样本、当前层已经实际观察到的 active residuals。

这使模型具备利用 sample-specific transformation patterns 的能力，但不保证它一定学到有用的迁移。若 active residuals 对当前 Ghost token 不具有预测力，复杂分支可能并不优于 local-only。

此外，本文的 reactivation head 使 Ghost 预算集中于未来更可能重新进入 Full 的候选；这是对 skipped-state 更新资源的分配，不是对原 Reroute Full selection 的替代。

“是否优于普通 Adapter”由第 57—58 节固定的对照实验回答，而不是写成已经证明的理论结论。

---

# 52. 这个设计的另一个重要特点：不是所有 skipped token 都补

以 `N=576`、`ours_vs_pdrop / avg64` 的首阶段为例：

```math
\boxed{80\text{ Full}+128\text{ Ghost}+368\text{ Cold}.}
```

每一个非决策层，只让 128 个 Ghost tokens 执行小型 predictor，而不是对全部 496 个 skipped tokens 都执行完整的 Self/Context 分支。

但开销必须完整记账：阶段入口仍需对全部 skipped candidates 计算低维 code 与 reactivation logits；原 decision scoring 也仍访问全部逻辑视觉位置。

所以准确的描述是：

```math
\boxed{\text{阶段入口：全部 Skip 参与候选打分；阶段内部：仅 Ghost 做 cheap state update。}}
```

Cold 不做逐层状态演化，但不等于整个输入处理过程中绝对零计算。

最后阶段由于无下一次 reactivation，Ghost 预算固定归零。

---

# 53. Ghost branch 到底比完整 Transformer 多出多少开销？

不能只写“Ghost 很便宜”，也不能把它的开销并入原 avg 档位之后继续声称总 FLOPs 完全相同。

对于每个确实执行 Ghost 的非决策层，主要新增计算包括：

```math
\boxed{O(K_F^{(s)}Dd_b)}
```

用于 active residual compression 与 prototype keys；

```math
\boxed{O(M_pK_F^{(s)}d_b)}
```

用于 prototype aggregation；

```math
\boxed{O(K_G^{(s)}Dd_b)}
```

用于 Ghost 的 down/query/up/context projections；

```math
\boxed{O(K_G^{(s)}M_pd_b+M_pd_b^2)}
```

用于低维 prototype attention；以及归一化、gate、residual 加法、deferred-buffer 读写与索引操作。

这里的大 O 省略常数；例如 Ghost projection 项包含多次 `D↔d_b` 投影，不是只进行一次 `D×32` 乘法。

在原 decision layer，还新增对全部 skipped candidates 的 reactivation scoring：

```math
\boxed{O(|\mathcal U_{d_s}|Dd_b)
+\text{Ghost candidate ranking 开销}.}
```

因此总成本为：

```math
\boxed{C_{\mathrm{Reroute\text{-}Ghost}}
=C_{\mathrm{original\ Reroute}}+C_{\mathrm{Ghost\ selection}}+C_{\mathrm{state\ evolution}}.}
```

原有的 full-range attention scoring 同样要包含在第一项中，不能因为它没有执行 V/FFN 就漏记。

---

本文保持的是：

```math
\boxed{\text{同一原 Full-token schedule 与同一 }K_F^{(s)}.}
```

而不是自动保持：

```math
\text{加入 Ghost 后总 FLOPs 仍与原方法严格相等。}
```

不为了重新追求总 FLOPs 匹配而降低 Full keep ratios，因为那会改变用户要求保留的原路由预算。

效率实验固定报告：原项目 prefill TFLOPs、post-prefill KV bytes、prefill latency 与 runtime 指标，并单独列出 Ghost 参数量、Ghost/deferred 临时内存以及完整峰值 GPU memory。

其中 cache 的视觉 token 数仍由原 Full schedule 决定；Ghost state 不是 KV entry，但仍占用显存。stagewise 路径不必每层构造完整张量，但每层 Ghost buffer 读写和 Full→Ghost 的串行依赖仍可能影响延迟。

扩展 profiler 时必须覆盖新增矩阵乘、prototype attention 与候选打分；已有统计工具未覆盖的算子要明确列明，不能一边漏计一边报告“等 FLOPs”。没有实际 GPU 测量前，不给出速度提升倍数。

---

# 54. 主实验完全沿用 Reroute，同时验证 State Staleness

## 54.1 模型、预算档位与评测框架不变

模型固定为项目已有的：

```math
\boxed{\text{LLaVA-1.5-7B}\quad+\quad\text{Qwen2.5-VL-7B}.}
```

保留原 checkpoint 标识、processor、attention backend、Qwen `max_pixels=451584`、`batch_size=1` 与模型族对应的 prompt/bbox preset。

预算档位固定保留：

```math
\boxed{\text{avg192},\quad\text{avg128},\quad\text{avg64}.}
```

评测继续通过 `lmms-eval` 和原仓库 `scripts/run_eval.py`、`scripts/run_setting.sh` 执行；不新增另一套 benchmark、不改分数聚合方式、不换测试划分。

## 54.2 全部评测任务与 split 原样保留

| 项目评测配置 | 实际 `lmms-eval` task |
| --- | --- |
| `pope` | `pope` |
| `gqa` | `gqa` |
| `mmbench` | `mmbench_en_dev` |
| `mme` | `mme` |
| `refcoco_val` | `refcoco_bbox_rec_val` |
| `refcoco_testA` | `refcoco_bbox_rec_testA` |
| `refcoco_testB` | `refcoco_bbox_rec_testB` |
| `refcoco+_val` | `refcoco+_bbox_rec_val` |
| `refcoco+_testA` | `refcoco+_bbox_rec_testA` |
| `refcoco+_testB` | `refcoco+_bbox_rec_testB` |
| `refcocog_val` | `refcocog_bbox_rec_val` |
| `refcocog_test` | `refcocog_bbox_rec_test` |

`paper_main.yaml` 实际包含 **POPE + 8 个 grounding splits**，不包含 GQA/MMBench/MME。后面三项由原 `--tasks vqa` 补齐。不能只跑脚本默认的 POPE 就声称完成全部评测。

原项目完整参照组通过以下两次任务分发覆盖：

```bash
bash scripts/run_paper_table.sh all --tier all --tasks paper_main
bash scripts/run_paper_table.sh all --tier all --tasks vqa
```

新增 Ghost 行在实现后使用同一组 task configs 和参数执行，不修改任务内容。原评测划分只用于评估和诊断，不参与 Ghost 训练或 checkpoint 选择。

## 54.3 Prompt 与 bbox 处理不变

LLaVA 保留 `bbox=normalized`、`prompt=full`；Qwen2.5-VL 保留 `bbox=pixel`、`prompt=nuwa`，继续应用项目的 RefCOCO utilities patches。

正式比较中清理会覆盖这些 family presets 的自定义环境变量，使原版与 Ghost 使用同一个项目默认。不得通过改变 grounding prompt 或 bbox parsing 为新增方法单独优化结果。

## 54.4 原基线与新增方法行分开

原始 38 个配置完整保留：2 个 dense baselines，以及 2 个模型 × 3 个档位 × 6 个原 routing variants。

这 6 个原 variant 是 `fastv_K3`、`pdrop_earlyL2`、`ours_vs_fastv`、`ours_vs_pdrop` 及后两者各自的 stagewise 版本。

只为 4 个可恢复 Reroute variants 添加 Ghost 对应行，形成 24 个新增配置。合计 62 个原始与新增配置，不能把原 Reroute 配置覆盖成 Ghost 后仍把它当原始基线。

`physical_delete` 的 FastV/PDrop 基线不添加 Ghost，因为它们已经物理删除 token，没有本文的 deferred state 可更新。

原 FastV `scoring_layer=3` 以及各档 `keep_ratio` 原样保留；原 physical PDrop 继续 `monotonic=true`，keep ratios 为：

| 档位 | 原 physical PDrop `keep_ratios` |
| --- | --- |
| `avg192` | `[0.5965,0.3558,0.2123,0.1266]` |
| `avg128` | `[0.4570,0.2088,0.0954,0.0436]` |
| `avg64` | `[0.2789,0.0937,0.0315,0.0106]` |

## 54.5 假设一：原 Reroute 中的 state error 是否随 skip age 增大？

对原 Reroute 实际发生重激活的位置，在进入该层 Full computation 前测量：

```math
\boxed{E_i^{d_s}=1-\cos(v_i^{d_s,\mathrm{Reroute}},v_i^{d_s,*}).}
```

按真实 Full-skip age 分组，统计 cosine error、归一化 L2 error 与样本数，绘制 age—error 关系。

这是需要检验的现象，不预设必然单调上升。误差还可能受 compact context、不同层和不同样本影响，因此分模型、分档位、分决策层报告，不把所有差异都归因于 staleness。

以上评测配置依据：`configs/eval/`、`scripts/run_eval.py`、`scripts/run_setting.sh`、`scripts/run_paper_table.sh`。

---

# 55. 假设二：Active residual 能不能预测 skipped residual？

本节保留原稿的 residual-sharing 诊断，但使用 Reroute 实际的阶段路由。

在同一个输入、同一层、同一原 Active Set 下，记录：

```math
\{d_j^{l,\mathrm{Reroute}}:j\in\mathcal A_l\}
```

以及 dense teacher 中 skipped tokens 的目标 residual：

```math
\{\Delta_i^{l,*}:i\in\mathcal U_l\}.
```

首先比较 Local-only predictor 与 Self+Residual-Transport predictor 对目标 residual 的归一化 reconstruction error 和 cosine similarity。

另做离线低秩诊断：在 dense teacher residual 中按同一 `A_l` 划分 active/skipped，检查 active residual 的低秩表示对 skipped residual 的重构能力。它只是共享结构的上限诊断，不能把 dense active residual 当作正式推理时可用的输入。

明确区分：

```math
\boxed{\text{teacher 内部的 residual 相关性}}
```

与：

```math
\boxed{\text{真实 compact active residual 对 Ghost 的实际预测价值。}}
```

第二个问题才直接决定 Residual Transport 是否值得加入。

诊断数据来自原评测任务中的固定样本，保留 sample IDs，只做观察和报告，不用这些评测样本训练 predictor 或选择超参数。

---

# 56. 假设三：未来重激活的 token 是否更值得补偿？

在每个非末阶段，把当前 skipped tokens 按实际下一次决策结果分为：

```math
\boxed{\mathcal U_s^+=\mathcal U_{d_s}\cap\mathcal A_{d_{s+1}}}
```

和：

```math
\boxed{\mathcal U_s^-=\mathcal U_{d_s}\setminus\mathcal A_{d_{s+1}}.}
```

`U_s^-` 的含义只是“下一次决策没有重激活”，不能叫做“之后永远不重要”；它仍然可能在更晚阶段被原 Reroute 选中。

固定报告 reactivation predictor 对下一决策的 precision、recall、AUROC，以及预算内覆盖率：

```math
\boxed{\operatorname{Recall@}K_G
=\frac{|\mathcal G_{d_s}\cap\mathcal U_s^+|}{|\mathcal U_s^+|}.}
```

无正样本的阶段不进入该 recall 的均值，单独记录数量。

为了把“状态补偿”与“下一次原 score 被改变”区分开，额外做固定轨迹诊断：从原 Reroute 记录一条 Full mask trace，仅在离线诊断中让 identity 与 Ghost 使用相同 Full masks，比较同一批重激活 token 的 freshness。正式主实验不锁定 masks，仍由原 scorer 读取最新 states 并选择。

固定轨迹结果用于研究机制，不能冒充正式推理算法或新主实验。所有任务、样本来源和指标框架仍来自原项目。

---

# 57. 最重要的 ablation 应该是什么？

消融固定使用项目已有的：

```bash
--tasks ablation
```

对应 `gqa`、`mmbench`、`refcoco_testA`、`refcoco_testB`，不另建一组评测任务。

消融锚点固定为 **LLaVA-1.5-7B、`ours_vs_pdrop` 调度、`avg64` 档位、stagewise 执行**。完整方法的主表仍覆盖两个模型、三个档位与两个 Reroute 调度族。

| 对照行 | 状态补偿设置 | 目的 |
| --- | --- | --- |
| 原 Reroute | 全部 skipped tokens identity bypass | 原始基线 |
| Reroute + Self only | 仅保留 Self residual；其余 Ghost 预算、预测、gate、训练目标不变 | 检查 local update 是否已足够 |
| Reroute + Context only | 仅保留 Residual Transport；保留路由和 gate 需要的 token code | 检查实际 active residual 的作用 |
| Reroute-Ghost w/o learned reactivation | 在每个原决策层，用 seed=42 的固定样本/阶段随机优先级选同量 Ghost；其余模块不变 | 检查预测下一次重激活的必要性 |
| Reroute-Ghost w/o freshness loss | 正式 rollout 中仅移除 `L_fresh`，其余不变 | 检查直接监督重激活状态的价值 |
| 完整 Reroute-Ghost | Self + Residual Transport + reactivation + gate + freshness | 正式方法 |

这些对照均保持原 Full decision layers、Full scoring、Top-K、keep ratios、token positions 和原任务设定；不靠更大的 Full budget 带来提升。

每个有训练模块的消融使用相同数据划分、训练步数和初始化 seed，按其定义重新训练。Self-only 的 context code固定置零，Context-only 的 self residual 固定置零，不能在消融时悄悄让该分支通过另一路径参与更新。

同时报告任务指标、重激活 freshness、实测 TFLOPs 与 latency，而不是只比较 accuracy。

这张表用于实验归因。正式方法始终是最后一行。

---

# 58. 如果 Self Adapter 已经和完整方法一样好，结果该怎样解释？

正式方案已经固定，但研究结论不能预先固定。

如果 Self-only 在同一训练和预算条件下与完整方法相当，甚至更好，就应如实报告：在当前 Reroute 设置下，Residual Transport 没有显示出额外收益。不能仅凭“用了 active residual”就宣称创新有效。

同样，如果 learned reactivation 不优于等量随机 Ghost，或 Freshness Loss 降低了 hidden-state error 却没有改善任务表现，也需要把这些结果分别写清楚。

最终判断必须同时看：

```math
\boxed{\text{原评测任务表现}+\text{重激活状态误差}+\text{新增计算与内存开销}.}
```

特别是不能把“Full-token 数量相同”写成“总计算开销严格相同”，也不能用单个示例 token 的改善代表整个 benchmark。

此外，README 中对原 stagewise 与非 stagewise 的数值一致性陈述不自动覆盖新增 Ghost 分支；两种执行路径需做同输入、同权重、同集合规则的独立一致性检查。

---

# 59. 本方案的确定实现参数与实验执行设置

## 59.1 Full routing：逐项复制原配置，不新增预算

所有原 `routing` 字段保持原值：`method`、`action`、`drop_layers`、`keep_ratios`、`monotonic`；FastV/PDrop 物理删除基线及 dense baseline 不添加 Ghost。

新方法只对原可恢复 Reroute 配置增加如下模块配置。以下是**实施时新增的字段约定**，当前上传代码尚未读取这些字段：

```yaml
ghost:
  enabled: true
  budget: 128
  bottleneck_dim: 32
  num_prototypes: 8
  horizon_decisions: 1
  max_skip_age: 8
  block_share_span: 4
  partition_at_decision_only: true
  update_every_layer_in_stage: true
  enable_after_last_decision: false
  write_original_kv: false
  epsilon: 1.0e-6
```

Full selector 直接复用原实现，不新增固定每层 token 数的硬编码配置。

## 59.2 Ghost 超参数固定

```math
\boxed{K_G=128,\quad d_b=32,\quad M_p=8,\quad H_{\mathrm{dec}}=1,\quad A_{\max}=8,\quad G=4.}
```

Self activation 固定 SiLU；新增大矩阵不带 bias；只有 gate 与 reactivation scalar heads 带 bias。共享和初始化严格采用第 37、45 节。

这些是当前确定的工程设定，不声称理论最优或已由实验验证。

## 59.3 训练输入与未提供资源的边界

训练清单固定为：

```text
data/ghost_train.jsonl
data/ghost_val.jsonl
```

每行字段固定为：

```json
{"sample_id":"...","image_id":"...","image_path":"...","question":"..."}
```

训练、验证按图片隔离，同一图片的不同问题不得跨两个清单。对原评测集按图像标识和规范化图像内容哈希做重叠检查；命中评测图片的条目不得进入 Ghost 训练或训练验证。

**这两个清单与图片资源没有包含在上传项目中。** 本文确定的是新增训练模块的数据接口与隔离规则，不把未提供的训练样本或数据集来源虚构成项目已有资源。清单缺失时训练必须明确报错，禁止自动拿评测数据或 `bench_data/` 代替。

输入只需要图片和问题，监督来自 teacher states 与路由轨迹；不要求使用 benchmark 的测试答案进行训练。

## 59.4 优化与 checkpoint 粒度固定

每个“骨干 × 档位 × Reroute 调度族”训练一个独立 Ghost checkpoint：

```math
\boxed{2\text{ 个骨干}\times3\text{ 个档位}\times2\text{ 个调度族}=12\text{ 个 checkpoints}.}
```

每个 checkpoint 同时用于其 `compact_route` 与 `compact_route_stagewise` 两种执行路径，不为它们分别训练两套权重。

| 参数 | 固定值 |
| --- | --- |
| 随机种子 | 42 |
| microbatch | 1 个样本 |
| gradient accumulation | 8 |
| effective batch size | 8 |
| optimizer | AdamW |
| Adam betas | `(0.9,0.999)` |
| Adam epsilon | `1e-8` |
| weight decay | `0.01`，仅矩阵权重；bias 与 prototype queries 不衰减 |
| Warm-up optimizer updates | 1000 |
| Warm-up learning rate | `1e-4`，常数 |
| Rollout optimizer updates | 2000 |
| Rollout learning rate | `3e-5`，常数 |
| 阶段切换 | 载入 warm-up 参数，重新初始化 rollout optimizer 状态 |
| gradient norm clipping | 1.0 |
| loss weights | `1, 0.1, 0.1, 1` |
| student training KV cache | `use_cache=False` |
| gradient checkpointing | 关闭，避免重复执行有状态路由 |
| 骨干 dropout 状态 | 保持 eval；只有新增参数参与梯度优化 |

训练数据按 seed=42 打乱，遍历完成后继续循环，直到达到指定 optimizer update 数。teacher 无梯度，student 允许梯度穿过冻结层传回 Ghost；混合精度随骨干计算 dtype 配置，损失归约与优化器主状态用 FP32，fp16 训练启用 loss scaling。

每 100 次 rollout optimizer updates 在独立 `ghost_val.jsonl` 上验证一次，以最小 `L_total` 选择最终 checkpoint；并列时选更早的 step。不用 POPE、GQA、MMBench、MME 或 RefCOCO 评测结果挑 checkpoint。

## 59.5 效率实验与项目保持一致

沿用 `bench_data/manifest.json` 中实际包含的 **3 个样本**，不把脚本残留注释中的“10-sample cohort”当成真实数量。

原项目参照组的 prefill profile 固定使用 shell dispatcher 的默认协议：

```bash
bash scripts/run_profile.sh all --tier all \
  --n-passes 1 --n-warmup-passes 0
```

Runtime 固定使用 shell dispatcher 的默认协议：

```bash
bash scripts/run_runtime_bench.sh all --tier all \
  --n-passes 5 --n-warmup-passes 2 --n-decode-tokens 64
```

新增 Ghost 配置使用完全相同的 cohort、预处理、generation 长度、warm-up、计时设备与聚合方式；新增训练不计入推理 latency，但 Ghost forward 的全部计算必须计入。

效率脚本保留已有结果字段；Ghost 参数、deferred 状态内存与新增算子开销以独立字段补充，不能用变化后的字段覆盖原 KV 统计含义。

## 59.6 固定实现落点

`models/router.py` 的 Full scoring、Full Top-K 与 stage cache 不改。

新增 `models/ghost.py`，实现 block-shared Self、Residual Prototype、Context、gate、reactivation head 及其训练损失所需输出。

在 `models/patching.py` 的三处 compact helpers 中接入 Ghost：`_forward_compact_route`、`_forward_compact_route_stagewise`、`_forward_compact_in_stage`；并在新 prefill reset 时清空新增动态状态。

新增 Ghost 配置时，从对应原配置完整复制 `routing`、model 与 eval 设置，仅加入 `ghost` 参数与 checkpoint 路径；文件名使用 `ghost_ours_vs_fastv`、`ghost_ours_vs_pdrop` 及其 stagewise 后缀。保留原 38 个配置，不覆盖它们。

`run_eval.py` 与两条 profiler loader 都必须加载同一 Ghost checkpoint，避免出现“评测开了 Ghost、profile 却仍是原 Reroute”的错配。扩展实验枚举仅增加新的配置名，任务选择器和原基线分发不变。

新增训练入口 `scripts/train_ghost.py`，按本节固定两阶段流程执行；新增模块缺失权重时直接报错，不默默以随机 Ghost 参数评测。

以上新增文件名与接入点是本次方案的实现约定，不代表本次已经修改或提交了代码仓库。

## 59.7 实施后的必要正确性检查

关闭 Ghost 时必须走原分支，核对与原 Reroute 的 routing logs 和输出；Ghost 开启时核对 Full scoring 公式、decision layers 与 `K_F` 不变。

在同输入、同权重下比较两条 compact 路径，核对 Ghost/Cold 索引、stagewise deferred 写回与输出误差；原始 token position IDs 不因 Ghost 重新编号。

检查最后阶段 `K_G=0`、没有视觉输入时不路由、decode 不运行 Ghost、每个新样本清空动态 cache，以及空有效 loss 集合不会产生 NaN。

在 rollout 梯度检查中确认：骨干无参数梯度，Ghost 有梯度，Freshness Loss 能到达此前的 Ghost updates。OOM 回退或缺失 scoring 的样本需明确标记，不能混入声称严格同 schedule 的对比结果。

这些是实现与验证要求，不是本次已完成 GPU 运行的报告。

---

# 60. 把所有主要符号汇总一次

| 符号 | 含义 | 类型 |
| --- | --- | --- |
| `L` | decoder layers 总数；LLaVA=32、Qwen=28 | 模型属性 |
| `l` | 当前 decoder layer 的 0-based index | 索引 |
| `N` | 当前样本实际 visual token 数 | 输入属性 |
| `M` | system/文本等全部非视觉 token 数 | 输入属性 |
| `D` | 当前骨干 hidden dimension | 模型属性 |
| `v_i^l` | token `i` 在第 `l` 层输入处的状态 | 动态 feature |
| `X^l` | 当前完整逻辑序列的 hidden states | 动态 feature |
| `\mathcal T` | 所有非视觉位置的索引集合 | 输入索引 |
| `\mathcal D={d_s}` | 原项目 decision layers | 原配置 |
| `S` | decision 次数，当前配置为 4 | 原配置属性 |
| `s(l)` | 当前层所在的原 routing stage | 派生索引 |
| `s_i^{d_s}` | 原最后位置 attention 在视觉区间上的 head-mean score | 动态标量 |
| `\rho_s` | 原 keep ratio | 原配置超参数 |
| `K_F^{(s)}` | 从原 ratio 和实际 `N` 得到的 Full 数量 | 动态预算值 |
| `\mathcal A_l` | 原 Reroute Full/Active token 集合，阶段内固定 | 动态集合 |
| `\mathcal U_l` | 本阶段被 skip 的视觉集合 | 动态集合 |
| `F_l` | 原第 `l` 个 compact decoder layer | 冻结模块 |
| `u_i^l` | Active token 经本层原 decoder 计算得到的输出 | 动态 feature |
| `\Delta_i^{l,*}` | dense teacher 的真实本层状态增量 | 训练监督 |
| `d_j^l` | 当前 compact Full 分支观察到的 active residual | 动态 feature |
| `\widehat\Delta_i^l` | Ghost gate 缩放后的预测 residual | 动态输出 |
| `d_b` | bottleneck dimension，固定 32 | 新增超参数 |
| `W_down,W_up` | Self branch 的降维/升维矩阵 | 可训练参数 |
| `z_i^l` | normalized state 的低维 code | 动态 feature |
| `W_R` | active residual 的压缩矩阵 | 可训练参数 |
| `c_j^l` | compressed active residual code | 动态 feature |
| `M_p` | prototype 数，固定 8 | 新增超参数 |
| `q_m^P,W_K^P` | prototype aggregation query 与 key projection | 可训练参数 |
| `\beta_{mj}^l` | prototype `m` 聚合 active token `j` 的权重 | 动态标量 |
| `p_m^l,P_l` | 动态 residual prototype 与其 bank | 动态 feature |
| `W_Q^G,W_K^G,W_V^G,W_ctx` | Ghost prototype retrieval 的投影矩阵 | 可训练参数 |
| `\alpha_{im}^l` | Ghost token 读取 prototype 的权重 | 动态标量 |
| `h_{i,ctx}^l` | context evolution code | 动态 feature |
| `g_i^l` | 状态更新幅度 gate | 动态标量 |
| `w_g,b_g` | gate 的参数 | 可训练参数 |
| `a_i^l` | 进入该层前连续跳过 Full computation 的层数 | 动态整数 |
| `A_max` | age 输入截断上限，固定 8 | 新增超参数 |
| `H_dec` | 预测未来决策次数，固定为下一次，即 1 | 新增超参数 |
| `\delta_s` | 从当前 stage 入口到下一决策的层数 | 原 schedule 派生值 |
| `\tau_s,m_i^{d_s}` | 原 Full 分数边界及 margin | 动态标量 |
| `B_s` | 新 reactivation head 使用的 score 归一化分母 | 动态标量 |
| `r_i^s` | 下一原 decision layer 重激活的预测概率 | 动态标量 |
| `w_react,b_react` | `d_b+4` 维输入的 reactivation head 参数 | 可训练参数 |
| `y_i^{s,react}` | 从实际路由轨迹得到的下一决策标签 | 训练监督 |
| `K_G` | Ghost 预算上限，固定 128 | 新增超参数 |
| `K_G^{(s)}` | 当前 stage 实际 Ghost 数；最后 stage 为 0 | 动态预算值 |
| `\mathcal G_l` | Ghost 集合，原 stage 内固定 | 动态集合 |
| `\mathcal C_l` | Cold identity-bypass 集合 | 动态集合 |
| `G,b(l)` | 参数共享跨度 4 与 `floor(l/4)` 的 block 编号 | 超参数/派生索引 |
| `\Omega_G,\Omega_W` | rollout 的 Ghost 监督集合与 warm-up 候选集合 | 训练索引 |
| `\mathcal R` | 实际 skip→Full 的 token/decision-layer 对 | 训练与诊断索引 |
| `q_i^l,q_{i,state}^{d_s}` | teacher residual/state 的损失尺度归一化量 | 训练标量 |
| `\lambda_\Delta,\lambda_dir,\lambda_react,\lambda_fresh` | 四项 loss 权重，固定 `1,0.1,0.1,1` | 新增超参数 |
| `\epsilon` | 数值稳定常数，固定 `1e-6` | 固定常数 |

---

# 61. 最后把整套算法浓缩成一条因果链

原 Reroute 的核心路由逻辑保留为：

```math
\boxed{\text{指定 decision layers 上的原 attention scoring}
\longrightarrow\text{原 Top-K Full selection}
\longrightarrow\text{阶段内复用 Full 集合}.}
```

原 skipped 分支是：

```math
\boxed{\text{当前不进入 Full}\longrightarrow\text{阶段内 state 保持不变}.}
```

本文只改变后半部分：

```math
\boxed{\text{当前不进入 Full}
\not\Rightarrow\text{representation 必须被冻结}.}
```

完整方法固定为：

```math
\boxed{
\begin{array}{c}
\text{原 Reroute decision layers / scores / Full Top-K / keep ratios}\\
\downarrow\\
\text{原 Full 集合与原 Skip 集合}\\
\downarrow\\
\text{仅在原 decision layer，对 Skip 预测下一次 reactivation}\\
\downarrow\\
\text{Full / Ghost / Cold：阶段内成员固定}\\
\downarrow\\
\text{每层先执行原 Full compact decoder}\\
\downarrow\\
\text{实际 active residuals}\rightarrow\text{8 个低维 residual prototypes}\\
\downarrow\\
\widehat\Delta_i^l=g_i^l(\widehat\Delta_{i,\mathrm{self}}^l+\widehat\Delta_{i,\mathrm{ctx}}^l)\\
\downarrow\\
\text{Ghost 写回 deferred state；Cold 保持 identity；不新增原 KV}\\
\downarrow\\
\text{下一原 decision layer 读取更新后的完整逻辑状态并重新 Top-K}\\
\downarrow\\
\text{最后一次决策后停止 Ghost；原 decode 与原实验协议不变}
\end{array}}
```

整篇工作的核心问题仍然是：

```math
\boxed{\textbf{Skipping computation should not necessarily mean skipping representation evolution.}}
```

在本项目中的具体含义是：

> **不改 Reroute 决定“谁做完整计算”的机制，只改 skipped token 在等待下一次原决策时“怎样维护自己的状态”。**

这里给出的正式方案只有一套：**阶段入口的 next-decision reactivation prediction + 固定 Ghost/Cold 划分 + 逐层 Self/Residual-Transport Ghost 演化 + gate + 两阶段训练与 Freshness Loss**。

保留原模型、原路由 schedule、原预算 ratios、原 benchmark 与原效率测量协议；新增计算独立报告。更好的 freshness、更高的任务分数以及可接受的额外延迟都是待实测的目标，而不是本文已经获得的结果。

