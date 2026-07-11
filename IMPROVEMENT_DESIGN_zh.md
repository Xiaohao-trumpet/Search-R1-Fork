# Search-R1 改进设计文档：奖励塑形 × RL 算法

> 本文档记录的是 **"怎么改" 和 "为什么这么改"**。
> 汇报时"为什么"往往比"怎么做"更重要，所以本文把**问题诊断 → 文献依据 → 设计决策 → 实验假设**的完整推理链都留了下来，方便反复回看与讲解。
>
> 配套文档：`REPRODUCTION_zh.md`（复现与源码结构）。本文只聚焦"改进"。
> 代码分支：`reward-algo-improvements`（基线 commit `8e0a77d` 之上逐步提交，可随时回退）。

---

## 0. 一句话主线（先讲结论）

> **原始 Search-R1 用"二值 EM 终局奖励 + 原味 GRPO"训练，把本可利用的学习信号从"奖励端"和"估计端"两头都丢掉了。我们从这两端分别把信号找回来——奖励端做稠密化+过程塑形（R+），算法端做去偏+防坍缩+动态采样（A+）——并验证一个核心假设：这两端在打同一个病根（"全同分组→零梯度"），因此存在部分替代关系。**

---

## 1. 出发点：当前实现到底长什么样（代码级，逐行核对过）

### 1.1 奖励（`verl/trainer/main_ppo.py` + `verl/utils/reward_score/qa_em.py`）

- 默认奖励函数：`_select_rm_score_fn` → `qa_em.compute_score_em`。
- 形式：**纯二值、纯终局** `r ∈ {0, 1}`。抽出响应里最后一个 `<answer>…</answer>`，做 `normalize_answer`（小写、去冠词、去标点、规范空格）后**与 gold 完全相等**才给 1，否则 0（`format_score` 默认 0）。
- 奖励张量只在**响应最后一个有效 token** 上非零：`reward_tensor[i, valid_response_length-1] = score`（`main_ppo.py:80`）。全程**没有**过程奖励、没有检索质量奖励、没有格式奖励。
- 一个隐藏脆点：`extract_solution` 要求 `<answer>` 出现 **≥2 次**（因为 prompt 模板里自带一个例子 `<answer> Beijing </answer>`），取最后一个。稍微跑偏格式就直接判 0。
- 仓库其实**已经写好**一个分档版 `qa_em_format.compute_score_em`（格式合法 + 检索命中 → `1 / 0.8 / 0.3 / 0.2 / 0.1 / 0`），带 `is_valid_sequence`（状态机校验 think/search/information/answer 的合法序列）和 `is_retrieval_correct`（gold 是否出现在某个 `<information>` 块里）——**但没有被接进 `_select_rm_score_fn`，等于摆着没用**。这为我们的改动提供了现成零件。

### 1.2 优势估计与更新（`verl/trainer/ppo/`）

- **GRPO 优势**（`core_algos.py:compute_grpo_outcome_advantage`）：把终局标量 `r_i` 按 prompt 分组（`uid`），做
  `Â_i = (r_i − mean(group)) / (std(group) + ε)`，再**广播到整条响应的每个 token**。
- **策略损失**（`core_algos.py:compute_policy_loss`）：标准 PPO clip，**对称、单一 `cliprange=0.2`**。
- **熵/KL**（`dp_actor.py:update_policy`）：`policy_loss = pg_loss − entropy_coeff·entropy + kl_loss_coef·KL(π‖π_ref)`；GRPO 用 `use_kl_loss=true, kl_loss_type=low_var_kl`。
- **状态掩码**：检索回来的 `<information>` token 通过 `loss_mask` 被排除出策略损失（`_create_loss_mask`），这一步做得对，保留。
- **分组结构**：`fit()` 里 `batch.repeat(n_agent)` 把每个 prompt 复制 `n_agent` 份作为 GRPO 组，`uid = index`（数据集行号）标识同组（`ray_trainer.py:702, 744`）。

---

## 2. 诊断：4 个真问题（"为什么现有方案弱"）

| # | 问题 | 机制 | 在我们场景（小模型+小库）的严重度 |
|---|------|------|----------|
| **P1** | **信号几乎全 0 → 梯度为 0** | 二值稀疏奖励下，一个 prompt 的一组 rollout 经常**全 0**；此时 `mean=0, r−mean=0`，整组优势归零，**这组白跑** | ⚠️ **头号病根**。小模型很少答对 → 几乎每组全 0 → 基本学不动 |
| **P2** | **搜索行为本身没被直接奖励** | 只有终局 EM，"学会搜索"只能靠又稀又噪的终局信号间接传导；检索质量、该不该搜都没人管 | 高 |
| **P3** | **EM 太脆 vs F1 易被刷 的两难** | EM 不可刷但极稀疏（"改述即判 0"）；F1/子串稠密但会被"长答案碰运气"刷分 | 高 |
| **P4** | **优势是一个标量摊到所有 token** + 小模型熵坍缩 | 一条好轨迹里废话 `<think>`、有用 search、最终 answer 拿到**完全一样**的优势，没有 turn 级信用分配；对称 clip 又容易早早冻住策略 | 中（turn 级信用是更深的问题，本轮先不动） |

---

## 3. 核心洞察：两个"病根"其实是同一个

P1（奖励端全 0）和"动态采样要解决的问题"（估计端全同分组白跑）**是同一个东西的两面**：

- **奖励端**：把奖励变稠（EM→F1、加过程分），组内 rollout 就不再"全同分"，组本身就有梯度；
- **估计端**：动态采样是"事后把全同分组丢掉、用有信息的组补齐"。

> 由此得到本项目的**核心研究假设**（见 §6）：
> **奖励一旦变稠，动态采样的边际收益就会大幅下降——二者是替代关系而非纯叠加。**

这个洞察决定了我们的实验一定要做成 **2×2（奖励轴 × 算法轴）**，才能读出交互项，而不是简单地"两个 trick 叠一起看涨了几个点"。

---

## 4. 文献调研（2024–2025，两轴，每条：机制 / 治哪个病 / 效果 / 代码）

### 4.1 奖励塑形轴

- **Search-R1**（2503.09516，基线本身）：`r = EM`，无格式奖励（作者说模型已自觉遵守结构，把格式塑形留给未来工作），检索 token 掩码。→ 我们要改进的正是它 P1/P2 的短板。
- **R1-Searcher**（2503.05592）：两阶段。阶段1（学会搜）`R = R_retrieve + R_format`，`R_retrieve=0.5 if 检索≥1 else 0`，**故意忽略答对与否**先把"搜索习惯"引导出来；阶段2（学对）`R = F1 + R_format`，格式项翻成**非法输出罚 −2**。→ 治 P1（用 F1）+ 冷启动"从不搜索"。
- **R1-Searcher++**（2505.17005）：`R = R_format + R_answer + R_group`。答案分用 **Cover-EM**（gold 被包含即算对）+ **≤10 词硬上限**防刷；新增 **group reward**：同一问题的正确回答里，用最少检索次数的给奖励。→ 治**过度搜索**。
- **ReSearch**（2503.19470）：分段 `r = F1 (F1>0) / 0.1 (F1=0 但格式对) / 0 (格式错)`。**格式只当"答错时的安慰分/闸门"**，防止模型放弃结构。→ 治 P1（F1）。
- **AutoRefine**（2505.11277，NeurIPS'25）：加 `<refine>` 蒸馏步 + **检索奖励** `R_ret = 1[gold ⊆ refine]`；非线性组合：答对=1.0，答错但检索命中=0.1，否则 0。→ **给"检索到了但没用好"以部分信用**，直接补 P2/P1。
- **StepSearch**（2505.15107，EMNLP'25）：最丰富的过程奖励。全局 `F1(answer)+γ·F1(search_keys)`；每步 token 级 `r_step = 信息增益 − 冗余惩罚`（信息增益=本轮检索文档对 gold 支撑文档的 TF-IDF 相似度较历史最大值的正向提升；冗余=本轮重复出现过的文档比例）。→ 强力治 P1/P2。**⚠️ 需要 gold 支撑文档标注 → 只适用多跳数据（我们的 NQ 单跳没有）**。
- **ZeroSearch**（2505.04588）：奖励仅 F1；贡献在于**用 LLM 模拟检索文档**（课程式逐步降质），省 API/带宽。→ 与我们带宽受限的处境相关，备选。
- **R-Search**（2506.04185）：**显式消融 EM vs Cover-EM vs F1，结论 F1 最好，相对 EM 平均 +52.6%**，并论证 EM 诱发 reward hacking / 脆性优化。→ "训练用 F1 而非 EM" 的直接实证依据。
- **HiPRAG**（2510.07794）：分层过程奖励，把每步标记为 最优/冗余(过搜)/不足(欠搜) 并分别奖惩。→ 同时治过搜与欠搜。（需 novel-info 判定，偏重多跳）
- **DeSA**（2510.04695）：把单一终局信号**拆成 search reward + answer reward**，因为终局奖励分不清"蒙对"和"真检索"。→ 治 P2。
- **格式奖励的坑**（"One Token to Fool"，2507.08794）：**正向格式奖励极易被刷**（一个 `Thought:` 就能骗到 60–90% 假阳性）。→ 结论：**格式要当"闸门/惩罚"，不当"正向 bonus"**。

**奖励轴小结（给我们的可落地结论）**：
1. **EM→词级 F1 + 短答案上限** 是单点最高性价比、几乎零成本（治 P1、P3）。
2. **格式当 gate/惩罚**，不当 bonus（治 P3 的刷分侧）。
3. **gold∈检索 的过程分**（AutoRefine 式，仓库已有 `is_retrieval_correct`）几乎零成本补 P2。
4. **小的冗余/搜索次数惩罚**（系数要小，否则知识密集问题会直接不搜）。
5. **放弃** StepSearch/HiPRAG 的信息增益 step 奖励（需多跳标注，NQ 不具备）。

### 4.2 GRPO 算法轴

- **DAPO**（2503.14476，代码在 verl）：4 个可组合 trick。
  - **clip-higher**：把 PPO clip 拆成 `ε_low=0.2 / ε_high=0.28`，只抬上界，给"低概率但正优势"的探索 token 成长空间 → **治熵坍缩**。
  - **dynamic sampling**：过滤掉一组内**全对或全错**的 prompt（要求 `0<正确数<G`），重采样补满 → **直接治 P1（全 0/全 1 组零梯度）**，稀疏二值奖励下最相关。
  - **token-level loss**：按 batch 总 token 归一（而非按序列平均）→ 治长响应的长度加权偏置。
  - **overlong reward shaping** + **丢掉 KL 项**（32B base 场景）。
- **Dr. GRPO**（2503.20783）：去掉两个归一化。
  - 去 **÷std**：二值奖励下 `std=√(p(1−p))`，对"太易/太难"的 prompt 特别小，优势被放大 → **去难度偏置**（我们二值场景正中要害）。
  - 去 **按响应长度平均 1/|o_i|**：否则"错误答案越长每 token 惩罚越小"→ 模型学会把错答案拉长 → **去长度偏置**。
  - **最便宜、最低风险**：优势函数改两行。**注意：单独用它救不了 P1**（全 0 组减完均值还是 0）。
- **GSPO**（2507.18071，Qwen3）：**序列级重要性比 + 序列级 clip**，修 GRPO 的 per-token 比高方差、长序列累积不稳（MoE 尤甚）。GSPO-token 变体保留 per-token 优势，适合多轮。→ 中等改动、动核心 loss，本轮先不上。
- **RLOO / LOOP**（2402.14740 / 2502.01600）：**留一法(leave-one-out) 基线**（用同组其他样本均值当 baseline，无偏）；LOOP 把它移植到多轮 agent。→ 便宜的优势替代方案，备选。
- **GiGPO**（2505.10978，NeurIPS'25）：**两级优势**（episode 级 GRPO + step 级 anchor-state 分组）做 critic-free 的 turn 级信用。→ 原理最对味 P4。**⚠️ 关键限制：它靠"相同环境状态重现"，而开放域 QA 搜索状态几乎不精确重现 → 原版 anchor 分组会失效，得退化成按 turn/工具调用边界分组**。高投入，放后续。
- **熵坍缩机理**（2505.22617）：熵变 ∝ token log-prob 与其优势的协方差，少数高协方差 token 主导坍缩；缓解：Clip-Cov / KL-Cov（只约束高协方差 token）/ 熵 bonus / clip-higher。**纠偏**（2509.26114）：熵的好处其实来自 clip-**low** 让低概率正优势 token 长大，**别只靠 clip-higher**，小模型要配熵保护。
- **KL 取舍**：GRPO 保 KL（稳、限漂移、但压探索）；DAPO 直接丢（大 base 模型要大幅偏离初始）。→ **小模型+稀疏奖励别裸奔**，用小/衰减 KL 或 KL-Cov 是更稳的中间路线。

**算法轴小结（给我们的可落地结论，按性价比排序）**：
1. **Dr.GRPO 去偏**（去 ÷std + 去长度平均）——几乎免费，二值场景收益最大（治 P4 的偏置侧）。
2. **DAPO 动态采样**——对头号病根 P1 的正解，收益最大（但需采样器改动、生成开销↑）。
3. **clip-higher + 熵保护**——一行改动防坍缩（治 P4 的探索侧）。
4. **token 级 loss 归一** + 我们已有的检索 token 掩码天然契合。
5. **turn 级信用（GiGPO 改良）**——最费、放后续。

---

## 5. 设计决策：我们选什么、为什么、以及为什么不选别的

### 5.1 奖励 bundle `R+`（全在一个 reward 函数里，复用现有 `is_valid_sequence` / `is_retrieval_correct`）
```
R+ =  [格式合法? 用 is_valid_sequence 做 gate]
      + F1(pred[:≤N词], gold)               # 稠密答案分，治 P1/P3；短答案上限防刷
      + λ_ret · 1[gold ⊆ 某个 <information>] # 检索命中过程分 0.1~0.2，治 P2（复用 is_retrieval_correct）
      − λ_pen · (冗余/多余搜索次数)           # 小惩罚，治过度搜索（系数小）
```
- **为什么是 F1 不是 Cover-EM**：R-Search 实证 F1 最好；F1 天生惩罚啰嗦，配短答案上限双保险。
- **为什么格式当 gate 不当 bonus**："One Token to Fool" 表明正向格式奖励会被刷；gate 化零成本且不可刷。
- **为什么不上 StepSearch/HiPRAG 信息增益**：需 gold 支撑文档标注，NQ 单跳没有（要上就得换多跳数据集，是另一条岔路）。
- **为什么不上 LLM-as-judge 奖励**：每 rollout 一次判定太贵、且可被刷（同上文献），与我们算力/带宽约束冲突。

### 5.2 算法 bundle `A+`（分别落在优势函数 / clip / 采样器三处，互不干扰）
```
A+ = Dr.GRPO 去偏(去÷std, 可选去长度平均)   # core_algos.compute_grpo_outcome_advantage
   + clip-higher(ε_low/ε_high) + 熵 bonus  # core_algos.compute_policy_loss + dp_actor
   + 动态采样(过滤全同分组, 有界超采补齐)     # ray_trainer.fit（带 flag，最谨慎）
```
- **为什么这三个**：正好各治一个不同的病（P4偏置 / P4探索 / P1零梯度），且都能定位到单一函数、互不耦合 → 组成**干净可消融**的链条，符合汇报需要"能把每个改动的作用讲清楚"。
- **为什么保留（小）KL**：小模型+稀疏奖励下裸奔易早坍缩，遵循文献用小系数 KL 兜底，而非 DAPO 式直接丢。
- **为什么 GSPO/GiGPO/RLOO 本轮不上**：GSPO/GiGPO 要动核心 loss/需状态重现，投入大、对我们单跳小规模收益不确定；作为后续加强项列在 §11。

---

## 6. 核心研究假设（可证伪，是报告的骨架）

- **H1（稠密化恢复梯度）**：`R+` 相比二值 EM，会显著提高"非零奖励比例"与"非全同分组比例（即非零优势组比例）"，从而 actor 每步拿到更多有效梯度。
- **H2（去偏改善稳定性/长度）**：Dr.GRPO 去偏会降低对"过易/过难"prompt 的优势放大，并抑制错误答案的长度膨胀（response length 更短更稳）。
- **H3（探索保持）**：clip-higher + 熵 bonus 会减缓策略熵的早期坍缩（entropy 曲线下降更慢）。
- **H4（替代关系，核心）**：**当奖励已用 `R+` 稠密化后，再叠加动态采样带来的增益，明显小于在二值 EM 上叠加动态采样的增益**（即 2×2 存在负交互项）。同时，在二值 EM 下开动态采样会**丢弃绝大多数组**（可量化，见 §7 指标），这本身就佐证 P1 的严重性。

---

## 7. 实验设计

### 7.1 主矩阵（2×2）
| | vanilla GRPO | `A+`（去偏+clip-higher+动态采样） |
|---|---|---|
| **EM 二值奖励** | ① baseline | ③ |
| **`R+` 稠密奖励** | ② | ④ |
- 读法：②−① = 奖励主效应（H1）；③−① = 算法主效应（H2/H3/H4的采样部分）；(④−②)−(③−①) = **交互项（H4）**。
- 若算力紧：精简成 3 个 run ①/②/④ 也能讲主线。

### 7.2 关键指标（重点看**训练动态**，不是 val EM）
> 说明：本机 10 文档小库 + 0.5B 模型，val EM≈0 且不是本阶段的重点；**我们比较的是"机制层面"的量**，这些量在小规模下就会有区别，正好对应 H1–H4。
- `reward/mean`、**`reward/nonzero_frac`（非零奖励比例）**
- **`grpo/nonuniform_group_frac`（非全同分组比例 = 有非零优势的组占比）** ← 验证 H1/H4 的核心量（需埋点）
- `dynamic_sampling/kept_frac`（动态采样保留比例）← 佐证 P1
- `actor/entropy_loss`（熵）← H3
- `response_length`（平均响应长度）← H2
- `actor/pg_clipfrac`、`actor/ppo_kl`、`actor/grad_norm`、`env/number_of_valid_search`

### 7.3 次要（若时间允许）
- 构造一个**答案可检索的迷你语料**（几百文档，保证选定问题的 gold 出现在库中）把 base 成功率抬到 >0，观察是否出现 val EM 的分化。

---

## 8. regime 决策与理由（为什么这么定）

- **数据/模型/库**：沿用已跑通的 smoke 底座——NQ 数据、10 文档 e5 索引、Qwen2.5-0.5B-Instruct、2 GPU 全量 FSDP offload、检索服务器 :8002。
- **为什么不换大库/大模型**：本机带宽 ~0.4MB/s、GPU 与他人 peft 作业共享（每卡 ~12GB 空闲），换大库(65GB wiki)/大模型不现实（见 `REPRODUCTION_zh.md` §6）。
- **为什么这样仍然有意义**：我们的假设 H1–H4 都是**机制层面**的（梯度信号是否被丢、熵是否坍缩、长度是否膨胀），**在稀疏小规模下反而更容易暴露**——正是二值 EM 让信号消失的地方。把它当成一个"受控诊断台"而非"刷 SOTA"。
- **动态采样的特别说明**：稀疏 regime 下"重采样到满"可能永远采不到有信息的组而**挂死**。因此我们实现的是**有界超采+过滤**版本（超采 k 倍→优先选非全同分组→不足则回退填满，绝不无限循环），并默认关闭、先小步验证再决定是否进长跑。

---

## 9. 代码改动清单（文件 / 函数 / flag；便于回退）

| 改动 | 文件 : 函数 | 新增 flag（默认值=保持原行为） |
|---|---|---|
| 奖励类型分发 | `verl/trainer/main_ppo.py : RewardManager` | `data.reward_type=em`（可选 `f1_shaped`） |
| F1+gate+检索分+惩罚 | `verl/utils/reward_score/qa_em.py`（新增 `compute_score_f1_shaped`） | 复用 `qa_em_format` 的零件 |
| Dr.GRPO 去偏 | `verl/trainer/ppo/core_algos.py : compute_grpo_outcome_advantage` | `algorithm.norm_adv_by_std=True` |
| 非全同分组埋点 | 同上 | 始终输出到 metrics |
| clip-higher | `core_algos.py : compute_policy_loss` + `dp_actor.py` | `actor.clip_ratio_low/high`（缺省=`clip_ratio`） |
| 动态采样 | `ray_trainer.py : fit` | `algorithm.dynamic_sampling=False`, `.oversample_factor` |

每个改动**单独一个 git commit**，commit message 写清"改了什么+为什么"，随时可 `git revert`。

---

## 10. 运行命令（示例，实际脚本见 `run_scripts/`）
```bash
# baseline ①
bash run_scripts/train_smoke_grpo.sh            # data.reward_type=em, vanilla
# 奖励主效应 ②
... data.reward_type=f1_shaped
# 算法主效应 ③
... algorithm.norm_adv_by_std=False actor.clip_ratio_high=0.28 algorithm.dynamic_sampling=True
# 全叠加 ④
... data.reward_type=f1_shaped algorithm.norm_adv_by_std=False actor.clip_ratio_high=0.28 algorithm.dynamic_sampling=True
```

---

## 11. 结果（实验后回填）

> 见 §7 指标；每个 run 的日志在 `/mnt/backup1/lgc/search-r1-data/exp_logs/`。

_（待填）_

---

## 12. 后续加强项（本轮不做，列出便于规划）
- **turn 级信用**：GiGPO 改良版（按 turn/工具调用边界分组，而非 anchor-state）——治 P4 最彻底。
- **信息增益过程奖励**：换多跳数据集（HotpotQA/2Wiki/MuSiQue）解锁 StepSearch/HiPRAG（2510.14967 是正对口的 Info-Gain PO for search agents）。
- **GSPO-token**：序列级 IS，若出现长序列/多轮不稳再上。
- **检索后端**：HNSW / BM25 / rerank / 在线搜索。

---

## 参考文献（arXiv）
奖励：Search-R1 2503.09516 · R1-Searcher 2503.05592 · R1-Searcher++ 2505.17005 · ReSearch 2503.19470 · AutoRefine 2505.11277 · StepSearch 2505.15107 · ZeroSearch 2505.04588 · R-Search 2506.04185 · HiPRAG 2510.07794 · DeSA 2510.04695 · LeTS 2505.17447 · One-Token-to-Fool 2507.08794 · survey 2510.16724
算法：DAPO 2503.14476 · Dr.GRPO 2503.20783 · GSPO 2507.18071 · RLOO 2402.14740 · LOOP 2502.01600 · GiGPO 2505.10978 · Entropy-Mechanism 2505.22617 · clip-low/high 2509.26114 · Info-Gain-PO(search) 2510.14967
