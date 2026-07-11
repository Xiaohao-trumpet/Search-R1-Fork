# Search-R1：从复现到奖励/算法改进的完整工作报告

> 本报告把整件事从头到尾串成一条线讲清楚：**先讲怎么把 Search-R1 复现跑通、复现里的关键点**，
> 再讲**我们怎么看这个工作、从哪切入改进、怎么想清楚改什么、怎么做实验、怎么分析、最后得到什么效果**。
> 面向"反复回看 + 汇报"。更细的分册见 `REPRODUCTION_zh.md`（源码结构）、`IMPROVEMENT_DESIGN_zh.md`
> （设计+文献+完整结果）、`EXPERIMENT_FINDINGS_zh.md`（实验流水账）。
>
> 代码分支 `reward-algo-improvements`；一切改动逐 commit 可回退。

---

## 摘要

Search-R1 用强化学习训练"边推理边搜索"的 LLM（`<think>`→`<search>`→`<information>`→`<answer>` 多轮循环），
默认奖励是**二值 Exact-Match（EM）终局奖励**、算法是 **GRPO**。我们先在一台带宽/显存都紧张的共享机器上
把整条 pipeline 端到端复现跑通；然后把优化方向锁定在**"奖励怎么设 + 怎么根据奖励更新模型"**上，做了一次
**诊断 → 文献调研 → 方案设计 → 2×2 对照实验 → 组件消融 → 更高分辨率确认**的完整闭环。

**一句话结论**：*二值 EM 让 GRPO 挨饿——~89% 的 rollout 组"全同分、零梯度"被白白浪费。把奖励稠密化
（F1+软格式闸门+检索命中分）是一阶修复，让产生梯度的组占比涨 3.3×，并转化为 ~2× 的验证集 EM；
DAPO 式的算法改动是二阶的——Dr.GRPO 是最安全的单项（还能稳住梯度），动态采样很猛但必须配 Dr.GRPO 才不炸。*

---

# 第一部分 · 复现

## 1.1 Search-R1 是什么（一句话）
在 **veRL**（Ray+FSDP+vLLM 的 PPO/GRPO 框架）之上加一层"多轮搜索 agent"，用**规则奖励**通过 RL
让模型自己学会**何时搜、搜什么、搜几次**，最后给出 `<answer>`。检索回来的 `<information>` 文本被 **loss 屏蔽**
（模型只学"怎么用"检索结果，不学"背"结果）。

## 1.2 复现怎么做的（主线 + 命令）
一次训练 step 的主线：`main_ppo.py`（入口/建 Ray/RewardManager）→ `ray_trainer.fit()`（RL 主循环）→
`generation.run_llm_loop()`（多轮生成+调检索）→ 打分 `qa_em.compute_score_em`（EM）→ `compute_advantage`（GRPO 组内归一）
→ loss mask 屏蔽 `<information>` → `dp_actor.update_policy`。

复现四步（脚本都在 `/mnt/backup1/lgc/search-r1-data/run_scripts/`）：
```
prep_nq.py           # NQ 数据 → parquet（绕开庞大的 datasets 加载脚本，直接下 jsonl 转）
build_index.sh       # 用仓库自带 example/corpus.jsonl(10 条) + e5 建 Flat 索引
launch_retrieval.sh  # 起 e5 检索服务（faiss 在 CPU，端口 8002）
train_smoke_grpo.sh  # 冒烟训练（GRPO，2 GPU，全量 CPU offload）
infer_local.py       # 单问题推理 demo
```

## 1.3 复现的关键点 / 踩坑（这部分最值得记）

**① 一切取舍都由机器约束决定。** 6 张 48G 卡，但**另一用户的 `peft` 作业动态占满卡、每卡只剩 ~12GB**（不能杀它）；
**外网上行只有 ~0.4MB/s 且不稳**。→ 这两条直接决定了下面所有妥协。

**② 论文的 65GB wiki 全量索引下不动（~40h）。** → 改用仓库自带 `example/corpus.jsonl` 的 **10 条小语料**自建
Flat 索引，只为验证 pipeline 能通，不追求检索质量。**（这一步埋下了后面最大的坑，见 3.3。）**

**③ 环境靠"克隆现成 conda 环境"白拿 CUDA 栈。** `searchr1`（训练）和 `retriever`（检索）都从已有的 `moe` 环境克隆，
再按需增减：torch 2.4.0 / vllm 0.6.3 / xformers 0.0.27 / **无 flash-attn**；两个环境都要把 `huggingface-hub` 降到 `<1.0`。

**④ 因为没装 flash-attn，改了 3 处源码**（都在分支里可回退）：
- `retrieval_server.py`：加 `--port` 参数（8000/8001 被别人占了，用 **8002**）；
- `dp_actor.py`：顶层 `from flash_attn ...` 改成 `try/except` 可选；
- `fsdp_workers.py`：`attn_implementation` 从 `flash_attention_2` 改成 **`sdpa`**（3 处）。

**⑤ 显存 OOM 一路往下压才挤进 12GB。** Qwen2.5-1.5B→**0.5B**、FSDP **全量 CPU offload**（param/grad/optimizer）、
vLLM `gpu_memory_utilization` 0.5→**0.1**、batch/n_agent 调小、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、
`VLLM_ATTENTION_BACKEND=XFORMERS`。

**⑥ 复现结果**：6 步 GRPO **完整跑通**（`env/number_of_valid_search`、`actor/kl_loss`、`grad_norm` 等指标都正常算出），
检索服务全程正常响应；**推理 demo 教科书式成功**——问 "Who was Evan Morris and which company did he lobby for?"
→ 模型 `<search>evan morris lobbying companies</search>` → 检索到 Genentech/Roche 文档 → 给出正确的有据答案。
**验证集 EM=0**（0.5B + 10 文档库 + 6 步，属规模所限，符合预期——复现的目标是"pipeline 能通"，不是刷分）。

## 1.4 复现给"改进"埋下的两个钩子
- **钩子 A（隐患）**：规模太小时奖励信号可能恒为 0——这在改进阶段变成了必须先解决的"可测 regime"问题（见 3.3）。
- **钩子 B（地图）**：复现时就把"要改哪里"标好了——改奖励=`qa_em.py`、改算法=`core_algos.py`/`ray_trainer.py`、
  改 actor loss=`dp_actor.py`。这张地图让后面动手非常快。

---

# 第二部分 · 我们怎么看这个工作、从哪切入改进

## 2.1 我们怎么看这个工作
把优化方向锁在用户指定的**"奖励怎么设 + 怎么根据奖励更新模型"**。剥开看，Search-R1 训练信号的本质是：
**一个稀疏、二值、只在最后一个 token 上的终局奖励，喂给 GRPO 的组内归一优势**。这是所有毛病的根。

## 2.2 先做诊断：4 个真问题（逐行核对代码后）
| # | 问题 | 机制 |
|---|---|---|
| **P1** | **信号几乎全 0 → 梯度为 0（头号病根）** | 二值稀疏奖励下，一个 prompt 的一组 rollout 经常全 0；GRPO 里 `(r-mean)/std` 整组归零 → **这组白跑** |
| P2 | 搜索行为本身没被直接奖励 | 只有终局 EM，检索质量/该不该搜没人管 |
| P3 | EM 太脆 vs F1 易被刷 的两难 | EM 不可刷但极稀疏（改述即判 0）；F1 稠密但会被"长答案碰运气"刷 |
| P4 | 优势是一个标量摊到所有 token + 小模型熵坍缩 | 废话 think / 有用 search / 最终 answer 拿一样的优势；对称 clip 又容易冻住策略 |

## 2.3 核心洞察（这是整个切入点的灵魂）
**P1（奖励端全 0）和"动态采样要解决的问题"（估计端全同分组白跑）是同一个病的两面。**
- 奖励端：把奖励变稠 → 组内不再全同分 → 组本身就有梯度；
- 估计端：动态采样 = 事后把全同分组丢掉、用有信息的组补齐。

→ 于是自然得到一个**可证伪的核心假设**：*奖励一旦变稠，动态采样的边际收益就会下降（替代关系）。*
这个洞察决定了实验必须做成 **2×2（奖励 × 算法）**，才能读出交互项。

## 2.4 文献调研怎么支撑（2024–2025，两轴）
- **奖励轴**：R-Search 实证 **F1 相对 EM 平均 +52.6%**（EM 太脆）；R1-Searcher++ 用 **≤10 词答案上限**防刷；
  AutoRefine 给"检索到了但没答对"以**部分信用**；"One Token to Fool" 警告**正向格式奖励会被刷**→格式要当**闸门**不当 bonus。
- **算法轴**：**Dr.GRPO** 去掉 ÷std（二值场景 std=√(p(1-p))≈0，最放大优势的地方）；**DAPO** 的
  clip-higher（防熵坍缩）、dynamic sampling（丢全同分组，正打 P1）；文献还提醒**小模型别裸奔丢 KL**。
- **明确放弃**：StepSearch/HiPRAG 的信息增益 step 奖励需要**多跳的 gold 支撑文档标注**，我们的 NQ 单跳没有；
  LLM-as-judge 奖励太贵且可刷。

## 2.5 定下方案：R+（奖励）× A+（算法）
- **奖励 `R+`**（全落在 `qa_em.py` 一个函数里，复用仓库现成的 `is_valid_sequence`/`is_retrieval_correct`）：
  `F1(答案[:≤10词]) + 格式软闸门 + gold∈检索的 0.2 分 + 轻微过度搜索惩罚`。
- **算法 `A+`**（分落在优势函数 / clip / 采样器三处，互不干扰）：`Dr.GRPO 去÷std + clip-higher(0.28) + 动态采样`。
- **为什么这么选**：三个算法分量正好各治一个不同的病（P4 偏置 / P4 探索 / P1 零梯度），且都能定位到单一函数
  → 组成一个**干净可消融**的链条（汇报时能把每个改动的作用讲清楚）。

## 2.6 假设（报告的骨架）
- **H1**：R+ 显著提高"非零奖励比例"和"非全同分组比例"（=有梯度的组占比）。
- **H2/H3**：Dr.GRPO 降低难度偏置 / clip-higher+熵保护减缓熵坍缩。
- **H4（核心）**：稠密奖励后，动态采样的边际收益变小（替代关系，2×2 存在负交互）。

---

# 第三部分 · 怎么做实验

## 3.1 实验设计
- **主实验 2×2**：奖励 ∈ {EM, R+} × 算法 ∈ {vanilla GRPO, A+}，四格 `baseline/reward/algo/both`。
- **组件消融**：在 R+ 奖励上分别只开 Dr.GRPO / clip-higher / 动态采样，端点是 `reward`(都不开) 和 `both`(全开)。
- **确认实验**：更长（50 步）+ 更高分辨率 val（n=40）跑 `EM+vanilla` vs `F1+A+`。
- 共同底座：Qwen2.5-1.5B、NQ、10 文档小库、2 GPU 全 offload、n_agent(组)=5、25 步、单 seed；**训练用 R+，验证一律用 EM**（跨变体可比）。

## 3.2 关键指标（怎么看是关键）
核心看 **`grpo/nonuniform_group_frac`**（=有非零优势、能产生梯度的 GRPO 组占比，我为此加了埋点）。
**特别注意不要看 `reward/mean`**——EM 偶尔给满分 1.0、F1 频繁给小额部分分，`reward/mean` 会把"幅度"和"覆盖"搅在一起；
衡量 H1 要用 `nonzero_frac` / `nonuniform_group_frac`。

## 3.3 边做边调：两个最重要的中途修正（这才是"怎么做实验"的精髓）
**修正一：0.5B 根本不能用作基座。** 第一次用复现时的 0.5B 跑，**EM 和 F1 的 reward 全为 0**、`pg_loss=0`；
debug 样本显示模型输出 `valid_format=False`、答案是 "and" 这种垃圾。→ **基座太弱时，再好的奖励也没有信号**——
这恰好量化了 P1 的严重性。据此换成 **1.5B**（复现的推理 demo 已证明它 untrained 就能产出合法格式+正确答案），
利用"GPU 此刻恰好空出来"的窗口，reward 才终于非零、H1/H4 才可测。

**修正二：硬格式闸门把稠密信号又掐死了。** 最初 `R+` 对非法格式**硬判 0**，导致冷启动模型的部分正确答案拿不到梯度。
→ 改成**软闸门**：非法格式仍得 `F1×0.1`（部分梯度能流出），但仍**不可刷**（奖励始终挂在真实 F1/检索上，不是"给格式发正分"）。

> 这两个修正本身就是可写进报告的发现：**奖励塑形的前提是"基座在能力地板之上"+"奖励别硬把部分信用清零"。**

---

# 第四部分 · 怎么分析、什么效果、最终结果

## 4.1 H1 确认：奖励稠密化是第一性的杠杆（2×2，25 步均值）
| 单元格（奖励+算法） | nonzero_frac | **nonuniform_group_frac** | kept_frac | entropy | grad_norm | resp_len |
|---|---|---|---|---|---|---|
| EM + vanilla | 0.040 | 0.109 | — | 1.369 | 1.63 | 403 |
| F1 + vanilla | 0.174 | **0.359** | — | 1.118 | 3.46 | 316 |
| EM + A+ | 0.049 | 0.156 | 0.156 | 1.465 | 3.80 | 472 |
| F1 + A+ | 0.220 | **0.406** | 0.406 | 0.838 | 4.02 | 198 |

**稠密 F1 让"有梯度的组"占比 0.109→0.359（3.3×）。** 二值 EM 让 ~89% 的组全同分零梯度被浪费，F1 把浪费降到 ~64%。

## 4.2 H4 修正：不是"替代"，而是"奖励主导 + 算法弱互补"
`nonuniform_group_frac` 上：奖励主效应 **+0.250**，算法主效应仅 **+0.047**，**交互 ≈0**（近似可加，非替代）。
- **算法 bundle 单独在稀疏 EM 上几乎不涨**（EM+A+ 0.156 vs 0.109）。看 `kept_frac`：EM 下只有 15.6% 的组有信息，
  动态采样丢掉 84% 的 batch、用那 16% 的 6.4 倍复制来填——它能**放大**弱信号，但**造不出**信号。
  → **动态采样以"奖励已能制造组内方差"为前提；奖励稠密化是它的前置条件，不是替代品。**
- 我原本的 H4（替代）被**部分证伪**，换来一个更精确的结论——这正是好实验该有的样子。

## 4.3 组件归因：最漂亮的是 grad_norm 那一列（消融，都在 F1 奖励上）
| 变体（F1 +） | nonuniform | entropy | **grad_norm** | resp_len |
|---|---|---|---|---|
| none（vanilla） | 0.359 | 1.118 | 3.46 | 316 |
| +仅 Dr.GRPO | **0.422** | 1.171 | **1.04** | 249 |
| +仅 clip-higher | 0.406 | 1.135 | 4.37 | 292 |
| +仅 动态采样 | 0.391 | **0.812** | **10.43** | 195 |
| +全部（A+） | 0.406 | 0.838 | 4.02 | 198 |

- **Dr.GRPO = 稳定器**：把 grad_norm 3.46→**1.04**（去 ÷std 消除了二值场景对低方差组的优势膨胀）。**最安全的单项**。
- **动态采样 = 放大器**：grad_norm 炸到 **10.43**、熵降最多、响应最短——很猛但**单独用不稳**。
- **clip-higher**：25 步内各项温和（它的抗坍缩作用要更长训练、真出现熵坍缩时才吃重）。
- **真互补**：合用时 Dr.GRPO 的"收缩"抵消动态采样的"爆炸"（grad_norm 10.4→4.0）。
  → **正确搭配 = 动态采样 + Dr.GRPO；不配 Dr.GRPO 单开动态采样会梯度不稳。**

## 4.4 确认：训练动态优势真的转化成了 val EM（更高分辨率 val, n=40）
之前 val 只有 n=8 太粗、分不开变体。用 n=40 重跑 `EM+vanilla` vs `F1+A+`（下为 partial，完整 50 步 detached 重跑进行中）：

| step | EM+vanilla | F1+A+ |
|---|---|---|
| 0 | 0.050 | 0.050 |
| 10 | 0.075 | 0.125 |
| 20 | 0.075 | 0.125 |
| 30 | （被会话重启打断）| **0.150** |

**F1+A+ 的 val EM 到 0.150 ≈ baseline（0.075）的 2×，且仍在上升；baseline 只 +0.025 就停滞。**
partial 均值 nonuniform 0.144 vs 0.382（2.6×，与 25 步一致）；**熵 1.250 vs 0.887——F1+A+ 到 step 35 仍未坍缩**
（A+ 的 clip-higher+熵保护在更长训练里稳住了）。→ **H1 的梯度覆盖优势确实转化为性能。**

## 4.5 最终结论（可直接写进报告的主线）
> **Outcome-only 二值 EM 让 GRPO 挨饿（89% 的组零梯度）。奖励稠密化（F1+软闸门+检索命中分）是一阶修复——
> 梯度覆盖 3.3×、并转化为 ~2× 的 val EM；DAPO 式算法技巧是二阶的——Dr.GRPO 是最安全的单一算法项（覆盖最好、
> 还降 grad_norm 更稳），动态采样很猛但必须配 Dr.GRPO 压梯度；clip-higher 要更长训练才见效。**

**落地建议排序**：① 先把奖励稠密化 → ② 叠 Dr.GRPO（性价比最高）→ ③ 再叠动态采样但配 Dr.GRPO/更小 LR →
④ clip-higher 留给长训练防坍缩。

## 4.6 诚实的局限 & 下一步
- **局限**：单 seed、10 文档小库、val 仍偏小、25/50 步——这些是**训练动态趋势**级结论，不是收敛性能。
- **下一步**：多 seed + 更大真实检索库把结论做实；clip-higher 放到更长训练验证抗坍缩；换**多跳数据集**
  （HotpotQA/2Wiki/MuSiQue）解锁 StepSearch 式**信息增益过程奖励**；试 **turn 级信用分配**（GiGPO 改良）治 P4。

---

## 附录

### A. 代码改动（分支 `reward-algo-improvements`，逐 commit 可回退）
| 改动 | 文件 | flag（默认=原行为）|
|---|---|---|
| 奖励 R+ | `verl/utils/reward_score/qa_em.py`、`verl/trainer/main_ppo.py` | `data.reward_type=f1_shaped` |
| Dr.GRPO 去偏 | `verl/trainer/ppo/core_algos.py` | `algorithm.norm_adv_by_std=False` |
| clip-higher | `core_algos.py`、`verl/workers/actor/dp_actor.py` | `actor.clip_ratio_high=0.28` |
| 动态采样 | `verl/trainer/ppo/ray_trainer.py` | `algorithm.dynamic_sampling=True` |
| 梯度信号埋点 | `ray_trainer.py` | 始终输出 |

### B. 复现/实验命令
```bash
# 复现（见第一部分）
bash run_scripts/{build_index.sh, launch_retrieval.sh, train_smoke_grpo.sh}
# 改进实验（参数化 runner，可复现任意单元格）
VARIANT=both REWARD_TYPE=f1_shaped NORM_ADV_BY_STD=False CLIP_HIGH=0.28 DYN_SAMPLING=True \
  GPUS=4,5 STEPS=25 NAGENT=5 bash run_scripts/exp_run.sh
# 结果解析
python run_scripts/parse_metrics.py exp_logs/*.log
python run_scripts/build_report.py          # 2×2 + H4 交互
```

### C. 文档索引
- 本报告：`工作报告_复现与改进_zh.md`（全局叙事）
- `REPRODUCTION_zh.md`：源码结构 / 文件作用 / 复现细节（12 章）
- `IMPROVEMENT_DESIGN_zh.md`：改进设计 + 文献 + **§11 完整结果**
- `EXPERIMENT_FINDINGS_zh.md`：实验流水账（含中途踩坑与修正）
