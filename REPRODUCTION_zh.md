# Search-R1 复现文档（中文）

> 本文档记录了在本机对 **Search-R1**（用强化学习训练"边推理边搜索"的 LLM）的一次端到端复现：**代码结构、文件关系与作用、复现步骤、结果、运行日志，以及为后续"改进 Search-R1 写实验报告"准备的源码入手点。**
> 复现日期：2026-07-08。仓库路径：`/mnt/backup1/lgc/exp/Search-R1`。
>
> 复现产物统一放在 `/mnt/backup1/lgc/search-r1-data/`（数据、模型、索引、脚本、日志）。

## 目录

1. [Search-R1 是什么](#1-search-r1-是什么)
2. [代码结构总览 + 端到端数据流](#2-代码结构总览--端到端数据流)
3. [`search_r1/` 详解（Search-R1 自己的代码）](#3-search_r1-详解search-r1-自己的代码)
4. [`verl/` 详解（RL 训练框架）](#4-verl-详解rl-训练框架)
5. [配置与超参速查](#5-配置与超参速查)
6. [本次复现：环境与机器约束](#6-本次复现环境与机器约束)
7. [复现步骤（可照抄命令）](#7-复现步骤可照抄命令)
8. [复现结果与日志解读](#8-复现结果与日志解读)
9. [我对源码做的修改（patch 清单）](#9-我对源码做的修改patch-清单)
10. [面向改进的源码入手点⭐](#10-面向改进的源码入手点)
11. [踩坑与 FAQ](#11-踩坑与-faq)
12. [附录：关键文件与符号速查表](#12-附录关键文件与符号速查表)

---

## 1. Search-R1 是什么

**Search-R1** = 在 **veRL**（火山引擎的 Ray+FSDP+vLLM 的 PPO/GRPO 训练框架）之上，加一层**"多轮搜索 agent"**，用**规则奖励（answer 是否 Exact Match）**通过 RL 让模型学会：

```
<think> 推理 </think>  →  <search> 查询 </search>  →  <information> 检索结果 </information>
  →  ...（可多轮）...  →  <answer> 最终答案 </answer>
```

- 论文1：https://arxiv.org/pdf/2503.09516 ；论文2（更详细的实证）：https://arxiv.org/abs/2505.15117
- 相对普通 RAG 的关键区别：**检索时机、查询内容、检索次数都是模型自己学出来的**（RL 决定），而不是固定流程。
- 相对 DeepSeek-R1 的区别：把"纯推理"扩展成"推理 + 调用搜索引擎"。

**核心机制三件套**（后面会反复出现）：
1. **多轮 rollout**：`search_r1/llm_agent/generation.py` 里的循环，负责生成→解析动作→调检索→注入结果→继续。
2. **规则奖励**：`verl/utils/reward_score/qa_em.py`，答案 EM 命中给 1 分，否则 0 分。
3. **信息屏蔽（state/loss masking）**：检索回来的 `<information>` 文本**不参与 loss**（模型不该"背"检索结果，只学怎么用）。

---

## 2. 代码结构总览 + 端到端数据流

### 2.1 顶层目录

| 路径 | 作用 |
|---|---|
| **`search_r1/`** | Search-R1 自己加的东西：`llm_agent/`（多轮搜索 rollout）+ `search/`（各种检索服务器）。**这是理解 Search-R1 的重点。** |
| **`verl/`** | RL 训练框架（veRL 的一个 fork）。trainer / workers / 配置 / 奖励函数 / PPO&GRPO 算法都在这。 |
| `scripts/` | 数据预处理 `data_process/`、语料下载 `download.py`、各版本训练脚本 `nq_hotpotqa/{v0.1,v0.2,v0.3}/`。 |
| `example/` | 现成启动脚本：`retriever/*.sh`（bm25/ann/serpapi/google）、`multinode/*.sh`（32B/72B），以及 `corpus.jsonl`（10 条样例语料）。 |
| `docs/` | `retriever.md`（各种检索器怎么起）、`multinode.md`、`experiment_log.md`。 |
| `infer.py` | 单条问题的**推理 demo**（HF generate + 遇到 `</search>` 停下来 → 调检索 → 继续）。不参与训练。 |
| `train_ppo.sh` / `train_grpo.sh` | 顶层启动器 → `python -m verl.trainer.main_ppo`。二者只差 `algorithm.adv_estimator=gae`(PPO) vs `=grpo`(GRPO)。 |
| `retrieval_launch.sh` | 启动 e5 稠密检索服务（默认端口 8000）。 |
| `setup.py`/`pyproject.toml`/`requirements.txt` | 打包安装（包名就叫 `verl`）。 |

> ⚠️ 注意：`train_grpo.sh` 其实也是调 `main_ppo`（GRPO/PPO 只靠 `adv_estimator` 分流）；且它里面用了未定义变量 `$TRAIN_DATA_DIR`/`$TEST_DATA_DIR`（原仓库的小 bug，需要自己设或改成 `$DATA_DIR`）。

### 2.2 一次训练 step 的端到端数据流（最重要的一张图）

```
                          verl/trainer/main_ppo.py  (Hydra 入口, 建 Ray/tokenizer/RewardManager)
                                        │
                                        ▼
              verl/trainer/ppo/ray_trainer.py :: RayPPOTrainer.fit()   ← RL 主循环
                                        │
   ┌────────────────────────────────────┼─────────────────────────────────────────────┐
   │ 1. 取一个 batch, 复制 n_agent 份 (GRPO 分组)                                        │
   │ 2. 生成 (do_search=true):                                                          │
   │      search_r1/llm_agent/generation.py :: run_llm_loop()                           │
   │        for turn in range(max_turns):                                               │
   │          vLLM 生成 → 截到 </search> 或 </answer>                                    │
   │          execute_predictions(): 解析 <search>/<answer>                             │
   │            └── <search> ─POST /retrieve──▶  search_r1/search/retrieval_server.py   │
   │                            ◀── <information>...</information> ── 注入回上下文        │
   │        末轮再生成一次(do_search=false) 逼出 <answer>                                │
   │ 3. 算 log_prob(actor), ref log_prob(ref), [critic values 仅 GAE]                   │
   │ 4. 奖励: RewardManager → qa_em.compute_score_em (answer EM?) 打在最后一个 token     │
   │ 5. 优势: compute_advantage() → GRPO 组内归一 / GAE                                  │
   │ 6. Loss mask: _create_loss_mask() 把 <information> token 屏蔽掉                     │
   │ 7. 更新: dp_actor.update_policy() [+ dp_critic.update_critic() 仅 GAE]              │
   │ 8. 周期性 _validate() / _save_checkpoint()                                          │
   └────────────────────────────────────────────────────────────────────────────────────┘
```

记住这条主线，后面每一节都是在讲这条线上的某一环。

---

## 3. `search_r1/` 详解（Search-R1 自己的代码）

### 3.1 `search_r1/llm_agent/generation.py` — 多轮搜索的心脏 ⭐

- **`GenerationConfig`**（dataclass）：`max_turns, max_start_length, max_prompt_length, max_response_length, max_obs_length, num_gpus, no_think_rl, search_url, topk`。
- **`LLMGenerationManager`**：持有 `tokenizer`、`actor_rollout_wg`（vLLM actor worker group）、config、`TensorHelper`。核心方法：

| 方法 | 作用 |
|---|---|
| **`run_llm_loop(gen_batch, initial_input_ids)`** | 主循环。维护左侧(prompt，截到 `max_start_length`)和右侧(累积的 responses + `responses_with_info_mask`)。`for turn in range(max_turns)`：切 padding → 按 `active_mask` 取还没结束的样本 → 生成 → `execute_predictions` → 更新 `active_mask`。循环后**再生成一次 `do_search=False`** 逼出答案。同时统计 `turns/valid_action/valid_search`。 |
| `_postprocess_responses` | 解码后**在第一个 `</search>` 或 `</answer>` 处截断**（保留标签），让生成停在动作边界。 |
| **`execute_predictions(predictions, pad_token, active_mask, do_search)`** | 环境 `step`：解析动作，把所有 `search` 查询**合并成一次** `batch_search`；然后逐样本：`answer`→obs 空、`done=1`；`search`→obs=`\n\n<information>{结果}</information>\n\n`、`done=0`；**非法动作**→给一段"你上次动作无效，请用 `<search>` 或 `<answer>`"的提示文本、`done=0`（注意：非法动作只给提示，**不扣分**）。 |
| `postprocess_predictions` | 正则 `r'<(search\|answer)>(.*?)</\1>'` 抽出 `(动作, 内容)`。 |
| `batch_search` / `_batch_search` | POST `{queries, topk, return_scores:True}` 到 `config.search_url`。 |
| `_passages2string` | 把检索结果拼成 `Doc i(Title: ...) 正文`。 |
| `_process_next_obs` | 把 obs 截到 `max_obs_length`（就是日志里 `OBSERVATION TOO LONG` 警告的来源）。 |
| `_compose_final_output` | 产出 `prompts/input_ids/responses/responses_with_info_mask/attention_mask/`**`info_mask`**`/position_ids`。 |
| `_generate_with_gpu_padding` | 把 batch 补齐到 `num_gpus` 的整数倍再生成。 |

> **`info_mask`** 是关键产物：它标记出哪些 token 是"检索回来的 `<information>` 内容"。后面 trainer 用它做 **loss 屏蔽**——模型不在检索文本上算 loss。

- **`tensor_helper.py`**：`TensorHelper` 一堆张量拼接/mask/position_id 的工具，纯管道。

### 3.2 `search_r1/search/` — 检索服务器（可插拔）

所有服务器都实现同一个契约：`POST /retrieve  {queries, topk, return_scores}` → `{result: [[{document:{contents}, score}, ...], ...]}`。**换检索后端只需换启动哪个服务器**，训练侧只认 `retriever.url`。

| 文件 | 作用 |
|---|---|
| **`retrieval_server.py`** | 主服务（本次用的就是它）。`Encoder`(e5/bge/dpr/t5 pooling) + `DenseRetriever`(faiss) + `BM25Retriever`(pyserini)。`get_retriever(config)`：名字含 bm25 走稀疏，否则走稠密。`--faiss_gpu` 把索引分片到 GPU。 |
| `index_builder.py` | `Index_Builder`：把语料编码成向量，`faiss.index_factory(dim, faiss_type, METRIC_INNER_PRODUCT)` 建索引。**`--faiss_type Flat`=精确，`HNSW64`=近似(ANN, 省显存能跑 CPU)**。 |
| `retrieval.py` | 离线批量检索库（不起服务），类同 `Encoder/DenseRetriever/BM25Retriever`。 |
| `serp_search_server.py` | 在线 **SerpAPI** 搜索，同 `/retrieve` 契约。 |
| `google_search_server.py` | 在线 **Google CSE** 搜索（可选抓网页正文 `--snippet_only`）。 |
| `rerank_server.py` / `retrieval_rerank_server.py` | 交叉编码器重排 / 检索+重排组合服务。 |
| `retrieval_request.py` | 一个 POST 到 8000 的客户端自测小脚本。 |

---

## 4. `verl/` 详解（RL 训练框架，看 FSDP 路径即可）

> `verl/` 里 `megatron*` 是另一条并行/大模型路径，**复现和改进只需看 FSDP 路径**，megatron 可忽略。

### 4.1 入口与奖励管理
- **`verl/trainer/main_ppo.py`**：Hydra 入口（配置名 `ppo_trainer`），起 Ray、建 tokenizer、映射角色（Actor/Critic/Ref → worker）、建 `RayPPOTrainer` 并 `fit()`。
  - **`RewardManager.__call__`**：遍历 batch，解码 prompt+response，**把标量分数写在最后一个有效 response token 上**：`reward_tensor[i, valid_response_length-1] = score`。
  - **`_select_rm_score_fn(data_source)`**：`data_source ∈ {nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle}` → `qa_em.compute_score_em`。**加新数据集要在这里登记。**
- **`main_ppo_format.py`**：**格式奖励版**（v0.3 脚本用 `-m verl.trainer.main_ppo_format`）。用 `qa_em_format.compute_score_em`，并把 `structure_format_score/final_format_score/retrieval_score` 传进去。

### 4.2 `verl/trainer/ppo/ray_trainer.py` — RL 主循环 ⭐
模块级算法函数：
- **`compute_advantage(data, adv_estimator, ...)`**：**按 `adv_estimator` 分流**——`gae`→`core_algos.compute_gae_advantage_return`（要 critic 的 values）；`grpo`→`compute_grpo_outcome_advantage`（用 `uid` 做组，无需 critic）。
- **`apply_kl_penalty(...)`**：把 `beta·KL(old‖ref)` 从奖励里扣掉（**只在 PPO 即 `use_kl_loss=False` 时用**；GRPO 把 KL 放进 actor loss 里）。有 `info_mask` 时会用它当 response mask。
- **`compute_data_metrics`**：除常规指标外，还记录 Search-R1 的环境指标 `env/number_of_actions`、`env/finish_ratio`、`env/number_of_valid_action`、`env/ratio_of_valid_action`、`env/number_of_valid_search`。

`RayPPOTrainer`：
- `init_workers`：**只有 `adv_estimator=='gae'` 才建 critic**（`use_critic=True`）；`grpo` 不建 critic。
- **`fit()`** 单步流程（对应 2.2 的图）：验证→复制 n_agent→生成(`run_llm_loop`)→log_prob→[critic values]→**奖励(reward_fn)→[apply_kl_penalty]→compute_advantage**→[update_critic]→**update_actor**→`_create_loss_mask`（`do_search and state_masking` 时把 `info_mask` 变成 `loss_mask`）→周期验证/存档。
- `_validate()`：同生成流程但 `do_sample=False`，汇总 `val/test_score/{data_source}`。

### 4.3 `verl/workers/`（FSDP）
- **`fsdp_workers.py`**：`ActorRolloutRefWorker`（一个类兼 actor/rollout/ref 三个角色）、`CriticWorker`、`RewardModelWorker`(默认关)。里面 `_build_model_optimizer / _build_rollout / update_actor / compute_log_prob / generate_sequences / compute_ref_log_prob`。
- **`actor/dp_actor.py`::`update_policy`** ⭐：actor 更新。构造 `response_mask`，**`state_masking` 时改用 `loss_mask`**；调 `compute_policy_loss` + 熵损失；**`use_kl_loss` 时加 `kl_loss·kl_loss_coef`（GRPO 的 KL-in-loss 路径）**。
- **`critic/dp_critic.py`::`update_critic`**：仅 GAE 用。
- **`rollout/`**：`vllm_rollout/vllm_rollout.py`(`vLLMRollout`，默认)、`hf_rollout.py`(HF 生成，无 vLLM 时的后备)、`naive/`。
- **`sharding_manager/`**：`fsdp_vllm.py`（FSDP↔vLLM 权重重分片）、`fsdp_ulysses.py`（序列并行）。

### 4.4 `verl/trainer/ppo/core_algos.py` — 优势/损失/KL ⭐
| 函数 | 作用 |
|---|---|
| `compute_gae_advantage_return` | 标准 GAE（用 critic values、gamma、lam）。 |
| **`compute_grpo_outcome_advantage`** | **GRPO 组内归一优势**：把每条回答的 token 奖励求和成标量，按 `uid` 分组，`(score − 组均值)/(组标准差+eps)`，再广播回整条回答。无需 critic。 |
| `compute_policy_loss` | 带 clip 的 PPO surrogate（`clip_ratio`）。 |
| `compute_value_loss` | 带 clip 的 value loss。 |
| `compute_entropy_loss` / `kl_penalty` | 熵；KL 支持 `kl/abs/mse/`**`low_var_kl`**`/full`（GRPO 默认 low_var_kl）。 |

### 4.5 奖励函数 `verl/utils/reward_score/` ⭐（改进高频区）
- **`qa_em.py`（默认）**：`normalize_answer`(SQuAD 式规范化)、`em_check`(精确)、`subem_check`(子串)。**`extract_solution`**：抽所有 `<answer>...</answer>`，**≥2 个才返回最后一个**（因为 prompt 模板里自带一个示例 `<answer> Beijing </answer>`）。**`compute_score_em`**：抽不到答案→0；EM 命中→`score`(1.0)；否则→`format_score`(0.0)。**即：默认就是二值 EM，无格式塑形。**
- **`qa_em_format.py`（格式奖励版）**：
  - `is_valid_sequence`：状态机，要求 `think/search/information/answer` 标签配对且顺序合法。
  - `is_retrieval_correct`：金答案是否出现在某个 `<information>` 块里。
  - `compute_score_em` 分档：EM对+格式对→**1.0**；EM对但格式错→**0.8**；答错但格式对且检索命中→**0.3**；仅格式对→**0.2**；有答案但错且格式错→**0.1**；啥都不对→**0**。**这是做奖励塑形（reward shaping）的直接抓手。**

### 4.6 数据管线 `scripts/data_process/`
产出 **parquet**，schema：
```python
{ "data_source": "nq",                       # 决定用哪个 reward
  "prompt": [{"role":"user","content": <make_prefix 输出>}],
  "ability": "fact-reasoning",
  "reward_model": {"style":"rule","ground_truth":{"target":[golden_answers]}},
  "extra_info": {"split":..., "index":...} }  # index 会被复制成 uid 供 GRPO 分组
```
- **`make_prefix(dp, template_type='base')`** 就是**提示词模板**（`<think>/<search>/<information>/<answer>` 那段）。`infer.py` 里硬编码了同一段。
- `nq_search.py`/`nq.py`：单数据集 NQ。`nq_rag.py`：把检索结果直接塞进 prompt 的普通 RAG 基线（无 agent 循环）。`qa_search_{train,test}_merge.py`：多数据集合并（`--data_sources nq,hotpotqa,...`）。

---

## 5. 配置与超参速查

默认值在 **`verl/trainer/config/ppo_trainer.yaml`**，每次运行用 Hydra CLI 覆盖（`key=value`）。常改的：

| 配置 | 含义 |
|---|---|
| `algorithm.adv_estimator` | `gae`(PPO,要critic) / `grpo`(无critic) |
| `actor_rollout_ref.actor.use_kl_loss` | GRPO 设 true（KL 放 loss 里）；PPO 设 false |
| `actor_rollout_ref.actor.kl_loss_coef` / `kl_loss_type` | GRPO 的 KL 系数 / 类型(`low_var_kl`) |
| `actor_rollout_ref.actor.state_masking` | 是否屏蔽 `<information>` 的 loss（Search-R1 设 true） |
| `actor_rollout_ref.rollout.n_agent` | 每个 prompt 采样几条轨迹（GRPO 组大小，如 5） |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | vLLM 占显存比例 |
| `data.max_prompt_length / max_response_length / max_start_length / max_obs_length` | 长度上限（满足 `max_prompt ≈ start + response·(turns-1) + obs·turns`） |
| `max_turns` / `do_search` / `retriever.url` / `retriever.topk` | 搜索轮数 / 是否搜索 / 检索地址 / topk |
| `reward_model.{structure_format_score,final_format_score,retrieval_score}` | 格式奖励档位（需配 `main_ppo_format`） |

---

## 6. 本次复现：环境与机器约束

### 6.1 机器（共享，注意资源竞争）
- 6 张卡：GPU0/1/4/5 = RTX A6000 48G，GPU2/3 = RTX 5880 Ada 48G。
- **另一个用户的 `peft` 任务会动态占满所有卡（每张 ~35GB，只剩 ~12GB）**——别杀它；训练要用小模型 + 全量 CPU offload + 低 vLLM 显存占用挤进去。
- 内存 1TB（CPU offload / CPU faiss 很划算）、`/mnt/backup1` 有 11TB。
- **外网上行只有 ~0.4MB/s 且不稳**（代理和清华直连都慢，多连接不叠加）。→ **论文原版 65GB wiki 全量 e5 索引下不动（~40h），本次改用本地自建 10 条小索引验证 pipeline。**

### 6.2 两个 conda 环境（建在 `/mnt/backup1/lgc/envs/`）
| 环境 | 内容 | 怎么来的 |
|---|---|---|
| **`searchr1`**（训练） | torch 2.4.0+cu121 / vllm 0.6.3 / xformers 0.0.27.post2 / transformers 4.46.3 / numpy 1.26.4 / verl。**无 flash-attn**。 | **克隆 `moe`**（白拿 CUDA 栈）→ 降 torch 到 2.4.0 → 装 vllm → 装 verl。 |
| **`retriever`**（检索） | torch 2.4.1 / transformers 4.44.2 / **faiss-cpu 1.14** / fastapi / uvicorn。 | 克隆 `moe` + `pip install faiss-cpu fastapi uvicorn`。 |

> 两个环境都需要把 `huggingface-hub` 降到 `<1.0`（`moe` 带的 1.10.2 与 transformers<4.48 冲突）。

### 6.3 产物目录 `/mnt/backup1/lgc/search-r1-data/`
```
nq_search/{train,test}.parquet   # NQ 数据 79168/3610 条
e5-base-v2/                      # 检索模型 intfloat/e5-base-v2
index_small/e5_Flat.index        # 用 example/corpus.jsonl(10条) 自建的小索引
run_scripts/                     # ★可复用脚本（见下）
run_scripts/logs/                # ★本次运行日志
ckpt/                            # 训练 checkpoint
```
`run_scripts/` 里：`prep_nq.py`(数据)、`build_index.sh`(建索引)、`launch_retrieval.sh`(起检索)、`train_smoke_grpo.sh`(训练)、`infer_local.py`(推理)、`dl_curl.sh`/`dl_chunked.py`(鲁棒下载器)、`searchr1_deps.sh`/`searchr1_finalize.sh`(装环境的记录)。

---

## 7. 复现步骤（可照抄命令）

> 前提：两个 conda 环境已建好（见 6.2；若要在别的机器重建，参考 `run_scripts/searchr1_finalize.sh`）。

```bash
# ── (0) 数据：下载 NQ 的 jsonl 并转 parquet（不用庞大的 datasets 加载脚本）
python /mnt/backup1/lgc/search-r1-data/run_scripts/prep_nq.py
#   → /mnt/backup1/lgc/search-r1-data/nq_search/{train,test}.parquet

# ── (1) 建小索引：用仓库自带 example/corpus.jsonl(10条) + e5 建 Flat 索引（GPU3 编码）
bash /mnt/backup1/lgc/search-r1-data/run_scripts/build_index.sh
#   → /mnt/backup1/lgc/search-r1-data/index_small/e5_Flat.index

# ── (2) 起检索服务（retriever 环境，e5 在 GPU3，faiss 在 CPU，端口 8002）
bash /mnt/backup1/lgc/search-r1-data/run_scripts/launch_retrieval.sh
#   自测: curl -s --noproxy '*' -X POST http://127.0.0.1:8002/retrieve \
#          -H 'Content-Type: application/json' \
#          -d '{"queries":["Who is Evan Morris?"],"topk":3,"return_scores":true}'

# ── (3) 冒烟训练（searchr1 环境，GRPO，GPU4/5，全量 CPU offload）
bash /mnt/backup1/lgc/search-r1-data/run_scripts/train_smoke_grpo.sh

# ── (4) 推理 demo（searchr1 环境，GPU4）
bash -c 'source /data/mambaforge/etc/profile.d/conda.sh; conda activate searchr1;
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
  python /mnt/backup1/lgc/search-r1-data/run_scripts/infer_local.py'
```

**本次冒烟训练的关键配置**（`train_smoke_grpo.sh`，为在 ~12GB 余量里跑通而缩小）：
```
模型 Qwen2.5-0.5B-Instruct | adv_estimator=grpo | use_kl_loss=true kl_loss_type=low_var_kl
train_batch_size=4 n_agent=2 ppo_mini=4 ppo_micro=2 | max_turns=2 do_search=true
max_prompt=1536 max_response=200 max_start=768 max_obs=200
FSDP 全量 offload(param/grad/optimizer=true) | vllm gpu_memory_utilization=0.1
n_gpus_per_node=2 (GPU4,5) | total_training_steps=6 | logger=console (无需 wandb)
retriever.url=http://127.0.0.1:8002/retrieve topk=3
环境变量 CUDA_DEVICE_ORDER=PCI_BUS_ID  VLLM_ATTENTION_BACKEND=XFORMERS  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

> **等 GPU 空出来 / 换更大机器时**，把模型换回 `Qwen2.5-1.5B/3B/7B`、`gpu_memory_utilization` 调回 0.5、batch 和 `n_agent` 调大、`total_training_steps` 跑满（论文 ~1000 步），offload 可关掉提速。

---

## 8. 复现结果与日志解读

完整日志在 `/mnt/backup1/lgc/search-r1-data/run_scripts/logs/`：
`train_smoke_full.log`（全量）、`train_smoke_steps.log`（只留 step 指标）、`infer_evan_morris.log`（推理）、`retrieval_server.log`（检索服务，含 29 次请求）。

### 8.1 训练：完整跑完 6 步（GRPO）
```
step:0 - val/test_score/nq:0.000            ← 训练前初始验证
epoch 0, step 1 ... step:1 ...
...
step:5 - ... val/test_score/nq:0.000        ← test_freq=5 触发的周期验证
step:6 - val/test_score/nq:0.000            ← 末步 + 最终验证，干净退出
```
step:1 的关键指标解读：
| 指标 | 值 | 含义 |
|---|---|---|
| `actor/kl_loss` | 0.001 | 与 ref 的 KL（GRPO 放 loss 里） |
| `actor/entropy_loss` | ~1.4 | 策略熵 |
| `actor/pg_loss` / `grad_norm` | 0.0 / 0.03 | 策略梯度损失 / 梯度范数（更新在发生） |
| `env/number_of_actions/mean` | ~2.7–3.0 | 每条轨迹的动作数（多轮生效） |
| **`env/number_of_valid_search`** | **step2=0.625, step3=0.375, step4/5=0.25** | **rollout 里真的发出了有效 `<search>`** |
| `response_length/mean` | 340–480 | 生成长度合理 |
| `timing_s/step` | ~44–50s | 每步耗时（offload 会慢些） |

- **检索服务本轮处理了 29 次 `POST /retrieve`（200 OK）** → 训练↔检索的集成确认打通。
- val 分数为 0：因为是 **0.5B 小模型 + 10 条语料 + 仅 6 步**，与"pipeline 是否跑通"无关。**我们验证的是整条链路能跑，不是模型质量。**

### 8.2 推理：一次教科书级的 Search-R1 行为
问题：*Who was Evan Morris and which company did he lobby for?*（未训练的 Qwen2.5-1.5B-Instruct）
```
<think> Evan Morris is an American political consultant... </think>
<search> evan morris lobbying companies </search>            ← 自主发起搜索
<information> Doc 1(Title: "Evan Morris") ... was a lobbyist for
   Genentech and its parent corporation Roche ... Medicare and Medicaid... </information>
<answer> Evan Morris ... worked primarily for Genentech and its parent
   company Roche ... Medicare and Medicaid </answer>              ← 基于检索、事实正确
```
完整闭环 **推理 → 自主搜索 → 真实检索 → 基于证据作答**，答案有出处（Genentech/Roche 全来自 Doc 1）。

---

## 9. 我对源码做的修改（patch 清单）

为在"无 flash-attn + 端口被占 + 显存紧张"的本机跑通，改了 3 处（都很小、可回退）：

| 文件 | 改动 | 原因 |
|---|---|---|
| `search_r1/search/retrieval_server.py` | 加 `--port` 参数（默认仍 8000），`uvicorn.run(..., port=args.port)` | 8000/8001 被别的用户占了，改用 8002 |
| `verl/workers/actor/dp_actor.py` | 顶层 `from flash_attn.bert_padding import ...` 包成 `try/except`（失败设 None） | 没装 flash-attn；这些函数只在 `use_remove_padding=True` 时才调用 |
| `verl/workers/fsdp_workers.py` | 3 处 `attn_implementation='flash_attention_2'` → `'sdpa'` | 没装 flash-attn；sdpa 数值等价、Qwen2 支持 |

> 这些是**为了在本机无 flash-attn 环境跑通**的适配。**正式复现/提速**时可反过来：装 flash-attn 的 prebuilt wheel，把上面 3 处改回、并开 `actor.model.use_remove_padding=True`。

---

## 10. 面向改进的源码入手点⭐

> 这一节专门为"根据文献/灵感改进 Search-R1 → 写实验报告"准备。按"你想改什么"直接定位到文件/函数。

### (a) 改**奖励**（最常见的改进方向）
- 二值 EM：`verl/utils/reward_score/qa_em.py :: compute_score_em`。
- 带格式/检索塑形的分档奖励：`verl/utils/reward_score/qa_em_format.py :: compute_score_em`（+ `is_valid_sequence` / `is_retrieval_correct`）。**想加"搜索次数惩罚 / 冗余惩罚 / 过程奖励 / 检索质量奖励"就改这里。**
- 新数据集/新奖励的登记：`verl/trainer/main_ppo.py :: _select_rm_score_fn`（或 `main_ppo_format.py`）。分数落点在 `RewardManager.__call__`（打在最后一个 token）。
- 开启格式奖励：启动改成 `-m verl.trainer.main_ppo_format`，并设 `reward_model.structure_format_score/final_format_score/retrieval_score`。

### (b) 改**搜索交互 / 轮数 / 观测处理**
- 全在 `search_r1/llm_agent/generation.py`：主循环 `run_llm_loop`；动作解析 `postprocess_predictions` / 截断 `_postprocess_responses`；环境转移与非法动作文本 `execute_predictions`；检索调用 `_batch_search`、结果拼装 `_passages2string`；观测注入/截断 `_process_next_obs`。
- 想改**轮数/观测长度/topk**：config `max_turns`、`data.max_obs_length`、`retriever.topk`。
- 想**惩罚非法动作 / 奖励高质量查询**：`execute_predictions` 目前对非法动作只给提示不扣分——这是一个明确的可改点（配合 (a) 的奖励）。

### (c) 改**RL 算法 / 优势 / KL**
- 优势与损失：`verl/trainer/ppo/core_algos.py`（`compute_grpo_outcome_advantage` / `compute_gae_advantage_return` / `compute_policy_loss` / `compute_value_loss` / `kl_penalty`）。
- 分流与 KL 施加：`ray_trainer.py :: compute_advantage`（gae/grpo 分支）、`apply_kl_penalty`；critic 是否创建看 `init_workers`。
- actor 侧 loss 组装（GRPO 的 KL-in-loss、熵、loss-mask）：`verl/workers/actor/dp_actor.py :: update_policy`。
- 检索 token 的 loss 屏蔽：`ray_trainer.py :: _create_loss_mask`（喂给 `dp_actor` 的 `loss_mask`）。**想改"是否/如何屏蔽检索内容"就在这。**

### (d) 改**prompt 模板 / 数据集**
- 模板：`scripts/data_process/*.py :: make_prefix`（同一段也硬编码在 `infer.py`）。**改标签体系要同步改** rollout(generation.py 的正则)、reward(qa_em 的正则)、这两处，否则解析会崩。
- 换数据集：`qa_search_{train,test}_merge.py` 的 `--data_sources`（并在 `_select_rm_score_fn` 登记），再把 `data.train_files/val_files` 指过去。

### (e) 改**检索后端**
- 换 BM25 / HNSW-ANN / SerpAPI / Google / rerank：起 `search_r1/search/` 下对应的服务器即可（同 `/retrieve` 契约），训练侧只改 `retriever.url`。参考 `docs/retriever.md`。

### (f) 超参
- 默认：`verl/trainer/config/ppo_trainer.yaml`。每次运行覆盖：`train_ppo.sh`/`train_grpo.sh`/`run_scripts/train_smoke_grpo.sh`。

---

## 11. 踩坑与 FAQ

| 现象 | 原因 / 解决 |
|---|---|
| `ImportError: flash_attn ... FlashAttention2 has been toggled on` | verl 默认 `attn_implementation='flash_attention_2'`。装 flash-attn，或按第 9 节改成 `sdpa`。 |
| `torch.OutOfMemory ... Process xxxx(peft) has 34.9GB` | 别人的任务占满卡。用小模型 + 全量 offload + `gpu_memory_utilization` 调低 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。 |
| `address already in use :8000` | 端口被占。`retrieval_server.py --port 8002`，训练 `retriever.url` 同步改。 |
| `[WARNING] OBSERVATION TOO LONG 482 & 256` | 检索段落超过 `max_obs_length`，会自动截断，无害（说明检索确实被调用了）。 |
| vllm `Detected different devices` 警告 | 卡型号不一致。训练只用同构 A6000（GPU4,5）即可；设 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。 |
| `huggingface-hub>=0.23.2,<1.0 is required` | clone 带来的 hub 太新。`pip install "huggingface-hub<1.0"`。 |
| 全量 65GB wiki 索引下不动 | 本机带宽限制。需更快外网通道或拷贝副本；验证 pipeline 用小索引即可。 |
| pip/HF 下大文件反复断 | 用 `run_scripts/dl_curl.sh`（curl 续传，清华源，`--noproxy`）；HF 单文件用 `dl_chunked.py`（32MB 分块）。 |

---

## 12. 附录：关键文件与符号速查表

| 你要看/改… | 去这个文件的这个符号 |
|---|---|
| 多轮搜索循环 | `search_r1/llm_agent/generation.py :: LLMGenerationManager.run_llm_loop` |
| 动作解析/检索调用 | `generation.py :: execute_predictions / _batch_search / postprocess_predictions` |
| 检索服务 | `search_r1/search/retrieval_server.py :: get_retriever / DenseRetriever / Encoder` |
| 建索引 | `search_r1/search/index_builder.py :: Index_Builder.build_dense_index` |
| RL 主循环 | `verl/trainer/ppo/ray_trainer.py :: RayPPOTrainer.fit` |
| 优势分流 / KL | `ray_trainer.py :: compute_advantage / apply_kl_penalty / _create_loss_mask` |
| GRPO/GAE/loss | `verl/trainer/ppo/core_algos.py` |
| actor 更新 | `verl/workers/actor/dp_actor.py :: update_policy` |
| 奖励(默认/格式) | `verl/utils/reward_score/qa_em.py`（`compute_score_em`）/ `qa_em_format.py` |
| 奖励挂载/数据源映射 | `verl/trainer/main_ppo.py :: RewardManager / _select_rm_score_fn` |
| 提示词模板 | `scripts/data_process/nq_search.py :: make_prefix`（同 `infer.py`） |
| 默认超参 | `verl/trainer/config/ppo_trainer.yaml` |
| 推理 demo | `infer.py`（本地版：`run_scripts/infer_local.py`） |

---

*本文档由复现过程整理。后续改进 Search-R1 时，建议在第 10 节对应位置改代码，并在这里追加"改动记录 + 实验结果"小节，便于形成实验报告。*
