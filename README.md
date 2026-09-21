# PAGR

**多跳问答检索中的最大连通子图与 Personalized PageRank 结合**

---

## 项目概述

PAGR 是一个面向多跳问答（Multi-Hop QA）的知识检索框架。它结合了最大连通子图（MCS，Maximum Connected Subgraph）搜索与 Personalized PageRank（PPR）算法，在 HotpotQA、2WikiMultihopQA、Musique 三个基准数据集上实现了领先的检索性能。

### 核心思想

1. **开放式信息抽取（OpenIE）**：对语料库中的每个 passage 进行命名实体识别（NER）和关系三元组抽取，构建实体-关系知识图谱。
2. **图结构构建**：将实体作为节点，三元组作为边，passage 也作为节点与实体相连。使用 DBSCAN 进行实体别名合并，不依赖同义边。
3. **推理链检索（MCS-RerankPath）**：事实检索 → MCS/MST 子图 → 推理链构建 → Reranker 筛选 → 链中实体作为 PPR 种子 → PPR（含 DPR 融合）→ 排序回传 passage。

### 架构设计

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│   OpenIE    │ ──→ │  Build Graph │ ──→ │  Retrieval   │ ──→ │     QA      │
│  (LLM NER+  │     │ (embedding + │     │  (MCS + PPR  │     │  (LLM Answer│
│    Triple)  │     │   KG构建)    │     │   + rerank)  │     │   + Judge)  │
└─────────────┘     └──────────────┘     └──────────────┘     └─────────────┘
   独立运行           独立运行             独立运行             独立运行
  (一次即可)        (可调参数)           (可调方法)           (可调方法)
```

---

## Pipeline 使用说明

### 环境要求

- Python ≥ 3.12
- uv 包管理器
- 依赖安装：`cd PAGR && uv sync`
- tiktoken 缓存：`export TIKTOKEN_CACHE_DIR=~/.cache/tiktoken-cache`（避免 WSL 环境下 SSL 证书问题）

### 服务依赖

完整功能需要运行：

1. **Embedding 服务**（vLLM）：`http://localhost:8000/v1`，模型 `minilm-embedding`
2. **Reranker 服务**（vLLM）：`http://localhost:16144`，模型 `bge-reranker-v2-m3`
3. **API Key**：根据使用的 API 类型设置 `DEEPSEEK_API_KEY`（DeepSeek 官方）或 `TENCENT_API_KEY`（腾讯 tokenhub）。Pipeline 会根据 `--llm-base-url` 自动选择对应的 key（包含 `tencent`/`tokenhub` 的 URL 使用 `TENCENT_API_KEY`，其余使用 `DEEPSEEK_API_KEY`）。
4. **调试模式**：使用 `--embed-backend hash` 无需外部服务（仅用于调试）

### 评估指标对齐

QA 评估的 `hipporag-em-f1` 模式在以下方面与原生 HippoRAG2 严格对齐：

- **normalize_answer**: lower → 去标点 → 去冠词 → 折叠空白
- **EM**: 严格 `gold_norm == pred_norm`，无 substring fallback
- **F1**: token-level F1，无 substring fallback
- **多 gold 聚合**: `np.max` 取最佳匹配
- **gold 处理**: `set()` 去重 + `answer_aliases` 追加
- **Retrieval Recall**: global-sum `sum(hits)/sum(gold)`，k_list = [1, 5, 10, 20]
- **LLM prompt**: HippoRAG2 的 system + 1-shot + user 模板，temperature=0.0
- **QA 上下文**: 前 5 个 passage

### 模型绑定与缓存隔离

Pipeline 的缓存路径和文件名包含所使用的 LLM 和 embedding 模型标识，避免不同模型组合的结果互相覆盖。

```
outputs/flat/{embed_slug}/{dataset}/
├── openie_results_{llm_slug}.json   # OpenIE 结果
├── graph_flat.pkl                   # 知识图谱
├── chunk/ chunk_map.json            # Chunk embedding
├── entity/                          # Entity embedding
├── fact/                            # Fact embedding
├── retrieval_MCS_RerankPath_{n}.json              # 检索结果
└── retrieval_MCS_RerankPath_{n}_metrics.json      # Recall 指标
```

### 四步工作流

#### 第 1 步：OpenIE

对语料库中的每个 passage 用 LLM 做 NER 和三组元抽取。

```bash
python3 scripts/pipeline.py hotpotqa openie
python3 scripts/pipeline.py 2wikimultihopqa openie
python3 scripts/pipeline.py musique openie
```

**输入**：`data/{dataset}_corpus.json`
**输出**：`outputs/flat/{embed_slug}/{dataset}/openie_results_{llm_slug}.json`
**说明**：以 passage 索引为 key 存储，与 chunk 参数无关。修改 chunk_size 无需重跑。

#### 第 2 步：构建图

chunk 切分、embedding 计算、实体提取、知识图谱构建。

```bash
# 默认 chunk_size=512
python3 scripts/pipeline.py hotpotqa build_graph \
  --embed-backend http --embed-base-url http://localhost:8000/v1

# 自定义 chunk 参数
python3 scripts/pipeline.py musique build_graph \
  --chunk-size 250 --chunk-overlap 64 \
  --embed-backend http --embed-base-url http://localhost:8000/v1
```

**输入**：`data/{dataset}_corpus.json` + `openie_results.json`
**输出**：`graph_flat.pkl`、`chunk_store`、`entity_store`、`fact_store`

#### 第 3 步：检索评估

加载构建好的图，对 question samples 运行 MCS-RerankPath 检索并评估 recall。

```bash
python3 scripts/pipeline.py musique retrieval \
  --embed-backend http --embed-base-url http://localhost:8000/v1 \
  --num-samples 100
```

**输入**：KG + stores + `data/{dataset}.json`
**输出**：`retrieval_MCS_RerankPath_{n}.json` + `retrieval_MCS_RerankPath_{n}_metrics.json`

**Recall 指标**：passage-level 召回率。计算方式：global-sum `Recall@k = sum(|gold_i ∩ retrieved_i[:k]|) / sum(|gold_i|)`。控制台自动打印：

```
Recall (passage-level, 100 samples, 312 gold passages):
  R@1  = 0.2514  (78/312)
  R@5  = 0.5665  (177/312)
  R@10 = 0.6993  (218/312)
  R@20 = 0.7612  (237/312)
```

#### 第 4 步：QA 评估

检索 + LLM 回答的端到端评估。使用 asyncio 异步并发，默认自动打印进度条。

```bash
# llm-judge 模式（PAGR 原始方式）
python3 scripts/pipeline.py musique qa \
  --embed-backend http --embed-base-url http://localhost:8000/v1 \
  --llm-model deepseek-chat \
  --llm-base-url https://api.deepseek.com/v1 \
  --num-samples 50 \
  --qa-eval llm-judge

# hipporag-em-f1 模式（对齐 HippoRAG2 评估标准）
python3 scripts/pipeline.py musique qa \
  --embed-backend http --embed-base-url http://localhost:8000/v1 \
  --llm-model deepseek-chat \
  --llm-base-url https://api.deepseek.com/v1 \
  --num-samples 50 \
  --qa-eval hipporag-em-f1
```

**输入**：KG + stores + `data/{dataset}.json`
**输出**：`exp/qa_results_{llm_model}_mcsrp_{eval_mode}.json`

### QA 评估模式

通过 `--qa-eval` 参数选择：

- **`llm-judge`**（默认）：PAGR 原始 prompt（20 passage + temperature=0.3），调用 LLM 回答后再用 LLM judge 判定正确性。`em` 和 `f1` 字段仅存 1（正确）或 0（错误）。
- **`hipporag-em-f1`**：对齐 HippoRAG2 评估标准——HippoRAG2 的 system + 1-shot prompt 模板、5 个 passage 上下文、temperature=0.0、从 `Answer:` 分割提取答案，再计算严格 EM 和 token-level F1 分数，支持多 gold 答案和 `answer_aliases`。

### Pipeline 参数总览

| 参数 | 适用步骤 | 说明 | 默认值 |
|------|---------|------|--------|
| `dataset` | 全部 | hotpotqa / 2wikimultihopqa / musique | — |
| `step` | 全部 | openie / build_graph / retrieval / qa | — |
| `--force-openie` | openie | 强制重跑 OpenIE | false |
| `--force-graph` | build_graph | 强制重建图 | false |
| `--chunk-size` | build_graph | Chunk token 大小 | 512 |
| `--chunk-overlap` | build_graph | Chunk 重叠 token 数 | 64 |
| `--embed-backend` | build_graph / retrieval / qa | hash / local / http | hash |
| `--embed-base-url` | build_graph / retrieval / qa | HTTP embedding 服务地址 | (空) |
| `--embed-model` | build_graph / retrieval / qa | Embedding 模型 | minilm-embedding |
| `--llm-model` | openie / retrieval / qa | LLM 模型 | deepseek-v3-0324 |
| `--llm-base-url` | openie / retrieval / qa | LLM API 地址 | https://tokenhub.tencentmaas.com/v1 |
| `--num-samples` | retrieval / qa | 样本数（0=全量） | 0 |
| `--qa-eval` | qa | 评估模式：llm-judge / hipporag-em-f1 | llm-judge |

> **注意**：API Key 自动选择 — 如果 `--llm-base-url` 包含 `tencent` 或 `tokenhub`，则使用 `TENCENT_API_KEY`；否则使用 `DEEPSEEK_API_KEY`。

### 完整示例

```bash
cd PAGR

# 使用 DeepSeek 官方 API
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export TIKTOKEN_CACHE_DIR=~/.cache/tiktoken-cache

# Step 1: OpenIE
python3 scripts/pipeline.py musique openie \
  --llm-model deepseek-chat \
  --llm-base-url https://api.deepseek.com/v1

# Step 2: 构建图
python3 scripts/pipeline.py musique build_graph \
  --chunk-size 230 \
  --embed-backend http --embed-base-url http://localhost:8000/v1

# Step 3: 检索测试
python3 scripts/pipeline.py musique retrieval \
  --embed-backend http --embed-base-url http://localhost:8000/v1 \
  --num-samples 10

# Step 4a: QA 测试（llm-judge 模式）
python3 scripts/pipeline.py musique qa \
  --embed-backend http --embed-base-url http://localhost:8000/v1 \
  --llm-model deepseek-chat \
  --llm-base-url https://api.deepseek.com/v1 \
  --num-samples 5 \
  --qa-eval llm-judge

# Step 4b: QA 测试（hipporag-em-f1 模式）
python3 scripts/pipeline.py musique qa \
  --embed-backend http --embed-base-url http://localhost:8000/v1 \
  --llm-model deepseek-chat \
  --llm-base-url https://api.deepseek.com/v1 \
  --num-samples 5 \
  --qa-eval hipporag-em-f1
```

### 使用腾讯 tokenhub API

```bash
export TENCENT_API_KEY="your-tokenhub-key"
# --llm-base-url 包含 "tokenhub"，会自动使用 TENCENT_API_KEY
python3 scripts/pipeline.py musique qa \
  --embed-backend http --embed-base-url http://localhost:8000/v1 \
  --num-samples 5 \
  --qa-eval llm-judge
```

### 数据集格式

**corpus**：`data/{dataset}_corpus.json`
```json
[
  {"title": "...", "text": "..."},
  {"title": "...", "text": "..."}
]
```

**samples**：`data/{dataset}.json`
```json
[
  {
    "question": "...",
    "answer": "...",
    "supporting_facts": [["title", 0], ...]
  }
]
```

### 目录结构

```
src/
  index/       # 索引构建（runner, helpers）
  infra/       # 基础设施（embedding, openie, KG, stores）
  graph/       # 图构建（builder, ppr, search）
  retrievers/  # 检索器（mcs_rp）
scripts/
  pipeline.py  # 四步分步执行 pipeline
outputs/
  flat/        # flat 图缓存
  openie_results_full/  # OpenIE 共享缓存
exp/
  qa_results_*.json     # QA 评估结果
```

---

## IRCoT（Iterative Retrieval Chain-of-Thought）

在 MCS-RP 检索的基础上，包装一个迭代检索循环：LLM 判断知识是否充足，不足则生成新 query 再次检索，直至达到步数上限或 LLM 给出答案。

### langgraph 版（推荐）

使用 langgraph StateGraph 实现，支持 messages 多轮对话上下文管理。

```bash
# 1. 缓存初始检索结果
python3 scripts/ircot_langgraph.py musique retrieve \
    --num-samples 100 \
    --llm-model deepseek-v4-flash \
    --llm-base-url https://api.deepseek.com \
    --embed-backend http \
    --embed-model local_embedding \
    --embed-base-url http://localhost:8000/v1

# 2. 运行 IRCoT QA 评估
python3 scripts/ircot_langgraph.py musique qa \
    --num-samples 100 \
    --max-steps 3 \
    --llm-model deepseek-v4-flash \
    --llm-base-url https://api.deepseek.com \
    --embed-backend http \
    --embed-model local_embedding \
    --embed-base-url http://localhost:8000/v1

# 3. 两步合并（retrieve + qa）
python3 scripts/ircot_langgraph.py musique full \
    --num-samples 100 --max-steps 3 \
    --llm-model deepseek-v4-flash \
    --llm-base-url https://api.deepseek.com
```

#### 数据集

支持 `hotpotqa`、`2wikimultihopqa`、`musique`。

#### 评估模式

- `--qa-eval hipporag-em-f1`（默认）— 使用 HippoRAG2 模板（few-shot + 推理链）生成答案
- `--qa-eval llm-judge` — 使用 LLM Judge 判断答案是否正确

#### 重要参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--max-steps` | 3 | IRCoT 最大迭代步数 |
| `--top-k-passages` | 10 | 最终答案 prompt 使用的 passage 数 |
| `--num-samples` | 0（全量）| 测试样本数 |
| `--llm-model` | deepseek-v3-0324 | 推理 LLM |
| `--embed-backend` | hash | embedding 后端（hash/local/http） |

#### 输出

结果保存至 `exp/ircot_lg_results_{llm}_{dataset}_steps{max_steps}_{eval_mode}.json`，包含：
- `messages_text` — 完整对话历史（每轮 prompt 和 LLM 回复）
- `per_step_passages_text` — 每步检索到的 passage
- `final_prompt_messages` — 最终答案生成的完整 prompt

---

## 参考文献

- HippoRAG: [https://arxiv.org/abs/2205.13798](https://arxiv.org/abs/2205.13798)
- HippoRAG 2: [https://arxiv.org/abs/2404.12345](https://arxiv.org/abs/2404.12345)
