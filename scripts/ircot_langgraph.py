#!/usr/bin/env python3
"""
IRCoT-LangGraph: Iterative Retrieval Chain-of-Thought using langgraph.

与 scripts/ircot.py 功能相同，但使用 langgraph StateGraph 实现多步 agent。
支持 retrieve/qa/full 三种模式。

Usage:
    python3 scripts/ircot_langgraph.py hotpotqa retrieve              # 初始检索缓存
    python3 scripts/ircot_langgraph.py hotpotqa qa                    # IRCoT QA 评估
    python3 scripts/ircot_langgraph.py hotpotqa full                  # 检索 + QA 一步
"""

import json
import os
import sys
import time
import re
import argparse
import hashlib as _hashlib
import asyncio
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Annotated

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from openai import AsyncOpenAI, RateLimitError
import asyncio


async def _llm_call_with_retry(
    async_client: AsyncOpenAI,
    model: str,
    messages: list,
    temperature: float = 0.0,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> str:
    """带指数退避重试的 LLM 调用。"""
    for attempt in range(max_retries + 1):
        try:
            completion = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return completion.choices[0].message.content or ""
        except RateLimitError as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                log(f"[RateLimit] 429, 第{attempt+1}次重试, 等待{delay:.0f}s...")
                await asyncio.sleep(delay)
            else:
                log(f"[RateLimit] 重试{max_retries}次后仍失败: {e}")
                raise
        except Exception as e:
            # 非限流错误直接抛
            log(f"[LLM-Error] {e}")
            raise
    return ""


# ── 目录 / 常量 ────────────────────────────────────────────────────────────
DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(DIR, '..'))
DATA_DIR = os.path.join(BASE, 'data')
CACHE_BASE = os.path.join(BASE, 'outputs')
EXP_DIR = os.path.join(BASE, 'exp')

DATASET_MAP = {
    "hotpotqa": "hotpotqa",
    "2wikimultihopqa": "2wikimultihopqa",
    "musique": "musique",
}

DEFAULT_LLM_MODEL = os.environ.get("PIPELINE_LLM_MODEL", "deepseek-v3-0324")
DEFAULT_LLM_BASE_URL = os.environ.get("PIPELINE_LLM_BASE_URL", "https://tokenhub.tencentmaas.com/v1")
DEFAULT_RERANKER_BASE_URL = "http://localhost:16144"

DEFAULT_MAX_STEPS = 3
DEFAULT_TOP_K_PASSAGES = 10


# ═══════════════════════════════════════════════════════════════════════════
# 工具函数（复制自 scripts/ircot.py 中 step_qa 内部的嵌套函数）
# 它们原是嵌套在 step_qa 内部的闭包。为保持独立性和可测试性，在此重新定义。
# ═══════════════════════════════════════════════════════════════════════════

_VERBOSE = False

def set_verbose(v: bool):
    global _VERBOSE
    _VERBOSE = v


def log(msg: str):
    if not _VERBOSE:
        return
    tqdm = __import__('tqdm').tqdm
    tqdm.write(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _make_model_slug(model_name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]', '_', model_name).rstrip('._-')


def _get_api_key(llm_base_url: str) -> str:
    """根据 base_url 自动选择 API key"""
    if "tencent" in llm_base_url.lower() or "tokenhub" in llm_base_url.lower():
        return os.environ.get("TENCENT_API_KEY", "")
    return os.environ.get("DEEPSEEK_API_KEY", "")


def _clean_llm_content(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    t = re.sub(r"[\u200B-\u200D\uFEFF]", "", t)
    m = re.compile(r"^\s*```(?:\s*\w+)?\s*\n(?P<body>[\s\S]*?)\n\s*```\s*$", re.MULTILINE).match(t)
    if m:
        t = m.group("body").strip()
    elif t.startswith("```") and t.endswith("```") and len(t) >= 6:
        t = t[3:-3].strip()
    if t.lower().startswith("json\n"):
        t = t.split("\n", 1)[1].strip()
    return t


def _extract_hipporag_answer(text: str) -> str:
    if "So the answer is:" in text:
        try:
            return text.split("So the answer is:")[1].split("\n")[0].strip()
        except Exception:
            pass
    if "Answer:" in text:
        try:
            parts = text.split("Answer:")
            if len(parts) > 1:
                answer = parts[-1].strip()
                answer = answer.split("\n")[0].strip()
                return answer
        except Exception:
            pass
    return text.strip()


def _extract_final_answer(text: str) -> str:
    """从 final answer LLM response 中提取答案。

    Response 格式预期为 "Thought: ... Answer: xxx"，
    兼容 "Answer:" / "So the answer is:" / 裸答案。
    """
    for marker in ["Answer:", "So the answer is:"]:
        if marker in text:
            try:
                return text.split(marker)[-1].split("\n")[0].strip()
            except Exception:
                pass
    return text.strip()


def _normalize_answer(text: str) -> str:
    def lower(t: str) -> str:
        return t.lower()
    def remove_punc(t: str) -> str:
        return re.sub(r'[^\w\s]', '', t)
    def remove_articles(t: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', t)
    def white_space_fix(t: str) -> str:
        return ' '.join(t.split())
    return white_space_fix(remove_articles(remove_punc(lower(text))))


def _exact_match(gold: str, predicted: str) -> float:
    return 1.0 if _normalize_answer(gold) == _normalize_answer(predicted) else 0.0


def _f1_score(gold: str, predicted: str) -> float:
    from collections import Counter
    gn = _normalize_answer(gold)
    pn = _normalize_answer(predicted)
    gt = gn.split()
    pt = pn.split()
    common = Counter(pt) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pt)
    recall = 1.0 * num_same / len(gt)
    return 2.0 * (precision * recall) / (precision + recall)


def _exact_match_multi(gold_list: list, predicted: str, aggregation_fn=None) -> float:
    if aggregation_fn is None:
        aggregation_fn = np.max
    return float(aggregation_fn([_exact_match(g, predicted) for g in gold_list]))


def _f1_score_multi(gold_list: list, predicted: str, aggregation_fn=None) -> float:
    if aggregation_fn is None:
        aggregation_fn = np.max
    return float(aggregation_fn([_f1_score(g, predicted) for g in gold_list]))


def _build_gold_list(sample):
    gold_raw = sample["answer"]
    gold_list_data = gold_raw if isinstance(gold_raw, list) else [gold_raw]
    gold_set = set(gold_list_data)
    if "answer_aliases" in sample and isinstance(sample["answer_aliases"], list):
        gold_set.update(sample["answer_aliases"])
    return list(gold_set)


async def _llm_judge_ask(async_client: Any, model: str, prompt: str) -> str:
    """LLM 单轮问答，用于 llm-judge 评测（对齐 pipeline 的 _llm_judge_ask）。"""
    try:
        raw = await _llm_call_with_retry(
            async_client, model,
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return _clean_llm_content(raw)
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"[ERROR] _llm_judge_ask failed: {e}")
        return ""


def _format_passage_context(passages: List[Tuple[str, str]]) -> str:
    """将 passage 格式化为 IRCoT 可读的上下文"""
    lines = []
    for i, (title, text) in enumerate(passages):
        lines.append(f"=== Passage {i+1}: {title} ===\n{text}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Prompt 模板
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert multi-hop knowledge assistant. Your task is to answer a multi-hop question by iteratively retrieving information.

## Process
[Step 0] You will be given an initial set of retrieved passages.
[Step N] You can request additional searches by outputting a search query. The results will be returned to you.

## Instructions
- If you can determine the final answer from the current knowledge, end your response with:
  So the answer is: <your final answer>

- If you are missing a specific piece of information that you can search for, end your response with:
  The new query is: <a short, specific search query for the missing entity or fact>

- After receiving new search results, you MUST review them and continue reasoning in the same format.

## Alternative Search Queries (only if truly ambiguous)
If the query refers to an entity that could be one of several distinct possibilities (e.g., "the performer" could refer to two different people), you MAY include alternative queries targeting DIFFERENT entities, NOT just rephrasings of the same query.

Good alternatives (different entities):
  Main query: "Janis Joplin children"
  Alternative 1: "Roger Miller children"

Bad alternatives (same entity rephrased):
  Main query: "Janis Joplin children"
  Alternative 1: "Did Janis Joplin have kids"  ← SAME entity, useless

If all possible interpretations refer to the same entity, do NOT output alternatives.
Output at most 2 alternatives (Main query + max 2 Alts = max 3 queries total).

When you do output alternatives, follow this format:
Main query: <your primary search query>
Alternative 1: <first alternative query>
Alternative 2: <second alternative query>

The system will search ALL queries simultaneously and combine results.
Then end with "The new query is: <main_query>" as usual.

## Example
Question: When was Neville A. Stanton's employer founded?

Wikipedia Title: Southampton
The University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students.

Wikipedia Title: Neville A. Stanton
Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton.

Thought: The question asks about the founding year of Neville A. Stanton's employer. From the passage about Neville A. Stanton, I can see his employer is the University of Southampton. From the passage about Southampton, the University of Southampton was founded in 1862. So the answer is 1862.
So the answer is: 1862."""

FINAL_ANSWER_GUIDE = """When you are asked to produce the final answer:

1. Your response must start with "Thought: " followed by your step-by-step reasoning.
2. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.
3. Use the information from the provided knowledge context and your own knowledge to answer the question.
4. If the knowledge is insufficient, answer the question based on your own knowledge.
5. Be precise, concise, and answer the question directly and completely.
6. Use the exact wording from the passages (e.g. "Mondays" not "Monday", "for crafting and voting on legislation" not paraphrases).
7. For factual questions, provide the specific fact or entity name.
8. For temporal questions, provide the specific date, year, or time period.
9. Do NOT include explanatory preambles like "Based on..." or "The passages state..." or "According to..." in the Answer line.

Example:
Wikipedia Title: Southampton
The University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students.

Wikipedia Title: Neville A. Stanton
Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton.

Question: When was Neville A. Stanton's employer founded?
Thought: The question asks about the founding year of Neville A. Stanton's employer. From the passage about Neville A. Stanton, I can see his employer is the University of Southampton. From the passage about Southampton, the University of Southampton was founded in 1862. So the answer is 1862.
Answer: 1862."""

# Final answer 模板（根据 llm_judge_mode 选择，对齐 pipeline.py 的 ANSWER_TEMPLATE）
CLOSED_BOOK_FINAL_GUIDE = """1. Use ONLY the information from the provided knowledge context and try your best to answer the question.
2. If the knowledge is insufficient, reject to answer the question.
3. Be precise, concise, and answer the question directly and completely.
4. Do NOT include explanatory preambles like "Based on..." or "The passages state..." or "According to..." in the Answer line.
5. For factual questions, provide the specific fact or entity name.
6. For temporal questions, provide the specific date, year, or time period."""

OPEN_BOOK_FINAL_GUIDE = """1. If the knowledge is insufficient, answer the question based on your own knowledge.
2. Be precise and concise in your answer.
3. Do NOT include explanatory preambles like "Based on..." or "The passages state..." or "According to..." in the Answer line.
4. For factual questions, provide the specific fact or entity name.
5. For temporal questions, provide the specific date, year, or time period."""

IRCOT_PROMPT_SHORT = """## Search Results
{context}

Continue your reasoning. If you can answer, end with "So the answer is: <answer>". If you need more information, end with "The new query is: <query>"."""

JUDGE_PROMPT = """You are an expert evaluator. Determine if the predicted answer is correct based on the question and gold answer.
Be lenient — if the predicted answer contains the gold answer or is semantically equivalent, consider it correct.

Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {predicted}

Return only "1" (correct) or "0" (incorrect):"""


# ═══════════════════════════════════════════════════════════════════════════
# LangGraph State
# ═══════════════════════════════════════════════════════════════════════════

class IRCoTState(TypedDict):
    """每个 question 的 IRCoT 状态。"""
    question: str                                           # 原始问题
    current_query: str                                      # 当前检索 query
    all_passages_by_step: List[Dict[str, str]]              # 按步分片: [step0_dict, step1_dict, ...]
    all_passage_fts: set                                    # 全文去重用 set（全局去重）
    messages: Annotated[List, add_messages]                 # 对话历史（自动累积）
    per_step_passages: List[List[Tuple[str, str, float]]]   # 每步检索原始结果（含 score）
    step: int                                               # 当前步数
    max_steps: int                                          # 最大步数
    step0_passages: List[Tuple[str, str, float]]            # 初始检索缓存
    # 外部依赖（由编译时注入，运行时不变）
    retriever: Any                                          # MCSRerankPathRetriever
    async_client: Any                                       # AsyncOpenAI
    embed_client: Any                                       # EmbeddingClient
    qa_eval_mode: str                                       # "hipporag-em-f1" | "llm-judge"
    llm_judge_mode: str                                     # "open-book" | "closed-book"
    top_k_passages: int                                     # 最终 QA 的 top-k
    llm_model: str                                          # LLM 模型名
    query_history: List[str]                                # 历史 query 记录
    alternative_queries: List[str]                          # 候选查询列表（并行检索用）
    multi_queries_per_step: List[List[str]]                 # 每一步实际检索的 query 列表
    fresh_count: int                                        # 最近一次检索的新增 passage 数
    # 内部路由标记
    _decision: str                                          # "answer" | "retrieve" | "stop"
    # 输出
    final_answer: Optional[str]
    ordered_passage_keys: List[str]
    final_prompt_messages: Optional[List[Dict]]


# ═══════════════════════════════════════════════════════════════════════════
# Node: Initialize Graph
# ═══════════════════════════════════════════════════════════════════════════

def node_initialize(state: IRCoTState) -> IRCoTState:
    """Initialize state with SystemMessage + Step0 passages as first user message."""
    step0 = state["step0_passages"]
    step0_dict: Dict[str, str] = {}
    all_passage_fts: set = set()

    for title, text, _score in step0:
        ft = ' '.join((title + '\n' + text).split())
        if ft not in all_passage_fts:
            all_passage_fts.add(ft)
            if title not in step0_dict:
                step0_dict[title] = text
            else:
                key = f"{title}___{hash(ft) & 0xFFFFFFFF:08x}"
                step0_dict[key] = text

    all_passages_by_step = [step0_dict]
    per_step_passages = [step0]

    # 构建对话消息：SystemMessage + 第一轮 user message
    # step0 是 (title, text, score) 三元组，取前两个字段
    context = _format_passage_context([(t, txt) for t, txt, _ in step0])
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"## Question\n{state['question']}\n\n## Initial Retrieved Passages\n{context}\n\nPlease analyze and determine if you can answer, or need to search for more information.")
    ]

    return {
        **state,
        "all_passages_by_step": all_passages_by_step,
        "all_passage_fts": all_passage_fts,
        "messages": messages,
        "per_step_passages": per_step_passages,
        "current_query": state["question"],
        "step": 0,
        "final_answer": None,
        "ordered_passage_keys": [],
        "final_prompt_messages": None,
        "_decision": "continue",
        "query_history": [],
        "alternative_queries": [],
        "multi_queries_per_step": [],
        "fresh_count": 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node: LLM Reasoning
# ═══════════════════════════════════════════════════════════════════════════

async def node_llm_reason(state: IRCoTState) -> IRCoTState:
    """LLM 推理：基于 messages 对话历史，判断 answer 或 new_query。

    System message 已在 initialize 时设置，本轮只追加简短的 user content
    （当前检索结果 + 当前 query），不再重复系统指令。
    """
    all_passages_by_step = state["all_passages_by_step"]
    messages = list(state["messages"])
    current_query = state.get("current_query", state["question"])
    step = state["step"] + 1
    async_client = state["async_client"]
    llm_model = state["llm_model"]

    # 当前最新一步的检索结果作为上下文
    context_titles_texts: List[Tuple[str, str]] = []
    if all_passages_by_step:
        last_step_dict = all_passages_by_step[-1]
        for t in list(last_step_dict.keys())[:10]:
            context_titles_texts.append((t, last_step_dict[t]))
    context = _format_passage_context(context_titles_texts)

    # 简短 user message（系统指令已由 system message 提供）
    user_content = IRCOT_PROMPT_SHORT.format(context=context)
    human_msg = HumanMessage(content=user_content)
    messages.append(human_msg)

    # 调用 LLM（完整对话历史）
    try:
        openai_messages = []
        for m in messages:
            if isinstance(m, SystemMessage):
                role = "system"
            elif isinstance(m, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            openai_messages.append({"role": role, "content": m.content})
        raw = await _llm_call_with_retry(
            async_client, llm_model, openai_messages,
            temperature=0.0,
        )
        response = _clean_llm_content(raw)
    except Exception as e:
        log(f"[ERROR] IRCoT step {step} LLM call failed: {e}")
        response = ""

    ai_msg = AIMessage(content=response)
    messages.append(ai_msg)

    answer_found = "So the answer is:" in response
    has_new_query = "The new query is:" in response

    new_state: IRCoTState = {
        **state,
        "messages": messages,
        "step": step,
    }

    if answer_found:
        log(f"  → Step {step}: answer found, stopping")
        new_state["_decision"] = "answer"
        return new_state

    if has_new_query:
        try:
            new_query = response.split("The new query is:")[1].strip()
        except Exception:
            log(f"  → Step {step}: cannot parse new query, using existing knowledge for final answer")
            new_state["_decision"] = "answer"
            return new_state

        if not new_query or new_query == current_query:
            log(f"  → Step {step}: new query same as current or empty, using existing knowledge for final answer")
            new_state["_decision"] = "answer"
            return new_state

        # 解析 Alternative queries（多候选并行检索）
        alt_queries = []
        if "Main query:" in response or "Alternative 1:" in response:
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("Alternative 1:"):
                    alt_q = line.split("Alternative 1:")[1].strip()
                    if alt_q and alt_q != new_query:
                        alt_queries.append(alt_q)
                elif line.startswith("Alternative 2:"):
                    alt_q = line.split("Alternative 2:")[1].strip()
                    if alt_q and alt_q != new_query and alt_q not in alt_queries:
                        alt_queries.append(alt_q)

        if alt_queries:
            new_state["alternative_queries"] = alt_queries
            log(f"  → Step {step}: new query (with {len(alt_queries)} alternatives)")
            log(f"     [main] {new_query[:80]}")
            for ai, aq in enumerate(alt_queries):
                log(f"     [alt_{ai+1}] {aq[:80]}")
        else:
            new_state["alternative_queries"] = []
            log(f"  → Step {step}: new query = '{new_query[:80]}...'")

        new_state["current_query"] = new_query

        # 语义相似度检测：如果新 query 与历史 query 高度相似，提前结束检索
        query_history = state.get("query_history", [])
        if query_history:
            embed_client = state.get("embed_client")
            if embed_client:
                vec_new = embed_client.encode(new_query)
                vec_old = embed_client.encode(query_history[-1])
                norm_new = np.linalg.norm(vec_new)
                norm_old = np.linalg.norm(vec_old)
                if norm_new > 0 and norm_old > 0:
                    similarity = float(np.dot(vec_new, vec_old) / (norm_new * norm_old))
            if similarity > 0.85:
                log(f"  → Step {step}: new query too similar to previous (cosine={similarity:.2f} > 0.85), early stop")
                log(f"     Last: {query_history[-1][:80]}")
                log(f"     New:  {new_query[:80]}")
                new_state["_decision"] = "answer"
                return new_state

        new_state["query_history"] = state.get("query_history", []) + [current_query]
        new_state["_decision"] = "retrieve"
        return new_state

    # 既没有 answer 也没有 new_query
    log(f"  → Step {step}: no clear action, using existing knowledge for final answer")
    new_state["_decision"] = "answer"
    return new_state


# ═══════════════════════════════════════════════════════════════════════════
# Router: decide next node after llm_reason
# ═══════════════════════════════════════════════════════════════════════════

def router_after_reason(state: IRCoTState) -> str:
    decision = state.get("_decision", "stop")
    return decision  # "retrieve" | "answer" | "stop"


# ═══════════════════════════════════════════════════════════════════════════
# Node: Retrieve (MCS-RP)
# ═══════════════════════════════════════════════════════════════════════════




# ═══════════════════════════════════════════════════════════════════════════
# Router: check max steps after retrieve
# ═══════════════════════════════════════════════════════════════════════════

def router_after_retrieve(state: IRCoTState) -> str:
    """检查步数上限，超限则进 final_answer，未超限则继续 LLM 推理。"""
    step = state["step"]
    max_steps = state["max_steps"]
    if step >= max_steps:
        log(f"  → Step {step}: max_steps reached ({max_steps})")
        return "final_answer"
    return "llm_reason"


# ═══════════════════════════════════════════════════════════════════════════
# Node: Multi-Retrieve（多候选并行检索）
# ═══════════════════════════════════════════════════════════════════════════

async def node_multi_retrieve(state: IRCoTState) -> IRCoTState:
    """并行检索当前 query + alternative queries（最多 3 个），合并结果。

    检索结果同时写入 messages（符合 agent 对话范式），
    以及保留 all_passages_by_step / per_step_passages（供下游 recall 计算）。
    """
    messages = list(state.get("messages", []))
    all_passages_by_step = list(state.get("all_passages_by_step", []))
    all_passage_fts = state.get("all_passage_fts", set())
    per_step_passages = list(state.get("per_step_passages", []))
    retriever = state["retriever"]
    current_query = state["current_query"]
    step = state.get("step", 0)

    # 收集需要检索的 query 列表
    queries_to_search = [current_query]
    alt_queries = state.get("alternative_queries", [])
    for q in alt_queries[:2]:
        if q and q != current_query:
            queries_to_search.append(q)

    merged_step_dict: Dict[str, str] = {}
    merged_passages: List[Tuple[str, str, float]] = []
    total_new_count = 0

    for qi, q in enumerate(queries_to_search):
        top_k = 10 if qi == 0 and step == 0 else 5
        try:
            retrieved = retriever.retrieve(q, top_k=top_k)
            for p, score in retrieved:
                ft_new = ' '.join((p.title + '\n' + p.text).split())
                if ft_new not in all_passage_fts:
                    all_passage_fts.add(ft_new)
                    key = p.title
                    if key in merged_step_dict or any(key in step_dict for step_dict in all_passages_by_step):
                        key = f"{p.title}___{hash(ft_new) & 0xFFFFFFFF:08x}"
                    merged_step_dict[key] = p.text
                    total_new_count += 1
                merged_passages.append((p.title, p.text, float(score)))
        except Exception as e:
            log(f"[ERROR] Multi-retrieve failed for query '{q}': {e}")
            continue

    multi_queries_this_step = queries_to_search.copy()

    if merged_passages:
        all_passages_by_step.append(merged_step_dict)
        per_step_passages.append(merged_passages)
        fresh_count = total_new_count
        if len(queries_to_search) > 1:
            log(f"  → multi-retrieve ({len(queries_to_search)} queries, top-5 each): {total_new_count} new passages")
            for qi, q in enumerate(queries_to_search):
                tag = "main_query" if qi == 0 else f"alt_{qi}"
                log(f"     [{tag}] {q}")
    else:
        all_passages_by_step.append({})
        per_step_passages.append([])
        fresh_count = 0
        log(f"  → All queries failed, inserted empty step")

    # △ 检索结果作为 tool result 追加入 messages
    if merged_step_dict:
        result_lines = []
        for key in list(merged_step_dict.keys())[:10]:
            text = merged_step_dict[key]
            title = key.split("___")[0] if "___" in key else key
            result_lines.append(f"Wikipedia Title: {title}\n{text}")
        result_text = "\n\n".join(result_lines)
        messages.append(HumanMessage(content=(
            f"## Search Results\n"
            f"Queries: {', '.join(queries_to_search)}\n\n"
            f"{result_text}"
        )))

    prev_multi = state.get("multi_queries_per_step", [])
    multi_queries_per_step = prev_multi + [multi_queries_this_step]

    return {
        **state,
        "messages": messages,
        "all_passages_by_step": all_passages_by_step,
        "all_passage_fts": all_passage_fts,
        "per_step_passages": per_step_passages,
        "fresh_count": fresh_count,
        "multi_queries_per_step": multi_queries_per_step,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Node: Stop (no answer found, or early stop)
# ═══════════════════════════════════════════════════════════════════════════

def node_stop(state: IRCoTState) -> IRCoTState:
    """走不投了，强制结束。"""
    return {**state, "final_answer": ""}


# ═══════════════════════════════════════════════════════════════════════════
# Node: Final Answer (HippoRAG2 / LLM-Judge)
# ═══════════════════════════════════════════════════════════════════════════

async def node_final_answer(state: IRCoTState) -> IRCoTState:
    """Final answer：基于 messages 对话历史 + CoT 回答问题。

    不加重复的 passage 或 system 指令（已在 system message 中），
    让 LLM 先思考再回答，直接基于已有对话推理。
    """
    question = state["question"]
    async_client = state["async_client"]
    llm_model = state["llm_model"]
    llm_judge_mode = state.get("llm_judge_mode", "closed-book")
    messages = list(state.get("messages", []))

    # 根据 llm_judge_mode 选择 final answer guide（对齐 pipeline 的 ANSWER_TEMPLATE）
    final_guide = CLOSED_BOOK_FINAL_GUIDE if llm_judge_mode == "closed-book" else OPEN_BOOK_FINAL_GUIDE

    # 基于已有对话 + final_guide 约束，不重复塞 passage
    messages.append(HumanMessage(content=(
        f"Based on the entire conversation above (including all retrieved information and your reasoning steps), "
        f"please answer the original question.\n\n"
        f"{final_guide}\n\n"
        f"Original question: {question}\n\nThought: "
    )))

    # 转 openai 格式
    openai_messages = []
    for m in messages:
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, HumanMessage):
            role = "user"
        else:
            role = "assistant"
        openai_messages.append({"role": role, "content": m.content})

    final_prompt_messages = list(openai_messages)

    try:
        raw = await _llm_call_with_retry(
            async_client, llm_model, openai_messages,
            temperature=0.0,
        )
        response = _clean_llm_content(raw)
        final_answer = _extract_final_answer(response)
        messages.append(AIMessage(content=response))
        log(f"[IRCoT-Final] answer: {final_answer[:80]}...")
    except Exception as e:
        log(f"[ERROR] Final answer LLM call failed: {e}")
        final_answer = ""
        final_prompt_messages = None

    return {
        **state,
        "messages": messages,
        "final_answer": final_answer,
        "ordered_passage_keys": [],
        "final_prompt_messages": final_prompt_messages,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Build LangGraph
# ═══════════════════════════════════════════════════════════════════════════

def build_qa_graph():
    """构建 IRCoT 的 langgraph StateGraph。

    图结构：
       START → initialize → llm_reason
                               │
                     ┌─────────┼─────────┐
                     │         │         │
                   answer multi_retrievestop
                     │         │         │
                     │    ┌────┴────┐    │
                     │    │  步数上限?│    │
                     │    ├──是──┬──否─┤    │
                     │    │     │     │    │
                     ▼    ▼     │                 │
               final_answer    │
                               │  (步数未超, 返回 llm_reason 继续迭代)
                               ▼
                           llm_reason  ←──────────┐
                               │                  │
                     ┌─────────┼─────────┐        │
                     │         │         │        │
                   answer multi_retrievestop       │
                     │         │         │        │
                     │    ┌────┴────┐    │        │
                     │    │  步数上限?│    │        │
                     │    ├──是──┬──否─┤    │        │
                     │    │     │     │    │        │
                     ▼    ▼     │     ▼    │        │
               final_answer ←──┘   ───────┘─────────┘
                     │
                     ▼
                    END

    多步迭代说明：
    - 每一步 LLM 推理后都可能产出 Main query + Alternative 1~2（最多 3 个 query）
    - multi_retrieve 并行检索所有 query（各 top-5），合并去重后返回
    - LLM 综合看到所有候选结果，继续推理
    - 每一步都可能产生新的 alternative queries，直到答案或步数超限 → final_answer
    """
    builder = StateGraph(IRCoTState)

    builder.add_node("initialize", node_initialize)
    builder.add_node("llm_reason", node_llm_reason)
    builder.add_node("retrieve", node_multi_retrieve)
    builder.add_node("stop", node_stop)
    builder.add_node("final_answer", node_final_answer)

    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "llm_reason")

    # LLM reason → 分支
    builder.add_conditional_edges(
        "llm_reason",
        router_after_reason,
        {
            "retrieve": "retrieve",
            "answer": "final_answer",
            "stop": "stop",
        },
    )

    # retrieve → 检查步数 → llm_reason 或 final_answer
    builder.add_conditional_edges(
        "retrieve",
        router_after_retrieve,
        {
            "final_answer": "final_answer",
            "llm_reason": "llm_reason",
        },
    )

    builder.add_edge("stop", END)
    builder.add_edge("final_answer", END)

    return builder.compile()


# ═══════════════════════════════════════════════════════════════════════════
# _build_result_dict: 从 graph final_state 构建结果字典
# ═══════════════════════════════════════════════════════════════════════════

async def _build_result_dict(
    idx: int,
    sample: dict,
    final_state: IRCoTState,
    retrieval_time: float,
    ircot_time: float,
    llm_model: str,
    qa_eval_mode: str,
    async_client: Any,
    dataset: str,
) -> dict:
    """从 graph 的 final_state 构建单条结果字典。"""
    llm_judge_mode = final_state.get("llm_judge_mode", "closed-book")
    final_answer = final_state.get("final_answer", "") or ""
    final_prompt_messages = final_state.get("final_prompt_messages", None)
    all_passages_by_step = final_state.get("all_passages_by_step", [])
    messages = final_state.get("messages", [])
    per_step_passages = final_state.get("per_step_passages", [])
    ordered_passage_keys = final_state.get("ordered_passage_keys", [])

    # 展平 all_passages_by_step 用于查找
    all_passages_flat: Dict[str, str] = {}
    for step_dict in all_passages_by_step:
        for k, v in step_dict.items():
            if k not in all_passages_flat:
                all_passages_flat[k] = v

    if not ordered_passage_keys:
        # 如果 final_answer node 没走到，手动构建
        seen_titles = set()
        for step_passes in per_step_passages:
            for t, _, _ in step_passes:
                if t in all_passages_flat and t not in seen_titles:
                    seen_titles.add(t)
                    ordered_passage_keys.append(t)
                else:
                    for k in all_passages_flat:
                        if k.startswith(t + "___") and k not in seen_titles:
                            seen_titles.add(k)
                            ordered_passage_keys.append(k)
                            break

    # ── 计算 recall（兼容 hotpotqa/2wikimultihopqa/musique 的数据格式） ──
    def _normalize_text(t):
        return " ".join(t.split())
    gold_fulltexts = set()
    n_gold = 0
    # 格式1: paragraphs（musique/hotpotqa）
    if "paragraphs" in sample:
        for p in sample.get("paragraphs", []):
            if p.get("is_supporting", False):
                full = p["title"] + "\n" + p.get("text", p.get("paragraph_text", ""))
                gold_fulltexts.add(_normalize_text(full))
                n_gold += 1
    # 格式2: context + supporting_facts（2wikimultihopqa/hotpotqa）
    if not gold_fulltexts and "context" in sample and "supporting_facts" in sample:
        gold_titles = set(sf[0] for sf in sample.get("supporting_facts", []))
        for title, paras in sample.get("context", []):
            if title in gold_titles:
                full = title + "\n" + " ".join(paras)
                gold_fulltexts.add(_normalize_text(full))
                n_gold += 1
    final_fulltexts_all = [
        _normalize_text(k.split("___")[0] + "\n" + all_passages_flat.get(k, ""))
        for k in ordered_passage_keys
    ]
    final_recall_all = sum(1 for ft in final_fulltexts_all if ft in gold_fulltexts) / max(n_gold, 1)
    final_recall_10 = sum(1 for ft in final_fulltexts_all[:10] if ft in gold_fulltexts) / max(n_gold, 1) if n_gold else 0

    gold_list = _build_gold_list(sample)

    if qa_eval_mode == "hipporag-em-f1":
        em_score = _exact_match_multi(gold_list, final_answer) if final_answer else 0.0
        f1_score_val = _f1_score_multi(gold_list, final_answer) if final_answer else 0.0
        is_correct = em_score >= 1.0
    else:
        # llm-judge 模式：LLM 调用 JUDGE_PROMPT 打分
        if final_answer:
            judge_raw = await _llm_judge_ask(
                async_client, llm_model,
                JUDGE_PROMPT.format(question=sample["question"], gold_answer=gold_list[0], predicted=final_answer),
            )
            is_correct = (judge_raw == "1")
        else:
            is_correct = False
        em_score = 1.0 if is_correct else 0.0
        f1_score_val = em_score

    total_steps = len(per_step_passages)
    step0_count = len(per_step_passages[0]) if per_step_passages else 0
    extra_steps = total_steps - 1
    extra_passages_count = sum(len(p) for p in per_step_passages[1:]) if total_steps > 1 else 0
    n_ircot_steps_used = min(total_steps, final_state.get("max_steps", 3))

    result = {
        "id": sample.get("id", sample.get("_id", str(idx))),
        "question": sample["question"],
        "gold": gold_list,
        "predicted": final_answer,
        "correct": is_correct,
        "em": em_score,
        "f1": f1_score_val,
        "messages_text": [
            {"role": "system" if isinstance(m, SystemMessage) else ("user" if isinstance(m, HumanMessage) else "assistant"),
             "content": m.content}
            for m in messages
        ],
        "steps_used": n_ircot_steps_used,
        "step0_count": step0_count,
        "extra_steps": extra_steps,
        "extra_passages": extra_passages_count,
        "final_recall_10": final_recall_10,
        "final_recall_all": final_recall_all,
        "final_gold_titles": (
            [p["title"] for p in sample.get("paragraphs", []) if p.get("is_supporting", False)]
            if "paragraphs" in sample
            else [sf[0] for sf in sample.get("supporting_facts", [])]
        ),
        "total_passages_used": len(set(t for t, _, _ in [p for step in per_step_passages for p in step])),
        "multi_queries_per_step": final_state.get("multi_queries_per_step", []),
        "per_step_passages_text": [
            [{"title": t, "text": text} for t, text, _ in step]
            for step in per_step_passages
        ],
        "final_prompt_messages": final_prompt_messages,
        "retrieval_time": retrieval_time,
        "ircot_time": ircot_time,
        "llm_judge_mode": llm_judge_mode,
    }
    return result


async def _run_sample(
    idx: int,
    sample: dict,
    step0_cache: list,
    retriever: Any,
    embed_client: Any,
    async_client: Any,
    sem: asyncio.Semaphore,
    llm_model: str,
    max_steps: int,
    top_k_passages: int,
    qa_eval_mode: str,
    llm_judge_mode: str,
    pbar,
) -> dict:
    """运行单个 sample 的 IRCoT 流程。"""
    async with sem:
        graph = build_qa_graph()
        q = sample["question"]
        t0 = time.time()

        step0_passages = []
        if idx >= 0 and idx < len(step0_cache):
            for entry in step0_cache[idx]:
                if len(entry) >= 2:
                    title, text = entry[0], entry[1]
                    score = entry[2] if len(entry) >= 3 else 0.0
                    step0_passages.append((title, text, score))

        if not step0_passages:
            retrieved = retriever.retrieve(q, top_k=10)
            for p, score in retrieved:
                step0_passages.append((p.title, p.text, float(score)))

        step0_passages = step0_passages[:10]
        t1 = time.time()

        initial_state: IRCoTState = {
            "question": q,
            "current_query": q,
            "all_passages_by_step": [],
            "all_passage_fts": set(),
            "messages": [],
            "per_step_passages": [],
            "step": 0,
            "max_steps": max_steps,
            "step0_passages": step0_passages,
            "retriever": retriever,
            "async_client": async_client,
            "embed_client": embed_client,
            "qa_eval_mode": qa_eval_mode,
            "llm_judge_mode": llm_judge_mode,
            "top_k_passages": top_k_passages,
            "llm_model": llm_model,
            "final_answer": None,
            "ordered_passage_keys": [],
            "final_prompt_messages": None,
            "_decision": "continue",
            "query_history": [],
            "alternative_queries": [],
            "fresh_count": 0,
        }

        try:
            final_state = await graph.ainvoke(initial_state)
        except Exception as e:
            log(f"[ERROR] graph.ainvoke failed for idx {idx}: {e}")
            import traceback
            traceback.print_exc()
            final_state = initial_state
            final_state["final_answer"] = ""

        t2 = time.time()
        result = await _build_result_dict(
            idx, sample, final_state, t1 - t0, t2 - t0,
            llm_model, qa_eval_mode, async_client, "",
        )
        em_score = result.get("em", 0.0)
        is_correct = result.get("correct", False)
        q = result.get("question", "")
        pbar.update(1)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# step_qa_langgraph: 主入口（并行执行所有 samples）
# ═══════════════════════════════════════════════════════════════════════════

def step_qa_langgraph(
    dataset: str, working_dir: str,
    embed_client, reranker_client,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
    num_samples: int = 0,
    max_steps: int = DEFAULT_MAX_STEPS,
    top_k_passages: int = DEFAULT_TOP_K_PASSAGES,
    qa_eval_mode: str = "hipporag-em-f1",
    llm_judge_mode: str = "closed-book",
    resume_from: Optional[str] = None,
):
    """IRCoT QA 的 langgraph 实现。

    整体流程：
      1. 加载 KG + Stores + Retriever
      2. 构建并编译 StateGraph
      3. 对每个 sample 构造初始 state → graph.ainvoke(state)
      4. 评估 & 保存结果

    resume_from: 已有结果文件路径。只重跑其中 predicted 为空的样本。
    """
    from retrievers.mcs_rp import MCSRerankPathRetriever
    from infra.knowledge_graph import KnowledgeGraph
    from infra.embedding_store import EmbeddingStore
    from infra.retrieval import Passage

    # ── 加载 KG + Stores ──
    log(f"[IRCoT-LG-QA] 加载 {dataset} 的 KG 和 stores...")
    llm_slug = _make_model_slug(llm_model)
    kg_path = os.path.join(working_dir, f"graph_flat_{llm_slug}.pkl")
    if not os.path.exists(kg_path):
        raise FileNotFoundError(f"请先运行 build_graph: {kg_path} 不存在")

    kg = KnowledgeGraph.load(kg_path)
    chunk_store = EmbeddingStore("chunk", working_dir)
    entity_store = EmbeddingStore("entity", working_dir)
    fact_store = EmbeddingStore("fact", working_dir)

    chunk_map_path = os.path.join(working_dir, "chunk_map.json")
    chunk_to_passage = {}
    if os.path.exists(chunk_map_path):
        with open(chunk_map_path) as f:
            cmap = json.load(f)
            chunk_to_passage = cmap.get("chunk_to_passage", cmap)

    chunks_list = [(_hashlib.md5(t.encode()).hexdigest(), t) for t in chunk_store.texts]

    corpus_path = os.path.join(DATA_DIR, f"{dataset}_corpus.json")
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join(DATA_DIR, f"{dataset}.json")
    with open(corpus_path) as f:
        corpus = json.load(f)
    if isinstance(corpus, dict):
        passage_list = [Passage(id=t, title=t, text=' '.join(paras) if isinstance(paras, list) else str(paras))
                        for t, paras in corpus.items()]
    else:
        passage_list = [Passage(id=p.get("id", p["title"]), title=p["title"], text=p["text"]) for p in corpus]
    log(f"[IRCoT-LG-QA] Total passages: {len(passage_list)}")

    index_result = {
        "kg": kg, "chunk_store": chunk_store,
        "entity_store": entity_store, "fact_store": fact_store,
        "chunk_to_passage": chunk_to_passage,
        "chunks_list": chunks_list, "passages": passage_list,
    }

    samples_path = os.path.join(DATA_DIR, f"{dataset}.json")
    if not os.path.exists(samples_path):
        log(f"[IRCoT-LG-QA] 没有 samples 文件: {samples_path}，跳过")
        return

    with open(samples_path) as f:
        all_samples = json.load(f)
    if num_samples > 0:
        all_samples = all_samples[:num_samples]
    n = len(all_samples)
    log(f"[IRCoT-LG-QA] {n} 个 sample | LLM: {llm_model} | max_steps: {max_steps}")

    # ── 加载 Step0 缓存 ──
    step0_path = os.path.join(working_dir, "ircot_retrieval_step0.json")
    step0_cache = []
    if os.path.exists(step0_path):
        with open(step0_path) as f:
            step0_cache = json.load(f)
        log(f"[IRCoT-LG-QA] 加载 Step0 缓存: {len(step0_cache)} 条")

    # ── 准备 retriever ──
    _api_key = _get_api_key(llm_base_url)
    retriever_llm = ChatOpenAI(
        model=llm_model,
        api_key=_api_key,
        base_url=llm_base_url,
        temperature=0.0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    retriever = MCSRerankPathRetriever(
        index_result=index_result, embedding_client=embed_client,
        reranker_client=reranker_client, llm=retriever_llm,
    )

    from httpx import Timeout
    async_client = AsyncOpenAI(
        base_url=llm_base_url,
        api_key=_api_key,
        timeout=Timeout(120.0, connect=30.0),
    )

    # ── 构建编译好的 graph ──
    graph = build_qa_graph()
    log(f"[IRCoT-LG-QA] StateGraph compiled")

    # ── Resume 模式：加载已有结果，只跑 failed 样本 ──
    resume_existing_results = None
    resume_out_path = None
    if resume_from:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"Resume file not found: {resume_from}")
        with open(resume_from) as f:
            resume_data = json.load(f)
        resume_existing_results = resume_data.get("results", [])
        log(f"[Resume] 加载已有结果: {len(resume_existing_results)} 条")

        # 筛选出 predicted 为空的 sample id
        resume_failed_ids = set()
        for r in resume_existing_results:
            pred = r.get("predicted")
            if pred is None or pred == "" or pred == []:
                resume_failed_ids.add(r.get("id", ""))

        log(f"[Resume] 需要重跑的 failed 问题: {len(resume_failed_ids)} 个")

        # 过滤 all_samples，只保留这些 id
        filtered_samples = [s for s in all_samples
                           if s.get("id", s.get("_id", "")) in resume_failed_ids]
        log(f"[Resume] 过滤后样本数: {len(filtered_samples)} (原始: {len(all_samples)})")

        if len(filtered_samples) == 0:
            log("[Resume] 没有需要重跑的样本，跳过")
            return

        all_samples = filtered_samples
        n = len(all_samples)

        # 输出到独立文件，避免覆盖原文件
        base, ext = os.path.splitext(resume_from)
        resume_out_path = f"{base}_resume{ext}"
        log(f"[Resume] 重跑结果将保存到: {resume_out_path}")

    # ── 结果收集 ──
    if resume_from:
        out_path = resume_out_path
    else:
        out_filename = f"ircot_lg_results_{llm_model}_{dataset}_steps{max_steps}_{qa_eval_mode}.json"
        out_path = os.path.join(EXP_DIR, out_filename)
    os.makedirs(EXP_DIR, exist_ok=True)

    results: List[Optional[Dict]] = [None] * n
    _all_final_recalls_10: List[float] = []
    _all_final_recalls_all: List[float] = []

    # ── 并行执行 ──
    from tqdm import tqdm
    pbar = tqdm(total=n, desc="IRCoT-LG", unit="q")
    log(f"[IRCoT-LG-QA] 开始评估 (max_steps={max_steps}, concurrency=8)...")

    sem = asyncio.Semaphore(8)
    if resume_from:
        # Resume 模式下 step0_cache 索引不匹配，跳过缓存（fallback 到实时检索）
        log("[Resume] 跳过 step0_cache（索引不匹配），使用实时检索")
        tasks = [
            _run_sample(
                -1, all_samples[i], step0_cache, retriever, embed_client,
                async_client, sem, llm_model, max_steps, top_k_passages,
                qa_eval_mode, llm_judge_mode, pbar,
            )
            for i in range(n)
        ]
    else:
        tasks = [
            _run_sample(
                i, all_samples[i], step0_cache, retriever, embed_client,
                async_client, sem, llm_model, max_steps, top_k_passages,
                qa_eval_mode, llm_judge_mode, pbar,
            )
            for i in range(n)
        ]

    async def _run_all():
        results_list = await asyncio.gather(*tasks)
        return results_list

    all_results = asyncio.run(_run_all())
    pbar.close()

    for i, result in enumerate(all_results):
        if result is not None:
            results[i] = result
            _all_final_recalls_10.append(result.get("final_recall_10", 0.0))
            _all_final_recalls_all.append(result.get("final_recall_all", 0.0))
    results = [r for r in results if r is not None]

    # ── Resume 模式：合并新结果到已有结果 ──
    if resume_from and resume_existing_results is not None:
        log(f"[Resume] 合并新结果到已有结果...")
        # 按 id 索引新结果
        new_by_id = {}
        for r in results:
            rid = r.get("id", "")
            new_by_id[rid] = r

        # 替换已有结果中对应 id 的条目
        updated_count = 0
        for i, existing in enumerate(resume_existing_results):
            eid = existing.get("id", "")
            if eid in new_by_id:
                resume_existing_results[i] = new_by_id[eid]
                updated_count += 1
        log(f"[Resume] 更新了 {updated_count}/{len(results)} 条结果")

        # 保存合并后的完整结果
        merged_out_path = resume_from  # 覆盖原文件
        resume_data["results"] = resume_existing_results
        # 重新计算 metrics
        done_list = [r for r in resume_existing_results if r is not None]
        if done_list:
            correct_done = sum(1 for r in done_list if r["correct"])
            avg_em = float(np.mean([r["em"] for r in done_list]))
            avg_f1 = float(np.mean([r["f1"] for r in done_list]))
            avg_steps = float(np.mean([r["steps_used"] for r in done_list]))
            avg_extra_passages = float(np.mean([r["extra_passages"] for r in done_list]))
            resume_data["qa"] = {"ExactMatch": round(avg_em, 4), "F1": round(avg_f1, 4)}
            resume_data["accuracy"] = correct_done / max(len(done_list), 1) * 100
            resume_data["correct"] = correct_done
            resume_data["total"] = len(done_list)
            resume_data["avg_ircot_steps"] = round(avg_steps, 2)
            resume_data["avg_extra_passages"] = round(avg_extra_passages, 1)

        with open(merged_out_path, "w") as f:
            json.dump(resume_data, f, indent=2, ensure_ascii=False)
        log(f"[Resume] 合并结果已保存到: {merged_out_path}")
        out_path = merged_out_path
    else:
        _save_checkpoint(results, out_path, dataset, llm_model, qa_eval_mode,
                         max_steps, top_k_passages, llm_judge_mode)

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    avg_em = float(np.mean([r["em"] for r in results]))
    avg_f1 = float(np.mean([r["f1"] for r in results]))
    avg_steps = float(np.mean([r["steps_used"] for r in results]))
    avg_extra = float(np.mean([r["extra_passages"] for r in results]))
    accuracy = correct / max(total, 1) * 100

    print(f"\n{'='*55}", flush=True)
    print(f"IRCoT-LangGraph Final Results ({dataset}, {qa_eval_mode}):", flush=True)
    final_recall_mean_10 = float(np.mean(_all_final_recalls_10)) if _all_final_recalls_10 else 0.0
    final_recall_mean_all = float(np.mean(_all_final_recalls_all)) if _all_final_recalls_all else 0.0
    print(f"  EM={avg_em:.4f}  F1={avg_f1:.4f}  Acc={accuracy:.1f}%  ({correct}/{total})", flush=True)
    print(f"  Final Recall@10: {final_recall_mean_10:.1%}  Recall@all: {final_recall_mean_all:.1%}", flush=True)
    print(f"  Avg IRCoT steps: {avg_steps:.1f}  Avg extra passages: {avg_extra:.0f}", flush=True)
    print(f"{'='*55}", flush=True)
    print(f"Saved to {out_path}", flush=True)


def _save_checkpoint(results, out_path, dataset, llm_model, qa_eval_mode,
                     max_steps, top_k_passages, llm_judge_mode):
    """保存检查点（与 ircot.py 的 _save_checkpoint 对齐）"""
    done_list = [r for r in results if r is not None]
    if not done_list:
        return
    correct_done = sum(1 for r in done_list if r["correct"])
    avg_em = float(np.mean([r["em"] for r in done_list]))
    avg_f1 = float(np.mean([r["f1"] for r in done_list]))
    avg_steps = float(np.mean([r["steps_used"] for r in done_list]))
    avg_extra_passages = float(np.mean([r["extra_passages"] for r in done_list]))
    with open(out_path, "w") as f:
        json.dump({
            "method": "IRCoT_LangGraph_MCSRP",
            "dataset": dataset,
            "llm_model": llm_model,
            "qa_eval_mode": qa_eval_mode,
            "config": {
                "max_steps": max_steps,
                "top_k_passages": top_k_passages,
                "llm_judge_mode": llm_judge_mode,
            },
            "qa": {
                "ExactMatch": round(avg_em, 4),
                "F1": round(avg_f1, 4),
            },
            "accuracy": correct_done / max(len(done_list), 1) * 100,
            "correct": correct_done,
            "total": len(done_list),
            "avg_ircot_steps": round(avg_steps, 2),
            "avg_extra_passages": round(avg_extra_passages, 1),
            "results": [
                {k: r[k] for k in ["id", "question", "gold", "predicted", "correct",
                                    "em", "f1", "messages_text", "steps_used",
                                    "step0_count", "extra_steps", "extra_passages",
                                    "final_recall_10", "final_recall_all",
                                    "per_step_passages_text", "final_prompt_messages",
                                    "multi_queries_per_step"]}
                for r in done_list
            ],
        }, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# Main CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IRCoT-LangGraph wrapper for MCS-RerankPath")
    parser.add_argument("dataset", choices=list(DATASET_MAP.keys()))
    parser.add_argument("mode", choices=["retrieve", "qa", "full"])
    parser.add_argument("--embed-backend", choices=["hash", "local", "http"], default="hash")
    parser.add_argument("--embed-base-url", default="")
    parser.add_argument("--embed-model", default="minilm-embedding")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--reranker-base-url", default=DEFAULT_RERANKER_BASE_URL)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--top-k-passages", type=int, default=DEFAULT_TOP_K_PASSAGES)
    parser.add_argument("--qa-eval", choices=["llm-judge", "hipporag-em-f1"], default="hipporag-em-f1")
    parser.add_argument("--llm-judge-mode", choices=["open-book", "closed-book"], default="closed-book")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="输出详细日志（默认只显示进度条）")
    parser.add_argument("--resume", type=str, default=None,
                        help="重跑模式：指定已有的结果 JSON，只重跑其中 predicted 为空的样本")

    args = parser.parse_args()
    dataset = DATASET_MAP[args.dataset]

    embed_slug = _make_model_slug(args.embed_model)
    working_dir = os.path.join(CACHE_BASE, "flat", embed_slug, dataset)
    os.makedirs(working_dir, exist_ok=True)

    from infra.embeddings import EmbeddingClient, RerankerClient
    embed_client = EmbeddingClient(
        mode=args.embed_backend, base_url=args.embed_base_url, model=args.embed_model,
    )
    reranker_client = RerankerClient(
        mode='http', model='bge-reranker-v2-m3',
        base_url=args.reranker_base_url,
        api_key='vllm', threshold=0.0,
    )

    set_verbose(args.verbose)

    log(f"IRCoT-LangGraph Pipeline: {dataset} / mode={args.mode}")
    log(f"LLM: {args.llm_model} | Embedding: {args.embed_model} (dim={embed_client.dim})")
    log(f"Working dir: {working_dir}")

    if args.mode in ("retrieve", "full"):
        _step_retrieve(dataset, working_dir, embed_client, reranker_client,
                       llm_model=args.llm_model, llm_base_url=args.llm_base_url,
                       num_samples=args.num_samples)
        if args.mode == "retrieve":
            log("[Done] IRCoT-LangGraph Retrieve")
            return

    if args.mode in ("qa", "full"):
        step_qa_langgraph(dataset, working_dir, embed_client, reranker_client,
                          llm_model=args.llm_model, llm_base_url=args.llm_base_url,
                          num_samples=args.num_samples,
                          max_steps=args.max_steps,
                          top_k_passages=args.top_k_passages,
                          qa_eval_mode=args.qa_eval,
                          llm_judge_mode=args.llm_judge_mode,
                          resume_from=args.resume)
        log("[Done] IRCoT-LangGraph QA")


# ── Retrieve-only mode（预缓存 MCS-RP 检索结果） ─────────────────────────
def _step_retrieve(
    dataset: str, working_dir: str,
    embed_client: Any,
    reranker_client: Any,
    num_samples: int = 0,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
):
    from retrievers.mcs_rp import MCSRerankPathRetriever
    from infra.knowledge_graph import KnowledgeGraph
    from infra.embedding_store import EmbeddingStore
    from infra.retrieval import Passage
    from tqdm import tqdm

    log(f"[Retrieve] 加载 {dataset} 的 KG 和 stores...")
    llm_slug = _make_model_slug(llm_model)
    kg_path = os.path.join(working_dir, f"graph_flat_{llm_slug}.pkl")
    if not os.path.exists(kg_path):
        raise FileNotFoundError(f"请先运行 build_graph: {kg_path} 不存在")

    kg = KnowledgeGraph.load(kg_path)
    chunk_store = EmbeddingStore("chunk", working_dir)
    entity_store = EmbeddingStore("entity", working_dir)
    fact_store = EmbeddingStore("fact", working_dir)

    chunk_map_path = os.path.join(working_dir, "chunk_map.json")
    chunk_to_passage = {}
    chunks_list = []
    if os.path.exists(chunk_map_path):
        with open(chunk_map_path) as f:
            cmap = json.load(f)
            chunk_to_passage = cmap.get("chunk_to_passage", cmap)

    corpus_path = os.path.join(DATA_DIR, f"{dataset}_corpus.json")
    if not os.path.exists(corpus_path):
        corpus_path = os.path.join(DATA_DIR, f"{dataset}.json")
    with open(corpus_path) as f:
        corpus = json.load(f)
    if isinstance(corpus, dict):
        passage_list = [
            Passage(id=t, title=t, text=' '.join(paras) if isinstance(paras, list) else str(paras))
            for t, paras in corpus.items()
        ]
    else:
        passage_list = [
            Passage(id=p.get("id", p["title"]), title=p["title"], text=p["text"])
            for p in corpus
        ]
    log(f"[Retrieve] Total passages: {len(passage_list)}")

    index_result = {
        "kg": kg, "chunk_store": chunk_store,
        "entity_store": entity_store, "fact_store": fact_store,
        "chunk_to_passage": chunk_to_passage,
        "chunks_list": chunks_list, "passages": passage_list,
    }

    samples_path = os.path.join(DATA_DIR, f"{dataset}.json")
    if not os.path.exists(samples_path):
        log(f"[Retrieve] 没有 samples 文件: {samples_path}，跳过")
        return
    with open(samples_path) as f:
        all_samples = json.load(f)
    if num_samples > 0:
        all_samples = all_samples[:num_samples]
    log(f"[Retrieve] {len(all_samples)} 个 sample")

    if "tencent" in llm_base_url.lower() or "tokenhub" in llm_base_url.lower():
        _api_key = os.environ.get("TENCENT_API_KEY", "")
    else:
        _api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    llm = ChatOpenAI(
        model=llm_model, api_key=_api_key,
        base_url=llm_base_url, temperature=0.0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    retriever = MCSRerankPathRetriever(
        index_result=index_result, embedding_client=embed_client,
        reranker_client=reranker_client, llm=llm,
    )

    log(f"[Retrieve] Running MCS-RerankPath...")
    results = []
    for sample in tqdm(all_samples, desc="Retrieve", unit="q"):
        q = sample["question"]
        retrieved = retriever.retrieve(q, top_k=20)
        seen = set()
        deduped = []
        for p, score in retrieved:
            ft = p.title + '\n' + p.text
            if ft not in seen:
                seen.add(ft)
                deduped.append((p, score))
        results.append([(p.title, p.title + '\n' + p.text) for p, _ in deduped])

    out_path = os.path.join(working_dir, f"retrieval_MCS_RerankPath_{len(all_samples)}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    log(f"[Retrieve] 结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
