"""
Open Information Extraction (OpenIE) for HippoRAG.

Extracts named entities (with descriptions) and RDF triples from document chunks
using DeepSeek LLM in batches of 10, with parallel threading.
"""

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from langchain_openai import ChatOpenAI

NER_SYSTEM_PROMPT = """You are a knowledge extraction assistant. Given a batch of document passages, extract named entities from each passage.

For each entity, output:
  - "name": the entity name as it appears in the text
  - "description": a short (5-15 word) description that disambiguates this entity (e.g. "a technology company founded by Steve Jobs" vs "a type of fruit")

Requirements:
- If the same entity appears in multiple passages, use the SAME name
- A passage may have 0 entities
- Be thorough: extract people, organizations, locations, events, products, dates, and key concepts

Output JSON format:
{
  "chunks": [
    {
      "chunk_index": 0,
      "entities": [{"name": "...", "description": "..."}, ...]
    },
    ...
  ]
}
输出纯 JSON，不要用 markdown 代码块包裹。"""

TRIPLE_SYSTEM_PROMPT = """You are a knowledge graph construction assistant. Given document passages and their extracted entities, extract RDF-style triples (subject, predicate, object) from each passage.

Requirements:
- Each triple should contain at least one of the named entities
- If the same subject/object entity appears across multiple chunks, use the SAME name
- Format each triple as [subject, predicate, object]
- Extract meaningful relationships, not obvious/trivial ones

Output JSON format:
{
  "chunks": [
    {
      "chunk_index": 0,
      "triples": [["subject", "predicate", "object"], ...]
    },
    ...
  ]
}
输出纯 JSON，不要用 markdown 代码块包裹。"""

NER_USER_TEMPLATE = """Extract named entities from the following {num_chunks} passages. For each passage, list all named entities with name and description.

Passages:
{passage_text}

Respond with the JSON format specified."""

TRIPLE_USER_TEMPLATE = """Given these passages and their extracted entities, extract RDF triples from each passage.

Passages:
{passage_text}

Entities (for entity grounding):
{entity_json}

Respond with the JSON format specified."""


def _make_llm(model_name, api_key, base_url):
    """Create a per-thread ChatOpenAI instance."""
    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        # DeepSeek v4-flash 默认启用思考模式，显式关闭以避免生成冗长的 reasoning_content
        extra_body={"thinking": {"type": "disabled"}},
    )


def _retry_invoke(llm, messages, max_retries=5):
    """Invoke LLM with retry on transient errors."""
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = min(2 ** attempt, 30)
                print(f"    [OpenIE] LLM error (attempt {attempt+1}), retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"    [OpenIE] LLM failed after {max_retries} attempts: {e}")
                return ""
    return ""


def _process_ner_batch(batch, model_name, api_key, base_url):
    try:
        llm = _make_llm(model_name, api_key, base_url)
        passage_lines = [f"[Passage {j}]\n{text}" for j, (ch, text) in enumerate(batch)]
        passage_text = "\n\n".join(passage_lines)
        prompt = NER_USER_TEMPLATE.format(num_chunks=len(batch), passage_text=passage_text)
        messages = [
            {"role": "system", "content": NER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        content = _retry_invoke(llm, messages)
        if not content:
            return batch, {"chunks": []}
        parsed = _extract_json(content)
        return batch, parsed
    except Exception as e:
        print(f"    [OpenIE] NER batch error: {e}")
        return batch, {"chunks": []}


def _process_triple_batch(batch, ner_results, model_name, api_key, base_url):
    try:
        llm = _make_llm(model_name, api_key, base_url)
        passage_lines = [f"[Passage {j}]\n{text}" for j, (ch, text) in enumerate(batch)]
        entity_lines = [f"[Passage {j}] Entities: {json.dumps(ner_results.get(ch, []), ensure_ascii=False)}" for j, (ch, text) in enumerate(batch)]
        passage_text = "\n\n".join(passage_lines)
        entity_json = "\n\n".join(entity_lines)
        prompt = TRIPLE_USER_TEMPLATE.format(passage_text=passage_text, entity_json=entity_json)
        messages = [
            {"role": "system", "content": TRIPLE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        content = _retry_invoke(llm, messages)
        if not content:
            return batch, {"chunks": []}
        parsed = _extract_json(content)
        return batch, parsed
    except Exception as e:
        print(f"    [OpenIE] Triple batch error: {e}")
        return batch, {"chunks": []}


def _extract_json(text: str) -> dict:
    """Extract JSON object from LLM response (handles markdown fences and minor formatting issues)."""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end+1]
    # Try strict first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Lenient: try fixing common issues
    # Remove trailing commas before ] or }
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    # Fix unquoted strings
    text = re.sub(r'([{,]])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\\1"\\2":', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Last resort: return empty structure
        print(f"    [OpenIE] JSON parse error (falling back to empty): {e}")
        return {"chunks": []}


class OpenIE:
    def __init__(self, llm: ChatOpenAI, max_workers: int = 30):
        self._llm = llm
        self.max_workers = max_workers

    def _get_llm_params(self):
        return (self._llm.model_name, self._llm.openai_api_key.get_secret_value() if hasattr(self._llm.openai_api_key, 'get_secret_value') else str(self._llm.openai_api_key), self._llm.openai_api_base)

    def batch_ner(self, chunks: List[Tuple[str, str]], batch_size: int = 10
                  ) -> Dict[str, List[Dict]]:
        results = {}
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        total_batches = len(batches)
        print(f"  [OpenIE] NER: {total_batches} batches, {self.max_workers} workers")

        model_name, api_key, base_url = self._get_llm_params()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            fut_to_idx = {}
            for i, batch in enumerate(batches):
                fut = executor.submit(_process_ner_batch, batch, model_name, api_key, base_url)
                fut_to_idx[fut] = i

            done = 0
            for future in as_completed(fut_to_idx):
                batch, parsed = future.result()
                try:
                    for item in parsed.get("chunks", []):
                        chunk_idx = item["chunk_index"]
                        chunk_hash = batch[chunk_idx][0]
                        results[chunk_hash] = item.get("entities", [])
                except (KeyError, IndexError, json.JSONDecodeError):
                    for ch, _ in batch:
                        results[ch] = []
                done += 1
                if done % 25 == 0:
                    print(f"    [NER] {done}/{total_batches} batches", flush=True)
        return results

    def batch_triple(self, chunks: List[Tuple[str, str]],
                     ner_results: Dict[str, List[Dict]],
                     batch_size: int = 10) -> Dict[str, List[List[str]]]:
        results = {}
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        total_batches = len(batches)
        print(f"  [OpenIE] Triple: {total_batches} batches, {self.max_workers} workers")

        model_name, api_key, base_url = self._get_llm_params()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            fut_to_idx = {}
            for i, batch in enumerate(batches):
                fut = executor.submit(_process_triple_batch, batch, ner_results, model_name, api_key, base_url)
                fut_to_idx[fut] = i

            done = 0
            for future in as_completed(fut_to_idx):
                batch, parsed = future.result()
                try:
                    for item in parsed.get("chunks", []):
                        chunk_idx = item["chunk_index"]
                        chunk_hash = batch[chunk_idx][0]
                        triples = item.get("triples", [])
                        valid = [t for t in triples if isinstance(t, list) and len(t) == 3]
                        results[chunk_hash] = valid
                except (KeyError, IndexError, json.JSONDecodeError):
                    for ch, _ in batch:
                        results[ch] = []
                done += 1
                if done % 25 == 0:
                    print(f"    [Triple] {done}/{total_batches} batches", flush=True)
        return results

    def batch_process(self, chunks: List[Tuple[str, str]]) -> Tuple[Dict, Dict]:
        print(f"[OpenIE] Starting NER for {len(chunks)} chunks ({self.max_workers} threads)...")
        ner_results = self.batch_ner(chunks)
        print(f"[OpenIE] Starting Triple extraction for {len(chunks)} chunks ({self.max_workers} threads)...")
        triple_results = self.batch_triple(chunks, ner_results)
        return ner_results, triple_results

    def batch_process_passages(
        self,
        passages_text: Dict[str, str],
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, List[List[str]]]]:
        """Passage 级 OpenIE。

        将 passage 包装成 chunks 格式传给 batch_ner/batch_triple，
        但 key 改用 passage_id 对应的文本 hash（与 batch_ner 内部一致）。

        Args:
            passages_text: {passage_id: passage_text}
                其中 passage_text = title + " " + text

        Returns:
            (ner_results, triple_results)
            ner_results:    {passage_id: [{"name":..., "description":...}]}
            triple_results: {passage_id: [["subj","pred","obj"], ...]}
        """
        print(f"[OpenIE] Starting Passage-level NER for {len(passages_text)} passages "
              f"({self.max_workers} threads)...")

        # 排序，保持顺序
        sorted_ids = sorted(passages_text.keys(),
                            key=lambda x: int(x) if x.isdigit() else 0)
        # 构建 chunks: (passage_id, passage_text) - batch_ner 内部用 text MD5 做 key
        passages_list: List[Tuple[str, str]] = [
            (pid, passages_text[pid]) for pid in sorted_ids
        ]

        ner_raw = self.batch_ner(passages_list)
        print(f"[OpenIE] Starting Passage-level Triple extraction for {len(passages_text)} passages "
              f"({self.max_workers} threads)...")
        triple_raw = self.batch_triple(passages_list, ner_raw)

        # batch_ner 内部用 batch[chunk_idx][0]（即 passage_id）做 key
        ner_results = {}
        triple_results = {}
        for pid in sorted_ids:
            ner_results[pid] = ner_raw.get(pid, [])
            triple_results[pid] = triple_raw.get(pid, [])

        return ner_results, triple_results
