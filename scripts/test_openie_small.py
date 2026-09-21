#!/usr/bin/env python3
"""
快速测试 OpenIE 流程是否走得通（只取 3 个 passage）。
用法: DEEPSEEK_API_KEY='***' python3 scripts/test_openie_small.py
"""
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from langchain_openai import ChatOpenAI
from infra.openie import OpenIE

# 读取前 3 个 passage
corpus = json.load(open("data/hotpotqa_corpus.json"))
import itertools
titles = list(corpus.keys())[:3]
passages_text = {}
for i, t in enumerate(titles):
    paragraphs = corpus[t]
    if isinstance(paragraphs, list):
        text = t + " " + " ".join(paragraphs)
    else:
        text = t + " " + str(paragraphs)
    passages_text[str(i)] = text
print(f"Passages: {len(passages_text)}")
for pid, text in passages_text.items():
    print(f"  [{pid}] {text[:80]}...")

# 创建 LLM（关闭思考模式）
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com/v1",
    temperature=0.0,
    extra_body={"thinking": {"type": "disabled"}},
)

openie = OpenIE(llm, max_workers=3)
print(f"\n[Test] 开始 NER...")
ner, triple = openie.batch_process_passages(passages_text)

print(f"\n[Test] NER 结果:")
for pid, entities in ner.items():
    print(f"  [{pid}]: {len(entities)} entities")
    for e in entities[:2]:
        print(f"    - {e['name']}: {e['description']}")

print(f"\n[Test] Triple 结果:")
for pid, triples in triple.items():
    print(f"  [{pid}]: {len(triples)} triples")
    for t in triples[:2]:
        print(f"    - {t}")

print(f"\n✅ 测试完成！OpenIE 流程正常。")
