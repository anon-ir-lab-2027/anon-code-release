#!/usr/bin/env python3
"""给结果文件补充 final_recall_10 / final_recall_all 字段。

用法：
    uv run python3 scripts/fill_recall.py <结果JSON>
"""
import json
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def normalize_text(t):
    return " ".join(t.split())


def compute_recall(result, samples_by_id):
    sample = samples_by_id.get(result.get("id"))
    if not sample:
        return None, None

    # ── gold passage fulltexts ──
    gold_fulltexts = set()
    # hotpotqa: context + supporting_facts
    if "context" in sample and "supporting_facts" in sample:
        gold_titles = set(sf[0] for sf in sample.get("supporting_facts", []))
        for title, paras in sample.get("context", []):
            if title in gold_titles:
                full = title + "\n" + " ".join(paras)
                gold_fulltexts.add(normalize_text(full))
    n_gold = len(gold_fulltexts)
    if n_gold == 0:
        return None, None

    # ── predicted passage fulltexts ──
    per_step_passages = result.get("per_step_passages_text", [])
    ordered_keys = []
    seen_titles = set()
    for step in per_step_passages:
        for p in step:
            title = p.get("title", "")
            if title not in seen_titles:
                seen_titles.add(title)
                ordered_keys.append(normalize_text(title + "\n" + p.get("text", "")))

    if not ordered_keys:
        return None, None

    final_recall_all = sum(1 for ft in ordered_keys if ft in gold_fulltexts) / n_gold
    final_recall_10 = sum(1 for ft in ordered_keys[:10] if ft in gold_fulltexts) / n_gold

    return final_recall_10, final_recall_all


def main():
    result_path = sys.argv[1] if len(sys.argv) > 1 else "exp/ircot_lg_results_deepseek-v3-0324_hotpotqa_steps3_llm-judge.json"
    # 推测 dataset 从文件名
    path = result_path
    dataset_name = "hotpotqa"
    for d in ["hotpotqa", "2wikimultihopqa", "musique"]:
        if d in path:
            dataset_name = d
            break

    print(f"[FillRecall] 加载结果文件: {path}")
    print(f"[FillRecall] 推测 dataset: {dataset_name}")

    with open(path) as f:
        data = json.load(f)

    # 加载 samples by id
    samples_path = os.path.join(os.path.dirname(__file__), "..", "data", f"{dataset_name}.json")
    with open(samples_path) as f:
        all_samples = json.load(f)
    samples_by_id = {}
    for s in all_samples:
        sid = s.get("id", s.get("_id", ""))
        samples_by_id[sid] = s
    print(f"[FillRecall] 加载 {len(samples_by_id)} 个样本")

    results = data["results"]
    filled = 0
    skipped = 0
    err = 0

    for r in results:
        if "final_recall_10" in r and "final_recall_all" in r:
            skipped += 1
            continue
        try:
            recall10, recall_all = compute_recall(r, samples_by_id)
            if recall10 is not None:
                r["final_recall_10"] = recall10
                r["final_recall_all"] = recall_all
                filled += 1
            else:
                # 没有样本 ID 或没有 gold passages
                r["final_recall_10"] = 0.0
                r["final_recall_all"] = 0.0
                filled += 1
        except Exception as e:
            r["final_recall_10"] = 0.0
            r["final_recall_all"] = 0.0
            filled += 1
            err += 1

    total = len(results)
    avg_recall10 = sum(r.get("final_recall_10", 0.0) for r in results) / max(total, 1)
    avg_recall_all = sum(r.get("final_recall_all", 0.0) for r in results) / max(total, 1)

    print(f"[FillRecall] 总计 {total} 条")
    print(f"[FillRecall] 已填充 {filled} 条 | 已有跳过 {skipped} | 异常 {err}")
    print(f"[FillRecall] 平均 Recall@10: {avg_recall10:.1%}")
    print(f"[FillRecall] 平均 Recall@all: {avg_recall_all:.1%}")

    # 回写文件
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[FillRecall] 已保存到 {path}")


if __name__ == "__main__":
    main()
