#!/usr/bin/env python3
"""参数扫描的官方指标汇总 + 出图脚本。

从 retrieval JSON 直接计算 Recall@k（与 scripts/pipeline.py 的官方 gold 口径一致：
hotpotqa 段落拼接用 ''.join，2wiki 用 ' '.join，musique 用 is_supporting 字段），
生成每数据集 CSV 与三面板超参图（depth / width / damping）。

用法：
    uv run python3 scripts/param_sweep_report.py
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs", "grid_search")
DATASETS = ["2wikimultihopqa", "musique", "hotpotqa"]
DS_LABEL = {"2wikimultihopqa": "2WikiMultihopQA", "musique": "Musique", "hotpotqa": "HotpotQA"}
DS_COLOR = {"2wikimultihopqa": "#2E86C1", "musique": "#27AE60", "hotpotqa": "#E74C3C"}
TOP_KS = [1, 5, 10, 20]


def normalize_text(t):
    return " ".join(t.split())


def build_gold(dataset):
    """与 pipeline.py get_gold_fulltexts 完全一致的 gold 口径。"""
    samples = json.load(open(os.path.join(DATA_DIR, f"{dataset}.json")))  # 仅取前 200 在调用处做
    gold = []
    for s in samples[:200]:
        if "supporting_facts" in s:  # hotpotqa, 2wikimultihopqa
            gold_titles = set(sf[0] for sf in s.get("supporting_facts", []))
            res = []
            for title, paras in s.get("context", []):
                if title in gold_titles:
                    if dataset == "hotpotqa":
                        full = title + "\n" + "".join(paras)
                    else:
                        full = title + "\n" + " ".join(paras)
                    res.append(normalize_text(full))
        elif "paragraphs" in s:  # musique
            res = []
            for p in s.get("paragraphs", []):
                if p.get("is_supporting", False):
                    full = p["title"] + "\n" + p.get("text", p.get("paragraph_text", ""))
                    res.append(normalize_text(full))
        else:
            res = []
        gold.append(list(set(res)))
    return gold


def recall_at_k(fp, gold, k):
    ret = json.load(open(fp))
    hits = 0
    total = 0
    for entries, si in zip(ret, range(len(gold))):
        gs = set(gold[si])
        total += len(gs)
        preds = set(normalize_text(ft) for _, ft in entries[:k])
        hits += len(gs & preds)
    return hits / total if total > 0 else 0.0


def collect(dataset, gold):
    bd = os.path.join(ROOT, "outputs", "flat", "local_embedding", dataset)
    families = {
        "depth":   (("d{}w8p0_4", [1, 2, 3, 5, 10]), "MCS Depth", "1,2,3,5,10"),
        "width":   (("d3w{}p0_4", [1, 2, 4, 8, 16, 32]), "MCS Width", "1,2,4,8,16,32"),
        "damping": (("d3w8p{}", ["0_1", "0_2", "0_3", "0_4", "0_5", "0_6", "0_7", "0_8", "0_9"]), "PPR Damping", "0.1..0.9"),
    }
    rows = []  # (family, x, {k: r})
    for fam, ((pattern, xs), _xlab, _note) in families.items():
        for xraw in xs:
            fp = os.path.join(bd, f"retrieval_MCS_RerankPath_{pattern.format(xraw)}_200.json")
            if not os.path.exists(fp):
                print(f"  [warn] missing {os.path.basename(fp)}")
                continue
            # damping 的 x 统一为浮点数（0.1..0.9），depth/width 保持整数
            x = float(xraw.replace("_", ".")) if fam == "damping" else xraw
            rows.append((fam, x, {k: recall_at_k(fp, gold, k) for k in TOP_KS}))
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_data = {}
    for ds in DATASETS:
        gold = build_gold(ds)
        rows = collect(ds, gold)
        all_data[ds] = rows
        # CSV
        csv_path = os.path.join(OUT_DIR, f"param_sweep_official_{ds}.csv")
        with open(csv_path, "w") as f:
            f.write("family,param,R@1,R@5,R@10,R@20\n")
            for fam, x, m in rows:
                f.write(f"{fam},{x},{m[1]:.4f},{m[5]:.4f},{m[10]:.4f},{m[20]:.4f}\n")
        print(f"CSV: {csv_path}")
        # 摘要
        for fam in ["depth", "width", "damping"]:
            vals = [m[10] for f_, x, m in rows if f_ == fam]
            if vals:
                print(f"  {DS_LABEL[ds]:<14} {fam:<8} ΔR@10 = {max(vals)-min(vals):.4f}")
        damp = [(x, m[10]) for f_, x, m in rows if f_ == "damping"]
        if damp:
            best_x, best_v = max(damp, key=lambda t: t[1])
            print(f"  {DS_LABEL[ds]:<14} best damping = {best_x} (R@10={best_v:.4f})")

    # 图：3 面板
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=200)
    panels = [
        (([1, 2, 3, 5, 10], "depth"), "MCS Depth", "(a) Depth"),
        (([1, 2, 4, 8, 16, 32], "width"), "MCS Width", "(b) Width"),
        (([i / 10 for i in range(1, 10)], "damping"), "PPR Damping", "(c) Damping"),
    ]
    for ax, ((xs, fam), xlab, title) in zip(axes, panels):
        for ds in DATASETS:
            by_x = {x: m[10] for f_, x, m in all_data[ds] if f_ == fam}
            pairs = [(x, by_x[x]) for x in xs if x in by_x]
            if pairs:
                xs_plot, ys = zip(*pairs)
                # plot in percent to match the paper's R@k convention (CSVs stay fractional)
                ax.plot(xs_plot, [100.0 * v for v in ys], marker="o", markersize=5, linewidth=2,
                        color=DS_COLOR[ds], label=DS_LABEL[ds])
        ax.set_xlabel(xlab, fontsize=12)
        ax.set_ylabel("Recall@10 (%)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xticks(xs)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "hyperparam_sensitivity.pdf")
    fig.savefig(fig_path)
    fig.savefig(os.path.join(OUT_DIR, "hyperparam_sensitivity.png"))
    print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
