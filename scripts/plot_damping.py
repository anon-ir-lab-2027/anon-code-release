#!/usr/bin/env python3
import json, os, re, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

grid_dir = "outputs/grid_search"
os.makedirs(grid_dir, exist_ok=True)

datasets = ['musique', 'hotpotqa', '2wikimultihopqa']
ds_labels = {'musique':'Musique', 'hotpotqa':'HotpotQA', '2wikimultihopqa':'2WikiMultiHopQA'}

def normalize(t):
    return ' '.join(t.split())

all_data = {}
for ds in datasets:
    with open(f'data/{ds}.json') as f:
        samples = json.load(f)[:200]
    
    gold_list = []
    for s in samples:
        if 'paragraphs' in s:
            paras = s['paragraphs']
            results = []
            for p in paras:
                if p.get('is_supporting', False):
                    full = p['title'] + '\n' + (p.get('text', p.get('paragraph_text', '')))
                    results.append(normalize(full))
        elif 'supporting_facts' in s:
            gold_titles = set(sf[0] for sf in s.get('supporting_facts', []))
            results = []
            for title, paras in s.get('context', []):
                if title in gold_titles:
                    # 与 pipeline.py 官方口径一致：hotpotqa 段落拼接不加空格，2wiki 加空格
                    full = title + '\n' + (''.join(paras) if ds == 'hotpotqa' else ' '.join(paras))
                    results.append(normalize(full))
        else:
            results = []
        gold_list.append(list(set(results)))
    
    bd = f'outputs/flat/local_embedding/{ds}'
    files = glob.glob(os.path.join(bd, 'retrieval_MCS_RerankPath_d3w8p*_200.json'))
    
    collected = {}
    for fp in files:
        fname = os.path.basename(fp)
        m = re.search(r'_d3w8p(.+?)_200\.json$', fname)
        if not m:
            continue
        raw = m.group(1)
        p = float(raw.replace('_', '.'))
        p_tenth = round(p * 10)
        if p_tenth < 1 or p_tenth > 9:
            continue
        
        with open(fp) as f:
            ret = json.load(f)
        
        total = 0
        hits = {k:0 for k in [1,5,10,20]}
        for entries, si in zip(ret, range(200)):
            gs = set(gold_list[si])
            total += len(gs)
            for k in hits:
                ps = set(normalize(ft) for _, ft in entries[:k])
                hits[k] += len(gs & ps)
        
        collected[p_tenth] = {f'R@{k}': hits[k]/total for k in [1,5,10,20]}
    
    all_data[ds] = collected
    print(f'{ds}: ok')

# ── 画图 ──
colors = {'R@1':'#E74C3C', 'R@5':'#F39C12', 'R@10':'#27AE60', 'R@20':'#2E86C1'}
metrics = ['R@1','R@5','R@10','R@20']
markers = {'R@1':'o', 'R@5':'s', 'R@10':'^', 'R@20':'D'}

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
for idx, ds in enumerate(datasets):
    ax = axes[idx]
    cd = all_data[ds]

    kk = sorted(cd.keys())
    xs = [k/10.0 for k in kk]
    for m in metrics:
        ys = [cd[k][m] for k in kk]
        ax.plot(xs, ys, marker=markers[m], color=colors[m], label=m, linewidth=2.5, markersize=7)
    ax.set_xlabel('PPR Damping', fontsize=13)
    ax.set_ylabel('Recall', fontsize=13)
    ax.set_title(ds_labels[ds], fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.arange(0.1, 1.0, 0.1))
    ax.set_xlim(0.05, 0.95)

plt.suptitle('Recall vs PPR Damping (fixed depth=3, width=8)', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(grid_dir, 'damping_scan_detailed.png'), dpi=200, bbox_inches='tight')
print(f'✅ 图已保存: {grid_dir}/damping_scan_detailed.png')

# ── 打印表格 ──
print()
print(f'{"="*130}')
print(f'{"damping":<8}  {"R@1":<8} {"R@5":<8} {"R@10":<8} {"R@20":<8}  '
      f'{"R@1":<8} {"R@5":<8} {"R@10":<8} {"R@20":<8}  '
      f'{"R@1":<8} {"R@5":<8} {"R@10":<8} {"R@20":<8}')
print(f'{"─"*130}')

for ti in range(1, 10):
    p = ti / 10.0
    line = f'{p:<8.1f}  '
    for ds in datasets:
        d = all_data[ds][ti]
        line += f'{d["R@1"]:<8.4f} {d["R@5"]:<8.4f} {d["R@10"]:<8.4f} {d["R@20"]:<8.4f}  '
    print(line)

# ── 变化幅度 ──
print(f'{"─"*130}')
print("变化幅度 (max-min):")
for ds in datasets:
    d = all_data[ds]
    v = {m: [d[k][m] for k in d] for m in metrics}
    print(f'  {ds_labels[ds]:<20}: '
          f'ΔR@1={max(v["R@1"])-min(v["R@1"]):.4f}, '
          f'ΔR@5={max(v["R@5"])-min(v["R@5"]):.4f}, '
          f'ΔR@10={max(v["R@10"])-min(v["R@10"]):.4f}')

# ── 最佳 ──
print(f'{"─"*130}')
print("各数据集最佳 damping (by R@10):")
for ds in datasets:
    d = all_data[ds]
    best = max(d.keys(), key=lambda k: d[k]["R@10"])
    print(f'  {ds_labels[ds]:<20}: damping={best/10.0:.1f}, R@10={d[best]["R@10"]:.4f}, R@1={d[best]["R@1"]:.4f}')
