"""Eğitim eğrileri (gerçek metrics.jsonl'den): sonuç oranları, M adıma göre. Ara rapor sunumunda kullanıldı.

Kullanım:  python scripts/plot_curves.py experiments <çıktı klasörü>
Renkler dataviz doğrulayıcısından geçti (açık renkler soluk: kesik çizgi + doğrudan etiketle destekli).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

exp, out = sys.argv[1], sys.argv[2]
INK, SOFT, GRID, BG = "#15202B", "#6B757D", "#E4DFD4", "#FBFAF7"
# Sonuç → (etiket, renk, çizgi). Kovalayan kazanır: turuncu tonları; kaçan kazanır: mavi tonları.
SERIES = [(1, "ebeleme", "#B8401B", "-"), (3, "kaçan çarptı", "#E0874F", "--"),
          (2, "süre doldu", "#2457B3", "-"), (4, "kovalayan çarptı", "#5E92EC", "--")]
K = 5


def load(run):
    rows = [json.loads(l) for l in open(os.path.join(exp, run, "metrics.jsonl"), encoding="utf-8")]
    steps = np.array([r["steps"] for r in rows]) / 1e6
    C = np.array([r["outcome_counts"] for r in rows])
    frac = C / C[:, 1:].sum(1, keepdims=True).clip(1)
    sm = lambda y: np.convolve(y, np.ones(K) / K, "valid")
    return steps[K // 2: len(steps) - K // 2], {k: sm(frac[:, k]) * 100 for k, *_ in SERIES}


def panel(ax, run, title, label_at=None):
    x, ys = load(run)
    for k, name, col, ls in SERIES:
        ax.plot(x, ys[k], color=col, lw=2.4, ls=ls, solid_capstyle="round", label=name)
    ax.set_xlim(0, x[-1]); ax.set_ylim(-2, 104)
    ax.set_yticks([0, 50, 100]); ax.set_yticklabels(["%0", "%50", "%100"])
    ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#B9B2A4")
    ax.tick_params(colors=SOFT, labelsize=15, length=0)
    ax.set_facecolor(BG)
    ax.set_title(title, loc="left", fontsize=17, color=INK, pad=8, fontweight="bold")
    # Seçici doğrudan etiketler (metin rengi mürekkep, çizgi rengi değil)
    for k, xpos, txt, dy in (label_at or []):
        i = int(np.argmin(np.abs(x - xpos)))
        ax.annotate(txt, (x[i], ys[k][i]), xytext=(6, dy), textcoords="offset points", fontsize=15, color=INK)


# ---- Kovalayan: iki deneme alt alta ----
fig, axes = plt.subplots(2, 1, figsize=(7.88, 5.6), dpi=200, sharex=True, facecolor=BG)
panel(axes[0], "ppo-chaser-empty-s0", "Deneme 1 · WCT kuralı",
      [(3, 300, "kaçan çarptı (yere düştü)", -20), (1, 300, "ebeleme", 6)])
panel(axes[1], "ppo-chaser-empty-nocrashrew-s0", "Deneme 2 · rakibi çarptırmak ödülsüz",
      [(1, 330, "ebeleme", -22), (2, 40, "süre doldu", -4)])
axes[1].set_xlabel("eğitim adımı (milyon)", fontsize=15, color=SOFT)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=13, frameon=False,
           bbox_to_anchor=(0.5, 1.0), handlelength=2.2, labelcolor=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(out, "curves_chaser.png"), facecolor=BG)

# ---- Kaçan ----
fig, ax = plt.subplots(figsize=(7.88, 5.6), dpi=200, facecolor=BG)
panel(ax, "ppo-evader-empty-nocrashrew-s0", "Kaçan · elle yazılmış kovalayana karşı",
      [(1, 20, "ebeleme", -22), (4, 122, "kovalayan çarptı", 4), (2, 330, "süre doldu", -24)])
ax.set_xlabel("eğitim adımı (milyon)", fontsize=15, color=SOFT)
fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=13, frameon=False,
           bbox_to_anchor=(0.5, 1.0), handlelength=2.2, labelcolor=INK)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(out, "curves_evader.png"), facecolor=BG)
print("ok")
