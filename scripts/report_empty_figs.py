"""Boş arena raporu için grafikler: hepsi experiments/ altındaki koşu dosyalarından çizilir.

Kullanım (repo kökünden; sadece numpy + matplotlib gerekir):
  python scripts/report_empty_figs.py experiments <grafik_klasörü>
  python scripts/report_empty_pdf.py experiments <grafik_klasörü> docs/bos_arena_raporu.pdf
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = sys.argv[1]          # experiments klasörü
OUT = sys.argv[2]          # grafiklerin yazılacağı klasör
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
C_CH, C_EV = "#c0392b", "#2471a3"      # kovalayan kırmızı, kaçan mavi (Elo matrisiyle aynı)
O_COL = {"ebeleme": "#c0392b", "süre doldu": "#2471a3", "kaçan çarptı": "#e67e22", "kovalayan çarptı": "#7d3c98",
         "ikisi çarptı": "#7f8c8d"}


def rows(run):
    return [json.loads(l) for l in open(os.path.join(EXP, run, "metrics.jsonl"), encoding="utf-8")]


def frac(r, k):
    oc = np.asarray(r["outcome_counts"], float)
    return oc[k] / max(oc[1:].sum(), 1)


def smooth(y, k=5):
    return np.convolve(y, np.ones(k) / k, "valid")


def steps_of(r):
    return r.get("steps", r.get("total_steps")) / 1e6


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


# ---- Şekil 1: temel ajanların eğitim eğrisi, eski kodla kıyas ----
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for run, k, col, lab in [("ppo-chaser-empty-fix-s0", 1, C_CH, "kovalayan (yeni kod)"),
                         ("ppo-chaser-empty-nocrashrew-s0", 1, C_CH, "kovalayan (eski kod)"),
                         ("ppo-evader-empty-fix-s0", 2, C_EV, "kaçan (yeni kod)"),
                         ("ppo-evader-empty-nocrashrew-s0", 2, C_EV, "kaçan (eski kod)")]:
    R = rows(run)
    x = np.array([steps_of(r) for r in R])
    ls = "-" if "fix" in run else "--"
    ax[0].plot(x[2:-2], smooth([frac(r, k) for r in R]), ls, color=col, label=lab, lw=1.4)
    ax[1].plot(x, [r["ep_len"] / 50 for r in R], ls, color=col, lw=1.2)
ax[0].set_title("kazanma oranı (eğitim, gürültülü politika)"); ax[0].set_xlabel("milyon adım"); ax[0].set_ylim(0, 1.02)
ax[0].legend(fontsize=7, loc="lower right")
ax[1].set_title("ortalama maç süresi (s)"); ax[1].set_xlabel("milyon adım")
save(fig, "fig1_temel.png")

# ---- Şekil 2: eski self-play'lerde kaçanın keşfi (std) ----
fig, ax = plt.subplots(figsize=(6.5, 2.8))
for run, col, lab in [("sp-empty-s0", "#7f8c8d", "sp-empty-s0 (entropi 0, eski kod)"),
                      ("sp-empty-ent-s0", "#e67e22", "sp-empty-ent-s0 (entropi 0,003, eski kod)"),
                      ("sp-empty-fix2-s0", C_EV, "sp-empty-fix2-s0 (entropi 0,001, düzeltilmiş kod)")]:
    R = [r for r in rows(run) if r["learner"] == "evader"]
    ax.plot([r["total_steps"] / 1e6 for r in R], [r["std"] for r in R], ".", ms=1.5, color=col, label=lab)
ax.set_title("kaçan eğitilirken eylem gürültüsü (std): düşükse keşif bitmiş demek")
ax.set_xlabel("toplam milyon adım (sadece kaçanın nesilleri)"); ax.set_ylabel("std"); ax.set_ylim(0, 0.25)
ax.legend(fontsize=7, markerscale=6)
save(fig, "fig2_std.png")

# ---- Şekil 3: ölçüm hatası (sp-empty-fix-s0, nesil 19) ----
R = [r for r in rows("sp-empty-fix-s0") if r["gen"] == 19]
bc = np.array([r["block_counts"] for r in R])[:, :4]          # en son rakiple oynayan 4 blok
tags, tos = bc[..., 1].sum(1), bc[..., 2].sum(1)
wins = bc[..., [2, 4]].sum((1, 2)); ended = bc[..., 1:].sum((1, 2))
win5 = [wins[max(0, i - 4):i + 1].sum() / max(ended[max(0, i - 4):i + 1].sum(), 1) for i in range(len(R))]
u = np.arange(1, len(R) + 1)
fig, ax = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
ax[0].bar(u, tags, color=O_COL["ebeleme"], label="yakalandı (kovalayan kazandı)")
ax[0].bar(u, tos, bottom=tags, color=O_COL["süre doldu"], label="süre doldu (kaçan kazandı)")
ax[0].set_yscale("log"); ax[0].set_ylabel("o turda biten maç\n(log ölçek)")
ax[0].set_title("sp-empty-fix-s0, nesil 19 (kaçan öğreniyor): her turda biten maçlar")
ax[0].legend(fontsize=7, loc="upper left")
ax[1].plot(u, win5, "o-", color=C_EV, ms=3, label="logdaki ölçü: son 5 turda kaçanın kazanma oranı")
ax[1].axhline(0.6, color="k", lw=0.8, ls=":", label="sıra değişimi eşiği %60")
post = (wins[-3:].sum() / ended[-3:].sum())
ax[1].axhline(post, color="#27ae60", lw=1.2, ls="--", label=f"patlama sonrası turlar: %{100 * post:.0f}")
ax[1].set_xlabel("nesil içindeki tur (1 tur = 64 adım = 1,28 s, 16.384 dünya)"); ax[1].set_ylim(0, 1)
ax[1].set_xticks(u[::2])
ax[1].legend(fontsize=7, loc="upper left")
save(fig, "fig3_olcum.png")

# ---- Şekil 4: temiz self-play, kontrol maçları + nesil uzunlukları ----
R = rows("sp-empty-fix2-s0")
fig, ax = plt.subplots(2, 1, figsize=(8, 5), gridspec_kw={"height_ratios": [2, 1]})
gens = sorted(set(r["gen"] for r in R))
for g in gens:
    G = [r for r in R if r["gen"] == g]
    ev = [r for r in G if r["win_latest_eval"] is not None]
    col = C_CH if G[0]["learner"] == "chaser" else C_EV
    ax[0].plot([r["total_steps"] / 1e6 for r in ev], [r["win_latest_eval"] for r in ev], "o-", color=col, ms=3, lw=1)
    ax[0].axvline(G[-1]["total_steps"] / 1e6, color="#bbb", lw=0.5)
ax[0].axhline(0.6, color="k", lw=0.8, ls=":")
ax[0].plot([], [], "o-", color=C_CH, label="kovalayan öğreniyor"); ax[0].plot([], [], "o-", color=C_EV, label="kaçan öğreniyor")
ax[0].set_ylim(0, 1.02); ax[0].set_xlabel("toplam milyon adım")
ax[0].set_title("kontrol maçları: öğrenenin EN SON rakibe karşı kazanma oranı (4096 maç, baştan sona)")
ax[0].legend(fontsize=7, loc="lower right")
gs = [[r for r in R if r["gen"] == g][-1] for g in gens]
ax[1].bar(gens, [r["gen_steps"] / 1e6 for r in gs], color=[C_CH if r["learner"] == "chaser" else C_EV for r in gs])
ax[1].axhline(20, color="k", lw=0.6, ls=":"); ax[1].axhline(100, color="k", lw=0.6, ls=":")
ax[1].set_xticks(gens); ax[1].set_xlabel("nesil"); ax[1].set_ylabel("M adım")
ax[1].set_title("nesil uzunluğu (en az 20 M, en fazla 100 M)")
fig.tight_layout()
save(fig, "fig4_selfplay.png")

# ---- Şekil 5: kontrol maçı sonuç dağılımı + std, nesil başına ----
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
last_ev = [[r for r in R if r["gen"] == g and r["win_latest_eval"] is not None][-1] for g in gens]
bottom = np.zeros(len(gens))
for name in ["ebeleme", "kaçan çarptı", "süre doldu", "kovalayan çarptı", "ikisi çarptı"]:
    v = np.array([r["eval_pct"][name] for r in last_ev])
    ax[0].bar(gens, v, bottom=bottom, color=O_COL[name], label=name)
    bottom += v
ax[0].set_xticks(gens[::2]); ax[0].set_xlabel("nesil (çift: kovalayan, tek: kaçan öğrendi)"); ax[0].set_ylabel("%")
ax[0].set_title("nesil sonu kontrol maçının sonuçları")
ax[0].legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5, frameon=False)
for role, col, lab in [("chaser", C_CH, "kovalayan"), ("evader", C_EV, "kaçan")]:
    g_r = [g for g, r in zip(gens, gs) if r["learner"] == role]
    ax[1].plot(g_r, [r["std"] for r in gs if r["learner"] == role], "o-", color=col, ms=3, label=lab)
ax[1].set_ylim(0, 0.25); ax[1].set_xlabel("nesil"); ax[1].set_ylabel("std")
ax[1].set_title("öğrenenin eylem gürültüsü (nesil sonu)"); ax[1].legend(fontsize=7)
save(fig, "fig5_sonuclar.png")

# ---- Şekil 6: Elo matrisi + puanlar ----
E = json.load(open(os.path.join(EXP, "sp-empty-fix2-s0", "elo", "elo.json"), encoding="utf-8"))
D = json.load(open(os.path.join(EXP, "sp-empty-fix2-s0", "elo_det", "elo.json"), encoding="utf-8"))
W = np.array(E["chaser_score"])
fig, ax = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.2, 1]})
im = ax[0].imshow(W, cmap="RdBu_r", vmin=0, vmax=1)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        ax[0].text(j, i, f"{100 * W[i, j]:.0f}", ha="center", va="center", fontsize=7.5)
ax[0].set_xticks(range(W.shape[1])); ax[0].set_yticks(range(W.shape[0]))
ax[0].set_xlabel("kaçan sürümü"); ax[0].set_ylabel("kovalayan sürümü")
ax[0].set_title("kovalayanın kazanma oranı (%), olasılıklı\nkırmızı: kovalayan, mavi: kaçan kazanır")
fig.colorbar(im, ax=ax[0], fraction=0.046)
v = np.arange(len(E["elo_chaser"]))
ax[1].plot(v, E["elo_chaser"], "o-", color=C_CH, label="kovalayan, olasılıklı")
ax[1].plot(v, D["elo_chaser"], "o--", color=C_CH, alpha=0.5, label="kovalayan, gürültüsüz")
ax[1].plot(v, E["elo_evader"], "s-", color=C_EV, label="kaçan, olasılıklı")
ax[1].plot(v, D["elo_evader"], "s--", color=C_EV, alpha=0.5, label="kaçan, gürültüsüz")
ax[1].set_xlabel("sürüm (havuza eklenme sırası)"); ax[1].set_ylabel("Elo (ortalama 1000)"); ax[1].legend(fontsize=7)
ax[1].set_title("Bradley–Terry puanı")
fig.tight_layout()
save(fig, "fig6_elo.png")
print("grafikler hazır:", sorted(os.listdir(OUT)))
