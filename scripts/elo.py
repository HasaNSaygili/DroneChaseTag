"""Bir self-play koşusunun bütün sürümleriyle Elo turnuvası (round-robin): her kovalayan × her kaçan.

Kullanım:
  python scripts/elo.py --run experiments/sp-empty-fix2-s0                 # çift başına 256 maç
  python scripts/elo.py --run experiments/x --games 64 --deterministic     # hızlı / gürültüsüz

Oyuncular: kovalayan_0 / kaçan_0 = koşunun başlangıç politikaları (config'teki init_*), kovalayan_k / kaçan_k =
havuza k. sırada eklenen sürüm (<rol>_genNN.pkl dosyaları, nesil sırasıyla). Arenalar eğitimdekinden farklı
seed'le üretilir (seed + 999, watch_policy ile aynı). Çıktı: <run>/elo/ altında elo.json (sayılar, makale için)
ve elo.png (kazanma matrisi + nesillere göre puan).
"""
import argparse
import glob
import json
import os
import pickle
import time

import jax
import numpy as np

from chase_tag.arenas import make_arena_set
from chase_tag.env import ChaseTagEnv
from chase_tag.eval import bradley_terry, round_robin, win_matrix
from chase_tag.rules import OUTCOME_NAMES

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--games", type=int, default=256, help="çift başına maç")
p.add_argument("--deterministic", action="store_true", help="gürültüsüz oynat (varsayılan: eğitimdeki gibi olasılıklı)")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

cfg = json.load(open(os.path.join(a.run, "config.json")))


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


players = {}
for role, tr in (("chaser", "kovalayan"), ("evader", "kaçan")):
    files = [cfg[f"init_{role}"]] + sorted(glob.glob(os.path.join(a.run, f"{role}_gen*.pkl")))
    players[role] = [(f"{tr}_{k}", load(f)) for k, f in enumerate(files)]
names_c = [n for n, _ in players["chaser"]]
names_e = [n for n, _ in players["evader"]]
print(f"{len(names_c)} kovalayan × {len(names_e)} kaçan, çift başına {a.games} maç, "
      f"{'gürültüsüz' if a.deterministic else 'olasılıklı'}")

arena_seed = cfg["seed"] + 999
arenas = make_arena_set(200, phase=cfg["phase"], n_obst=tuple(cfg["n_obst"]), seed=arena_seed, verbose=False)
n_blocks = len(names_e)                       # aynı anda bir kovalayanın bütün rakipleri kadar çift
env = ChaseTagEnv(n_blocks * a.games, arenas, opp_crash_reward=cfg["opp_crash_reward"],
                  full_obs=cfg.get("full_obs", False))
t0 = time.perf_counter()
C = round_robin(env, [q for _, q in players["chaser"]], [q for _, q in players["evader"]], a.games,
                jax.random.PRNGKey(a.seed), squash=cfg.get("squash_mean", False), deterministic=a.deterministic)
print(f"turnuva: {time.perf_counter() - t0:.0f} s")

wins, games = win_matrix(C, env.reward_table)
W = wins / np.maximum(games, 1)                 # kovalayanın puanı (kazanma oranı)
elo_c, elo_e = bradley_terry(wins, games)

print("\nPuanlar (Elo ölçeği, ortalama 1000; 200 puan fark ≈ %76 kazanma):")
for k in range(max(len(names_c), len(names_e))):
    c = f"{names_c[k]:13s} {elo_c[k]:6.0f}" if k < len(names_c) else " " * 20
    e = f"{names_e[k]:9s} {elo_e[k]:6.0f}" if k < len(names_e) else ""
    print(f"  {c}    {e}")
print(f"\nKovalayanın ortalama puanı (bütün çiftler): %{100 * wins.sum() / games.sum():.1f}")

out = os.path.join(a.run, "elo" + ("_det" if a.deterministic else ""))
os.makedirs(out, exist_ok=True)
res = dict(games_per_pair=a.games, deterministic=a.deterministic, arena_seed=arena_seed, seed=a.seed,
           chasers=names_c, evaders=names_e, elo_chaser=elo_c.round(1).tolist(), elo_evader=elo_e.round(1).tolist(),
           chaser_score=W.round(4).tolist(), outcome_names=OUTCOME_NAMES, outcome_counts=C.tolist())
with open(os.path.join(out, "elo.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)

# ---- Grafik: kazanma matrisi + nesillere göre puan ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
im = ax[0].imshow(W, cmap="RdBu_r", vmin=0, vmax=1)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        ax[0].text(j, i, f"{100 * W[i, j]:.0f}", ha="center", va="center", fontsize=7)
ax[0].set_xticks(range(len(names_e)), [n.split("_")[1] for n in names_e])
ax[0].set_yticks(range(len(names_c)), [n.split("_")[1] for n in names_c])
ax[0].set_xlabel("kaçan sürümü"); ax[0].set_ylabel("kovalayan sürümü")
ax[0].set_title("kovalayanın kazanma oranı (%)\nkırmızı: kovalayan kazanır, mavi: kaçan kazanır")
fig.colorbar(im, ax=ax[0], fraction=0.046)
ax[1].plot(range(len(elo_c)), elo_c, "o-", label="kovalayan")
ax[1].plot(range(len(elo_e)), elo_e, "s-", label="kaçan")
ax[1].set_xlabel("sürüm (havuza eklenme sırası)"); ax[1].set_ylabel("Elo"); ax[1].legend()
ax[1].set_title("Bradley–Terry puanı\n(yükselen eğri = rakip tarafın BÜTÜN sürümlerine karşı daha iyi)")
fig.suptitle(os.path.basename(os.path.normpath(a.run)))
fig.savefig(os.path.join(out, "elo.png"), dpi=120, bbox_inches="tight")
print(f"\nkaydedildi: {out}/elo.json, elo.png")
