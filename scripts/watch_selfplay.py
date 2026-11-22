"""İki ÖĞRENEN ajanın maçını videoya çek: bir self-play koşusundan kovalayan_i vs kaçan_j.

Kullanım (repo kökünden):
  python scripts/watch_selfplay.py --run experiments/sp-empty-lr1e4-s0                  # son sürümler, 4 maç
  python scripts/watch_selfplay.py --run experiments/x --chaser 3 --evader 5 --games 2
  python scripts/watch_selfplay.py --run experiments/x --deterministic                  # gürültüsüz
  python scripts/watch_selfplay.py --run experiments/x --live                           # MuJoCo penceresi

Sürüm numaraları Elo'daki gibi: 0 = başlangıç politikası (config'teki init_*), k = havuza k. eklenen sürüm.
Varsayılan olasılıklı oynatma: self-play ve Elo maçlarındaki gibi (ajanın gerçekten oynadığı oyun).
Arenalar görülmemiş (seed + 999). Videolar --out klasörüne: <koşu>_k<i>_e<j>_mac<g>.mp4
"""
import argparse
import glob
import json
import os
import pickle

import jax
import jax.numpy as jnp

from chase_tag.arenas import make_arena_set
from chase_tag.env import ChaseTagEnv
from chase_tag.play import make_runner, split_games
from chase_tag.ppo import actor_dist
from chase_tag.rules import CHASER, EVADER, MAX_STEPS, OUTCOME_NAMES
from chase_tag.viewer import play_live, save_video

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--chaser", type=int, default=-1, help="kovalayan sürümü (varsayılan: son)")
p.add_argument("--evader", type=int, default=-1, help="kaçan sürümü (varsayılan: son)")
p.add_argument("--games", type=int, default=4)
p.add_argument("--deterministic", action="store_true")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--live", action="store_true")
p.add_argument("--out", default="videos")
a = p.parse_args()

cfg = json.load(open(os.path.join(a.run, "config.json")))
files = {r: [cfg[f"init_{r}"]] + sorted(glob.glob(os.path.join(a.run, f"{r}_gen*.pkl"))) for r in ("chaser", "evader")}
ic = a.chaser % len(files["chaser"])
ie = a.evader % len(files["evader"])
load = lambda f: jax.tree.map(jnp.asarray, pickle.load(open(f, "rb")))
p_ch, p_ev = load(files["chaser"][ic]), load(files["evader"][ie])
squash = cfg.get("squash_mean", False)
print(f"kovalayan_{ic} ({os.path.basename(files['chaser'][ic])}) vs kaçan_{ie} ({os.path.basename(files['evader'][ie])}) | "
      f"faz {cfg['phase']}, engel {tuple(cfg['n_obst'])} | {'gürültüsüz' if a.deterministic else 'olasılıklı'}")

arenas = make_arena_set(200, phase=cfg["phase"], n_obst=tuple(cfg["n_obst"]), seed=cfg["seed"] + 999, verbose=False)
env = ChaseTagEnv(2, arenas, opp_crash_reward=cfg["opp_crash_reward"], full_obs=cfg.get("full_obs", False))


def policy(s, mem, key):
    """İki ağ da kendi (maskeli) gözlemiyle oynar; eğitimdeki gibi olasılıklı ya da gürültüsüz."""
    obs = env.observe(s)
    acts = []
    for params, me, k in ((p_ch, CHASER, 0), (p_ev, EVADER, 1)):
        mu, std = actor_dist(params, obs[:, me], squash)
        acts.append(mu if a.deterministic else mu + std * jax.random.normal(jax.random.fold_in(key, k), mu.shape))
    return jnp.stack(acts, 1), mem


run = make_runner(env, policy, lambda mem, done: mem)
_, (_, _, _, rec) = run(env.reset(jax.random.PRNGKey(a.seed)), None, jax.random.PRNGKey(a.seed + 1),
                        MAX_STEPS * a.games)
games = split_games(rec)[:a.games]
os.makedirs(a.out, exist_ok=True)
name = os.path.basename(os.path.normpath(a.run))
for g, game in enumerate(games):
    print(f"maç {g}: {OUTCOME_NAMES[game[2]]}, {(game[1] - game[0] + 1) / 50:.1f} s")
    if a.live:
        play_live(rec, arenas, game)
    else:
        path = os.path.join(a.out, f"{name}_k{ic}_e{ie}{'_det' if a.deterministic else ''}_mac{g}.mp4")
        print("  →", save_video(rec, arenas, game, path))
