"""Unutma testi: bir self-play koşusunun bütün sürümlerini SABİT, elle yazılmış rakiplere karşı oynatır.

Kullanım (repo kökünden, CPU yeter):
  python scripts/vs_scripted.py --run experiments/sp-empty-lr1e4-s0
  python scripts/vs_scripted.py --run experiments/x --worlds 256 --every 2

Neden: self-play'de ajan sadece havuzdaki rakiplere karşı uzmanlaşıp başka rakiplere karşı kötüleşebilir
("unutma"). Elle yazılmış rakip hiç değişmediği için mutlak bir cetvel: kovalayan_k OPEN kaçanına, kaçan_k elle
yazılmış kovalayana karşı oynar. Sürüm 0 = temel ajan (config'teki init_*), yani kıyas noktası.
Ölçüm temel ajanların değerlendirmesiyle (watch_policy) aynı: gürültüsüz, görülmemiş arenalar (seed + 999),
dünya başına 2 × 20 s. Çıktı: <run>/vs_scripted.json.
"""
import argparse
import glob
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from chase_tag.arenas import make_arena_set
from chase_tag.env import ChaseTagEnv
from chase_tag.play import crash_breakdown, summarize
from chase_tag.ppo import actor_dist, make_opponent
from chase_tag.rules import CHASER, EVADER, MAX_STEPS

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--worlds", type=int, default=512)
p.add_argument("--every", type=int, default=1, help="her k. sürüm (0 ve sonuncu hep dahil)")
a = p.parse_args()

cfg = json.load(open(os.path.join(a.run, "config.json")))
arena_seed = cfg["seed"] + 999
arenas = make_arena_set(200, phase=cfg["phase"], n_obst=tuple(cfg["n_obst"]), seed=arena_seed, verbose=False)
env = ChaseTagEnv(a.worlds, arenas, opp_crash_reward=cfg["opp_crash_reward"], full_obs=cfg.get("full_obs", False))
squash = cfg.get("squash_mean", False)
OPP = {"chaser": "open", "evader": "scripted_chaser"}      # temel ajanların eğitildiği rakipler


def make_eval(role):
    """eval(params, key) -> (sonuçlar, süreler, çarpma yerleri). Parametreler girdi: tek derleme, bütün sürümler."""
    me = CHASER if role == "chaser" else EVADER
    opp_fn, opp_init, opp_reset = make_opponent(OPP[role])

    def run(params, key):
        def body(carry, k):
            s, mem = carry
            k1, k2 = jax.random.split(k)
            mu, _ = actor_dist(params, env.observe(s)[:, me], squash)          # gürültüsüz: ortalama eylem
            a_opp, mem = opp_fn(s, mem, k1, None, None)
            act = jnp.zeros((a.worlds, 2, 3)).at[:, me].set(mu).at[:, 1 - me].set(a_opp)
            s, _, done, info = env.step(s, act, k2)
            return (s, opp_reset(mem, done)), (info["outcome"], info["length"], info["crash_kind"])
        k0, k1 = jax.random.split(key)
        _, out = jax.lax.scan(body, (env.reset(k0), opp_init(a.worlds)), jax.random.split(k1, 2 * MAX_STEPS))
        return out
    return jax.jit(run)


res = dict(run=os.path.basename(os.path.normpath(a.run)), worlds=a.worlds, steps=2 * MAX_STEPS, arena_seed=arena_seed,
           deterministic=True, opponents=OPP, versions={})
for role, tr in (("chaser", "kovalayan"), ("evader", "kaçan")):
    files = [cfg[f"init_{role}"]] + sorted(glob.glob(os.path.join(a.run, f"{role}_gen*.pkl")))
    idx = sorted(set(list(range(0, len(files), a.every)) + [len(files) - 1]))
    ev = make_eval(role)
    print(f"\n== {tr} sürümleri vs {OPP[role]} (gürültüsüz, {a.worlds} dünya × {2 * MAX_STEPS} adım)")
    rows = []
    for k in idx:
        t0 = time.perf_counter()
        params = jax.tree.map(jnp.asarray, pickle.load(open(files[k], "rb")))
        outcomes, lengths, kinds = ev(params, jax.random.PRNGKey(1))
        total, pct, mean_len = summarize(outcomes, lengths)
        crash = crash_breakdown(kinds)
        rows.append(dict(version=k, file=os.path.basename(files[k]), n_games=int(total), outcomes_pct=pct,
                         mean_len_s=mean_len / 50, crash_pct=crash))
        print(f"{tr}_{k:<2d} | {total:5d} maç | " + " | ".join(f"{n} %{v:5.1f}" for n, v in pct.items())
              + f" | {mean_len / 50:4.1f} s | {time.perf_counter() - t0:.0f} s")
    res["versions"][role] = rows

out = os.path.join(a.run, "vs_scripted.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
print(f"\nkaydedildi: {out}")
