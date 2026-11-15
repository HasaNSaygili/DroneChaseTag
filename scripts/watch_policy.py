"""Eğitilmiş bir politikayı değerlendir ve maçlarını videoya çek.

Kullanım:
  python scripts/watch_policy.py --run results/ppo-chaser-empty-s0
  python scripts/watch_policy.py --run results/x --params params_00200.pkl --games 3 --set "n_obst=(4,8)"

Arenalar eğitimdekinden farklı bir seed'le üretilir (hiç görülmemiş düzenler). Politika gürültüsüz oynar.
"""
import argparse
import ast
import glob
import json
import os
import pickle

import jax

from chase_tag.arenas import make_arena_set
from chase_tag.env import ChaseTagEnv
from chase_tag.play import crash_breakdown, make_runner, split_games, summarize
from chase_tag.ppo import learned_vs_opponent
from chase_tag.rules import MAX_STEPS, OUTCOME_NAMES
from chase_tag.viewer import save_video

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--params", default=None, help="varsayılan: en son params_*.pkl")
p.add_argument("--games", type=int, default=3)
p.add_argument("--eval-worlds", type=int, default=256)
p.add_argument("--set", nargs="*", default=[], help="config'i değiştir (örn. opponent=open_mem phase=3)")
p.add_argument("--out", default="videos")
a = p.parse_args()

cfg = json.load(open(os.path.join(a.run, "config.json")))
for kv in a.set:
    k, v = kv.split("=", 1)
    try:
        cfg[k] = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        cfg[k] = v
path = os.path.join(a.run, a.params) if a.params else sorted(glob.glob(os.path.join(a.run, "params_*.pkl")))[-1]
params = pickle.load(open(path, "rb"))
print(f"politika: {path} | öğrenen: {cfg['learner']} | rakip: {cfg['opponent']} | faz {cfg['phase']}, "
      f"engel {cfg['n_obst']}")
arenas = make_arena_set(200, phase=cfg["phase"], n_obst=tuple(cfg["n_obst"]), seed=cfg["seed"] + 999, verbose=False)

# ---- Değerlendirme: çok arenada, gürültüsüz ----
ocr = cfg.get("opp_crash_reward", 1.0)
env = ChaseTagEnv(a.eval_worlds, arenas, opp_crash_reward=ocr)
policy, opp_init, opp_reset = learned_vs_opponent(params, cfg["learner"], cfg["opponent"], env)
run = make_runner(env, policy, opp_reset)
_, (outcomes, lengths, kinds, _) = run(env.reset(jax.random.PRNGKey(1)), opp_init(a.eval_worlds),
                                       jax.random.PRNGKey(2), MAX_STEPS * 2)
total, rows, mean_len = summarize(outcomes, lengths)
print(f"değerlendirme: {total} maç | " + " | ".join(f"{k} %{v:.1f}" for k, v in rows.items())
      + f" | ort. süre {mean_len / 50:.1f} s")
crash = crash_breakdown(kinds)
for role, where in crash.items():
    if where:
        print(f"  {role} çarpma yerleri: " + ", ".join(f"{k} %{v:.0f}" for k, v in where.items() if v > 0))
# Sonucu koşu klasörüne de yaz: makaledeki sayılar bu dosyadan gelecek
ev = dict(params=os.path.basename(path), eval_worlds=a.eval_worlds, eval_steps=MAX_STEPS * 2,
          arena_seed=cfg["seed"] + 999, overrides=a.set, n_games=int(total), outcomes_pct=rows,
          mean_len_s=mean_len / 50, crash_pct=crash)
tag_ = "_".join(s.replace("=", "-") for s in a.set)
with open(os.path.join(a.run, f"eval_{os.path.splitext(ev['params'])[0]}{'_' + tag_ if tag_ else ''}.json"), "w",
          encoding="utf-8") as f:
    json.dump(ev, f, indent=2, ensure_ascii=False)

# ---- Video ----
env = ChaseTagEnv(2, arenas, opp_crash_reward=ocr)
policy, opp_init, opp_reset = learned_vs_opponent(params, cfg["learner"], cfg["opponent"], env)
run = make_runner(env, policy, opp_reset)
_, (_, _, _, rec) = run(env.reset(jax.random.PRNGKey(3)), opp_init(2), jax.random.PRNGKey(4), MAX_STEPS * a.games)
os.makedirs(a.out, exist_ok=True)
name = os.path.basename(os.path.normpath(a.run))
tag = os.path.splitext(os.path.basename(path))[0]
for g, game in enumerate(split_games(rec)[:a.games]):
    out = os.path.join(a.out, f"{name}_{tag}_mac{g}.mp4")
    print(f"maç {g}: {OUTCOME_NAMES[game[2]]}, {(game[1] - game[0] + 1) / 50:.1f} s → {save_video(rec, arenas, game, out)}")
