"""Elle yazılmış kovalayan vs kaçan maçlarını kaydedip izle.

Kullanım:
  python scripts/watch_match.py                       # faz 1, 6 engel, 3 maç → videos/*.mp4
  python scripts/watch_match.py --phase 3 --obst 8    # karışık arena
  python scripts/watch_match.py --live                # MuJoCo penceresinde canlı izle (fareyle kamera)
"""
import argparse
import functools
import os

import jax

from chase_tag.arenas import make_arena_set
from chase_tag.env import ChaseTagEnv
from chase_tag.play import make_runner, split_games
from chase_tag.rules import MAX_STEPS, OUTCOME_NAMES
from chase_tag.scripted import reset_mem, scripted_actions, scripted_init
from chase_tag.viewer import play_live, save_video

p = argparse.ArgumentParser()
p.add_argument("--phase", type=int, default=1)
p.add_argument("--obst", type=int, default=6)
p.add_argument("--games", type=int, default=3)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--evader", default="open", choices=["open", "open_mem"])
p.add_argument("--live", action="store_true")
p.add_argument("--out", default="videos")
a = p.parse_args()

arenas = make_arena_set(20, phase=a.phase, n_obst=(a.obst, a.obst), seed=a.seed, verbose=False)
env = ChaseTagEnv(2, arenas)
run = make_runner(env, functools.partial(scripted_actions, evader=a.evader), reset_mem)
# Kayıt sadece 0. dünyadan; istenen sayıda maç bitene kadar yetecek kadar adım
_, (_, _, rec) = run(env.reset(jax.random.PRNGKey(a.seed)), scripted_init(2), jax.random.PRNGKey(a.seed + 1),
                     MAX_STEPS * a.games)
games = split_games(rec)[:a.games]
os.makedirs(a.out, exist_ok=True)
for g, game in enumerate(games):
    dur = (game[1] - game[0] + 1) / 50
    print(f"maç {g}: {OUTCOME_NAMES[game[2]]}, {dur:.1f} s")
    if a.live:
        play_live(rec, arenas, game)
    else:
        path = os.path.join(a.out, f"scripted_{a.evader}_faz{a.phase}_engel{a.obst}_mac{g}.mp4")
        print("  →", save_video(rec, arenas, game, path))
