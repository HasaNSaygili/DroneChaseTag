"""Elle yazılmış kovalayan vs kaçan: engel sayısına göre sonuç dağılımı.

Kullanım:  python scripts/scripted_match.py [n_worlds] [n_steps] [faz] [kaçan: open | open_mem]
"""
import functools
import sys
import time

import jax

from chase_tag.arenas import make_arena_set
from chase_tag.env import ChaseTagEnv
from chase_tag.play import make_runner, summarize
from chase_tag.scripted import reset_mem, scripted_actions, scripted_init

n_worlds = int(sys.argv[1]) if len(sys.argv) > 1 else 64
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
phase = int(sys.argv[3]) if len(sys.argv) > 3 else 1
evader = sys.argv[4] if len(sys.argv) > 4 else "open"
print("kaçan:", evader)

for k in [0, 4, 8]:
    arenas = make_arena_set(200, phase=phase, n_obst=(k, k), seed=100 + k, verbose=False)
    env = ChaseTagEnv(n_worlds, arenas)
    run = make_runner(env, functools.partial(scripted_actions, evader=evader), reset_mem)
    t0 = time.perf_counter()
    _, (outcomes, lengths, _) = run(env.reset(jax.random.PRNGKey(k)), scripted_init(n_worlds),
                                    jax.random.PRNGKey(1000 + k), n_steps)
    total, rows, mean_len = summarize(outcomes, lengths)
    print(f"faz {phase}, engel {k}: {total} maç ({time.perf_counter() - t0:.0f} s) | "
          + " | ".join(f"{name} %{v:.0f}" for name, v in rows.items())
          + f" | ort. süre {mean_len / 50:.1f} s")
