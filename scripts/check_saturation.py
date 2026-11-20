"""Politikaların önerdiği ortalama eylemin boyu: 1'den büyükse ortam kırpıyor (doyma)."""
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import chase_tag  # noqa: F401
from chase_tag.arenas import make_arena_set
from chase_tag.env import ChaseTagEnv
from chase_tag.ppo import actor_dist

E = "d:/Masaüstü/DL PROJE/experiments"
F = "C:/Users/USER/AppData/Local/Temp/claude/45b85a352bf8698a/7cb14c4d-1c0d-4cf7-8b4d-6417ff9c08bb/scratchpad/faz1_zip/sp-faz1-s0"
pols = {
    "kovalayan_0 (doğrulama)": (f"{E}/ppo-chaser-empty-nocrashrew-s0/params_00476.pkl", 0),
    "kaçan_0 (doğrulama)": (f"{E}/ppo-evader-empty-nocrashrew-s0/params_00476.pkl", 1),
    "boş self-play kovalayan_10": (f"{E}/sp-empty-s0/chaser_gen18.pkl", 0),
    "boş self-play kaçan_10": (f"{E}/sp-empty-s0/evader_gen19.pkl", 1),
    "engelli self-play kovalayan_10": (f"{F}/chaser_gen18.pkl", 0),
    "engelli self-play kaçan_10": (f"{F}/evader_gen19.pkl", 1),
}
arenas = make_arena_set(20, phase=1, n_obst=(8, 12), seed=3, verbose=False)
env = ChaseTagEnv(256, arenas)
obs = []
s = env.reset(jax.random.PRNGKey(0))
step = jax.jit(env.step)
for t in range(40):                      # farklı durumlardan gözlem topla (rastgele eylemlerle)
    obs.append(np.asarray(env.observe(s)))
    s, *_ = step(s, jax.random.uniform(jax.random.PRNGKey(t), (256, 2, 3), minval=-1, maxval=1), jax.random.PRNGKey(100 + t))
obs = jnp.asarray(np.concatenate(obs))   # (N, 2, 33)
for name, (path, role) in pols.items():
    p = jax.tree.map(jnp.asarray, pickle.load(open(path, "rb")))
    mu, std = actor_dist(p, obs[:, role])
    n = np.linalg.norm(np.asarray(mu), axis=-1)
    print(f"{name:32s} ortalama eylem boyu: medyan {np.median(n):4.2f}, %10 {np.percentile(n, 10):4.2f}, "
          f"%90 {np.percentile(n, 90):4.2f} | boyu > 1 olan %{100 * np.mean(n > 1):3.0f} | std {np.asarray(std).round(3)}")
