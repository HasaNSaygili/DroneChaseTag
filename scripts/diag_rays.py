"""Sensör kör noktası tanısı (CPU, ~2 dk). Repo kökünden:
  python scripts/diag_rays.py experiments/sp-faz1-lr1e4-s0

Soru: 16 yatay ışın 22,5° aralıklı; ince bir silindir iki ışının arasına düşüp görünmez olabilir mi
(teoride r / sin 11,25° ≈ 5,1·r mesafeden ötesi)? Engelli arenada kovalayanın ~%10'luk çarpması bundan mı?

1) Statik: rastgele noktalarda yatay ışınların gösterdiği en yakın mesafe vs gerçek en yakın mesafe.
2) Çarpma incelemesi: koşunun son kovalayanı vs son kaçanı, görülmemiş arenalar; her engel çarpmasından önceki
   adımlarda engel ışınlarda görünüyor muydu?
"""
import glob
import os
import pickle
import sys

sys.stdout.reconfigure(encoding="utf-8")
import jax
import jax.numpy as jnp
import numpy as np

from chase_tag import geometry as G
from chase_tag.arenas import make_arena_set
from chase_tag.env import ChaseTagEnv
from chase_tag.ppo import actor_dist
from chase_tag.rules import ARENA, CHASER, EVADER

RUN = sys.argv[1]
arenas = make_arena_set(200, phase=1, n_obst=(8, 12), seed=999, verbose=False)
A = G.arena_from_set(arenas)
NH = G.N_RAYS_H


def true_clear(p, arena):
    """Gerçek en yakın yüzey (engel ya da duvar; yatay düzlemde ışınların göreceği şeyler)."""
    wall = jnp.min(jnp.concatenate([p[:2], ARENA[:2] - p[:2]]))
    return jnp.minimum(G.obstacle_dist(p, arena), wall), G.obstacle_dist(p, arena)


# ---------------------------------------------------------------- 1) statik geometri
rng = np.random.default_rng(0)
M = 200_000
k = rng.integers(0, A.boxes.shape[0], M)
pts = np.c_[rng.uniform(0.06, ARENA[0] - 0.06, M), rng.uniform(0.06, ARENA[1] - 0.06, M), np.full(M, 1.0)]
arena_k = jax.tree.map(lambda x: jnp.asarray(x)[k], A)
f = jax.jit(jax.vmap(lambda p, a: (*true_clear(p, a), jnp.min(G.rays(p, a)[:NH]))))
tc, od, ray_min = map(np.asarray, f(jnp.asarray(pts, jnp.float32), arena_k))
ok = od > G.DRONE_R                                   # engelin içinde olmayan noktalar
near_obst = ok & (od <= tc + 1e-6)                    # en yakın yüzey bir ENGEL (duvar değil)
print("== 1) Statik: en yakın yüzey bir engel olan noktalar (z = 1 m, görülmemiş arenalar)")
print("gerçek mesafe | nokta | ışınların gösterdiği − gerçek: medyan | %90'lık | 'kör' oranı (fark > 0,2 m)")
for lo, hi in [(0.06, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.0), (1.0, 1.5)]:
    m = near_obst & (tc >= lo) & (tc < hi)
    gap = ray_min[m] - tc[m]
    print(f"{lo:.2f}–{hi:.2f} m | {m.sum():6d} | {np.median(gap):.3f} | {np.quantile(gap, 0.9):.3f} | "
          f"%{100 * np.mean(gap > 0.2):.1f}")

# ---------------------------------------------------------------- 2) çarpma incelemesi
gens = lambda r: sorted(glob.glob(os.path.join(RUN, f"{r}_gen*.pkl")))
load = lambda p: jax.tree.map(jnp.asarray, pickle.load(open(p, "rb")))
p_ch, p_ev = load(gens("chaser")[-1]), load(gens("evader")[-1])
print(f"\n== 2) Çarpma incelemesi: {os.path.basename(gens('chaser')[-1])} vs {os.path.basename(gens('evader')[-1])}")
N, T = 1024, 1000
env = ChaseTagEnv(N, arenas, opp_crash_reward=1.0)


def step(carry, key):
    s = carry
    k1, k2, k3 = jax.random.split(key, 3)
    obs = env.observe(s)
    mu_c, sd_c = actor_dist(p_ch, obs[:, CHASER])
    mu_e, sd_e = actor_dist(p_ev, obs[:, EVADER])
    a = jnp.stack([mu_c + sd_c * jax.random.normal(k1, mu_c.shape), mu_e + sd_e * jax.random.normal(k2, mu_e.shape)], 1)
    pos, vel = s.data.states.pos, s.data.states.vel
    rays = env._rays(s)                                                         # (N, 2, 18)
    tcl = jax.vmap(jax.vmap(lambda p, ar: true_clear(p, ar)[0], (0, None)))(pos, s.arena)   # (N, 2)
    s2, _, _, info = env.step(s, a, k3)
    rec = dict(ray_min=jnp.min(rays[..., :NH], -1), clear=tcl, speed=jnp.linalg.norm(vel, axis=-1),
               crash=info["crash_kind"], outcome=info["outcome"])
    return s2, rec


run = jax.jit(lambda s, k: jax.lax.scan(step, s, jax.random.split(k, T)))
_, rec = run(env.reset(jax.random.PRNGKey(0)), jax.random.PRNGKey(1))
rec = {kk: np.asarray(v) for kk, v in rec.items()}                              # (T, N, ...)

LOOK = [1, 5, 10, 15, 25]                                                       # kaç adım önce (×20 ms)
for role, name in ((CHASER, "kovalayan"), (EVADER, "kaçan")):
    tt, ww = np.nonzero(rec["crash"][:, :, role] == G.HIT_OBSTACLE)
    keep = tt >= max(LOOK)
    tt, ww = tt[keep], ww[keep]
    print(f"\n{name}: {len(tt)} engel çarpması (en az 0,5 s süren maçlarda)")
    if not len(tt):
        continue
    print("çarpmadan önce | gerçek mesafe medyan | ışın gösterdiği medyan | 'görünmüyor' oranı (fark > 0,15 m) | hız medyan")
    for L in LOOK:
        c = rec["clear"][tt - L + 1, ww, role]          # kayıt t = çarpan adımdan ÖNCEKİ durum; L=1 → son adım
        r = rec["ray_min"][tt - L + 1, ww, role]
        v = rec["speed"][tt - L + 1, ww, role]
        print(f"{20 * L:4d} ms | {np.median(c):.2f} m | {np.median(r):.2f} m | %{100 * np.mean(r - c > 0.15):.0f} | {np.median(v):.2f} m/s")
    # Engel son 0,3 s boyunca hiç ışında görüldü mü (ışın ≈ gerçek, fark < 0,1)?
    seen = np.zeros(len(tt), bool)
    for L in range(1, 16):
        seen |= (rec["ray_min"][tt - L + 1, ww, role] - rec["clear"][tt - L + 1, ww, role]) < 0.1
    print(f"son 0,3 s'de engeli ışınlarda en az bir kez doğru gören: %{100 * seen.mean():.0f}")
