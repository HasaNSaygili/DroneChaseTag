"""Iskalanan yakalama fırsatları: kovalayan kaçana çok yaklaşıp neden yakalamadan uzaklaşıyor?

Kullanım (repo kökünden, CPU ~3 dk):
  python scripts/near_miss.py --run experiments/sp-empty-lr1e4-s0            # son kovalayan vs son kaçan

Yaklaşma olayı: 3B mesafe NEAR (0,5 m) altına iner. Olay şöyle biter: ebeleme (yakaladı), mesafe EXIT (0,6 m)
üstüne çıkar (ıska) ya da oyun başka şekilde biter (çarpma / süre). Iska ile yakalama, en yakın anda şu ölçülerle
karşılaştırılır (videodaki "dibindeyken uzaklaşıyor" gözleminin olası açıklamaları):
  1) |Δz|: yükseklik farkı (üstten görünümde üst üste ama dikeyde ayrı mı?)
  2) kaçanın yön değişimi ve kovalayanın hızı (son anda yana kaçış + kovalayanın ıskalayıp geçmesi mi?)
  3) kaçanın en yakın sınıra (zemin, tavan, duvar) mesafesi (kovalayan çarpma riskinden mi kaçınıyor?)
  4) olasılıklı vs gürültüsüz oynatmada ıska sıklığı (eylem gürültüsü mü?)
"""
import argparse
import glob
import json
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
from chase_tag.rules import ARENA, CHASER, EVADER, TAG, TAG_R

NEAR, EXIT = 0.5, 0.6
p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--chaser", type=int, default=-1)
p.add_argument("--evader", type=int, default=-1)
p.add_argument("--worlds", type=int, default=1024)
p.add_argument("--steps", type=int, default=1000)
a = p.parse_args()

cfg = json.load(open(os.path.join(a.run, "config.json")))
files = {r: [cfg[f"init_{r}"]] + sorted(glob.glob(os.path.join(a.run, f"{r}_gen*.pkl"))) for r in ("chaser", "evader")}
load = lambda f: jax.tree.map(jnp.asarray, pickle.load(open(f, "rb")))
ic, ie = a.chaser % len(files["chaser"]), a.evader % len(files["evader"])
p_ch, p_ev = load(files["chaser"][ic]), load(files["evader"][ie])
squash = cfg.get("squash_mean", False)
arenas = make_arena_set(200, phase=cfg["phase"], n_obst=tuple(cfg["n_obst"]), seed=cfg["seed"] + 999, verbose=False)
env = ChaseTagEnv(a.worlds, arenas, opp_crash_reward=cfg["opp_crash_reward"])
print(f"kovalayan_{ic} vs kaçan_{ie} | {a.worlds} dünya × {a.steps} adım | faz {cfg['phase']} engel {tuple(cfg['n_obst'])}")


def simulate(det):
    def step(s, key):
        k1, k2, k3 = jax.random.split(key, 3)
        obs = env.observe(s)
        acts = []
        for params, me, k in ((p_ch, CHASER, k1), (p_ev, EVADER, k2)):
            mu, std = actor_dist(params, obs[:, me], squash)
            acts.append(mu if det else mu + std * jax.random.normal(k, mu.shape))
        rec = dict(pos=s.data.states.pos, vel=s.data.states.vel, t=s.t, vis=s.visible[:, 0],
                   obst_e=jax.vmap(G.obstacle_dist)(s.data.states.pos[:, EVADER], s.arena))   # boş arenada inf
        s, _, _, info = env.step(s, jnp.stack(acts, 1), k3)
        rec["outcome"] = info["outcome"]
        return s, rec
    run = jax.jit(lambda s, k: jax.lax.scan(step, s, jax.random.split(k, a.steps)))
    _, rec = run(env.reset(jax.random.PRNGKey(0)), jax.random.PRNGKey(1))
    return {k: np.asarray(v) for k, v in rec.items()}


def events(rec):
    pos, vel, t, out = rec["pos"], rec["vel"], rec["t"], rec["outcome"]
    d = np.linalg.norm(pos[:, :, 0] - pos[:, :, 1], axis=-1)                       # (T, N)
    rows = []
    T = d.shape[0]
    for w in range(d.shape[1]):
        i = 1
        while i < T:
            if d[i, w] < NEAR and d[i - 1, w] >= NEAR and t[i, w] > 0:
                j, kind = i, None
                while j < T:
                    if out[j, w] != 0:
                        kind = "yakaladı" if out[j, w] == TAG else "diğer"
                        break
                    if d[j, w] > EXIT:
                        kind = "ıska"
                        break
                    j += 1
                if kind is None or i < 10 or j + 50 >= T:
                    break                                                           # kayıt sınırında yarım olay
                m = i + int(np.argmin(d[i:j + 1, w]))
                pc, pe, vc, ve = pos[m, w, 0], pos[m, w, 1], vel[m, w, 0], vel[m, w, 1]
                ve0, ve1 = vel[i - 10, w, 1], vel[min(m + 5, T - 1), w, 1]
                cosang = ve0 @ ve1 / (np.linalg.norm(ve0) * np.linalg.norm(ve1) + 1e-9)
                rel = vc - ve
                rows.append(dict(kind=kind, dmin=d[m, w], dz=abs(pc[2] - pe[2]), dxy=np.linalg.norm(pc[:2] - pe[:2]),
                                 vc=np.linalg.norm(vc), ve=np.linalg.norm(ve), relv=np.linalg.norm(rel),
                                 turn=np.degrees(np.arccos(np.clip(cosang, -1, 1))),
                                 bound_e=min(pe.min(), (ARENA - pe).min()), bound_c=min(pc.min(), (ARENA - pc).min()),
                                 ez=pe[2], obst_e=float(rec["obst_e"][m, w]),
                                 hidden=float(np.mean(~rec["vis"][max(i - 10, 0):j + 1, w])),
                                 dmax_after=d[j:j + 50, w].max() if kind == "ıska" else np.nan))
                i = j + 1
            else:
                i += 1
    return rows


def table(rows, name):
    k = np.array([r["kind"] for r in rows])
    n_t, n_m, n_o = (k == "yakaladı").sum(), (k == "ıska").sum(), (k == "diğer").sum()
    print(f"\n== {name}: {len(rows)} yaklaşma olayı (mesafe < {NEAR} m) → yakaladı {n_t}, ıska {n_m}, diğer {n_o} | "
          f"yaklaşmaların ıskayla bitme oranı %{100 * n_m / max(n_t + n_m, 1):.0f}")
    cols = [("dmin", "en yakın mesafe m"), ("dz", "|Δz| m"), ("dxy", "yatay mesafe m"), ("vc", "kovalayan hızı m/s"),
            ("ve", "kaçan hızı m/s"), ("relv", "bağıl hız m/s"), ("turn", "kaçanın yön değişimi °"),
            ("bound_e", "kaçanın sınıra mesafesi m"), ("bound_c", "kovalayanın sınıra mesafesi m"), ("ez", "kaçanın z'si m")]
    if np.isfinite([r["obst_e"] for r in rows]).any():          # engelli arena
        cols += [("obst_e", "kaçanın engele mesafesi m"), ("hidden", "görüş kapalı zaman oranı")]
    print(f"{'ölçü (en yakın anda), medyan':32s} | {'yakaladı':>9s} | {'ıska':>9s}")
    for c, lab in cols:
        f = lambda kind: np.median([r[c] for r in rows if r["kind"] == kind]) if (k == kind).any() else np.nan
        print(f"{lab:32s} | {f('yakaladı'):9.2f} | {f('ıska'):9.2f}")
    miss = [r for r in rows if r["kind"] == "ıska"]
    if miss:
        dz = np.array([r["dz"] for r in miss]); dxy = np.array([r["dxy"] for r in miss])
        print(f"ıskalarda: |Δz| > yatay mesafe (dikeyde ayrı) %{100 * np.mean(dz > dxy):.0f} | "
              f"kaçan sınıra < 0,3 m %{100 * np.mean([r['bound_e'] < 0.3 for r in miss]):.0f} | "
              f"kaçanın yön değişimi > 60° %{100 * np.mean([r['turn'] > 60 for r in miss]):.0f} | "
              f"ıskadan sonraki 1 s'de en uzak mesafe medyanı {np.nanmedian([r['dmax_after'] for r in miss]):.2f} m")
    hits = [r for r in rows if r["kind"] == "yakaladı"]
    if miss and np.isfinite([r["obst_e"] for r in rows]).any():
        f = lambda rs, c: 100 * np.mean([c(r) for r in rs]) if rs else np.nan
        print(f"engel: kaçan engele < 0,3 m: yakaladı %{f(hits, lambda r: r['obst_e'] < 0.3):.0f} vs ıska "
              f"%{f(miss, lambda r: r['obst_e'] < 0.3):.0f} | kaçan zemine/tavana/duvara < 0,3 m: yakaladı "
              f"%{f(hits, lambda r: r['bound_e'] < 0.3):.0f} vs ıska %{f(miss, lambda r: r['bound_e'] < 0.3):.0f} | "
              f"olay boyunca görüş hiç kapanmadı: yakaladı %{f(hits, lambda r: r['hidden'] == 0):.0f} vs ıska "
              f"%{f(miss, lambda r: r['hidden'] == 0):.0f}")
    if hits and miss:
        print(f"karşılaştırma: kaçan sınıra < 0,3 m: yakaladı %{100 * np.mean([r['bound_e'] < 0.3 for r in hits]):.0f} "
              f"vs ıska %{100 * np.mean([r['bound_e'] < 0.3 for r in miss]):.0f} | yön değişimi > 60°: yakaladı "
              f"%{100 * np.mean([r['turn'] > 60 for r in hits]):.0f} vs ıska %{100 * np.mean([r['turn'] > 60 for r in miss]):.0f}")


print(f"(yakalama yarıçapı {TAG_R} m)")
for det, name in ((False, "olasılıklı (eğitimdeki gibi)"), (True, "gürültüsüz")):
    table(events(simulate(det)), name)
