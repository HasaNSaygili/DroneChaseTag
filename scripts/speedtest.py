"""Hız testi: eğitim bütçesini planlamak için. Colab A100'de çalıştırılır; sonunda kopyalanacak bir özet basar.

Ölçülenler:
  1. Arena üretimi (CPU, NumPy): 1000 arena kaç saniye?
  2. Ortam + rastgele ağırlıklı iki MLP (kovalayan + kaçan, 2×256): saniyede kaç oyun adımı?
     PPO'nun veri toplama aşaması tam olarak bu: ağ eylem seçer, oyun ilerler, 128 adım GPU'da birikir.
  3. Ortam + elle yazılmış rakipler: yol bulma pahalı mı? (eğitimde rakip olarak kullanılacak)
  4. PPO güncellemesi: aktör + asimetrik critic, 4 epoch × 8 mini-batch; bir rollout'un verisi kaç saniye?

Kullanım:  python scripts/speedtest.py            (Colab, tam boy)
           python scripts/speedtest.py --small    (yerel CPU, sadece çalışıyor mu)
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from chase_tag.arenas import make_arena_set
from chase_tag.env import CRITIC_DIM, OBS_DIM, ChaseTagEnv
from chase_tag.scripted import reset_mem, scripted_actions, scripted_init

p = argparse.ArgumentParser()
p.add_argument("--small", action="store_true")
a = p.parse_args()

T = 128                                    # rollout uzunluğu (PPO'da bir veri toplama turu)
WORLDS = [64, 256] if a.small else [1024, 4096, 16384, 32768]
SCRIPTED_WORLDS = [64] if a.small else [1024, 4096]
HIDDEN = 256
dev = jax.devices()[0]
print(f"JAX {jax.__version__} | cihaz: {dev.platform} {getattr(dev, 'device_kind', '')}")
results = {}

# ---- 1. Arena üretimi ----
t0 = time.perf_counter()
arenas = make_arena_set(100 if a.small else 1000, phase=3, n_obst=(0, 8), seed=0, verbose=False)
results["arena_1000_s"] = (time.perf_counter() - t0) * (10 if a.small else 1)
print(f"[1] arena üretimi: 1000 arena ≈ {results['arena_1000_s']:.1f} s")


# ---- Rastgele ağırlıklı MLP (gerçek politika ağıyla aynı boyutta) ----
def mlp_init(key, sizes):
    ks = jax.random.split(key, len(sizes) - 1)
    return [(jax.random.normal(k, (i, o)) * jnp.sqrt(2.0 / i), jnp.zeros(o)) for k, i, o in zip(ks, sizes[:-1], sizes[1:])]


def mlp(params, x):
    for w, b in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def bench_rollout(n, policy_kind):
    env = ChaseTagEnv(n, arenas)
    pc = mlp_init(jax.random.PRNGKey(1), [OBS_DIM, HIDDEN, HIDDEN, 3])
    pe = mlp_init(jax.random.PRNGKey(2), [OBS_DIM, HIDDEN, HIDDEN, 3])

    def body(carry, key):
        s, mem = carry
        k1, k2 = jax.random.split(key)
        if policy_kind == "mlp":
            obs = env.observe(s)
            act = jnp.stack([jnp.tanh(mlp(pc, obs[:, 0])), jnp.tanh(mlp(pe, obs[:, 1]))], 1)
            _ = env.critic_obs(s)   # PPO'da critic girdisi de her adım hesaplanacak
        else:
            act, mem = scripted_actions(s, mem, k1)
        s, r, d, info = env.step(s, act, k2)
        if policy_kind != "mlp":
            mem = reset_mem(mem, d)
        return (s, mem), r

    run = jax.jit(lambda s, m, k: jax.lax.scan(body, (s, m), jax.random.split(k, T)))
    s, mem = env.reset(jax.random.PRNGKey(0)), scripted_init(n)
    t0 = time.perf_counter()
    (s, mem), r = run(s, mem, jax.random.PRNGKey(1))
    jax.block_until_ready(r)
    compile_s = time.perf_counter() - t0
    reps = 3
    t0 = time.perf_counter()
    for i in range(reps):
        (s, mem), r = run(s, mem, jax.random.PRNGKey(10 + i))
    jax.block_until_ready(r)
    dt = (time.perf_counter() - t0) / reps
    return n * T / dt, dt, compile_s


# ---- 2 + 3. Rollout hızı ----
for kind, worlds in (("mlp", WORLDS), ("scripted", SCRIPTED_WORLDS)):
    for n in worlds:
        try:
            sps, dt, cs = bench_rollout(n, kind)
            results[(kind, n)] = sps
            print(f"[{'2' if kind == 'mlp' else '3'}] {kind:8s} n_worlds={n:6d}: {sps / 1e6:6.2f} M adım/s "
                  f"(128 adım: {dt:.2f} s, derleme {cs:.0f} s)")
        except Exception as e:  # bellek yetmezse devam et
            print(f"    {kind} n_worlds={n}: HATA {type(e).__name__}: {str(e)[:120]}")
            break


# ---- 4. PPO güncelleme maliyeti ----
def bench_update(n_samples, epochs=4, n_mb=8):
    k = jax.random.PRNGKey(0)
    params = (mlp_init(k, [OBS_DIM, HIDDEN, HIDDEN, 3]), mlp_init(k, [CRITIC_DIM, HIDDEN, HIDDEN, 1]))
    opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
    st = opt.init(params)
    obs, cobs = jax.random.normal(k, (n_samples, OBS_DIM)), jax.random.normal(k, (n_samples, CRITIC_DIM))
    act, adv = jax.random.normal(k, (n_samples, 3)), jax.random.normal(k, (n_samples,))

    def loss(p, o, c, ac, ad):
        mu = jnp.tanh(mlp(p[0], o))
        logp = -jnp.sum((ac - mu) ** 2, -1)
        v = mlp(p[1], c)[:, 0]
        return -jnp.mean(logp * ad) + jnp.mean((v - ad) ** 2)

    @jax.jit
    def update(params, st, key):
        def epoch(carry, k):
            perm = jax.random.permutation(k, n_samples).reshape(n_mb, -1)

            def mb(carry, idx):
                params, st = carry
                g = jax.grad(loss)(params, obs[idx], cobs[idx], act[idx], adv[idx])
                u, st = opt.update(g, st, params)
                return (optax.apply_updates(params, u), st), None
            return jax.lax.scan(mb, carry, perm)[0], None
        return jax.lax.scan(epoch, (params, st), jax.random.split(key, epochs))[0]

    out = update(params, st, k)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    out = update(params, st, jax.random.PRNGKey(1))
    jax.block_until_ready(out)
    return time.perf_counter() - t0


for n in [key[1] for key in results if isinstance(key, tuple) and key[0] == "mlp"]:
    samples = n * T            # bir rol için (iki rol sırayla eğitileceği için tek rol yeterli)
    dt = bench_update(samples)
    results[("update", n)] = dt
    print(f"[4] PPO güncellemesi n_worlds={n:6d} ({samples / 1e6:.1f} M örnek): {dt:.2f} s")

# ---- Özet: bu bloğu kopyala ----
print("\n========== ÖZET (bunu kopyala) ==========")
print(f"cihaz: {dev.platform} {getattr(dev, 'device_kind', '')} | JAX {jax.__version__}")
print(f"arena üretimi: 1000 arena ≈ {results['arena_1000_s']:.0f} s")
for n in WORLDS:
    if ("mlp", n) in results:
        roll = n * T / results[("mlp", n)]
        upd = results.get(("update", n), float("nan"))
        total = n * T / (roll + upd)
        print(f"n_worlds={n:6d}: rollout {results[('mlp', n)] / 1e6:5.2f} M adım/s | güncelleme {upd:.2f} s | "
              f"toplam eğitim ≈ {total / 1e6:5.2f} M adım/s → 1e9 adım ≈ {1e9 / total / 3600:.1f} saat")
for n in SCRIPTED_WORLDS:
    if ("scripted", n) in results:
        print(f"elle yazılmış rakipler n_worlds={n}: {results[('scripted', n)] / 1e6:.2f} M adım/s")
