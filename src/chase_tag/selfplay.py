"""Self-play: kovalayan ve kaçan SIRAYLA, birbirlerinin donmuş sürümlerine karşı öğrenir.

Bir NESİL = bir tarafın öğrendiği dönem. Öğrenen, rakip rolün HAVUZUNDAKİ ağlara karşı PPO ile eğitilir;
rakipler donuktur (öğrenmez). Nesil bitince öğrenenin donmuş kopyası kendi rolünün havuzuna eklenir ve sıra
öbür tarafa geçer. Böylece iki taraf birbirine cevap vererek ilerler: kaçan daire çizmeyi bulur → kovalayan
önünü kesmeyi öğrenir → kaçan yeni bir şey bulur → ...

Neden sırayla (ikisi aynı anda değil)? Aynı anda öğrenirlerse her biri sürekli kayan bir hedefe nişan alır;
öğrenme kararsızlaşır. Sırayla eğitimde her nesil sabit bir rakip kümesine karşı, iyi tanımlı bir problem.

Neden havuz (sadece en son rakip değil)? Sadece en son rakibe karşı oynayan ajan eski taktikleri UNUTUR ve
taraflar bir döngüde dönebilir (kovalayan daireyi kesmeyi unutur, kaçan daireye geri döner...). Bu yüzden:
  * blokların YARISI her zaman en son rakiple oynar (asıl soru: "son hamleye cevap verebildin mi?"),
  * diğer yarısı havuzdan PFSP ile seçilir: zorlandığın rakip daha sık gelir, ağırlık (1 − x)²,
    x = öğrenenin o rakibe karşı kazanma oranı (Gavin & Bronz 2026'nın tarifi).

Sıra değişimi: öğrenen EN SON rakibe karşı son `win_window` turda ≥ `switch_win` kazanınca (en az `min_steps`
adımdan sonra) ya da `max_steps` dolunca.

Uygulama ayrıntısı: arenalar `n_blocks` bloğa bölünür, her blok bir rakip ağla oynar. Rakip ağırlıkları
argüman olarak geçtiği için havuz büyüdükçe yeniden derleme olmaz (havuz `pool_max` boyutlu sabit dizi).
PFSP blokları her `resample_every` turda yeniden çekilir; maçlar turlara yayıldığı için bir maçın ortasında
rakip değişebilir (yaklaşıklık; en son rakip blokları hiç değişmez).
"""
import collections
import dataclasses
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

from .arenas import make_arena_set
from .env import OBS_DIM, ChaseTagEnv
from .geometry import CRASH_KIND_NAMES
from .ppo import PPOConfig, _crash_summary, actor_dist, make_update
from .rules import C_CRASH, CHASER, CTRL_HZ, E_CRASH, EVADER, OUTCOME_NAMES, TAG, TIMEOUT

ROLES = ("chaser", "evader")
ROLE_TR = {"chaser": "kovalayan", "evader": "kaçan"}


@dataclasses.dataclass
class SelfPlayConfig:
    # ---- ortam ----
    phase: int = 1
    n_obst: tuple = (0, 0)
    n_arenas: int = 1000
    opp_crash_reward: float = 1.0         # WCT kuralı: rakip çarparsa sen kazanırsın
    # ---- ölçek ----
    n_worlds: int = 16384
    n_steps: int = 64
    n_blocks: int = 8
    # ---- PPO (iki rol için aynı; γ hariç) ----
    epochs: int = 4
    n_minibatches: int = 8
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    lr: float = 3e-4                      # SABİT: self-play'de öğrenme "bitmez", rakip hep değişir
    max_grad_norm: float = 0.5
    hidden: int = 256
    gamma_chaser: float = 0.995           # ufuk ~4 s
    gamma_evader: float = 0.998           # ufuk ~10 s (kaçanın ödülü 20 s sonunda)
    # ---- self-play ----
    init_chaser: str = "experiments/ppo-chaser-empty-nocrashrew-s0/params_00476.pkl"   # kovalayan_0
    init_evader: str = "experiments/ppo-evader-empty-nocrashrew-s0/params_00476.pkl"   # kaçan_0
    first_learner: str = "chaser"         # kaçan_0 daire çiziyor; önce kovalayan cevap versin
    n_gens: int = 20                      # toplam nesil (yarısı kovalayan, yarısı kaçan)
    min_steps: int = 20_000_000           # bir nesilde en az bu kadar adım (gürültüyle erken geçişi önler)
    max_steps: int = 100_000_000          # bir nesilde en fazla (oyun yapısal olarak dengesizse takılmasın)
    switch_win: float = 0.6               # en son rakibe karşı bu orana ulaşınca sıra değişir
    win_window: int = 5                   # bu oran son kaç turun toplamından hesaplanır
    p_latest: float = 0.5                 # blokların bu kadarı hep en son rakiple oynar
    resample_every: int = 10              # PFSP blokları kaç turda bir yeniden çekilir
    pool_max: int = 32
    win_ema: float = 0.1                  # rakip başına kazanma oranının üstel ortalama katsayısı
    seed: int = 0
    log_every: int = 5


def role_ppo_cfg(sp, role):
    """Bir rol için PPO ayarları (ppo.make_update bunları kullanır)."""
    keys = ("phase", "n_obst", "n_arenas", "opp_crash_reward", "n_worlds", "n_steps", "n_blocks", "epochs",
            "n_minibatches", "gae_lambda", "clip_eps", "vf_coef", "ent_coef", "lr", "max_grad_norm", "hidden",
            "seed")
    return PPOConfig(learner=role, opponent="pool", anneal_lr=False,
                     gamma=sp.gamma_chaser if role == "chaser" else sp.gamma_evader,
                     **{k: getattr(sp, k) for k in keys})


def make_pool_opponent(n, n_blocks, opp_role):
    """Havuzdaki donmuş ağlarla oynayan rakip. opp_args = (havuz dizisi, blok başına havuz indeksi)."""
    bs = n // n_blocks

    def fn(es, mem, key, opp_args, obs_all):
        pool, slot = opp_args
        p = jax.tree.map(lambda x: x[slot], pool)                          # (B, ...) her bloğun rakip ağı
        obs = obs_all[:, opp_role].reshape(n_blocks, bs, OBS_DIM)
        mu, std = jax.vmap(actor_dist)(p, obs)                             # (B, bs, 3), (B, 3)
        a = mu + std[:, None, :] * jax.random.normal(key, mu.shape)        # rakip de olasılıklı oynar
        return a.reshape(n, 3), mem

    return fn, (lambda n: None), (lambda mem, done: mem)


def _load_params(path):
    with open(path, "rb") as f:
        return jax.tree.map(jnp.asarray, pickle.load(f))


def _save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(jax.device_get(obj), f)
    os.replace(tmp, path)       # yarım yazılmış checkpoint kalmasın (Colab bağlantısı koparsa)


class Pool:
    """Bir rolün donmuş sürümleri: sabit boyutlu dizi + doluluk sayısı."""

    def __init__(self, template, size):
        self.arr = jax.tree.map(lambda x: jnp.zeros((size,) + x.shape, x.dtype), template)
        self.n, self.size = 0, size

    def add(self, params):
        i = self.n if self.n < self.size else 1 + (self.n - 1) % (self.size - 1)   # dolunca 0. hariç en eskiyi ez
        self.arr = jax.tree.map(lambda P, x: P.at[i].set(x), self.arr, params)
        self.n = min(self.n + 1, self.size)
        return i


def train_selfplay(sp, out_dir, resume=True, log=print):
    os.makedirs(out_dir, exist_ok=True)
    log_file = open(os.path.join(out_dir, "train_log.txt"), "a", encoding="utf-8")
    screen = log

    def log(msg):
        screen(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(sp), f, indent=2)

    log(f"arenalar üretiliyor ({sp.n_arenas}, faz {sp.phase}, engel {sp.n_obst})...")
    arenas = make_arena_set(sp.n_arenas, phase=sp.phase, n_obst=tuple(sp.n_obst), seed=sp.seed, verbose=False)
    env = ChaseTagEnv(sp.n_worlds, arenas, opp_crash_reward=sp.opp_crash_reward)
    role_idx = {"chaser": CHASER, "evader": EVADER}
    cfgs = {r: role_ppo_cfg(sp, r) for r in ROLES}
    updates, txs = {}, {}
    for r in ROLES:
        other = role_idx["evader" if r == "chaser" else "chaser"]
        updates[r], txs[r], _ = make_update(cfgs[r], env, role_idx[r],
                                            opponent=make_pool_opponent(sp.n_worlds, sp.n_blocks, other))

    # ---- durum: iki rolün öğrenen ağı, Adam durumu, havuzu, rakip başına kazanma oranları ----
    ckpt = os.path.join(out_dir, "checkpoint.pkl")
    key = jax.random.PRNGKey(sp.seed)
    if resume and os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            c = pickle.load(f)
        params = {r: jax.tree.map(jnp.asarray, c["params"][r]) for r in ROLES}
        opt = {r: jax.tree.map(jnp.asarray, c["opt"][r]) for r in ROLES}
        pools = {}
        for r in ROLES:
            pools[r] = Pool(params[r], sp.pool_max)
            pools[r].arr, pools[r].n = jax.tree.map(jnp.asarray, c["pool"][r]), c["pool_n"][r]
        ema, gen0, key, total = c["ema"], c["gen"], jnp.asarray(c["key"]), c["total_steps"]
        log(f"checkpoint'ten devam: nesil {gen0}")
    else:
        init = {"chaser": _load_params(sp.init_chaser), "evader": _load_params(sp.init_evader)}
        params = dict(init)
        opt = {r: txs[r].init(params[r]) for r in ROLES}
        pools = {r: Pool(init[r], sp.pool_max) for r in ROLES}
        for r in ROLES:
            pools[r].add(init[r])                                  # havuz: kovalayan_0 ve kaçan_0
        ema = {r: np.full(sp.pool_max, 0.5) for r in ROLES}        # ema[r][j]: r'nin, rakip havuzu j'ye karşı kazanması
        gen0, total = 0, 0

    batch = sp.n_worlds * sp.n_steps
    n_latest = max(1, int(round(sp.n_blocks * sp.p_latest)))
    mfile = open(os.path.join(out_dir, "metrics.jsonl"), "a")
    rng = np.random.default_rng(sp.seed + 1000 * gen0)
    t_start = time.perf_counter()

    for gen in range(gen0, sp.n_gens):
        role = ROLES[(ROLES.index(sp.first_learner) + gen) % 2]
        other = "evader" if role == "chaser" else "chaser"
        me = role_idx[role]
        table = env.reward_table
        win_k = [k for k in range(1, len(OUTCOME_NAMES)) if table[k, me] > 0]   # öğrenenin kazandığı sonuçlar
        pool = pools[other]
        latest = pool.n - 1
        log(f"\n=== nesil {gen} · öğrenen: {ROLE_TR[role]} · rakip havuzu: {pool.n} {ROLE_TR[other]} "
            f"(en son: {ROLE_TR[other]}_{latest}) ===")
        key, k_env = jax.random.split(key)
        es, mem = env.reset(k_env), None
        slot = np.full(sp.n_blocks, latest, np.int32)
        recent = collections.deque(maxlen=sp.win_window)            # (kazanılan, biten) en son rakibe karşı
        window = np.zeros(len(OUTCOME_NAMES))
        crash_window = np.zeros((2, len(CRASH_KIND_NAMES)))
        steps, u, t_gen = 0, 0, time.perf_counter()
        while True:
            if u % sp.resample_every == 0:                           # PFSP bloklarını yeniden çek
                w = (1.0 - ema[role][:pool.n]) ** 2
                w = w / w.sum() if w.sum() > 0 else np.full(pool.n, 1.0 / pool.n)
                slot[n_latest:] = rng.choice(pool.n, size=sp.n_blocks - n_latest, p=w)
            params[role], opt[role], es, mem, key, m = updates[role](
                params[role], opt[role], es, mem, key, (pool.arr, jnp.asarray(slot)))
            m = jax.device_get(m)
            steps += batch
            total += batch
            u += 1
            bc = m["block_counts"]                                   # (B, 6)
            wins_b, ended_b = bc[:, win_k].sum(1), bc[:, 1:].sum(1)
            recent.append((wins_b[:n_latest].sum(), ended_b[:n_latest].sum()))
            for j in np.unique(slot):                                # rakip başına kazanma oranını güncelle
                sel = slot == j
                if ended_b[sel].sum() > 0:
                    ema[role][j] = (1 - sp.win_ema) * ema[role][j] + sp.win_ema * wins_b[sel].sum() / ended_b[sel].sum()
            win_latest = float(sum(a for a, _ in recent) / max(sum(b for _, b in recent), 1))
            window += m["outcome_counts"]
            crash_window += m["crash_counts"]
            row = dict(gen=gen, learner=role, update=u, gen_steps=steps, total_steps=total,
                       time_s=time.perf_counter() - t_start, slot=slot.tolist(), win_latest_window=win_latest,
                       ema=ema[role][:pool.n].tolist(),
                       **{k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in m.items()})
            mfile.write(json.dumps(row) + "\n")
            done_gen = steps >= sp.max_steps or (steps >= sp.min_steps and len(recent) == sp.win_window
                                                 and win_latest >= sp.switch_win)
            if u % sp.log_every == 0 or done_gen:
                mfile.flush()
                tot = window[1:].sum()
                pct = lambda k: 100 * window[k] / max(tot, 1)
                sps = steps / (time.perf_counter() - t_gen)
                log(f"nesil {gen:2d} {ROLE_TR[role]:9s} tur {u:3d} | {steps / 1e6:4.0f} M | {sps / 1e6:.2f} M/s | "
                    f"en son rakibe karşı kazanma %{100 * win_latest:5.1f} | havuz ort. %{100 * ema[role][:pool.n].mean():5.1f} | "
                    f"ebeleme %{pct(TAG):4.1f} süre %{pct(TIMEOUT):4.1f} K.çarp %{pct(C_CRASH):4.1f} "
                    f"E.çarp %{pct(E_CRASH):4.1f} | süre {m['ep_len'] / CTRL_HZ:4.1f} s | std {m['std']:.2f} "
                    f"kl {m['approx_kl']:.4f} | çarpma K[{_crash_summary(crash_window, CHASER)}] "
                    f"E[{_crash_summary(crash_window, EVADER)}]")
                window[:] = 0
                crash_window[:] = 0
            if done_gen:
                break
        # ---- nesil sonu: öğreneni havuzuna ekle, kaydet ----
        why = "eşik" if win_latest >= sp.switch_win else "en fazla adım"
        j = pools[role].add(params[role])
        ema[other][j] = 0.5                                          # yeni rakip: henüz bilinmiyor
        _save(os.path.join(out_dir, f"{role}_gen{gen:02d}.pkl"), params[role])
        log(f"--- nesil {gen} bitti ({why}): {ROLE_TR[role]}, {steps / 1e6:.0f} M adım, "
            f"{(time.perf_counter() - t_gen) / 60:.1f} dk, en son rakibe karşı %{100 * win_latest:.1f} → "
            f"havuza {ROLE_TR[role]}_{j} olarak eklendi")
        _save(ckpt, dict(params=params, opt=opt, pool={r: pools[r].arr for r in ROLES},
                         pool_n={r: pools[r].n for r in ROLES}, ema=ema, gen=gen + 1, key=key, total_steps=total))
    mfile.close()
    log(f"\nbitti: {total / 1e6:.0f} M adım, {(time.perf_counter() - t_start) / 60:.1f} dk")
    log_file.close()
    return params, pools
