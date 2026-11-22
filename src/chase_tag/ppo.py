"""PPO (Proximal Policy Optimization): bir rolü (kovalayan ya da kaçan) sabit bir rakibe karşı eğitir.

Büyük resim — her PPO turu iki aşama:
  1. VERİ TOPLA (rollout): binlerce arenada T adım oyna. Her adımda aktör bir eylem seçer, critic
     "bu durum ne kadar iyi" der, ortam ödül verir. Hepsi kaydedilir.
  2. ÖĞREN (update): kaydedilen veriyle iki ağı güncelle, birkaç tur (epoch) boyunca:
       - aktör: avantajı pozitif olan eylemlerin olasılığını artır, negatif olanlarınkini azalt,
                ama her turda en fazla ~%20 değiş (PPO'nun kırpması)
       - critic: tahminlerini gerçekte olanlara yaklaştır
  Sonra yeni aktörle tekrar veri topla. Self-play (iki rolün sırayla eğitilmesi) bunun üstüne kurulacak.

Kavramlar (koda girdikleri yerde tekrar açıklanıyor):
  avantaj A = (gerçekte ne oldu) − (critic ne bekliyordu); GAE bunu kısa ve uzun vadeyi karıştırarak hesaplar.
  oran r = π_yeni(a|s) / π_eski(a|s); PPO r'yi [1−ε, 1+ε] dışına itmeyi ödüllendirmez → küçük, güvenli adımlar.

Uygulama ayrıntıları "The 37 Implementation Details of PPO" (Huang vd., ICLR blog 2022) ve PureJaxRL'i izler:
ortogonal başlatma, avantaj normalizasyonu, kırpılmış değer kaybı, gradyan kırpma, öğrenme oranının azalması.
"""
import dataclasses
import json
import os
import pickle
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import scripted
from .arenas import make_arena_set
from .env import CRITIC_DIM, OBS_DIM, ChaseTagEnv
from .geometry import CRASH_KIND_NAMES
from .rules import C_CRASH, CHASER, CTRL_HZ, E_CRASH, EVADER, OUTCOME_NAMES, TAG, TIMEOUT


@dataclasses.dataclass
class PPOConfig:
    # ---- ne öğreniyor, kime karşı, nerede ----
    learner: str = "chaser"          # "chaser" | "evader"
    opponent: str = "open"           # "open" | "open_mem" (kaçan rakipler) | "scripted_chaser"
    phase: int = 1                   # engel aşaması (arenas.PHASE_KINDS)
    n_obst: tuple = (0, 0)           # arena başına engel sayısı aralığı
    n_arenas: int = 1000             # önceden üretilecek arena sayısı
    opp_crash_reward: float = 1.0    # rakip çarpınca öbür tarafın ödülü: 1 = WCT kuralı, 0 = Gavin (ablasyon 7)
    # ---- ölçek ----
    n_worlds: int = 16384            # aynı anda oynanan arena sayısı
    n_steps: int = 64                # T: bir turda her arenada kaç adım (batch = n_worlds × T)
    n_blocks: int = 8                # arenalar bu kadar bloğa bölünür; self-play'de her blok bir rakiple oynar
    total_steps: int = 500_000_000   # toplam oyun adımı (tek rol)
    # ---- PPO ----
    epochs: int = 4                  # aynı veriyle kaç tur güncelleme
    n_minibatches: int = 8           # her turda batch kaç parçaya bölünür
    gamma: float = 0.995             # indirim: 50 Hz'de ufuk ≈ 1/(1−γ) = 200 adım = 4 s
    gae_lambda: float = 0.95         # GAE: avantajda kısa/uzun vade karışımı
    clip_eps: float = 0.2            # oran kırpması: her güncellemede en fazla ~%20 değiş
    vf_coef: float = 0.5             # critic kaybının ağırlığı
    ent_coef: float = 0.0            # entropi (keşif) bonusu; std zaten öğrenilen bir parametre
    lr: float = 3e-4
    anneal_lr: bool = True           # öğrenme oranı eğitim boyunca doğrusal olarak 0'a iner
    max_grad_norm: float = 0.5       # gradyan kırpma
    # KL ile erken durdurma (CleanRL / Stable-Baselines'taki target_kl). Bir turda aynı veriyle epochs × n_minibatches
    # adım atılıyor; her adımdan önce politikanın turun başından beri ne kadar değiştiğine (approx_kl) bakılır, bu
    # sınırı geçtiyse turun KALAN adımları atlanır. Neden: oran kırpması (clip_eps) tek tek örnekleri sınırlar ama
    # üst üste binen adımları sınırlamaz; sp-faz1-fix2-s0'da clip_frac 0,4'e, kl 0,05'e çıkıp kovalayan çöktü.
    # 0 = kapalı (2026-11-22 öncesi bütün koşular böyle).
    target_kl: float = 0.0
    hidden: int = 256
    init_log_std: float = -0.5       # başlangıç keşif gürültüsü: std = e^-0.5 ≈ 0.6
    squash_mean: bool = True         # eylem ortalaması tanh ile [-1, 1]'de; False = eski (2026-11-20 öncesi) davranış
    full_obs: bool = False           # True: aktör rakibi hep görür (kontrol deneyi / ablasyon 2), bkz. ChaseTagEnv
    # ---- kayıt ----
    seed: int = 0
    log_every: int = 5               # kaç turda bir satır yazdır
    ckpt_every: int = 50             # kaç turda bir checkpoint


# ---------------------------------------------------------------------------
# Ağlar: düz bir MLP, parametreler iç içe sözlük/liste (pytree). Kütüphane yok, her şey açıkta.
# ---------------------------------------------------------------------------
def init_mlp(key, sizes, last_scale):
    """Ortogonal başlatma: katmanların çıktı ölçeğini korur (derin ağlarda sinyal sönmez/patlamaz).
    Son katman ölçeği küçük (aktörde 0.01): başta bütün eylemler ~0 ve belirsizlik eşit; ajan bir yöne
    önyargılı başlamaz."""
    keys = jax.random.split(key, len(sizes) - 1)
    layers = []
    for i, (k, a, b) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        scale = last_scale if i == len(sizes) - 2 else np.sqrt(2)
        layers.append({"w": jax.nn.initializers.orthogonal(scale)(k, (a, b)), "b": jnp.zeros(b)})
    return layers


def mlp(layers, x):
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x @ layers[-1]["w"] + layers[-1]["b"]


def init_params(key, cfg):
    ka, kc = jax.random.split(key)
    return {
        "actor": init_mlp(ka, [OBS_DIM, cfg.hidden, cfg.hidden, 3], 0.01),   # gözlem → eylem ortalaması
        "log_std": jnp.full(3, cfg.init_log_std),                            # keşif gürültüsü (öğrenilir)
        "critic": init_mlp(kc, [CRITIC_DIM, cfg.hidden, cfg.hidden, 1], 1.0),  # tam durum → değer
    }


def actor_dist(params, obs, squash=True):
    """Politika π(a|s): ortalaması ağdan gelen, std'si öğrenilen Gauss dağılımı.

    squash=True: ortalama tanh ile her eksende [-1, 1]'e sınırlanır. Neden: ortam eylemin boyunu 1'e kırpıyor.
    Sınırsız ortalamayla eğitilen ilk politikalar ortalamayı ~4,5'e kaydırdı (boş arenada hep tam hız en iyiydi);
    std küçükken denenen bütün eylemler kırpılınca AYNI eyleme dönüşüyor, farklı sonuç çıkmıyor, avantaj sıfır
    oluyor ve ajan fren yapmayı öğrenemiyor (engelli self-play'de engellere tam hızla çarpma). Sınırlı ortalamada
    denemeler gerçekten farklı eylemler üretir. (Eski koşuların politikaları squash=False ile açılmalı.)"""
    mu = mlp(params["actor"], obs)
    return (jnp.tanh(mu) if squash else mu), jnp.exp(params["log_std"])


def log_prob(mu, std, a):
    """log π(a|s): 3 bağımsız Gauss boyutunun log yoğunluklarının toplamı."""
    return jnp.sum(-0.5 * ((a - mu) / std) ** 2 - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi), -1)


def entropy(std):
    """Gauss entropisi: std büyükse ajan 'kararsız', yani daha çok keşfediyor."""
    return jnp.sum(0.5 * jnp.log(2 * jnp.pi * jnp.e * std ** 2))


def critic_value(params, cobs):
    """V(s): bu durumdan itibaren beklenen (indirimli) toplam ödül."""
    return mlp(params["critic"], cobs)[..., 0]


# ---------------------------------------------------------------------------
# Rakipler: (eylem fonksiyonu, bellek başlatma, bellek sıfırlama)
# ---------------------------------------------------------------------------
#
# Rakip fonksiyonunun imzası: fn(es, mem, key, opp_args, obs_all) -> (eylem (n, 3), mem)
#   opp_args: rakibin değişen girdileri (self-play'de rakip ağların ağırlıkları); elle yazılmışlar için None
#   obs_all:  (n, 2, OBS_DIM) iki drone'un gözlemi; öğrenen için zaten hesaplandığından rakip tekrar hesaplamaz
def make_opponent(name):
    if name in scripted.EVADERS:
        fn = scripted.EVADERS[name]
        return (lambda es, mem, key, args, obs: fn(es, mem, key)), (lambda n: None), (lambda mem, done: mem)
    if name == "scripted_chaser":
        return ((lambda es, mem, key, args, obs: scripted.chaser_policy(es, mem, key)), scripted.chaser_init,
                scripted.reset_mem)
    raise ValueError(f"bilinmeyen rakip: {name}")


class Transition(NamedTuple):
    obs: jax.Array
    cobs: jax.Array
    action: jax.Array
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    outcome: jax.Array
    length: jax.Array
    crash_kind: jax.Array   # (n, 2) her drone nereye çarptı (geometry.CRASH_KIND_NAMES)


# ---------------------------------------------------------------------------
# Eğitim
# ---------------------------------------------------------------------------
def make_update(cfg, env, me, opponent=None):
    """Tek bir PPO turunu (veri topla + öğren) yapan, derlenmiş fonksiyonu üretir.
    opponent: (fn, init, reset) üçlüsü; verilmezse cfg.opponent'tan (elle yazılmış rakip)."""
    opp = 1 - me
    opp_fn, _, opp_reset = opponent or make_opponent(cfg.opponent)
    n, T = cfg.n_worlds, cfg.n_steps
    batch = n * T
    mb_size = batch // cfg.n_minibatches
    n_updates = cfg.total_steps // batch
    lr = (optax.linear_schedule(cfg.lr, 0.0, n_updates * cfg.epochs * cfg.n_minibatches)
          if cfg.anneal_lr else cfg.lr)
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr, eps=1e-5))

    # ---- 1. Veri toplama: bir oyun adımı ----
    def env_step(carry, key):
        params, es, mem, opp_args = carry
        k_act, k_opp, k_env = jax.random.split(key, 3)
        cobs_all = env.critic_obs(es)                            # (n, 2, CRITIC_DIM); ilk OBS_DIM'i aktör gözlemi
        obs_all = cobs_all[..., :OBS_DIM]
        obs, cobs = obs_all[:, me], cobs_all[:, me]
        mu, std = actor_dist(params, obs, cfg.squash_mean)
        a = mu + std * jax.random.normal(k_act, mu.shape)      # dağılımdan örnekle (keşif burada)
        a_opp, mem = opp_fn(es, mem, k_opp, opp_args, obs_all)
        actions = jnp.zeros((n, 2, 3)).at[:, me].set(a).at[:, opp].set(a_opp)
        es, reward, done, info = env.step(es, actions, k_env)
        mem = opp_reset(mem, done)
        tr = Transition(obs, cobs, a, log_prob(mu, std, a), critic_value(params, cobs), reward[:, me], done,
                        info["outcome"], info["length"], info["crash_kind"])
        return (params, es, mem, opp_args), tr

    # ---- GAE: avantajları sondan başa doğru hesapla ----
    def compute_gae(tr, last_value):
        """δ_t = r_t + γ·V(s_{t+1}) − V(s_t)      ("bir adımlık sürpriz": beklenenden ne kadar iyi gitti)
           A_t = δ_t + γλ·A_{t+1}                  (sürprizlerin indirimli toplamı)
        λ = 0 → sadece bir adım ileri bak (critic'e çok güven: az gürültü, ama critic yanılırsa yanlı)
        λ = 1 → oyunun sonuna kadar gerçek ödüller (yansız ama gürültülü). 0.95 ikisinin arası.
        Oyun bittiyse (done) sonraki durumun değeri sayılmaz: yeni oyun başka bir hikâye."""
        def step(carry, t):
            adv_next, v_next = carry
            live = 1.0 - t.done
            delta = t.reward + cfg.gamma * v_next * live - t.value
            adv = delta + cfg.gamma * cfg.gae_lambda * live * adv_next
            return (adv, t.value), adv
        _, adv = jax.lax.scan(step, (jnp.zeros(n), last_value), tr, reverse=True)
        return adv, adv + tr.value          # hedef getiri = avantaj + tahmin (critic bunu öğrenir)

    # ---- 2. Öğrenme: PPO kaybı ----
    def loss_fn(params, mb):
        obs, cobs, a, logp_old, v_old, adv, ret = mb
        mu, std = actor_dist(params, obs, cfg.squash_mean)
        logp = log_prob(mu, std, a)
        ratio = jnp.exp(logp - logp_old)                       # π_yeni / π_eski
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)          # normalize: ölçekten bağımsız, kararlı adım
        # Kırpılmış amaç: oran 1±ε dışına çıkınca daha fazla itmenin faydası kalmaz → küçük adımlar
        pg_loss = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv))
        # Critic: tahmini hedef getiriye yaklaştır (aynı mantıkla değişimi de kırpılmış)
        v = critic_value(params, cobs)
        v_clip = v_old + jnp.clip(v - v_old, -cfg.clip_eps, cfg.clip_eps)
        vf_loss = 0.5 * jnp.mean(jnp.maximum((v - ret) ** 2, (v_clip - ret) ** 2))
        ent = entropy(std)
        total = pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * ent
        aux = dict(pg_loss=pg_loss, vf_loss=vf_loss, entropy=ent,
                   approx_kl=jnp.mean((ratio - 1) - jnp.log(ratio)),          # politika ne kadar değişti
                   clip_frac=jnp.mean(jnp.abs(ratio - 1) > cfg.clip_eps))     # kırpmaya takılan oran
        return total, aux

    def update(params, opt_state, es, mem, key, opp_args=None):
        key, k_roll, k_upd = jax.random.split(key, 3)
        (_, es, mem, _), tr = jax.lax.scan(env_step, (params, es, mem, opp_args), jax.random.split(k_roll, T))
        last_value = critic_value(params, env.critic_obs(es)[:, me])
        adv, ret = compute_gae(tr, last_value)

        flat = lambda x: x.reshape((batch,) + x.shape[2:])
        data = (flat(tr.obs), flat(tr.cobs), flat(tr.action), flat(tr.logp), flat(tr.value), flat(adv), flat(ret))

        def epoch(carry, k):
            perm = jax.random.permutation(k, batch)
            mbs = jax.tree.map(lambda x: x[perm].reshape((cfg.n_minibatches, mb_size) + x.shape[1:]), data)

            def mb_step(carry, mb):
                params, opt_state, stop = carry
                (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, mb)
                # Erken durdurma: aux["approx_kl"] = bu adımdan ÖNCEKİ politikanın, veriyi toplayan politikadan
                # uzaklığı (bu minibatch üstünde). Sınır bir kez aşılınca turun geri kalanında stop hep True kalır.
                # jit içinde "break" olmadığı için adım yine hesaplanır ama uygulanmaz (eski değerler korunur).
                if cfg.target_kl > 0:
                    stop = stop | (aux["approx_kl"] > cfg.target_kl)
                updates, new_opt = tx.update(grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                keep = lambda old, new: jax.tree.map(lambda o, n_: jnp.where(stop, o, n_), old, new)
                aux = dict(aux, kl_skip=stop.astype(jnp.float32))           # bu adım atlandı mı (ölçüm)
                return (keep(params, new_params), keep(opt_state, new_opt), stop), aux
            return jax.lax.scan(mb_step, carry, mbs)
        (params, opt_state, _), aux = jax.lax.scan(epoch, (params, opt_state, jnp.array(False)),
                                                   jax.random.split(k_upd, cfg.epochs))

        # ---- ölçümler ----
        d = tr.done
        counts = jnp.sum(jax.nn.one_hot(tr.outcome, len(OUTCOME_NAMES)) * d[..., None], (0, 1))
        # Çarpma yerleri, rol başına: (2, 5) → [kovalayan, kaçan] × [-, engel, duvar, zemin, tavan]
        crash_counts = jnp.sum(jax.nn.one_hot(tr.crash_kind, len(CRASH_KIND_NAMES)), (0, 1))
        # Blok başına sonuç sayıları (n_blocks, 6): self-play'de hangi rakibe karşı ne olduğunu ayırmak için.
        # Bloklar ardışık: dünya w, w // (n / n_blocks) numaralı blokta.
        block_counts = jnp.sum((jax.nn.one_hot(tr.outcome, len(OUTCOME_NAMES)) * d[..., None])
                               .reshape(T, cfg.n_blocks, n // cfg.n_blocks, len(OUTCOME_NAMES)), (0, 2))
        var_ret = jnp.var(ret)
        # ---- gaz pedalı (doyma) ölçüleri ----
        # Ortam eylemin boyunu 1'e kırpıyor. Denenen eylemlerin çoğu kırpılıyorsa denemeler aynı eyleme dönüşür.
        a_norm = jnp.linalg.norm(tr.action, axis=-1)
        # tanh'tan ÖNCEKİ ham ortalama: |x| > 2,65 ise tanh(x) > 0,99, yani o eksende eğim ~0 ve ajan
        # "tam gaz" alışkanlığından ancak çok yavaş vazgeçebilir. (Gözlemlerin bir örneği üstünde ölçülür.)
        pre = mlp(params["actor"], data[0][:8192])
        metrics_sat = dict(
            act_clip_frac=jnp.mean(a_norm > 1.0),              # denenen eylemlerin kırpılan oranı
            act_speed=jnp.mean(jnp.minimum(a_norm, 1.0)),      # komut edilen ortalama hız (1 = tam gaz)
            pre_abs=jnp.mean(jnp.abs(pre)),                    # ham ortalamanın büyüklüğü
            pre_sat_frac=jnp.mean(jnp.abs(pre) > 2.65),         # tanh'ın düz bölgesindeki eksen oranı
        )
        metrics = {k: v.mean() for k, v in aux.items()}
        metrics.update(metrics_sat)
        metrics.update(
            outcome_counts=counts,
            crash_counts=crash_counts,
            block_counts=block_counts,
            ep_len=jnp.sum(tr.length * d) / jnp.maximum(d.sum(), 1),
            reward_per_step=tr.reward.mean(),
            value_mean=tr.value.mean(),
            # Critic ne kadar iyi tahmin ediyor? 1 = mükemmel, 0 = sabit tahmin kadar, <0 = ondan da kötü
            explained_var=1.0 - jnp.var(ret - tr.value) / (var_ret + 1e-8),
            std=jnp.exp(params["log_std"]).mean(),
        )
        return params, opt_state, es, mem, key, metrics

    return jax.jit(update), tx, n_updates


def learner_win_rate(counts, me, reward_table):
    """Öğrenen rolün kazandığı biten oyunların oranı: sonuç ödülü pozitifse kazanmış sayılır
    (opp_crash_reward = 0 iken rakibi çarptırmak kazanma sayılmaz)."""
    total = counts[1:].sum()
    wins = sum(counts[k] for k in range(1, len(counts)) if reward_table[k, me] > 0)
    return wins / max(total, 1)


def _crash_summary(crash_counts, role):
    c = crash_counts[role, 1:]
    tot = c.sum()
    if tot == 0:
        return "-"
    return " ".join(f"{CRASH_KIND_NAMES[i + 1][:3]}{100 * c[i] / tot:.0f}" for i in range(len(c)) if c[i])


def _save(path, obj):
    with open(path, "wb") as f:
        pickle.dump(jax.device_get(obj), f)


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def train(cfg, out_dir, resume=True, log=print):
    """Eğitimi çalıştırır. out_dir'e config.json, metrics.jsonl ve checkpoint'ler yazılır.
    resume=True ve out_dir'de checkpoint varsa kaldığı yerden devam eder (Colab bağlantısı koparsa)."""
    os.makedirs(out_dir, exist_ok=True)
    log_file = open(os.path.join(out_dir, "train_log.txt"), "a", encoding="utf-8")
    screen = log

    def log(msg):  # ekrana ve train_log.txt'ye (makale için her satır saklansın)
        screen(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    me = CHASER if cfg.learner == "chaser" else EVADER
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    log(f"arenalar üretiliyor ({cfg.n_arenas}, faz {cfg.phase}, engel {cfg.n_obst})...")
    arenas = make_arena_set(cfg.n_arenas, phase=cfg.phase, n_obst=tuple(cfg.n_obst), seed=cfg.seed, verbose=False)
    env = ChaseTagEnv(cfg.n_worlds, arenas, opp_crash_reward=cfg.opp_crash_reward, full_obs=cfg.full_obs)
    update, tx, n_updates = make_update(cfg, env, me)
    _, opp_init, _ = make_opponent(cfg.opponent)

    key = jax.random.PRNGKey(cfg.seed)
    key, k_init, k_env = jax.random.split(key, 3)
    params, opt_state, start = init_params(k_init, cfg), None, 0
    ckpt = os.path.join(out_dir, "checkpoint.pkl")
    if resume and os.path.exists(ckpt):
        c = _load(ckpt)
        params, opt_state, start, key = c["params"], c["opt_state"], c["update"], c["key"]
        log(f"checkpoint'ten devam: tur {start}")
    if opt_state is None:
        opt_state = tx.init(params)
    es, mem = env.reset(k_env), opp_init(cfg.n_worlds)

    batch = cfg.n_worlds * cfg.n_steps
    log(f"{n_updates} tur × {batch:,} adım = {n_updates * batch / 1e6:.0f} M adım | öğrenen: {cfg.learner}, "
        f"rakip: {cfg.opponent}")
    mfile = open(os.path.join(out_dir, "metrics.jsonl"), "a")
    t0, window = time.perf_counter(), np.zeros(len(OUTCOME_NAMES))
    crash_window = np.zeros((2, len(CRASH_KIND_NAMES)))
    for u in range(start, n_updates):
        params, opt_state, es, mem, key, m = update(params, opt_state, es, mem, key)
        m = jax.device_get(m)
        window += m["outcome_counts"]
        crash_window += m["crash_counts"]
        row = {k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in m.items()}
        row.update(update=u + 1, steps=(u + 1) * batch, time_s=time.perf_counter() - t0)
        mfile.write(json.dumps(row) + "\n")
        if (u + 1) % cfg.log_every == 0 or u == start:
            mfile.flush()
            total = window[1:].sum()
            pct = lambda k: 100 * window[k] / max(total, 1)
            sps = (u + 1 - start) * batch / (time.perf_counter() - t0)
            log(f"tur {u + 1:4d} | {(u + 1) * batch / 1e6:6.0f} M adım | {sps / 1e6:.2f} M/s | maç {int(total):6d} | "
                f"kazanma %{100 * learner_win_rate(window, me, env.reward_table):5.1f} | ebeleme %{pct(TAG):4.1f} "
                f"süre %{pct(TIMEOUT):4.1f} K.çarp %{pct(C_CRASH):4.1f} E.çarp %{pct(E_CRASH):4.1f} | "
                f"süre {m['ep_len'] / CTRL_HZ:4.1f} s | std {m['std']:.2f} kl {m['approx_kl']:.4f} "
                f"atlanan %{100 * m['kl_skip']:.0f} "
                f"ev {m['explained_var']:.2f} | gaz: hız {m['act_speed']:.2f} kırpılan %{100 * m['act_clip_frac']:.0f} "
                f"düz %{100 * m['pre_sat_frac']:.0f} | çarpma yeri K[{_crash_summary(crash_window, CHASER)}] "
                f"E[{_crash_summary(crash_window, EVADER)}]")
            window[:] = 0
            crash_window[:] = 0
        if (u + 1) % cfg.ckpt_every == 0 or u + 1 == n_updates:
            _save(ckpt, dict(params=params, opt_state=opt_state, update=u + 1, key=key))
            _save(os.path.join(out_dir, f"params_{u + 1:05d}.pkl"), params)
    mfile.close()
    log_file.close()
    return params


# ---------------------------------------------------------------------------
# Eğitilmiş politikayı oynatmak için (play.make_runner arayüzü)
# ---------------------------------------------------------------------------
def learned_vs_opponent(params, learner, opponent, env, deterministic=True, squash=True):
    """policy(es, mem, key) -> ((n, 2, 3), mem): öğrenen rol ağla, diğeri rakip fonksiyonuyla oynar.
    deterministic: değerlendirmede gürültüsüz (ortalama eylem) oynat. squash: koşunun squash_mean ayarı."""
    me = CHASER if learner == "chaser" else EVADER
    opp_fn, opp_init, opp_reset = make_opponent(opponent)

    def policy(es, mem, key):
        k1, k2 = jax.random.split(key)
        mu, std = actor_dist(params, env.observe(es)[:, me], squash)
        a = mu if deterministic else mu + std * jax.random.normal(k1, mu.shape)
        a_opp, mem = opp_fn(es, mem, k2, None, None)
        n = a.shape[0]
        return jnp.zeros((n, 2, 3)).at[:, me].set(a).at[:, 1 - me].set(a_opp), mem

    return policy, opp_init, opp_reset
