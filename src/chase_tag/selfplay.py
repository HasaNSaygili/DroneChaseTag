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

Sıra değişimi: öğrenen EN SON rakibe karşı ≥ `switch_win` kazanınca (en az `min_steps` adımdan sonra) ya da
`max_steps` dolunca. Kazanma oranı AYRI KONTROL MAÇLARIYLA ölçülür (`match_eval`): `eval_worlds` maç sıfırdan
başlar, her biri sonuna kadar oynanır, her maç tam bir kez sayılır.

Neden eğitim maçlarından ölçülmüyor (2026-11-21'e kadar öyleydi)? Nesil başında bütün dünyalar AYNI ANDA başlıyor.
Yakalanmalar dağınık zamanlarda biter, ama süre dolan maçların hepsi aynı turda (1000. adım) biter. "Son birkaç
turda biten maçlar" sayılınca bu patlamanın denk geldiği pencere kaçanın oranını şişirir (sp-empty-fix-s0: log
%71–87, gerçek ~%50), denk gelmeyen pencere sıfırlar. Eğitim içi sonuç oranları (logdaki ebeleme/süre yüzdeleri,
PFSP'nin rakip başına `ema`'sı) bu yanlılığı hâlâ taşır: log sadece gösterge, PFSP ağırlığı yaklaşık.

Uygulama ayrıntısı: arenalar `n_blocks` bloğa bölünür, her blok bir rakip ağla oynar. Rakip ağırlıkları
argüman olarak geçtiği için havuz büyüdükçe yeniden derleme olmaz (havuz `pool_max` boyutlu sabit dizi).
PFSP blokları her `resample_every` turda yeniden çekilir; maçlar turlara yayıldığı için bir maçın ortasında
rakip değişebilir (yaklaşıklık; en son rakip blokları hiç değişmez).
"""
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
from .eval import make_pair_runner, stack
from .geometry import CRASH_KIND_NAMES
from .ppo import PPOConfig, _crash_summary, actor_dist, make_opponent, make_update
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
    target_kl: float = 0.0                # >0: KL ile erken durdurma (bkz. PPOConfig.target_kl); 0 = kapalı
    warmup_steps: int = 0                 # >0: lr ısınması, ilk bu kadar optimizer adımı (bkz. PPOConfig); 0 = kapalı
    hidden: int = 256
    squash_mean: bool = True             # eylem ortalaması tanh ile sınırlı (bkz. ppo.actor_dist)
    full_obs: bool = False                # True: iki taraf da rakibi hep görür (ablasyon 2)
    gamma_chaser: float = 0.995           # ufuk ~4 s
    gamma_evader: float = 0.998           # ufuk ~10 s (kaçanın ödülü 20 s sonunda)
    # ---- kaçanın keşfi (2026-11-22, sp-faz1-lr1e4-s0'da kaçanın std'si 0,059 → 0,028'e çöktü, kovalayan baskın) ----
    # Engelli temel kaçan self-play'e düşük gürültüyle başlıyor (std 0,064; boş arenadaki 0,22). Gürültü azalınca yeni
    # hareket denenmiyor, yeni taktik bulunamıyor. İki ayrı deneme için iki ayrı ayar:
    ent_coef_evader: float = -1.0         # ≥0: kaçanın entropi bonusu bu olur (kovalayan ent_coef'te kalır); <0 = ent_coef
    evader_init_std: float = 0.0          # >0: self-play BAŞINDA öğrenen kaçanın std'si bu değere ayarlanır (ağın
    #                                       ortalaması aynı kalır). Havuzdaki kaçan_0 değişmez. 0 = kapalı
    # ---- self-play ----
    init_chaser: str = "experiments/ppo-chaser-empty-nocrashrew-s0/params_00476.pkl"   # kovalayan_0
    init_evader: str = "experiments/ppo-evader-empty-nocrashrew-s0/params_00476.pkl"   # kaçan_0
    first_learner: str = "chaser"         # kaçan_0 daire çiziyor; önce kovalayan cevap versin
    n_gens: int = 20                      # toplam nesil (yarısı kovalayan, yarısı kaçan)
    min_steps: int = 20_000_000           # bir nesilde en az bu kadar adım (gürültüyle erken geçişi önler)
    max_steps: int = 100_000_000          # bir nesilde en fazla (oyun yapısal olarak dengesizse takılmasın)
    switch_win: float = 0.6               # en son rakibe karşı bu orana ulaşınca sıra değişir
    eval_every: int = 5_000_000           # min_steps'ten sonra kaç adımda bir kontrol maçı yapılır
    eval_worlds: int = 4096               # kontrol maçı sayısı (her dünya 1 maç; ±%1,5 belirsizlik)
    p_latest: float = 0.5                 # blokların bu kadarı hep en son rakiple oynar
    resample_every: int = 10              # PFSP blokları kaç turda bir yeniden çekilir
    pool_max: int = 32
    win_ema: float = 0.1                  # rakip başına kazanma oranının üstel ortalama katsayısı
    # ---- çapa rakipler (2026-11-22): unutmaya karşı ----
    # Sadece öğrenen ajanlardan oluşan havuzda iki taraf birbirine karşı DAR UZMANLAŞIYOR: boş ve engelli arenada
    # self-play kaçanları elle yazılmış kovalayana karşı %100/%63 → %0 düştü (unutma testi, scripts/vs_scripted.py).
    # Çapa: son anchor_blocks blok HER ZAMAN elle yazılmış bir rakiple oynar (öğrenen kovalayan → anchor_vs_chaser,
    # öğrenen kaçan → anchor_vs_evader). Ajan hem birbirine hem dış rakibe karşı iyi kalmak zorunda.
    anchor_blocks: int = 0                # 0 = kapalı (2026-11-22 öncesi bütün koşular)
    anchor_vs_chaser: str = "open"        # öğrenen kovalayanın çapa rakibi (elle yazılmış kaçan)
    anchor_vs_evader: str = "scripted_chaser"   # öğrenen kaçanın çapa rakibi (elle yazılmış kovalayan)
    anchor_opp_crash_reward: float = 0.0  # çapa maçında rakip çarparsa öğrenenin ödülü. 0: elle yazılmış rakibi
    #                                       çarptırmak ödül getirmez (yoksa rezonans hilesi geri gelir, bkz. LOG
    #                                       ppo-chaser-empty-s0); temel ajanlar da böyle eğitildi
    seed: int = 0
    log_every: int = 5


def role_ppo_cfg(sp, role):
    """Bir rol için PPO ayarları (ppo.make_update bunları kullanır)."""
    keys = ("phase", "n_obst", "n_arenas", "opp_crash_reward", "n_worlds", "n_steps", "n_blocks", "epochs",
            "n_minibatches", "gae_lambda", "clip_eps", "vf_coef", "ent_coef", "lr", "max_grad_norm", "target_kl", "warmup_steps", "hidden",
            "squash_mean", "full_obs", "seed")
    kw = {k: getattr(sp, k) for k in keys}
    if role == "evader" and sp.ent_coef_evader >= 0:
        kw["ent_coef"] = sp.ent_coef_evader
    return PPOConfig(learner=role, opponent="pool", anneal_lr=False,
                     gamma=sp.gamma_chaser if role == "chaser" else sp.gamma_evader, **kw)


def set_std(params, std):
    """Politikanın keşif gürültüsünü (std) değiştirir, ortalamaya (ağın kendisine) dokunmaz.
    log_std ağdan bağımsız, öğrenilen bir parametre: bütün eksenlere aynı değeri yazıyoruz."""
    return dict(params, log_std=jnp.full_like(params["log_std"], jnp.log(std)))


def make_pool_opponent(n, n_blocks, opp_role, squash=True, anchor_blocks=0, anchor_name=None):
    """Havuzdaki donmuş ağlarla oynayan rakip. opp_args = (havuz dizisi, blok başına havuz indeksi).
    anchor_blocks > 0: son anchor_blocks bloktaki dünyalarda rakibi ağ değil elle yazılmış anchor_name oynar
    (o blokların slot'u önemsiz). Elle yazılmış rakip basitlik için bütün dünyalarda hesaplanıp sadece çapa
    dünyalarında kullanılır; elle yazılmış kovalayanın hafızası (mem) bu yüzden bütün dünyalar için tutulur."""
    bs = n // n_blocks
    if anchor_blocks:
        s_fn, s_init, s_reset = make_opponent(anchor_name)
        is_anchor = (jnp.arange(n) >= n - anchor_blocks * bs)[:, None]    # (n, 1) çapa dünyası mı
    else:
        s_init, s_reset = (lambda n: None), (lambda mem, done: mem)

    def fn(es, mem, key, opp_args, obs_all):
        pool, slot = opp_args
        k_net, k_s = jax.random.split(key)
        p = jax.tree.map(lambda x: x[slot], pool)                          # (B, ...) her bloğun rakip ağı
        obs = obs_all[:, opp_role].reshape(n_blocks, bs, OBS_DIM)
        mu, std = jax.vmap(lambda pp, oo: actor_dist(pp, oo, squash))(p, obs)   # (B, bs, 3), (B, 3)
        a = (mu + std[:, None, :] * jax.random.normal(k_net, mu.shape)).reshape(n, 3)   # rakip de olasılıklı oynar
        if anchor_blocks:
            a_s, mem = s_fn(es, mem, k_s, None, None)
            a = jnp.where(is_anchor, a_s, a)
        return a, mem

    return fn, s_init, s_reset


def anchor_reward_adjust(n, n_blocks, anchor_blocks, me, table, anchor_ocr):
    """make_update'in reward_adjust'u: çapa dünyalarında 'rakip çarptı' sonucunun öğrenen ödülünü anchor_ocr yapar.
    Döner (n, n_sonuç): ödüle eklenecek fark (çapa dışı dünyalarda 0). Ortamın tablosu WCT kuralı (rakip çarparsa +1)."""
    adj = np.zeros((n, len(OUTCOME_NAMES)), np.float32)
    if anchor_blocks:
        opp_crash = E_CRASH if me == CHASER else C_CRASH                    # öğrenenin RAKİBİ çarptı
        adj[n - anchor_blocks * (n // n_blocks):, opp_crash] = anchor_ocr - table[opp_crash, me]
    return adj


def make_match_eval(env, squash=True):
    """Kontrol maçları: run(p_kovalayan, p_kaçan, key) -> (n,) her dünyanın İLK maçının sonucu.

    Bütün dünyalar sıfırdan başlar ve tam MAX_STEPS adım oynanır; süre sınırı yüzünden her maç en geç son adımda
    biter, yani her dünya tam bir maç verir (sonrası otomatik yeniden başlar, sayılmaz). İki taraf da eğitimdeki
    gibi olasılıklı oynar: ölçülen şey, eğitilen politikanın gerçekten kazandığı oran.
    (eval.make_pair_runner'ın tek çiftlik hali: bütün dünyalar aynı çifti oynar.)"""
    run = make_pair_runner(env, 1, squash)
    return lambda p_ch, p_ev, key: run(stack([p_ch]), stack([p_ev]), key)


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
    env = ChaseTagEnv(sp.n_worlds, arenas, opp_crash_reward=sp.opp_crash_reward, full_obs=sp.full_obs)
    # Kontrol maçları için ayrı, küçük ortam (aynı arenalar, aynı kurallar)
    match_eval = make_match_eval(ChaseTagEnv(sp.eval_worlds, arenas, opp_crash_reward=sp.opp_crash_reward,
                                             full_obs=sp.full_obs), sp.squash_mean)
    role_idx = {"chaser": CHASER, "evader": EVADER}
    cfgs = {r: role_ppo_cfg(sp, r) for r in ROLES}
    n_latest = max(1, int(round(sp.n_blocks * sp.p_latest)))
    assert n_latest + sp.anchor_blocks <= sp.n_blocks, "en son rakip + çapa blokları blok sayısını aşıyor"
    anchor_name = {"chaser": sp.anchor_vs_chaser, "evader": sp.anchor_vs_evader}
    updates, txs, opp_inits = {}, {}, {}
    for r in ROLES:
        other = role_idx["evader" if r == "chaser" else "chaser"]
        opponent = make_pool_opponent(sp.n_worlds, sp.n_blocks, other, sp.squash_mean, sp.anchor_blocks, anchor_name[r])
        opp_inits[r] = opponent[1]
        adj = anchor_reward_adjust(sp.n_worlds, sp.n_blocks, sp.anchor_blocks, role_idx[r], env.reward_table,
                                   sp.anchor_opp_crash_reward) if sp.anchor_blocks else None
        updates[r], txs[r], _ = make_update(cfgs[r], env, role_idx[r], opponent=opponent, reward_adjust=adj)
    if sp.anchor_blocks:
        log(f"çapa: son {sp.anchor_blocks}/{sp.n_blocks} blok elle yazılmış rakiple (kovalayan öğrenirken "
            f"{sp.anchor_vs_chaser}, kaçan öğrenirken {sp.anchor_vs_evader}); çapa maçında rakip çarpması ödülü "
            f"{sp.anchor_opp_crash_reward}")

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
        if sp.evader_init_std > 0:          # sadece öğrenen kaçan; havuza aşağıda değişmemiş init[...] girer
            params["evader"] = set_std(params["evader"], sp.evader_init_std)
            log(f"kaçanın başlangıç std'si {float(jnp.exp(init['evader']['log_std']).mean()):.3f} → {sp.evader_init_std}")
        opt = {r: txs[r].init(params[r]) for r in ROLES}
        pools = {r: Pool(init[r], sp.pool_max) for r in ROLES}
        for r in ROLES:
            pools[r].add(init[r])                                  # havuz: kovalayan_0 ve kaçan_0
        ema = {r: np.full(sp.pool_max, 0.5) for r in ROLES}        # ema[r][j]: r'nin, rakip havuzu j'ye karşı kazanması
        gen0, total = 0, 0

    batch = sp.n_worlds * sp.n_steps
    n_net = sp.n_blocks - sp.anchor_blocks                          # ağ rakipli bloklar: [0, n_latest) en son, sonra PFSP
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
        es, mem = env.reset(k_env), opp_inits[role](sp.n_worlds)       # çapa elle yazılmış kovalayansa hafızası
        slot = np.full(sp.n_blocks, latest, np.int32)                    # çapa bloklarının slot'u kullanılmaz
        opp_latest = jax.tree.map(lambda x: x[latest], pool.arr)      # kontrol maçlarının rakibi
        win_eval, eval_pct, last_eval = None, None, -np.inf
        window = np.zeros(len(OUTCOME_NAMES))
        crash_window = np.zeros((2, len(CRASH_KIND_NAMES)))
        anchor_window = np.zeros(len(OUTCOME_NAMES))                     # çapa bloklarının sonuçları (log için)
        steps, u, t_gen = 0, 0, time.perf_counter()
        while True:
            if u % sp.resample_every == 0:                           # PFSP bloklarını yeniden çek
                w = (1.0 - ema[role][:pool.n]) ** 2
                w = w / w.sum() if w.sum() > 0 else np.full(pool.n, 1.0 / pool.n)
                slot[n_latest:n_net] = rng.choice(pool.n, size=n_net - n_latest, p=w)
            params[role], opt[role], es, mem, key, m = updates[role](
                params[role], opt[role], es, mem, key, (pool.arr, jnp.asarray(slot)))
            m = jax.device_get(m)
            steps += batch
            total += batch
            u += 1
            bc = m["block_counts"]                                   # (B, 6)
            wins_b, ended_b = bc[:, win_k].sum(1), bc[:, 1:].sum(1)
            net = np.arange(sp.n_blocks) < n_net                     # çapa blokları PFSP hesabına girmez
            for j in np.unique(slot[net]):                           # rakip başına kazanma oranını güncelle (PFSP için)
                sel = (slot == j) & net
                if ended_b[sel].sum() > 0:
                    ema[role][j] = (1 - sp.win_ema) * ema[role][j] + sp.win_ema * wins_b[sel].sum() / ended_b[sel].sum()
            # ---- kontrol maçları: en az adımdan sonra her eval_every adımda bir (ve en fazla adımda son kez) ----
            evaled = False
            if steps >= sp.min_steps and (steps - last_eval >= sp.eval_every or steps >= sp.max_steps):
                key, k_eval = jax.random.split(key)
                p_ch, p_ev = (params[role], opp_latest) if role == "chaser" else (opp_latest, params[role])
                first = np.asarray(match_eval(p_ch, p_ev, k_eval))
                first = first[first > 0]                             # (süre sınırı yüzünden hepsi biter)
                win_eval = float(np.isin(first, win_k).mean())
                eval_pct = {OUTCOME_NAMES[k]: 100 * float(np.mean(first == k)) for k in range(1, len(OUTCOME_NAMES))}
                last_eval, evaled = steps, True
            window += m["outcome_counts"]
            crash_window += m["crash_counts"]
            anchor_window += bc[n_net:].sum(0)
            row = dict(gen=gen, learner=role, update=u, gen_steps=steps, total_steps=total,
                       time_s=time.perf_counter() - t_start, slot=slot.tolist(),
                       win_latest_eval=win_eval if evaled else None, eval_pct=eval_pct if evaled else None,
                       ema=ema[role][:pool.n].tolist(),
                       **{k: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for k, v in m.items()})
            mfile.write(json.dumps(row) + "\n")
            done_gen = steps >= sp.max_steps or (evaled and win_eval >= sp.switch_win)
            if evaled:
                log(f"nesil {gen:2d} kontrol maçı ({len(first)} maç, en son {ROLE_TR[other]}_{latest}'e karşı): "
                    f"kazanma %{100 * win_eval:.1f} | " + " ".join(f"{k} %{v:.1f}" for k, v in eval_pct.items()))
            if u % sp.log_every == 0 or done_gen:
                mfile.flush()
                tot = window[1:].sum()
                pct = lambda k: 100 * window[k] / max(tot, 1)
                sps = steps / (time.perf_counter() - t_gen)
                log(f"nesil {gen:2d} {ROLE_TR[role]:9s} tur {u:3d} | {steps / 1e6:4.0f} M | {sps / 1e6:.2f} M/s | "
                    f"havuz ort. %{100 * ema[role][:pool.n].mean():5.1f} | eğitim maçları: "
                    f"ebeleme %{pct(TAG):4.1f} süre %{pct(TIMEOUT):4.1f} K.çarp %{pct(C_CRASH):4.1f} "
                    f"E.çarp %{pct(E_CRASH):4.1f} | süre {m['ep_len'] / CTRL_HZ:4.1f} s | std {m['std']:.2f} "
                    f"kl {m['approx_kl']:.4f} clip %{100 * m['clip_frac']:.0f} atlanan %{100 * m['kl_skip']:.0f} | "
                    f"gaz: hız {m['act_speed']:.2f} kırpılan %{100 * m['act_clip_frac']:.0f} "
                    f"düz %{100 * m['pre_sat_frac']:.0f} | çarpma K[{_crash_summary(crash_window, CHASER)}] "
                    f"E[{_crash_summary(crash_window, EVADER)}]"
                    + (f" | çapa ({anchor_name[role]}): ebeleme %{100 * anchor_window[TAG] / max(anchor_window[1:].sum(), 1):.0f}"
                       f" süre %{100 * anchor_window[TIMEOUT] / max(anchor_window[1:].sum(), 1):.0f}"
                       if sp.anchor_blocks else ""))
                window[:] = 0
                crash_window[:] = 0
                anchor_window[:] = 0
            if done_gen:
                break
        # ---- nesil sonu: öğreneni havuzuna ekle, kaydet ----
        why = "eşik" if win_eval >= sp.switch_win else "en fazla adım"
        j = pools[role].add(params[role])
        ema[other][j] = 0.5                                          # yeni rakip: henüz bilinmiyor
        _save(os.path.join(out_dir, f"{role}_gen{gen:02d}.pkl"), params[role])
        log(f"--- nesil {gen} bitti ({why}): {ROLE_TR[role]}, {steps / 1e6:.0f} M adım, "
            f"{(time.perf_counter() - t_gen) / 60:.1f} dk, kontrol maçında en son rakibe karşı %{100 * win_eval:.1f} → "
            f"havuza {ROLE_TR[role]}_{j} olarak eklendi")
        _save(ckpt, dict(params=params, opt=opt, pool={r: pools[r].arr for r in ROLES},
                         pool_n={r: pools[r].n for r in ROLES}, ema=ema, gen=gen + 1, key=key, total_steps=total))
    mfile.close()
    log(f"\nbitti: {total / 1e6:.0f} M adım, {(time.perf_counter() - t_start) / 60:.1f} dk")
    log_file.close()
    return params, pools
