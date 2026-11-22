"""Self-play testleri: kontrol maçları her dünyada tam bir maç sayıyor mu?"""
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from chase_tag.arenas import pack_layout
from chase_tag.env import ChaseTagEnv
from chase_tag.rules import TIMEOUT
from chase_tag.ppo import make_opponent
from chase_tag.rules import C_CRASH, CHASER, E_CRASH, EVADER
from chase_tag.selfplay import (SelfPlayConfig, Pool, anchor_reward_adjust, make_match_eval, make_pool_opponent,
                                role_ppo_cfg, set_std)

N = 4


def test_match_eval_counts_one_full_game_per_world():
    """İki taraf da yerinde dursun (ağ çıktısı 0, gürültü yok): boş arenada karşı köşelerde kimse kimseyi
    yakalayamaz, yani HER maç süre dolarak bitmeli. Eğitim maçlarındaki sayım hatası (süre dolan maçların
    sadece belli turlarda görünmesi) burada olamaz: her dünya baştan sona tam bir maç oynar."""
    with open("experiments/ppo-chaser-empty-fix-s0/params_00476.pkl", "rb") as f:
        template = pickle.load(f)
    still = jax.tree.map(jnp.zeros_like, template)        # bütün ağırlıklar 0 → ortalama eylem tanh(0) = 0
    still["log_std"] = jnp.full_like(still["log_std"], -20.0)   # std ≈ 0: gürültüsüz
    packed = pack_layout(np.zeros((0, 2, 3)), np.zeros((0, 5)))
    arenas = {k: v[None] for k, v in packed.items()}       # tek boş arena
    first = np.asarray(make_match_eval(ChaseTagEnv(N, arenas))(still, still, jax.random.PRNGKey(0)))
    assert first.shape == (N,)
    assert (first == TIMEOUT).all(), first


def test_anchor_blocks_use_scripted_opponent():
    """Çapa: son blok(lar)daki dünyalarda rakibin hareketi elle yazılmış rakipten gelir, diğerlerinde ağdan."""
    n, n_blocks = 8, 4                                       # blok boyu 2; son 1 blok = dünya 6 ve 7
    packed = pack_layout(np.zeros((0, 2, 3)), np.zeros((0, 5)))
    env = ChaseTagEnv(n, {k: v[None] for k, v in packed.items()})
    with open("experiments/ppo-evader-empty-fix-s0/params_00476.pkl", "rb") as f:
        p_ev = jax.tree.map(jnp.asarray, pickle.load(f))
    pool = Pool(p_ev, 2); pool.add(p_ev)
    fn, init, _ = make_pool_opponent(n, n_blocks, EVADER, True, anchor_blocks=1, anchor_name="open")
    es, key = env.reset(jax.random.PRNGKey(0)), jax.random.PRNGKey(1)
    a, _ = fn(es, init(n), key, (pool.arr, jnp.zeros(n_blocks, jnp.int32)), env.observe(es))
    s_fn, _, _ = make_opponent("open")
    a_s, _ = s_fn(es, None, jax.random.split(key)[1], None, None)
    assert np.allclose(np.asarray(a[6:]), np.asarray(a_s[6:]))          # çapa dünyaları: elle yazılmış kaçan
    assert not np.allclose(np.asarray(a[:6]), np.asarray(a_s[:6]))      # diğerleri: ağ


def test_anchor_reward_adjust():
    """Çapa dünyalarında 'rakip çarptı' ödülü anchor_ocr olur (WCT tablosunda +1); başka hiçbir şey değişmez."""
    table = ChaseTagEnv(4, {k: v[None] for k, v in pack_layout(np.zeros((0, 2, 3)), np.zeros((0, 5))).items()},
                        opp_crash_reward=1.0).reward_table
    adj = anchor_reward_adjust(8, 4, 1, CHASER, table, 0.0)            # kovalayan öğreniyor: rakip = kaçan
    assert np.allclose(adj[6:, E_CRASH], -1.0) and np.count_nonzero(adj) == 2
    adj = anchor_reward_adjust(8, 4, 1, EVADER, table, 0.0)            # kaçan öğreniyor: rakip = kovalayan
    assert np.allclose(adj[6:, C_CRASH], -1.0) and np.count_nonzero(adj) == 2
    assert np.count_nonzero(anchor_reward_adjust(8, 4, 0, CHASER, table, 0.0)) == 0


def test_evader_entropy_override():
    """ent_coef_evader sadece kaçanı değiştirir; verilmezse (<0) iki rol de ent_coef kullanır."""
    sp = SelfPlayConfig(ent_coef=0.001, ent_coef_evader=0.003)
    assert role_ppo_cfg(sp, "evader").ent_coef == 0.003
    assert role_ppo_cfg(sp, "chaser").ent_coef == 0.001
    sp = SelfPlayConfig(ent_coef=0.001)
    assert role_ppo_cfg(sp, "evader").ent_coef == 0.001


def test_evader_min_std_only_for_evader():
    """evader_min_std sadece kaçanın PPO ayarına geçer; kovalayan sınırsız kalır."""
    sp = SelfPlayConfig(evader_min_std=0.1)
    assert role_ppo_cfg(sp, "evader").min_std == 0.1
    assert role_ppo_cfg(sp, "chaser").min_std == 0.0
    assert role_ppo_cfg(SelfPlayConfig(), "evader").min_std == 0.0


def test_set_std_changes_only_noise():
    """set_std: std istenen değer olur, ağın ağırlıkları (ortalama eylem) aynı kalır."""
    with open("experiments/ppo-evader-faz1-long-s0/params_01430.pkl", "rb") as f:
        p = jax.tree.map(jnp.asarray, pickle.load(f))
    q = set_std(p, 0.2)
    assert np.allclose(np.exp(np.asarray(q["log_std"])), 0.2)
    for a, b in zip(jax.tree.leaves(p["actor"]), jax.tree.leaves(q["actor"])):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.allclose(np.exp(np.asarray(p["log_std"])), 0.2)     # orijinal değişmedi
