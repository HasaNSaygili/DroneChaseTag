"""Self-play testleri: kontrol maçları her dünyada tam bir maç sayıyor mu?"""
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from chase_tag.arenas import pack_layout
from chase_tag.env import ChaseTagEnv
from chase_tag.rules import TIMEOUT
from chase_tag.selfplay import SelfPlayConfig, make_match_eval, role_ppo_cfg, set_std

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


def test_evader_entropy_override():
    """ent_coef_evader sadece kaçanı değiştirir; verilmezse (<0) iki rol de ent_coef kullanır."""
    sp = SelfPlayConfig(ent_coef=0.001, ent_coef_evader=0.003)
    assert role_ppo_cfg(sp, "evader").ent_coef == 0.003
    assert role_ppo_cfg(sp, "chaser").ent_coef == 0.001
    sp = SelfPlayConfig(ent_coef=0.001)
    assert role_ppo_cfg(sp, "evader").ent_coef == 0.001


def test_set_std_changes_only_noise():
    """set_std: std istenen değer olur, ağın ağırlıkları (ortalama eylem) aynı kalır."""
    with open("experiments/ppo-evader-faz1-long-s0/params_01430.pkl", "rb") as f:
        p = jax.tree.map(jnp.asarray, pickle.load(f))
    q = set_std(p, 0.2)
    assert np.allclose(np.exp(np.asarray(q["log_std"])), 0.2)
    for a, b in zip(jax.tree.leaves(p["actor"]), jax.tree.leaves(q["actor"])):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.allclose(np.exp(np.asarray(p["log_std"])), 0.2)     # orijinal değişmedi
