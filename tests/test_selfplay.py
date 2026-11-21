"""Self-play testleri: kontrol maçları her dünyada tam bir maç sayıyor mu?"""
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from chase_tag.arenas import pack_layout
from chase_tag.env import ChaseTagEnv
from chase_tag.rules import TIMEOUT
from chase_tag.selfplay import make_match_eval

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
