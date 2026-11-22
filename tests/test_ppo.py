"""PPO ağ testleri."""
import jax
import jax.numpy as jnp
import numpy as np

from chase_tag.env import OBS_DIM
from chase_tag.ppo import PPOConfig, actor_dist, init_params


def test_squashed_mean_stays_in_action_range():
    """Eylem ortalaması [-1, 1]'de kalmalı, ağırlıklar ne kadar büyürse büyüsün. Sınırsız ortalama ~4,5'e kayıp
    doyuyordu (ortam boyu 1'e kırptığı için bütün denemeler aynı eyleme dönüşüyor, öğrenme duruyordu)."""
    p = init_params(jax.random.PRNGKey(0), PPOConfig())
    p["actor"][-1]["w"] = p["actor"][-1]["w"] * 1e4                   # çıktıyı bilerek patlat
    obs = jax.random.normal(jax.random.PRNGKey(1), (512, OBS_DIM))
    mu, _ = actor_dist(p, obs)
    assert np.abs(np.asarray(mu)).max() <= 1.0
    mu_raw, _ = actor_dist(p, obs, squash=False)                        # eski davranış: sınırsız
    assert np.abs(np.asarray(mu_raw)).max() > 1.0


def _one_update(target_kl, warmup_steps=0):
    """Küçük boş arenada tek PPO turu (8 dünya × 16 adım, 2 epoch × 2 minibatch = 4 adım)."""
    from chase_tag.arenas import pack_layout
    from chase_tag.env import ChaseTagEnv
    from chase_tag.ppo import make_update
    packed = pack_layout(np.zeros((0, 2, 3)), np.zeros((0, 5)))
    env = ChaseTagEnv(8, {k: v[None] for k, v in packed.items()})
    cfg = PPOConfig(n_worlds=8, n_steps=16, n_blocks=1, epochs=2, n_minibatches=2, total_steps=8 * 16,
                    opponent="open", target_kl=target_kl, warmup_steps=warmup_steps, anneal_lr=False)
    update, tx, _ = make_update(cfg, env, 0)
    p0 = init_params(jax.random.PRNGKey(0), cfg)
    p1, *_, m = update(p0, tx.init(p0), env.reset(jax.random.PRNGKey(1)), None, jax.random.PRNGKey(2))
    return p0, p1, m


def test_kl_early_stop():
    """target_kl: politika bir turda sınırdan fazla değişince turun kalan adımları atlanır.
    İlk adımda politika henüz değişmemiş (kl = 0) → hep uygulanır. Sınır çok küçükse ikinci adımdan itibaren
    hepsi atlanır: 4 adımın 3'ü. Kapalıyken (0) hiçbiri atlanmaz. Ayrıca atlansa da ilk adım parametreleri değiştirir."""
    _, _, m_off = _one_update(0.0)
    assert float(m_off["kl_skip"]) == 0.0
    p0, p1, m_on = _one_update(1e-12)
    assert np.isclose(float(m_on["kl_skip"]), 3 / 4)
    moved = jax.tree.leaves(jax.tree.map(lambda a, b: np.abs(np.asarray(a) - np.asarray(b)).max(), p0, p1))
    assert max(moved) > 0


def _max_change(a, b):
    return max(jax.tree.leaves(jax.tree.map(lambda x, y: np.abs(np.asarray(x) - np.asarray(y)).max(), a, b)))


def test_warmup_softens_first_steps():
    """Isınma: taze Adam'ın ilk adımları her ağırlığı ~lr oynatır; ısınmayla (100 adım) ilk 4 adımda lr en fazla
    3/100 · lr, yani ağırlık değişimi ısınmasıza göre çok küçük olmalı."""
    p0, p_cold, _ = _one_update(0.0)
    _, p_warm, _ = _one_update(0.0, warmup_steps=100)
    assert _max_change(p0, p_warm) < 0.1 * _max_change(p0, p_cold)


def test_fresh_policy_starts_near_zero_with_exploration():
    """Yeni ağ: ortalama ~0 (önyargısız), std ~0,6 (keşif)."""
    p = init_params(jax.random.PRNGKey(0), PPOConfig())
    mu, std = actor_dist(p, jnp.ones((4, OBS_DIM)))
    assert np.abs(np.asarray(mu)).max() < 0.1
    assert np.allclose(np.asarray(std), np.exp(-0.5))
