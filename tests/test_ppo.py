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


def test_fresh_policy_starts_near_zero_with_exploration():
    """Yeni ağ: ortalama ~0 (önyargısız), std ~0,6 (keşif)."""
    p = init_params(jax.random.PRNGKey(0), PPOConfig())
    mu, std = actor_dist(p, jnp.ones((4, OBS_DIM)))
    assert np.abs(np.asarray(mu)).max() < 0.1
    assert np.allclose(np.asarray(std), np.exp(-0.5))
