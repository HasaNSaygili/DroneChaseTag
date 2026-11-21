"""Değerlendirme testleri: Bradley–Terry puanları doğru mu, turnuva her çifte doğru sayıda maç veriyor mu?"""
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from chase_tag.arenas import pack_layout
from chase_tag.env import ChaseTagEnv
from chase_tag.eval import ELO_SCALE, bradley_terry, round_robin, win_matrix
from chase_tag.rules import TIMEOUT


def test_bradley_terry_recovers_known_ratings():
    """Bilinen puanlardan beklenen galibiyet sayılarını üret; uydurma aynı puan FARKLARINI geri bulmalı."""
    r = np.array([0.0, 1.0, 2.0])            # kovalayanlar (doğal ölçek)
    s = np.array([0.5, 1.5])                 # kaçanlar
    games = np.full((3, 2), 100_000.0)
    wins = games / (1.0 + np.exp(-(r[:, None] - s[None, :])))
    elo_c, elo_e = bradley_terry(wins, games)
    true = ELO_SCALE * np.concatenate([r, s])
    got = np.concatenate([elo_c, elo_e])
    np.testing.assert_allclose(got - got.mean(), true - true.mean(), atol=1.0)   # 1 Elo puanı içinde
    assert abs(got.mean() - 1000.0) < 1e-6


def test_win_matrix_wct():
    """WCT kuralı: ebeleme ve kaçan çarptı kovalayanın, süre doldu ve kovalayan çarptı kaçanın; ikisi çarptı 0,5."""
    env = ChaseTagEnv(2, {k: v[None] for k, v in pack_layout(np.zeros((0, 2, 3)), np.zeros((0, 5))).items()})
    C = np.array([[[0, 10, 20, 30, 40, 50]]])   # devam, ebeleme, süre, kaçan çarptı, kovalayan çarptı, ikisi
    wins, games = win_matrix(C, env.reward_table)
    assert games[0, 0] == 150 and wins[0, 0] == 10 + 30 + 0.5 * 50


def test_round_robin_counts_every_pair():
    """2 kovalayan × 2 kaçan = 4 çift, aynı anda 3 çift oynanır → ikinci grup 1 çift + 2 dolgu. Herkes yerinde
    durursa her maç süre dolarak biter; her çift tam `games` maç almalı (dolgu sayılmamalı)."""
    with open("experiments/ppo-chaser-empty-fix-s0/params_00476.pkl", "rb") as f:
        still = jax.tree.map(jnp.zeros_like, pickle.load(f))
    still["log_std"] = jnp.full_like(still["log_std"], -20.0)
    games = 2
    env = ChaseTagEnv(3 * games, {k: v[None] for k, v in pack_layout(np.zeros((0, 2, 3)), np.zeros((0, 5))).items()})
    C = round_robin(env, [still, still], [still, still], games, jax.random.PRNGKey(0), log=lambda m: None)
    assert C.shape[:2] == (2, 2)
    assert (C[..., 1:].sum(-1) == games).all()
    assert (C[..., TIMEOUT] == games).all()
