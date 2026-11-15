"""Maç oynatma: bir politika çiftini ortamda koşturup sonuçları ve (istenirse) bir maçın kaydını döndürür.

Politika arayüzü: policy(env_state, mem, key) -> (actions (n, 2, 3), mem). Elle yazılmış rakipler de,
ileride eğitilmiş ağlar da bu arayüze uyar; böylece değerlendirme ve video kodu ikisi için aynı.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .rules import OUTCOME_NAMES


def make_runner(env, policy, reset_mem):
    """n_steps adımlık oyunu tek derlenmiş scan olarak döndüren fonksiyon üretir.

    Kayıt: her adımda 0. dünyanın drone konumları/yönelimleri ve sonucu (videolar için).
    """
    def body(carry, key):
        s, mem = carry
        k1, k2 = jax.random.split(key)
        actions, mem = policy(s, mem, k1)
        s_next, reward, done, info = env.step(s, actions, k2)
        mem = reset_mem(mem, done)
        rec = dict(pos=s.data.states.pos[0], quat=s.data.states.quat[0], visible=s.visible[0, 0],
                   arena_idx=s.arena_idx[0], t=s.t[0], outcome=info["outcome"][0])
        return (s_next, mem), (info["outcome"], info["length"], rec)

    def run(s, mem, key, n_steps):
        return jax.lax.scan(body, (s, mem), jax.random.split(key, n_steps))

    return jax.jit(run, static_argnums=3)


def summarize(outcomes, lengths):
    """Biten oyunların sonuç dağılımı (%) ve ortalama süreleri."""
    outcomes, lengths = np.asarray(outcomes).ravel(), np.asarray(lengths).ravel()
    ended = outcomes > 0
    total = int(ended.sum())
    rows = {OUTCOME_NAMES[k]: 100.0 * np.mean(outcomes[ended] == k) for k in range(1, len(OUTCOME_NAMES))}
    return total, rows, float(lengths[ended].mean()) if total else float("nan")


def split_games(rec):
    """0. dünyanın kaydını oyunlara böl: [(başlangıç adımı, bitiş adımı, sonuç), ...] (bitiş dahil)."""
    t, outcome = np.asarray(rec["t"]), np.asarray(rec["outcome"])
    starts = np.flatnonzero(t == 0)
    games = []
    for a in starts:
        ends = np.flatnonzero(outcome[a:] > 0)
        if len(ends):
            games.append((int(a), int(a + ends[0]), int(outcome[a + ends[0]])))
    return games
