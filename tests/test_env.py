"""Ortam testleri: fizik makul mü, her kural doğru sonucu ve ödülü veriyor mu?"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chase_tag import geometry as G
from chase_tag.arenas import pack_layout
from chase_tag.env import CRITIC_DIM, OBS_DIM, ChaseTagEnv
from chase_tag.rules import (C_CRASH, CRASH_PEN, CTRL_HZ, MAX_STEPS, NONE, TAG, TIMEOUT, V_MAX)

N = 2
WALL = [[2.8, 0.0, 0.0], [3.2, 6.0, 2.0]]   # arenayı ikiye bölen tam boy duvar (sadece test için)


def arena_set(*layouts):
    packed = [pack_layout(np.array(b).reshape(-1, 2, 3), np.zeros((0, 5))) for b in layouts]
    return {k: np.stack([p[k] for p in packed]) for k in packed[0]}


@pytest.fixture(scope="module")
def env():
    return ChaseTagEnv(N, arena_set([]))   # tek boş arena


@pytest.fixture(scope="module")
def step(env):
    return jax.jit(env.step)


def place(s, pos, arena=None):
    """Drone'ları elle istenen yere koy (hız sıfır)."""
    pos = jnp.asarray(pos, jnp.float32)
    data = s.data.replace(states=s.data.states.replace(pos=pos, vel=jnp.zeros_like(pos)))
    s = s._replace(data=data, ref=pos, dist=jnp.linalg.norm(pos[:, 0] - pos[:, 1], axis=-1),
                   last_seen=pos[:, ::-1])
    if arena is not None:
        s = s._replace(arena=jax.tree.map(lambda a: jnp.broadcast_to(a, (N,) + a.shape), arena))
    return s


def run(step, s, actions, n_steps, key=0):
    """Sabit eylemle n adım; her adımın (konum, hız, bitti, sonuç) kaydı."""
    keys = jax.random.split(jax.random.PRNGKey(key), n_steps)
    hist = []
    for k in keys:
        s, r, d, info = step(s, actions, k)
        hist.append((np.asarray(s.data.states.pos), np.asarray(s.data.states.vel), np.asarray(d),
                     np.asarray(info["outcome"]), np.asarray(r)))
    return s, hist


ZERO = jnp.zeros((N, 2, 3))


# ---------------- başlangıç ----------------
def test_reset(env):
    s = env.reset(jax.random.PRNGKey(0))
    pos = np.asarray(s.data.states.pos)
    assert np.linalg.norm(pos[:, 0] - pos[:, 1], axis=-1).min() > 5.0      # karşı köşeler
    assert ((pos[..., :2] < 1.0) | (pos[..., :2] > 5.0)).all()              # köşe bölgelerinde
    assert (np.asarray(s.t) == 0).all()
    obs, cobs = env.observe(s), env.critic_obs(s)
    assert obs.shape == (N, 2, OBS_DIM) and cobs.shape == (N, 2, CRITIC_DIM)
    assert np.isfinite(obs).all() and np.isfinite(cobs).all()


# ---------------- fizik ----------------
def test_hover_is_stable(env, step):
    s = place(env.reset(jax.random.PRNGKey(1)), [[[2, 2, 1], [4, 4, 1]]] * N)
    s, hist = run(step, s, ZERO, 2 * CTRL_HZ)                                 # 2 s
    pos = np.stack([h[0] for h in hist])
    assert not any(h[2].any() for h in hist)                                 # kimse çarpmadı
    assert np.isfinite(pos).all()
    assert np.abs(pos[-1] - np.array([[2, 2, 1], [4, 4, 1]])).max() < 0.12  # yerinde durdu (≤ ~8 cm sapma)


def test_velocity_tracking_and_speed_limit(env, step):
    s = place(env.reset(jax.random.PRNGKey(2)), [[[1, 1, 1], [5, 5, 1.5]]] * N)
    act = jnp.zeros((N, 2, 3)).at[:, 0].set(jnp.array([1.0, 1.0, 0.0]))     # kovalayan yatay çapraz tam gaz
    s, hist = run(step, s, act, 3 * CTRL_HZ // 2)                            # 1,5 s
    pos = np.stack([h[0][0, 0] for h in hist])
    speed = np.array([np.linalg.norm(h[1][0, 0]) for h in hist])
    print("hız profili (m/s):", np.round(speed[::5], 2))
    assert speed[-1] == pytest.approx(V_MAX, abs=0.2)                        # komut verilen hıza ulaştı
    assert np.linalg.norm(pos[-1] - pos[-1 - CTRL_HZ]) < V_MAX * 1.1         # sürdürülebilir hız ≈ 2 m/s, √2 katı değil
    assert speed.max() < V_MAX * 1.4                                         # anlık aşma sınırlı


# ---------------- kurallar ----------------
def test_tag(env, step):
    s = place(env.reset(jax.random.PRNGKey(3)), [[[3, 3, 1], [3.2, 3, 1]]] * N)
    s2, r, d, info = step(s, ZERO, jax.random.PRNGKey(0))
    assert np.asarray(d).all() and (np.asarray(info["outcome"]) == TAG).all()
    assert np.asarray(r[:, 0]) == pytest.approx(1.0, abs=0.02)
    assert np.asarray(r[:, 1]) == pytest.approx(-1.0, abs=0.02)
    assert (np.asarray(s2.t) == 0).all()                                     # otomatik yeniden başladı
    pos = np.asarray(s2.data.states.pos)
    assert np.linalg.norm(pos[:, 0] - pos[:, 1], axis=-1).min() > 5.0


def test_crash_into_wall(env, step):
    s = place(env.reset(jax.random.PRNGKey(4)), [[[0.3, 3, 1], [4, 4, 1]]] * N)
    act = jnp.zeros((N, 2, 3)).at[:, 0, 0].set(-1.0)                         # kovalayan duvara doğru
    s, hist = run(step, s, act, CTRL_HZ)
    first = next(i for i, h in enumerate(hist) if h[2].any())
    assert (hist[first][3] == C_CRASH).all()
    assert hist[first][4][0] == pytest.approx([-1.0 - CRASH_PEN, 1.0], abs=0.02)


def test_timeout(env, step):
    s = place(env.reset(jax.random.PRNGKey(5)), [[[1, 1, 1], [5, 5, 1]]] * N)
    s = s._replace(t=jnp.full(N, MAX_STEPS - 1))
    _, r, d, info = step(s, ZERO, jax.random.PRNGKey(0))
    assert (np.asarray(info["outcome"]) == TIMEOUT).all()
    assert (np.asarray(info["length"]) == MAX_STEPS).all()
    assert np.asarray(r) == pytest.approx(np.array([[-1.0, 1.0]] * N), abs=0.02)


def test_no_outcome_in_open_play(env, step):
    s = place(env.reset(jax.random.PRNGKey(6)), [[[1, 1, 1], [5, 5, 1]]] * N)
    _, _, d, info = step(s, ZERO, jax.random.PRNGKey(0))
    assert not np.asarray(d).any() and (np.asarray(info["outcome"]) == NONE).all()


def test_visibility_memory(env, step):
    wall = G.arena_from_set(pack_layout(np.array([WALL]), np.zeros((0, 5))))
    s = place(env.reset(jax.random.PRNGKey(7)), [[[1.5, 3, 1], [4.5, 3, 1]]] * N, arena=wall)
    s, _ = run(step, s, ZERO, 10)
    assert not np.asarray(s.visible).any()
    assert np.asarray(s.unseen) == pytest.approx(10 / CTRL_HZ)
    # Görmüyorken rakibin yeri son bilinen yer olarak kalır (place() başlangıçta gerçek yeri yazdı)
    np.testing.assert_allclose(np.asarray(s.last_seen[:, 0]), [[4.5, 3, 1]] * N, atol=0.05)
    obs = np.asarray(env.observe(s))
    assert (obs[..., 9] == 0).all()                        # görünürlük bayrağı
    assert (obs[..., 11:14] == 0).all()                    # rakip hızı gizli


def test_scripted_actions_valid():
    """Elle yazılmış rakipler karışık (faz 3) arenalarda sonlu, birim küre içinde eylem üretmeli."""
    from chase_tag.arenas import make_arena_set
    from chase_tag.scripted import scripted_actions, scripted_init
    e = ChaseTagEnv(N, make_arena_set(4, phase=3, n_obst=(6, 8), seed=0, verbose=False))
    s = e.reset(jax.random.PRNGKey(0))
    a, _ = jax.jit(scripted_actions)(s, scripted_init(N), jax.random.PRNGKey(1))
    a = np.asarray(a)
    assert a.shape == (N, 2, 3) and np.isfinite(a).all()
    assert (np.linalg.norm(a, axis=-1) <= 1.0 + 1e-5).all()


def test_jitted_scan(env):
    """Bütün oyun döngüsü tek bir derlenmiş scan içinde dönebiliyor mu (eğitimde böyle çalışacak)?"""
    def body(s, k):
        s, r, d, info = env.step(s, ZERO, k)
        return s, d
    run_scan = jax.jit(lambda s, k: jax.lax.scan(body, s, jax.random.split(k, 50)))
    s, d = run_scan(env.reset(jax.random.PRNGKey(8)), jax.random.PRNGKey(9))
    assert np.isfinite(np.asarray(s.data.states.pos)).all()
