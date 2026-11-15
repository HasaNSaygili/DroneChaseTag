"""Geometri testleri. Ortamdaki bir geometri hatası çökme üretmez; ajan onu sömürmeyi öğrenir.
O yüzden hem elle hesaplanmış basit durumlar hem de bağımsız bir kaba yöntemle karşılaştırma var."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chase_tag.arenas import dist_to_box, dist_to_cyl, make_arena_set, pack_layout
from chase_tag.geometry import RAY_MAX, arena_from_set, crashed, line_of_sight, ray_dist
from chase_tag.rules import ARENA, DRONE_R

X, Y, Z = np.eye(3, dtype=np.float32)
H = float(ARENA[2])


def arena(boxes=(), cyls=()):
    return arena_from_set(pack_layout(np.array(boxes).reshape(-1, 2, 3), np.array(cyls).reshape(-1, 5)))


def p(*xyz):
    return jnp.array(xyz, jnp.float32)


# x = 2..2.5 arasında, y = 2.5..3.5 boyunca uzanan duvarlar (tam boy / alçak / sarkan)
FULL_BOX = [[2.0, 2.5, 0.0], [2.5, 3.5, H]]
LOW_BOX = [[2.0, 2.5, 0.0], [2.5, 3.5, 0.8]]
CYL = [3.0, 3.0, 0.5, 0.0, H]           # (3, 3) merkezli, 0,5 m yarıçaplı tam boy silindir
OVERHANG_CYL = [3.0, 3.0, 0.5, 1.0, H]  # tavandan z = 1'e kadar sarkıyor


# ---------------- ışın mesafesi ----------------
def test_ray_open_arena_hits_wall():
    assert ray_dist(p(4, 3, 1), X, arena()) == pytest.approx(2.0)
    assert ray_dist(p(3, 3, 1.5), Z, arena()) == pytest.approx(0.5)
    assert ray_dist(p(1, 3, 1), -X, arena()) == pytest.approx(1.0)


def test_ray_capped_at_ray_max():
    assert ray_dist(p(0.5, 3, 1), X, arena()) == pytest.approx(RAY_MAX)


def test_ray_box():
    assert ray_dist(p(1, 3, 1), X, arena([FULL_BOX])) == pytest.approx(1.0)


def test_ray_cylinder():
    assert ray_dist(p(1, 3, 1), X, arena(cyls=[CYL])) == pytest.approx(1.5)
    assert ray_dist(p(3, 1, 1), Y, arena(cyls=[CYL])) == pytest.approx(1.5)


def test_ray_passes_over_low_box_but_hits_its_top():
    a = arena([LOW_BOX])
    assert ray_dist(p(1, 3, 1.5), X, a) == pytest.approx(RAY_MAX)       # üstünden geçer
    assert ray_dist(p(2.25, 3, 1.5), -Z, a) == pytest.approx(0.7)       # aşağı bakınca üst yüzey


def test_ray_passes_under_overhang():
    a = arena(cyls=[OVERHANG_CYL])
    assert ray_dist(p(1, 3, 0.5), X, a) == pytest.approx(RAY_MAX)       # altından geçer
    assert ray_dist(p(3, 3, 0.5), Z, a) == pytest.approx(0.5)           # yukarı bakınca alt yüzey


def test_floating_box_over_and_under():
    a = arena([[[2.0, 2.5, 0.8], [2.5, 3.5, 1.3]]])   # havada duran platform
    assert ray_dist(p(1, 3, 0.5), X, a) == pytest.approx(RAY_MAX)   # altından geçer
    assert ray_dist(p(1, 3, 1.6), X, a) == pytest.approx(RAY_MAX)   # üstünden geçer
    assert ray_dist(p(1, 3, 1.0), X, a) == pytest.approx(1.0)       # ortadan çarpar
    assert ray_dist(p(2.25, 3, 0.3), Z, a) == pytest.approx(0.5)    # alttan bakınca alt yüzey
    assert not line_of_sight(p(1, 3, 1.0), p(4, 3, 1.0), a)
    assert line_of_sight(p(1, 3, 0.5), p(4, 3, 0.5), a)


def test_inactive_obstacles_are_ignored():
    s = pack_layout(np.array([FULL_BOX]), np.array([CYL]))
    s["box_on"][:] = False
    s["cyl_on"][:] = False
    assert ray_dist(p(1, 3, 1), X, arena_from_set(s)) == pytest.approx(RAY_MAX)


# ---------------- görüş hattı ----------------
def test_line_of_sight():
    assert line_of_sight(p(1, 3, 1), p(5, 3, 1), arena())
    assert not line_of_sight(p(1, 3, 1), p(5, 3, 1), arena([FULL_BOX]))
    assert not line_of_sight(p(1, 3, 1), p(5, 3, 1), arena(cyls=[CYL]))
    assert line_of_sight(p(1, 3, 1.5), p(5, 3, 1.5), arena([LOW_BOX]))          # alçak duvarın üstünden
    assert line_of_sight(p(1, 3, 0.5), p(5, 3, 0.5), arena(cyls=[OVERHANG_CYL]))  # sarkanın altından


def test_line_of_sight_segment_only():
    # Engel iki drone'un arasında değil, arkada: görüşü kesmez
    assert line_of_sight(p(0.5, 3, 1), p(1.5, 3, 1), arena([FULL_BOX]))


# ---------------- çarpma ----------------
def test_crash_obstacles():
    assert crashed(p(2.0 - 0.05, 3, 1), arena([FULL_BOX]))       # yüzeye 5 cm
    assert not crashed(p(2.0 - 0.1, 3, 1), arena([FULL_BOX]))    # yüzeye 10 cm
    assert crashed(p(3.0 - 0.5 - 0.05, 3, 1), arena(cyls=[CYL]))
    assert not crashed(p(3.0 - 0.5 - 0.1, 3, 1), arena(cyls=[CYL]))
    assert crashed(p(2.25, 3, 0.85), arena([LOW_BOX]))           # alçak kutunun üstüne 5 cm
    assert not crashed(p(2.25, 3, 0.95), arena([LOW_BOX]))


def test_crash_boundaries():
    a = arena()
    assert not crashed(p(3, 3, 1), a)
    for q in [(0.03, 3, 1), (5.97, 3, 1), (3, 0.03, 1), (3, 5.97, 1), (3, 3, 0.03), (3, 3, 1.97)]:
        assert crashed(p(*q), a), q


# ---------------- bağımsız yöntemle karşılaştırma ----------------
@pytest.fixture(scope="module")
def random_arenas():
    return make_arena_set(20, phase=3, n_obst=(4, 8), seed=123, verbose=False)


def _np_min_dist(q, s, i):
    """NumPy ile noktadan en yakın engele mesafe (arenas.py'deki formüller)."""
    d = [dist_to_box(q, b) for b in s["boxes"][i][s["box_on"][i]]]
    d += [dist_to_cyl(q, c) for c in s["cyls"][i][s["cyl_on"][i]]]
    return np.min(d, axis=0) if d else np.full(q.shape[:-1], np.inf)


def test_crash_matches_numpy_distance(random_arenas):
    rng = np.random.default_rng(0)
    for i in range(len(random_arenas["n_obst"])):
        a = jax.tree.map(lambda x: x[i], arena_from_set(random_arenas))
        q = rng.uniform(0, ARENA, size=(500, 3)).astype(np.float32)
        got = np.asarray(jax.vmap(crashed, (0, None))(jnp.asarray(q), a))
        near_wall = np.any((q < DRONE_R) | (q > ARENA - DRONE_R), -1)
        want = (_np_min_dist(q, random_arenas, i) < DRONE_R) | near_wall
        assert (got == want).all()


def test_ray_matches_brute_force_marching(random_arenas):
    """Işın boyunca 1 mm adımlarla yürü; engele ya da arena dışına ilk girilen yer = beklenen mesafe."""
    rng = np.random.default_rng(1)
    ts = np.linspace(0, RAY_MAX, 3001)
    for i in range(len(random_arenas["n_obst"])):
        a = jax.tree.map(lambda x: x[i], arena_from_set(random_arenas))
        for _ in range(25):
            o = rng.uniform(0.2, ARENA - 0.2)
            if _np_min_dist(o[None], random_arenas, i)[0] < 0.05:
                continue  # engelin içinden başlayan ışın anlamsız
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            pts = o + ts[:, None] * d
            blocked = (_np_min_dist(pts, random_arenas, i) <= 0) | np.any((pts < 0) | (pts > ARENA), -1)
            want = ts[np.argmax(blocked)] if blocked.any() else RAY_MAX
            got = float(ray_dist(jnp.asarray(o, jnp.float32), jnp.asarray(d, jnp.float32), a))
            assert got == pytest.approx(want, abs=2e-3), (i, o, d)
