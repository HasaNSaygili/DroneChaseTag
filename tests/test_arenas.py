"""Arena üretimi testleri: bağlantılılık kontrolü gerçekten kapalı cepleri ve dar geçitleri yakalıyor mu?"""
import numpy as np

from chase_tag.arenas import (CORNER_MASK, PASSAGE_MIN, is_connected, make_arena_set, obstacle_mask,
                              unpack_layout)
from chase_tag.rules import ARENA

H = float(ARENA[2])


def wall_across(gap, z0=0.0, z1=H):
    """Arenayı x = 3'te ikiye bölen duvar; ortasında `gap` genişliğinde bir boşluk."""
    y0, y1 = 3.0 - gap / 2, 3.0 + gap / 2
    return [np.array([[2.9, 0.0, z0], [3.1, y0, z1]]), np.array([[2.9, y1, z0], [3.1, 6.0, z1]])]


def test_empty_arena_connected():
    assert is_connected([], [])


def test_closed_room_is_rejected():
    # Ortada dört duvarlı kapalı oda: içi ayrı bir parça → kapalı cep
    room = [np.array([[2.0, 2.0, 0.0], [4.0, 2.2, H]]), np.array([[2.0, 3.8, 0.0], [4.0, 4.0, H]]),
            np.array([[2.0, 2.0, 0.0], [2.2, 4.0, H]]), np.array([[3.8, 2.0, 0.0], [4.0, 4.0, H]])]
    assert not is_connected(room, [])
    assert is_connected(room[:3], [])   # bir duvarı kaldırınca oda açılır


def test_narrow_gap_counts_as_closed():
    assert not is_connected(wall_across(gap=0.0), [])
    assert not is_connected(wall_across(gap=PASSAGE_MIN - 0.15), [])   # drone sığmaz
    assert is_connected(wall_across(gap=PASSAGE_MIN + 0.2), [])


def test_low_and_overhang_walls_are_passable():
    assert is_connected(wall_across(gap=0.0, z1=0.8), [])    # üstünden geçilir
    assert is_connected(wall_across(gap=0.0, z0=1.0), [])    # altından geçilir
    assert is_connected(wall_across(gap=0.0, z0=0.8, z1=1.3), [])   # havada: altından ya da üstünden


def test_generated_arenas_are_valid():
    s = make_arena_set(30, phase=3, n_obst=(2, 8), seed=7, verbose=False)
    assert s["boxes"].shape[0] == 30
    assert ((s["n_obst"] >= 2) & (s["n_obst"] <= 8)).all()
    for i in range(30):
        boxes, cyls = unpack_layout(s, i)
        assert len(boxes) + len(cyls) == s["n_obst"][i]
        assert is_connected(boxes, cyls)
        assert not (obstacle_mask(boxes, cyls) & CORNER_MASK).any()   # köşeler boş


def test_same_seed_same_arenas():
    a = make_arena_set(5, phase=2, seed=3, verbose=False)
    b = make_arena_set(5, phase=2, seed=3, verbose=False)
    assert all(np.array_equal(a[k], b[k]) for k in a)
