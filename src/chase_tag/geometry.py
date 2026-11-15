"""Oyun sırasında çalışan geometri (JAX): ışın mesafesi, görüş hattı, çarpma.

Bunlar her adımda binlerce arenada aynı anda çalışır, o yüzden JAX ile ve sabit boyutlu yazıldı.
Fonksiyonlar TEK nokta ve TEK arena için yazılır; çoğaltmayı çağıran taraf jax.vmap ile yapar.

Ortak fikir — "giriş-çıkış aralığı": ışın o + t*d biçiminde. Her engel için ışının engelin İÇİNDE kaldığı
[t_giriş, t_çıkış] aralığını buluruz. Aralık boş değilse ve ışının önündeyse (t_çıkış >= 0) ışın çarpar.
  kutu:     x, y ve z eksenlerinde ayrı ayrı aralık bul, kesiştir ("slab" yöntemi)
  silindir: yatayda daire aralığı (ikinci derece denklem) ile dikeyde z aralığını kesiştir
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .rules import ARENA, DRONE_R


class Arena(NamedTuple):
    """Bir arenanın engelleri (gösterim için bkz. arenas.py). NamedTuple JAX'te doğrudan pytree'dir:
    jit/vmap içinden geçebilir; başına arena boyutu eklenince binlerce arena tek nesnede durur."""
    boxes: jax.Array    # (N_BOX_MAX, 2, 3)  [[min], [max]]
    box_on: jax.Array   # (N_BOX_MAX,)       kullanılmayan yerler False
    cyls: jax.Array     # (N_CYL_MAX, 5)     [cx, cy, r, z_min, z_max]
    cyl_on: jax.Array   # (N_CYL_MAX,)


def arena_from_set(s):
    """arenas.py'nin NumPy sözlüğünden (tek arena ya da başında arena boyutu olan küme) Arena yapar."""
    return Arena(*(jnp.asarray(s[k]) for k in Arena._fields))


# ---- Işın sensörü: 16 yatay yön + yukarı + aşağı ----
N_RAYS_H = 16
RAY_MAX = 3.0   # bundan uzağı "boş" sayılır; gözlemi sınırlı tutar
_ang = np.linspace(0, 2 * np.pi, N_RAYS_H, endpoint=False)
RAY_DIRS = np.concatenate([
    np.stack([np.cos(_ang), np.sin(_ang), np.zeros_like(_ang)], -1),
    np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
]).astype(np.float32)  # (18, 3), hepsi birim vektör
N_RAYS = RAY_DIRS.shape[0]

_EPS = 1e-9


def _safe_inv(d):
    """1/d; d≈0 ise çok büyük sayı. O eksende ışın hareket etmiyor demek: aralık ya (-∞, ∞) ya boş olur."""
    return 1.0 / jnp.where(jnp.abs(d) < _EPS, _EPS, d)


def _box_intervals(o, d, arena):
    inv = _safe_inv(d)
    t1 = (arena.boxes[:, 0] - o) * inv    # (NB, 3) her eksende min yüzeye varış
    t2 = (arena.boxes[:, 1] - o) * inv    # (NB, 3) her eksende max yüzeye varış
    t_in = jnp.max(jnp.minimum(t1, t2), -1)   # üç eksenin hepsine girmiş olmak gerekir → en geç giriş
    t_out = jnp.min(jnp.maximum(t1, t2), -1)  # birinden çıkmak yeter → en erken çıkış
    return t_in, t_out


def _cyl_intervals(o, d, arena):
    c = arena.cyls
    # Yatay: |o_xy + t d_xy - merkez|² = r²  →  a t² + b t + k = 0
    oc = o[:2] - c[:, :2]                       # (NC, 2)
    a = jnp.sum(d[:2] ** 2)
    b = 2.0 * oc @ d[:2]
    k = jnp.sum(oc ** 2, -1) - c[:, 2] ** 2     # k <= 0: başlangıç noktası dairenin içinde
    disc = b * b - 4.0 * a * k
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    a_safe = jnp.maximum(a, _EPS)
    r0, r1 = (-b - sq) / (2 * a_safe), (-b + sq) / (2 * a_safe)
    miss = disc < 0
    r0 = jnp.where(miss, jnp.inf, r0)
    r1 = jnp.where(miss, -jnp.inf, r1)
    # Dik ışın (yukarı/aşağı): yatayda hiç hareket yok; daire içindeyse bütün doğru içeride, değilse hiç
    vertical = a < _EPS
    r0 = jnp.where(vertical, jnp.where(k <= 0, -jnp.inf, jnp.inf), r0)
    r1 = jnp.where(vertical, jnp.where(k <= 0, jnp.inf, -jnp.inf), r1)
    # Dikey: z aralığı (kutudaki gibi tek eksenli slab)
    inv = _safe_inv(d[2])
    s1, s2 = (c[:, 3] - o[2]) * inv, (c[:, 4] - o[2]) * inv
    t_in = jnp.maximum(r0, jnp.minimum(s1, s2))
    t_out = jnp.minimum(r1, jnp.maximum(s1, s2))
    return t_in, t_out


def _intervals(o, d, arena):
    """Bütün engeller için giriş/çıkış t'leri ve 'ışının önünde çarpıyor mu' bayrağı."""
    bi, bo = _box_intervals(o, d, arena)
    ci, co = _cyl_intervals(o, d, arena)
    t_in, t_out = jnp.concatenate([bi, ci]), jnp.concatenate([bo, co])
    on = jnp.concatenate([arena.box_on, arena.cyl_on])
    hit = on & (t_out >= jnp.maximum(t_in, 0.0))
    return t_in, t_out, hit


def ray_dist(o, d, arena):
    """o noktasından birim vektör d yönünde en yakın engel ya da sınır yüzeyine mesafe (en fazla RAY_MAX)."""
    t_in, _, hit = _intervals(o, d, arena)
    t_obs = jnp.min(jnp.where(hit, jnp.maximum(t_in, 0.0), jnp.inf))
    inv = _safe_inv(d)
    t_wall = jnp.where(d > 0, (ARENA - o) * inv, jnp.where(d < 0, -o * inv, jnp.inf))
    return jnp.minimum(jnp.minimum(t_obs, jnp.min(t_wall)), RAY_MAX)


def rays(p, arena):
    """p noktasından 18 ışının mesafeleri: (N_RAYS,). Gözlemin 'etrafımda ne var' kısmı."""
    return jax.vmap(lambda d: ray_dist(p, d, arena))(jnp.asarray(RAY_DIRS))


def line_of_sight(a, b, arena):
    """a ile b arasındaki doğru parçasını hiçbir engel kesmiyorsa True (rakip görünüyor).
    Işınla aynı hesap; tek fark aralığın t ∈ [0, 1] ile, yani sadece iki nokta arasıyla sınırlı olması."""
    t_in, _, hit = _intervals(a, b - a, arena)
    return ~jnp.any(hit & (t_in <= 1.0))


def crashed(p, arena):
    """Drone (DRONE_R yarıçaplı küre) bir engele ya da sınıra değdi mi?"""
    # Kutuya mesafe: p'yi kutuya kırp → kutudaki en yakın nokta
    d_box = jnp.linalg.norm(p - jnp.clip(p, arena.boxes[:, 0], arena.boxes[:, 1]), axis=-1)
    # Silindire mesafe: yatayda daireden, dikeyde z aralığından taşmaların bileşkesi
    c = arena.cyls
    rho = jnp.hypot(p[0] - c[:, 0], p[1] - c[:, 1])
    d_xy = jnp.maximum(rho - c[:, 2], 0.0)
    d_z = jnp.maximum(jnp.maximum(c[:, 3] - p[2], p[2] - c[:, 4]), 0.0)
    d_cyl = jnp.hypot(d_xy, d_z)
    hit_obs = jnp.any(arena.box_on & (d_box < DRONE_R)) | jnp.any(arena.cyl_on & (d_cyl < DRONE_R))
    out = jnp.any(p < DRONE_R) | jnp.any(p > ARENA - DRONE_R)   # zemin, tavan, duvarlar
    return hit_obs | out
