"""Elle yazılmış rakipler: öğrenmeyen, kuralları bizim yazdığımız drone'lar.

Üç işleri var: (1) RL'den önce kuralların doğru çalıştığını görmek, (2) eğitimin ilk antrenman rakibi
(rastgele iki ağ birbirine karşı oynarsa öğrenecek sinyal çıkmaz), (3) makalede kıyaslama çubuğu.
Zayıf bir baseline'ı yenmek bir şey kanıtlamaz; o yüzden makul derecede akıllı olmaları gerekiyor.

Bilgi adaleti: ikisi de arenanın haritasını bilir (WCT'de sporcular sahayı görür), ama rakibi sadece
görüş hattından görür: RL ajanıyla aynı kural. Rakip görünmüyorsa ellerinde son görüldüğü yer var.

  Kovalayan: hedefe giden yol açıksa düz uçar; değilse ızgarada en kısa yolu izler (engelin etrafından).
             Rakibi kaybedince tahmini yerine gider, orada da yoksa rastgele noktaları gezer.
  Kaçan:     OPEN'ın (Chen vd., RA-L 2025) potansiyel alanlı kaçanı, kodlarındaki formülle
             (github.com/thu-uav/Multi-UAV-pursuit-evasion, hideandseek.py: _get_dummy_policy_prey).

Hepsi JAX ile yazıldı: eğitimde rakip olarak binlerce arenada aynı anda çalışacaklar.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import geometry as G
from .rules import ARENA, CHASER, DRONE_R, EVADER, V_MAX

# ---------------------------------------------------------------------------
# Yol bulma ızgarası (2B, kovalayanın uçtuğu yükseklikte)
# ---------------------------------------------------------------------------
CELL = 0.2                                   # ızgara hücresi (m)
NX, NY = int(round(ARENA[0] / CELL)), int(round(ARENA[1] / CELL))   # 30 × 30
PLAN_CLEAR = DRONE_R + 0.15                  # planlanan yol engellerden en az bu kadar uzak geçsin
N_RELAX = 90                                 # dalga yayılımı adım sayısı (en uzun dolambaçlı yol için yeterli)
_BIG = 1e6
_CX = (np.arange(NX) + 0.5) * CELL
_CY = (np.arange(NY) + 0.5) * CELL
CELL_XY = np.stack(np.meshgrid(_CX, _CY, indexing="ij"), -1).astype(np.float32)   # (NX, NY, 2)
# 8 komşu ve adım maliyetleri (çapraz √2)
_NBR = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
_NBR_COST = np.array([np.hypot(dx, dy) for dx, dy in _NBR], np.float32)


def free_cells(arena, z):
    """z yüksekliğinde hangi hücreler uçulabilir? (NX, NY) bool."""
    pts = jnp.concatenate([jnp.asarray(CELL_XY), jnp.full((NX, NY, 1), z)], -1)
    d = jax.vmap(jax.vmap(G.obstacle_dist, (0, None)), (0, None))(pts, arena)
    xy = jnp.asarray(CELL_XY)
    near_wall = jnp.any((xy < PLAN_CLEAR) | (xy > ARENA[:2] - PLAN_CLEAR), -1)
    return (d > PLAN_CLEAR) & ~near_wall


def _shift(a, dx, dy, fill):
    """a[i, j] → a[i+dx, j+dy]; kenardan taşan yerler fill."""
    a = jnp.pad(a, 1, constant_values=fill)
    return a[1 + dx:1 + dx + NX, 1 + dy:1 + dy + NY]


def distance_field(free, goal_ij):
    """Her hücreden hedef hücreye ızgarada en kısa yol uzunluğu (hücre birimi; ulaşılamıyorsa büyük).
    Dalga yayılımı: hedef 0'dan başlar, her turda her hücre 'komşum + adım maliyeti'nden küçükse güncellenir.
    N_RELAX tur sonra dalga bütün ulaşılabilir hücrelere yayılmış olur (Bellman-Ford'un ızgara hali)."""
    goal = jnp.zeros((NX, NY), bool).at[goal_ij[0], goal_ij[1]].set(True)
    passable = free | goal

    def relax(_, D):
        cand = jnp.stack([_shift(D, dx, dy, _BIG) + c for (dx, dy), c in zip(_NBR, _NBR_COST)])
        return jnp.where(passable, jnp.minimum(D, cand.min(0)), _BIG)

    return jax.lax.fori_loop(0, N_RELAX, relax, jnp.where(goal, 0.0, _BIG))


def _cell_of(p):
    i = jnp.clip((p[0] / CELL).astype(jnp.int32), 0, NX - 1)
    j = jnp.clip((p[1] / CELL).astype(jnp.int32), 0, NY - 1)
    return jnp.stack([i, j])


def path_direction(me, goal, arena):
    """me'den goal'e gitmek için yatay yön (birim vektör). Yol açıksa düz, değilse ızgarada yokuş aşağı."""
    free = free_cells(arena, me[2])
    D = distance_field(free, _cell_of(goal))
    ij = _cell_of(me)
    # Komşular arasında hedefe en yakın olanı seç
    nb = jnp.array(_NBR) + ij
    ok = (nb[:, 0] >= 0) & (nb[:, 0] < NX) & (nb[:, 1] >= 0) & (nb[:, 1] < NY)
    nbc = jnp.clip(nb, 0, jnp.array([NX - 1, NY - 1]))
    vals = jnp.where(ok, D[nbc[:, 0], nbc[:, 1]] + jnp.asarray(_NBR_COST), _BIG)
    best = nbc[jnp.argmin(vals)]
    step_dir = jnp.asarray(CELL_XY)[best[0], best[1]] - me[:2]
    reachable = vals.min() < _BIG / 2
    # Doğrudan görüş açıksa dolambaca gerek yok (ve ızgaranın kaba kenarlarını da atlamış oluruz)
    straight = goal[:2] - me[:2]
    clear = G.line_of_sight(me, goal.at[2].set(me[2]), arena)
    d = jnp.where(clear | ~reachable, straight, step_dir)
    return d / (jnp.linalg.norm(d) + 1e-6)


# ---------------------------------------------------------------------------
# Ortak yardımcılar
# ---------------------------------------------------------------------------
def _unit(v):
    return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)


def _to_ball(v):
    return v / jnp.maximum(1.0, jnp.linalg.norm(v, axis=-1, keepdims=True))


def obstacle_push(p, arena, d_safe=0.6):
    """Işınlardan engel itmesi: yakındaki her yüzey drone'u ters yöne iter (yaklaştıkça karesiyle artar)."""
    r = G.rays(p, arena)
    push = (jnp.maximum(d_safe - r, 0.0) / d_safe) ** 2
    return -jnp.sum(jnp.asarray(G.RAY_DIRS) * push[:, None], 0), r


A_BRAKE = 3.0           # güvenlik filtresinin varsaydığı fren ivmesi (m/s²); ölçülen fren ~0,4 m @ 2 m/s
TAU = 0.15              # kontrolcünün tepki gecikmesi (s): komut değişince drone ancak bu kadar sonra tepki verir
SAFE_MARGIN = DRONE_R + 0.12


def safety_filter(v, rays, vel):
    """Fren mesafesine dayalı hız sınırı ('control barrier' fikri; arabadaki takip mesafesi kuralı).

    v: eylem (birim küre, × V_MAX = m/s), vel: drone'un şu anki gerçek hızı. Her ışın yönü için:
      gecikme payı = o yöne şu anki hız × TAU  (tepki verene kadar zaten alınacak yol)
      v_izin = √(2·a·(mesafe − pay − gecikme payı))      (v² = 2·a·s: kalan yolda durabilmek)
    Komutun o yöndeki bileşeni v_izin'den büyükse fazlası kırpılır; diğer bileşenler kalır, yani drone
    yüzey boyunca kayar. Gerçek hızı hesaba katmak şart: sadece komuta bakan sürümde drone 2 m/s ile
    duvara giderken filtre 'yavaşla' diyordu ama drone yavaşlayamadan çarpıyordu.
    """
    dirs = jnp.asarray(G.RAY_DIRS)
    v_ms = v * V_MAX
    reaction = jnp.maximum(dirs @ vel, 0.0) * TAU
    allow = jnp.sqrt(2.0 * A_BRAKE * jnp.maximum(rays - SAFE_MARGIN - reaction, 0.0))
    for i in range(G.N_RAYS):  # sırayla (biri kırpınca diğerinin bileşeni değişebilir)
        c = v_ms @ dirs[i]
        v_ms = v_ms - jnp.maximum(c - allow[i], 0.0) * dirs[i]
    return v_ms / V_MAX


# ---------------------------------------------------------------------------
# Kovalayan
# ---------------------------------------------------------------------------
class ChaserMem(NamedTuple):
    goal: jax.Array        # (n, 3) arama modundaki hedef nokta
    searching: jax.Array   # (n,)   rastgele gezme modunda mı
    opp_vel: jax.Array     # (n, 3) rakibin son görülen hızı (tahmin için)


def chaser_init(n):
    return ChaserMem(goal=jnp.full((n, 3), 3.0), searching=jnp.zeros(n, bool), opp_vel=jnp.zeros((n, 3)))


def _random_free_point(key, arena):
    """Arenada uçulabilir rastgele bir nokta (arama hedefi)."""
    free = free_cells(arena, 1.0).ravel()
    k = jax.random.categorical(key, jnp.where(free, 0.0, -jnp.inf))
    xy = jnp.asarray(CELL_XY).reshape(-1, 2)[k]
    return jnp.concatenate([xy, jnp.array([1.0])])


def _chaser_one(me, my_vel, opp, vis, last_seen, unseen, opp_vel_now, arena, mem_goal, mem_search, mem_vel, key):
    opp_vel = jnp.where(vis, opp_vel_now, mem_vel)
    # Görmüyorsam: rakip son görüldüğü yerden son hızıyla gitmeye devam etmiş olabilir (en fazla 1 s tahmin)
    guess = jnp.clip(last_seen + opp_vel * jnp.minimum(unseen, 1.0), 0.3, ARENA - 0.3)
    near = lambda a, b: jnp.linalg.norm(a[:2] - b[:2]) < 0.4
    # Mod geçişleri: görünce kovala; tahmini yere varıp bulamayınca ya da arama hedefine varınca yeni nokta
    arrive_guess = ~vis & ~mem_search & near(me, guess)
    arrive_goal = ~vis & mem_search & near(me, mem_goal)
    searching = ~vis & (mem_search | arrive_guess)
    new_goal = _random_free_point(key, arena)
    goal = jnp.where(arrive_guess | arrive_goal, new_goal, mem_goal)
    target = jnp.where(vis, opp, jnp.where(searching, goal, guess))

    d_xy = path_direction(me, target, arena)
    dz = jnp.clip(target[2] - me[2], -1.0, 1.0)
    push, rays = obstacle_push(me, arena)
    v = _unit(jnp.concatenate([d_xy, dz[None]]) + 0.8 * push)
    return safety_filter(v, rays, my_vel), goal, searching, opp_vel


def chaser_policy(es, mem, key):
    """Kovalayanın eylemi (n, 3) ve güncellenmiş belleği. es: env.EnvState."""
    pos, vel = es.data.states.pos, es.data.states.vel
    keys = jax.random.split(key, pos.shape[0])
    a, goal, searching, opp_vel = jax.vmap(_chaser_one)(
        pos[:, CHASER], vel[:, CHASER], pos[:, EVADER], es.visible[:, CHASER], es.last_seen[:, CHASER], es.unseen[:, CHASER],
        vel[:, EVADER], es.arena, mem.goal, mem.searching, mem.opp_vel, keys)
    return a, ChaserMem(goal=goal, searching=searching, opp_vel=opp_vel)


# ---------------------------------------------------------------------------
# Kaçan
# ---------------------------------------------------------------------------
_EPS = 1e-5   # OPEN'daki ile aynı


def _closest_points(p, arena):
    """Her engelde p'ye en yakın nokta: kutular (NB, 3), silindirler (NC, 3)."""
    q_box = jnp.clip(p, arena.boxes[:, 0], arena.boxes[:, 1])
    c = arena.cyls
    h = p[:2] - c[:, :2]
    rho = jnp.linalg.norm(h, axis=-1, keepdims=True)
    q_xy = c[:, :2] + h / (rho + _EPS) * jnp.minimum(rho, c[:, 2:3])
    q_cyl = jnp.concatenate([q_xy, jnp.clip(p[2], c[:, 3], c[:, 4])[:, None]], -1)
    return q_box, q_cyl


def _open_force(me, threat, use_threat, arena):
    """OPEN'ın toplam itme kuvveti F (aşağıdaki açıklamaya bkz.)."""
    away = me - threat
    d = jnp.linalg.norm(away)
    F = jnp.where(use_threat, away / (d + _EPS) ** 2, 0.0)
    F = F + 1.0 / (me + _EPS) - 1.0 / (ARENA - me + _EPS)
    q_box, q_cyl = _closest_points(me, arena)
    q = jnp.concatenate([q_box, q_cyl])
    on = jnp.concatenate([arena.box_on, arena.cyl_on])
    vec = me - q
    dist = jnp.linalg.norm(vec, axis=-1, keepdims=True)
    return F + jnp.sum(jnp.where(on[:, None], vec / (dist + _EPS) ** 2, 0.0), 0)


def _evader_one(me, my_vel, chaser, vis, arena):
    """OPEN'ın potansiyel alanı. Her kaynak, kaçanı kendinden uzağa, mesafeyle ters orantılı iter:
         F = Σ (uzaklaşma yönü) / mesafe
    Kaynaklar: kovalayan (sadece GÖRÜNÜYORSA), dört duvar, zemin + tavan, engeller. Hız sabit (= V_MAX),
    yönü F'nin yönü. Katsayı yok; hepsi eşit ağırlıkta.

    OPEN'dan farklar (gerekçeli):
      * Engeller: OPEN'da sadece sonsuz boy silindir var (xy'de merkezden uzaklaşma). Biz aynı formülü engelin
        EN YAKIN NOKTASINDAN hesaplıyoruz; tam boy silindirde bu OPEN'ınkiyle aynı, kutu/alçak/sarkan engellere
        de doğal olarak genelleşiyor.
      * Normalizasyon: OPEN kodu hızı eksen eksen bölüyor (norm(dim=1), boyutu 1 olan eksen) ve çaprazda
        hız √3 katına çıkıyor; makale "sabit hız" diyor. Makaledeki tanımı uyguluyoruz.
      * Fren filtresi: OPEN kaçanın hızını fizik motorunda doğrudan atıyor (anında döner); bizim drone gerçek
        kontrolcüyle uçuyor (fren ~0,4 m), o yüzden üstte safety_filter var.
    """
    # Kovalayan: (ben − o)/d² = birim uzaklaşma yönü × 1/d (sadece görünüyorsa)
    # Duvarlar, zemin, tavan: her yüzey kendi normal yönünde 1/mesafe ile iter
    # Engeller: en yakın noktadan uzaklaşma yönü × 1/mesafe
    F = _open_force(me, chaser, vis, arena)
    return safety_filter(_unit(F), G.rays(me, arena), my_vel)


F_FULL = 0.5   # OPEN + bellek: toplam kuvvet bundan büyükse tam hız (≈ tek başına 2 m'deki kovalayanın kuvveti)


def _evader_mem_one(me, my_vel, last_seen, arena):
    """OPEN + bellek (bizim eklediğimiz, daha güçlü baseline). OPEN'dan iki farkı var:
      1. Bellek: kovalayan görünmüyorsa yok sayılmaz; SON GÖRÜLDÜĞÜ yerden kaçılır (last_seen, görünürken
         gerçek konumdur). OPEN'da görünmeyen kovalayan unutuluyor ve kaçan köşede boşta kalıyordu.
      2. Hız kuvvetle orantılı (en fazla tam hız): kuvvetlerin dengelendiği noktada OPEN tam hızla ileri
         geri gidip geliyordu (titreşim, 'chattering'); burada kuvvet zayıflayınca yavaşlayıp havada durur.
    """
    F = _open_force(me, last_seen, True, arena)
    speed = jnp.minimum(1.0, jnp.linalg.norm(F) / F_FULL)
    return safety_filter(_unit(F) * speed, G.rays(me, arena), my_vel)


def evader_policy(es, mem, key):
    """Kaçanın eylemi (n, 3): sadık OPEN."""
    pos, vel = es.data.states.pos, es.data.states.vel
    return jax.vmap(_evader_one)(pos[:, EVADER], vel[:, EVADER], pos[:, CHASER], es.visible[:, EVADER],
                                 es.arena), mem


def evader_mem_policy(es, mem, key):
    """Kaçanın eylemi (n, 3): OPEN + bellek."""
    pos, vel = es.data.states.pos, es.data.states.vel
    return jax.vmap(_evader_mem_one)(pos[:, EVADER], vel[:, EVADER], es.last_seen[:, EVADER], es.arena), mem


EVADERS = {"open": evader_policy, "open_mem": evader_mem_policy}


# ---------------------------------------------------------------------------
# İkisi birlikte
# ---------------------------------------------------------------------------
def scripted_init(n):
    return chaser_init(n)


def scripted_actions(es, mem, key, evader="open"):
    """Her iki rol için elle yazılmış eylemler: (n, 2, 3) ve yeni bellek. evader: 'open' ya da 'open_mem'."""
    a_c, mem = chaser_policy(es, mem, key)
    a_e, _ = EVADERS[evader](es, None, key)
    return jnp.stack([a_c, a_e], 1), mem


def reset_mem(mem, done):
    """Biten oyunlarda belleği sıfırla (yeni oyun, yeni arena)."""
    fresh = chaser_init(done.shape[0])
    return jax.tree.map(lambda f, m: jnp.where(done.reshape((-1,) + (1,) * (m.ndim - 1)), f, m), fresh, mem)
