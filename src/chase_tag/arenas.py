"""Arena üretimi: eğitimden önce, bir kez, sade NumPy ile.

Neden önceden? Eğitim JAX içinde derlenmiş, sabit boyutlu kodla çalışıyor. "Düzen kötüyse yeniden üret"
gibi kaç kez döneceği belli olmayan bir döngü orada hem zor yazılır hem yavaştır. Burada ise acele yok:
binlerce düzen üretip her birini kontrol ediyor, geçenleri kaydediyoruz. Eğitim sadece indeks seçer.

Her düzen iki kontrolden geçer:
  1. Dört köşe engelsiz (başlangıç noktaları bir engelin içine düşmesin).
  2. Bağlantılı: drone'un sığabileceği boş alan tek parça (kapalı cep yok). Yoksa kaçan oraya girip
     bedavaya kazanır ve RL ajanı bu açığı "taktik" diye öğrenir.

Engel gösterimi (JAX tarafı da aynısını kullanır, bkz. geometry.Arena):
  kutu:    (2, 3)  = [[x_min, y_min, z_min], [x_max, y_max, z_max]]   eksen hizalı
  silindir: (5,)   = [cx, cy, r, z_min, z_max]                           dikey
Sabit boyut için her arena N_BOX_MAX kutu ve N_CYL_MAX silindir yeri taşır; kullanılmayanlar maskeyle
kapatılır (box_on / cyl_on = False).
"""
import numpy as np
from scipy import ndimage

from .rules import ARENA

N_BOX_MAX = 10
N_CYL_MAX = 10

VOXEL = 0.1          # bağlantılılık ızgarasının çözünürlüğü (m)
PASSAGE_MIN = 0.4    # drone'un güvenle geçebileceği en dar boşluk (m); drone çapı ~0,12 m
INFLATE = PASSAGE_MIN / 2  # engelleri bu kadar şişirince, PASSAGE_MIN'den dar geçitler kapanır
CORNER_CLEAR = 1.0   # dört köşede engelsiz bırakılan karenin kenarı (m); başlangıçlar burada

# ---- Engel türleri: hepsi sadece z aralığıyla tanımlanır ----
H = float(ARENA[2])
KIND_Z = {
    "full":     lambda rng: (0.0, H),                        # yerden tavana: oyun neredeyse 2B
    "low":      lambda rng: (0.0, rng.uniform(0.6, 1.0)),    # alçak: üstünden geçilir (WCT atlama)
    "overhang": lambda rng: (rng.uniform(0.7, 1.0), H),      # tavandan sarkan: altından geçilir (threading)
    "float":    lambda rng: _float_z(rng),                   # havada duran platform: altından da üstünden de
}


def _float_z(rng):
    """Havada duran platform (WCT'deki ince ayaklı platformların basitleştirmesi; ayaklar modellenmiyor).
    Alt yüzey 0,6–1,0 m, kalınlık 0,2–0,5 m: altında da üstünde de drone'un sığacağı boşluk kalır."""
    z0 = rng.uniform(0.6, 1.0)
    return z0, z0 + rng.uniform(0.2, 0.5)


# Aşamalar birikimli: her aşama öncekinin türlerini de içerir; 3. aşama = nihai karışık arena.
PHASE_KINDS = {1: ("full",), 2: ("full", "low"), 3: ("full", "low", "overhang", "float")}


# ---------------------------------------------------------------------------
# Noktadan engele mesafe (NumPy). JAX'teki çarpma kontrolüyle aynı formül; testler ikisini karşılaştırır.
# ---------------------------------------------------------------------------
def dist_to_box(p, box):
    """p: (..., 3). Kutunun içindeyse 0. Kutudaki en yakın nokta = p'nin kutuya kırpılmış hali."""
    q = np.clip(p, box[0], box[1])
    return np.linalg.norm(p - q, axis=-1)


def dist_to_cyl(p, cyl):
    """Dikey silindir: yatayda daireye, dikeyde z aralığına olan taşma; ikisinin bileşkesi."""
    cx, cy, r, z0, z1 = cyl
    rho = np.hypot(p[..., 0] - cx, p[..., 1] - cy)
    d_xy = np.maximum(rho - r, 0.0)
    d_z = np.maximum(np.maximum(z0 - p[..., 2], p[..., 2] - z1), 0.0)
    return np.hypot(d_xy, d_z)


# ---------------------------------------------------------------------------
# Izgara ve kontroller
# ---------------------------------------------------------------------------
_shape = tuple(np.round(ARENA / VOXEL).astype(int))                 # (60, 60, 20)
_axes = [(np.arange(n) + 0.5) * VOXEL for n in _shape]               # voxel merkezleri
GRID = np.stack(np.meshgrid(*_axes, indexing="ij"), -1)              # (60, 60, 20, 3)

# Köşe bölgeleri: xy'de köşeye CORNER_CLEAR'dan yakın voxeller (bütün yükseklik boyunca)
_x, _y = GRID[..., 0], GRID[..., 1]
_near = lambda v, L: (v < CORNER_CLEAR) | (v > L - CORNER_CLEAR)
CORNER_MASK = _near(_x, ARENA[0]) & _near(_y, ARENA[1])


def obstacle_mask(boxes, cyls):
    """Şişirilmiş engellerin kapladığı voxeller (duvarlar hariç)."""
    m = np.zeros(_shape, bool)
    for b in boxes:
        m |= dist_to_box(GRID, b) < INFLATE
    for c in cyls:
        m |= dist_to_cyl(GRID, c) < INFLATE
    return m


def wall_mask():
    """Sınırlara INFLATE'ten yakın voxeller: duvar dibi de dar geçit gibi davranır."""
    return np.any((GRID < INFLATE) | (GRID > ARENA - INFLATE), axis=-1)


_WALLS = wall_mask()


def is_connected(boxes, cyls):
    """Drone merkezinin bulunabileceği boş alan tek parça mı? (6-komşulu flood fill)"""
    free = ~(obstacle_mask(boxes, cyls) | _WALLS)
    _, n_parts = ndimage.label(free)
    return n_parts == 1


# ---------------------------------------------------------------------------
# Rastgele düzen
# ---------------------------------------------------------------------------
def _sample_obstacle(rng, kinds, p_cyl):
    z0, z1 = KIND_Z[rng.choice(kinds)](rng)
    cx, cy = rng.uniform(0.3, ARENA[0] - 0.3), rng.uniform(0.3, ARENA[1] - 0.3)
    if rng.random() < p_cyl:
        return "cyl", np.array([cx, cy, rng.uniform(0.1, 0.35), z0, z1])
    hx, hy = rng.uniform(0.1, 0.7, size=2)  # yarı kenarlar; biri ince biri uzunsa duvar olur
    lo = np.clip([cx - hx, cy - hy, z0], 0.0, ARENA)
    hi = np.clip([cx + hx, cy + hy, z1], 0.0, ARENA)
    return "box", np.stack([lo, hi])


def sample_layout(rng, n_obst, kinds, p_cyl, max_tries=200):
    """n_obst engelli, köşeleri boş ve bağlantılı bir düzen. Bulamazsa None."""
    for _ in range(max_tries):
        boxes, cyls = [], []
        for _ in range(n_obst):
            for _ in range(50):  # köşeye taşan engeli hemen yeniden çek (bütün düzeni atmaya gerek yok)
                kind, ob = _sample_obstacle(rng, kinds, p_cyl)
                m = obstacle_mask([ob], []) if kind == "box" else obstacle_mask([], [ob])
                if not (m & CORNER_MASK).any():
                    (boxes if kind == "box" else cyls).append(ob)
                    break
        if (len(boxes) + len(cyls) == n_obst and len(boxes) <= N_BOX_MAX and len(cyls) <= N_CYL_MAX
                and is_connected(boxes, cyls)):
            return boxes, cyls
    return None


def pack_layout(boxes, cyls):
    """Engel listelerini sabit boyutlu dizilere yerleştirir; boş yerler maskeyle kapalı."""
    out = dict(boxes=np.zeros((N_BOX_MAX, 2, 3), np.float32), box_on=np.zeros(N_BOX_MAX, bool),
               cyls=np.zeros((N_CYL_MAX, 5), np.float32), cyl_on=np.zeros(N_CYL_MAX, bool))
    if len(boxes):
        out["boxes"][:len(boxes)] = boxes
        out["box_on"][:len(boxes)] = True
    if len(cyls):
        out["cyls"][:len(cyls)] = cyls
        out["cyl_on"][:len(cyls)] = True
    return out


def unpack_layout(arena_set, i):
    """pack_layout'un tersi: kümedeki i. arenanın etkin kutu ve silindir listeleri."""
    s = arena_set
    return list(s["boxes"][i][s["box_on"][i]]), list(s["cyls"][i][s["cyl_on"][i]])


def make_arena_set(n_arenas, phase=1, n_obst=(0, 8), p_cyl=0.3, seed=0, verbose=True):
    """n_arenas düzen üretir. Her düzenin engel sayısı n_obst aralığından (iki uç dahil) eşit olasılıkla.

    Döner: başında arena boyutu olan sabit boyutlu diziler (doğrudan JAX'e verilebilir) + üretim bilgisi.
    Aynı seed her zaman aynı arenaları verir: koşular tekrarlanabilir olsun.
    """
    rng = np.random.default_rng(seed)
    kinds = PHASE_KINDS[phase]
    packed, counts = [], []
    while len(packed) < n_arenas:
        k = int(rng.integers(n_obst[0], n_obst[1] + 1))
        layout = sample_layout(rng, k, kinds, p_cyl)
        if layout is None:
            continue  # bu sayıda engelle bağlantılı düzen bulunamadı; başka bir sayı çek
        packed.append(pack_layout(*layout))
        counts.append(k)
        if verbose and len(packed) % 1000 == 0:
            print(f"  {len(packed)}/{n_arenas} arena")
    out = {key: np.stack([p[key] for p in packed]) for key in packed[0]}
    out["n_obst"] = np.array(counts, np.int32)
    out["info"] = np.array(f"phase={phase} n_obst={n_obst} p_cyl={p_cyl} seed={seed}")
    return out


def save_arena_set(path, arena_set):
    np.savez_compressed(path, **arena_set)


def load_arena_set(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}
