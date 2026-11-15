"""Maç izleme: kaydedilmiş bir maçı Crazyflow'un MuJoCo görüntüleyicisinde 3B olarak tekrar oynatır.

Neden "kayıt + tekrar oynatma"? Eğitimde binlerce maç GPU'da görüntüsüz oynanır; çizmek hepsini yüzlerce kat
yavaşlatırdı. Biz birkaç maçın konum/yönelim kaydını alıp (play.make_runner) sonra çiziyoruz.

İki çıktı:
  * video (mp4): solda eğik 3B görünüm, sağda üstten görünüm (saklanmayı görmek için)
  * canlı pencere: MuJoCo'nun etkileşimli penceresi; fareyle kamerayı döndürüp yakınlaştırabilirsin

Drone'lar gerçek Crazyflie modeli (~10 cm), 6 m arenada küçük kalır: etraflarına renkli bir hale
(kırmızı kovalayan, mavi kaçan) ve arkalarına iz çiziyoruz. Birbirlerini görürken aralarında yeşil çizgi var.
"""
import time

import imageio.v2 as imageio
import jax.numpy as jnp
import mujoco
import numpy as np
from matplotlib import font_manager
from crazyflow.sim import Sim
from crazyflow.sim.sim import sync_sim2mjx
from crazyflow.sim.visualize import draw_line
from PIL import Image, ImageDraw, ImageFont

from .rules import ARENA, CTRL_HZ, OUTCOME_NAMES

RED = np.array([0.90, 0.20, 0.20, 1.0])
BLUE = np.array([0.20, 0.45, 0.95, 1.0])
KIND_RGBA = {"tam boy": [0.55, 0.55, 0.58, 1.0], "alçak": [0.30, 0.70, 0.35, 1.0],
             "sarkan": [0.95, 0.60, 0.20, 1.0], "havada": [0.60, 0.40, 0.80, 1.0]}
H = float(ARENA[2])
TRAIL = 40          # izde kaç geçmiş nokta (her karede bir)

CAM_3D = dict(distance=8.6, azimuth=135.0, elevation=-40.0, lookat=np.array([3.0, 3.0, 0.3]))
# Üstten görünüm ortografik (perspektifsiz, harita gibi): perspektifte uzun engellerin tepesi kameraya yakın
# kaldığı için devasa görünüyordu. Ortografikte fovy = görünen dikey genişlik (m).
CAM_TOP = dict(distance=10.0, azimuth=90.0, elevation=-90.0, lookat=np.array([3.0, 3.0, 0.0]), ortho_fovy=6.6)
# Türkçe karakterli font: matplotlib'le gelen DejaVu Sans (Windows'ta da Colab'da da var)
_FONT = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 18)


def _kind(z0, z1):
    if z0 <= 1e-3 and z1 >= H - 1e-3:
        return "tam boy"
    if z0 <= 1e-3:
        return "alçak"
    if z1 >= H - 1e-3:
        return "sarkan"
    return "havada"


class Replayer:
    """Tek dünyalık, sadece çizim için bir Crazyflow simülasyonu. Fizik çalıştırmıyoruz; her karede
    drone'ları kayıttaki konum/yönelime koyup çiziyoruz."""

    def __init__(self, width=800, height=600):
        self.w, self.h = width, height
        self.sim = Sim(n_worlds=1, n_drones=2, device="cpu")
        self.sim.reset()

    def _pose(self, pos, quat):
        st = self.sim.data.states.replace(pos=jnp.asarray(pos)[None], quat=jnp.asarray(quat)[None])
        self.sim.data = self.sim.data.replace(states=st)
        self.sim.data, self.sim.mjx_data = sync_sim2mjx(self.sim.data, self.sim.mjx_data, self.sim.mjx_model)

    def _markers(self, arena, trail, visible):
        """Engeller, arena sınırı, haleler, izler, görüş çizgisi. MuJoCo işaretçileri her çizimden sonra
        silindiği için her karede yeniden eklenir."""
        v = self.sim.viewer.viewer
        eye = np.eye(3).ravel()
        boxes, box_on, cyls, cyl_on = arena
        for b in boxes[box_on]:
            v.add_marker(type=mujoco.mjtGeom.mjGEOM_BOX, pos=(b[0] + b[1]) / 2, size=(b[1] - b[0]) / 2,
                         mat=eye, rgba=np.array(KIND_RGBA[_kind(b[0, 2], b[1, 2])]), label="")
        for c in cyls[cyl_on]:
            v.add_marker(type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=np.array([c[0], c[1], (c[3] + c[4]) / 2]),
                         size=np.array([c[2], c[2], (c[4] - c[3]) / 2]), mat=eye,
                         rgba=np.array(KIND_RGBA[_kind(c[3], c[4])]), label="")
        # Arena sınırı: zeminde ve tavanda çerçeve çizgileri
        corners = np.array([[0, 0], [ARENA[0], 0], [ARENA[0], ARENA[1]], [0, ARENA[1]], [0, 0]])
        for z in (0.01, H):
            draw_line(self.sim, np.c_[corners, np.full(5, z)], rgba=np.array([1, 1, 1, 0.8]),
                      start_size=2.0, end_size=2.0)
        # Drone haleleri ve izleri
        for i, col in enumerate((RED, BLUE)):
            v.add_marker(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=trail[-1, i], size=np.full(3, 0.11), mat=eye,
                         rgba=col * np.array([1, 1, 1, 0.35]), label="")
            if len(trail) > 1:
                draw_line(self.sim, trail[:, i], rgba=col, start_size=0.5, end_size=4.0)
        if visible:
            draw_line(self.sim, trail[-1], rgba=np.array([0.3, 0.9, 0.3, 0.9]), start_size=1.5, end_size=1.5)

    def frame(self, pos, quat, arena, trail, visible, text):
        """Bir video karesi: solda 3B, sağda üstten görünüm, üstte bilgi yazısı."""
        self._pose(pos, quat)
        views = []
        vis_global = self.sim.mj_model.vis.global_
        fovy0 = vis_global.fovy
        for cam in (CAM_3D, CAM_TOP):
            if self.sim.viewer is None:
                self.sim.render("rgb_array", width=self.w, height=self.h, cam_config=CAM_3D)
            c = self.sim.viewer.viewer.cam
            c.distance, c.azimuth, c.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
            c.lookat[:] = cam["lookat"]
            ortho = "ortho_fovy" in cam
            c.orthographic = int(ortho)
            vis_global.fovy = cam["ortho_fovy"] if ortho else fovy0
            self._markers(arena, trail, visible)
            views.append(self.sim.render("rgb_array", width=self.w, height=self.h))
        vis_global.fovy = fovy0
        img = Image.fromarray(np.concatenate(views, 1))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, img.width, 34], fill=(20, 20, 30))
        d.text((12, 6), text, fill=(255, 255, 255), font=_FONT)
        return np.asarray(img)

    def close(self):
        self.sim.close()


def _game_arrays(rec, arena_set, start, end):
    """Kayıttan bir oyunun dilimini ve o oyunun arenasını çıkarır."""
    sl = slice(start, end + 1)
    pos, quat = np.asarray(rec["pos"][sl]), np.asarray(rec["quat"][sl])
    vis = np.asarray(rec["visible"][sl])
    k = int(np.asarray(rec["arena_idx"])[start])
    arena = tuple(np.asarray(arena_set[key][k]) for key in ("boxes", "box_on", "cyls", "cyl_on"))
    return pos, quat, vis, arena


def _caption(t, vis, outcome, done):
    s = f"t = {t / CTRL_HZ:4.1f} s   |   kırmızı: kovalayan, mavi: kaçan   |   " \
        f"{'birbirlerini GÖRÜYOR' if vis else 'görüş KAPALI (engel arkasında)'}"
    if done:
        s += f"   |   SONUÇ: {OUTCOME_NAMES[outcome]}"
    return s


def save_video(rec, arena_set, game, path, every=2, hold_s=1.5, replayer=None):
    """game = (başlangıç adımı, bitiş adımı, sonuç), play.split_games'ten. 50 Hz kayıt, her `every` adımda
    bir kare → every=2 ile 25 fps gerçek zaman. Sonda son kare hold_s saniye bekletilir."""
    start, end, outcome = game
    pos, quat, vis, arena = _game_arrays(rec, arena_set, start, end)
    rp = replayer or Replayer()
    fps = CTRL_HZ // every
    with imageio.get_writer(path, fps=fps, macro_block_size=8) as w:
        idx = list(range(0, len(pos), every))
        if idx[-1] != len(pos) - 1:
            idx.append(len(pos) - 1)
        for j in idx:
            trail = pos[max(0, j - TRAIL * every):j + 1:every]
            done = j == len(pos) - 1
            img = rp.frame(pos[j], quat[j], arena, trail, bool(vis[j]), _caption(j, vis[j], outcome, done))
            for _ in range(int(hold_s * fps) if done else 1):
                w.append_data(img)
    if replayer is None:
        rp.close()
    return path


def play_live(rec, arena_set, game, speed=1.0):
    """MuJoCo penceresinde gerçek zamanlı oynatır (fareyle kamera döner). Pencereyi kapatınca biter."""
    start, end, outcome = game
    pos, quat, vis, arena = _game_arrays(rec, arena_set, start, end)
    rp = Replayer()
    sim = rp.sim
    for j in range(len(pos)):
        t0 = time.perf_counter()
        rp._pose(pos[j], quat[j])
        if sim.viewer is None:
            sim.render("human", cam_config=CAM_3D)
        rp._markers(arena, pos[max(0, j - TRAIL * 2):j + 1:2], bool(vis[j]))
        sim.render("human")
        time.sleep(max(0.0, 1.0 / (CTRL_HZ * speed) - (time.perf_counter() - t0)))
    print("Sonuç:", OUTCOME_NAMES[outcome], f"({len(pos) / CTRL_HZ:.1f} s)")
    time.sleep(2.0)
    rp.close()
