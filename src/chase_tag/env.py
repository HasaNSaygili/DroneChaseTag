"""Oyun ortamı: Crazyflow fiziği + WCT kuralları, binlerce arena aynı anda.

RL'de "ortam" ajanın içinde yaşadığı dünyadır. Döngü:
    gözlem → (ağ) → eylem → ortam bir adım ilerler → ödül + yeni gözlem → ...
Bu dosya o döngünün dünya tarafı. Ağ ve öğrenme ayrı dosyalarda.

Tasarım:
  * Saf fonksiyonlar: step(durum, eylem, anahtar) → (yeni durum, ödül, bitti, bilgi). Durum bir pytree
    (EnvState); içeride gizli değişken yok. Böylece bütün oyun jax.jit + lax.scan ile tek parça derlenir
    ve GPU'da Python döngüsü olmadan döner.
  * Otomatik yeniden başlatma: biten arena aynı adımda yeni düzen + yeni başlangıçla tekrar başlar.
    Dönen gözlem o zaman YENİ oyunun ilk gözlemidir; öğrenme tarafı bunu `done` bayrağından bilir.
  * Süre dolması burada gerçek bir oyun sonu (kaçan kazanır) ve kalan süre gözlemin içinde. Yani
    "zaman yüzünden kesildi" (truncation) değil "oyun bitti" (termination): ayrıca bir düzeltme gerekmez.

Eksen düzeni: dizilerin başında dünya (arena) boyutu n, sonra drone boyutu 2 (0 = kovalayan, 1 = kaçan).
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Dynamics, Sim
import crazyflow.sim.functional as F

from . import geometry as G
from .rules import (ARENA, BOTH_CRASH, C_CRASH, CHASER, CTRL_HZ, E_CRASH, EVADER, K_APPROACH, K_SURVIVE,
                    MAX_STEPS, N_SUB, NONE, REWARD_TABLE, SIM_HZ, TAG, TAG_R, TIMEOUT, V_MAX)

# ---- Gözlem boyutları ----
# Aktör (politika) sadece kendi görebildiğini görür: rakip engel arkasındaysa son görüldüğü yer.
OBS_DIM = 3 + 3 + 3 + 1 + 1 + 3 + G.N_RAYS + 1          # = 33
# Critic (değer ağı, sadece eğitimde) her şeyi görür: rakibin gerçek konumu/hızı ve rakibin ışınları.
CRITIC_DIM = OBS_DIM + 3 + 3 + G.N_RAYS                   # = 57
UNSEEN_CAP = 5.0   # "kaç saniyedir görmüyorum" bu değerde doyar

# ---- Başlangıç: karşı köşeler (arenas.py köşeleri engelsiz bırakır) ----
_CORNERS = np.array([[0.6, 0.6], [5.4, 5.4], [0.6, 5.4], [5.4, 0.6]], np.float32)
START_Z = 1.0

# ---- Hız komutu → Crazyflie kontrolcüsü: "tasmalı sanal hedef nokta" ----
# Politika bir hız verir; bu hızla bir hedef nokta (ref) ilerletilir, kontrolcü o noktayı kovalar.
# Neden doğrudan "şimdiki konum + v·dt" değil (prototip)? Kontrolcünün itki modeli simülasyondaki
# drone'la tam uyuşmuyor (gerçek firmware'de de böyle); bunu kontrolcünün integral terimi düzeltir,
# ama o terim konum hatasına bakar. Hedef hep drone'un dibinde olunca hata hep ~0 kalıyor ve drone
# sıfır komutta bile 0,21 m/s ile tavana süzülüyordu. Hedef nokta bağımsız ilerleyince integral çalışır.
# Tasma: hedef drone'dan en fazla LEASH uzaklaşabilir; yoksa geride kalan drone yetişmek için çok
# hızlanır (ölçüm: tasmasız tepe 3,4 m/s). Ölçülen davranış (L = 0,1 m, so_rpy):
#   havada sapma ≤ 8 cm; 2 m/s'nin %90'ına 0,3 s; fren mesafesi 0,4 m;
#   anlık tepe hız ~2,6 m/s ama 1 s'de en uzun yol düz tam gazla 2,09 m (yön değiştirme hileleri
#   daha kısa) → sürdürülebilir hız iki drone için de 2 m/s.
LEASH = 0.1


class EnvState(NamedTuple):
    data: object           # Crazyflow SimData (fizik durumu: konum, hız, yönelim, kontrolcü iç durumu)
    ref: jax.Array         # (n, 2, 3) kontrolcünün kovaladığı sanal hedef nokta (bkz. LEASH)
    arena: G.Arena         # her dünyanın şu anki engelleri, (n, ...)
    arena_idx: jax.Array   # (n,)   arena kümesindeki indeks (analiz için)
    t: jax.Array           # (n,)   bu oyunda geçen adım
    visible: jax.Array     # (n, 2) drone i rakibini görüyor mu
    last_seen: jax.Array   # (n, 2, 3) drone i'nin rakibini son gördüğü MUTLAK konum
    unseen: jax.Array      # (n, 2) kaç saniyedir görmüyor
    dist: jax.Array        # (n,)   drone'lar arası mesafe (yaklaşma ipucu için bir önceki değer)


def _swap(x):
    """(n, 2, ...) dizide drone eksenini ters çevir: her drone için 'rakibin değeri'."""
    return x[:, ::-1]


def clip_speed(actions):
    """Eylem ∈ birim küre → hedef hız (m/s). Vektörün boyu kırpılır, eksenler ayrı ayrı değil:
    yoksa çapraz uçuşta hız V_MAX·√3'e çıkardı."""
    norm = jnp.linalg.norm(actions, axis=-1, keepdims=True)
    return actions / jnp.maximum(1.0, norm) * V_MAX


class ChaseTagEnv:
    """n_worlds arenada aynı anda oyun. Arenalar önceden üretilmiş kümeden (arenas.make_arena_set) çekilir.

    Sınıf sadece sabit şeyleri tutar (fizik fonksiyonları, arena kümesi); oyunun durumu EnvState'tedir.
    """

    def __init__(self, n_worlds, arena_set, dynamics=Dynamics.so_rpy, device=None, opp_crash_reward=1.0):
        """opp_crash_reward: biri çarptığında DİĞER tarafın aldığı ödül. 1.0 = WCT kuralı (sahadan çıkan
        kaybeder, rakibi kazanır). 0.0 = Gavin vd.'nin yolu: rakibi çarptırmak ödüllendirilmez (ablasyon 7).
        İlk kovalayan eğitiminde ajan elle yazılmış kaçanı rezonansla yere düşürmeyi öğrendi; 0.0 bunu kapatır."""
        self.n = n_worlds
        table = np.array(REWARD_TABLE)
        table[E_CRASH, CHASER] = opp_crash_reward
        table[C_CRASH, EVADER] = opp_crash_reward
        self.reward_table = table
        self.arena_set = G.arena_from_set(arena_set)           # (n_set, ...)
        self.n_set = self.arena_set.boxes.shape[0]
        device = device or ("gpu" if jax.default_backend() == "gpu" else "cpu")
        # Control.state: drone'a "şu konum ve hızda ol" deriz, Crazyflie'ın kendi kontrolcüsü (Mellinger)
        # motorları ayarlar. Yani biz sadece strateji katmanını öğreniyoruz (Tencent'in hiyerarşisi gibi).
        self.sim = Sim(n_worlds=n_worlds, n_drones=2, control=Control.state, dynamics=dynamics,
                       freq=SIM_HZ, state_freq=CTRL_HZ * 2, device=device)
        self.sim.reset()
        self._sim_step, self._sim_reset = self.sim._step, self.sim._reset
        self._default_data = self.sim.default_data

    # ------------------------------------------------------------------ başlatma
    def _sample_starts(self, key):
        """Rastgele bir köşegen, rollere rastgele köşe; küçük gürültü. Aralarında ~6 m."""
        k1, k2, k3 = jax.random.split(key, 3)
        diag = jax.random.bernoulli(k1, 0.5, (self.n,)).astype(jnp.int32)     # 0: (0,1) köşeleri, 1: (2,3)
        swap = jax.random.bernoulli(k2, 0.5, (self.n,)).astype(jnp.int32)
        a = jnp.asarray(_CORNERS)[2 * diag + swap]
        b = jnp.asarray(_CORNERS)[2 * diag + 1 - swap]
        xy = jnp.stack([a, b], 1)                                              # (n, 2, 2)
        noise = jax.random.uniform(k3, (self.n, 2, 3), minval=-1.0, maxval=1.0) * jnp.array([0.25, 0.25, 0.15])
        return jnp.concatenate([xy, jnp.full((self.n, 2, 1), START_Z)], -1) + noise

    def _reset_where(self, s, mask, key):
        """mask'teki dünyaları yeni oyuna başlatır; diğerlerine dokunmaz."""
        k1, k2 = jax.random.split(key)
        data = self._sim_reset(s.data, self._default_data, mask)   # fizik + kontrolcü iç durumu sıfırlanır
        m3 = mask[:, None, None]
        pos = jnp.where(m3, self._sample_starts(k1), data.states.pos)
        vel = jnp.where(m3, 0.0, data.states.vel)
        data = data.replace(states=data.states.replace(pos=pos, vel=vel))

        idx = jnp.where(mask, jax.random.randint(k2, (self.n,), 0, self.n_set), s.arena_idx)
        arena = jax.tree.map(lambda new, old: jnp.where(mask.reshape((-1,) + (1,) * (old.ndim - 1)), new[idx], old),
                             self.arena_set, s.arena)

        vis = jax.vmap(G.line_of_sight)(pos[:, 0], pos[:, 1], arena)
        visible = jnp.where(mask[:, None], vis[:, None], s.visible)
        # Başta rakibin başlangıç köşesini biliyoruz (WCT'de oyuncular birbirini görerek başlar)
        last_seen = jnp.where(m3, _swap(pos), s.last_seen)
        return EnvState(data=data, ref=jnp.where(m3, pos, s.ref), arena=arena, arena_idx=idx,
                        t=jnp.where(mask, 0, s.t),
                        visible=visible, last_seen=last_seen,
                        unseen=jnp.where(mask[:, None], 0.0, s.unseen),
                        dist=jnp.where(mask, jnp.linalg.norm(pos[:, 0] - pos[:, 1], axis=-1), s.dist))

    def reset(self, key):
        n = self.n
        empty = EnvState(data=self.sim.data, ref=jnp.zeros((n, 2, 3)), arena=jax.tree.map(lambda a: a[jnp.zeros(n, int)], self.arena_set),
                         arena_idx=jnp.zeros(n, jnp.int32), t=jnp.zeros(n, jnp.int32),
                         visible=jnp.zeros((n, 2), bool), last_seen=jnp.zeros((n, 2, 3)),
                         unseen=jnp.zeros((n, 2)), dist=jnp.zeros(n))
        return self._reset_where(empty, jnp.ones(n, bool), key)

    # ------------------------------------------------------------------ gözlem
    def _rays(self, s):
        """(n, 2, N_RAYS): her drone'un 18 ışın mesafesi."""
        per_drone = jax.vmap(G.rays, (0, None))                 # aynı arenada iki drone
        return jax.vmap(per_drone)(s.data.states.pos, s.arena)  # n arena

    def observe(self, s, rays=None):
        """Aktör gözlemi (n, 2, OBS_DIM), her drone kendi bakış açısından. Değerler kabaca [-1, 1]."""
        pos, vel = s.data.states.pos, s.data.states.vel
        rays = self._rays(s) if rays is None else rays
        vis = s.visible[..., None]
        t_left = jnp.broadcast_to((1.0 - s.t / MAX_STEPS)[:, None, None], vis.shape)
        return jnp.concatenate([
            pos / ARENA * 2 - 1,                              # kendi konumum
            vel / V_MAX,                                      # kendi hızım
            (s.last_seen - pos) / ARENA[0],                   # rakibe (ya da son görüldüğü yere) vektör
            vis.astype(jnp.float32),                          # şu an görüyor muyum
            jnp.minimum(s.unseen, UNSEEN_CAP)[..., None] / UNSEEN_CAP,  # kaç saniyedir görmüyorum
            jnp.where(vis, _swap(vel), 0.0) / V_MAX,          # rakibin hızı (sadece görüyorsam)
            rays / G.RAY_MAX,                                 # etrafımdaki engeller
            t_left,                                           # kalan süre
        ], -1)

    def critic_obs(self, s):
        """Critic gözlemi (n, 2, CRITIC_DIM): aktör gözlemi + gizli bilgiler (asimetrik critic).
        Critic sadece eğitimde 'bu durum ne kadar iyi' tahmini için kullanılır; oyunda drone'u o yönetmez.
        Bu yüzden ona hile serbest: tahmini ne kadar iyiyse öğrenme o kadar kararlı (OpenAI saklambaç)."""
        pos, vel = s.data.states.pos, s.data.states.vel
        rays = self._rays(s)
        return jnp.concatenate([
            self.observe(s, rays),
            (_swap(pos) - pos) / ARENA[0],                    # rakibin GERÇEK göreli konumu
            _swap(vel) / V_MAX,                               # rakibin gerçek hızı
            _swap(rays) / G.RAY_MAX,                          # rakibin etrafı (köşede mi, sıkışmış mı)
        ], -1)

    # ------------------------------------------------------------------ adım
    def step(self, s, actions, key):
        """actions: (n, 2, 3), her bileşen kabaca [-1, 1] → hedef hız.

        Döner: (yeni durum, ödül (n, 2), bitti (n,), bilgi). bilgi['outcome'] ve bilgi['length'] biten
        oyunların sonucunu ve süresini (adım) verir; değerlendirme metrikleri bunlardan hesaplanır.
        """
        data = s.data
        v = clip_speed(actions)
        # Sanal hedef noktayı v ile ilerlet, sonra tasmayla drone'un LEASH yakınına çek
        ref = s.ref + v / CTRL_HZ
        off = ref - data.states.pos
        ref = data.states.pos + off * jnp.minimum(1.0, LEASH / jnp.maximum(jnp.linalg.norm(off, axis=-1, keepdims=True), 1e-9))
        # Komut: "şu noktada, şu hızla ol". Kuaterniyon (x,y,z,w) = (0,0,0,1): düz dur (w=0 olursa NaN).
        cmd = jnp.zeros((self.n, 2, 16))
        cmd = cmd.at[..., 0:3].set(ref).at[..., 3:6].set(v).at[..., 12].set(1.0)
        data = self._sim_step(F.state_control(data, cmd), N_SUB)    # 10 fizik adımı = 1/50 s

        pos = data.states.pos
        t = s.t + 1
        dist = jnp.linalg.norm(pos[:, 0] - pos[:, 1], axis=-1)
        crash = jax.vmap(jax.vmap(G.crashed, (0, None)))(pos, s.arena)   # (n, 2)
        c_cr, e_cr = crash[:, CHASER], crash[:, EVADER]
        # Öncelik sırası: aynı adımda ebeleme ve çarpma olursa ebeleme sayılır (dokunuş önce oldu).
        outcome = jnp.select([dist < TAG_R, c_cr & e_cr, e_cr, c_cr, t >= MAX_STEPS],
                             [TAG, BOTH_CRASH, E_CRASH, C_CRASH, TIMEOUT], NONE)
        done = outcome != NONE

        # Ödül = sonuç tablosu (asıl sinyal) + küçük ipuçları.
        # Yaklaşma ipucu K·(önceki mesafe − yeni mesafe) "potansiyel tabanlı": bir oyun boyunca toplamı
        # sadece başlangıç ve bitiş mesafesine bağlı. Bu yüzden ajanı yanlış bir hedefe saptıramaz
        # (Ng vd. 1999), sadece "yaklaşmak iyi" ipucunu erken verir.
        r_shape = jnp.stack([K_APPROACH * (s.dist - dist), jnp.full(self.n, K_SURVIVE)], -1)
        reward = jnp.asarray(self.reward_table)[outcome] + r_shape

        # Görüş belleği: görüyorsa rakibin yerini güncelle, görmüyorsa son bilinen yer kalır
        vis = jax.vmap(G.line_of_sight)(pos[:, 0], pos[:, 1], s.arena)
        visible = jnp.broadcast_to(vis[:, None], (self.n, 2))       # görüş simetrik: biri görüyorsa öteki de
        last_seen = jnp.where(visible[..., None], _swap(pos), s.last_seen)
        unseen = jnp.where(visible, 0.0, s.unseen + 1.0 / CTRL_HZ)

        s = EnvState(data=data, ref=ref, arena=s.arena, arena_idx=s.arena_idx, t=t, visible=visible,
                     last_seen=last_seen, unseen=unseen, dist=dist)
        kind = jax.vmap(jax.vmap(G.crash_kind, (0, None)))(pos, s.arena)        # (n, 2) nereye çarptı
        info = dict(outcome=outcome, length=t, arena_idx=s.arena_idx, crash_kind=kind * crash)
        s = self._reset_where(s, done, key)
        return s, reward, done, info
