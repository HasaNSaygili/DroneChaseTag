"""Oyun kuralları v1: bütün sabitler tek yerde (CLAUDE.md'deki tabloyla birebir).

Bir kuralı değiştirmek gerekirse sadece bu dosya değişir. Sabitler NumPy dizisi; JAX kodunda
doğrudan kullanılabilirler (derleme sırasında sabit olarak gömülürler).
"""
import numpy as np

# ---- Arena ve zaman ----
ARENA = np.array([6.0, 6.0, 2.0])   # (x, y, z) m; zemin, tavan ve dört duvar sınır
CTRL_HZ = 50                         # politika saniyede 50 karar verir
SIM_HZ = 500                         # fizik saniyede 500 adım (kararlı olsun diye daha sık)
N_SUB = SIM_HZ // CTRL_HZ            # her politika adımında 10 fizik adımı
EPISODE_S = 20.0                     # WCT kovalamacası 20 s
MAX_STEPS = int(EPISODE_S * CTRL_HZ) # = 1000 politika adımı

# ---- Drone ve temas ----
DRONE_R = 0.06    # drone'u bu yarıçapta bir küre sayıyoruz; engele/sınıra bundan yakınsa çarptı
TAG_R = 0.3       # merkezler arası mesafe bundan küçükse ebeleme (WCT'de el ile dokunma)
V_MAX = 2.0       # iki drone için eşit azami hız (m/s): sonucu hız değil taktik belirlesin

# ---- Roller ----
CHASER, EVADER = 0, 1

# ---- Sonuçlar ve ödüller ----
# Her adımın sonunda oyun bu altı durumdan birinde olur.
NONE, TAG, TIMEOUT, E_CRASH, C_CRASH, BOTH_CRASH = range(6)
OUTCOME_NAMES = ["devam", "ebeleme", "süre doldu", "kaçan çarptı", "kovalayan çarptı", "ikisi çarptı"]

CRASH_PEN = 0.5   # ek çarpma cezası: çarpmak, normal kaybetmekten daha kötü olsun
_P = 1.0 + CRASH_PEN
# REWARD_TABLE[sonuç] = (kovalayanın ödülü, kaçanın ödülü)
REWARD_TABLE = np.array([
    [0.0, 0.0],    # devam: oyun bitmedi, sonuç ödülü yok
    [1.0, -1.0],   # ebeleme: kovalayan kazanır
    [-1.0, 1.0],   # süre doldu: kaçan kazanır
    [1.0, -_P],    # kaçan çarptı: kovalayan kazanır (WCT: sahadan çıkan kaybeder)
    [-_P, 1.0],    # kovalayan çarptı: kaçan kazanır
    [-_P, -_P],    # ikisi aynı adımda çarptı: kazanan yok
])

# ---- Küçük ipuçları (reward shaping) ----
# Asıl sinyal yukarıdaki ±1. Bunlar sadece başta yol göstersin diye çok küçük tutuldu.
K_APPROACH = 0.05   # kovalayan: mesafedeki azalma başına
K_SURVIVE = 0.0005  # kaçan: hayatta kaldığı her adım için
