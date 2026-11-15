"""Drone Chase Tag: WCT kurallarıyla 1'e 1 drone ebelemecesi, self-play RL."""
import os

# Crazyflow, SciPy'nin bu ayarla yüklenmesini şart koşuyor; arenas.py SciPy'yi Crazyflow'dan önce
# yükleyebildiği için ayarı paketin en başında yapıyoruz.
os.environ.setdefault("SCIPY_ARRAY_API", "1")
