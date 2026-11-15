# Deney kaydı

| Tarih | Koşu | Commit | Config | Seed | Durum / sonuç |
|---|---|---|---|---|---|
| 2026-11-09 | cpu-scripted-smoke | — | elle yazılmış vs elle yazılmış, 64 arena, 0/4/8 kutu, so_rpy | 0 | Ebeleme %100 / %90 / %54 (küçük örnek, sadece duman testi; ESKİ prototip kodu) |
| 2026-11-15 | colab-speedtest-a100 | 6f48eb1 | scripts/speedtest.py; A100-SXM4-40GB, JAX 0.11.1; faz 3 arenalar, so_rpy, 2×256 MLP, T=128 | 0 | Rollout 0.04 / 0.19 / 0.72 / 1.20 M adım/s (1024 / 4096 / 16384 / 32768 dünya); 128 adım hep ~3 s (GPU doymamış, adım başı ~23 ms sabit); PPO güncellemesi 0.03–0.46 s; toplam ≈ 1.06 M adım/s @32768 → 1e9 adım ≈ 16 dk. Elle yazılmış rakipler 0.10 M adım/s @4096. Arena üretimi 1000 arena ≈ 39 s (CPU). |
