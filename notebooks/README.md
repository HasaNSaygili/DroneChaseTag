# Notebook'lar

Her koşunun kendi notebook'u `runs/` altında; kod o koşunun sürümüne sabitli. Colab'da açmak için linke tıkla,
*Çalışma zamanı → A100 GPU* seç, hücreleri sırayla çalıştır. Sonuçlar: `../experiments/LOG.md`.

| Koşu | Colab |
|---|---|
| `eval-sp-faz1-evstd-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/eval-sp-faz1-evstd-s0.ipynb) |
| `ppo-chaser-empty-nocrashrew-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/ppo-chaser-empty-nocrashrew-s0.ipynb) |
| `ppo-chaser-empty-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/ppo-chaser-empty-s0.ipynb) |
| `ppo-chaser-faz1-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/ppo-chaser-faz1-s0.ipynb) |
| `ppo-evader-empty-nocrashrew-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/ppo-evader-empty-nocrashrew-s0.ipynb) |
| `ppo-evader-faz1-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/ppo-evader-faz1-s0.ipynb) |
| `sp-empty-anchor-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sp-empty-anchor-s0.ipynb) |
| `sp-empty-ent-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sp-empty-ent-s0.ipynb) |
| `sp-empty-fix-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sp-empty-fix-s0.ipynb) |
| `sp-empty-fix2-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sp-empty-fix2-s0.ipynb) |
| `sp-empty-kl-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sp-empty-kl-s0.ipynb) |
| `sp-empty-lr1e4-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sp-empty-lr1e4-s0.ipynb) |
| `sp-empty-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sp-empty-s0.ipynb) |
| `sp-faz1-anchor-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sp-faz1-anchor-s0.ipynb) |
| `sp-faz1-anchor2-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sp-faz1-anchor2-s0.ipynb) |
| `sp-faz1-event-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sp-faz1-event-s0.ipynb) |
| `sp-faz1-evstd-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sp-faz1-evstd-s0.ipynb) |
| `sp-faz1-fix2-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sp-faz1-fix2-s0.ipynb) |
| `sp-faz1-kl-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sp-faz1-kl-s0.ipynb) |
| `sp-faz1-lr1e4-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sp-faz1-lr1e4-s0.ipynb) |
| `sp-faz1-s0` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sp-faz1-s0.ipynb) |
| `sweep-base-empty` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sweep-base-empty.ipynb) |
| `sweep-base-faz1` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sweep-base-faz1.ipynb) |
| `sweep-ent-chaser-faz1` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sweep-ent-chaser-faz1.ipynb) |
| `sweep-obs-chaser-faz1` | [aç](https://colab.research.google.com/github/HasaNSaygili/DroneChaseTag/blob/main/notebooks/runs/sweep-obs-chaser-faz1.ipynb) |

Diğerleri: `train.ipynb` (tek koşu şablonu), `sweep.ipynb` (tarama şablonu: birkaç koşu arka arkaya + karşılaştırma), `eval.ipynb` (bir self-play koşusunun değerlendirmesi: Elo + unutma testi, GPU; runs/ altında `eval-<koşu>`), `speedtest.ipynb` (A100 hız testi).
