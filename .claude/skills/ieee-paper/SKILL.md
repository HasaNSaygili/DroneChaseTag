---
name: ieee-paper
description: Ara sunum ya da 4 sayfalık final IEEE makalesini yazarken, derlerken veya sayfa sınırını kontrol ederken kullan.
---

# IEEE makale

- Şablon: `paper/IEEEtran.cls` (resmi V1.8b, Overleaf'teki ile aynı), `\documentclass[conference]{IEEEtran}`.
- Final makale **en fazla 4 sayfa**, en az bir ablasyon içerir (ders şartı, `docs/DL_Project_Hand-Out.pdf`).
- Derle ve kontrol et:
  ```
  cd paper && pdflatex -interaction=nonstopmode final.tex && pdflatex -interaction=nonstopmode final.tex
  pdfinfo final.pdf | grep Pages        # 4'ü geçmemeli
  grep -E "Overfull|undefined" final.log
  ```
- Bölüm iskeleti: Giriş (WCT + motivasyon) · İlgili çalışmalar (`docs/literatur_notlari.md`, karşılaştırma
  tablosu) · Yöntem (kurallar, gözlem, ödül, PPO, self-play) · Deneyler (yoğunluk taraması, Elo, taktik
  metrikleri) · Ablasyon · Sınırlılıklar (sadece simülasyon, ideal algılama) · Sonuç.
- Kaynakçada makale başlıklarını **tam** yaz (proposal'da yer için "RL" kısaltılmıştı).
- Her sayı `experiments/LOG.md`'deki bir koşuya dayanmalı. Şekiller vektör (PDF) olsun.
- Dil sade: hoca bu literatürü bilmiyor; terimleri ilk geçtiği yerde açıkla.
