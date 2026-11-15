---
name: experiment-log
description: Bir eğitim, değerlendirme ya da ablasyon koşusu başlatırken veya sonucu geldiğinde kullan. Her koşuyu experiments/LOG.md'ye kaydeder ki makaledeki her sayı bir koşuya dayansın.
---

# Deney kaydı

Makaledeki her tablo ve grafik bir koşuya dayanmalı. Her koşu için:

1. **Başlatmadan önce** `experiments/LOG.md` tablosuna bir satır ekle:
   `| tarih | koşu adı | git commit | config (engel aşaması, n_worlds, self-play türü, ablasyon) | seed | durum |`
   Koşu adı kısa ve anlamlı: `sp-pool-faz1-s0`, `abl-nopool-faz1-s0` gibi.
2. Config'i `experiments/<koşu adı>/config.json` olarak kaydet. Seed her zaman belirtilir.
3. **Sonuç gelince** aynı satırı güncelle: kaçış oranı, ortalama yakalama süresi, çarpma oranı, Elo
   (varsa), toplam adım ve süre. Ham sayıları `experiments/<koşu adı>/metrics.json`'a yaz.
4. Ablasyonlarda karşılaştırılan koşular aynı seed'leri ve aynı adım bütçesini kullanır; en az 3 seed hedefle.
5. Beklenmedik bir şey olduysa (çöküş, reward hacking, garip taktik) satıra bir cümlelik not düş.

Sayı uydurma ya da tahmin yazma: koşu yoksa tablo hücresi boş kalır.
