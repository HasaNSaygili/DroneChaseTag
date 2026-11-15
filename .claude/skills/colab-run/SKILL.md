---
name: colab-run
description: Kodu Colab A100'de çalıştırmak gerektiğinde kullan (eğitim, hız testi, ablasyon). Yerel CPU testinden Colab notebook'una geçiş adımları.
---

# Colab'da çalıştırma

Eğitim yerelde değil, Colab Pro A100'de yapılır. Kullanıcı notebook'u kendisi çalıştırır; sen hazırlarsın.

1. **Önce CPU'da küçük test:** aynı kodu 64 arena, birkaç yüz adımla yerelde çalıştır. Hata yoksa devam.
2. **Kodu repoya koy:** değişiklikleri commit et ve push'la (kullanıcı onaylarsa). Notebook kodu kopyalamaz,
   repoyu çeker:
   ```
   !git clone <repo-url> proje && cd proje && git checkout <commit>
   !pip install -q "crazyflow[gpu]==0.4.0" -e .
   ```
   Kurulumdan sonra import hata verirse: Çalışma zamanı → Oturumu yeniden başlat, kurulum hücresini atla.
3. **Notebook'u kısa tut:** 1) GPU kontrol (`!nvidia-smi`), 2) kurulum, 3) tek bir `run(config)` çağrısı,
   4) sonuçları `results/<koşu-adı>/` altına kaydet ve Google Drive'a kopyala (Colab oturumu kapanınca silinir).
4. **Sonuç dönüşü:** notebook en sonda kısa bir özet yazdırsın; kullanıcıdan o çıktıyı yapıştırmasını iste.
5. Gelen sonucu `experiment-log` skill'iyle `experiments/LOG.md`'ye işle.

Uzun eğitimlerde her N iterasyonda checkpoint kaydet; Colab bağlantısı koparsa kaldığı yerden devam edebilsin.
