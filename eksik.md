# Eksikler

Ertelenen işler. Biri yapılınca buradan silinir, sonucu `experiments/LOG.md`'ye yazılır.

## Boş arena (engelsiz)

- [ ] **Self-play'i 2 seed daha koş** (`sp-empty-fix2` ayarları, seed 1 ve 2) ve her birine Elo turnuvası yap.
  Neden: şu an tek koşu var (`sp-empty-fix2-s0`). Engelli–engelsiz karşılaştırması (ablasyon 3) tek koşuya karşı
  tek koşuyla zayıf kalır, fark şans eseri olabilir. Maliyet: Colab'da ~14 dk/koşu, Elo CPU'da ~10 dk/koşu.
- [ ] **İki öğrenen ajanın maçlarını videoya çek ve izle.** Bunun için iki öğrenen ajanı karşı karşıya oynatıp video
  çeken bir script lazım (şu anki `watch_policy.py` sadece öğrenen ajan ile elle yazılmış rakibi oynatıyor).
  Bakılacaklar: son sürümlerin taktikleri (kaçan hâlâ dikey daire mi çiziyor?); kovalayanın "kaçan çarptı" kazançları
  (son nesillerde %6–16) gerçek bir sıkıştırma taktiği mi, sömürü mü? Neden: engelli arenada "engeller yeni taktik
  getirdi" diyebilmek için önce engelsiz taktikleri bilmek gerekiyor.
