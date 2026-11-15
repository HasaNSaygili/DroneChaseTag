# Drone Chase Tag — proje bağlamı

İki simüle drone, World Chase Tag (WCT) kurallarıyla engelli bir arenada 1'e 1 ebelemece oynuyor ve
self-play RL ile oynamayı sıfırdan öğreniyor. Derin Öğrenme dersi projesi (Hasan Saygılı, bireysel, Rota 2).
Proposal teslim edildi: `docs/proposal.pdf`. Ders şartları: `docs/DL_Project_Hand-Out.pdf`.

## Kullanıcıyla çalışma
- Türkçe, kısa ve sohbet havasında yaz. Uzun dosya/rapor sevmiyor; açıklamaları sohbette ver.
- Amaç RL'i **öğrenmek**. Yeni bir kavram (PPO, GAE, self-play, PFSP...) koda girerken 2–3 cümleyle ne
  olduğunu ve neden gerektiğini anlat. Kodu açıklamalı yaz, sihirli kısım bırakma.
- Kararı kullanıcıya ait olan şeyleri (kurallar, ablasyon seçimi, kapsam) değiştirmeden önce sor.

## Takvim (sabit)
| Tarih | İş |
|---|---|
| 13–22 Kasım | Oyun ortamı (arena, engeller, kurallar, görüş hattı) + elle yazılmış rakipler |
| 23 Kasım–6 Aralık | Elle yazılmış kaçana karşı PPO kovalayan; boş arenada self-play (doğrulama) |
| 7–18 Aralık | Engellerle self-play + rakip havuzu; **son derste ara sunum** (21 Aralık'tan önceki hafta) |
| 19 Aralık–teslim | Ablasyonlar, taktik analizi, videolar; 4 sayfa IEEE makale + kod (finaller 21 Aralık'ta başlıyor) |

Final makalesinde **en az bir ablasyon** zorunlu. Not ağırlığı %25.

## Hesap
- Eğitim **sadece Colab Pro (A100, yüksek RAM)**. Yerel bilgisayar Windows, 4 GB GPU: eğitim için değil.
- Kod bir Python paketi olarak yazılsın; Colab notebook'u repoyu `git clone` edip import etsin
  (bkz. `.claude/skills/colab-run`). Her şey önce CPU'da küçük boyutla test edilir, sonra Colab'a gider.

## Teknik yığın
- JAX + **Crazyflow 0.4.0 (sürüm sabit)**, PPO PureJaxRL tarzında (tek dosya, her şey `jit`/`scan` içinde).
- Crazyflow API notları (belgeler eski, bunlar 0.4.0'da doğrulandı):
  - `from crazyflow.sim import Sim, Dynamics` (`Physics` değil); `dynamics=` argümanı.
  - `Control.state` komutu 16 boyutlu: pos 0:3, vel 3:6, acc 6:9, quat xyzw 9:13 (**index 12 = w = 1**, yoksa NaN), rates 13:16.
  - Saf/jit uyumlu fonksiyonlar: `sim._step(data, n_steps)`, `sim._reset(data, sim.default_data, mask)`,
    `crazyflow.sim.functional.state_control(data, cmd)`. Durum: `data.states.pos/vel` (n_worlds, n_drones, 3).
  - Dinamik: önce `Dynamics.so_rpy` (soyutlanmış, hızlı), sonra `first_principles` + CTBR.
  - Drone–drone teması fizikte yok: ebeleme/çarpma mesafe kontrolüyle bizim kodda.
  - Crazyflow `SCIPY_ARRAY_API=1` ister; `chase_tag/__init__.py` bunu SciPy'den önce ayarlıyor.
  - Varsayılan drone `cf21B_500` (Crazyflie 2.1 Brushless, model 43 g); seçenekler: cf2x_L250/P250/T350, hb_x500.
  - Kontrolcünün itki modeli simdeki drone'la uyuşmuyor (gerçek firmware gibi): hedef = "konum + v·dt"
    verilirse integral terimi çalışmıyor, drone sıfır komutta 0,21 m/s ile yükseliyor. Çözüm (env.py):
    **tasmalı sanal hedef nokta**, LEASH = 0,1 m. Ölçüm: havada ≤ 8 cm sapma, 2 m/s'ye 0,3–0,4 s, fren 0,4 m,
    anlık tepe ~2,6 m/s ama 1 s'de en uzun yol düz uçuşla 2,09 m (hileler daha kısa). İvme sınırı denendi,
    sömürüyü azaltmadı, freni uzattı → kullanılmadı.

## Oyun kuralları v1 (değiştirmeden önce sor)
- Arena 6×6×2 m; zemin, tavan, duvarlar sınır. Süre 20 s, kontrol 50 Hz (1000 adım), fizik 500 Hz.
- Ebeleme: merkezler arası < 0,3 m. Kaçan 20 s dayanırsa kazanır. Puanı sadece kaçış getirir.
- Engele ya da sınıra değen **kaybeder, rakibi kazanır** (WCT: sahadan çıkan kaybeder).
- Eşit azami hız (2 m/s başlangıç). Görüş 360°: aradaki doğru parçasında engel yoksa rakip görülür;
  görünmezse son görülen konum + görünmeme süresi.
- Başlangıç: karşı köşeler, ~1 m yükseklik. Roller için iki ayrı ağ (kovalayan, kaçan).
- Ödül (kovalayan, kaçan): ebeleme (+1, −1); süre doldu (−1, +1); kaçan çarptı (+1, −1−c); kovalayan
  çarptı (−1−c, +1); ikisi çarptı (−1−c, −1−c). c = ek çarpma cezası (0,5). Küçük ipuçları: kovalayana
  yaklaşma, kaçana hayatta kalma (çok küçük).
- Engeller kutu ve dikey silindir; her biri sadece z aralığıyla tanımlı. Türler: tam boy (0→2), alçak
  (0→0,6–1,0; üstünden geçilir), sarkan (0,7–1,0→2; altından geçilir, WCT "threading"), havada duran
  platform (alt 0,6–1,0, kalınlık 0,2–0,5; ince ayaklı platformların basitleştirmesi).
- Aşamalar birikimli (karar 2026-11-15): (1) tam boy → (2) + alçak → (3) + sarkan + havada = nihai karışık.
- Değerlendirme maçı: 16 kovalamaca, kazanan bir sonraki turda kaçan olur (WCT formatı).

## Yöntem kararları (literatürden; ayrıntı `docs/literatur_notlari.md`)
- Gözlem: kendi konum/hız, rakibin (son görülen) göreli konumu, görünürlük bayrağı, ışın mesafeleri
  (16 yatay + yukarı/aşağı), kalan süre. Sonra: rakip geçmiş penceresi ya da GRU.
- Asimetrik critic: actor maskeli gözlemi, critic tam durumu görür.
- Self-play: taraflar sırayla eğitilir, eşik geçilince sıra değişir; rakip havuzu, maçların %50'si en
  son rakiple, %50'si PFSP ağırlığı (1−x)² ile.
- Müfredat: başarı oranı 0,5–0,9 olan arenalar; gerekirse ebeleme yarıçapı büyük başlayıp 0,3'e iner.
- Arenalar **önceden üretilir** (karar 2026-11-15): NumPy ile binlerce düzen üretilir, ızgarada BFS ile
  bağlantılılık kontrol edilir (kapalı cep olmasın, geçitler drone'un sığacağı genişlikte), geçenler dosyaya
  kaydedilir; JAX eğitimde sadece indeks seçer. Bir kısmı test için ayrılır (eğitimde hiç görülmez).
  Neden JAX içinde değil: "tutana kadar yeniden üret" döngüsü jit/sabit boyutla zor ve yavaş.
- Baseline: elle yazılmış kovalayan/kaçan + engelsiz eğitilmiş politika.
- Metrikler: kaçış oranı, sansürlü yakalama süresi, çarpma türleri, engel yoğunluğuna göre; Elo
  (round-robin, çift başına çok maç); taktik metrikleri (WCT sözlüğü tablosu notlarda, bölüm 10).
- Ablasyon önceliği: (1) havuz vs naive self-play, (2) maskeli vs tam gözlem, (3) engelli vs engelsiz
  eğitim, (4) seyrek vs yoğun ödül, (5) MLP vs GRU, (6) hız komutu vs CTBR, (7) çarpmada rakip 0 alır,
  (8) görüş hattı merkezden merkeze vs çok noktalı (rakip kürenin merkezi + yüzeyde 6 nokta; biri açıksa görünür).

## Mevcut durum
- 2026-11-15: implementasyon **sıfırdan** yazılıyor (kurallar ve yöntem kararları aynı kalıyor).
  Eski prototip referans olarak `drone_chase_tag_baslangic/handoff/src/chase_tag_prototype.py`'de
  (git'e girmiyor). Körü körüne kopyalama; aşağıdaki sorunlu noktaları yeniden düşün.
- Eski CPU ön sonucu (elle yazılmış rakipler, 64 arena): ebeleme oranı engel 0 → %100, 4 → %90,
  8 → %54. Küçük örnek; eski koddan, yeni kodla tekrarlanmalı.
- Prototipte bulunan sorunlar (yeni kodda çözülecek):
  - Başlangıç noktası bir kutunun içine düşebiliyordu (kutular 0,6 m'ye kadar uzanıyor, başlangıç 0,6 ± 0,25).
  - Bağlantılılık kontrolü yok (kapalı cep → kaçan bedavaya kazanır).
  - Elle yazılmış kovalayan rakibi kaybedince son görüldüğü yerde bekliyor (arama yok).
  - Sadece kutu engel; proposal silindir de diyor.
- A100 hız testi hiç koşmadı: hesap bütçesi bilinmiyor.
- Yazılanlar (testli): `rules.py`, `arenas.py`, `geometry.py`, `env.py`. Sırada: `scripted.py`
  (kovalayan: ızgarada en kısa yol + kaybedince arama; kaçan: potansiyel alan, OPEN).
- Arena yoğunluğu: 8 engel alanın sadece %10–15'ini kaplıyor (seyrek). Kaplama oranıyla ayarlama ve
  faz 3'te tam boy payını artırma fikri var; maçları izledikten sonra karar verilecek.
- Yerel ortam: `.venv` (Python 3.12, `pip install -e .`), testler `.venv\Scripts\python.exe -m pytest tests`.

## Klasör yapısı
```
src/chase_tag/   env.py (kurallar+ortam)  geometry.py  scripted.py  ppo.py  selfplay.py  eval.py
tests/           CPU'da küçük boyutlu testler (pytest)
notebooks/       Colab notebook'ları (repoyu clone edip src'yi import eder)
experiments/     LOG.md (her koşu bir satır) + koşu klasörleri
paper/           IEEEtran.cls, proposal.tex, final makale
docs/            proposal, ders dokümanı, literatür notları
```

## Dürüstlük kuralları
- Sonuçlar sadece simülasyonda; algılama ideal (konum + görüş maskesi). Makalede sınırlılık olarak yaz.
- Sayı uydurma: her tablo/grafik bir koşuya (`experiments/LOG.md`) dayanmalı.
