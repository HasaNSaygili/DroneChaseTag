# Literatür Notları: Drone Chase Tag

Nov 9, 2026 · @hasan

## Karşılaştırma tablosu

Her makaleyi okudukça bir satır eklenecek.

| Çalışma | Ne yapmış | Engel | Self-play | Eksik / bizim farkımız |
| --- | --- | --- | --- | --- |
| [AgilePE](https://arxiv.org/abs/2608.14135) (2026) | 1v1 drone kovalamaca, CTBR, MAPPO, OmniDrones | Yok (sadece görüş maskesinde) | SP, FSP, PFSP | Engelsiz arena; dokunma kuralı yok; taktikler ölçülmemiş |
| [KTH](https://arxiv.org/html/2506.02849v2) (2025) | 1v1 drone kovalamaca, 0,5 m yakalama, 8 s; CTBR ve hız komutu karşılaştırması; Isaac Sim | Yok | Sıralı eğitim + rakip havuzu (p\_old = 0,75) | Engelsiz; yalnızca simülasyon; taktik analizi yok |
| [Tencent](https://arxiv.org/html/2308.15143v2) (2024) | WCT'den esinlenen kovalamaca, dört ayaklı robot, 3 katmanlı hiyerarşi, bayrakla rol değişimi | Küçük (çit, sırık), 4,5 m arena | PFSP + Elo turnuvası | Drone değil; büyük engel yok; taktikler sadece nitel |
| [OpenAI saklambaç](https://arxiv.org/pdf/1909.07528) (2020) | Takımlı saklambaç, 2B yürüyen ajanlar, altı taktik aşaması; taktikler davranış istatistikleriyle izlenmiş | Var: duvarlar, taşınabilir kutu ve rampa | Self-play (PPO), merkezi critic | Robot/drone değil; çok büyük hesap; kovalamaca değil saklambaç |
| [OPEN](https://arxiv.org/html/2409.15866v4) (RA-L 2025) | 3 drone 1 kaçanı kovalıyor; CTBR, MAPPO + kaçan tahmini + uyarlamalı ortam üretici; gerçek Crazyflie | Var ama seyrek: 4–5 silindir, küçük arena | Yok: kaçan sabit potansiyel alan scripti | Kaçan öğrenmiyor; 1v1 değil; yoğun engel yok |
| [DualCL](https://arxiv.org/html/2312.12255v2) (2023) | N drone hızlı bir kaçanı kovalıyor; yakalama yarıçapı ve kaçan hızı üzerinden müfredat; OmniDrones | Var ama seyrek: 0–3 silindir | Yok: kaçan sabit script | OPEN ile aynı eksikler |
| [Gavin vd. 1v1](https://arxiv.org/pdf/2603.16279) (2026) | Ağlı drone ile 1v1 havada yakalama; CTBR, PPO; JAX simülatör, 2×10⁹ adım yaklaşık 1,5 saat (RTX 4090) | Yok | Birlikte eğitim, havuz yok (aşırı uyum) | Engelsiz; havuz yok; engelli ortam gelecek iş |
| [Gavin ve Bronz 3v1](https://arxiv.org/pdf/2607.05939) (2026) | 3 ağlı drone 1 kaçana karşı; MAPPO; eşikle sıralı eğitim | Yok | Havuz: %50 en son rakip + %50 PFSP | Engelsiz; 1v1 değil; taktikler sadece nitel |
| [UPenn yarış](https://arxiv.org/html/2512.11781v1) (2025) | İki drone yarışıyor; seyrek "kazan" ödülü yoğun ödülü yeniyor; Isaac Lab, gerçek Crazyflie | Var: parkurda 2–4 engel | IPPO birlikte eğitim, havuz yok | Kovalamaca değil yarış; taktikler ölçülmemiş |
| [CRUISE](https://arxiv.org/abs/2510.22570v1) (2025) | 2–4 drone yarışı; aşamalı müfredat + tekrarlı self-play; kendi Gym ortamı | Yok | Dondurulmuş rakipler, eşikle güncelleme | Kovalamaca değil; gerçek test yok |
| [Sandia / GT](https://coogan.ece.gatech.edu/papers/pdf/llanes2026marl.pdf) (ICRA 2026) | 2v2 ve 4v4 "ebelemece": kaçanlar hedefe ulaşmaya çalışıyor; MAPPO; gerçek Crazyflie | Yok | Yok: kovalayanlar sabit (PP, APN) | Kovalayan öğrenmiyor; engelsiz; gerçek sonuç nitel |
| [SJTU](https://arxiv.org/pdf/2407.10648) (2025) | Derinlik kamerasıyla engelli ortamda 20 m/s uçuş; nokta kütle + türevlenebilir fizik | Var: yoğun orman, hareketli engeller | Yok (tek ajan ve sürü, rekabet yok) | Kovalamaca yok; bizim için basit dinamik seçiminin dayanağı |

## 1. AgilePE: özet ve çıkarımlar

En yakın çalışma bu: 1v1 drone kovalamacasında iki ajan da self-play ile öğreniyor, ama deneylerin hepsi engelsiz bir arenada yapılmış. [arXiv 2608.14135](https://arxiv.org/abs/2608.14135) · Tang ve ark. (Tsinghua, Zhejiang, Chongqing), Ağustos 2026.

**Ne yapmışlar**

- **Problem:** 1v1 kovalamaca. Üç eksiği hedefliyor: yüksek seviyeli komutlar çevikliği sınırlıyor, self-play kararsız, sim2real zor.
- **Kurulum:** 6×6×3 m arena, OmniDrones üzerinde 2048 paralel arena. Hedef dokunmak değil, rakibi kameranın görüş alanında (FOV) tutmak. Ana metrik FOV içinde kalma oranı.
  - Yakalama koşulu ve bölüm süresi yazılmamış.
- **Kontrol:** CTBR (itki + gövde hızları), 60 Hz. Dinamik RK4 ile 10 alt adımda çözülüyor.
  - Gecikme: itki 45 ms, açısal hız 30 ms.
  - Gürültü: itki ±%10, açısal hız ±%30.
- **Gözlem:** kendi durumu ve rakibin geçmiş penceresi. Rakip konik FOV dışındaysa ya da silindir arkasında kaldıysa maskeleniyor. Engeller yalnızca bu gözlem modelinde var.
- **Ödül:** mesafe terimi d̃ (rakip FOV dışındaysa D\_max) ile hız, kenar, aşırı yakınlık ve yumuşaklık cezaları.
- **Öğrenme:** MAPPO ile naive SP, FSP ve PFSP. Hiperparametreler ve ağ mimarisi verilmemiş.

**Sonuçlar**

- Naive SP çöküyor: kaçan baskın çıkıyor (>0,9).
- FSP: kaçan 0,28'den 0,81–0,82'ye çıkıyor, kovalayan geriliyor.
- PFSP daha dengeli, ama kovalayan dalgalanıyor (6. iterasyonda 0,58'den 0,30'a düşüyor).
- Gözlenen taktikler: dash-and-flank ve mesafe yönetimi. Yalnızca nitel anlatılmış.
- Gerçek dünya: Jetson Orin NX + PX4, zero-shot transfer. Sayısal sonuç yok.
- Kendi yazdıkları sınırlar: deneyler engelsiz ve sadece 1v1, karmaşık ortam gelecek iş, gözlem görüntüden değil durumdan.

**Bizim için çıkarımlar**

1. Boşluk gerçek. Engeller sadece görüş maskesinde var, arenada değil; makale de engelli ortamı gelecek iş olarak yazıyor.
2. Hazır alınabilecek tasarım parçaları: arena boyutu, ceza terimleri, FOV-mesafe ödülü, gecikme ve gürültü değerleri.
3. Bizim oyunumuz farklı: net bir dokunma kuralı ve 20 s'lik turlar var (WCT).
4. Self-play dengesizliği gerçek bir risk. İki tarafın sonucunu da raporla, rakip havuzu kullan.
5. Taktikleri ölçmemişler. Bizim taktik metriklerimiz (görüş kırılması, sıkıştırma) burada katkı olur.
6. Metin ile tablo arasında tutarsızlık var (ör. metindeki 0,88 "yakalama oranı"). Atıf yaparken tablo değerlerini kullan.

## 2. KTH: özet ve çıkarımlar

Kural seti bizimkine çok yakın: 0,5 m'de yakalama var, süre dolarsa kaçan kazanıyor. Ama arena engelsiz ve testler yalnızca simülasyonda. [arXiv 2506.02849](https://arxiv.org/html/2506.02849v2) · Sánchez Roncero, Cai, Andersson, Ögren (KTH), 2025.

**Ne yapmışlar**

- **Kurulum:** 10×10×4 m engelsiz arena. Isaac Sim'de 62,5 Hz, 256 paralel ortam. Drone AscTec Hummingbird (0,716 kg), tam 6-DoF dinamik.
- **Kurallar:** aradaki mesafe 0,5 m'nin altına inerse yakalama. Bölüm en fazla 600 adım (yaklaşık 8 s). Süre dolarsa kaçan kazanır. Kaçan arenadan çıkarsa cezalandırılır.
- **Kontrol:** CTBR (\[ωx, ωy, ωz, T/m\]) ile hız komutu karşılaştırılmış. CTBR'yi bir PID motor hızlarına çeviriyor.
- **Gözlem:**
  - Rakibin şu anki ve son 4 adımdaki göreli konumu.
  - Kendi dönüş matrisi, hızı ve açısal hızları.
  - Konum: kaçan için global konum, kovalayan için sadece yükseklik.
  - Kalan süre.
- **Ödül (sıfır toplamlı değil):**
  - Kovalayan: yaklaşma (mesafedeki azalma) ×0,05 ve yakalama +10.
  - Kaçan: her adımda hayatta kalma +0,005 ve süre dolunca +10.
  - İkisine de: açısal hız cezası ×0,0005, sınır ihlali ×0,1.
- **Öğrenme:** PPO, 3 katmanlı MLP. Tam hiperparametre tablosu var: lr 5e-4, clip 0,1, γ 0,99, λ 0,95, batch 2048, 4 epoch, entropi 0,001.
- **AMSPB:**
  - Taraflar sırayla eğitiliyor.
  - Rakip %75 olasılıkla havuzdaki eski politikalardan seçiliyor (p\_old = 0,75), aksi halde en günceli.
  - Havuz sezgisel politikalarla (Hover, Circular, Repel) başlıyor.

**Sonuçlar**

- **Rakip havuzu unutmayı engelliyor (6. aşama, yakalama oranı):**
  - Repel'e karşı: havuzlu 0,81, havuzsuz 0,47.
  - Eski bir rate kaçana (s1) karşı: havuzlu 0,57, havuzsuz 0,16.
- **CTBR hız komutundan daha çevik:**
  - CTBR kovalayan, hız komutlu kaçanı %57 yakalıyor; tersi yalnızca %13,7.
  - Tepe hız: CTBR 12,9 m/s ve 12,85 rad/s, hız komutu 10,5 m/s ve 6,4 rad/s.
- **Eksikler:** taktik analizi, gerçek uçuş testi, sınırlar bölümü ve eğitim süresi yok. "Monoton iyileşme" iddiası tam tutmuyor (örn. Repel 0,89'dan 0,68'e düşüyor).

> **Doğrulama notu (15 Kasım 2026):** Repo (`github.com/KTH-RPL/AMSPB_PEG`) artık 404. Makale (v2) yeniden
> okundu ve bu bölümle çelişen noktalar çıktı: makalede **p\_old = 0,75 geçmiyor**, ablasyonda 0,5 kullanılıyor
> (Ek A: p\_old ≥ 0,5 yeterli). Yukarıdaki PPO hiperparametre tablosu (lr, clip, γ, λ, batch, entropi) makalede
> **bulunamadı**; makale sadece "çevrimiçi erişilebilir" diyor. Ödül katsayıları makalede farklı: yaklaşma
> κa = 0,5, adım κs = 0,007, yakalama/kaçış κc = κt = 10, gövde hızı 0,005, sınır 0,1 (Tablo II). Ağ 3 katman ×
> 256. Makaleye bu bölümden sayı aktarmadan önce makalenin kendisinden kontrol et.

**Bizim için çıkarımlar**

1. Kuralları neredeyse aynen alabiliriz: 0,5 m yakalama, süre dolunca kaçan kazanır. Bizdeki fark engeller ve 20 s'lik WCT turu.
2. Başlangıç tasarımı hazır: ödül katsayıları ve PPO hiperparametreleri. Kaçana mesafe değil, hayatta kalma ödülü veriliyor; bu reward hacking riskini azaltır.
3. Rakip havuzu basit ve işe yarıyor: tek parametre p\_old = 0,75. Bizim "havuz ve naive self-play" ablasyonumuzun dayanağı bu.
4. CTBR'ye geçmek değer: çeviklik farkı büyük. "Hız komutu ve CTBR" ikinci bir ablasyon olabilir.
5. Havuzu sezgisel politikalarla başlatmak, bizim scripted baseline'larımızla birebir örtüşüyor.
6. Isaac Sim ve 256 ortam kullandılar. Biz Colab'da Crazyflow ile binlerce ortam çalıştıracağız; bunu hız avantajı olarak yazabiliriz.

## 3. Tencent: özet ve çıkarımlar

World Chase Tag'dan esinlenen tek robot çalışması bu. Ama robotlar dört ayaklı, arena 4,5 m ve engeller küçük. [arXiv 2308.15143](https://arxiv.org/html/2308.15143v2) · Han ve ark. (Tencent Robotics X), Nature Machine Intelligence 2024.

**Ne yapmışlar**

- **Üç katmanlı hiyerarşi:**
  - Alt katman köpek hareket verisini taklit ediyor (Labrador, mocap).
  - Orta katman engel aşmayı öğreniyor: çit, basamak, merdiven.
  - Üst katman strateji: orta katmana yön ve hız komutu veriyor. Alttaki iki katman donduruluyor.
- **Oyun:** 4,5×4,5 m arena, çitler ve bir sırık. Daha büyük engeller arenaya sığmamış.
  - Kovalayan aradaki mesafe 0,6 m'nin altına inerse kazanıyor.
  - Kaçan bayrağa 0,3 m yaklaşırsa roller değişiyor. Bu WCT'de yok, onların eklediği bir kural.
  - 1000 adımda kimse kazanamazsa berabere; ikisine de 0 ödül.
- **Ödül:** seyrek ve sıfır toplamlı: kazanan +1, kaybeden -1.
- **Öğrenme:**
  - PPO ve PFSP (α = 2). Rakip, zorlandığı eski sürümlerden seçiliyor.
  - Politika LSTM, 32 birim.
  - Strateji katmanı 4 V100'de yaklaşık 40 saat eğitilmiş.
- **Değerlendirme:** saatte bir kayıt, toplam 40 model. Her çift 100 maç oynuyor (round-robin), sonuçlar Elo ile sıralanıyor. Sonraki modeller öncekileri tutarlı biçimde yeniyor.

**Sonuçlar**

- **Ortaya çıkan davranışlar:**
  - Kovalayan, kaçanı bayraktan önce yakalayamayacağını anlayınca vazgeçiyor.
  - Yakın mesafede sıçrayıp yakalıyor.
  - Kaçan da bayrağa sıçrıyor.
  - Hepsi yalnızca nitel anlatılmış.
- **Gerçek robot:** 14 kg MAX robotları, zero-shot transfer. Konum bilgisi mocap'ten geliyor. Güvenlik için hız 0,5 m/s'ye sabitlenmiş.
- **İnsana karşı:** joystick ile oynayan bir araştırmacıya karşı 2-0 kazanmış. Tek kişi, iki maç.

**Bizim için çıkarımlar**

1. WCT fikri robotlarda yapılmış, bunu kabul edip atıf yapıyoruz. Bizim farkımız drone, 3B hareket, yoğun engeller ve asıl WCT kuralları (süre dolunca kaçan kazanır, bayrak yok).
2. Hiyerarşi fikri bize de uyuyor. Crazyflow'un hazır hız/durum denetleyicisi "alt katman" olur, biz yalnızca strateji katmanını öğreniriz. Başta hız komutuyla başlama kararımızın gerekçesi bu.
3. Değerlendirme protokolünü aynen alabiliriz: belirli aralıklarla kayıt, çift başına 100 maç, Elo matrisi.
4. Ödül konusunda iki farklı yol var: seyrek ±1 (Tencent) ya da yoğun şekillendirilmiş ödül (KTH). Biz yoğunla başlayıp yakınlık terimini zamanla azaltabiliriz; bu bir ablasyon adayı.
5. LSTM bellek sağlıyor. Bizde rakip engel arkasında kayıp olduğunda belleğe ihtiyaç var; MLP ve GRU karşılaştırması maskeli gözlemle birlikte anlamlı.
6. Taktikler burada da ölçülmemiş, sadece anlatılmış. Bizim sayısal taktik metriklerimiz yine katkı.
7. Ekip 4 V100'de 40 saat harcamış. Biz alt katmanı öğrenmediğimiz ve basit dinamik kullandığımız için çok daha ucuz olmalı; yine de Colab bütçesini erken ölçelim.

## 4. OpenAI saklambaç: özet ve çıkarımlar

Basit bir ±1 ödül ve rekabet, altı aşamalı bir taktik yarışı doğurmuş. Bizim "engeller taktikleri nasıl değiştirir" sorumuzun ilham kaynağı bu. [arXiv 1909.07528](https://arxiv.org/pdf/1909.07528) · Baker ve ark. (OpenAI), ICLR 2020.

**Ne yapmışlar**

- **Ortam:** MuJoCo, 18 m kare alan. Rastgele odalar ve duvarlar var; 3–9 taşınabilir kutu, 2 rampa.
  - 1–3 saklanan ve 1–3 arayan.
  - Ajanlar nesneleri tutup kilitleyebiliyor.
- **Kurallar:** bölüm 240 adım. İlk %40 hazırlık evresi: arayanlar donuk, ödül yok.
  - Saklananların hepsi görünmezse saklananlar +1, biri bile görülürse -1. Arayanlar tersini alıyor.
  - Alandan çıkma cezası -10.
- **Gözlem:**
  - 30 lidar ışını.
  - Nesnelerin konum, hız ve boyutu.
  - Görüş hattı kesilen ya da 135° görüş konisinin dışında kalan nesneler maskeleniyor.
- **Mimari:**
  - Nesne tabanlı self-attention ve LSTM (256), yaklaşık 1,6M parametre.
  - Ajanlar tek politikayı paylaşıyor.
  - Critic her şeyi görüyor (merkezi); actor yalnızca maskeli gözlemi görüyor.
- **Öğrenme ve ölçek:**
  - PPO ile self-play.
  - 4. aşamaya 132 milyon bölümde, 34 saatte ulaşılmış. Büyük bir küme kullanılmış.
  - Küçük batch'ler (8k ve 16k) hiç yakınsamamış.

**Sonuçlar**

- **Altı taktik aşaması:**
  1. Koşma ve kovalama
  2. Kutularla sığınak kurma
  3. Arayanların rampa kullanması
  4. Saklananların rampayı kilitlemesi
  5. Kutu sörfü
  6. Saklananların tüm kutuları kilitlemesi
- **Ölçüm:** ortaya çıkan taktikler basit davranış istatistikleriyle izlenmiş: kutu ve rampa hareketi, kilitlenen nesne sayısı. Bunların eğitim boyunca çizilen eğrilerindeki sıçramalar aşamaları gösteriyor.
- **Rastgelelik önemli:**
  - Tam rastgele ortam: 6 aşama.
  - Duvarlar kaldırılınca: 4 aşama.
  - Sabit düzen ve duvarsız: 2 aşama.
- **Critic:** maskeli critic 3. aşamayı geçememiş; her şeyi gören critic 4. aşamaya ulaşmış.
- **İç motivasyonla kıyas:** sayım tabanlı keşif ve RND, self-play'e göre çok daha az anlamlı etkileşim üretmiş.
- **Sınırlar:**
  - Çok fazla deneyim gerekiyor.
  - Ajanlar simülatör hatalarını sömürmüş.
  - Transfer testleri karışık sonuç vermiş.

**Bizim için çıkarımlar**

1. Ana hipotezimize doğrudan dayanak: ortamdaki engeller ne kadar zenginse taktikler o kadar zengin (duvarlar kalkınca 6 aşama 4'e iniyor). Bizim engel yoğunluğu taramamız bu sorunun drone versiyonu.
2. Taktik ölçme yöntemini alıyoruz: görüş kırılması sayısı ve sıkışma noktasında geçen süre gibi metrikleri eğitim boyunca çizip aşama geçişlerini aramak.
3. Asimetrik critic kullanacağız: actor maskeli gözlem, critic tam durum görür. Maliyeti yok, farkı büyük. Ablasyon adayı.
4. Gözlem tasarımımız (ışın mesafeleri ve görüş hattı maskesi) bununla birebir uyumlu; atıfla savunulabilir.
5. Büyük batch şart. JAX ile binlerce paralel arena çalıştırmak bu yüzden kritik.
6. Beklentiyi ayarla: onların ölçeği milyonlarca bölüm ve büyük bir küme. Bizde 6 aşama çıkmaz. Hedef birkaç net taktik: engel arkasına saklanma, köşeye sıkıştırma, kesme.
7. Simülatör sömürüsüne dikkat: drone'un engel köşesinden geçmesi ya da sınır hilesi gibi. Videoları izleyerek kontrol et.

## 5. RA-L 2025 OPEN: özet ve çıkarımlar

Engelli ortamda yapılan en yakın drone kovalamacası bu. Ama kaçan öğrenmiyor: bir potansiyel alanını izleyen sabit bir script. Ayrıca 3'e karşı 1 oynanıyor ve self-play yok. [arXiv 2409.15866](https://arxiv.org/html/2409.15866v4) · Chen ve ark. (Tsinghua), IEEE RA-L 2025.

**Ne yapmışlar**

- **Kurulum:**
  - 3 kovalayan drone (hız 1,0) ve 1 kaçan (hız 1,3). Kaçan her kovalayandan hızlı.
  - Arena küçük: yarıçap 0,9, yükseklik 1,2.
  - Engel olarak 4–5 rastgele silindir var (yarıçap 0,1), yani seyrek.
- **Kurallar:** yakalama yarıçapı 0,3; bölüm en fazla 800 adım. Kaçan ancak görüş hattı açıksa görülüyor; gören drone bilgiyi takımına paylaşıyor.
- **Simülatör ve kontrol:** OmniDrones (Crazyflie, sistem tanımlamalı), CTBR + PID.
- **Gözlem:** kendi durumu, takım arkadaşları, en yakın 3 engel ve kaçanın konumu. Kaçan görünmezken konumu -5 ile maskeleniyor.
- **Yöntem:** MAPPO ve attention tabanlı kodlayıcı. Yanında iki ekleme var:
  - **Kaçan tahmin ağı (EPN):** LSTM ile kaçanın sonraki adımlarını tahmin ediyor.
  - **Uyarlamalı ortam üretici (AEG):** başarı oranı 0,5–0,9 arasında kalan görevleri saklıyor ve yakınında yenilerini üretiyor (%70). Kalan %30'da tüm görev uzayından rastgele seçiyor.
- **Ödül:** yakalama +6, mesafe cezası ×0,1, çarpışma -10.
  - İkinci aşamada yumuşaklık ödülü ekleniyor. Bu olmadan gerçek drone'lar çakılıyor.

**Sonuçlar (300 bölüm)**

- **Dört test senaryosu:** Wall, Narrow Gap (eğitim dağılımı içinde), Random ve Passage (dağılım dışı).
- **Yakalama oranı:**
  - OPEN tüm senaryolarda 0,99–1,00.
  - Düz MAPPO 0,62–0,92.
  - Sezgisel kovalayanlar (APF vb.) Wall'da yaklaşık 0,01, diğerlerinde çoğunlukla 0,31'in altında.
- **Ablasyon (dağılım dışı, Passage):**
  - MAPPO 0,62
  - MAPPO + AEG 0,73
  - MAPPO + EPN 0,89
  - İkisi birlikte 1,00
- **Maliyet:** düz MAPPO'nun %90'a ulaşması 2 milyar örnek sürmüş. OPEN bunu %50'den fazla kısaltıyor. Kullandıkları GPU ve eğitim süresi verilmemiş.
- **Gerçek dünya:** 3 Crazyflie 2.1, mocap, zero-shot transfer. Kaçan sanal.
- **Gelecek iş:** görüntü tabanlı yöntemler. Öğrenen bir kaçan ya da self-play hiç anılmamış.

**Bizim için çıkarımlar**

1. Boşluk teyit edildi. Engelli çalışmada kaçan sabit bir script; engelsiz çalışmalarda ise iki taraf da öğreniyor. İkisini birleştiren yok.
2. Hazır baseline: potansiyel alanlı kaçan (drone'lardan, engellerden ve sınırlardan mesafeyle ters orantılı itilme). Aynen kodlayıp "scripted evader" olarak kullanırız.
   - **Kod (15 Kasım 2026'da okundu):** github.com/thu-uav/Multi-UAV-pursuit-evasion (MIT), `omni_drones/envs/hide_and_seek/hideandseek.py`, `_get_dummy_policy_prey`. Formül: her kaynak (uzaklaşma yönü)/mesafe ile iter; kovalayan sadece görüş hattı açıksa sayılır; arena sınırı, zemin/tavan (1/z − 1/(H−z)) ve silindirler (yüzeye mesafe); katsayı yok; hız sabit v\_e, yön = toplam kuvvet. Kodda hız normalizasyonu eksen eksen yapılıyor (`norm(dim=1)`, boyutu 1 olan eksen) → çaprazda hız √3 katı; makale "sabit hız" diyor. Bizim uyarlamamız `src/chase_tag/scripted.py`'de (en yakın noktadan mesafe, doğru normalizasyon, fren filtresi).
3. Sezgisel kovalayanlar engelde ne kadar kötü olduğunu gösteriyor (Wall'da yaklaşık 0,01). Bu, RL kullanmanın doğrudan gerekçesi.
4. AEG fikri bizim yoğunluk müfredatımıza uyar: başarı oranı 0,5–0,9 arasındaki arenaları tut. Self-play'de iki tarafı dengede tutmak için de işe yarayabilir.
5. Test seti tasarımını alıyoruz: adlandırılmış, dağılım dışı arena tipleri (duvar, dar geçit, koridor) ile metrikler (yakalama oranı, yakalama süresi, çarpışma oranı; 300 bölüm, ortalama ± std).
6. Kaçanı tahmin etmek, görüş kesildiğinde çok işe yarıyor (0,62'den 0,89'a). Bu, bellek (GRU) ablasyonumuzu güçlendiriyor.
7. Hesap uyarısı: 2 milyar örnek. Colab'da bu sayıya JAX ile ancak saatler mertebesinde ulaşılabilir; erken bir hız testi şart.
8. Arenaları küçük ve seyrek; bizimki WCT gibi kutu ve duvarlarla yoğun olacak. Farkı metinde net yaz.

## 6. Gavin ve Bronz (iki makale): özet ve çıkarımlar

Aynı ekipten (ENAC ve LAAS, Toulouse) iki makale. İkisinde de ağ taşıyan bir drone öğrenen bir hedefi yakalıyor; arenalar engelsiz, simülatör JAX. Bizim için en değerli kısım hesap verileri: JAX ile tek GPU'da milyarlarca adım saatler içinde koşturulabiliyor. [arXiv 2603.16279](https://arxiv.org/pdf/2603.16279) (1v1) · [arXiv 2607.05939](https://arxiv.org/pdf/2607.05939) (3v1), 2026.

**Ortak kurulum**

- **Kontrol:** CTBR, 100 Hz; alt döngü 1 kHz.
- **Simülatör:** kendi yazdıkları JAX simülatörü; hava sürüktünmesi ve gecikmeler dahil. 1024 paralel ortam, saniyede yaklaşık 3,5×10⁵ adım.
- **Donanım:** tek RTX 4090.
- **Kurallar:** ağın yakınına giren hedef yakalanıyor. Bölüm 10 s; süre dolarsa kaçan kazanıyor.
  - Bir ajan çarparsa rakibine ödül verilmiyor. Amaç, rakibi çarpmaya zorlamayı ödüllendirmemek.
- **Kaçan:** yalnızca duvarlara yaklaşınca cezalandırılıyor. Amaç onu ortadaki bölgede tutmak.
- **Mimari:**
  - 2×256 MLP; geçmiş gözlem yok.
  - Critic rakibin tam durumunu ve eylemini de görüyor (asimetrik critic).

**1v1 makalesi (2603.16279)**

- **Eğitim:** PPO, iki taraf birlikte eğitiliyor. Rakip havuzu yok, her zaman en son rakip kullanılıyor.
- **Süre:** yaklaşık 2×10⁹ adım, 1 saat 35 dakika. Tabloda 4×10⁹ yazıyor; metinle tablo tutarsız.
- **Arenalar:** küçük 8×8×5 m, büyük 40×40×14 m.
- **Yakalama oranı (öğrenen kovalayan, öğrenen kaçana karşı):** küçük arenada %78,8, büyük arenada %66,5.
- **Sezgisel kovalayan (pure pursuit), öğrenen kaçana karşı:** küçük arenada %58, büyük arenada %25. Ayrıca çok çarpıyor (%39–48).
- **Gözlenen taktikler:**
  - Kaçan: dikey kaçış, dalış, ani durma, çalma.
  - Kovalayan: ağı kaçana doğru çevirmeyi kendi keşfediyor.
- **Gerçek test:** kaçan simülasyonda, kovalayan gerçek drone. 28 s uçuşta 7 yakalama.
- **Kendi yazdıkları sınırlar:**
  - Rakip havuzu olmadığı için aşırı uyum (overfitting) var. Çözüm olarak self-play ya da popülasyon öneriyorlar.
  - Engelli ortam gelecek iş.

**3v1 makalesi (2607.05939)**

- **Eğitim:** MAPPO. Taraflar sırayla eğitiliyor; eşik geçilince sıra diğer tarafa geçiyor.
  - Rakip havuzu var.
  - Bölümlerin %50'si en son rakiple, %50'si PFSP ile oynanıyor. PFSP ağırlığı f(x) = (1 − x)².
- **Süre:** taraf başına 5×10⁹ adım, toplam yaklaşık 5,5 saat.
- **Ablasyonlar:**
  - Self-play olmadan eğitilen politika (noSP) kendi rakibine aşırı uyuyor, çevik kaçanlara karşı bozuluyor.
  - Hız komutlu kovalayan, CTBR'li kaçanı yalnızca %17,7 yakalıyor.
  - Sezgisel FRPN kovalayan, öğrenen kaçana karşı %2,9.
- **Gözlenen taktikler (nitel):** kovalayanlar ayrılıp kaçanı duvarlara doğru itiyor; biri atak yaparken diğerleri bekliyor. Kuşatma nadir görülüyor.
- **Gerçek test:** yok. Arena engelsiz.
- Metin ile tablo burada da tutarsız. Atıf yaparken tablo değerlerini kullan.

**Bizim için çıkarımlar**

1. Boşluk yine teyit edildi: iki makale de engelsiz, ilki engelli ortamı açıkça gelecek iş olarak yazıyor.
2. Yapılabilirliğin en güçlü kanıtı bu: JAX ile tek GPU'da 2×10⁹ adım yaklaşık 1,5 saat. Colab A100 benzer sınıfta. Proposal'da hesap bütçesine bu dayanak yazılabilir.
3. Self-play tarifi hazır: sırayla eğitim ve eşikle geçiş, rakip havuzu, %50 en son rakip ve %50 PFSP \[(1 − x)²\].
4. Kritik tasarım kararı: çarpan ajanın rakibine ödül verilmeli mi?
   - Onlar vermiyor, çünkü rakibi çarpmaya zorlamayı ödüllendirmek istemiyorlar.
   - Bizde engel var. Kovalayanı duvara çarptırmak WCT'de meşru bir taktik olabilir.
   - Karar verilecek; iki seçenek ablasyon olarak denenebilir.
5. Arena boyutu sonucu çok değiştiriyor (pure pursuit %58'den %25'e). Arena boyutunu sabit tut ve gerekçelendir; yoğunluk taramasını o boyutta yap.
6. Değerlendirme protokolü: eşleşme başına 10.000 bölüm. Sansürlü yakalama süresi kullanılıyor: çarpışma ya da süre dolması 10 s sayılıyor. Çarpma türleri ayrı ayrı raporlanıyor.
7. CTBR'nin üstünlüğü üçüncü kez görüldü. Ama başta hız komutuyla başlayıp sonra CTBR'ye geçme planımız mantıklı kalıyor.
8. "Kaçanı duvara itmek", WCT'deki herding'in karşılığı. Onlar bunu yalnızca anlatmış; biz ölçeceğiz.

## 7. UPenn drone yarışı: özet ve çıkarımlar

Bu çalışma, engelli parkurda "yarışı kazan" gibi seyrek bir ödülün self-play ile birleşince yoğun (ilerleme) ödülünden çok daha iyi çalıştığını gösteriyor. Yoğun ödül engelli parkurda %0 başarıya düşüyor. [arXiv 2512.11781](https://arxiv.org/html/2512.11781v1) · Pasumarti, Bianchi, Loquercio (UPenn, Tor Vergata), 2025.

**Ne yapmışlar**

- **Kurulum:** iki drone 3 tur yarışıyor. İki parkur var:
  - CT: 8×7 m, 6 kapı, 4 engel.
  - LT: 5×5 m, 5 kapı, 2 engel.
- **Drone ve kontrol:** Crazyflie 2.1 Brushless, CTBR. Simülatör Isaac Lab; domain randomization var.
- **Ödül:**
  - Öndeyken kapı geçmek +10, gerideyken +5.
  - Öndeyken tur bitirmek +50.
  - Komut ve çarpışma cezaları.
  - Karşılaştırılan baseline ise ilerlemeyi ödüllendiren yoğun ödül.
- **Öğrenme:** IPPO; iki ajanın kendi politikası ve critic'i var, birlikte self-play ile eğitiliyorlar. Rakip havuzu yok.

**Sonuçlar**

- **Tek ajan, engelli parkur:**
  - Yoğun ödül iki parkurda da %0 başarı.
  - Seyrek ödül LT'de %98.
  - CT'de ikisi de başarısız.
- **Kafa kafaya (simülasyon):** seyrek ödüllü çok ajanlı yöntem ortalama %91 kazanıyor.
- **Sim2real:** hız farkı 0,43 m/s, baseline'da 0,76 m/s; yaklaşık %45 daha az. Engelli LT'de gerçekte parkuru yalnızca bu yöntem tamamlıyor.
- **Gözlenen davranışlar:**
  - Güçlü rakip varken daha hızlı uçuyor (tepe 9,9 m/s).
  - Rakip düşünce temkinli uçmaya geçiyor.
  - Gerçek dünyada bir "bloklama" mançası gözlenmiş: öndeki drone geniş bir yay çizerek arkadakini kapı çerçevesine çarptırıyor.
- **Sınırlar:**
  - Gerçek yarışlar çok az: eşleşme başına 3.
  - Çok ajanlı eğitim eğrileri dalgalı; iki ajan sırayla öne geçiyor.
  - Eğitim süresi verilmemiş.

**Bizim için çıkarımlar**

1. Ödül tasarımımızı doğrudan destekliyor: seyrek WCT sonucu artı küçük bir şekillendirme. Engelli ortamda yoğun ödül ajanı engele kilitleyebiliyor. "Seyrek ve yoğun ödül" güçlü bir ablasyon adayı.
2. Engeli taktik olarak kullanmak (rakibi çerçeveye çarptırmak) yarışta bile kendiliğinden çıktı. Bizim "çarptırmak ödüllendirilmeli mi" sorumuzun yanıtını doğrudan etkiliyor.
3. Havuzsuz self-play dalgalı. Rakip havuzu kararımızı bir kez daha destekliyor.
4. Engel var ama oyun yarış, kovalamaca değil; ölçülen tek şey tur zamanı ve kazanma oranı. Bizim farkımız korunuyor.

## 8. Sandia / Georgia Tech: özet ve çıkarımlar

Basında "drone'lara ebelemece öğretmek" diye geçen çalışma bu. Ama yalnızca kaçanlar öğreniyor; kovalayanlar sabit algoritma (proportional navigation) ve ortam engelsiz. [Sandia haberi](https://www.sandia.gov/labnews/2026/08/13/teaching-drones-to-play-tag/) · [makale](https://coogan.ece.gatech.edu/papers/pdf/llanes2026marl.pdf): Llanes, Williams, Jensen, Coogan, "Learning Cooperative Strategies for Drone Swarms using MARL", ICRA 2026.

**Ne yapmışlar**

- **Oyun:**
  - Senaryolar 2v2 ve 4v4. Kaçanlar bir hedefe ulaşmaya, kovalayanlar ise hedefi korumaya çalışıyor.
  - Arena 15 m yarıçaplı bir küre ve engel yok.
  - Yakalama mesafesi 0,2 m.
- **Kovalayanlar:** pure pursuit ve geliştirilmiş proportional navigation (APN); öğrenmiyorlar.
- **Kaçanlar:** MAPPO, merkezi critic, 128 paralel ortam. Kendi yazdıkları basit 6-DoF model kullanılıyor.
- **Ödül:**
  - Hedefe ulaşmak +30.
  - İki kovalayanın birbirine çarpması +10 (2v2) ya da +20 (4v4).
  - Sınırı geçmek -5.

**Sonuçlar**

- **Başarı oranı (kaçanlar):**
  - 2v2: %96,5 (pure pursuit karşısında) ve %94 (APN karşısında).
  - 4v4: %86 ve %54.
  - Kovalayansız eğitilen baseline yalnızca %6–14.
- **Gözlenen taktikler:**
  - Kaçanlar kovalayanları birbirine çarptırıyor.
  - Dikey ivmeyle APN'nin kör noktasını sömürüyorlar.
- **Gerçek test:** 4 Crazyflie 2.1. Yalnızca nitel sonuç; başarı oranı verilmemiş.
- **Sınırlar:** model basit, iletim gecikmesi var, mocap'e bağımlı. Self-play ya da öğrenen kovalayan konuşulmamış.

**Bizim için çıkarımlar**

1. "Drone ebelemecesi" ilgi çeken, haber olan bir konu. Bu, özgeçmiş ve sunum açısından bir artı.
2. Yine tek taraf öğreniyor ve engel yok; boşluk korunuyor.
3. Sabit bir rakibe karşı eğitilen ajan, o rakibin kör noktasını sömürüyor (APN'ye karşı dikey kaçış). Bu, scripted rakiplerin neden yetmediğinin ve self-play'in gerekçesi.
4. "Rakibi çarptırmak" burada açıkça ödüllendirilmiş ve ana taktik olmuş. Gavin ve Bronz ise bunu bilerek ödüllendirmiyor. İki uç örnek var; biz arada bir karar verip gerekçelendireceğiz.

## 9. Crazyflow (simülatörümüz): özet ve çıkarımlar

Engelleri dolduracağımız arena, ışın mesafeleri ve çarpışma kontrolü için gereken her şey hazır: MuJoCo MJX üzerinden geliyor. Fizikte temas çözücü yok; yakalama ve çarpmayı mesafe kontrolüyle kendimiz yazacağız. [arXiv 2606.01478](https://arxiv.org/pdf/2606.01478) · Schuck, Rath, Hua, Goudar, Zhou, Schoellig (TU Münih, Toronto, SFU), Haziran 2026 · `pip install crazyflow`.

**Ne sunuyor**

- **İki dinamik ailesi:**
  - **First-principles:** 6-DoF rijit gövde. Motor dinamiği, itki ve tork eğrileri, jiroskopik etki ve doğrusal sürüktünme var.
  - **Soyutlanmış:** nokta kütle. Tutum 2. derece, itki 1. derece gecikmeli; daha hızlı.
- **Sistem tanımlama:** itki ve tork eğrileri yük hücresiyle ölçülmüş. Soyutlanmış model 4 dakikadan az uçuş verisiyle uydurulmuş. Crazyflie 2.1, 2.1+, Brushless ve kendi yaptıkları 660 g'lık bir drone için model var.
- **Kontrol arayüzleri:**
  - Motor hızı (en az 500 Hz).
  - İtki + gövde torku (en az 250 Hz).
  - İtki + tutum (50 Hz öneriliyor).
  - Konum hedefi: Crazyflie'ın kendi Mellinger kontrolcüsü taklit ediliyor.
- **Engeller:** MuJoCo XML sahnesinden yükleniyor. Ray casting, mesafe ve çarpışma kontrolü MJX ile yapılıyor. Derinlik kamerası: 1024 ortamda 64×64 çözünürlükle saniyede 350 bin kare.
- **Hız:** tüketici GPU'da 1 milyon paralel dünyada saniyede yaklaşık 700 milyon adım. Özet kısmı 900 milyon diyor; metin ile tutarsız. Aerial Gym, DiffAero ve gym-pybullet-drones'tan en az 10 kat hızlı.
- **Doğruluk (gerçek Crazyflie 2.1, RMS konum hatası):**
  - Daire, 9 s: Crazyflow 19 mm; gym-pybullet-drones 49 mm; CrazySim 102 mm.
  - Toplamda gym-pybullet-drones'a göre hata %61'e kadar daha az.
- **RL örnekleri:**
  - PPO ile tutum kontrolünde iz takibi 1,6–17,6 saniyede eğitiliyor.
  - Motor seviyesinde RL 12 s'de eğitilip gerçek drone'da uçmuş.
- **Sınırlar:** drone'lar arası aerodinamik etkileşim ya da temas modellenmiyor. PX4 ve Betaflight kontrolcüsü yok. Foto-gerçekçi görüntü yok, sadece derinlik var.

**Bizim için çıkarımlar**

1. Arenayı MuJoCo XML olarak üreteceğiz: kutular ve silindirler. Işın mesafeleri ve görüş hattı kontrolü aynı MJX sorgularıyla yapılır; ekstra kod az.
2. Yakalama (0,3 m), drone–drone çarpışması ve engel çarpması mesafe kontrolüyle bizim ortam kodumuzda olacak. Fizik temas çözmediği için çarpma ani bitiş sayılmalı.
3. Başta soyutlanmış (nokta kütle) model ve konum/hız hedefi; sonra tutum ya da gövde hızı arayüzü. Bu, "önce hız komutu, sonra CTBR" planımızla birebir uyuyor.
4. "Sistem tanımlamalı dinamik" iddiamızın sayısal dayanağı bu: 19 ve 49 mm.
5. Darboğaz simülatör değil, politika ağı ve PPO güncellemesi olacak. Hesap bütçesi buna göre planlanmalı.
6. Sürümü sabitle. Bizim test ettiğimiz 0.4.0'da API adları belgelerden farklıydı (`Physics` yerine `Dynamics`, 16 boyutlu durum komutu).

## 10. World Chase Tag kuralları: özet ve çıkarımlar

Resmi kurallar kısa ve RL'e kolay aktarılıyor. Asıl kazanç sporun kendi taktik sözlüğü: çoğu terim doğrudan ölçülebilir bir metriğe dönüşüyor. [WCT kuralları](https://worldchasetag.com/rules) · Quad boyutu için [Wikiwand](https://www.wikiwand.com/en/articles/World_Chase_Tag).

**Kurallar**

- **Maç formatı:**
  - İki takım, en fazla 6'şar kişi. Maç 16 kovalamacadan oluşuyor.
  - Her kovalamacada 1 kovalayan ve 1 kaçan var; süre 20 s. Aralarda 25 s dinlenme.
- **Alan:** "Quad" 12×12 m (resmi sayfada boyut yok, Wikiwand'den). İçi parkur engelleriyle dolu; oyuncular rakip köşelerden başlıyor.
- **Dokunma ve sınır:**
  - Dokunma el ile olmalı, ayakla sayılmaz.
  - Quad'dan çıkan kovalamacayı kaybeder.
  - Dokunma ya da çıkış kovalamacayı anında bitirir.
- **Puan ve roller:**
  - Sadece kaçış puan getirir: 20 s yakalanmazsan +1. Yakalamak puan getirmez.
  - Kazanan bir sonraki kovalamacada kaçan olur.
- **Beraberlik:** iki kovalamacalık chase-off oynanır, iki takım birer kez kaçar. Daha uzun kaçan kazanır; ikisi de 20 s'yi tamamlarsa tekrarlanır.

**Taktik sözlüğü ve bizim metriğimiz**

| WCT terimi | Anlamı | Bizde nasıl ölçülür |
| --- | --- | --- |
| Evasion ratio | Kaçışla biten kovalamaca oranı | Ana metrik: kaçış oranı |
| Spike | 5 s'den kısa süren kovalamaca | 5 s'den kısa yakalama oranı |
| EQ (Evasion Quality) | Bir engelin kaçana sağladığı koruma | Engel başına kaçış oranı; arena ısı haritası |
| Herding | Kovalayanın kaçanı kolay yakalanacak bölgeye sürmesi | Kaçanın sıkışma noktalarında geçirdiği süre |
| Reverse herding | Kaçanın kovalayanı zor bölgeye çekmesi | Kovalayanın yüksek EQ bölgelerinde geçirdiği süre |
| Pinch point | Kaçış yolu az olan bölge | Serbest ışın sayısı düşük hücreler |
| Idling | Kaçanın yüksek EQ'lu engel arkasında zaman öldürmesi | Görüş hattı kapalıyken kaçanın düşük hızda geçirdiği süre |
| Benny Hill | Kovalayanın aynı engelin etrafında dönmesi (zayıf) | Bir engel etrafında toplam dönüş açısı |
| Tailing | Kovalayanın kaçanın izini birebir takip etmesi (zayıf) | Kovalayan yolunun kaçanın geçmiş yoluna uzaklığı |
| Centre control | Kaçırdıktan sonra kovalayanın merkeze dönmesi | Iska sonrası kovalayanın merkeze uzaklığı |
| Threading | Dar boşluktan sızıp geçmek | Dar geçitlerden geçiş sayısı |

**Bizim için çıkarımlar**

1. Kurallar aynen aktarılıyor: 20 s, sınır dışı kaybeder, sadece kaçış puan, kazanan kaçan olur. El ile dokunma bizde 0,3 m yakalama yarıçapı oluyor.
2. Değerlendirmede gerçek maç formatı kullanılabilir: 16 kovalamacalık maçlar, kazanan kaçan olarak kalır, berabere biterse kaçış süresi belirler. Elo bu maçlardan hesaplanır. Sunumda çok çarpıcı durur.
3. Taktik sözlüğü makalenin taktik analizi bölümünün iskeleti olur. "WCT sporcularının terimleri drone'larda kendiliğinden ortaya çıktı mı?" sorusu net ve ölçülebilir.
4. Proposal'daki 12×12 m bilgisi resmi sayfada yok; kaynağı Wikipedia/Wikiwand. Gerekirse atıfa ekle.

## 11. DualCL: özet ve çıkarımlar

OPEN'ın öncülü; aynı Tsinghua ekibinden. Burada da kaçan potansiyel alanlı bir script, yani boşluk etkilenmiyor. [arXiv 2312.12255](https://arxiv.org/html/2312.12255v2) · Chen, Li, Yu, Yang, Xu, Yang, Wang.

- **Kurulum:**
  - N drone (hız 1,0), kendilerinden çok daha hızlı bir kaçanı (2,4) kovalıyor.
  - Arena yarıçapı 0,9; 0–3 silindir engel var.
  - Yakalama yarıçapı 0,12; bölüm 800 adım.
  - OmniDrones, CTBR, MAPPO.
- **İkili müfredat:**
  - **İç parametre müfredatı:** yakalama yarıçapı arena boyutundan, kaçan hızı 0'dan başlıyor. 10 adımda hedef değerlere iniyor; yakalama oranı 0,98'i geçince bir adım ilerliyor.
  - **Dış ortam üretici:** çözülemeyen düzenleri DFS ile eliyor (TaskFilter). Çözülemeyen senaryoları bir arşivde tutup %70 olasılıkla oradan örnekliyor.
- **Sonuçlar:**
  - Yakalama oranı 0,89–0,96 (dağılım dışı senaryolar dahil). En iyi baseline 0,51–0,84.
  - Düz MAPPO bu zorlukta yaklaşık 0 alıyor.
- **Gerçek test:** Crazyflie 2.1, sanal kaçan. Sonuç yalnızca nitel.

**Bizim için çıkarımlar**

1. Boşluk yine sağlam: kaçan öğrenmiyor, engeller seyrek (0–3 silindir).
2. Çok kullanışlı bir hile: kovalayan başta hiç yakalayamıyorsa, yakalama yarıçapını büyük başlatıp adım adım 0,3 m'ye indir. Self-play'in ilk aşamasındaki kilitlenmeyi çözer.
3. Rastgele arena üretirken her düzende bağlantılılığı kontrol et (BFS/DFS). Kapalı cep oluşursa kaçan oraya saklanıp bedavaya kazanır.

## 12. CRUISE: özet ve çıkarımlar

İstanbul'dan bir çalışma (Türk-Alman Üniversitesi). Çok drone'lu yarışta müfredat ile tekrarlı self-play birleştiriliyor; kovalamaca ya da engel yok. [arXiv 2510.22570](https://arxiv.org/abs/2510.22570v1) · Onur Akgün, 2025, hakem sürecinde.

- **Kurulum:** 2–4 drone, iki parkur (halka 58,9 m, sekiz 56,8 m). Kendi Gymnasium ortamları. Politika ivme komutu veriyor.
- **Müfredat:** 5 aşama, toplam 4×10⁷ adım. Her aşamada referans hız 1'den 10 m/s'ye, çeviklik katsayısı ve çarpışma cezası artıyor.
- **Self-play:**
  - Tek bir aktif politika, n−1 dondurulmuş rakibe karşı eğitiliyor.
  - Kazanma oranı eşiği geçince rakipler aktif politikanın kopyasıyla güncelleniyor.
  - Havuz yok.
- **Sonuçlar:** halka parkurda oyun kuramsal planlayıcının (SE-IBR) yaklaşık 1,9 katı hız (4,4 ve 2,3 m/s). Müfredatsız versiyon 3 ve daha fazla drone'da çoğunlukla başarısız.
- **Zayıf yönler:** gerçek test yok. Kazanma oranları raporlanmamış. Metin ile tablo tutarsız ("yakınsamadı" denen baseline tabloda %30–43 başarı gösteriyor).

**Bizim için çıkarımlar**

1. Özgünlüğümüzü etkilemiyor: yarış, kovalamaca değil, engel yok.
2. Yerel bir referans. Hocanın ilgisini çekebilir; atıf vermek ya da yazarla iletişim kurmak düşünülebilir.
3. "Eşik geçince rakibi güncelle" fikri Gavin ve Bronz'unkiyle aynı. Bizim tarifimiz (havuz + PFSP) bunun bir adım ötesi.

## 13. SJTU "Back to Newton's Laws": özet ve çıkarımlar

Basit bir nokta kütle modeliyle eğitilen politika gerçek dünyada ormanda 20 m/s ile uçmuş. Kovalamaca yok, ama basitleştirilmiş dinamik seçimimizi savunmak için en güçlü kanıt bu. [arXiv 2407.10648](https://arxiv.org/pdf/2407.10648) · Zhang, Hu, Song, Zou, Lin (Shanghai Jiao Tong); önceki taramada Nature Machine Intelligence 2025 olarak geçti.

- **Model:** nokta kütle, sabit gecikme (1/15 s) ve üstel yumuşatma. Hava sürüktünmesi var.
- **Eğitim:** RL değil, türevlenebilir fizikle doğrudan gradyan.
  - 64 paralel ortam.
  - Zamanla sönen gradyan (α = 0,92) olmadan patlıyor.
  - PPO'nun yaklaşık %10'u kadar örnekle yakınsıyor.
- **Politika:** 16×12 derinlik görüntüsü giriyor, CNN ve GRU'dan geçiyor, itki ivmesi vektörü çıkıyor. 21 dolarlık bir kart üzerinde çalışıyor.
- **Gerçek dünya:**
  - Ormanda 20 m/s; önceki en iyi yöntem 10 m/s.
  - Hareketli engellerden kaçıyor.
  - 6 drone'luk sürü dar bir kapıdan iletişimsiz geçiyor.

**Bizim için çıkarımlar**

1. "Nokta kütle ve hız/ivme komutu gerçekçi değil" itirazına cevap: doğru gecikme ve sınırlarla bu model gerçek dünyaya taşınıyor. Crazyflow'un soyutlanmış modeli aynı fikir.
2. GRU bellek burada da kritik. Bizim MLP/GRU ablasyonu için bir dayanak daha.
3. Türevlenebilir fizik (BPTT) Crazyflow'da da var. Gelecek iş olarak yazılabilir; bu projede kapsam dışı.

## Genel sonuç ve tasarım kararları

Okunan 12 çalışmanın hiçbiri üç şeyi birlikte yapmıyor: 1v1 drone, iki tarafın da öğrenmesi ve yoğun engel. Taktikleri sayısal olarak ölçen de yok. Aşağıdaki tarif bu okumalardan derlendi.

**Tasarım tarifi (kimden aldık)**

- **Kurallar:**
  - Yakalama yarıçapı yaklaşık 0,3 m; Crazyflie için OPEN'deki değer.
  - 20 s'lik WCT turu; süre dolarsa kaçan kazanır (KTH, Gavin).
  - Arena dışına çıkmak ya da çarpmak kaybettirir.
- **Kontrol:** önce hız komutu kullanılır, Crazyflow'un kontrolcüsü alt katman olur (Tencent'in hiyerarşisi). Sonra CTBR'ye geçilir, çünkü daha çevik (KTH, Gavin).
- **Gözlem:**
  - Kendi durumu.
  - Rakibin son birkaç adımdaki göreli konumu (KTH).
  - Işın mesafeleri ve görüş hattı maskesi (OpenAI, AgilePE).
  - Kalan süre.
- **Critic:** asimetrik. Actor maskeli gözlemi, critic tam durumu görür (OpenAI, Gavin).
- **Ödül:**
  - Seyrek WCT sonucu ve küçük bir yaklaşma terimi (UPenn, KTH).
  - Kaçana her adım için hayatta kalma ödülü.
  - Başlangıç katsayıları KTH'den.
- **Self-play:**
  - Taraflar sırayla eğitilir, eşik geçilince sıra değişir.
  - Rakip havuzu kullanılır: %50 en son rakip, %50 PFSP \[(1 − x)²\] (Gavin ve Bronz, KTH).
- **Müfredat:** başarı oranı 0,5–0,9 aralığında olan arenalar tutulur ve yoğunluk adım adım artırılır (OPEN).
- **Baseline'lar:** pure pursuit kovalayan, potansiyel alanlı kaçan (OPEN) ve engelsiz eğitilmiş politika.
- **Değerlendirme:**
  - Kaydedilen politikalar arasında round-robin maçlar ve Elo (Tencent).
  - Kaçış oranı, sansürlü yakalama süresi, çarpışma türleri (Gavin).
  - Adlandırılmış dağılım dışı arenalar: duvar, dar geçit, koridor (OPEN).
  - Taktik metrikleri eğitim boyunca çizilir (OpenAI).

**Ablasyon adayları (öncelik sırasıyla)**

1. Rakip havuzu ve naive self-play (KTH, Gavin)
2. Maskeli ve tam gözlem
3. Engelli ve engelsiz eğitim (OpenAI)
4. Seyrek ve yoğun ödül (UPenn)
5. MLP ve GRU bellek (Tencent, OPEN)
6. Hız komutu ve CTBR (KTH)

**Açık karar**

- [ ] Rakibi engele çarptırmak ödüllendirilsin mi? Gavin ödüllendirmiyor, Sandia ödüllendiriyor; WCT ruhuna göre meşru sayılabilir.

**Hesap**

Benzer bir JAX simülatörü tek RTX 4090'da 2×10⁹ adımı yaklaşık 1,5 saatte koşturuyor (Gavin). İlk hafta Colab A100'de gerçek hız test edilecek.

**Araç ve kural okumasından eklenen kararlar**

- **Arena:** MuJoCo XML olarak üretilir. Işın mesafeleri ve görüş hattı MJX sorgularıyla hesaplanır (Crazyflow).
- **Bağlantılılık:** her rastgele düzende kontrol edilir; kapalı cep oluşmamalı (DualCL).
- **Yakalama ve çarpma:** mesafe kontrolüyle yazılır. Fizikte temas yok, bu yüzden çarpma bölümü anında bitirir (Crazyflow).
- **Başlangıç kilitlenmesine karşı:** yakalama yarıçapı büyük başlar, adım adım 0,3 m'ye iner (DualCL).
- **Dinamik:** önce soyutlanmış (nokta kütle) model, sonra first-principles ve CTBR. Dayanak: SJTU ve Crazyflow.
- **Değerlendirme formatı:** 16 kovalamacalık WCT maçları oynanır, Elo bunlardan hesaplanır. Taktik analizi WCT sözlüğüne göre yapılır (bölüm 10'daki tablo).

## Oyun kuralları (v1)

Kurallar WCT'den uyarlandı. Her maç tek bir kovalamaca: 20 s, 1'e 1. Puanı yalnızca kaçış getirir.

| Kural | Değer | Dayanak |
| --- | --- | --- |
| Arena | 6×6×2 m; zemin, tavan ve duvarlar sınır | WCT Quad, Crazyflie ölçeğine küçültülmüş |
| Süre | 20 s; kontrol 50 Hz, yani 1000 adım | WCT; Crazyflow 50 Hz öneriyor |
| Ebeleme | Merkezler arası mesafe 0,3 m'nin altında | WCT'de el ile dokunma; OPEN de 0,3 m kullanıyor |
| Kazanma | Kaçan 20 s dayanırsa kazanır | WCT |
| Sınır ve çarpma | Sınıra ya da bir engele değen kaybeder, rakibi kazanır | WCT'de Quad'dan çıkan kaybeder. Engele çarpan drone düşer; bu, sahadan çıkmakla aynı sayılır |
| Hız | İki drone'un azami hızı eşit | WCT'de iki insan; sonucu taktik belirlesin |
| Görüş | 360°; aradaki doğruda engel yoksa rakip görülür | Saklanma taktiği buradan doğar |
| Başlangıç | Karşı köşelerde, aralarında en az 3 m, havada asılı | WCT'de oyuncular karşı taraflardan başlar |
| Roller | Kovalayan ve kaçan için iki ayrı ağ | Her rolün kendi stratejisi var |
| Değerlendirme maçı | 16 kovalamaca; kazanan bir sonraki turda kaçan olur | WCT maç formatı; yalnızca test için |

**Sonuçlar ve ödüller (karar: WCT yolu)**

| Ne oldu? | Kazanan | Kovalayanın ödülü | Kaçanın ödülü |
| --- | --- | --- | --- |
| Kovalayan 0,3 m'ye girdi (ebeledi) | Kovalayan | +1 | −1 |
| 20 s doldu | Kaçan | −1 | +1 |
| Kaçan engele ya da sınıra çarptı | Kovalayan | +1 | −1 ve ek çarpma cezası |
| Kovalayan engele ya da sınıra çarptı | Kaçan | −1 ve ek çarpma cezası | +1 |
| İkisi aynı anda çarptı (nadir) | Kimse | −1 ve ek çarpma cezası | −1 ve ek çarpma cezası |

- **Ek çarpma cezası:** çarpmayı normal kaybetmekten daha kötü yapar; drone'lar güvenli uçmayı öğrenir (KTH, Gavin). Değeri eğitimde ayarlanacak.
- **Oyun sırasındaki küçük ipuçları:** kovalayan yaklaştıkça, kaçan her saniye dayandıkça az puan alır. Yalnızca başlangıçta yol gösterir; asıl puan tablodan gelir.
- **Ablasyon (Gavin yolu):** biri çarptığında rakip +1 yerine 0 alır. Soru: rakibi çarptırmak ödüllendirilince taktikler nasıl değişiyor?

**Engel aşamaları**

1. **Aşama 1, tam boy:** yerden tavana sütunlar ve duvarlar. Oyun neredeyse 2B. Öğrenmenin çalıştığı burada doğrulanır.
2. **Aşama 2, üstünden geçilebilen:** alçak kutular ve duvarlar (yaklaşık 0,8 m). WCT'deki atlamaya karşılık gelir.
3. **Aşama 3, altından geçilebilen:** havada duran platformlar ve çubuklar (altında yaklaşık 0,8 m boşluk). WCT'deki "threading"e, yani dar boşluktan sızıp geçmeye karşılık gelir.

Nihai arena üç türü karışık içerir. Her aşamaya bir öncekinin politikasıyla devam edilir.

**Ablasyon olarak denenecek alternatifler:** 360° yerine kamera konisi (AgilePE).

## Kalan okumalar

Literatür taraması 9 Kasım'da tamamlandı. Kalan iki kaynak kod aracı; kodlama sırasında bakılacak.

| Kaynak | Durum | Ne zaman |
| --- | --- | --- |
| Crazyflow makalesi | Okundu (bölüm 9) | — |
| WCT kuralları | Okundu (bölüm 10) | — |
| DualCL, CRUISE, SJTU | Okundu (bölüm 11–13); boşluğu kapatan yok | — |
| PureJaxRL | PPO kodunun temeli | Kodlarken |
| gym-pybullet-drones | Yedek simülatör | Gerekirse |
