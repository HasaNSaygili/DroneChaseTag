# Drone Chase Tag: projeye baştan bakış

Bu not, projeyi hiç bilmeyen birinin baştan sona okuyup genel fikri edinmesi için yazıldı (20 Kasım 2026).
İlk kısım projeyi sırayla anlatıyor; ikinci kısım sık sorulan soruları ayrıntılı cevaplıyor.

**İçindekiler**

- Kısım A: Proje baştan sona
  1. Fikir
  2. Pekiştirmeli öğrenme (RL) en baştan
  3. Simülasyon
  4. Oyunun kuralları ve ödüller
  5. Arenalar ve ajanın "gözü"
  6. Öğrenme algoritması (PPO)
  7. Elle yazılmış rakipler
  8. Şimdiye kadarki deneyler
  9. Yol boyunca çıkan dersler
  10. Şu an neredeyiz?
- Kısım B: Sorular ve cevaplar
  1. Boş arenada kovalayanın hep kazanması literatürde var mı?
  2. Drone'un boyutu
  3. Critic neye göre tahmin ediyor?
  4. Ajanın gördükleri literatürde nasıl?
  5. γ ve std neden önemli?
  6. Elle yazılmış rakipler: talimatlar, sonuçlar, atıflar
  7. PFSP ve diğer rakip seçme kuralları
  8. Self-play'in kovalayan lehine çıkmasının iki nedeni
- Kaynaklar

---

# Kısım A: Proje baştan sona

## 1. Fikir

### World Chase Tag nedir?
World Chase Tag (WCT) profesyonel bir **ebelemece sporu**. 12×12 metrelik bir sahada kutular, duvarlar ve barlar var.
Her turda iki kişi oynuyor: biri **kovalayan**, biri **kaçan**.

- Kovalayan 20 saniye içinde kaçana elle dokunursa kazanır.
- Dokunamazsa kaçan kazanır ve takımına puan getirir.
- Sahadan çıkan kaybeder.

Bu sporu ilginç yapan, kazanmanın hızdan çok **taktiğe** bağlı olması. Kaçan bir engelin arkasına saklanır, engelin
etrafında döner. Kovalayan kaçanı köşeye sürer, önünü keser. Sporcuların bu taktikler için kendi terimleri bile var:
*herding* (rakibi sürmek), *Benny Hill* (engelin etrafında dönmek) gibi.

### Biz ne yapıyoruz?
Bu oyunu **iki drone'a** oynatıyoruz ve onlara oynamayı kimse öğretmiyor. Drone'lar oyunu **sıfırdan, deneme
yanılmayla** öğreniyor. Başta rastgele uçuyorlar; milyonlarca maç oynadıkça neyin işe yaradığını keşfediyorlar.

### Araştırma sorusu
> Engelli bir arenada, birbirine karşı öğrenen iki drone hangi taktikleri kendiliğinden keşfeder?

Örneğin kaçan, kimse söylemeden engel arkasına saklanmayı öğrenir mi? Kovalayan onu köşeye sürmeyi öğrenir mi?

### Neden "yeni"?
Ders, projenin daha önce tam bu şekilde yapılmamış bir şey olmasını istiyor. Literatürde 12 çalışma okuduk. Hepsinde
şu üç parçadan biri eksik:

1. **İki taraf da öğreniyor mu?** Bazı çalışmalarda sadece kovalayan öğreniyor; kaçan sabit bir program.
2. **Arena engelli mi?** İki tarafın da öğrendiği çalışmaların hepsi boş arenada.
3. **Drone mu?** WCT'yi deneyen tek çalışma dört ayaklı robotlarla yapılmış.

Bu proje üçünü birleştiriyor. Ek olarak taktikleri sayısal olarak **ölçmeyi** hedefliyor; önceki çalışmalar
taktikleri sadece sözle anlatıyor.

---

## 2. Pekiştirmeli öğrenme (RL) en baştan

### Temel fikir: deneme yanılma
Bir köpeğe "otur" öğretmeyi düşün. Ona nasıl oturacağını anlatamazsın. Bir şeyler yapar; oturduğunda ödül verirsin.
Zamanla ödül getiren davranışı tekrarlamaya başlar. RL tam olarak bu.

| Kavram | Anlamı | Bu projede |
|---|---|---|
| **Ajan** | Öğrenen taraf | Bir drone (kovalayan ya da kaçan) |
| **Ortam** | Ajanın içinde yaşadığı dünya | Arena + fizik + kurallar |
| **Gözlem** | Ajanın o an bildiği şeyler | Konumu, hızı, rakibin yeri, etrafındaki engeller… |
| **Eylem** | Ajanın yaptığı şey | "Şu yöne, şu hızla uç" |
| **Ödül** | Yaptığının ne kadar iyi olduğunu söyleyen sayı | Kazanınca +1, kaybedince −1 |
| **Politika** | Ajanın "beyni": gözlemden eyleme kural | Bir sinir ağı |

Döngü saniyede 50 kez dönüyor:

```
gözlem → (sinir ağı) → eylem → ortam ilerler → ödül + yeni gözlem → ...
```

### Neden RL, neden normal (denetimli) öğrenme değil?
Denetimli öğrenmede her örnek için **doğru cevabı** veririz ("bu resim kedi"). Burada doğru cevap yok. Bir maçın
ortasındaki bir anda "doğru hamle şuydu" diyebilecek bir uzman yok; en iyi stratejiyi kimse bilmiyor. Elimizde sadece
maçın sonu var: kazandın ya da kaybettin. RL tam olarak bu durum için var: **sadece sonucu bilerek davranış öğrenmek**.

### Asıl zorluk: hangi hamle sonucu belirledi?
Bir maç 1000 karardan oluşuyor ve sonunda tek bir sayı geliyor: +1 ya da −1. Maçı hangi karar kazandırdı? Buna
*kredi atama problemi* deniyor ve RL'in en zor kısmı. Bölüm 6'da nasıl çözüldüğü anlatılıyor.

---

## 3. Simülasyon

### Neden simülasyon?
Gerçek drone'larla milyonlarca maç oynamak imkânsız; her çarpmada drone kırılır. Bu yüzden her şey bilgisayarda
simüle ediliyor. Projenin "verisi", drone'ların simülasyonda oynayarak ürettiği maçlar.

### Katmanlar
Bir drone'un her kararı dört katmandan geçiyor:

1. **Sinir ağı (öğrettiğimiz):** Gözlemi alıp bir hedef hız üretiyor, örneğin "sağa 1,5 m/s, yukarı 0,3 m/s".
2. **Kontrolcü (hazır):** Gerçek Crazyflie drone'unun kendi kontrol yazılımının kopyası. Hedef hızı motorların ne kadar
   dönmesi gerektiğine çeviriyor. Biz sadece stratejiyi öğretiyoruz; "nasıl uçulur" kısmını bu kontrolcü yapıyor.
3. **Fizik (hazır):** **Crazyflow** simülatörü. Gerçek Crazyflie drone'unun uçuş verisine göre ayarlanmış; ivmesi ve
   tepki gecikmesi gerçek drone'unkiyle aynı. Saniyede 500 fizik adımı hesaplıyor.
4. **Oyun katmanı (bu projede yazıldı):** Arena, engeller, kurallar, çarpma ve ebeleme kontrolü, görüş hattı, ödüller.

### Neden bu kadar hızlı?
Crazyflow **JAX** ile yazılmış. JAX hesaplamaları ekran kartında (GPU) binlerce kopya halinde aynı anda yapabiliyor.
**16.384 arena aynı anda** oynuyor. Google Colab'daki A100 ekran kartında saniyede ~1,9 milyon oyun adımı; 500 milyon
adımlık bir eğitim ~5 dakika sürüyor. Eğitimler bu yüzden Colab'da yapılıyor; yerel bilgisayar sadece küçük testler için.

---

## 4. Oyunun kuralları ve ödüller

### Kurallar

| Kural | Değer | Neden |
|---|---|---|
| Arena | 6×6×2 m | WCT sahasının drone ölçeğine küçültülmüşü |
| Süre | 20 s | WCT ile aynı |
| Ebeleme | iki drone'un merkezi 0,3 m'den yakın | "elle dokunma"nın karşılığı |
| Çarpma | engele, duvara, zemine ya da tavana değen kaybeder | WCT: sahadan çıkan kaybeder |
| Hız | ikisi de en fazla 2 m/s | Sonucu hız değil taktik belirlesin |
| Görüş | 360°, ama arada engel varsa göremez | Saklanma buradan doğar |
| Başlangıç | karşı köşeler | WCT'de oyuncular karşı taraflardan başlar |

### Ödüller

| Ne oldu? | Kovalayan | Kaçan |
|---|---|---|
| Ebeleme | +1 | −1 |
| Süre doldu | −1 | +1 |
| Kaçan çarptı | +1 | −1,5 |
| Kovalayan çarptı | −1,5 | +1 |

- Çarpan taraf ek 0,5 ceza alıyor: çarpmak normal kaybetmekten daha kötü olsun, drone'lar güvenli uçmayı öğrensin.
- **Küçük ipuçları:** Kovalayan rakibe yaklaştıkça minik bir ödül alıyor; kaçan her hayatta kaldığı an için çok çok
  küçük bir ödül alıyor. Bunlar başlangıçta yol göstermek için; asıl ödül maç sonundaki ±1.

---

## 5. Arenalar ve ajanın "gözü"

### Engeller
Engeller kutular ve dikey silindirler. Dört türü var:

- **Tam boy:** yerden tavana; üstünden de altından da geçilemez.
- **Alçak:** üstünden geçilebilir.
- **Sarkan:** tavandan sarkıyor, altından geçilebilir.
- **Havada duran platform:** altından da üstünden de geçilebilir.

Öğrenme kademeli planlandı: önce sadece tam boy engeller (faz 1), sonra alçaklar (faz 2), sonra hepsi (faz 3).

### Arenaların kontrolü
Binlerce rastgele arena önceden üretiliyor. Her biri iki kontrolden geçiyor:

- **Köşeler boş mu?** Drone'lar engelin içinde başlamasın.
- **Kapalı cep var mı?** Engeller bir bölgeyi tamamen kapatırsa kaçan oraya girip kovalayanın asla ulaşamayacağı yerde
  bekler ve bedavaya kazanır. RL ajanı böyle bir açığı mutlaka bulur. Bu yüzden her arena 10 cm'lik bir ızgara üzerinde
  taranıyor; kapalı cep varsa atılıyor.

Değerlendirmeler hep eğitimde hiç görülmemiş arenalarda yapılıyor; ajan arenayı ezberlemiş olmasın.

### Ajan ne görüyor? (Gözlem: 33 sayı)
- Kendi konumu ve hızı.
- Rakibe doğru yön ve mesafe. **Arada engel varsa sadece son gördüğü yeri** biliyor.
- Şu an rakibi görüyor mu, kaç saniyedir görmüyor.
- Rakibin hızı (sadece görüyorsa).
- **18 ışın:** Etrafına 18 yöne lazer gibi bakıp en yakın engelin ne kadar uzakta olduğunu ölçüyor (yarasaların sesle
  "görmesi" gibi).
- Kalan süre.

### Ajan ne yapıyor? (Eylem)
Üç sayı: hedef hızın x, y, z bileşenleri, en fazla 2 m/s.

### Beyin: sinir ağı
Her rolün kendi ağı var: 2 katman, katman başına 256 nöronluk basit bir ağ (MLP). 33 sayıyı alıp 3 sayı veriyor.

---

## 6. Öğrenme algoritması (PPO)

### İki ağ: aktör ve critic
- **Aktör:** Drone'u yöneten ağ (Bölüm 5'teki).
- **Critic (eleştirmen):** "Şu anki durum ne kadar iyi?" tahmini yapan ikinci bir ağ. Örneğin "kaçan köşeye sıkışmış,
  kovalayan için iyi durum, büyük ihtimalle +1 gelecek" der. Sadece eğitimde kullanılıyor; oyunda drone'u yönetmiyor.

### Avantaj: kredi atamanın çözümü
Her eylem için şu hesaplanıyor:

> **Avantaj = (gerçekte ne oldu) − (critic ne bekliyordu)**

- Pozitifse eylem beklenenden iyi sonuç verdi; aktör o eylemi daha sık yapmaya başlar.
- Negatifse beklenenden kötü; daha seyrek yapar.

Kazanılan bir maçta bile, beklenenden kötü giden anlardaki eylemler cezalandırılıyor. Maçın tamamı yerine her an ayrı
ayrı değerlendiriliyor. Bu hesabın özel bir türü kullanılıyor (GAE).

### Bir PPO turu
1. **Oyna:** 16.384 arenada 64'er adım; ~1 milyon karar toplanıyor.
2. **Değerlendir:** Her karar için avantaj hesaplanıyor.
3. **Güncelle:** Aktör ve critic güncelleniyor. PPO'nun kendine has özelliği **kırpma (clipping)**: her turda politika
   en fazla ~%20 değişebiliyor. Büyük bir güncelleme iyi giden bir ajanı bir anda bozabilir; PPO küçük ve güvenli
   adımlarla ilerliyor.
4. Tekrar. Bir eğitim ~476 tur.

### Önemli ayarlar
- **γ (gamma), "gelecek ne kadar önemli?":** Kovalayan için 0,995 (~4 saniye ilerisini önemsiyor), kaçan için 0,998
  (~10 saniye). Kaçanın ödülü 20 saniye sonunda geldiği için ufku daha uzun olmalı. Ayrıntı: Kısım B, soru 5.
- **Std (keşif gürültüsü):** Aktör eylemini biraz rastgelelikle seçiyor. Keşif için şart: ajan hep aynı şeyi yaparsa
  daha iyisini hiç denemez. Ayrıntı: Kısım B, soru 5.
- **Asimetrik critic:** Critic rakibin gerçek konumunu da görüyor, aktör görmüyor. Critic oyunu oynamadığı için bu bir
  hile değil; daha iyi tahmin, daha temiz avantaj, daha hızlı öğrenme demek.

### Neden PPO?
- **DQN** gibi yöntemler sayılabilir eylemler ("sol, sağ, ateş") istiyor; bizim eylemimiz sürekli bir hız.
- **SAC** gibi yöntemler eski maçları tekrar tekrar kullanıyor; rakip sürekli değişirken eski maçlar yanıltıcı.
- PPO kararlı ve en yakın çalışmaların hepsi onu kullanıyor.

---

## 7. Elle yazılmış rakipler

RL'den önce kuralları elle yazılmış iki "bot" yazıldı:

- **Kovalayan:** Arenanın haritasını biliyor, GPS gibi engellerin etrafından en kısa yolu buluyor. Rakibi kaybedince
  son görüldüğü yere gidip arıyor.
- **Kaçan:** OPEN adlı makalenin kaçanı, kendi kodlarından. Mıknatıs gibi kovalayandan, duvarlardan ve engellerden
  itiliyor.

Üç işleri var:

1. Kuralların doğru çalıştığını görmek.
2. RL ajanlarının ilk antrenman rakibi olmak (iki rastgele ağ birbirine karşı oynarsa hiçbir şey öğrenilmez).
3. Makalede kıyas çubuğu olmak: "RL, el yazımı stratejiden şu kadar iyi."

Ayrıntılı talimatlar ve sonuçlar: Kısım B, soru 6.

---

## 8. Şimdiye kadarki deneyler

### Deney 1: Kovalayan, OPEN'ın kaçanına karşı (ödül hilesi)
Kovalayanın kazanma oranı %99,9'a çıktı ama **ebeleme %0**'dı. Kovalayan hiç dokunmadan, uzaktan küçük dikey
hareketlerle kaçanın aşağı yukarı salınımını büyüttü (salıncaktaki çocuğu doğru anda itmek gibi). Kaçan sonunda yere
çakıldı. Kural "rakip çarparsa sen kazanırsın" dediği için kovalayan bunu kazanmanın en kolay yolu olarak buldu.

Buna **ödül hilesi (reward hacking)** deniyor: ajan istediğimizi değil, ödülün söylediğini yapıyor. RL'in en meşhur
dersi.

### Deney 2: Aynısı, rakibi çarptırmak ödülsüz
Kovalayanın tek kazanma yolu ebelemek oldu ve gerçekten yakalamayı öğrendi: görülmemiş arenalarda **%99,8 ebeleme,
ortalama 3,8 saniyede**. Elle yazılmış kovalayan aynı kaçanı ~%90'da ve ~9,6 saniyede yakalıyordu.

### Deney 3: Kaçan, elle yazılmış kovalayana karşı
Kaçan **%99,8** dayanmayı öğrendi. Taktiği dikey düzlemde daire çizmekti: "her an rakibin şu anki yerine uç" diyen bir
kovalayana karşı matematikte bilinen bir kaçış yolu.

Deney 2 ve 3 **doğrulamaydı**: PPO iki rolü de öğretebiliyor mu? Cevap evet.

### Deney 4: Self-play (iki taraf birbirine karşı)
Asıl yöntem bu. Elle yazılmış rakiplerin zaafı var ve ajan onu buluyor (Deney 1 ve 3). Rakip de öğrenirse bu zaaflar
kapanır ve taraflar birbirini sürekli zorlar.

- **Nesil:** Bir taraf öğrenir, öbürü donuk kalır. Öğrenen, en son rakibine karşı %60 kazanınca sıra değişir.
- **Rakip havuzu:** Her neslin sonunda öğrenenin bir kopyası havuza eklenir. Öğrenen maçların yarısını en son rakiple,
  yarısını eski sürümlerle oynar; eski taktiklere karşı nasıl oynanacağını unutmasın.
- **PFSP:** Eski sürümlerden zorlandığı rakibi daha sık seçme kuralı (Kısım B, soru 7).

**Sonuç: kovalayan tamamen baskın çıktı.** 20 neslin çoğunda kovalayan en son kaçanı ~%100 yakaladı; kaçan
nesillerinde en son rakibe karşı kazanma %0'da kaldı. İki neden bulundu (ayrıntı: Kısım B, soru 8):

1. **Oyunun yapısı:** Kapalı bir alanda, eşit hızda ve sıfırdan büyük bir yakalama mesafesi varsa iyi oynayan bir
   kovalayan her zaman yakalar. Boş arenada kaçanın saklanacak ya da etrafında döneceği bir şey yok. WCT sahalarının
   engellerle dolu olmasının nedeni de bu.
2. **Keşfin çökmesi:** Kaçan sürekli kaybettiği için rastgeleliğini neredeyse sıfıra indirdi ve yeni şey denemeyi
   bıraktı.

**Düzeltme (20 Kasım):** Boş arenayı entropi bonusuyla tekrar koşunca kaçan nesilleri en son rakibe karşı %15–61
kazandı ve taraflar sırayla birbirini yenmeye başladı. Yani asıl neden keşfin çökmesiymiş, oyunun yapısı değil. Ayrıca
engelli ilk self-play'de bir uygulama kusuru (eylemlerin doyması) bulundu ve düzeltildi. Ayrıntı: Kısım B, soru 8.

---

## 9. Yol boyunca çıkan dersler

- **Ödül hilesi gerçek:** Ajan ödülün açıklarını buluyor (Deney 1).
- **Sabit rakip sömürülür:** İki ajan da elle yazılmış rakibin akla gelmeyen bir zaafını buldu.
- **Ortam hataları sessizdir:** Testler iki hata yakaladı: bir ışın sensörünün −560 milyon metre göstermesi ve drone'un
  sıfır komutta tavana süzülmesi. İkisi de çökme üretmiyor, sadece "garip davranış" olarak görünüyordu. RL'de en
  tehlikeli hata türü bu.
- **Taktikler kendiliğinden çıkıyor:** Rakibi sürmek, kovalayanı çarptırmak, dikey daire, önünü kesmek. Hiçbiri
  öğretilmedi.

---

## 10. Şu an neredeyiz?

### Takvim

| Dönem | İş | Durum (20 Kasım) |
|---|---|---|
| 13–22 Kasım | Ortam, kurallar, elle yazılmış rakipler | Bitti |
| 23 Kasım–6 Aralık | PPO doğrulama, boş arenada self-play | Bitti (takvimden önde) |
| 7–18 Aralık | Engelli self-play, ara sunum | Sırada |
| 19 Aralık–teslim | Ablasyon, taktik analizi, makale | — |

### Sıradaki deney
**Engelli arenada self-play.** Asıl soru ilk kez sorulacak: engeller dengeyi değiştiriyor mu, kaçan saklanmayı öğreniyor
mu? Öneriler: faz 1 engeller (arena başına 8–12), keşfin çökmemesi için entropi bonusu, karşılaştırma için boş arenayı
entropiyle tekrar koşmak.

### İş akışı
1. Kod yazılıp yerel bilgisayarda küçük boyutla test ediliyor.
2. GitHub'a gönderiliyor.
3. `notebooks/runs/` altına o deneyin notebook'u konuyor.
4. Notebook Colab'da açılıp çalıştırılıyor; sonuçlar Drive'a yazılıyor.
5. Sonuçlar `experiments/` altına ve deney kaydına (`experiments/LOG.md`) işleniyor.

### Nerede ne var?

| Klasör | İçerik |
|---|---|
| `src/chase_tag/` | Asıl kod: kurallar, arenalar, fizik bağlantısı, elle yazılmış rakipler, PPO, self-play |
| `tests/` | Kodun doğru çalıştığını kontrol eden testler |
| `scripts/` | Eğitimi başlatma, video çekme, grafik ve analiz araçları |
| `notebooks/runs/` | Her deneyin Colab notebook'u |
| `experiments/` | Her deneyin sonuçları + `LOG.md` (deney günlüğü) |
| `docs/` | Proposal, ders dokümanı, literatür notları, bu rehber |

---

# Kısım B: Sorular ve cevaplar

## 1. Boş arenada kovalayanın hep kazanması literatürde var mı? Atıf yapılabilir mi?

**Evet, iki farklı literatürde.**

### a) Matematik: "aslan ve insan" problemi
1930'lardan beri bilinen bir kovalama problemi: bir arenada aslan (kovalayan) ve insan (kaçan) eşit hızda koşuyor;
aslan yakalayabilir mi?

- **Besicovitch'in sonucu:** Yakalama "tam aynı noktaya gelmek" demekse ve zaman sürekliyse, insan spiral bir yolla
  **sonsuza kadar kaçabilir**. Aslan çok yaklaşır ama asla tam yakalayamaz.
- **Yakalama yarıçapı eklenince durum tersine dönüyor.** Aslanın "uzanan bir pençesi" varsa, yani belli bir mesafeye
  girmek yakalama sayılıyorsa (bizdeki 0,3 m gibi), aslan kazanıyor. Alexander, Bishop ve Ghrist, tek bir kovalayanın
  bu kuralla belli tür kapalı alanlarda (kutu şeklindeki arenamız gibi dışbükey alanlar bunlara dahil) kaçanı her zaman
  yakalayabildiğini göstermiş.
- Alonso, Goldstein ve Reingold yuvarlak bir arenada yakalama süresi için bir üst sınır vermiş: arena büyüdükçe artan,
  yakalama mesafesi küçüldükçe yavaşça artan bir süre.

**Dikkat:** Bu teoremler idealize bir oyun için. Ataletsiz, anında yön değiştirebilen noktalar varsayılıyor; bizim
drone'larımızın ataleti ve tepki gecikmesi var. Teoremlerin tam ifadeleri orijinal makalelerden henüz okunmadı. Dürüst
ifade şu olur: *"Bulgumuz, yakalama yarıçaplı aslan-insan probleminin teorik sonucuyla **uyumlu**."* "Kanıtlıyor"
denemez.

### b) RL literatürü
Literatür notlarına göre Gavin ve arkadaşlarının 1'e 1 drone çalışmasında ikisi de öğrenen drone'lar boş arenada
oynamış: kovalayan küçük arenada (8×8×5 m) kaçanı **%78,8**, büyük arenada (40×40×14 m) **%66,5** yakalamış
(maç 10 s). Orada da kovalayan lehine, ve arena küçüldükçe daha da lehine. Bizim arena daha küçük (6×6×2 m) ve maç
daha uzun (20 s); daha güçlü bir kovalayan üstünlüğü beklenen bir şey. (Bu sayılar notlardan; makaleye geçmeden önce
orijinal tablodan kontrol edilmeli.)

**Ters örnek:** AgilePE'de havuzsuz self-play'de kaçan baskın çıkmış. Ama orada "yakalama" dokunmak değil, rakibi
kameranın görüş alanında tutmak; kurallar farklı.

**Sonuç:** Atıf yapılabilir ve makaleyi güçlendirir: "Boş arenada self-play kovalayanın üstünlüğüne yakınsadı; bu hem
aslan-insan teorisiyle hem Gavin ve arkadaşlarının bulgusuyla uyumlu. WCT'nin sahayı engellerle doldurmasının nedeni de
bu."

---

## 2. Drone'un boyutu

- **Gerçek Crazyflie 2.1 Brushless:** Çapraz motor merkezinden motor merkezine **100 mm**. Kalkış ağırlığı bacaklarıyla
  **34 g**, pervane korumalarıyla 37 g (Bitcraze ürün sayfası).
- **Simülasyondaki model** (`cf21B_500`): 43,4 g. Gerçek ağırlıktan biraz fazla; modelin pil gibi ek parçaları içerip
  içermediği bilinmiyor. Kaynaklar arasında da 32–37 g gibi farklı değerler var.
- **Çarpma hesabında** drone 6 cm yarıçaplı bir küre, yani 12 cm çaplı. 10 cm'lik gövdeyi ve pervaneleri kabaca kapsıyor.

---

## 3. Critic neye göre tahmin ediyor?

**Ne tahmin ediyor:** "Bu durumdan itibaren, *şu anki* politikayla oynamaya devam edersem, gelecekte toplam ne kadar
ödül alırım?" Gelecek ödüller γ ile indirimli toplanıyor. Kısaca **durumun değeri**, V(s).

Örnek, kovalayanın critic'i:

- Kaçan köşeye sıkışmış, aradaki mesafe 1 m, görüş açık: değer yüksek, yakında +1 bekleniyor.
- Kaçan engel arkasında kaybolmuş, 15 saniye kalmış: değer orta.
- 1 saniye kalmış, kaçan 4 m uzakta: değer düşük, süre dolacak ve −1 gelecek.

**Neye bakarak:** Critic'in girdisi 57 sayı: aktörün gördüğü 33 sayıya ek olarak rakibin *gerçek* konumu, gerçek hızı
ve rakibin etrafındaki engeller (onun ışın ölçümleri). Rakip engel arkasında olsa bile critic onun nerede olduğunu
biliyor.

**Nasıl öğreniyor:** Normal bir regresyon problemi gibi. Her turdan sonra elimizde "bu durumdan sonra gerçekte şu
ödüller geldi" bilgisi var; critic tahminini bu gerçek değere yaklaştırıyor (kayıp: farkın karesi). Gerçek değer tam
maç sonuna kadar beklenmeden, kısmen critic'in kendi sonraki tahminleri kullanılarak hesaplanıyor (GAE).

**Ne kadar iyi tahmin ediyor:** Log'lardaki `ev` (açıklanan varyans) bunu ölçüyor: 1 kusursuz tahmin, 0 "hep ortalamayı
söylemek" kadar iyi. Bu projedeki koşularda genelde 0,85–0,99.

**Neden önemli:** Avantaj = gerçek − critic'in tahmini. Critic ne kadar iyi tahmin ederse avantaj o kadar temiz olur ve
ajan hangi eylemin iyi olduğunu o kadar net görür.

---

## 4. Ajanın gördükleri literatürde nasıl?

| Çalışma | Ajan ne görüyor |
|---|---|
| **OpenAI saklambaç** | 30 lidar ışını; nesnelerin konum ve hızı; görüş hattı kesilen ya da 135°'lik görüş konisinin dışında kalan nesneler gizleniyor |
| **AgilePE** | Kendi durumu; rakibin **son birkaç adımlık geçmişi**; rakip görüş alanı dışındaysa ya da engel arkasındaysa gizleniyor |
| **KTH** | Rakibin şu anki ve **son 4 adımdaki** göreli konumu; kendi yönelimi, hızı, açısal hızı; konum ya da yükseklik; kalan süre |
| **OPEN** | Kendi durumu; takım arkadaşları; **en yakın 3 engel**; kaçan görünmüyorsa konumu sabit bir sayıyla (−5) gizleniyor |
| **Gavin vd.** | Geçmiş yok; critic rakibin durumunu ve eylemini de görüyor (asimetrik critic) |
| **Bu proje** | Kendi konum ve hızı; rakibin göreli konumu (görmüyorsa **son gördüğü yer**); görünürlük ve kaç saniyedir görmediği; 18 ışın; kalan süre |

**Kimden ne alındı:**

- **Işınlar + görüş hattı maskesi:** OpenAI saklambaç ve AgilePE.
- **Göreli konum + kalan süre:** KTH.
- **Rakip görünmezken bir şey koymak:** OPEN. Burada sabit bir sayı yerine son görülen yer ve görünmeme süresi
  veriliyor; daha bilgilendirici.
- **Asimetrik critic:** OpenAI ve Gavin.

**Eksik olan:** Rakibin **geçmişi**. KTH ve AgilePE son birkaç adımı veriyor; bizim ajan sadece şu anı görüyor. Rakibin
nereye gittiğini tahmin etmek için geçmiş ya da bir bellek (GRU) faydalı olabilir. Ablasyon listesinde 5. sırada
("MLP ve GRU").

---

## 5. γ ve std neden önemli?

### γ (gamma): "gelecek ne kadar önemli?"
Ajan bir ödülü ne kadar ileride alırsa ödül o kadar indirimli değerlendiriliyor: t adım sonraki bir ödülün bugünkü
değeri **γᵗ** ile çarpılıyor. Saniyede 50 adım olduğu için:

| γ | 2 s sonra (100 adım) | 4 s sonra | 20 s sonra (1000 adım) |
|---|---|---|---|
| 0,99 | 0,37 | 0,13 | ~0,00004 |
| 0,995 | 0,61 | 0,37 | 0,007 |
| 0,998 | 0,82 | 0,67 | 0,14 |

- **Çok küçükse ajan miyop olur.** γ = 0,99'da 4 saniye sonraki ebeleme neredeyse değersiz; kovalayan uzaktaki kaçanı
  kovalamanın değerini göremez. Kaçan için 20 saniye sonundaki +1 tamamen görünmez olur.
- **Çok büyükse öğrenme gürültülenir.** Uzak gelecek belirsiz; critic'in tahmini zorlaşır, avantajlar gürültülü olur.
- Bu yüzden role göre seçildi: kovalayan 0,995 (ebeleme genelde ~4 saniyede oluyor), kaçan 0,998 (ödülü maçın sonunda).

### Std: keşif gürültüsü
Aktör her an bir **ortalama hız** öneriyor; gerçek eylem bu ortalamanın etrafında rastgele seçiliyor. Std bu
rastgeleliğin büyüklüğü.

- **Neden gerekli:** RL deneme yanılmayla öğreniyor ve denemek için farklı şeyler yapmak lazım. Std sıfırsa ajan hep
  aynı şeyi yapar; daha iyi bir hareket olsa bile onu hiç denemez. Buna **keşif (exploration)** deniyor.
- **Denge:** Başta std büyük olmalı (ajan bir şey bilmiyor). Ajan öğrendikçe std doğal olarak küçülür; doğrulama
  koşularında 0,6'dan ~0,2'ye düştü, sağlıklı. Ama **çok** küçülürse ajan yeni şeyleri denemeyi bırakır; self-play'de
  kaçana olan buydu (soru 8).

---

## 6. Elle yazılmış rakipler: talimatlar, sonuçlar, atıflar

### Elle yazılmış kaçan: OPEN'ın kaçanı
**Talimatı (kuvvet alanı):** Kaçan her an kendisini iten kuvvetleri topluyor ve toplam kuvvetin yönünde tam hızla
uçuyor:

- Kovalayandan uzaklaştıran kuvvet; büyüklüğü 1/mesafe. **Sadece kovalayan görünüyorsa** sayılıyor.
- Her duvardan, zeminden ve tavandan uzaklaştıran kuvvet; 1/mesafe.
- Her engelden uzaklaştıran kuvvet; 1/mesafe.

Mıknatıslar gibi: her şey kaçanı kendinden itiyor, kaçan itilmenin en az olduğu yöne kaçıyor.

**Kaynak:** Chen ve arkadaşları, "OPEN", IEEE RA-L 2025; formül kendi açık kodlarından alındı. Atıf yapılabilir.
Belirtilmesi gerekenler: kodlarındaki hız hesabında bir hata vardı (çapraz giderken √3 kat hızlanıyordu), makaledeki
tarife uyuldu; drone gerçek bir kontrolcüyle uçtuğu için üstüne bir fren filtresi eklendi.

**Sonuçları:**
- Elle yazılmış kovalayana karşı ~%10 dayandı: boş arenada %89, 4 engelde %90, 8 engelde %82 yakalandı (CPU duman
  testi; resmi koşu kaydı yok).
- RL kovalayanına karşı %99,8 yakalandı; WCT kuralıyla da rezonansla yere düşürüldü.
- Zaafı: kovalayan görünmeyince onu unutuyor, köşede duvar kuvvetlerinin dengelendiği noktada salınıyor ve sıkışıyor.

### Elle yazılmış kovalayan: bu projenin tasarımı
**Talimatı:**

1. **Kaçanı görüyorsa ve arada engel yoksa:** Ona doğru düz uç. Bu, *saf takip (pure pursuit)* denen klasik kural:
   her an hedefin *şu anki* yerine yönel. KTH, Sandia ve Gavin bunu baseline olarak kullanıyor.
2. **Arada engel varsa:** Arenayı 20 cm'lik karelere bölüp hedefe en kısa yolu bul (GPS gibi) ve onu izle. Klasik
   ızgarada en kısa yol algoritmasının (Bellman-Ford/Dijkstra) bir türü.
3. **Kaçanı göremiyorsa:** Son gördüğü yere git; kaçanın son hızıyla "şimdi nerede olabilir" diye tahmin et. Bulamazsa
   rastgele noktaları gez.
4. **Hep:** Fren filtresi. "Şu hızla giderken o engelde durabilir miyim?" kontrolü; durabileceği hızı aşan kısmı
   kırpıyor. Robotikteki *control barrier function* fikrinin basit bir hali.

**Kaynak:** Bir makaleden alınmadı; parçaları klasik yöntemlerden. Makalede "kendi tasarladığımız, haritayı bilen bir
baseline" diye tanıtılır ve parçaların kaynakları verilir (saf takip için KTH ya da Gavin, fren filtresi için kontrol
bariyer fonksiyonları literatürü).

**Sonuçları:**
- OPEN'ın kaçanını ~%90 yakaladı.
- RL kaçanına karşı **çöktü**: kaçan dikey düzlemde daire çizerek %99,8 dayandı. Saf takibin bilinen zaafı: hedef daire
  çizerken kovalayan hep geride bir iç daireye oturuyor.

---

## 7. PFSP nedir? Başka kurallar var mı?

### Kökeni
1. **Fictitious play** (Brown, 1951; oyun teorisi): Her turda rakibin şimdiye kadar oynadığı **bütün** stratejilerin
   karışımına karşı en iyi cevabı ver.
2. **Fictitious Self-Play** (Heinrich, Lanctot ve Silver, 2015): Bu fikrin RL ve self-play versiyonu.
3. **Prioritized FSP (PFSP)** (DeepMind AlphaStar, StarCraft II, Nature 2019): Geçmiş rakipleri eşit sıklıkta değil,
   **zorlandıklarını daha sık** seçerek oyna.

### Bu projedeki kural
Rakip j'ye karşı kazanma oranın x ise o rakibin seçilme ağırlığı **(1 − x)²** (Gavin ve Bronz'un tarifi).

- x = %90 (kolay rakip) → ağırlık 0,01, neredeyse hiç seçilmez.
- x = %50 → ağırlık 0,25.
- x = %10 (zor rakip) → ağırlık 0,81, sık seçilir.

Kare almak zor rakiplere odağı keskinleştiriyor. AlphaStar da benzer bir zorluk ağırlıklandırması kullanıyor; tam
formülü kaynağında doğrulanmadı.

### Başka kurallar

| Kural | Nasıl çalışıyor | Kim kullandı |
|---|---|---|
| **Naive (saf) self-play** | Sadece en son rakip | Ablasyon 1'in "havuzsuz" kolu |
| **δ-uniform** | Rakip, geçmişin son (1−δ) kısmından eşit olasılıkla seçilir; δ = 1 sadece en son rakip, δ = 0 bütün geçmiş | Bansal vd. 2018; sadece en son rakibe karşı eğitim en kötü sonucu vermiş |
| **%80 en son, %20 geçmiş** | Geçmiş rakipler bir "kalite puanına" göre seçiliyor | OpenAI Five (Dota 2); amaç strateji çöküşünü önlemek |
| **p_old** | p olasılıkla eski rakiplerden, aksi halde en sonuncu | KTH |
| **Lig (league)** | Ana ajanların yanında sadece onların zaafını bulmak için eğitilen "sömürücü" ajanlar | AlphaStar |
| **PSRO** | Popülasyondaki stratejiler arasında oyun teorisiyle en iyi karışım hesaplanıyor | Lanctot vd. 2017 |
| **Bu proje** | %50 en son rakip + %50 PFSP | Gavin ve Bronz'un tarifi |

Hepsinin çıkış noktası aynı: sadece en son rakibe karşı oynamak unutmaya ve döngüye yol açıyor.

---

## 8. Self-play'in kovalayan lehine çıkmasının iki nedeni

### Neden 1: Oyun yapısal olarak kovalayan lehine

**Teori (soru 1):** Yakalama yarıçapı varsa, kapalı bir arenada eşit hızlı bir kovalayanın kazanma stratejisi var.

**Arenanın geometrisi:**

- Arena 6 m genişliğinde. 2 m/s'de kaçan arenayı 3 saniyede geçiyor. 20 saniyelik bir maçta toplam 40 m yol, yani en az
  6–7 kez bir duvara gelip dönmek zorunda.
- **Her dönüşte kovalayan kazanıyor.** Kaçan duvar boyunca yay çizerek dönerken kovalayan iç taraftan kirişi kesiyor.
  Eşit hızda kısa yolu alan taraf mesafeyi kapatır.
- **Kaçan köşede başlıyor.** İki tarafı duvar; kaçabileceği yönlerin çoğu kapalı. Kovalayan köşegen boyunca geldiği
  için kaçanın iki kaçış yönünü de aynı anda kesebiliyor.
- **Dikey kaçış alanı dar.** Arena 2 m yüksek; çarpmamak için drone ~0,06–1,94 m arasında kalmak zorunda. Kaçanın dikey
  daire taktiği en fazla ~0,9 m yarıçaplı daireye izin veriyor; öğrenen kovalayan nesil 0'da bunun içinden kesmeyi
  öğrendi.

**Verideki kanıt: kaçan öğreniyor ama kazanamıyor.**

- Her kaçan neslinde maç süresi uzuyor (2,5 s'den 3,5–4,8 s'ye); yakalanmayı geciktirmeyi öğreniyor.
- İlk kovalayanlara karşı hâlâ kazanabiliyor (kovalayan_0'a karşı %58'e kadar); öğrenme mekanizması çalışıyor.
- Gelişmiş kovalayanlara karşı oyunun yapısı kaçana kazanma şansı bırakmıyor.

### Neden 2: Kaçanın keşfinin çökmesi

**Veri:** Kaçanın std'si nesiller boyunca düştü: 0,14 → 0,10 → 0,08 → … → **0,03**. Kovalayanınki de düştü ama daha
az (0,17 → 0,10).

**Mekanizma:**

- PPO std'yi her turda şu soruya göre ayarlıyor: rastgele sapmalar ortalamada iyi mi kötü mü sonuç verdi?
- Kaçan neredeyse her maçı kaybediyor. Rastgele bir sapma nadiren işe yarıyor; çoğu zaman kaçanı kovalayana yaklaştırıp
  yakalanmayı hızlandırıyor.
- Gradyan her turda "daha az rastgele ol" diyor; buna karşı koyan hiçbir şey yok (entropi bonusu 0'dı).
- Sonuç: ajan bildiği tek stratejiye kilitlendi.

**Neden kötü:**

1. **Yeni taktik bulamaz.** Keşif olmadan, örneğin "kovalayanın kör noktasından geçmek" gibi bir hareketi hiç denemez.
2. **Tahmin edilebilir olur.** Hep aynı hareketi yapan bir kaçanı kovalayan kolayca ezberler.
3. **Öğrenme kararsızlaşır.** Std çok küçükken ortalamadaki küçücük bir değişiklik eylemin olasılığını çok büyük oranda
   değiştiriyor; PPO'nun "en fazla %20 değiş" kırpması sık sık devreye giriyor ve güncellemeler gürültüleniyor. Veride
   de görünüyor: kaçanın KL'si (bir turda ne kadar değiştiğinin ölçüsü) 0,007'den 0,02'nin üstüne çıktı.

### Hangisi asıl neden? (Düzeltme, 20 Kasım)
İlk yorum "Neden 1 asıl, Neden 2 ikincil" idi. **Kontrol koşusu bunu yanlışladı.** Boş arenayı sadece entropi bonusuyla
tekrar koşunca (`sp-empty-ent-s0`) kaçanın std'si ~0,18'de kaldı ve kaçan nesilleri en son rakibe karşı %0 yerine
%15–61 kazandı; taraflar sırayla birbirini yenmeye başladı. Yani boş arenadaki kovalayan üstünlüğünün **asıl nedeni
keşfin çökmesiydi**. Aslan-insan teorisi idealize bir oyun için; 20 saniyelik süre sınırı ve drone dinamiği varken
kaçana pratikte yer kalıyor.

### Üçüncü bir sorun: eylemlerin doyması
Engelli arenadaki ilk self-play'de (`sp-faz1-s0`) drone'lar 1,2 milyar adım boyunca engellere çarpmaya devam etti.
Nedeni bir uygulama kusuruydu: aktör eylemin *ortalamasını* sınırsız üretiyordu, ortam ise eylemin boyunu 1'e
kırpıyordu. Boş arenada hep tam hız iyi olduğu için ortalamalar ~4,5'e kaydı. Std küçükken denenen bütün eylemler
kırpılınca aynı eyleme dönüştü: farklı deneme yok, farklı sonuç yok, avantaj sıfır; ajan fren yapmayı öğrenemedi.
Düzeltme: ortalama tanh ile [-1, 1]'e sınırlandı. Sıfırdan eğitilen bir kovalayan engelli arenada birkaç yüz bin adımda
çarpmayı büyük ölçüde bıraktı. Ders: sürekli eylemli PPO'da eylemi sınırsız bir Gauss'tan seçip ortamda kırpmak bilinen
bir tuzak; ortalama ya da eylem sınırlanmalı.

---

# Kaynaklar

- [Lion and Man: Can Both Win? (arXiv 0909.2524)](https://arxiv.org/pdf/0909.2524)
- [Alexander, Bishop, Ghrist: Capture pursuit games on unbounded domains](https://ems.press/journals/lem/articles/12139)
- [Alexander, Bishop, Ghrist: Total curvature and simple pursuit](https://www2.math.upenn.edu/%7Eghrist/preprints/totalcurvature.pdf)
- [Bramson, Burdzy, Kendall: Shy couplings, CAT(0) spaces, and the lion and man](https://arxiv.org/pdf/1007.3199)
- [Bopardikar, Bullo, Hespanha: discrete-time pursuit with sensing radius](https://web.ece.ucsb.edu/~hespanha/published/2007ACC_0610_FI.pdf)
- [Bitcraze: Crazyflie 2.1 Brushless](https://bitcraze.io/products/crazyflie-2-1-brushless)
- [DeepMind: AlphaStar](https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
- [Bansal vd.: Emergent Complexity via Multi-Agent Competition](https://arxiv.org/pdf/1710.03748)
- [Berner vd.: Dota 2 with Large Scale Deep RL (OpenAI Five)](https://arxiv.org/pdf/1912.06680)
- [OPEN kodu: thu-uav/Multi-UAV-pursuit-evasion](https://github.com/thu-uav/Multi-UAV-pursuit-evasion)
- Literatür notları: `docs/literatur_notlari.md` (AgilePE, KTH, Gavin vd., OPEN, OpenAI saklambaç ve diğerleri)

**Doğrulanmamış olanlar:** Gavin ve arkadaşlarının yakalama oranları (notlardan), aslan-insan teoremlerinin tam
ifadeleri (arama özetlerinden), AlphaStar'ın PFSP formülü, Brown (1951), Heinrich vd. (2015) ve Lanctot vd. (2017)
referansları. Makaleye geçirmeden önce orijinallerinden kontrol edilmeli.
