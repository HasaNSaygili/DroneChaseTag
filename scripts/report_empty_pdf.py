"""Boş arena raporu (PDF). Sayılar experiments/LOG.md ve koşu dosyalarından; grafikler report_empty_figs.py'den.

Kullanım: önce report_empty_figs.py (bkz. orası), sonra
  python scripts/report_empty_pdf.py experiments <grafik_klasörü> docs/bos_arena_raporu.pdf
reportlab gerekir (proje .venv'inde yok; sistem Python'unda kurulu). Türkçe karakterler için Windows'un Arial
fontu kullanılır (reportlab'ın yerleşik fontlarında ğ, ş, ı yok).
"""
import json
import os
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

EXP, FIGS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
F = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("Arial", F + "arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", F + "arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", F + "ariali.ttf"))
pdfmetrics.registerFont(TTFont("Arial-BoldItalic", F + "arialbi.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-BoldItalic")

INK, MUTED, ACCENT, BOXBG = colors.HexColor("#1f2933"), colors.HexColor("#5f6b7a"), colors.HexColor("#2471a3"), \
    colors.HexColor("#eef4fa")
S = {
    "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=20, leading=24, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="Arial", fontSize=10.5, leading=14, textColor=MUTED, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="Arial-Bold", fontSize=14, leading=18, textColor=ACCENT, spaceBefore=14,
                         spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=11, leading=14, textColor=INK, spaceBefore=8,
                         spaceAfter=4),
    "p": ParagraphStyle("p", fontName="Arial", fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=6),
    "li": ParagraphStyle("li", fontName="Arial", fontSize=9.5, leading=13.5, textColor=INK, leftIndent=12,
                         bulletIndent=2, spaceAfter=2),
    "cap": ParagraphStyle("cap", fontName="Arial-Italic", fontSize=8.3, leading=11, textColor=MUTED, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Arial", fontSize=8.3, leading=10.5, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Arial-Bold", fontSize=8.3, leading=10.5, textColor=INK),
    "box": ParagraphStyle("box", fontName="Arial", fontSize=9.3, leading=13, textColor=INK, spaceAfter=3),
}


def P(t, s="p"):
    return Paragraph(t, S[s])


def bullets(items):
    return [Paragraph(t, S["li"], bulletText="•") for t in items]


def fig(name, width_cm, caption):
    path = os.path.join(FIGS, name)
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(path).getSize()
    img = Image(path, width=width_cm * cm, height=width_cm * cm * h / w)
    return KeepTogether([img, Spacer(1, 3), P(caption, "cap")])


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), S["cellb"] if (header and i == 0) else S["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], hAlign="LEFT")
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d1db")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde7f0")))
    t.setStyle(TableStyle(st))
    return t


def box(title, items):
    inner = [P(f"<b>{title}</b>", "box")] + [Paragraph(t, S["box"], bulletText="•") for t in items]
    t = Table([[inner]], colWidths=[17 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BOXBG), ("BOX", (0, 0), (-1, -1), 0.6, ACCENT),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def footer(canv, doc):
    canv.saveState()
    canv.setFont("Arial", 7.5)
    canv.setFillColor(MUTED)
    canv.drawString(2 * cm, 1.2 * cm, "Drone Chase Tag · Boş arena raporu · 21 Kasım 2026")
    canv.drawRightString(19 * cm, 1.2 * cm, f"sayfa {doc.page}")
    canv.restoreState()


E = json.load(open(os.path.join(EXP, "sp-empty-fix2-s0", "elo", "elo.json"), encoding="utf-8"))
D = json.load(open(os.path.join(EXP, "sp-empty-fix2-s0", "elo_det", "elo.json"), encoding="utf-8"))

st = []
st += [P("Boş arenada öğrenme ve self-play", "title"),
       P("Drone Chase Tag · engelsiz arenadaki bütün deneylerin özeti: temel ajanlar, self-play, bulunan ölçüm hatası "
         "ve Elo turnuvası · 21 Kasım 2026", "sub")]

st.append(box("Kısaca sonuç", [
    "Düzeltilmiş kodla eğitilen temel ajanlar boş arenada elle yazılmış rakiplerini neredeyse kusursuz yeniyor: "
    "kovalayan <b>%99,6</b> yakalıyor (2,9 s), kaçan maçların <b>%100</b>'ünde dayanıyor.",
    "Self-play'de sıra değişimini ölçen kodda bir <b>ölçüm hatası</b> bulundu ve düzeltildi: senkron başlayan maçlar "
    "kaçanın kazanma oranını %76 gösteriyordu, gerçekte ~%51'di.",
    "Düzeltilmiş self-play'de (20 nesil, 765 M adım) iki taraf <b>sırayla birbirini yeniyor</b>.",
    "Elo turnuvası (121 eşleşme × 256 maç): <b>döngü yok, gerçek ilerleme var</b>. Her yeni sürüm rakip tarafın önceki "
    "bütün sürümlerini yeniyor. Puanlar kovalayanda 626 → 1550, kaçanda 84 → 1669; gürültüsüz turnuva aynı tabloyu verdi.",
    "Eğri 20 nesilde düzleşmedi: oyun henüz \"çözülmedi\", daha uzun eğitim daha güçlü ajanlar verirdi.",
]))

# ---------------- 1 ----------------
st += [P("1. Kurulum: ne öğretiyoruz, neden önce boş arena?", "h1"),
       P("İki simüle Crazyflie drone 6 × 6 × 2 m'lik bir arenada 1'e 1 ebelemece oynuyor (World Chase Tag kuralları). "
         "Kovalayan 20 saniye içinde rakibine 0,3 m'den fazla yaklaşırsa kazanıyor; kaçan 20 saniye dayanırsa kazanıyor. "
         "Duvara, zemine, tavana ya da bir engele değen kaybediyor ve rakibi kazanıyor. İki drone'un azami hızı eşit "
         "(2 m/s). Her drone'u ayrı bir sinir ağı yönetiyor: ağ her 1/50 saniyede gözlemini (kendi konumu ve hızı, rakibin "
         "göreli konumu, ışın mesafeleri, kalan süre) alıp bir hız komutu üretiyor."),
       P("Ağlar <b>PPO</b> ile eğitiliyor. PPO bir pekiştirmeli öğrenme algoritması: ajan bir sürü maç oynuyor, kazandıran "
         "eylemlerin olasılığını biraz artırıp kaybettirenlerinkini biraz azaltıyor. \"Biraz\" kısmı önemli: PPO tek bir "
         "güncellemede politikanın çok değişmesini engelliyor, öğrenme bu yüzden kararlı."),
       P("<b>Self-play</b>, iki ajanın birbirine karşı oynayarak öğrenmesi. Projenin asıl sorusu engelli arenada self-play'in "
         "saklanma ve arama gibi taktikler üretip üretmediği. Önce boş arenada çalışmamızın iki nedeni var:"),
       *bullets(["<b>Doğrulama:</b> Kod boş arenada çalışmıyorsa engelli arenada hiç çalışmaz. Boş arenada sonucu "
                 "yorumlamak da kolay.",
                 "<b>Kıyas noktası:</b> Makaledeki zorunlu ablasyon adaylarından biri \"engelli ve engelsiz eğitim\" "
                 "karşılaştırması. Engellerin neyi değiştirdiğini görmek için engelsiz sonucun elde olması lazım."]),
       P("Bu rapordaki bütün sayılar <font face='Arial-Bold'>experiments/LOG.md</font>'deki koşulara dayanıyor; "
         "koşu listesi en sonda.")]

# ---------------- 2 ----------------
st += [P("2. Adım 1: Temel ajanlar (sweep-base-empty)", "h1"),
       P("Self-play'e sıfırdan başlamak yerine iki ajan önce <b>elle yazılmış rakiplere</b> karşı eğitiliyor:"),
       *bullets(["Kovalayan, <b>OPEN</b> adlı potansiyel alan kaçanına karşı. OPEN, kovalayandan uzaklaşan ve duvarlardan "
                 "kaçınan kural tabanlı bir kaçan.",
                 "Kaçan, <b>elle yazılmış kovalayana</b> karşı. Bu kovalayan haritayı biliyor, ızgarada en kısa yolu "
                 "buluyor ve rakibi kaybedince arama yapıyor."]),
       P("Bu iki ajan self-play'in başlangıç oyuncuları oluyor. Daha önce de benzer ajanlar eğitmiştik. Ama o koddaki "
         "bir kusur yüzünden (bölüm 3.2, \"gaz pedalı\") koşuyu düzeltilmiş kodla tekrarlamamız ve düzeltmenin boş "
         "arenada bir şeyi bozup bozmadığını görmemiz gerekiyordu. Her iki ajan 500 M adım eğitildi; entropi katsayısı "
         "0,001."),
       fig("fig1_temel.png", 16.5, "Şekil 1. Temel ajanların eğitim eğrisi. Düz çizgi: düzeltilmiş kod; kesikli çizgi: "
           "eski kodla yapılan aynı koşu. Kovalayanın maçları kısalıyor (daha çabuk yakalıyor), kaçanınki 20 s'ye "
           "çıkıyor (hiç yakalanmıyor). Sol grafik eğitim sırasındaki gürültülü politikayla ölçüldü."),
       P("Eğitimden sonra ajanlar <b>hiç görmedikleri arenalarda</b> ve <b>gürültüsüz</b> (her adımda en olası eylemle) "
         "değerlendirildi:"),
       table([["Ajan", "Rakip", "Eski kod", "Düzeltilmiş kod"],
              ["Kovalayan", "OPEN kaçanı", "ebeleme %99,8, ort. 3,8 s (10.102 maç)",
               "ebeleme <b>%99,6</b>, ort. <b>2,9 s</b> (13.486 maç); kaçan çarptı %0,3"],
              ["Kaçan", "elle yazılmış kovalayan", "süre doldu %99,8 (2048 maç)",
               "süre doldu <b>%100</b> (2048 maç), çarpma %0"]], [2.4, 3.4, 4.8, 6.4]),
       Spacer(1, 6),
       P("<b>Sonuç: düzeltme hiçbir şeyi bozmadı</b>; kovalayan rakibini yaklaşık 1 saniye daha hızlı yakalıyor. Öğrenme hızı "
         "da benzer: kovalayan %50'ye 94 M, %98'e 409 M adımda ulaştı (150–300 M arasında %89'da bir plato var); kaçan %50'ye "
         "150 M, %95'e 199 M adımda ulaştı. Eğitim süreleri: kovalayan 7,8 dakika, kaçan 60 dakika (elle yazılmış kovalayan "
         "yavaş, her adımda bir yol araması yapıyor).")]

# ---------------- 3 ----------------
st += [P("3. Arka plan: önceki self-play denemelerinin iki kusuru", "h1"),
       P("Bu temel ajanlardan önce boş arenada iki self-play koşusu yapılmıştı. İkisinin de sonucu yanıltıcıydı. "
         "Düzeltilmiş self-play'in neden gerektiğini anlamak için ne olduğuna kısaca bakalım."),
       P("3.1 Keşfin çökmesi", "h2"),
       P("Ağ tek bir kesin komut üretmiyor: bir <b>ortalama</b> eylem ve onun etrafındaki sapmayı (<b>std</b>) üretiyor. "
         "Drone her adımda ortalamaya küçük bir rastgele sapma eklenmiş eylemi uyguluyor. PPO bu sapmalar sayesinde "
         "<b>keşfediyor</b>: biraz farklı bir şey deniyor, sonuç daha iyiyse o yöne kayıyor. std sıfıra yaklaşırsa ajan "
         "artık yeni bir şey denemiyor ve öğrenmeyi bırakıyor."),
       P("İlk self-play koşusunda (sp-empty-s0) kovalayan baskın çıktı: kaçan nesillerinin hiçbiri en son kovalayana karşı "
         "%0,3'ün üstüne çıkamadı. Önce bunun oyunun yapısından geldiğini düşündük (eşit hızda kovalayan avantajlı). Ama "
         "kontrol deneyi (sp-empty-ent-s0) başka bir şey gösterdi. Bu koşuda kayba küçük bir <b>entropi bonusu</b> eklendi; "
         "bu bonus ajanı std'sini korumaya teşvik ediyor. Bonusla kaçanın keşfi korundu ve oyun dengelendi. Asıl neden "
         "keşfin çökmesiydi."),
       fig("fig2_std.png", 14, "Şekil 2. Kaçan öğrenirken std. Entropi bonusu olmadan (gri) keşif 0,03'e çöküyor; bonusla "
           "(turuncu, mavi) 0,17–0,20 civarında kalıyor. Mavi, bu rapordaki düzeltilmiş koşu."),
       P("3.2 Gaz pedalı kusuru", "h2"),
       P("İlk engelli self-play denemesinde drone'lar 1,2 milyar adım boyunca engellere çarpıp durdu. Nedeni bir uygulama "
         "kusuruydu: ağın eylem ortalaması sınırsızdı ve boş arenada hep tam hız iyi olduğu için ~4,5'e kaymıştı. Ortam "
         "eylemin boyunu 1'e kırptığı için denenen bütün eylemler aynı tam gaz eylemine dönüşüyordu. Farklı deneme yoksa "
         "öğrenme de olmuyor; ajan fren yapmayı öğrenemedi. Düzeltme olarak ortalama artık <b>tanh</b> fonksiyonundan "
         "geçiyor ve her eksende [−1, 1] aralığında kalıyor. Bu kusur boş arenada görünmüyordu, çünkü orada tam gaz "
         "çoğunlukla zaten doğru eylem. Bölüm 2'deki temel ajanların düzeltilmiş kodla yeniden eğitilmesinin nedeni bu."),
       P("Sonuç olarak eski self-play'lerin iki kusuru vardı: keşif çöküyordu ve eylemler doymuştu. Düzeltilmiş kodla "
         "(tanh, entropi 0,001) yeni bir self-play koşusu yapıldı.")]

# ---------------- 4 ----------------
st += [P("4. Bulunan ölçüm hatası (sp-empty-fix-s0)", "h1"),
       P("Self-play sırayla ilerliyor. Bir <b>nesil</b> boyunca sadece bir taraf öğreniyor; öbür taraf donmuş durumda. "
         "Öğrenen taraf rakibinin <b>en son</b> sürümüne karşı maçların %60'ını kazanınca (en az 20 M, en fazla 100 M "
         "adımdan sonra) sıra değişiyor. Öğrenenin donmuş bir kopyası da kendi rolünün <b>havuzuna</b> ekleniyor. Öğrenen "
         "taraf, eğitim maçlarının yarısını en son rakiple, yarısını havuzdan seçilen eski rakiplerle oynuyor (PFSP: "
         "zorlandığın rakip daha sık seçiliyor). Bu sayede eski taktikleri unutmuyor."),
       P("Düzeltilmiş kodla yapılan ilk self-play koşusunda grafiklerde sürekli zıplayan dişler vardı ve kaçan nesillerinin "
         "hepsi tam 21 M adımda bitiyordu. Nedenini araştırınca bir <b>ölçüm hatası</b> bulduk. %60 eşiği, son 5 eğitim "
         "turunda biten maçlardan hesaplanıyordu. Ama:"),
       *bullets(["Her nesil başında 16.384 maçın hepsi <b>aynı anda</b> başlıyor.",
                 "Yakalanmalar dağınık zamanlarda bitiyor. Süre dolan maçların hepsi ise tam 20. saniyede, yani "
                 "<b>aynı turda</b> bitiyor.",
                 "5 turluk pencere bu patlamaya denk gelince kaçanın oranı şişiyor. Patlamadan önceki turlarda ise sadece "
                 "yakalanmalar bittiği için oran %0 görünüyor."]),
       fig("fig3_olcum.png", 14.5, "Şekil 3. Ölçüm hatasının kendisi. Üstte: 16. turda yaklaşık 4900 maç aynı anda "
           "\"süre doldu\" ile bitiyor. Altta: logdaki ölçü (mavi) önce %0, sonra %98 gösteriyor ve eşiği geçiyor. "
           "Patlama sonrası turlarda kaçanın gerçek oranı ~%51 (yeşil): eşiğin altında."),
       P("Bir benzetme: 100 öğrenci aynı anda koşuya başlıyor. Yorulanlar istedikleri an bırakıp çıkıyor, sonuna kadar "
         "dayananların hepsi 20. dakikada birlikte bitiriyor. Kapıda sadece son 1 dakikada çıkanları sayarsan 5. dakikada "
         "\"kimse dayanmıyor\", 20. dakikada \"herkes dayanıyor\" dersin. Doğrusu, herkes bitirdikten sonra sonuçları "
         "toplamak."),
       P("<b>Düzeltme:</b> sıra değişimine artık ayrı <b>kontrol maçları</b> karar veriyor. En az 20 M adım dolunca, "
         "sonra her 5 M adımda bir, öğrenen ile en son rakip arasında 4096 maç sıfırdan başlatılıyor ve hepsi sonuna kadar "
         "oynanıyor. Her maç tam bir kez sayılıyor. Kontrol maçında iki taraf da eğitimdeki gibi olasılıklı oynuyor. "
         "Maliyet küçük çıktı (A100'de kontrol başına ~5 s). Bir birim testi bu mantığı doğruluyor: hareket etmeyen iki "
         "drone verildiğinde bütün maçlar \"süre doldu\" çıkıyor."),
       P("Hatanın eski koşulara etkisi LOG'a not edildi. Hata kaçanın oranını <b>şişirdiği</b> için sp-empty-s0'daki "
         "\"kovalayan baskın\" sonucu geçerli kalıyor (kaçan buna rağmen %0'daydı). sp-empty-ent-s0'daki kaçan oranları "
         "(%15–61) ise muhtemelen şişirilmiş. Eğitim maçlarından hesaplanan diğer şeyler, yani logdaki sonuç yüzdeleri ve "
         "PFSP'nin \"hangi rakip zor\" tahmini, bu yanlılığı hâlâ taşıyor. Bunlar artık sadece gösterge; sıra değişimini "
         "etkilemiyorlar.")]

# ---------------- 5 ----------------
st += [P("5. Adım 2: Düzeltilmiş self-play (sp-empty-fix2-s0)", "h1"),
       P("Ayarlar: havuz bölüm 2'deki iki temel ajanla başlıyor (kovalayan_0, kaçan_0); WCT kuralı (rakip çarparsa sen "
         "kazanırsın); entropi 0,001; sabit öğrenme oranı 3·10<super>-4</super>; 20 nesil (10 kovalayan + 10 kaçan "
         "nesli), ilk olarak kovalayan öğreniyor. Koşu A100'de 13,8 dakika sürdü (765 M adım, 86 kontrol maçı)."),
       fig("fig4_selfplay.png", 16.5, "Şekil 4. Üstte: her neslin kontrol maçları. Bir nesil, öğrenenin rakibin yeni "
           "sürümüne karşı ~%30–45'ten başlayıp %60'ı geçmesiyle bitiyor. Altta: nesillerin uzunluğu. Kaçan nesilleri "
           "(mavi) çoğunlukla en kısa sürede, 21 M adımda bitiyor."),
       P("Gözlemler", "h2"),
       *bullets(["<b>Merdiven:</b> 20 neslin hepsi gerçek eşikle geçti. Her taraf öbürünün son hamlesine bir cevap buluyor.",
                 "<b>Asimetri:</b> Kaçan çok daha çabuk cevap buluyor. Kaçan nesillerinin 7'si ilk kontrolde (21 M) "
                 "%61–82 ile geçti. Kovalayan nesilleri 26–100 M sürdü ve %61–71'de bitti. Eski self-play'deki kovalayan "
                 "baskınlığı oyunun yapısından değil, bölüm 3'teki kusurlardan geliyordu.",
                 "<b>Sıkıştırma:</b> Kovalayanın kontrol maçı kazançlarının bir kısmı (nesil sonunda %6–16) kaçanın "
                 "çarpmasından geliyor. WCT kuralında rakibi duvara ya da yere sıkıştırmak da kazandırıyor. Bunun gerçek bir "
                 "taktik mi yoksa bir sömürü mü olduğuna videolarda bakılmadı.",
                 "<b>Keşif korunuyor:</b> Kaçanın std'si 0,21'den 0,17'ye, kovalayanınki 0,19'dan 0,11'e yavaşça iniyor. "
                 "Çökme yok, ama kovalayandaki düşüş daha uzun koşularda izlenmeli."]),
       fig("fig5_sonuclar.png", 16.5, "Şekil 5. Solda: her neslin son kontrol maçındaki sonuçlar. Kovalayan nesillerinde "
           "ebeleme ve kaçan çarptı (kırmızı + turuncu), kaçan nesillerinde süre doldu (mavi) çoğunlukta. Sağda: öğrenen "
           "tarafın nesil sonu std'si.")]

# ---------------- 6 ----------------
rows = [["Sürüm", "Kovalayan (olasılıklı)", "Kovalayan (gürültüsüz)", "Kaçan (olasılıklı)", "Kaçan (gürültüsüz)"]]
for k in range(len(E["elo_chaser"])):
    up = lambda x: str(int(x + 0.5))       # yarımı yukarı yuvarla (konsol çıktısı ve LOG ile aynı sayılar)
    rows.append([str(k), up(E["elo_chaser"][k]), up(D["elo_chaser"][k]), up(E["elo_evader"][k]), up(D["elo_evader"][k])])
st += [P("6. Elo turnuvası: ilerleme mi, döngü mü?", "h1"),
       P("Şekil 4'teki merdiven iki şekilde yorumlanabilir. <b>Gerçek ilerleme</b> olabilir: kaçan_10, kaçan_1'den "
         "gerçekten daha iyidir. Ya da <b>döngü</b> olabilir: taraflar taş-kâğıt-makas gibi aynı birkaç taktik arasında "
         "dönüyordur; her yeni sürüm sadece son rakibi yenip eskilere kaybediyordur. Bunu ayırt etmek için "
         "<b>bütün</b> sürümleri birbiriyle oynattık."),
       *bullets(["<b>Turnuva:</b> 11 kovalayan sürümü × 11 kaçan sürümü = 121 eşleşme. Her eşleşmeye 256 maç düştü, toplam "
                 "31 bin maç. Hepsi baştan sona oynandı ve hiç görülmemiş arenalarda yapıldı. CPU'da ~10 dakika sürdü.",
                 "<b>Puan (Bradley–Terry):</b> Satrançtaki Elo'nun toplu hali. Her oyuncunun bir puanı var. Kovalayan i'nin "
                 "kaçan j'yi yenme olasılığı, puan farkının lojistik fonksiyonu. 200 puan fark yaklaşık %76 kazanma demek. "
                 "Puanlar bütün sonuçlara birlikte uyduruluyor (en çok olabilirlik), bu yüzden maçların oynanma sırasına "
                 "bağlı değil. Ölçek ortalaması 1000.",
                 "<b>İki mod:</b> <i>Olasılıklı</i> modda iki taraf eğitimdeki gibi küçük bir sapmayla oynuyor; kontrol "
                 "maçlarıyla tutarlı olan bu. <i>Gürültüsüz</i> modda sadece ortalama eylem kullanılıyor. Sonuç ölçüm "
                 "seçimine bağlı mı diye ikisi de koşuldu."]),
       fig("fig6_elo.png", 17, "Şekil 6. Solda: kazanma matrisi. Sol alt üçgen kırmızı, yani her kovalayan önceki bütün "
           "kaçanları yeniyor. Sağ üst mavi, yani her kaçan önceki bütün kovalayanları yeniyor. Sağda: puanlar sürüm sürüm "
           "yükseliyor; iki mod neredeyse üst üste."),
       P("Sonuç", "h2"),
       *bullets(["<b>Döngü yok, ilerleme var.</b> Döngü olsaydı matriste karışık yamalar görürdük; örneğin kovalayan_10 "
                 "eski bir kaçana kaybederdi. Alt üçgende kovalayanın oranı %81–100, üst üçgende %0–25. Yeni sürümler "
                 "eski taktikleri unutmamış. Rakip havuzu işini görüyor.",
                 "<b>Ölçüm seçimine bağlı değil:</b> Olasılıklı ve gürültüsüz turnuva aynı tabloyu veriyor. En büyük fark "
                 "kaçan_8'de (1380 ve 1478). Olasılıklı turnuva iki kez koşuldu, sayılar birebir aynı çıktı.",
                 "<b>Henüz yakınsamadı:</b> Puan eğrisi 20 nesilde düzleşmedi. Daha uzun bir self-play daha güçlü ajanlar "
                 "verirdi.",
                 "<b>Küçük düzensizlikler:</b> Kovalayan_1, kovalayan_0'dan biraz düşük; sadece kaçan_0'a karşı 21 M adım "
                 "eğitildi ve ona fazla uydu. Gürültüsüz turnuvada kaçan_9 ilk kovalayanlara karşı kaçan_8'den zayıf; son "
                 "kovalayanlara uzmanlaşırken biraz unuttu. Kaçan_10'da bu düzeliyor.",
                 "<b>Kaçan_0'ın puanı çok düşük (84).</b> Dikey daire taktiğini daha ilk kovalayan bile %99 çözüyor."]),
       KeepTogether([P("Puan tablosu", "h2"), table(rows, [2.0, 3.6, 3.6, 3.6, 3.6])])]

# ---------------- 7 ----------------
st += [P("7. Sınırlılıklar ve açık sorular", "h1"),
       *bullets(["<b>Tek seed:</b> Her koşu bir kez yapıldı. Rakamların ne kadar oynayacağı bilinmiyor; makale için en "
                 "azından self-play'in 2–3 seed'le tekrarı iyi olur.",
                 "<b>Videolar izlenmedi:</b> \"Sıkıştırma\" (kaçan çarptı kazançları) ve kaçanların taktikleri henüz "
                 "gözle doğrulanmadı.",
                 "<b>PFSP tahmini yaklaşık:</b> Hangi eski rakibin zor olduğu hâlâ eğitim maçlarından tahmin ediliyor ve "
                 "senkron başlangıç yanlılığını taşıyor. Bu sadece rakip seçimini etkiliyor, sıra değişimini etkilemiyor.",
                 "<b>Değerlendirme yöntemleri farklı:</b> Elle yazılmış rakiplere karşı sonuçlar gürültüsüz, self-play ve "
                 "Elo sonuçları olasılıklı. Makalede ikisi ayrı tablolarda, yöntemiyle birlikte verilmeli.",
                 "<b>Sadece simülasyon:</b> Algılama ideal (konum + görüş maskesi), drone–drone teması fizikte yok. "
                 "Ebeleme mesafe kontrolüyle yapılıyor."]),
       P("8. Sonraki adımlar", "h1"),
       *bullets(["<b>Adım 3: engelli temel ajanlar (sweep-base-faz1).</b> Faz 1 engeller (tam boy, 8–12 tane), sıfırdan, "
                 "1,5 milyar adım, kovalayan ve kaçan için 3'er seed. Notebook hazır.",
                 "<b>Adım 4: engelli self-play.</b> Aynı turnuva orada da koşulacak. Bu rapordaki matris ve puanlar "
                 "engelsiz kıyas noktası olacak (ablasyon 3)."])]

# ---------------- Ek ----------------
st += [P("Ek: Koşular", "h1"),
       table([["Koşu", "Commit", "Ne"],
              ["ppo-chaser-empty-fix-s0", "e17b9e2", "temel kovalayan vs OPEN, 500 M adım"],
              ["ppo-evader-empty-fix-s0", "e17b9e2", "temel kaçan vs elle yazılmış kovalayan, 500 M adım"],
              ["ppo-chaser-empty-nocrashrew-s0", "3847d67", "eski kodla kovalayan (Şekil 1, kesikli)"],
              ["ppo-evader-empty-nocrashrew-s0", "007d0f9", "eski kodla kaçan (Şekil 1, kesikli)"],
              ["sp-empty-s0", "—", "eski self-play, entropi 0 (Şekil 2)"],
              ["sp-empty-ent-s0", "42fc3f3", "eski self-play, entropi 0,003 (Şekil 2)"],
              ["sp-empty-fix-s0", "26e4b68", "düzeltilmiş kod, eski ölçüm: hata bulundu (Şekil 3)"],
              ["sp-empty-fix2-s0", "d2ac672", "düzeltilmiş kod + kontrol maçları (Şekil 2, 4, 5)"],
              ["Elo turnuvası", "2ba620c", "scripts/elo.py, sp-empty-fix2-s0/elo ve elo_det (Şekil 6)"]],
             [5.0, 2.0, 10.0]),
       Spacer(1, 6),
       P("Bütün dosyalar repoda <font face='Arial-Bold'>experiments/&lt;koşu&gt;/</font> altında; ayrıntılar "
         "<font face='Arial-Bold'>experiments/LOG.md</font>'de.", "cap")]

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.8 * cm, title="Boş arenada öğrenme ve self-play",
                        subject="Drone Chase Tag: engelsiz arena deneyleri")
doc.build(st, onFirstPage=footer, onLaterPages=footer)
print("yazıldı:", OUT)
