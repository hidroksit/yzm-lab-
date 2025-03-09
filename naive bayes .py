import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

class OzelNaiveBayes:
    def __init__(self, duzeltme=1):
        self.duzeltme = duzeltme  # Laplace düzeltmesi parametresi
        
    def egitim(self, ozellikler, hedef):
        self.siniflar = np.unique(hedef)
        self.sinifSayisi = len(self.siniflar)
        self.ozellikOlasiliklari = {}  # {sinif: {ozellik: {ozellik_degeri: olasilik}}}
        self.sinifOnceligi = {}  # P(sinif)
        
        toplamOrnek = len(hedef)
        for sinif in self.siniflar:
            veriSinif = ozellikler[hedef == sinif]
            self.sinifOnceligi[sinif] = (len(veriSinif) + self.duzeltme) / (toplamOrnek + self.sinifSayisi * self.duzeltme)
            self.ozellikOlasiliklari[sinif] = {}
            
            for sutun in ozellikler.columns:
                degerler = ozellikler[sutun].unique()
                sayim = veriSinif[sutun].value_counts().to_dict()
                self.ozellikOlasiliklari[sinif][sutun] = {}
                toplamSayim = len(veriSinif[sutun])
                for deger in degerler:
                    sayi = sayim.get(deger, 0)
                    self.ozellikOlasiliklari[sinif][sutun][deger] = (sayi + self.duzeltme) / (toplamSayim + len(degerler) * self.duzeltme)
                    
    def tahmin(self, ozellikler):
        tahminler = []
        for index, satir in ozellikler.iterrows():
            sinifPuanlari = {}
            for sinif in self.siniflar:
                puan = np.log(self.sinifOnceligi[sinif])
                for sutun in ozellikler.columns:
                    olasilik = self.ozellikOlasiliklari[sinif][sutun].get(satir[sutun], self.duzeltme / (sum(self.ozellikOlasiliklari[sinif][sutun].values()) + self.duzeltme))
                    puan += np.log(olasilik)
                sinifPuanlari[sinif] = puan
            tahminler.append(max(sinifPuanlari, key=sinifPuanlari.get))
        return np.array(tahminler)

# Veri setini oku
veriSeti = pd.read_csv("E:\Downloads\car_evaluation.csv")

# Veri setine genel bakış
print("Veri seti şekli:", veriSeti.shape)
print("İlk 5 örnek:\n", veriSeti.head())
print("Eksik veri kontrolü:\n", veriSeti.isnull().sum())

# Tüm sütunların kategorik olduğunu varsayarak label encoding uygulayalım
etiketDonusturucu = LabelEncoder()
for sutun in veriSeti.columns:
    veriSeti[sutun] = etiketDonusturucu.fit_transform(veriSeti[sutun])

# Özellikler ve hedef ayrımı: son sütun hedef olarak kabul ediliyor.
ozellikler = veriSeti.iloc[:, :-1]
hedef = veriSeti.iloc[:, -1]

# Eğitim ve test setlerine böl
ozelliklerEgitim, ozelliklerTest, hedefEgitim, hedefTest = train_test_split(ozellikler, hedef, test_size=0.3, random_state=42)

# Custom Naive Bayes modelini oluştur ve eğit
ozel_nb = OzelNaiveBayes(duzeltme=1)

baslangic = time.time()
ozel_nb.egitim(ozelliklerEgitim, hedefEgitim)
ozelEgitimSuresi = time.time() - baslangic

baslangic = time.time()
tahminSonuclariOzel = ozel_nb.tahmin(ozelliklerTest)
ozelTahminSuresi = time.time() - baslangic

# Performans değerlendirmesi
dogrulukOzel = accuracy_score(hedefTest, tahminSonuclariOzel)
karmaMatrixOzel = confusion_matrix(hedefTest, tahminSonuclariOzel)

print("Custom Naive Bayes Doğruluk:", dogrulukOzel)
print("Custom model - Eğitim süresi: {:.6f} sn, Tahmin süresi: {:.6f} sn".format(ozelEgitimSuresi, ozelTahminSuresi))

# Karmaşıklık matrisini görselleştir
gostergeOzel = ConfusionMatrixDisplay(confusion_matrix=karmaMatrixOzel)
gostergeOzel.plot(cmap=plt.cm.Oranges)
plt.title("Custom Naive Bayes - Karmaşıklık Matrisi")
plt.show()
