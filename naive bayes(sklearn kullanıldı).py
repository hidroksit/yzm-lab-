import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

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

# Özellikler (ozellikler) ve hedef (hedef) ayrımı: son sütunun hedef olduğunu varsayıyoruz.
ozellikler = veriSeti.iloc[:, :-1]
hedef = veriSeti.iloc[:, -1]

# Eğitim ve test setlerine böl
ozelliklerEgitim, ozelliklerTest, hedefEgitim, hedefTest = train_test_split(ozellikler, hedef, test_size=0.3, random_state=42)

# Modeli oluştur
nb_model = MultinomialNB()

# Eğitim süresini ölç
baslangic = time.time()
nb_model.fit(ozelliklerEgitim, hedefEgitim)
egitimSuresi = time.time() - baslangic

# Tahmin süresini ölç
baslangic = time.time()
tahminSonuclari = nb_model.predict(ozelliklerTest)
tahminSuresi = time.time() - baslangic

# Performans değerlendirmesi
dogruluk = accuracy_score(hedefTest, tahminSonuclari)
karmaMatrix = confusion_matrix(hedefTest, tahminSonuclari)

print("scikit-learn MultinomialNB Doğruluk:", dogruluk)
print("Eğitim süresi: {:.6f} sn, Tahmin süresi: {:.6f} sn".format(egitimSuresi, tahminSuresi))

# Karmaşıklık matrisini görselleştir
gosterge = ConfusionMatrixDisplay(confusion_matrix=karmaMatrix)
gosterge.plot(cmap=plt.cm.Blues)
plt.title("scikit-learn MultinomialNB - Karmaşıklık Matrisi")
plt.show()
