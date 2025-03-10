Bu projede, araç değerlendirme veri seti kullanılarak, araçların sınıflandırılması (örneğin "unacc" gibi sınıflar) hedeflenmiştir. Amaç, farklı Naive Bayes yaklaşımlarını (scikit-learn ve custom model) karşılaştırarak en iyi performansı sağlayanı belirlemektir.

Eksik Veri: Hiçbir eksik veri bulunmamaktadır.

Yöntem
Modeller:
scikit-learn MultinomialNB modeli
Custom Naive Bayes modeli
Değerlendirme Metrikleri:
Doğruluk (Accuracy)
Karmaşıklık Matrisi (Confusion Matrix)
Precision, Recall, F1-Score (ek metrikler, problem ve sınıf dağılımı göz önünde bulundurularak)
Not: Jupyter Notebook kullanılarak çalışma gerçekleştirilmiştir. Dolayısıyla, kod dosyaları .py uzantılı olmayıp, Notebook formatında sunulmuştur.
Sonuçlar
scikit-learn MultinomialNB:
Doğruluk: %69.94
Eğitim ve tahmin süreleri de ölçülmüştür.
Custom Naive Bayes:
Doğruluk: %83.43
Eğitim ve tahmin süreleri farklılık göstermiştir.
Karmaşıklık matrisi ile modellerin hata dağılımları görselleştirilmiş, dengesiz sınıf dağılımlarının modele etkisi detaylandırılmıştır.
Yorum / Tartışma
Model Karşılaştırması: Custom Naive Bayes modeli, doğruluk açısından daha başarılı bulunmuştur.
Metrik Seçimi: Sınıf dağılımının dengesiz olduğu durumlarda, yalnızca doğruluk metriği yeterli olmamaktadır. Precision, Recall ve F1-Score gibi ek metrikler, modelin belirli sınıflar üzerindeki performansını daha doğru analiz etmemizi sağlamıştır.
Uygulama Önerisi: Jupyter Notebook kullanımı, veri ön işleme, model eğitimi ve sonuç görselleştirmeleri açısından etkileşimli bir çalışma ortamı sunmaktadır.
