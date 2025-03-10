Bu projede, car_evaluation.csv veri seti kullanılarak iki farklı Naive Bayes sınıflandırma modelinin performansları karşılaştırılmıştır. Veri seti, 1727 örnek ve 7 özelliğe sahiptir.

Veri setinde herhangi bir eksik veri bulunmamaktadır

Model Performansları ve Sonuçlar
Projede iki model uygulanmıştır:

scikit-learn MultinomialNB:

Doğruluk (Accuracy): 0.6994
Eğitim Süresi: 0.010301 sn
Tahmin Süresi: 0.009728 sn
Custom Naive Bayes:

Doğruluk (Accuracy): 0.8343
Eğitim Süresi: 0.007322 sn
Tahmin Süresi: 0.051956 sn
Performans Ölçümünde Kullanılan Metodoloji
Karmaşıklık Matrisi (Confusion Matrix) Görselleştirmesi
Model performansını değerlendirirken, sadece doğruluk oranına bakmak yeterli olmayabilir. Bu nedenle, her iki model için de karmaşıklık matrisi oluşturularak:

True Positive (TP), False Positive (FP), True Negative (TN) ve False Negative (FN) değerleri belirlenmiştir.
Karmaşıklık matrisi, sınıflandırma hatalarının hangi sınıflarda yoğunlaştığını görselleştirmek ve modelin hangi durumlarda hatalı kararlar verdiğini anlamak için kullanılmaktadır.
Bu görselleştirme, özellikle dengesiz sınıf dağılımlarında modelin performansını daha detaylı analiz etmeye olanak sağlar.
Değerlendirme Metrikleri ve Seçiminde Problem & Sınıf Dağılımının Rolü
Proje kapsamında kullanılan temel değerlendirme metrikleri şunlardır:

Doğruluk (Accuracy): Genel olarak doğru sınıflandırma oranını ölçer.
Precision (Kesinlik): Pozitif olarak tahmin edilen örnekler içerisindeki gerçek pozitif oranını gösterir.
Recall (Duyarlılık): Gerçek pozitiflerin ne kadarının doğru tespit edildiğini ifade eder.
F1-Score: Precision ve Recall arasındaki dengeyi ölçen harmonik ortalamadır.
Metrik Seçiminin Önemi:

Problem Tanımı: Sınıflandırma probleminin doğası (örneğin, tıbbi teşhis, dolandırıcılık tespiti gibi) hangi metriklerin daha kritik olduğunu belirler. Örneğin, sağlık uygulamalarında yanlış negatiflerin (hasta olmasına rağmen sağlıklı tahmini) maliyeti çok yüksek olabilir.
Sınıf Dağılımı: Veri setinde sınıflar arasında dengesizlik varsa, sadece doğruluk oranı yanıltıcı olabilir. Dengesiz veri setlerinde, precision, recall ve F1-score gibi metriklerin kullanımı modelin performansını daha sağlıklı değerlendirmeye yardımcı olur.
Bu nedenle, değerlendirme metriklerinin seçimi, problemi doğru tanımlamak ve sınıf dağılımını göz önünde bulundurmak açısından kritik öneme sahiptir. Hem modelin genel başarısını hem de belirli sınıflara yönelik performansını değerlendirebilmek için birden fazla metriğin kullanılması önerilmektedir.

