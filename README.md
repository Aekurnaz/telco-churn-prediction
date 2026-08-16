# Telco Customer Churn Tahmini

**Türkiye Yapay Zeka Akademisi — Makine Öğrenmesi Final Ödevi**

## Projenin Amacı

Bu proje, telekomünikasyon müşterilerinin hizmet kullanım ve demografik verilerini inceleyerek aboneliklerini iptal edip etmeyeceklerini (**Churn**) tahmin eder. Amaç; veri inceleme, veri ön işleme, öznitelik mühendisliği, model eğitimi, model karşılaştırma, hiperparametre ayarlama ve sonuç yorumlama adımlarını uçtan uca tek bir Python dosyasında uygulamaktır.

- **Problem türü:** İkili Sınıflandırma (Binary Classification)
- **Hedef değişken:** `Churn` (`Yes` → 1, `No` → 0)

## Veri Seti

| Özellik | Değer |
|---|---|
| Kaynak | [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) |
| Satır sayısı | 7.043 müşteri |
| Sütun sayısı | 21 (ham veri) |
| Hedef | Müşteri ayrıldı mı? (`Churn`) |

Veri seti; demografik bilgiler (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), hizmet kullanımı (`InternetService`, `PhoneService`, streaming servisleri), sözleşme tipi, ödeme yöntemi ve fatura tutarlarını (`tenure`, `MonthlyCharges`, `TotalCharges`) içerir.


Hedef değişkende **sınıf dengesizliği** vardır: kalan müşteriler (~5.170), ayrılanlardan (~1.870) yaklaşık 3 kat fazladır. Bu nedenle veri bölmede `stratify=y`, modellerde `class_weight='balanced'` kullanılmış ve model karşılaştırma metriği olarak **F1-Score** seçilmiştir.

![Müşteri Ayrılma (Churn) Dağılımı]

<img width="744" height="498" alt="image" src="https://github.com/user-attachments/assets/b005e5f9-3ebc-4fad-9096-a23400d3bf70" />


## Nasıl Çalıştırılır

```bash
# 1. Bağımlılıkları kurun
pip install -r requirements.txt

# 2. Analizi çalıştırın
python Makine-Ogrenmesi-Telco-Churn.py
```


## Uygulanan Adımlar (Ödev Soruları ile Eşleşme)

1. **Docstring:** Dosya başında amaç, adımlar ve kütüphaneleri açıklayan docstring yer alır.
2. **Veri okuma:** IBM Telco verisi pandas ile URL'den okunur; problem müşteri ayrılma tahminidir.
3. **Hedef değişken:** `Churn` 
4. **Temel inceleme:** Boyut (7043 × 21) ve `describe()` ile temel istatistikler yazdırılır.
5. **Eksik değer:** `TotalCharges` içindeki 11 gizli boşluk `NaN` yapılır ve **medyan** ile doldurulur.
6. **Encoding:** Kategorik değişkenler **One-Hot Encoding** (`get_dummies`, `drop_first=True`) ile dönüştürülür; `customerID` silinir.
7. **Aykırı değer:** `TotalCharges` aykırı değerleri **IQR** yöntemiyle üst sınıra baskılanır (capping).
8. **Ölçekleme:** Sayısal değişkenlere (`tenure`, `MonthlyCharges`, `TotalCharges`, `AvgChargePerMonth`) **StandardScaler** uygulanır (scaler sadece train üzerinde fit edilir).
9. **Öznitelik mühendisliği (2 adet):**
   - `AvgChargePerMonth` = `TotalCharges / (tenure + 1)` — aylık ortalama ödeme
   - `IsAutoPayment` — ödeme yöntemi otomatik mi? (1/0)
10. **Öznitelik seçimi:** Hedefle mutlak korelasyonu %1'den düşük sütunlar elenir (bu veri setinde `gender_Male` silindi).
11. **Veri bölme:** Tabakalı (`stratify`) bölme ile **Train: %80 / Validation: %10 / Test: %10**.
12. **3 model eğitimi:** Logistic Regression, SVM ve Random Forest.
13. **Validation karşılaştırması:** Accuracy ve F1-Score birlikte yazdırılır.
14. **Hiperparametre ayarı:** Seçilen model için **RandomizedSearchCV**.
15. **Test değerlendirmesi:** Confusion matrix + classification report (accuracy, precision, recall, F1).
16. **Model yorumu:** Aşağıdaki "Sonuç Yorumu" bölümünde.
17. **Açıklanabilirlik:** SHAP (KernelExplainer) ile en etkili değişkenler görselleştirilir.

## Model Karşılaştırması (Validation Sonuçları)

| Model | Validation Accuracy | Validation F1-Score |
|---|---|---|
| **Logistic Regression** | 0.7244 | **0.6025** |
| Random Forest | 0.7528 | 0.5797 |
| SVM (linear) | 0.6761 | 0.5714 |

Azınlık sınıfını (churn) yakalama başarısını ölçen **F1-Score'a göre Logistic Regression** en iyi model olarak seçilmiştir.

### Hiperparametre Optimizasyonu

Logistic Regression için Randomized Search (12 kombinasyon, 3-fold CV) ile bulunan en iyi parametreler:

```text
{'solver': 'saga', 'penalty': 'l1', 'max_iter': 1000, 'C': 10.0}
```

## Test Performansı

Optimize edilen model, ayrı tutulan test seti (705 kayıt) üzerinde değerlendirilmiştir.

![Confusion Matrix]

<img width="614" height="507" alt="image" src="https://github.com/user-attachments/assets/93c445ab-67ca-412f-bf4c-68acaadfdee1" />


| | Tahmin: Kalır (0) | Tahmin: Ayrılır (1) |
|---|---|---|
| **Gerçek: Kalır (0)** | 391 | 127 |
| **Gerçek: Ayrılır (1)** | 37 | 150 |

| Sınıf | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (Kalır) | 0.91 | 0.75 | 0.83 | 518 |
| 1 (Ayrılır) | 0.54 | 0.80 | 0.65 | 187 |
| **Accuracy** | | | **0.77** | 705 |
| Macro avg | 0.73 | 0.78 | 0.74 | 705 |

## Korelasyon Analizi

![En Etkili 15 Değişkenin Korelasyon Matrisi]

<img width="638" height="514" alt="image" src="https://github.com/user-attachments/assets/d05a1d9b-828b-47d8-b172-a25b28eb7965" />


- `tenure` churn ile en güçlü negatif ilişkiye sahiptir (**-0.35**): müşterilik süresi arttıkça ayrılma azalır.
- `InternetService_Fiber optic` (**+0.31**) ve `PaymentMethod_Electronic check` (**+0.30**) churn riskini artırır.
- `Contract_Two year` (**-0.30**): uzun sözleşme sadakati güçlendirir.

## SHAP ile Açıklanabilirlik

![SHAP Summary Plot]

<img width="430" height="507" alt="image" src="https://github.com/user-attachments/assets/86b5c6e9-7934-4569-97a4-42ecf61d4fd8" />


Modele en çok katkı yapan değişkenler:

1. **`InternetService_Fiber optic`:** En önemli değişkendir. Fiber kullanan müşterilerin (kırmızı noktalar sağda) ayrılma olasılığı belirgin biçimde yüksektir.
2. **`tenure`:** Yeni müşteriler (düşük tenure, mavi noktalar sağda) daha çok churn eder; uzun süreli müşteriler sadıktır.
3. **`AvgChargePerMonth`:** Ürettiğimiz bu öznitelikte yüksek ortalama ödeme yapanlar daha sadık, düşük ödeme yapanlar ayrılma eğilimindedir.
4. **`Contract_Two year`:** İki yıllık sözleşme churn ihtimalini ciddi oranda düşürür.

## Sonuç Yorumu

**Hangi model daha iyi oldu?** Validation F1 skoruna göre en iyi model **Logistic Regression** olmuştur (F1 = 0.6025). Random Forest accuracy'de daha yüksek kalsa da (0.7528) churn sınıfındaki F1'i daha düşüktür (0.5797); SVM her iki metrikte de geridedir. Dengesiz veri setlerinde accuracy yanıltıcı olabildiğinden model seçimi F1'e göre yapılmıştır.

**Test sonuçları yorumu** Test setinde accuracy **%77**'dir. Asıl önemli olan churn sınıfında model **%80 recall** üretir: gerçekten ayrılan 187 müşterinin 150'sini doğru yakalar, yalnızca 37'sini kaçırır. Precision'ın 0.54 olması, 127 sadık müşterinin de "riskli" olarak işaretlendiğini gösterir. `class_weight='balanced'` bu dengeyi bilinçli olarak recall lehine kaydırmıştır; müşteri kaybının maliyeti kampanya maliyetinden yüksek olduğu için iş açısından tercih edilen senaryodur.

**Hangi değişkenler önemli?** Hem korelasyon hem SHAP analizi tutarlı biçimde aynı değişkenleri işaret eder: fiber internet kullanımı ve elektronik çek ile ödeme churn riskini artırır; uzun müşterilik süresi, iki yıllık sözleşme ve otomatik ödeme sadakati güçlendirir.


## Kullanılan Kütüphaneler

`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`
