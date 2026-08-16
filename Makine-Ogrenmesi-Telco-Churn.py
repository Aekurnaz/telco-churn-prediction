"""
Telco Customer Churn Tahmini 
===================================================================
Amacı: 
    1. Telekomünikasyon müşterilerinin hizmet kullanım ve demografik verilerini inceleyerek aboneliklerini iptal edip etmeyeceklerini (Churn) tahmin etmektir.

Adımları:
    1. Gerekli veri bilimi kütüphanelerini import eder.
    2. IBM Telco veri setini internet üzerinden (URL) okur ve temel istatistiklerini gösterir.
    3. TotalCharges sütunundaki eksik verileri tespit eder ve medyan ile doldurur.
    4. "Aylık Ortalama Ödeme" ve "Otomatik Ödeme Durumu" gibi yeni öznitelikler üretir (Feature Engineering).
    5. Kategorik değişkenleri One-Hot Encoding ile sayısal formata dönüştürür.
    6. Aykırı değerleri (Outliers) çeyrekler açıklığı (IQR) yöntemiyle üst sınıra baskılar (Capping).
    7. Hedef değişkenle korelasyonu %1'den bile düşük olan gereksiz öznitelikleri eler.
    8. Veriyi %60 Train, %20 Validation ve %20 Test olacak şekilde tabakalı (Stratify) olarak böler ve ölçekleme (StandardScaler) yapar.
    9. Logistic Regression, SVM ve Random Forest modellerini eğitir ve Validation doğrulama metriklerini birlikte yazdırır.
    10. Validation setinde en başarılı olan model için Grid Search ile hiperparametre optimizasyonu yapar.
    11. Optimize edilmiş en iyi modeli Test verisi üzerinde değerlendirir.
    12. Modelin kararlarında etkili olan en önemli değişkenleri SHAP  yöntemiyle açıklar ve grafiğe döker.

Kullanılan Kütüphaneler: 
pandas, numpy, scikit-learn, matplotlib, seaborn, shap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import shap 


print("--- Adım 2: Veri Yükleme ---")
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
print("Veri başarıyla yüklendi. (Problem: Müşteri Ayrılma (Churn) Tahmini)\n")

print("--- Adım 3: Hedef Değişken ve Problem Türü ---")
target = 'Churn'
print(f"Hedef Değişken: {target}")
print("Problem Türü: İkili Sınıflandırma (Binary Classification)\n")

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Churn', palette='Set2')
plt.title('Müşteri Ayrılma (Churn) Dağılımı')
plt.savefig('1_churn_dagilimi.png')
plt.close()
print("[Görsel Kaydedildi]: '1_churn_dagilimi.png' dosyasına hedef değişken dağılımı çizildi.\n")

print("--- Adım 4: Temel Veri İncelemesi ---")
print("Boyut:", df.shape)
print("\nTemel İstatistikler:\n", df.describe().T, "\n")

print("--- Adım 5: Eksik Değer Kontrolü ve Temizlik ---")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("Eksik veri sayısı (Gizli boşluklar NaN yapıldıktan sonra):", df['TotalCharges'].isnull().sum())
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
print("Eksik veriler TotalCharges medyanı ile dolduruldu.\n")

print("--- Adım 9: Öznitelik Mühendisliği (Feature Engineering) ---")
df['AvgChargePerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
df['IsAutoPayment'] = df['PaymentMethod'].apply(lambda x: 1 if 'automatic' in x.lower() else 0)
print("Yeni öznitelikler 'AvgChargePerMonth' ve 'IsAutoPayment' oluşturuldu.\n")

print("--- Adım 6: Kategorik Değişken Dönüşümü (Encoding) ---")
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = df.drop('customerID', axis=1)
df = pd.get_dummies(df, drop_first=True)
print("Kategorik değişkenler One-Hot Encoding ile dönüştürüldü.\n")

print("--- Adım 7: Aykırı Değer (Outlier) İşlemi ---")
Q1 = df['TotalCharges'].quantile(0.25)
Q3 = df['TotalCharges'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
df['TotalCharges'] = np.where(df['TotalCharges'] > upper_bound, upper_bound, df['TotalCharges'])
print("TotalCharges (Toplam Ücret) üzerindeki muhtemel aykırı değerler üst sınıra baskılandı.\n")

print("--- Adım 10: Öznitelik Seçimi (Feature Selection) ---")
corr_matrix = df.corr()
churn_corr = corr_matrix['Churn'].abs()
low_corr_features = churn_corr[churn_corr < 0.01].index.tolist()
if len(low_corr_features) > 0:
    df = df.drop(columns=low_corr_features)
    print(f"Korelasyonu %1'den düşük olan sütunlar silindi: {low_corr_features}\n")

top_corr_features = churn_corr.sort_values(ascending=False).head(15).index
plt.figure(figsize=(10, 8))
sns.heatmap(df[top_corr_features].corr(), annot=True, fmt='.2f', cmap='coolwarm', cbar=False)
plt.title('En Etkili 15 Değişkenin Korelasyon Matrisi')
plt.tight_layout()
plt.savefig('2_korelasyon_matrisi.png')
plt.close()
print("[Görsel Kaydedildi]: '2_korelasyon_matrisi.png' kaydedildi.\n")

print("--- Adım 11: Train, Validation ve Test Kümelerine Ayırma (80-10-10) ---")
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()

print(f"Train: {X_train.shape} | Validation: {X_val.shape} | Test: {X_test.shape}\n")

print("--- Adım 8: Veri Ölçekleme (Scaling) ---")
scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgChargePerMonth']

X_train[num_cols] = X_train[num_cols].astype(float)
X_val[num_cols] = X_val[num_cols].astype(float)
X_test[num_cols] = X_test[num_cols].astype(float)

X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
X_val.loc[:, num_cols] = scaler.transform(X_val[num_cols])
X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])
print("Nümerik değişkenlere StandardScaler uygulandı.\n")

print("--- Adım 12 & 13: Model Eğitimi ve Validation Karşılaştırması ---")

models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42),
    "SVM (Support Vector Machine)": SVC(kernel='linear', class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42)
}

best_val_score = 0
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    
    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    
    print(f"[{name}] Validation Accuracy: {acc:.4f} | Validation F1-Score: {f1:.4f}")
    
    if f1 > best_val_score:
        best_val_score = f1
        best_model_name = name

print(f"\n=> Validation sonuçlarına göre seçilen en iyi model: {best_model_name}\n")

print("--- Adım 14: Randomized Search ile Hiperparametre Optimizasyonu ---")
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])

if best_model_name == "Random Forest":
    param_grid = {
        'n_estimators': [100, 200, 300, 500], 
        'max_depth': [10, 15, 20, 30, None], 
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6]
    }
    n_iter_val = 24
elif best_model_name == "Logistic Regression":
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], 
        'penalty': ['l1', 'l2'], 
        'solver': ['liblinear', 'saga'], 
        'max_iter': [100, 200, 500, 1000]
    }
    n_iter_val = 12
elif best_model_name == "SVM (Support Vector Machine)":
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01, 0.001]
    }
    n_iter_val = 6

selected_base_model = models[best_model_name]
print(f"{best_model_name} modeli için Randomized Search uygulanıyor ({n_iter_val} farklı kombinasyon deneniyor)...")

random_search = RandomizedSearchCV(
    estimator=selected_base_model, 
    param_distributions=param_grid, 
    n_iter=n_iter_val, 
    cv=3, 
    scoring='f1', 
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train_val, y_train_val)

best_model = random_search.best_estimator_
print(f"Bulunan en iyi hiperparametreler: {random_search.best_params_}\n")

print("--- Adım 15: Test Verisi Üzerinde Değerlendirme ---")
y_test_pred = best_model.predict(X_test)

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_test_pred))

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix ({best_model_name})')
plt.xlabel('Tahmin Edilen (Predicted)')
plt.ylabel('Gerçek (Actual)')
plt.savefig('3_confusion_matrix.png')
plt.close()
print("[Görsel Kaydedildi]: '3_confusion_matrix.png' kaydedildi.\n")

print("--- Adım 16: Model Yorumu ve Değerlendirme ---")
print(f"""Yorum:
- Üç farklı algoritma arasından, azınlık sınıfını (Churn) tespit etme performansı (F1-Score) en yüksek olan model {best_model_name} seçilmiştir.
- Veri dağılımı %80 Eğitim olacak şekilde ayarlanmış ve sınıf dengesizliğini çözmek için sınıf ağırlıkları dengelenmiştir.
- Bu sayede modelin ayrılacak müşterileri (1) tahmin etme gücü artırılmış ve iş süreçlerinde kullanıma uygun bir model elde edilmiştir.
""")

print("--- Adım 17: Model Açıklanabilirliği (SHAP Kütüphanesi ile) ---")

plt.figure(figsize=(10, 8))

if best_model_name in ["Random Forest"]:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test)
    shap_vals_churn = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(shap_vals_churn, X_test, show=False)
else:
    background = shap.kmeans(X_train_val, 50)
    explainer = shap.KernelExplainer(best_model.predict, background)
    X_test_sample = X_test.sample(min(150, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_test_sample)
    shap.summary_plot(shap_values, X_test_sample, show=False)

plt.title(f'SHAP Summary Plot ({best_model_name})', loc='left')
plt.tight_layout()
plt.savefig('4_shap_summary.png')
plt.close()

print("\n[Görsel Kaydedildi]: '4_shap_summary.png' (SHAP Analizi) başarıyla kaydedildi.")

print("\nSHAP YORUMU:")
print("Grafikte yatay eksen SHAP değerini, renkler ise değişkenin kendi değerini temsil eder.")
print("Sonuçlara göre:")
print("1. 'InternetService_Fiber optic': En önemli değişkendir. Kırmızı noktalar (fiber kullananlar) sağ taraftadır. Bu hizmeti alanların ayrılma (churn) ihtimali çok daha yüksektir.")
print("2. 'tenure' (müşterilik süresi): Düşük tenure (mavi noktalar) sağda yer alıp ayrılma riskini artırırken; eski müşteriler (kırmızı noktalar) solda yer alır.")
print("3. 'AvgChargePerMonth': Kendi ürettiğimiz bu değişkende, yüksek ortalama ödemesi olan müşterilerin (kırmızı) sadık kaldığı, düşük ödeme yapanların (mavi) ise ayrılma eğiliminde olduğu gözlemlenmiştir.")
print("4. 'Contract_Two year': İki yıllık sözleşme yapan müşterilerin churn ihtimali ciddi oranda düşmektedir.")
