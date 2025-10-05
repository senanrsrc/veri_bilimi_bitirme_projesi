import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('dava_sonuclari.csv')

print("GÖREV 1 : VERİ HAZIRLIĞI VE KONTROL")
# Eksik değer kontrolü
print("Eksik değer kontrolü:")
eksik = data.isnull().sum()
if eksik.sum() == 0:
    print("Eksik değer yok")
else:
    print(eksik[eksik > 0])

# Outcome kontrolü
print("Outcome (Hedef Değişken) Analizi:")
outcome_dagilim = data['Outcome'].value_counts()
print(outcome_dagilim)
print(f"Benzersiz değerler: {data['Outcome'].unique()}")
if len(data['Outcome'].unique()) == 1:
    print("Outcome sütununda sadece TEK SINIF (0 - Kaybedilen) var!")

# Case Type encoding
le = LabelEncoder()
if 'Case Type' in data.columns:
    data['Case Type Encoded'] = le.fit_transform(data['Case Type'])
    print("Case Type Sayısal Değişkene Dönüşümü:")
    for i, case_type in enumerate(le.classes_):
        print(f"{case_type}:{i}")

# Özellik seçimi
ozellik_sutunlari = [col for col in data.columns if col not in ['Outcome', 'Case Type']]
if 'Case Type Encoded' in data.columns:
    ozellik_sutunlari.append('Case Type Encoded')
X = data[ozellik_sutunlari]
y_orijinal = data['Outcome']
print(f"Kullanılan özellikler ({len(ozellik_sutunlari)} adet):")
for i, col in enumerate(ozellik_sutunlari, 1):
    print(f"  {i}. {col}")


print("SOLATION FOREST İLE FARKLI DAVALARIN TESPİTİ")
# Model parametreleri
anomali_orani = 0.15  # Verinin %15'ini "farklı" olarak işaretle
iso_forest = IsolationForest(contamination=anomali_orani,random_state=42,n_estimators=100)

# Anomali skorları hesaplama
anomali_etiket = iso_forest.fit_predict(X)
anomali_skor = iso_forest.score_samples(X)

y_sentetik = np.where(anomali_etiket == -1, 1, 0)
data['Potential_Win'] = y_sentetik
data['Anomaly_Score'] = anomali_skor
print(f"Tespit Sonuçları:")
print(f"Normal davalar (0): {(y_sentetik == 0).sum()} adet (%{(y_sentetik == 0).sum() / len(y_sentetik) * 100:.1f})")
print(f"Potansiyel kazanılabilir (1): {(y_sentetik == 1).sum()} adet (%{(y_sentetik == 1).sum() / len(y_sentetik) * 100:.1f})")
print("Yorum: Isolation Forest, 'kaybedilen davalar' kalıbından sapan davaları 'potansiyel kazanılabilir' olarak işaretledi.")


print("️GÖREV 2 : DECISION TREE İLE KAZANMA KOŞULLARININ ANALİZİ")

# Train-test ayrımı (sentetik hedef ile)
X_train, X_test, y_train, y_test = train_test_split(X, y_sentetik, test_size=0.2, random_state=42, stratify=y_sentetik)
print(f"Veri Bölünmesi:")
print(f"Eğitim seti: {X_train.shape[0]} örnek")
print(f"Test seti: {X_test.shape[0]} örnek")


print("️GÖREV 3 : DECISION TREE İLE KAZANMA KOŞULLARININ ANALİZİ")

# Decision Tree modeli
dt_model = DecisionTreeClassifier(max_depth=5,min_samples_split=10,min_samples_leaf=5,random_state=42)
dt_model.fit(X_train, y_train)
# Tahminler
y_train_tahmin = dt_model.predict(X_train)
y_test_tahmin = dt_model.predict(X_test)
print("Model eğitildi!")


print("GÖREV 4 : MODEL PERFORMANSI")

train_acc = accuracy_score(y_train, y_train_tahmin)
test_acc = accuracy_score(y_test, y_test_tahmin)
precision = precision_score(y_test, y_test_tahmin)
recall = recall_score(y_test, y_test_tahmin)
f1 = f1_score(y_test, y_test_tahmin)
print("Metrikler:")
print(f"Eğitim Doğruluğu: {train_acc:.2%}")
print(f"Test Doğruluğu: {test_acc:.2%}")
print(f"Precision (Kesinlik): {precision:.2%}")
print(f"Recall (Duyarlılık): {recall:.2%}")
print(f"F1-Score: {f1:.2%}")
print(f"Detaylı Rapor:")
print(classification_report(y_test, y_test_tahmin,target_names=['Kaybedilebilir', 'Kazanılabilir']))

# Karmaşıklık Matrisi
cm = confusion_matrix(y_test, y_test_tahmin)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=['Kaybedilebilir', 'Kazanılabilir'],
            yticklabels=['Kaybedilebilir', 'Kazanılabilir'])
plt.title('Confusion Matrix (Karmaşıklık Matrisi)', fontsize=14, weight='bold')
plt.ylabel('Gerçek Değer')
plt.xlabel('Tahmin')
plt.tight_layout()
plt.show()


print("GÖREV 5:HANGİ ÖZELLİKLER KAZANMA POTANSİYELİNİ BELİRLİYOR?")

# Özellik önemi
ozellik_onemi= pd.DataFrame({'Özellik': X.columns,'Önem': dt_model.feature_importances_}).sort_values('Önem', ascending=False)
print("Özellik Önem Sıralaması:")
for idx, row in ozellik_onemi.iterrows():
    if row['Önem'] > 0:
        print(f"{row['Özellik']:.<40} {row['Önem']:.5f}")

# Görselleştirme
plt.figure(figsize=(10, 6))
top_ozellikler= ozellik_onemi.head(10)
sns.barplot(data=top_ozellikler, y='Özellik', x='Önem', palette='viridis')
plt.title('En Önemli 10 Özellik (Kazanma Potansiyelini Etkileyen Faktörler)',fontsize=13, weight='bold')
plt.xlabel('Önem Derecesi', fontsize=11)
plt.ylabel('Özellik', fontsize=11)
plt.tight_layout()
plt.show()

# Karar Ağacı Yapısı
plt.figure(figsize=(20, 12))
plot_tree(dt_model, max_depth=3,feature_names=X.columns,class_names=['Kaybedilebilir', 'Kazanılabilir'],
          filled=True, rounded=True, fontsize=10,proportion=True)
plt.title('Karar Ağacı: Dava Kazanma Potansiyeli Karar Kuralları',fontsize=16, weight='bold')
plt.tight_layout()
plt.show()

# Metin formatında kurallar
agac_kurallari = export_text(dt_model, feature_names=list(X.columns), max_depth=4)
print("Karar Kuralları (İlk 4 Seviye):")
print(agac_kurallari[:2500] + "..." if len(agac_kurallari) > 2500 else agac_kurallari)
print("ANALİZ TAMAMLANDI")


