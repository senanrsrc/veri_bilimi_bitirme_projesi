import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv('dava.csv')
print(data.head())
print(f"Veri seti boyutu: {data.shape}")


print(f"\nEksik değer kontrolü:")
print(data.isnull().sum())


print("GÖREV 1: KÜMELEME İÇİN UYGUN ÖZELLİK SEÇME")
sayisal_ozellikler = ['Case Duration (Days)', 'Number of Witnesses',
                      'Legal Fees (USD)', 'Number of Evidence Items']
print(f"Kümeleme için seçilen özellikler: {sayisal_ozellikler}")
kumeleme = data[sayisal_ozellikler].copy()
print(f"Kümeleme veri seti boyutu: {kumeleme.shape}")
skaler = StandardScaler()
kumeleme_standart = skaler.fit_transform(kumeleme)
print("Veriler StandardScaler ile ölçeklendirildi.")


print("GÖREV 2: ELBOW METHODU İLE OPTİMAL KÜME SAYISI BULMA")
kume_range = range(2, 11)  # minimum anlamlı küme sayısı 2'dir.
wcss = []
for k in kume_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(kumeleme_standart)
    wcss.append(kmeans.inertia_)
plt.plot(kume_range, wcss, 'bo-')
wcss_fark= np.diff(wcss)
elbow_k = kume_range[np.argmin(wcss_fark[1:]) + 2]
wcss_df = pd.DataFrame({
    'K': kume_range,
    'WCSS': wcss })
print(wcss_df)
print(f"Elbow method önerisi: k={elbow_k}")


print("GÖREV 3: KMEANS İLE VERİYİ KÜMELEME")
optimal_k = elbow_k  # Silhouette'e göre en iyi k
print(f"Seçilen küme sayısı: {optimal_k}")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_final.fit(kumeleme_standart)
kume_kmeans = kmeans_final.predict(kumeleme_standart)
data_kumeli = data.copy()
data_kumeli['Kümelenme'] = kume_kmeans
print(data_kumeli.head())
kume_sayisi = pd.Series(kume_kmeans).value_counts().sort_index()
print(kume_sayisi)


print("GÖREV 4: GRAFİKLENDİRME")
plt.figure(figsize=(7, 5))
plt.plot(range(2, 11), wcss, marker='o', color='royalblue')
plt.title("Elbow Yöntemi - En Uygun Küme Sayısı")
plt.xlabel("Küme Sayısı (k)")
plt.ylabel("Toplam Hata (Inertia)")
plt.grid(True, alpha=0.3)
plt.show()


plt.figure(figsize=(7, 5))
colors = ['red', 'green', 'blue', 'purple', 'orange']
for i in range(optimal_k):
    mask = (data_kumeli['Kümelenme'] == i)
    plt.scatter(data_kumeli.loc[mask, 'Case Duration (Days)'],
                data_kumeli.loc[mask, 'Legal Fees (USD)'],
                color=colors[i],
                label=f'Küme {i}',
                alpha=0.7,
                s=80,
                edgecolors='black')
merkezler = skaler.inverse_transform(kmeans_final.cluster_centers_)
plt.scatter(merkezler[:, 0], merkezler[:, 2],  # 0: süre, 2: masraf (Legal Fees)
            color='yellow', marker='X', s=200, edgecolors='black', label='Merkezler')
plt.title("K-Means Kümeleme Sonuçları", fontsize=12)
plt.xlabel("Dava Süresi (Days)")
plt.ylabel("Hukuk Masrafları (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Değişkenler Arası Korelasyon Matrisi")
plt.show()


kume_sayisi = pd.Series(kume_kmeans).value_counts().sort_index()
plt.figure(figsize=(6, 4))
plt.bar(kume_sayisi.index, kume_sayisi.values, color=colors[:optimal_k])
plt.title("Küme Boyutları", fontsize=12)
plt.xlabel("Küme")
plt.ylabel("Eleman Sayısı")
for i, count in enumerate(kume_sayisi.values):
    plt.text(i, count + 0.2, str(count), ha='center')
plt.show()














