import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('default')
# Görselleştirme ayarları
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

data = pd.read_csv('50_Startups.csv')
print(data.head())

print("GÖREV 1 : R&D Harcaması ve Kâr Arasındaki İlişki")
plt.figure(figsize=(10, 6))
plt.scatter(data['R&D Spend'], data['Profit'], alpha=0.7, color='blue', s=50)
plt.xlabel('R&D Harcaması ($)', fontsize=12)
plt.ylabel('Kâr ($)', fontsize=12)
plt.title('R&D Harcaması vs Kâr İlişkisi', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# Korelasyon ( ilişki) hesaplama
correlation_rd = data['R&D Spend'].corr(data['Profit'])
print(f"R&D Harcaması ve Kâr arasındaki korelasyon: {correlation_rd:.4f}")


print("GÖREV 2 : Yönetim Harcamaları ve Kâr Arasındaki İlişki")
plt.figure(figsize=(10, 6))
plt.scatter(data['Administration'], data['Profit'], alpha=0.7, color='green', s=50)
plt.xlabel('Yönetim Harcaması ($)', fontsize=12)
plt.ylabel('Kâr ($)', fontsize=12)
plt.title('Yönetim Harcaması vs Kâr İlişkisi', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Korelasyon hesapla
correlation_admin = data['Administration'].corr(data['Profit'])
print(f"Yönetim Harcaması ve Kâr arasındaki korelasyon: {correlation_admin:.4f}")


print("GÖREV 3: Eyaletlere Göre Ortalama Kar")
ortalama = data.groupby('State')['Profit'].agg(['mean']).round(2)
print("Eyaletlere göre ortalama kar")
print(ortalama)
plt.figure(figsize=(10, 6))
bars = plt.bar(ortalama.index, ortalama['mean'],
               color=['skyblue', 'lightcoral', 'lightgreen'])
plt.xlabel('Eyalet', fontsize=12)
plt.ylabel('Ortalama Kâr ($)', fontsize=12)
plt.title('Eyaletlere Göre Ortalama Kâr', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
for i, ort in enumerate(ortalama['mean']):
    plt.text(i, ort, f'{ort:,.0f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()


print("GÖREV 4 : Harcama Türlerinin Karşılaştırması")
spending_data = pd.DataFrame({
    'R&D Harcaması': data['R&D Spend'],
    'Yönetim Harcaması': data['Administration'],
    'Pazarlama Harcaması': data['Marketing Spend'] })
plt.figure(figsize=(12, 6))
spending_data.boxplot()
plt.title('Harcama Türlerinin Dağılım Karşılaştırması', fontsize=14, fontweight='bold')
plt.ylabel('Harcama Miktarı ($)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()










