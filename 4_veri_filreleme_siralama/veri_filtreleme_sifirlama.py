import pandas as pd


data = pd.read_csv('country.csv')
print(data.head())
print(f"Veri seti boyutu: {data.shape}")


print("1. GÖREV: Nüfusa Göre Azalan Sırada Sıralama")
nufus_azalan = data.sort_values('Population', ascending=False)
print("En kalabalık 20 ülke:")
print(nufus_azalan[['Country', 'Population']].head(20))


print("2. GÖREV: GDP per capita'ya göre artan sırada sıralama")
gdp_artan = data.sort_values('GDP ($ per capita)', ascending=True)
print("En düşük GDP per capita'ya sahip 20 ülke:")
print(gdp_artan[['Country', 'GDP ($ per capita)']].head(20))


print("3. GÖREV: Nüfusu 10 milyonun üzerinde olan ülkeler")
nufus_10m_uzerinde = data[data['Population'] > 10_000_000]
print(f"Nüfusu 10 milyonun üzerinde olan ülke sayısı: {len(nufus_10m_uzerinde)}")
print("Bu ülkeler:")
print(nufus_10m_uzerinde[['Country', 'Population']])


print("4. GÖREV: En yüksek okur-yazarlık oranına sahip ilk 5 ülke")
literacy_orani = data.sort_values('Literacy (%)', ascending=False)
print("En yüksek okur-yazarlık oranına sahip 5 ülke:")
print(literacy_orani[['Country', 'Literacy (%)']].head())


print("5. GÖREV: GDP per capita 10,000'in üzerinde olan ülkeler")
zengin_ulkeler = data[data['GDP ($ per capita)'] > 10000]
print(f"GDP per capita 10,000'in üzerinde olan ülke sayısı: {len(zengin_ulkeler)}")
print("Bu ülkeler:")
zengin_sirali = zengin_ulkeler.sort_values('GDP ($ per capita)', ascending=False)
print(zengin_sirali[['Country', 'GDP ($ per capita)']].head(10))


print("6. GÖREV: En yüksek nüfus yoğunluğuna sahip ilk 10 ülke")
nufus_yogunlugu = data.sort_values('Pop. Density (per sq. mi.)', ascending=False)
print("En yüksek nüfus yoğunluğuna sahip 10 ülke:")
print(nufus_yogunlugu[['Country', 'Pop. Density (per sq. mi.)']].head(20))