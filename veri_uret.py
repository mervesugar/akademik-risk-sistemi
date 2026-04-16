import pandas as pd
import numpy as np

np.random.seed(42)

# Kaggle verisini yükle
df = pd.read_csv('data/Student_performance_data _.csv')

# Sadece ihtiyacımız olan sütunları al
df = df[['Absences', 'GPA', 'GradeClass']].copy()
df = df.sample(n=1000, random_state=42).reset_index(drop=True)

# Öğrenci ID ekle
df.insert(0, 'ogrenci_id', [f'OGR{str(i+1).zfill(4)}' for i in range(len(df))])

# Sütun isimlerini Türkçeleştir
df.rename(columns={
    'Absences': 'devamsizlik_oran',
    'GPA': 'gpa'
}, inplace=True)

# Devamsızlığı yüzdeye çevir
df['devamsizlik_oran'] = (df['devamsizlik_oran'] / 30 * 100).round(1)

# Dönem sayısı — 1'den 8'e rastgele
df['donem'] = np.random.randint(1, 9, len(df))

# GPA'yı düzelt — ortalama 2.2-2.5 civarında olsun
# Orijinal GPA'yı yukarı kaydır, 0-4 aralığında tut
df['gpa'] = np.clip(df['gpa'] + np.random.uniform(0.3, 0.7, len(df)), 0, 4).round(4)

# Dönem ile GPA korelasyonu — ileri dönemlerde GPA biraz yükselsin
donem_etkisi = (df['donem'] - 1) * 0.05
df['gpa'] = np.clip(df['gpa'] + donem_etkisi, 0, 4).round(4)

# Diğer sütunlar
df['ders_tekrar_sayisi'] = np.random.randint(0, 6, len(df))
df['kredi_yuku'] = np.random.randint(15, 35, len(df))
df['onceki_basari'] = np.clip(df['gpa'] + np.random.uniform(-0.3, 0.3, len(df)), 0, 4).round(2)

# Risk grubu
def risk_donustur(row):
    skor = 0

    if row['gpa'] < 1.5: skor += 3
    elif row['gpa'] < 2.0: skor += 2
    elif row['gpa'] < 2.8: skor += 1

    if row['devamsizlik_oran'] > 60: skor += 3
    elif row['devamsizlik_oran'] > 40: skor += 2
    elif row['devamsizlik_oran'] > 25: skor += 1

    if row['ders_tekrar_sayisi'] >= 4: skor += 2
    elif row['ders_tekrar_sayisi'] >= 2: skor += 1

    if row['kredi_yuku'] > 30: skor += 1

    if skor >= 6: return 2
    elif skor >= 3: return 1
    else: return 0

df['risk_grubu'] = df.apply(risk_donustur, axis=1)
df.drop(columns=['GradeClass'], inplace=True)

# Kaydet
df.to_csv('data/ogrenci_veri.csv', index=False)
print("✅ Veri hazır!")
print(f"GPA ortalaması: {df['gpa'].mean():.2f}")
print(df['risk_grubu'].value_counts())
print(df.head())