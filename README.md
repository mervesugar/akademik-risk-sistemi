# Akademik Erken Uyarı Sistemi

Üniversite öğrencilerinin akademik riskini dönem içinde tahmin eden, makine öğrenmesi tabanlı bir erken uyarı sistemi.

## Proje Hakkında

Mevcut üniversite sistemleri öğrenci başarısızlığını ancak dönem sonu not ortalamasına bakarak tespit edebilmektedir. Bu proje, devamsızlık oranı, not ortalaması (GPA), ders tekrar sayısı ve kredi yükü gibi çok boyutlu akademik verileri analiz ederek riski erken aşamada tahmin etmeyi hedeflemektedir.

## Özellikler

- Random Forest ve XGBoost ile üç sınıflı risk tahmini (Düşük / Orta / Yüksek)
- SMOTE ile sınıf dengesizliği giderimi
- SHAP ile model yorumlanabilirliği
- Streamlit tabanlı interaktif dashboard
- Açık/koyu tema desteği
- Öğrenci bazlı bireysel risk analizi

## Model Performansı

| Model | Accuracy | F1 Score | AUC-ROC |
|---|---|---|---|
| XGBoost | %97.35 | 0.9734 | 0.9965 |
| Random Forest | %96.46 | 0.9643 | 0.9957 |

## Kurulum

```bash
git clone https://github.com/mervesugar/akademik-risk-sistemi.git
cd akademik-risk-sistemi
pip install -r requirements.txt
```

## Kullanım

```bash
# 1. Veriyi üret
python veri_uret.py

# 2. Modeli eğit
python model_egit.py

# 3. SHAP analizini çalıştır
python shap_analiz.py

# 4. Dashboard'u başlat
streamlit run app.py
```

## Veri Seti

Kaggle Student Performance Dataset referans alınarak NumPy ile 1.000 adet sentetik öğrenci kaydı üretilmiştir. Risk grubu dağılımı OECD (2023) ve Frontiers in Education (2023) kaynaklarına dayandırılmıştır.

| Değişken | Açıklama |
|---|---|
| devamsizlik_oran | Dönemlik devamsızlık oranı (%) |
| gpa | Ağırlıklı not ortalaması (0–4) |
| ders_tekrar_sayisi | Tekrar alınan ders sayısı |
| kredi_yuku | Alınan kredi miktarı |
| donem | Kaçıncı dönemde olduğu |
| onceki_basari | Önceki dönem başarı durumu |

## Teknolojiler

Python • Scikit-learn • XGBoost • SHAP • Streamlit • Pandas • NumPy • imbalanced-learn

## Kaynaklar

- OECD (2023). Education at a Glance 2023. https://doi.org/10.1787/e13bef63-en
- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
- Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. KDD.
- Frontiers in Education (2023). Factors contributing to university dropout: a review.
