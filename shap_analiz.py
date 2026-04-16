import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt

# Model ve veriyi yükle
with open('model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('data/ogrenci_veri.csv')
X = df.drop(columns=['risk_grubu'])
y = df['risk_grubu']

# SHAP açıklayıcı oluştur
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 1. Summary Plot → genel değişken önemi
plt.figure()
shap.summary_plot(shap_values, X, class_names=['Düşük Risk', 'Orta Risk', 'Yüksek Risk'], show=False)
plt.tight_layout()
plt.savefig('model/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Summary plot kaydedildi → model/shap_summary.png")

# 2. Force Plot → tek öğrenci analizi (ilk öğrenci)
shap.initjs()
ogrenci_idx = 0
print(f"\n📊 Öğrenci {ogrenci_idx} risk grubu: {y[ogrenci_idx]}")
print("Özellik değerleri:")
print(X.iloc[ogrenci_idx])

# Bar plot olarak bireysel analiz
plt.figure()
shap.bar_plot(shap_values[y[ogrenci_idx]][ogrenci_idx], 
              feature_names=X.columns.tolist(),
              show=False)
plt.title(f'Öğrenci {ogrenci_idx} - Risk Grubu: {y[ogrenci_idx]}')
plt.tight_layout()
plt.savefig('model/shap_force.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Force plot kaydedildi → model/shap_force.png")