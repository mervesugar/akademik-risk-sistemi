import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle

# Veriyi yükle
df = pd.read_csv('data/ogrenci_veri.csv')
X = df.drop(columns=['risk_grubu', 'ogrenci_id'])
y = df['risk_grubu']

# SMOTE ile sınıf dengesi
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
print("✅ SMOTE sonrası dağılım:")
print(pd.Series(y_balanced).value_counts())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Random Forest
print("\n🌲 Random Forest eğitiliyor...")
rf_params = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='f1_macro')
rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_
rf_pred = rf_model.predict(X_test)
print(f"RF Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"RF F1 Score: {f1_score(y_test, rf_pred, average='macro'):.4f}")

# XGBoost
print("\n⚡ XGBoost eğitiliyor...")
xgb_params = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.1, 0.01]}
xgb_grid = GridSearchCV(xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'), xgb_params, cv=5, scoring='f1_macro')
xgb_grid.fit(X_train, y_train)
xgb_model = xgb_grid.best_estimator_
xgb_pred = xgb_model.predict(X_test)
print(f"XGB Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
print(f"XGB F1 Score: {f1_score(y_test, xgb_pred, average='macro'):.4f}")

# AUC-ROC
from sklearn.preprocessing import label_binarize
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
xgb_prob = xgb_model.predict_proba(X_test)
rf_prob = rf_model.predict_proba(X_test)
print(f"\nXGB AUC-ROC: {roc_auc_score(y_test_bin, xgb_prob, multi_class='ovr', average='macro'):.4f}")
print(f"RF AUC-ROC:  {roc_auc_score(y_test_bin, rf_prob, multi_class='ovr', average='macro'):.4f}")

# En iyi modeli seç ve kaydet
if f1_score(y_test, rf_pred, average='macro') >= f1_score(y_test, xgb_pred, average='macro'):
    best_model = rf_model
    best_name = "Random Forest"
else:
    best_model = xgb_model
    best_name = "XGBoost"

with open('model/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"\n🏆 En iyi model: {best_name}")
print("✅ Model kaydedildi → model/best_model.pkl")