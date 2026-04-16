import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(
    page_title="Akademik Erken Uyarı Sistemi",
    layout="wide",
    page_icon="🎓"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0f1117; }
.block-container { padding: 2rem 3rem; }
h1 { font-size: 2rem !important; font-weight: 600 !important; color: #ffffff !important; letter-spacing: -0.5px; }
h2, h3 { font-weight: 500 !important; color: #e2e8f0 !important; }
.metric-card { background: #1a1d2e; border: 1px solid #2d3148; border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; }
.metric-label { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.4rem; }
.metric-value { font-size: 2.2rem; font-weight: 600; color: #ffffff; font-family: 'DM Mono', monospace; }
.risk-high { color: #f87171 !important; }
.risk-mid  { color: #fbbf24 !important; }
.risk-low  { color: #34d399 !important; }
.section-title { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 2px; color: #4b5563; margin-bottom: 0.5rem; }
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid #2d3148; }
.stSidebar { background-color: #13151f !important; border-right: 1px solid #1f2235; }
hr { border-color: #1f2235 !important; margin: 2rem 0 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('model/best_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv('data/ogrenci_veri.csv')

@st.cache_resource
def get_shap_values(_model, _X):
    explainer = shap.TreeExplainer(_model)
    vals = explainer.shap_values(_X)
    # XGBoost bazen 3D array döner, liste formatına çevir
    if hasattr(vals, '__len__') and not isinstance(vals, list):
        import numpy as np
        vals = [vals[:, :, i] for i in range(vals.shape[2])]
    return explainer, vals

model = load_model()
df = load_data()
X = df.drop(columns=['risk_grubu'])

df['tahmin'] = model.predict(X)
explainer, shap_values = get_shap_values(model, X)

def risk_etiketi(val):
    if val == 0: return '🟢 Düşük Risk'
    elif val == 1: return '🟡 Orta Risk'
    else: return '🔴 Yüksek Risk'

df['risk_etiketi'] = df['tahmin'].apply(risk_etiketi)

# Sidebar
st.sidebar.markdown("## 🎓 EWS Panel")
st.sidebar.markdown("---")
st.sidebar.markdown("### Filtreler")

risk_filtre = st.sidebar.multiselect(
    "Risk Grubu",
    options=['🟢 Düşük Risk', '🟡 Orta Risk', '🔴 Yüksek Risk'],
    default=['🟢 Düşük Risk', '🟡 Orta Risk', '🔴 Yüksek Risk']
)
donem_filtre = st.sidebar.slider("Dönem Aralığı", 1, 8, (1, 8))
gpa_filtre = st.sidebar.slider("GPA Aralığı", 0.0, 4.0, (0.0, 4.0), step=0.1)

filtered_df = df[
    (df['risk_etiketi'].isin(risk_filtre)) &
    (df['donem'].between(donem_filtre[0], donem_filtre[1])) &
    (df['gpa'].between(gpa_filtre[0], gpa_filtre[1]))
].reset_index(drop=True)

# Başlık
st.markdown("# 🎓 Akademik Erken Uyarı Sistemi")
st.markdown("<p style='color:#6b7280; margin-top:-0.8rem; margin-bottom:1.5rem;'>Danışman akademisyenler için yapay zeka destekli öğrenci risk analizi</p>", unsafe_allow_html=True)

# Metrik kartları
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Toplam Öğrenci</div><div class="metric-value">{len(filtered_df)}</div></div>""", unsafe_allow_html=True)
with c2:
    cnt = len(filtered_df[filtered_df['tahmin'] == 2])
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Yüksek Risk</div><div class="metric-value risk-high">{cnt}</div></div>""", unsafe_allow_html=True)
with c3:
    cnt = len(filtered_df[filtered_df['tahmin'] == 1])
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Orta Risk</div><div class="metric-value risk-mid">{cnt}</div></div>""", unsafe_allow_html=True)
with c4:
    cnt = len(filtered_df[filtered_df['tahmin'] == 0])
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Düşük Risk</div><div class="metric-value risk-low">{cnt}</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# Öğrenci Tablosu
st.markdown("### 📋 Öğrenci Listesi")
kolon_adlari = {
    'devamsizlik_oran': 'Devamsızlık (%)',
    'gpa': 'GPA',
    'ders_tekrar_sayisi': 'Ders Tekrar',
    'kredi_yuku': 'Kredi Yükü',
    'donem': 'Dönem',
    'onceki_basari': 'Önceki Başarı',
    'risk_etiketi': 'Risk Durumu'
}
st.dataframe(
    filtered_df[list(kolon_adlari.keys())].rename(columns=kolon_adlari),
    use_container_width=True,
    height=320
)

st.divider()

# SHAP Analizi
st.markdown("### 🔍 SHAP Analizi")
col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.markdown("<p class='section-title'>Genel Değişken Önemi</p>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#1a1d2e')
    ax.set_facecolor('#1a1d2e')
    shap.summary_plot(shap_values, X, plot_type='bar',
                      class_names=['Düşük Risk', 'Orta Risk', 'Yüksek Risk'], show=False)
    ax.tick_params(colors='#9ca3af')
    ax.xaxis.label.set_color('#9ca3af')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3148')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("<p class='section-title'>Bireysel Öğrenci Analizi</p>", unsafe_allow_html=True)
    ogrenci_idx = st.number_input("Öğrenci No seç (0 – 999)", min_value=0, max_value=len(df)-1, value=0, step=1)
    idx = int(ogrenci_idx)
    ogrenci_risk = int(df['tahmin'].iloc[idx])
    
    try:
        sv = shap_values[ogrenci_risk][idx]
    except (IndexError, TypeError):
        import numpy as np
        sv = shap_values[idx]
    
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    fig2.patch.set_facecolor('#1a1d2e')
    ax2.set_facecolor('#1a1d2e')
    shap.bar_plot(sv, feature_names=X.columns.tolist(), show=False)
    ax2.set_title(f'Öğrenci {idx} → {risk_etiketi(ogrenci_risk)}', color='#e2e8f0', fontsize=11, pad=10)
    ax2.tick_params(colors='#9ca3af')
    ax2.xaxis.label.set_color('#9ca3af')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#2d3148')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.markdown("<br><p style='color:#374151; font-size:0.75rem; text-align:center;'>Makine Öğrenmesi Tabanlı Akademik Erken Uyarı Sistemi • XGBoost + SHAP</p>", unsafe_allow_html=True)