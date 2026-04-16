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

if 'tema' not in st.session_state:
    st.session_state.tema = 'koyu'

tema = st.session_state.tema
koyu = tema == 'koyu'

bg = '#0f1117' if koyu else '#f8f9fc'
sidebar_bg = '#13151f' if koyu else '#ffffff'
kart_bg = '#1a1d2e' if koyu else '#ffffff'
kart_border = '#2d3148' if koyu else '#e2e8f0'
text_primary = '#ffffff' if koyu else '#0f172a'
text_secondary = '#6b7280' if koyu else '#64748b'
divider = '#1f2235' if koyu else '#e2e8f0'
grafik_bg = '#1a1d2e' if koyu else '#ffffff'
grafik_text = '#9ca3af' if koyu else '#374151'
grafik_spine = '#2d3148' if koyu else '#e2e8f0'
footer_text = '#374151' if koyu else '#94a3b8'

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background-color: {bg}; }}
.block-container {{ padding: 2rem 2.5rem; }}

h1 {{ font-size: 1.6rem !important; font-weight: 600 !important; color: {text_primary} !important; letter-spacing: -0.3px; }}
h2, h3 {{ font-weight: 500 !important; color: {text_primary} !important; }}

.metric-card {{
    background: {kart_bg};
    border: 1px solid {kart_border};
    border-radius: 10px;
    padding: 1rem 1.25rem;
    text-align: center;
}}
.metric-label {{
    font-size: 0.7rem;
    color: {text_secondary};
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.3rem;
    font-weight: 500;
}}
.metric-value {{
    font-size: 2rem;
    font-weight: 600;
    color: {text_primary};
    font-family: 'JetBrains Mono', monospace;
}}

.risk-high {{ color: #f87171 !important; }}
.risk-mid  {{ color: #f59e0b !important; }}
.risk-low  {{ color: #10b981 !important; }}

.section-title {{
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: {text_secondary};
    margin-bottom: 0.75rem;
    font-weight: 500;
}}

div[data-testid="stDataFrame"] {{
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid {kart_border};
}}

section[data-testid="stSidebar"] {{
    background-color: {sidebar_bg} !important;
    border-right: 1px solid {kart_border} !important;
}}

section[data-testid="stSidebar"] * {{
    color: {text_primary} !important;
}}

hr {{ border-color: {divider} !important; margin: 1.5rem 0 !important; }}

.sidebar-label {{
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: {text_secondary};
    font-weight: 500;
    margin-bottom: 0.5rem;
    margin-top: 1.2rem;
}}

.sidebar-title {{
    font-size: 0.95rem;
    font-weight: 600;
    color: {text_primary};
    margin: 0;
}}
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
    if hasattr(vals, '__len__') and not isinstance(vals, list):
        import numpy as np
        vals = [vals[:, :, i] for i in range(vals.shape[2])]
    return explainer, vals

model = load_model()
df = load_data()
X = df.drop(columns=['risk_grubu', 'ogrenci_id'])

df['tahmin'] = model.predict(X)
explainer, shap_values = get_shap_values(model, X)

def risk_etiketi(val):
    if val == 0: return '🟢 Düşük Risk'
    elif val == 1: return '🟡 Orta Risk'
    else: return '🔴 Yüksek Risk'

df['risk_etiketi'] = df['tahmin'].apply(risk_etiketi)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding: 0.5rem 0 1.5rem;'>
        <p class='sidebar-title'>Erken Uyarı Sistemi</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("<p class='sidebar-label'>Filtreler</p>", unsafe_allow_html=True)

    risk_filtre = st.multiselect(
        "Risk grubu",
        options=['🟢 Düşük Risk', '🟡 Orta Risk', '🔴 Yüksek Risk'],
        default=['🟢 Düşük Risk', '🟡 Orta Risk', '🔴 Yüksek Risk'],
        label_visibility="collapsed"
    )

    st.markdown("<p class='sidebar-label'>Dönem</p>", unsafe_allow_html=True)
    donem_filtre = st.slider("", 1, 8, (1, 8), label_visibility="collapsed")

    st.markdown("<p class='sidebar-label'>GPA Aralığı</p>", unsafe_allow_html=True)
    gpa_filtre = st.slider("", 0.0, 4.0, (0.0, 4.0), step=0.1, label_visibility="collapsed")

    st.divider()
    st.markdown(f"<p style='font-size:0.7rem; color:{text_secondary};'>1.000 öğrenci kaydı</p>", unsafe_allow_html=True)

# ── Filtrele ──────────────────────────────────────────────────────────────────
filtered_df = df[
    (df['risk_etiketi'].isin(risk_filtre)) &
    (df['donem'].between(donem_filtre[0], donem_filtre[1])) &
    (df['gpa'].between(gpa_filtre[0], gpa_filtre[1]))
].reset_index(drop=True)

# ── Başlık + Tema Toggle ──────────────────────────────────────────────────────
title_col, toggle_col = st.columns([11, 1])
with title_col:
    st.markdown(f"<h1>Akademik Erken Uyarı Sistemi</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{text_secondary}; margin-top:-0.6rem; margin-bottom:1.5rem; font-size:0.9rem;'>Danışman akademisyenler için öğrenci risk analizi</p>", unsafe_allow_html=True)
with toggle_col:
    st.markdown("<br>", unsafe_allow_html=True)
    tema_sec = st.radio("", ["🌙", "☀️"], index=0 if koyu else 1, label_visibility="collapsed", horizontal=True)
    if (tema_sec == "🌙") != koyu:
        st.session_state.tema = 'koyu' if tema_sec == "🌙" else 'acik'
        st.rerun()

# ── Metrik kartları ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">Toplam</div><div class="metric-value" style="color:{text_primary}">{len(filtered_df)}</div></div>""", unsafe_allow_html=True)
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

# ── Öğrenci Tablosu ───────────────────────────────────────────────────────────
st.markdown("### Öğrenci Listesi")
kolon_adlari = {
    'ogrenci_id': 'ID',
    'devamsizlik_oran': 'Devamsızlık (%)',
    'gpa': 'GPA',
    'ders_tekrar_sayisi': 'Ders Tekrar',
    'kredi_yuku': 'Kredi Yükü',
    'donem': 'Dönem',
    'onceki_basari': 'Önceki Başarı',
    'risk_etiketi': 'Risk'
}
st.dataframe(
    filtered_df[list(kolon_adlari.keys())].rename(columns=kolon_adlari),
    use_container_width=True,
    height=300
)

st.divider()

# ── SHAP ─────────────────────────────────────────────────────────────────────
st.markdown("### SHAP Analizi")
col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.markdown("<p class='section-title'>Genel Değişken Önemi</p>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(grafik_bg)
    ax.set_facecolor(grafik_bg)
    shap.summary_plot(shap_values, X, plot_type='bar',
                      class_names=['Düşük Risk', 'Orta Risk', 'Yüksek Risk'], show=False)
    ax.tick_params(colors=grafik_text)
    ax.xaxis.label.set_color(grafik_text)
    for spine in ax.spines.values():
        spine.set_edgecolor(grafik_spine)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.markdown("<p class='section-title'>Bireysel Öğrenci Analizi</p>", unsafe_allow_html=True)
    ogrenci_idx = st.number_input("Öğrenci No (0 – 999)", min_value=0, max_value=len(df)-1, value=0, step=1)
    idx = int(ogrenci_idx)
    ogrenci_risk = int(df['tahmin'].iloc[idx])

    try:
        sv = shap_values[ogrenci_risk][idx]
    except (IndexError, TypeError):
        import numpy as np
        sv = shap_values[idx]

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    fig2.patch.set_facecolor(grafik_bg)
    ax2.set_facecolor(grafik_bg)
    shap.bar_plot(sv, feature_names=X.columns.tolist(), show=False)
    ax2.set_title(f'Öğrenci {idx} → {risk_etiketi(ogrenci_risk)}', color=grafik_text, fontsize=11, pad=10)
    ax2.tick_params(colors=grafik_text)
    ax2.xaxis.label.set_color(grafik_text)
    for spine in ax2.spines.values():
        spine.set_edgecolor(grafik_spine)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

st.markdown(f"<br><p style='color:{footer_text}; font-size:0.7rem; text-align:center;'>Makine Öğrenmesi Tabanlı Akademik Erken Uyarı Sistemi</p>", unsafe_allow_html=True)