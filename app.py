import streamlit as st

import pandas as pd

import pickle

import shap

import matplotlib.pyplot as plt

import matplotlib

import plotly.express as px

import numpy as np

from matplotlib.patches import Patch

matplotlib.use('Agg')



st.set_page_config(

    page_title="Akademik Erken Uyarı Sistemi",

    layout="wide",

    page_icon="🎓",

    initial_sidebar_state="collapsed"

)



# ── Tema Yönetimi ─────────────────────────────────────────────────────────────

if 'dark_mode' not in st.session_state:

    st.session_state.dark_mode = True

dark = st.session_state.dark_mode



# ── Tasarım Sistemi ───────────────────────────────────────────────────────────

if dark:

    BG       = '#0D0E12'

    BG_CARD  = '#13151e'

    BG_FILT  = '#11131b'

    BORDER   = 'rgba(255,255,255,0.07)'

    BORDER_T = 'rgba(255,255,255,0.12)'

    TEXT     = '#E0E0E0'

    TEXT_S   = '#9DA3AE'

    GRID     = '#1c1f2e'

    CARD_GR_A = 'rgba(255,255,255,0.055)'

    CARD_GR_B = 'rgba(255,255,255,0.022)'

    ALERT_BG  = 'rgba(255,77,77,0.06)'

    ALERT_BD  = 'rgba(255,77,77,0.16)'

    ALERT_BDT = 'rgba(255,77,77,0.25)'

    MPL_STYLE = 'dark_background'

    TOGGLE_ICON = '☀️'

    TOGGLE_BORDER = 'rgba(255,255,255,0.15)'

    TOGGLE_BG     = 'rgba(255,255,255,0.05)'

    TOGGLE_BG_HOV = 'rgba(255,255,255,0.10)'

else:

    BG       = '#F4F6FA'

    BG_CARD  = '#FFFFFF'

    BG_FILT  = '#ECEEF4'

    BORDER   = 'rgba(0,0,0,0.08)'

    BORDER_T = 'rgba(0,0,0,0.14)'

    TEXT     = '#111827'

    TEXT_S   = '#6B7280'

    GRID     = '#E5E7EB'

    CARD_GR_A = 'rgba(255,255,255,0.95)'

    CARD_GR_B = 'rgba(255,255,255,0.70)'

    ALERT_BG  = 'rgba(255,77,77,0.05)'

    ALERT_BD  = 'rgba(255,77,77,0.14)'

    ALERT_BDT = 'rgba(255,77,77,0.22)'

    MPL_STYLE = 'default'

    TOGGLE_ICON = '🌙'

    TOGGLE_BORDER = 'rgba(0,0,0,0.15)'

    TOGGLE_BG     = 'rgba(0,0,0,0.04)'

    TOGGLE_BG_HOV = 'rgba(0,0,0,0.09)'



R_HIGH   = '#FF4D4D'

R_MID    = '#FFBF00'

R_LOW    = '#00D4AA'

ACCENT   = '#6366f1'



RENK = {

    '🔴 Yüksek Risk': R_HIGH,

    '🟡 Orta Risk':   R_MID,

    '🟢 Düşük Risk':  R_LOW,

}



st.markdown(f"""

<style>

/* ── Space Grotesk ───────────────────────────────────────────── */

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');



html, body, [class*="css"] {{

    font-family: 'Space Grotesk', sans-serif !important;

}}



/* ── Zemin ── */

body, .stApp {{ background-color: {BG} !important; }}

.block-container {{ padding: 2.2rem 2.8rem 4rem !important; max-width: 100% !important; }}



/* ── Gizle: sidebar ── */

section[data-testid="stSidebar"] {{ display: none !important; }}

[data-testid="collapsedControl"]  {{ display: none !important; }}



/* ── Deploy butonu — birden fazla selector ── */

.stDeployButton                                              {{ display: none !important; }}

[data-testid="stDeployButton"]                               {{ display: none !important; }}

[data-testid="stToolbar"] [data-testid="stDeployButton"]     {{ display: none !important; }}

[data-testid="stToolbarActions"] a[href*="deploy"]           {{ display: none !important; }}

[data-testid="stToolbarActions"] a[href*="share"]            {{ display: none !important; }}

[data-testid="stToolbarActions"] > div:nth-child(2)          {{ display: none !important; }}



/* ── Light mod: Streamlit native bileşenler ── */

{"" if dark else f"""

[data-baseweb="select"] > div,

[data-baseweb="select"] div[class*="valueContainer"],

[data-baseweb="input"] input {{

    background-color: {BG_CARD} !important;

    color: {TEXT} !important;

    border-color: {BORDER} !important;

}}

[data-testid="stSlider"] {{

    color: {TEXT} !important;

}}

div[data-testid="stAlert"] {{

    background: {BG_CARD} !important;

    color: {TEXT} !important;

}}

[data-baseweb="tag"] {{

    background-color: rgba(99,102,241,0.15) !important;

    color: {TEXT} !important;

}}

"""}



/* ── Ana başlık ── */

h1 {{

    font-size: 1.55rem !important;

    font-weight: 700 !important;

    color: {TEXT} !important;

    letter-spacing: -0.04em !important;

    margin-bottom: 0 !important;

    line-height: 1.1 !important;

}}



/* ── Alt başlıklar ── */

h3 {{

    font-size: 0.92rem !important;

    font-weight: 600 !important;

    color: {TEXT} !important;

    letter-spacing: -0.02em !important;

}}



/* ── Filtre bandı ── */

.filtre-band {{

    background: {BG_FILT};

    border: 1px solid {BORDER};

    border-top: 1px solid {BORDER_T};

    border-radius: 12px;

    padding: 0.9rem 1.4rem;

    margin-bottom: 1.6rem;

}}



/* Streamlit bileşen etiketleri */

label, .stSlider label, .stMultiSelect label {{

    font-size: 0.7rem !important;

    font-weight: 500 !important;

    color: {TEXT_S} !important;

    letter-spacing: 0.1em !important;

    text-transform: uppercase !important;

}}



/* ── KPI kartları ── */

.metric-card {{

    background: linear-gradient(145deg, {CARD_GR_A} 0%, {CARD_GR_B} 100%);

    border: 1px solid {BORDER};

    border-top: 1px solid {BORDER_T};

    border-radius: 12px;

    padding: 1.5rem 1rem 1.2rem;

    text-align: center;

    height: 158px;

    box-sizing: border-box;

    display: flex; flex-direction: column;

    align-items: center; justify-content: center;

    position: relative;

    overflow: hidden;

}}

.metric-card::before {{

    content: '';

    position: absolute;

    top: 0; left: 50%;

    transform: translateX(-50%);

    width: 60%; height: 1px;

    background: linear-gradient(90deg, transparent, {BORDER_T}, transparent);

}}

.metric-label {{

    font-size: 0.58rem;

    font-weight: 500;

    color: {TEXT_S};

    text-transform: uppercase;

    letter-spacing: 0.18em;

    margin-bottom: 0.55rem;

}}

.metric-value {{

    font-size: 2rem;

    font-weight: 700;

    color: {TEXT};

    letter-spacing: -0.05em;

    line-height: 1;

}}

.metric-pct {{

    font-size: 0.8rem;

    font-weight: 600;

    letter-spacing: 0.06em;

    margin-top: 0.35rem;

}}

.metric-bar-wrap {{

    width: 55%; height: 2px;

    background: rgba(255,255,255,0.06);

    border-radius: 1px; margin-top: 0.55rem;

}}

.metric-bar-fill {{ height: 100%; border-radius: 1px; }}



/* Risk renkleri + glow */

.risk-high {{

    color: {R_HIGH} !important;

    text-shadow: 0 6px 25px rgba(255,77,77,0.45), 0 0 50px rgba(255,77,77,0.15);

}}

.risk-mid {{

    color: {R_MID} !important;

    text-shadow: 0 6px 25px rgba(255,191,0,0.45), 0 0 50px rgba(255,191,0,0.15);

}}

.risk-low {{

    color: {R_LOW} !important;

    text-shadow: 0 6px 25px rgba(0,212,170,0.45), 0 0 50px rgba(0,212,170,0.15);

}}



/* ── Destek önceliği kartları ── */

.alert-card {{

    background: linear-gradient(145deg, {ALERT_BG} 0%, rgba(255,77,77,0.02) 100%);

    border: 1px solid {ALERT_BD};

    border-top: 1px solid {ALERT_BDT};

    border-radius: 12px;

    padding: 1.15rem 1.1rem;

    min-height: 128px;

}}

.alert-id {{

    font-weight: 700;

    font-size: 0.82rem;

    color: {TEXT};

    letter-spacing: -0.01em;

    margin-bottom: 0.6rem;

}}

.alert-row {{

    font-size: 0.73rem;

    color: {TEXT_S};

    margin-top: 0.28rem;

    display: flex;

    justify-content: space-between;

    letter-spacing: 0.01em;

}}

.alert-val {{

    font-weight: 600;

    color: {TEXT};

    letter-spacing: -0.01em;

}}



/* ── Bölüm etiketleri ── */

.section-title {{

    font-size: 0.56rem;

    text-transform: uppercase;

    letter-spacing: 0.2em;

    margin-bottom: 0.7rem;

    font-weight: 600;

    color: {TEXT_S};

    display: block;

}}



/* ── Tablo ── */

div[data-testid="stDataFrame"] {{

    border-radius: 12px;

    overflow: hidden;

    border: 1px solid {BORDER};

}}



/* ── Multiselect ── */

div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {{

    opacity: 0.75;

    filter: saturate(0.65);

}}



/* ── Sekme ── */

button[data-baseweb="tab"] {{

    font-family: 'Space Grotesk', sans-serif !important;

    font-size: 0.78rem !important;

    font-weight: 600 !important;

    letter-spacing: 0.05em !important;

    padding: 0.55rem 1.2rem !important;

    color: {TEXT_S} !important;

    text-transform: uppercase !important;

}}

button[data-baseweb="tab"][aria-selected="true"] {{

    color: {TEXT} !important;

}}

[data-baseweb="tab-highlight"] {{ background-color: {R_HIGH} !important; height: 2px !important; }}

[data-baseweb="tab-border"]    {{ background-color: {BORDER} !important; }}



/* ── Divider ── */

hr {{ border-color: {BORDER} !important; margin: 1.3rem 0 !important; }}



/* ── Info kutusu ── */

div[data-testid="stAlert"] {{

    border-radius: 12px !important;

    border: 1px solid {BORDER} !important;

    background: {BG_CARD} !important;

    font-size: 0.8rem !important;

    color: {TEXT} !important;

}}



/* ── Download button ── */

div[data-testid="stDownloadButton"] button {{

    font-family: 'Space Grotesk', sans-serif !important;

    font-size: 0.72rem !important;

    font-weight: 600 !important;

    letter-spacing: 0.05em !important;

    border-radius: 8px !important;

}}



/* ── Tema toggle butonu ── */

div[data-testid="column"]:last-child button[data-testid="stBaseButton-secondary"] {{

    background: {TOGGLE_BG} !important;

    border: 1px solid {TOGGLE_BORDER} !important;

    border-radius: 8px !important;

    color: {TEXT} !important;

    font-size: 1rem !important;

    padding: 0.3rem 0.55rem !important;

    line-height: 1 !important;

    transition: background 0.15s ease !important;

}}

div[data-testid="column"]:last-child button[data-testid="stBaseButton-secondary"]:hover {{

    background: {TOGGLE_BG_HOV} !important;

}}

</style>

""", unsafe_allow_html=True)



# ── Model & Veri ──────────────────────────────────────────────────────────────

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

        vals = [vals[:, :, i] for i in range(vals.shape[2])]

    return explainer, vals



model = load_model()

df    = load_data()

X     = df.drop(columns=['risk_grubu', 'ogrenci_id'])



df['tahmin'] = model.predict(X)

explainer, shap_values = get_shap_values(model, X)



RISK_LABEL = {0: '🟢 Düşük Risk', 1: '🟡 Orta Risk', 2: '🔴 Yüksek Risk'}

df['risk_etiketi'] = df['tahmin'].apply(lambda v: RISK_LABEL[v])



# ── Başlık + Tema Toggle ──────────────────────────────────────────────────────

hd1, hd2 = st.columns([13, 1])

with hd1:

    st.markdown(

        f"<h1>Akademik Erken Uyarı <span style='color:{R_HIGH};'>Sistemi</span></h1>",

        unsafe_allow_html=True

    )

    st.markdown(

        f"<p style='margin-top:-0.1rem;margin-bottom:1.2rem;font-size:0.78rem;"

        f"color:{TEXT_S};font-weight:400;letter-spacing:0.06em;text-transform:uppercase;'>"

        f"Danışman akademisyenler — Öğrenci risk paneli</p>",

        unsafe_allow_html=True

    )

with hd2:

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    if st.button(TOGGLE_ICON, key="theme_btn", use_container_width=True):

        st.session_state.dark_mode = not dark

        st.rerun()



# ── Filtre Bandı ──────────────────────────────────────────────────────────────

st.markdown("<div class='filtre-band'>", unsafe_allow_html=True)

f1, f2, f3 = st.columns([2, 1, 1])

with f1:

    risk_filtre = st.multiselect(

        "Risk Grubu", options=list(RISK_LABEL.values()),

        default=list(RISK_LABEL.values())

    )

with f2:

    donem_filtre = st.slider("Dönem Aralığı", 1, 8, (1, 8))

with f3:

    gpa_filtre = st.slider("GPA Aralığı", 0.0, 4.0, (0.0, 4.0), step=0.1)

st.markdown("</div>", unsafe_allow_html=True)



# ── Filtre ────────────────────────────────────────────────────────────────────

filtered_df = df[

    (df['risk_etiketi'].isin(risk_filtre)) &

    (df['donem'].between(donem_filtre[0], donem_filtre[1])) &

    (df['gpa'].between(gpa_filtre[0], gpa_filtre[1]))

].reset_index(drop=True)



# ── KPI Kartları ──────────────────────────────────────────────────────────────

toplam = len(filtered_df)



def kpi(label, value, cls="", pct_num=0, bar_color=""):

    if pct_num:

        pct_html = f"<div class='metric-pct {cls}'>{pct_num:.0f}%</div>"

        bar_html  = (f"<div class='metric-bar-wrap'>"

                     f"<div class='metric-bar-fill' style='width:{min(pct_num,100):.0f}%;"

                     f"background:{bar_color};opacity:0.8;'></div></div>")

    else:

        pct_html = "<div class='metric-pct' style='visibility:hidden'>0%</div>"

        bar_html  = "<div class='metric-bar-wrap' style='visibility:hidden'></div>"

    return (f'<div class="metric-card">'

            f'<div class="metric-label">{label}</div>'

            f'<div class="metric-value {cls}">{value}</div>'

            f'{pct_html}{bar_html}</div>')



c1, c2, c3, c4 = st.columns(4)

with c1: st.markdown(kpi("Toplam Öğrenci", toplam), unsafe_allow_html=True)

with c2:

    n = len(filtered_df[filtered_df['tahmin'] == 2])

    st.markdown(kpi("Yüksek Risk", n, "risk-high", n/toplam*100 if toplam else 0, R_HIGH), unsafe_allow_html=True)

with c3:

    n = len(filtered_df[filtered_df['tahmin'] == 1])

    st.markdown(kpi("Orta Risk", n, "risk-mid", n/toplam*100 if toplam else 0, R_MID), unsafe_allow_html=True)

with c4:

    n = len(filtered_df[filtered_df['tahmin'] == 0])

    st.markdown(kpi("Düşük Risk", n, "risk-low", n/toplam*100 if toplam else 0, R_LOW), unsafe_allow_html=True)



st.markdown("<br>", unsafe_allow_html=True)



# ── Sekmeler ──────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["Genel Bakış", "Risk Faktörleri"])



# ═══════════════════════════════════════════════════════════════════════════════

with tab1:



    yuksek = filtered_df[filtered_df['tahmin'] == 2].sort_values(

        ['devamsizlik_oran', 'gpa'], ascending=[False, True]

    ).head(5)



    if not yuksek.empty:

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(

            f"<h3 style='color:{TEXT}'>Destek Önceliği Olan Öğrenciler</h3>",

            unsafe_allow_html=True

        )

        st.markdown(

            f"<p style='font-size:0.76rem;color:{TEXT_S};margin-top:-0.15rem;"

            f"margin-bottom:1rem;letter-spacing:0.01em;'>"

            "En riskli 5 öğrenci — devamsızlık yüksekten düşüğe, eşitlikte GPA düşükten yükseğe.</p>",

            unsafe_allow_html=True

        )

        acols = st.columns(min(len(yuksek), 5))

        for i, (_, row) in enumerate(yuksek.iterrows()):

            with acols[i]:

                st.markdown(f"""

                <div class="alert-card">

                    <div class="alert-id">{row['ogrenci_id']}</div>

                    <div class="alert-row"><span>GPA</span><span class="alert-val">{row['gpa']:.2f}</span></div>

                    <div class="alert-row"><span>Devamsızlık</span><span class="alert-val">{row['devamsizlik_oran']:.1f}%</span></div>

                    <div class="alert-row"><span>Ders Tekrar</span><span class="alert-val">{int(row['ders_tekrar_sayisi'])}</span></div>

                    <div class="alert-row"><span>Dönem</span><span class="alert-val">{int(row['donem'])}</span></div>

                </div>

                """, unsafe_allow_html=True)



    st.markdown("<br>", unsafe_allow_html=True)

    st.divider()

    st.markdown(f"<h3 style='color:{TEXT}'>Analiz Paneli</h3>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)



    ga1, ga2 = st.columns(2, gap="large")

    with ga1:

        st.markdown("<span class='section-title'>Dönem Bazlı Risk Dağılımı</span>", unsafe_allow_html=True)

        donem_risk = filtered_df.groupby(['donem', 'risk_etiketi']).size().reset_index(name='sayi')

        fig_d = px.bar(donem_risk, x='donem', y='sayi', color='risk_etiketi',

                       color_discrete_map=RENK,

                       labels={'donem': 'Dönem', 'sayi': 'Öğrenci Sayısı', 'risk_etiketi': ''},

                       barmode='stack')

        fig_d.update_layout(

            margin=dict(l=0, r=0, t=10, b=0), height=300,

            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,

                        font=dict(size=10, color=TEXT_S, family='Space Grotesk')),

            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,

            font=dict(family='Space Grotesk', color=TEXT_S),

            xaxis=dict(showgrid=False, dtick=1, color=TEXT_S,

                       tickfont=dict(color=TEXT_S, size=10)),

            yaxis=dict(gridcolor=GRID, color=TEXT_S,

                       tickfont=dict(color=TEXT_S, size=10)),

        )

        st.plotly_chart(fig_d, use_container_width=True)



    with ga2:

        st.markdown("<span class='section-title'>GPA — Devamsızlık İlişkisi</span>", unsafe_allow_html=True)

        fig_s = px.scatter(filtered_df, x='devamsizlik_oran', y='gpa',

                           color='risk_etiketi', color_discrete_map=RENK,

                           hover_data={'ogrenci_id': True, 'ders_tekrar_sayisi': True, 'risk_etiketi': False},

                           labels={'devamsizlik_oran': 'Devamsızlık (%)', 'gpa': 'GPA', 'risk_etiketi': ''},

                           opacity=0.7)

        fig_s.update_layout(

            margin=dict(l=0, r=0, t=10, b=0), height=300,

            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,

                        font=dict(size=10, color=TEXT_S, family='Space Grotesk')),

            paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,

            font=dict(family='Space Grotesk', color=TEXT_S),

            xaxis=dict(showgrid=False, color=TEXT_S, tickfont=dict(color=TEXT_S, size=10)),

            yaxis=dict(gridcolor=GRID, color=TEXT_S, tickfont=dict(color=TEXT_S, size=10)),

        )

        st.plotly_chart(fig_s, use_container_width=True)



    st.divider()

    hdr, btn = st.columns([7, 1])

    with hdr:

        st.markdown(f"<h3 style='color:{TEXT}'>Öğrenci Listesi</h3>", unsafe_allow_html=True)

    with btn:

        st.markdown("<br>", unsafe_allow_html=True)

        csv = filtered_df.to_csv(index=False).encode('utf-8')

        st.download_button("CSV İndir", data=csv,

                           file_name="ogrenci_risk_raporu.csv", mime="text/csv")



    kolon_adlari = {

        'ogrenci_id': 'ID', 'devamsizlik_oran': 'Devamsızlık (%)',

        'gpa': 'GPA', 'ders_tekrar_sayisi': 'Ders Tekrar',

        'kredi_yuku': 'Kredi Yükü', 'donem': 'Dönem',

        'onceki_basari': 'Önceki Başarı', 'risk_etiketi': 'Risk'

    }

    display_df = filtered_df[list(kolon_adlari.keys())].rename(columns=kolon_adlari)



    def highlight_risk(s):

        out = []

        for v in s:

            if '🔴' in str(v):

                out.append(f'background-color:rgba(255,77,77,0.1);color:{R_HIGH};font-weight:600')

            elif '🟡' in str(v):

                out.append(f'background-color:rgba(255,191,0,0.1);color:{R_MID};font-weight:600')

            elif '🟢' in str(v):

                out.append(f'background-color:rgba(0,212,170,0.1);color:{R_LOW};font-weight:600')

            else:

                out.append('')

        return out



    styled_df = (

        display_df.style

        .format({'Devamsızlık (%)': '{:.2f}', 'GPA': '{:.2f}', 'Önceki Başarı': '{:.2f}'})

        .apply(highlight_risk, subset=['Risk'])

    )

    st.dataframe(styled_df, use_container_width=True, height=320)



# ═══════════════════════════════════════════════════════════════════════════════

with tab2:

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(

        f"<p style='font-size:0.78rem;color:{TEXT_S};margin-bottom:1rem;letter-spacing:0.01em;'>"

        "Her özelliğin modelin risk tahminini ne ölçüde etkilediğini gösterir. "

        "Sağ panelde seçilen öğrenciye ait bireysel faktör analizi yer alır.</p>",

        unsafe_allow_html=True

    )



    secilen_id = st.selectbox(

        "Öğrenci Seç", options=df['ogrenci_id'].tolist(), key="shap_selectbox"

    )

    idx = df.index[df['ogrenci_id'] == secilen_id][0]

    ogrenci_risk = int(df['tahmin'].iloc[idx])

    try:

        sv = shap_values[ogrenci_risk][idx]

    except (IndexError, TypeError):

        sv = shap_values[idx]



    st.markdown("<br>", unsafe_allow_html=True)

    FIG_W, FIG_H = 9, 5



    def style_ax(fig, ax):

        fig.set_facecolor(BG_CARD)

        ax.set_facecolor(BG_CARD)

        ax.tick_params(colors=TEXT_S, labelsize=8.5)

        ax.xaxis.label.set_color(TEXT_S)

        ax.xaxis.label.set_size(8.5)

        for label in ax.get_xticklabels() + ax.get_yticklabels():

            label.set_color(TEXT_S)

        for spine in ax.spines.values():

            spine.set_edgecolor(GRID)



    sh1, sh2 = st.columns([1, 1], gap="large")



    with sh1:

        st.markdown("<span class='section-title'>Değişken Ağırlıkları</span>", unsafe_allow_html=True)

        plt.close('all')

        plt.style.use(MPL_STYLE)

        shap.summary_plot(shap_values, X, plot_type='bar',

                          class_names=['Düşük Risk', 'Yüksek Risk', 'Orta Risk'], show=False)

        fig1 = plt.gcf()

        fig1.set_size_inches(FIG_W, FIG_H)

        if fig1.axes:

            style_ax(fig1, fig1.axes[0])

        fig1.subplots_adjust(left=0.22, right=0.97, top=0.95, bottom=0.12)

        st.pyplot(fig1, use_container_width=True)

        plt.close('all')



    with sh2:

        st.markdown("<span class='section-title'>Öğrenci Detay Analizi</span>", unsafe_allow_html=True)



        feat_names = X.columns.tolist()

        abs_order  = np.argsort(np.abs(sv))

        s_vals     = sv[abs_order]

        s_names    = [feat_names[i] for i in abs_order]

        colors     = [R_HIGH if v >= 0 else '#4d9eff' for v in s_vals]



        plt.close('all')

        plt.style.use(MPL_STYLE)

        fig2, ax2 = plt.subplots(figsize=(FIG_W, FIG_H))

        style_ax(fig2, ax2)

        ax2.barh(s_names, s_vals, color=colors, alpha=0.85, height=0.52)

        ax2.axvline(x=0, color=TEXT_S, linewidth=0.5, alpha=0.4)

        ax2.set_xlabel('SHAP değeri', fontsize=8.5, color=TEXT_S)

        ax2.set_title(

            f'{secilen_id}  ·  {RISK_LABEL[ogrenci_risk]}',

            color=TEXT, fontsize=10, pad=10, fontweight='semibold',

            fontfamily='monospace'

        )

        ax2.legend(

            handles=[Patch(color=R_HIGH, label='Riski artırıyor', alpha=0.85),

                     Patch(color='#4d9eff', label='Riski azaltıyor', alpha=0.85)],

            loc='lower right', fontsize=7.5,

            facecolor=BG_CARD, edgecolor=GRID, labelcolor=TEXT_S

        )

        fig2.subplots_adjust(left=0.22, right=0.97, top=0.92, bottom=0.12)

        st.pyplot(fig2, use_container_width=True)

        plt.close('all')



    st.markdown("<br>", unsafe_allow_html=True)

    st.info(

        "Kırmızı çubuklar riski artıran, mavi çubuklar riski azaltan faktörleri temsil eder. "

        "Çubuk uzunluğu o faktörün göreli etkisini gösterir.",

        icon="ℹ️"

    )



st.markdown(

    f"<p style='font-size:0.6rem;text-align:center;color:{TEXT_S};opacity:0.3;"

    f"margin-top:3rem;letter-spacing:0.12em;text-transform:uppercase;'>"

    "Makine Öğrenmesi Tabanlı Akademik Erken Uyarı Sistemi</p>",

    unsafe_allow_html=True

)