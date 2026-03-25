import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go

# ======================
# KONFIGURASI HALAMAN & CSS
# ======================
st.set_page_config(page_title="Fertilizer Optimization", page_icon="🌱", layout="wide")

st.markdown("""
    <style>
    /* Styling Teks Premium untuk Dark Mode */
    .premium-title {
        font-family: 'Helvetica Neue', sans-serif;
        color: #FFFFFF;
        font-size: 2.8rem !important;
        font-weight: 800;
        margin-bottom: -10px;
    }
    .premium-subtitle {
        color: #81C784;
        font-size: 1.2rem !important;
        font-weight: 500;
        margin-bottom: 25px;
    }
    /* Memperbesar Metrik Hasil */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #A5D6A7;
        font-weight: 800;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #424242;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODEL AI
# ======================
@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "FO2rfmodel.pkl")
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"⚠️ File Model tidak ditemukan di: {MODEL_PATH}")
        return None

rf_model = load_model()

# ======================
# SMART DEFAULT DATA (PENGGANTI CSV)
# ======================
# Karena CSV dihapus, kita buat baseline default yang masuk akal
default_features = {
    'umur_tanaman': [10.0],
    'curah_hujan': [2500.0],
    'hari_hujan': [150.0],
    'populasi_ha': [143.0],
    'pupuk_N': [150.0],
    'pupuk_P': [100.0],
    'pupuk_K': [150.0]
}
sample_data = pd.DataFrame(default_features)

# ======================
# HEADER UTAMA
# ======================
st.markdown('<p class="premium-title">🌱 AI Fertilizer Optimization</p>', unsafe_allow_html=True)
st.markdown('<p class="premium-subtitle">Simulasi Dosis Pupuk Maksimal untuk Profitabilitas Kebun</p>', unsafe_allow_html=True)

if rf_model is None:
    st.stop() # Hentikan eksekusi jika model tidak ada

# ======================
# CONTROL PANEL INTERAKTIF
# ======================
with st.expander("⚙️ Control Panel: Parameter & Harga", expanded=True):
    st.markdown("Sesuaikan kondisi lapangan dan harga pasar saat ini:")
    
    col_env, col_base, col_price = st.columns(3)
    
    with col_env:
        st.caption("🌍 Kondisi Lingkungan")
        umur = st.number_input("Umur Tanaman (Tahun)", 1, 30, 10)
        curah_hujan = st.number_input("Curah Hujan Tahunan (mm)", 1000, 4000, 2500)
        harga_jual = st.number_input("Harga Jual TBS (Rp/kg)", 1000, 4000, 2500, step=100)
        
    with col_base:
        st.caption("🧪 Baseline Dosis Saat Ini (kg/ha)")
        base_N = st.slider("Nitrogen (N)", 0, 300, 150)
        base_P = st.slider("Fosfor (P)", 0, 300, 100)
        base_K = st.slider("Kalium (K)", 0, 300, 150)
        
    with col_price:
        st.caption("💰 Harga Pupuk (Rp/kg)")
        harga_N = st.number_input("Harga N (Urea/ZA)", 5000, 20000, 12000, step=500)
        harga_P = st.number_input("Harga P (TSP/RP)", 5000, 20000, 10000, step=500)
        harga_K = st.number_input("Harga K (MOP/KCl)", 5000, 20000, 9000, step=500)

# Update sample_data dengan input user terbaru
sample_data['umur_tanaman'] = umur
sample_data['curah_hujan'] = curah_hujan

# ======================
# ENGINE SIMULASI PROFIT
# ======================
def calculate_profit(yield_pred, n, p, k):
    biaya = (n * harga_N) + (p * harga_P) + (k * harga_K)
    revenue = yield_pred * harga_jual
    return revenue - biaya

N_range = np.linspace(0, 300, 50) # Dikurangi jadi 50 agar lebih cepat diproses
P_range = np.linspace(0, 300, 50)
K_range = np.linspace(0, 300, 50)

# Fungsi pembantu untuk iterasi
def get_profit_curve(nutrient_type, val_range, base_n, base_p, base_k):
    profits = []
    for val in val_range:
        temp = sample_data.copy()
        temp['pupuk_N'] = val if nutrient_type == 'N' else base_n
        temp['pupuk_P'] = val if nutrient_type == 'P' else base_p
        temp['pupuk_K'] = val if nutrient_type == 'K' else base_k
        
        y_pred = rf_model.predict(temp)[0]
        profit = calculate_profit(y_pred, temp['pupuk_N'].values[0], temp['pupuk_P'].values[0], temp['pupuk_K'].values[0])
        profits.append(profit)
    return profits

# Proses Kalkulasi
profits_N = get_profit_curve('N', N_range, base_N, base_P, base_K)
profits_P = get_profit_curve('P', P_range, base_N, base_P, base_K)
profits_K = get_profit_curve('K', K_range, base_N, base_P, base_K)

# Mencari Titik Optimal
opt_N = N_range[np.argmax(profits_N)]
opt_P = P_range[np.argmax(profits_P)]
opt_K = K_range[np.argmax(profits_K)]

max_profit_N = max(profits_N)
max_profit_P = max(profits_P)
max_profit_K = max(profits_K)

# ======================
# KARTU REKOMENDASI (METRICS)
# ======================
st.write("---")
st.markdown("### 🏆 Rekomendasi Dosis Optimal per Hektar")
st.caption("Titik puncak dimana penambahan pupuk menghasilkan selisih profit tertinggi sebelum *Law of Diminishing Returns* terjadi.")

m1, m2, m3 = st.columns(3)
with m1:
    st.metric(label="Dosis Optimal N", value=f"{opt_N:.0f} kg", delta=f"Max Profit: Rp {max_profit_N/1e6:.1f} Juta", delta_color="off")
with m2:
    st.metric(label="Dosis Optimal P", value=f"{opt_P:.0f} kg", delta=f"Max Profit: Rp {max_profit_P/1e6:.1f} Juta", delta_color="off")
with m3:
    st.metric(label="Dosis Optimal K", value=f"{opt_K:.0f} kg", delta=f"Max Profit: Rp {max_profit_K/1e6:.1f} Juta", delta_color="off")

st.write("---")

# ======================
# VISUALISASI PLOTLY (PREMIUM INTERACTIVE)
# ======================
st.markdown("### 📈 Kurva Analisis Profitabilitas")

def create_plotly_chart(x_data, y_data, opt_val, max_profit, title, color_hex, x_label):
    fig = go.Figure()
    
    # Garis Kurva
    fig.add_trace(go.Scatter(
        x=x_data, y=y_data, 
        mode='lines', 
        name='Profit',
        line=dict(color=color_hex, width=3),
        fill='tozeroy', 
        fillcolor=f"rgba{tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}"
    ))
    
    # Titik Optimal
    fig.add_trace(go.Scatter(
        x=[opt_val], y=[max_profit],
        mode='markers+text',
        name='Optimal',
        marker=dict(color='red', size=10, symbol='star'),
        text=[f"Optimal: {opt_val:.0f} kg"],
        textposition="top center",
        textfont=dict(color='white')
    ))
    
    # Garis Vertikal Optimal
    fig.add_vline(x=opt_val, line_dash="dash", line_color="red", opacity=0.7)
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Profit (Rupiah)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        hovermode="x unified"
    )
    
    fig.update_yaxes(gridcolor='#333333', zerolinecolor='#333333')
    fig.update_xaxes(gridcolor='#333333', zerolinecolor='#333333')
    
    return fig

# Tampilkan grafik dalam 3 kolom
c1, c2, c3 = st.columns(3)

with c1:
    fig_n = create_plotly_chart(N_range, profits_N, opt_N, max_profit_N, "Kurva Nitrogen (N)", "#4FC3F7", "N (kg/ha)")
    st.plotly_chart(fig_n, use_container_width=True)

with c2:
    fig_p = create_plotly_chart(P_range, profits_P, opt_P, max_profit_P, "Kurva Fosfor (P)", "#81C784", "P (kg/ha)")
    st.plotly_chart(fig_p, use_container_width=True)

with c3:
    fig_k = create_plotly_chart(K_range, profits_K, opt_K, max_profit_K, "Kurva Kalium (K)", "#BA68C8", "K (kg/ha)")
    st.plotly_chart(fig_k, use_container_width=True)