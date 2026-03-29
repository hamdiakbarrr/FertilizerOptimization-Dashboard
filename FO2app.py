import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "notebook"

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

# SMART DEFAULT DATA
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

if rf_model is None:
    st.stop()

# HEADER UTAMA
st.markdown('<p class="premium-title">🌱 AI Fertilizer Optimization</p>', unsafe_allow_html=True)
st.markdown('<p class="premium-subtitle">Simulasi Dosis Pupuk Maksimal untuk Profitabilitas Kebun</p>', unsafe_allow_html=True)

# CONTROL PANEL INTERAKTIF
with st.expander("⚙️ Control Panel: Parameter & Harga", expanded=True):
    st.markdown("Sesuaikan kondisi lapangan dan harga pasar saat ini:")
    
    col_env, col_limit, col_price = st.columns(3)
    
    with col_env:
        st.caption("🌍 Kondisi Lingkungan")
        populasi = st.number_input("Populasi (Pohon/ha)", 100, 200, 143)
        umur = st.number_input("Umur Tanaman (Tahun)", 1, 30, 10)
        curah_hujan = st.number_input("Curah Hujan Tahunan (mm)", 1000, 4000, 2500)
        harga_jual = st.number_input("Harga Jual TBS (Rp/kg)", 1000, 4000, 2500, step=100)
        
    with col_limit:
        # BUG FIX: Mengubah Baseline menjadi Batas Minimal agar slider berfungsi dan P=0 teratasi
        st.caption("🛡️ Batas Minimal Pupuk (kg/ha)")
        min_N = st.slider("Minimal Nitrogen (N)", 0, 400, 50)
        min_P = st.slider("Minimal Fosfor (P)", 0, 400, 50)
        min_K = st.slider("Minimal Kalium (K)", 0, 400, 50)
        
    with col_price:
        st.caption("💰 Harga Pupuk (Rp/kg)")
        harga_N = st.number_input("Harga N (Urea/ZA)", 5000, 20000, 12000, step=500)
        harga_P = st.number_input("Harga P (TSP/RP)", 5000, 20000, 10000, step=500)
        harga_K = st.number_input("Harga K (MOP/KCl)", 5000, 20000, 9000, step=500)

sample_data['populasi_ha'] = populasi
sample_data['umur_tanaman'] = umur
sample_data['curah_hujan'] = curah_hujan

def calculate_profit(yield_pred, n, p, k):
    biaya = (n * harga_N) + (p * harga_P) + (k * harga_K)
    revenue = yield_pred * harga_jual
    return revenue - biaya

# LANGKAH 1: OPTIMASI GLOBAL CEPAT
# BUG FIX: Menggunakan 20 titik agar Streamlit tidak lag (8.000 iterasi, bukan 125.000)
N_grid = np.linspace(min_N, 300, 20)
P_grid = np.linspace(min_P, 300, 20)
K_grid = np.linspace(min_K, 300, 20)

nn, pp, kk = np.meshgrid(N_grid, P_grid, K_grid, indexing='ij')

grid_df = pd.DataFrame({
    'pupuk_N': nn.ravel(),
    'pupuk_P': pp.ravel(),
    'pupuk_K': kk.ravel()
})

for col in sample_data.columns:
    if col not in ['pupuk_N', 'pupuk_P', 'pupuk_K']:
        grid_df[col] = sample_data[col].values[0]

grid_df = grid_df[sample_data.columns]

grid_df['yield_pred'] = rf_model.predict(grid_df)
grid_df['profit'] = calculate_profit(grid_df['yield_pred'], grid_df['pupuk_N'], grid_df['pupuk_P'], grid_df['pupuk_K'])

best_idx = grid_df['profit'].idxmax()
opt_N = grid_df.loc[best_idx, 'pupuk_N']
opt_P = grid_df.loc[best_idx, 'pupuk_P']
opt_K = grid_df.loc[best_idx, 'pupuk_K']

# --- Ambil Yield Prediksi Optimal ---
opt_yield = grid_df.loc[best_idx, 'yield_pred'] 
global_max_profit = grid_df.loc[best_idx, 'profit']

# LANGKAH 2: GENERATE KURVA HALUS (1D)
# Memastikan titik optimal absolut masuk ke dalam array grafik agar kurva tidak meleset
N_range = np.sort(np.unique(np.append(np.linspace(min_N, 300, 50), opt_N)))
P_range = np.sort(np.unique(np.append(np.linspace(min_P, 300, 50), opt_P)))
K_range = np.sort(np.unique(np.append(np.linspace(min_K, 300, 50), opt_K)))

def get_profit_curve_vectorized(nutrient_col, val_range, best_n, best_p, best_k):
    temp_df = pd.concat([sample_data] * len(val_range), ignore_index=True)
    temp_df['pupuk_N'] = best_n
    temp_df['pupuk_P'] = best_p
    temp_df['pupuk_K'] = best_k
    temp_df[nutrient_col] = val_range
    
    y_preds = rf_model.predict(temp_df)
    profits = calculate_profit(y_preds, temp_df['pupuk_N'], temp_df['pupuk_P'], temp_df['pupuk_K'])
    return profits.tolist()

profits_N = get_profit_curve_vectorized('pupuk_N', N_range, opt_N, opt_P, opt_K)
profits_P = get_profit_curve_vectorized('pupuk_P', P_range, opt_N, opt_P, opt_K)
profits_K = get_profit_curve_vectorized('pupuk_K', K_range, opt_N, opt_P, opt_K)

# KARTU REKOMENDASI (METRICS)
st.write("---")
st.markdown("### 🏆 Rekomendasi Dosis & Prediksi Hasil per Hektar")
st.caption("Titik puncak dimana penambahan pupuk menghasilkan selisih profit tertinggi.")

# --- BARU: Menjadi 4 Kolom agar Yield ikut tampil ---
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(label="Dosis Optimal N", value=f"{opt_N:.0f} kg")
with m2:
    st.metric(label="Dosis Optimal P", value=f"{opt_P:.0f} kg")
with m3:
    st.metric(label="Dosis Optimal K", value=f"{opt_K:.0f} kg")
with m4:
    # Menampilkan prediksi Yield dalam warna hijau
    st.metric(label="Prediksi Yield (Kg/ha/tahun)", value=f"{opt_yield:,.0f} kg", delta="Max Output", delta_color="normal")

st.write("---")

# VISUALISASI PLOTLY (PREMIUM INTERACTIVE)
st.markdown("### 📈 Kurva Analisis Profitabilitas")

def create_plotly_chart(x_data, y_data, opt_val, max_profit, title, color_hex, x_label):
    fig = go.Figure()
    
    # BUG FIX: Konversi Hex ke RGBA yang lebih aman dari error parsing Python
    hex_c = color_hex.lstrip('#')
    r, g, b = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
    fill_color_safe = f"rgba({r}, {g}, {b}, 0.1)"
    
    fig.add_trace(go.Scatter(
        x=x_data, y=y_data, 
        mode='lines', 
        name='Profit',
        line=dict(color=color_hex, width=3),
        fill='tozeroy', 
        fillcolor=fill_color_safe
    ))
    
    fig.add_trace(go.Scatter(
        x=[opt_val], y=[max_profit],
        mode='markers+text',
        name='Optimal',
        marker=dict(color='red', size=10, symbol='star'),
        text=[f"Optimal: {opt_val:.0f} kg"],
        textposition="top center",
        textfont=dict(color='white')
    ))
    
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

c1, c2, c3 = st.columns(3)

with c1:
    fig_n = create_plotly_chart(N_range, profits_N, opt_N, global_max_profit, "Kurva Nitrogen (N)", "#4FC3F7", "N (kg/ha)")
    st.plotly_chart(fig_n, use_container_width=True)

with c2:
    fig_p = create_plotly_chart(P_range, profits_P, opt_P, global_max_profit, "Kurva Fosfor (P)", "#81C784", "P (kg/ha)")
    st.plotly_chart(fig_p, use_container_width=True)

with c3:
    fig_k = create_plotly_chart(K_range, profits_K, opt_K, global_max_profit, "Kurva Kalium (K)", "#BA68C8", "K (kg/ha)")
    st.plotly_chart(fig_k, use_container_width=True)

from fpdf import FPDF
from datetime import datetime
import tempfile
import plotly.io as pio
import os

# --- KODE WAJIB ANTI-HANG STREAMLIT CLOUD ---
# Letakkan 2 baris ini di bagian paling atas app.py kamu (setelah import)
pio.kaleido.scope.mathjax = None
pio.kaleido.scope.chromium_args = tuple([
    "--headless", "--no-sandbox", "--disable-gpu", 
    "--single-process", "--disable-dev-shm-usage"
])

# Tambahkan semua variabel yang dibutuhkan ke dalam parameter (dalam tanda kurung)
def generate_pdf_report(fig1, fig2, fig3, umur, curah_hujan, populasi, harga_jual, opt_N, opt_P, opt_K, opt_yield, global_max_profit):
    pdf = FPDF()
    pdf.add_page()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(BASE_DIR, "aalilogo.png") 
    
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=15, y=10, w=30) 
        
    pdf.set_y(15) 
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(46, 125, 50)
    pdf.cell(0, 10, "Laporan Rekomendasi Pemupukan AI", ln=True, align='C')
    
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(100, 100, 100)
    tanggal_cetak = datetime.now().strftime("%d %B %Y - %H:%M WIB")
    pdf.cell(0, 5, f"Dicetak pada: {tanggal_cetak}", ln=True, align='C')
    pdf.ln(15) 
    
    # --- BAGIAN 1: KONDISI LAPANGAN ---
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "1. Parameter Lapangan & Ekonomi:", ln=True)
    pdf.set_font("Arial", '', 11)
    
    w_label = 45 
    w_colon = 5  
    
    pdf.cell(10); pdf.cell(w_label, 6, "- Umur Tanaman"); pdf.cell(w_colon, 6, ":"); pdf.cell(0, 6, f"{umur} Tahun", ln=True)
    pdf.cell(10); pdf.cell(w_label, 6, "- Curah Hujan"); pdf.cell(w_colon, 6, ":"); pdf.cell(0, 6, f"{curah_hujan} mm/tahun", ln=True)
    pdf.cell(10); pdf.cell(w_label, 6, "- Populasi Tanaman"); pdf.cell(w_colon, 6, ":"); pdf.cell(0, 6, f"{populasi} Pokok/ha", ln=True)
    pdf.cell(10); pdf.cell(w_label, 6, "- Harga Jual TBS"); pdf.cell(w_colon, 6, ":"); pdf.cell(0, 6, f"Rp {harga_jual:,.0f} / kg", ln=True)
    pdf.ln(5)
    
    # --- BAGIAN 2: REKOMENDASI ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "2. Rekomendasi Dosis & Prediksi AI:", ln=True)
    pdf.set_font("Arial", '', 11)
    
    pdf.cell(10); pdf.cell(w_label, 6, "- Pupuk Nitrogen (N)"); pdf.cell(w_colon, 6, ":"); pdf.cell(0, 6, f"{opt_N:.0f} kg/ha", ln=True)
    pdf.cell(10); pdf.cell(w_label, 6, "- Pupuk Fosfor (P)"); pdf.cell(w_colon, 6, ":"); pdf.cell(0, 6, f"{opt_P:.0f} kg/ha", ln=True)
    pdf.cell(10); pdf.cell(w_label, 6, "- Pupuk Kalium (K)"); pdf.cell(w_colon, 6, ":"); pdf.cell(0, 6, f"{opt_K:.0f} kg/ha", ln=True)
    
    pdf.set_text_color(46, 125, 50) 
    pdf.cell(10); pdf.cell(w_label, 6, "- Prediksi Produksi"); pdf.cell(w_colon, 6, ":"); pdf.cell(0, 6, f"{opt_yield:,.0f} kg/ha", ln=True)
    pdf.set_text_color(0, 0, 0) 
    pdf.ln(5)
    
    # --- BAGIAN 3: FINANSIAL ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "3. Proyeksi Finansial per Hektar:", ln=True)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(211, 47, 47)
    pdf.cell(10); pdf.cell(0, 10, f"Estimasi Profit Maksimal: Rp {global_max_profit:,.0f}", ln=True)
    
    # --- BAGIAN 4: INSERT GAMBAR GRAFIK ---
    pdf.add_page() 
    
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=15, y=10, w=20) 
        pdf.set_y(15)
        
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "4. Kurva Analisis Profitabilitas (N, P, K):", ln=True, align='C')
    pdf.ln(5)

    # Trik Pemanasan Kaleido (Mencegah Deadlock)
    try:
        dummy_fig = go.Figure()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as dummy_file:
            dummy_fig.write_image(dummy_file.name, format="png", engine="kaleido")
        os.unlink(dummy_file.name)
    except Exception:
        pass # Abaikan jika pemanasan gagal, biarkan proses berlanjut
        
    # --- Baru dilanjut dengan kode aslimu di bawah ini ---
    for idx, fig in enumerate([fig1, fig2, fig3]):
        # ... kode aslimu ...
        
    for idx, fig in enumerate([fig1, fig2, fig3]):
        fig_copy = go.Figure(fig) 
        
        fig_copy.update_layout(
            plot_bgcolor='white', 
            paper_bgcolor='white', 
            font=dict(color='black', size=14), 
            margin=dict(l=80, r=30, t=40, b=50), 
            title_font=dict(size=16)
        )
        
        fig_copy.update_yaxes(showticklabels=True, color='black', gridcolor='#E0E0E0', zerolinecolor='#9E9E9E')
        fig_copy.update_xaxes(showticklabels=True, color='black', gridcolor='#E0E0E0', zerolinecolor='#9E9E9E')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig_copy.write_image(tmpfile.name, format="png", width=800, height=350, scale=1, engine="kaleido") 
            pdf.image(tmpfile.name, x=15, w=180)
            pdf.ln(3) 
            
        os.unlink(tmpfile.name)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, "Laporan ini di-generate otomatis oleh AI Fertilizer Optimization Engine.", ln=True, align='C')
    
    return pdf.output(dest='S').encode('latin-1')

# TOMBOL DOWNLOAD DI STREAMLIT
st.markdown("### 📄 Export Laporan Eksekutif")
st.write("Unduh hasil kalkulasi beserta grafik kurva profitabilitas dalam format PDF formal.")

# 1. Tombol pemicu untuk mulai membuat PDF
if st.button("Buat Dokumen PDF"):
    # Munculkan animasi loading agar user tahu sistem sedang bekerja
    with st.spinner("Menyiapkan dokumen PDF... Proses ini memakan waktu beberapa detik."):
        try:
            # PANGGIL FUNGSINYA DI SINI
            # Penting: Pastikan nama variabel di bawah ini (fig_N, umur_input, dll) 
            # sesuai dengan nama variabel yang ada di kodemu!
            pdf_bytes = generate_pdf_report(
                fig1=fig_n, # Ganti dengan variabel grafik N kamu
                fig2=fig_p, # Ganti dengan variabel grafik P kamu
                fig3=fig_k, # Ganti dengan variabel grafik K kamu
                umur=umur,         # Ganti dengan variabel input umur
                curah_hujan=curah_hujan,   # Ganti dengan variabel input curah hujan
                populasi=populasi,   # Ganti dengan variabel input populasi
                harga_jual=harga_jual,      # Ganti dengan variabel input harga
                opt_N=opt_N,       # Ganti dengan output N optimal
                opt_P=opt_P,       # Ganti dengan output P optimal
                opt_K=opt_K,       # Ganti dengan output K optimal
                opt_yield=opt_yield,  # Ganti dengan output prediksi panen
                global_max_profit=global_max_profit # Ganti dengan output profit maksimal
            )
            
            # Simpan hasil PDF ke dalam memori sementara (session_state)
            st.session_state['laporan_pdf_siap'] = pdf_bytes
            st.success("Tadaaa! Dokumen PDF berhasil dibuat dan siap diunduh!")
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membuat PDF: {e}")

# 2. Tampilkan tombol Download HANYA JIKA PDF sudah selesai dibuat di atas
if 'laporan_pdf_siap' in st.session_state:
    st.download_button(
        label="⬇️ Unduh Laporan PDF Sekarang",
        data=st.session_state['laporan_pdf_siap'],
        file_name="Laporan_Rekomendasi_Pemupukan_AI.pdf",
        mime="application/pdf"
    )

# ---FOOTER---
st.caption('Dashboard dikembangkan oleh Ham D Roger v1.1 - 2026 | Powered by Machine Learning')
