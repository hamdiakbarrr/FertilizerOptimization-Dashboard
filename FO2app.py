import pandas as pd
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# ======================
# LOAD MODEL DAN DATA
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "FO2rfmodel.pkl")

with open(MODEL_PATH, "rb") as f:
    rf_model = pickle.load(f)

csv_path = os.path.join(BASE_DIR, "DatasetKS_FO.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File CSV tidak ditemukan di {csv_path}")
df = pd.read_csv(csv_path)

features = [
    'umur_tanaman',
    'curah_hujan',
    'hari_hujan',
    'populasi_ha',
    'pupuk_N',
    'pupuk_P',
    'pupuk_K'
]

# Ambil rata-rata hanya kolom yang sama seperti fitur model
sample_data = df[features].mean(numeric_only=True).to_frame().T

st.title("Fertilizer Optimization for Profit")

# ======================
# INPUT INTERAKTIF
# ======================
st.sidebar.header("Input Baseline")
base_N = st.sidebar.slider("Baseline N", 0, 300, 150)
base_P = st.sidebar.slider("Baseline P", 0, 300, 100)
base_K = st.sidebar.slider("Baseline K", 0, 300, 150)

N_range = np.linspace(0, 300, 100)
P_range = np.linspace(0, 300, 100)
K_range = np.linspace(0, 300, 100)

umur = st.sidebar.slider("Umur Tanaman", 1, 25, 5)
curah_hujan = st.sidebar.slider("Curah Hujan", 1800, 3200, 2500)

base_features = {
    'umur_tanaman': umur,
    'curah_hujan': curah_hujan,
    'hari_hujan': 150,
    'populasi_ha': 143
}

# ======================
# HARGA PUPUK
# ======================
harga_N = 12000
harga_P = 10000
harga_K = 9000

# ======================
# FUNCTION HITUNG PROFIT
# ======================
def calculate_profit(yield_pred, n, p, k):
    harga_jual = 2500
    biaya = (n * harga_N) + (p * harga_P) + (k * harga_K)
    revenue = yield_pred * harga_jual
    return revenue - biaya

# ======================
# HITUNG PROFIT UNTUK N, P, K
# ======================
profits_N = []
for n in N_range:
    temp = sample_data.copy()
    temp['pupuk_N'] = n
    temp['pupuk_P'] = base_P
    temp['pupuk_K'] = base_K
    y_pred = rf_model.predict(temp)[0]
    profit = calculate_profit(y_pred, n, base_P, base_K)
    profits_N.append(profit)

profits_P = []
for p in P_range:
    temp = sample_data.copy()
    temp['pupuk_N'] = base_N
    temp['pupuk_P'] = p
    temp['pupuk_K'] = base_K
    y_pred = rf_model.predict(temp)[0]
    profit = calculate_profit(y_pred, base_N, p, base_K)
    profits_P.append(profit)

profits_K = []
for k in K_range:
    temp = sample_data.copy()
    temp['pupuk_N'] = base_N
    temp['pupuk_P'] = base_P
    temp['pupuk_K'] = k
    y_pred = rf_model.predict(temp)[0]
    profit = calculate_profit(y_pred, base_N, base_P, k)
    profits_K.append(profit)

# ======================
# PLOT
# ======================
fig, axes = plt.subplots(1, 3, figsize=(15,5))
fig.patch.set_facecolor('white')

# --- N ---
axes[0].plot(N_range, profits_N, color='blue', marker='o')
idx_opt_N = np.argmax(profits_N)
opt_N = N_range[idx_opt_N]
axes[0].axvline(opt_N, color='red', linestyle='--')
y_mid = (max(profits_N) + min(profits_N)) / 2
axes[0].text(opt_N + 5, y_mid, f"Opt N = {opt_N:.0f}", color='red', fontsize=10, va='center')
axes[0].set_title(f"Pupuk N\n(P={base_P}, K={base_K})")
axes[0].set_xlabel("N (kg/ha)")
axes[0].set_ylabel("Profit (Juta Rp)")
axes[0].yaxis.set_major_formatter(
    lambda x, pos: f"{int(str(int(x/1e6))[:2])}"
)

# --- P ---
axes[1].plot(P_range, profits_P, color='green', marker='o')
idx_opt_P = np.argmax(profits_P)
opt_P = P_range[idx_opt_P]
axes[1].axvline(opt_P, color='red', linestyle='--')
y_mid = (max(profits_P) + min(profits_P)) / 2
axes[1].text(opt_P + 5, y_mid, f"Opt P = {opt_P:.0f}", color='red', fontsize=10, va='center')
axes[1].set_title(f"Pupuk P\n(N={base_N}, K={base_K})")
axes[1].set_xlabel("P (kg/ha)")
axes[1].set_ylabel("Profit (Juta Rp)")
axes[1].yaxis.set_major_formatter(
    lambda x, pos: f"{int(str(int(x/1e6))[:2])}"
)

# --- K ---
axes[2].plot(K_range, profits_K, color='purple', marker='o')
idx_opt_K = np.argmax(profits_K)
opt_K = K_range[idx_opt_K]
axes[2].axvline(opt_K, color='red', linestyle='--')
y_mid = (max(profits_K) + min(profits_K)) / 2
axes[2].text(opt_K + 5, y_mid, f"Opt K = {opt_K:.0f}", color='red', fontsize=10, va='center')
axes[2].set_title(f"Pupuk K\n(N={base_N}, P={base_P})")
axes[2].set_xlabel("K (kg/ha)")
axes[2].set_ylabel("Profit (Juta Rp)")
axes[2].yaxis.set_major_formatter(
    lambda x, pos: f"{int(str(int(x/1e6))[:2])}"
)

plt.tight_layout()
st.pyplot(fig)

# ======================
# OUTPUT RINGKAS
# ======================
st.subheader("Rekomendasi Optimal")
st.write(f"""
- N optimal: **{opt_N:.0f} kg/ha**
- P optimal: **{opt_P:.0f} kg/ha**
- K optimal: **{opt_K:.0f} kg/ha**
""")