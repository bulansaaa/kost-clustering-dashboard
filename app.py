import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="Dashboard Kos Clustering",
    layout="wide"
)

# =========================
# HEADER
# =========================
st.title("🏠 Dashboard Analisis Kos Jabodetabek")
st.markdown("### Clustering Segmen Pasar: Ekonomis - Standar - Premium")
st.markdown("---")

# =========================
# FUNGSI BANTUAN
# =========================
def bersihkan_harga(series):
    """Mengubah kolom harga berformat 'Rp1.500.000' dll menjadi numerik."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace("Rp", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


@st.cache_data(show_spinner=False)
def load_default_data():
    df = pd.read_csv("hasil_cluster.csv")
    df_pca = pd.read_csv("hasil_visualisasi.csv")
    return df, df_pca


def jalankan_clustering(df_raw, kolom_fitur, n_clusters=3):
    """
    Menjalankan pipeline clustering lengkap:
    1. Bersihkan & siapkan fitur numerik
    2. Standardisasi
    3. KMeans
    4. PCA untuk visualisasi 2D
    5. Beri label segmen (Ekonomis/Standar/Premium) berdasarkan rata-rata harga
    """
    df = df_raw.copy()

    # Pastikan kolom Harga numerik (kalau ada)
    if "Harga" in df.columns:
        df["Harga_Numeric"] = bersihkan_harga(df["Harga"])
    elif "Harga_Numeric" not in df.columns:
        st.error("Kolom 'Harga' tidak ditemukan di data. Tambahkan kolom Harga untuk pelabelan segmen.")
        return None, None, None

    # Siapkan matriks fitur untuk clustering
    fitur_df = df[kolom_fitur].copy()
    for col in kolom_fitur:
        fitur_df[col] = pd.to_numeric(fitur_df[col], errors="coerce")

    # Buang baris yang fiturnya kosong semua / tidak valid
    mask_valid = fitur_df.notna().all(axis=1)
    fitur_df = fitur_df[mask_valid]
    df = df[mask_valid].reset_index(drop=True)
    fitur_df = fitur_df.reset_index(drop=True)

    if len(fitur_df) < n_clusters:
        st.error("Jumlah data valid terlalu sedikit untuk jumlah cluster yang dipilih.")
        return None, None, None

    # Standardisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(fitur_df)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df["Cluster"] = cluster_labels

    # Hitung silhouette score
    sil_score = silhouette_score(X_scaled, cluster_labels)

    # Beri nama segmen berdasarkan rata-rata harga tiap cluster
    rata_harga = df.groupby("Cluster")["Harga_Numeric"].mean().sort_values()
    urutan_cluster = rata_harga.index.tolist()

    if n_clusters == 3:
        nama_segmen = ["Ekonomis", "Standar", "Premium"]
    else:
        nama_segmen = [f"Segmen {i+1}" for i in range(n_clusters)]

    mapping_nama = {cl: nama_segmen[i] for i, cl in enumerate(urutan_cluster)}
    df["Segmen_Pasar"] = df["Cluster"].map(mapping_nama)

    # PCA untuk visualisasi 2D
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame({
        "PCA1": pca_result[:, 0],
        "PCA2": pca_result[:, 1],
        "Segmen_Pasar": df["Segmen_Pasar"]
    })

    return df, df_pca, sil_score


# =========================
# SUMBER DATA: UPLOAD ATAU DEFAULT
# =========================
st.sidebar.header("⚙️ Sumber Data & Pengaturan")

uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV kos baru",
    type=["csv"],
    help="File harus memiliki kolom numerik seperti Harga, Ukuran (m2), dll."
)

gunakan_default = uploaded_file is None

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.sidebar.success(f"File berhasil diupload: {uploaded_file.name} ({len(df_input)} baris)")
    except Exception as e:
        st.sidebar.error(f"Gagal membaca file: {e}")
        df_input = None
else:
    st.sidebar.info("Belum ada file diupload. Menggunakan data default (hasil_cluster.csv).")
    df_input = None

# =========================
# JIKA UPLOAD FILE BARU -> JALANKAN CLUSTERING
# =========================
if uploaded_file is not None and df_input is not None:
    st.sidebar.markdown("### Pilih Kolom Fitur untuk Clustering")

    kolom_numerik_tersedia = df_input.select_dtypes(include=[np.number]).columns.tolist()

    # Kalau kolom Harga masih berupa teks (misal "Rp1.500.000"), tetap tawarkan
    kolom_kandidat = list(dict.fromkeys(kolom_numerik_tersedia + [c for c in df_input.columns if "harga" in c.lower() or "ukuran" in c.lower()]))

    default_pilihan = [c for c in kolom_kandidat if c.lower() in ["harga", "ukuran (m2)", "ukuran"]]
    if not default_pilihan:
        default_pilihan = kolom_kandidat[:2] if len(kolom_kandidat) >= 2 else kolom_kandidat

    kolom_fitur = st.sidebar.multiselect(
        "Kolom yang dipakai untuk clustering",
        options=kolom_kandidat,
        default=default_pilihan
    )

    # Kalau kolom Harga dipilih tapi formatnya teks, bersihkan dulu sebelum dipakai sebagai fitur numerik
    if "Harga" in kolom_fitur:
        df_input["Harga"] = bersihkan_harga(df_input["Harga"])

    n_clusters = st.sidebar.slider("Jumlah Cluster", min_value=2, max_value=6, value=3)

    tombol_proses = st.sidebar.button("🚀 Jalankan Clustering", use_container_width=True)

    if tombol_proses:
        if len(kolom_fitur) < 1:
            st.sidebar.error("Pilih minimal 1 kolom fitur.")
            st.stop()

        with st.spinner("Memproses clustering..."):
            df, df_pca, silhouette_score_value = jalankan_clustering(df_input, kolom_fitur, n_clusters)

        if df is None:
            st.stop()

        st.session_state["df"] = df
        st.session_state["df_pca"] = df_pca
        st.session_state["silhouette_score_value"] = silhouette_score_value
        st.sidebar.success("Clustering berhasil dijalankan!")
    elif "df" not in st.session_state:
        st.info("⬅️ Atur kolom fitur dan klik **Jalankan Clustering** di sidebar untuk memproses file yang diupload.")
        st.stop()

    df = st.session_state.get("df")
    df_pca = st.session_state.get("df_pca")
    silhouette_score_value = st.session_state.get("silhouette_score_value")

    if df is None:
        st.stop()

else:
    # Pakai data default bawaan
    try:
        df, df_pca = load_default_data()
    except FileNotFoundError:
        st.warning("File default (hasil_cluster.csv / hasil_visualisasi.csv) tidak ditemukan. Silakan upload file CSV di sidebar.")
        st.stop()

    silhouette_score_value = 0.7215

    # Pastikan kolom Harga_Numeric tersedia
    if "Harga_Numeric" not in df.columns and "Harga" in df.columns:
        df["Harga_Numeric"] = bersihkan_harga(df["Harga"])

# =========================
# RINGKASAN DATA
# =========================
st.subheader("Ringkasan Dataset")

col1, col2, col3 = st.columns(3)

col1.metric("Total Data", len(df))
col2.metric("Total Kolom", len(df.columns))
col3.metric("Cluster Unik", df["Segmen_Pasar"].nunique())

st.dataframe(df.head())

st.markdown("---")

# =========================
# DATA VISUALISASI
# =========================
cluster_counts = df["Segmen_Pasar"].value_counts()

if "Harga_Numeric" not in df.columns:
    df["Harga_Numeric"] = np.nan

harga_cluster = (
    df.groupby("Segmen_Pasar")["Harga_Numeric"]
    .mean()
    .reset_index(name="Rata_Rata_Harga")
)

kolom_ukuran = "Ukuran (m2)" if "Ukuran (m2)" in df.columns else None
if kolom_ukuran:
    ukuran_cluster = (
        df.groupby("Segmen_Pasar")[kolom_ukuran]
        .mean()
        .reset_index(name="Rata_Rata_Ukuran")
    )

# =========================
# VISUALISASI 2x2
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Evaluasi Model Clustering")

    st.metric("Silhouette Score", f"{silhouette_score_value:.4f}")
    st.progress(min(max(silhouette_score_value, 0.0), 1.0))

    if silhouette_score_value >= 0.71:
        st.success("Hasil clustering sangat baik.")
    elif silhouette_score_value >= 0.51:
        st.info("Hasil clustering cukup baik.")
    else:
        st.warning("Hasil clustering kurang optimal.")

with col2:
    st.subheader("Distribusi Segmen Pasar")

    st.bar_chart(cluster_counts)
    st.dataframe(cluster_counts)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Perbandingan Harga Rata-rata")
    st.bar_chart(
        harga_cluster,
        x="Segmen_Pasar",
        y="Rata_Rata_Harga"
    )

with col4:
    st.subheader("Perbandingan Ukuran Kamar")
    if kolom_ukuran:
        st.bar_chart(
            ukuran_cluster,
            x="Segmen_Pasar",
            y="Rata_Rata_Ukuran"
        )
    else:
        st.info("Kolom 'Ukuran (m2)' tidak ditemukan pada data ini.")

st.markdown("---")

# =========================
# VISUALISASI PCA
# =========================
st.subheader("Analisis Cluster dan Distribusi Harga")

col_pca1, col_pca2 = st.columns(2)

with col_pca1:
    st.markdown("### Visualisasi Cluster PCA")

    fig, ax = plt.subplots(figsize=(8, 6))

    for label in df_pca["Segmen_Pasar"].unique():
        sub = df_pca[df_pca["Segmen_Pasar"] == label]
        ax.scatter(
            sub["PCA1"],
            sub["PCA2"],
            label=label,
            alpha=0.5,
            s=20
        )

    ax.set_title("Visualisasi Cluster dengan PCA")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)

with col_pca2:
    st.markdown("### Distribusi Harga Kost")

    harga_juta = df["Harga_Numeric"] / 1_000_000

    fig2, ax2 = plt.subplots(figsize=(8, 6))

    ax2.hist(
        harga_juta.dropna(),
        bins=30,
        alpha=0.8
    )

    ax2.set_title("Distribusi Harga Kost")
    ax2.set_xlabel("Harga (Juta Rupiah)")
    ax2.set_ylabel("Jumlah Kost")
    ax2.grid(alpha=0.3)

    st.pyplot(fig2)

st.markdown("---")

# =========================
# FILTER DATA
# =========================
st.subheader("Filter Data Kos")

cluster_filter = st.selectbox(
    "Pilih Segmen",
    df["Segmen_Pasar"].unique(),
    key="filter_segmen"
)

filtered_df = df[df["Segmen_Pasar"] == cluster_filter]

st.dataframe(filtered_df.head(20))

st.markdown("---")

# =========================
# DOWNLOAD HASIL
# =========================
st.subheader("Unduh Hasil Clustering")
st.download_button(
    label="⬇️ Download hasil_cluster.csv",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="hasil_cluster.csv",
    mime="text/csv"
)

st.markdown("---")

# =========================
# INSIGHT CLUSTER
# =========================
st.subheader("Insight Cluster")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Cluster Ekonomis**

    Harga lebih rendah dengan fasilitas dasar.
    Cocok untuk mahasiswa atau pekerja dengan budget terbatas.
    """)

    st.info("""
    **Cluster Standar**

    Menawarkan keseimbangan harga, ukuran, dan fasilitas.
    Cocok untuk penyewa yang mencari value terbaik.
    """)

with col2:
    st.info("""
    **Cluster Premium**

    Memiliki fasilitas lebih lengkap dengan harga lebih tinggi.
    Cocok untuk penyewa yang mengutamakan kenyamanan.
    """)

    st.success(f"""
    **Kesimpulan**

    Segmentasi pasar membagi kos menjadi beberapa segmen pasar.

    Dengan Silhouette Score **{silhouette_score_value:.4f}**, kualitas clustering dinilai
    {"sangat baik" if silhouette_score_value >= 0.71 else "cukup baik" if silhouette_score_value >= 0.51 else "kurang optimal"}.
    """)