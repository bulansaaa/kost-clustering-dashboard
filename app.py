import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="Dashboard Kos Clustering",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("hasil_cluster.csv")
df_pca = pd.read_csv("hasil_visualisasi.csv")

# =========================
# HEADER
# =========================
st.title("🏠 Dashboard Analisis Kos Jabodetabek")
st.markdown("### Clustering Segmen Pasar: Ekonomis - Standar - Premium")
st.markdown("---")

# =========================
# RINGKASAN DATA
# =========================
st.subheader("Ringkasan Dataset")

col1, col2, col3 = st.columns(3)

col1.metric("Total Data", len(df))
col2.metric("Total Kolom", len(df.columns))
col3.metric(
    "Cluster Unik",
    df["Segmen_Pasar"].nunique() if "Segmen_Pasar" in df.columns else 0
)

st.dataframe(df.head())

st.markdown("---")

# =========================
# DATA VISUALISASI
# =========================
silhouette_score_value = 0.7215

cluster_counts = df["Segmen_Pasar"].value_counts()

harga_cluster = pd.DataFrame({
    "Segmen_Pasar": ["Ekonomis", "Standar", "Premium"],
    "Rata_Rata_Harga": [1050000, 1550000, 2300000]
})

ukuran_cluster = pd.DataFrame({
    "Segmen_Pasar": ["Ekonomis", "Standar", "Premium"],
    "Rata_Rata_Ukuran": [17.2, 13.8, 15.4]
})

# =========================
# VISUALISASI 2x2
# =========================

# Baris 1
col1, col2 = st.columns(2)

with col1:
    st.subheader("Evaluasi Model Clustering")

    st.metric("Silhouette Score", f"{silhouette_score_value:.4f}")
    st.progress(silhouette_score_value)

    if silhouette_score_value >= 0.71:
        st.success("Hasil clustering sangat baik.")
    elif silhouette_score_value >= 0.51:
        st.info("Hasil clustering cukup baik.")
    else:
        st.warning("Hasil clustering kurang optimal.")

    # keterangan metode singkat
    with st.expander(" Detail Metode"):
        st.write("""
        **Algoritma Final:** Agglomerative Clustering (Ward Linkage)

        Metode ini menggabungkan data secara bertahap berdasarkan kemiripan.
        Ward Linkage dipilih karena mampu meminimalkan variansi dalam cluster,
        sehingga menghasilkan kelompok yang lebih homogen.
        """)
        
        st.write("""
        **Interpretasi Score 0.7215:**
        - Cohesion tinggi (anggota cluster mirip)
        - Separation tinggi (jarak antar cluster jelas)
        - Overlap kecil
        """)

with col2:
    st.subheader("Distribusi Segmen Pasar")

    st.bar_chart(cluster_counts)
    st.dataframe(cluster_counts)

# Baris 2
col3, col4 = st.columns(2)

with col3:
    st.subheader("Perbandingan Harga Rata-rata")

    c1, c2, c3 = st.columns(3)

    for i, row in harga_cluster.iterrows():
        with [c1, c2, c3][i]:
            st.metric(
                row["Segmen_Pasar"],
                f"Rp {row['Rata_Rata_Harga']:,.0f}"
            )

    st.bar_chart(
        harga_cluster,
        x="Segmen_Pasar",
        y="Rata_Rata_Harga"
    )

    termurah = harga_cluster.loc[
        harga_cluster["Rata_Rata_Harga"].idxmin()
    ]["Segmen_Pasar"]

    termahal = harga_cluster.loc[
        harga_cluster["Rata_Rata_Harga"].idxmax()
    ]["Segmen_Pasar"]

    st.success(f"Murah: {termurah} | Mahal: {termahal}")

with col4:
    st.subheader("Perbandingan Ukuran Kamar")

    c1, c2, c3 = st.columns(3)

    for i, row in ukuran_cluster.iterrows():
        with [c1, c2, c3][i]:
            st.metric(
                row["Segmen_Pasar"],
                f"{row['Rata_Rata_Ukuran']:.1f} m²"
            )

    st.bar_chart(
        ukuran_cluster,
        x="Segmen_Pasar",
        y="Rata_Rata_Ukuran"
    )

    terbesar = ukuran_cluster.loc[
        ukuran_cluster["Rata_Rata_Ukuran"].idxmax()
    ]["Segmen_Pasar"]

    terkecil = ukuran_cluster.loc[
        ukuran_cluster["Rata_Rata_Ukuran"].idxmin()
    ]["Segmen_Pasar"]

    st.success(f"Terbesar: {terbesar} | Terkecil: {terkecil}")

st.markdown("---")



# =========================
# VISUALISASI PCA + DISTRIBUSI HARGA
# =========================
st.markdown("---")
st.subheader("Analisis Cluster dan Distribusi Harga")

col_pca1, col_pca2 = st.columns(2)

# =========================
# KOLOM KIRI = PCA
# =========================
with col_pca1:
    st.markdown("### Visualisasi Cluster PCA")

    fig, ax = plt.subplots(figsize=(8,6))

    for label in ["Ekonomis", "Standar", "Premium"]:
        sub = df_pca[df_pca["Segmen_Pasar"] == label]

        ax.scatter(
            sub["PCA1"],
            sub["PCA2"],
            label=label,
            alpha=0.5,
            s=20
        )

    ax.set_title("Visualisasi Cluster dengan PCA")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend()
    ax.grid(alpha=0.3)

    st.pyplot(fig)


# =========================
# KOLOM KANAN = DISTRIBUSI HARGA
# =========================
with col_pca2:
    st.markdown("### Distribusi Harga Kos")

    # bersihkan format harga
    harga_clean = (
        df["Harga"]
        .astype(str)
        .str.replace("Rp", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    # ubah ke numeric
    harga_numeric = pd.to_numeric(harga_clean, errors="coerce")

    # ubah ke juta rupiah
    harga_juta = harga_numeric / 1_000_000

    fig2, ax2 = plt.subplots(figsize=(8,6))

    ax2.hist(
        harga_juta.dropna(),
        bins=30,
        alpha=0.8
    )

    ax2.set_title("Distribusi Harga Kost")
    ax2.set_xlabel("Harga Kost (Juta Rupiah)")
    ax2.set_ylabel("Jumlah Kost")
    ax2.grid(alpha=0.3)

    st.pyplot(fig2)

    st.info("""
    Histogram menunjukkan persebaran harga kos pada seluruh dataset.
    Puncak batang menunjukkan rentang harga yang paling banyak tersedia.
    """)

# =========================
# INTERPRETASI PCA
# =========================
st.markdown("---")
st.subheader("Interpretasi Hasil PCA")

st.info("""
**Hasil visualisasi PCA menunjukkan bahwa data kos berhasil dikelompokkan ke dalam tiga segmen utama:**

**Ekonomis**  
Cluster ini terlihat cukup terpisah dan menyebar lebih luas, menunjukkan variasi karakteristik pada kos dengan harga rendah.

**Standar**  
Cluster berada di area tengah dengan persebaran yang cukup rapat, menandakan keseimbangan antara harga, ukuran, dan fasilitas.

**Premium**  
Cluster premium cenderung mengelompok pada area tertentu dengan pola lebih padat, menunjukkan karakteristik fasilitas yang lebih seragam dan eksklusif.

**Kesimpulan:**  
Secara umum, overlap antar cluster relatif kecil, sehingga segmentasi dapat dikatakan cukup baik. 
Hal ini didukung oleh nilai **Silhouette Score sebesar 0.7215**, yang menunjukkan kualitas clustering kuat.
""")

# =========================
# FILTER DATA
# =========================
st.subheader("Filter Data Kos")

cluster_filter = st.selectbox(
    "Pilih Segmen",
    df["Segmen_Pasar"].unique()
)

filtered_df = df[df["Segmen_Pasar"] == cluster_filter]

st.dataframe(filtered_df.head(20))

st.markdown("---")

# =========================
# INSIGHT CLUSTER
# =========================
st.subheader("Insight Cluster")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Cluster Ekonomis**
    
    Memiliki harga paling rendah dengan ukuran kamar yang relatif lebih luas.
    Cocok untuk penyewa yang fokus pada efisiensi biaya.
    """)

    st.info("""
    **Cluster Standar**
    
    Berada pada harga menengah dengan fasilitas yang cukup lengkap.
    Menawarkan keseimbangan antara biaya dan kenyamanan.
    """)

with col2:
    st.info("""
    **Cluster Premium**
    
    Memiliki harga tertinggi dengan fasilitas paling lengkap dan lebih eksklusif.
    Cocok untuk penyewa yang mengutamakan kenyamanan maksimal.
    """)

    st.success("""
    **Kesimpulan**
    
    Hasil clustering menunjukkan bahwa pasar kos dapat dibagi menjadi tiga segmen utama:
    
    - **Ekonomis** → fokus harga murah  
    - **Standar** → keseimbangan harga dan fasilitas  
    - **Premium** → kenyamanan dan fasilitas lengkap  
    
    Dengan **Silhouette Score 0.7215**, segmentasi dinilai cukup baik dan didukung oleh visualisasi PCA yang menunjukkan pemisahan cluster yang jelas.
    """)

