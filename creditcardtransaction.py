# --- Import Library --- #
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Pastikan matplotlib menggunakan backend yang sesuai
matplotlib.use("Agg")  # Non-interactive backend untuk Streamlit

# --- Streamlit Configuration --- #
st.set_page_config(
    page_title="Clustering dengan GMM",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar Input --- #
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])

n_clusters = st.sidebar.slider("Jumlah Cluster (n_clusters)", min_value=2, max_value=10, value=3, step=1)
random_state = st.sidebar.number_input("Random State", value=42, min_value=0)

# --- Fungsi Utama --- #
def main():
    if uploaded_file:
        # Membaca dataset
        st.title("Clustering dengan Gaussian Mixture Model (GMM)")
        st.write("Dataset berhasil diunggah. Berikut adalah beberapa baris dari dataset:")
        
        data = pd.read_csv(uploaded_file, delimiter=';')
        st.dataframe(data.head(10))
        
        # Membatasi data menjadi 500 baris
        data = data.head(500)
        
        # --- Data Cleaning --- #
        st.subheader("1. Data Cleaning")
        
        irrelevant_cols = ['Customer ID', 'Name', 'Surname', 'Birthdate']
        data_cleaned = data.drop(columns=[col for col in irrelevant_cols if col in data.columns], errors="ignore")
        
        if 'Date' in data_cleaned.columns:
            data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], errors='coerce').astype(int) / 10**9
        
        for col in data_cleaned.select_dtypes(include=['object']).columns:
            data_cleaned[col].fillna('Unknown', inplace=True)
        
        for col in data_cleaned.select_dtypes(include=['float64', 'int64']).columns:
            data_cleaned[col].fillna(data_cleaned[col].median(), inplace=True)
        
        label_encoder = LabelEncoder()
        for col in data_cleaned.select_dtypes(include=['object']).columns:
            data_cleaned[col] = label_encoder.fit_transform(data_cleaned[col])
        
        st.write("Data setelah dibersihkan:")
        st.dataframe(data_cleaned.head(10))
        
        # --- Scaling Data --- #
        scaler = StandardScaler()
        data_numeric = data_cleaned.select_dtypes(include=['float64', 'int64'])
        data_scaled = scaler.fit_transform(data_numeric)
        
        # --- GMM Clustering --- #
        st.subheader("2. Gaussian Mixture Model (GMM) Clustering")
        
        gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        clusters = gmm.fit_predict(data_scaled)
        data_cleaned['Cluster'] = clusters
        
        st.write(f"Hasil clustering dengan {n_clusters} cluster:")
        st.dataframe(data_cleaned.head(10))
        
        # --- Analisis Merchant --- #
        st.subheader("3. Analisis Transaksi Merchant")
        
        if 'Merchant Name' in data.columns and 'Transaction Amount' in data.columns:
            merchant_name = st.text_input("Masukkan Merchant Name:")
            
            if merchant_name:
                merchant_data = data[data['Merchant Name'] == merchant_name]
                total_transactions = len(merchant_data)
                total_amount = merchant_data['Transaction Amount'].sum()
                
                # Menentukan transaksi mencurigakan (contoh: cluster dengan transaksi tinggi)
                suspicious_transactions = merchant_data[merchant_data['Cluster'] == 2]  # Cluster mencurigakan
                
                st.write(f"Analisis untuk Merchant Name: {merchant_name}")
                st.write(f"Jumlah total transaksi: {total_transactions}")
                st.write(f"Total Transaction Amount: {total_amount}")
                st.write(f"Jumlah transaksi mencurigakan: {len(suspicious_transactions)}")
                
                if not suspicious_transactions.empty:
                    st.write("Detail transaksi mencurigakan:")
                    st.dataframe(suspicious_transactions[['Transaction Amount', 'Cluster']])
        
        # --- Visualisasi --- #
        st.subheader("4. Visualisasi Hasil Clustering")
        
        # Heatmap korelasi
        st.write("**Heatmap Korelasi:**")
        corr_matrix = data_cleaned.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
        # PCA untuk visualisasi 2D
        st.write("**PCA (Visualisasi 2D):**")
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_scaled)
        data_cleaned['PCA1'] = data_pca[:, 0]
        data_cleaned['PCA2'] = data_pca[:, 1]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(
            x='PCA1',
            y='PCA2',
            hue='Cluster',
            data=data_cleaned,
            palette='Set2',
            legend='full',
            ax=ax
        )
        ax.set_title("Hasil Clustering dengan PCA")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        st.pyplot(fig)

# --- Menjalankan Aplikasi --- #
if __name__ == "__main__":
    main()
