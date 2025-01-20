import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from io import BytesIO

# Streamlit Page Configuration
st.set_page_config(page_title="GMM Clustering App", layout="wide")

# --- Helper Functions --- #
def preprocess_data(data, irrelevant_cols):
    """Preprocess the uploaded dataset."""
    data_cleaned = data.drop(columns=[col for col in irrelevant_cols if col in data.columns])

    if 'Date' in data_cleaned.columns:
        data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], errors='coerce').astype(int) / 10**9

    for col in data_cleaned.select_dtypes(include=['object']).columns:
        data_cleaned[col].fillna('Unknown', inplace=True)

    for col in data_cleaned.select_dtypes(include=['float64', 'int64']).columns:
        data_cleaned[col].fillna(data_cleaned[col].median(), inplace=True)

    label_encoder = LabelEncoder()
    for col in data_cleaned.select_dtypes(include=['object']).columns:
        data_cleaned[col] = label_encoder.fit_transform(data_cleaned[col])

    return data_cleaned

def plot_distribution(data, feature):
    """Plot the distribution of a specific feature."""
    fig, ax = plt.subplots(figsize=(10, 6))
    data[feature].hist(bins=30, alpha=0.7, color='blue', ax=ax)
    ax.set_title(f'Distribution of {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

def plot_heatmap(corr_matrix):
    """Plot the correlation heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True, ax=ax)
    ax.set_title("Heatmap of Feature Correlations")
    st.pyplot(fig)

def download_file(data):
    """Allow the user to download the processed data."""
    output = BytesIO()
    data.to_csv(output, index=False)
    processed_data = output.getvalue()
    return processed_data

# --- Streamlit App --- #
st.title("Gaussian Mixture Model (GMM) Clustering App")

# Step 1: Upload Dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Load dataset
    data = pd.read_csv(uploaded_file, delimiter=';')
    st.write(f"### Uploaded Dataset ({len(data)} rows):")
    st.write(data.head())

    # Step 2: Preprocess Dataset
    irrelevant_cols = st.sidebar.multiselect(
        "Columns to Exclude:", options=data.columns.tolist(), default=['Customer ID', 'Name', 'Surname', 'Birthdate', 'Merchant Name']
    )

    data_cleaned = preprocess_data(data, irrelevant_cols)
    st.write("### Cleaned Dataset:")
    st.write(data_cleaned.head())

    # Step 3: Visualizations
    st.sidebar.header("EDA Options")
    if st.sidebar.checkbox("Show Feature Distributions"):
        numeric_features = data_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()
        selected_feature = st.sidebar.selectbox("Select Feature", options=numeric_features)
        plot_distribution(data_cleaned, selected_feature)

    if st.sidebar.checkbox("Show Correlation Heatmap"):
        corr_matrix = data_cleaned.corr()
        plot_heatmap(corr_matrix)

    # Step 4: Scaling Data
    scaler = StandardScaler()
    data_numeric = data_cleaned.select_dtypes(include=['float64', 'int64'])
    data_scaled = scaler.fit_transform(data_numeric)

    # Step 5: Clustering with GMM
    st.sidebar.header("Clustering Options")
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)

    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    clusters = gmm.fit_predict(data_scaled)
    data_cleaned['Cluster'] = clusters

    # Step 6: Evaluation
    st.write("### Cluster Analysis")
    silhouette_avg = silhouette_score(data_scaled, clusters)
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")

    st.write("#### Cluster Summary:")
    st.write(data_cleaned.groupby('Cluster').mean())

    # Step 7: Visualization
    st.write("### PCA Visualization")
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)
    data_cleaned['PCA1'] = data_pca[:, 0]
    data_cleaned['PCA2'] = data_pca[:, 1]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data_cleaned, palette='Set2', legend='full', ax=ax)
    ax.set_title('PCA Visualization of Clusters')
    st.pyplot(fig)

    # Step 8: Download Results
    st.sidebar.header("Download Results")
    csv_data = download_file(data_cleaned)
    st.sidebar.download_button(label="Download Clustered Data", data=csv_data, file_name="gmm_clustering_results.csv", mime="text/csv")

else:
    st.write("Upload a CSV file to get started.")
