import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# st.write("Clustering Model with K-Means on Web-Application")
st.set_page_config(page_title="Clustering Model with K-Means", page_icon=":clipboard:", layout="wide")
st.sidebar.title("Clustering Model with K-Means")
# อัพโหลดไฟล์ csv
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# อ่านไฟล์ csv และแสดงตัวอย่างข้อมูล
if uploaded_file is None:
    st.warning("Please upload a file.")
else: 
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    st.write("Uploaded file:")
    st.write(df.head())
    X = df.iloc[:, 1:].values  # assuming the first column is the ID and will not be used in clustering
    sc = StandardScaler()
    X = sc.fit_transform(X)
    kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    df["Cluster"] = y_kmeans
    st.write("Cluster assignments:")
    st.dataframe(df[["ID", "Cluster"]])
    fig, ax = plt.subplots()
    for i in range(5):
        ax.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], label=f"Cluster {i+1}")
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, marker="X", label="Centroids")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("K-Means Clustering")
    ax.legend()
    st.write(fig)
