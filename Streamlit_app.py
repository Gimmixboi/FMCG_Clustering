import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# อัพโหลดไฟล์ csv
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

# อ่านไฟล์ csv และแสดงตัวอย่างข้อมูล
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # สร้างกราฟ WCSS
    def wcss(k):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df)
        return kmeans.inertia_

    st.write('Elbow Curve')
    fig, ax = plt.subplots()
    k_range = range(1, 10)
    wcss_values = [wcss(k) for k in k_range]
    ax.plot(k_range, wcss_values)
    st.pyplot(fig)

    # กระบวนการ clustering
    st.write('Clustering Result')
    n_clusters = st.slider('Select Number of Clusters', min_value=2, max_value=10)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(df)

    # แสดงผลลัพธ์การ clustering บน scatter plot
    fig, ax = plt.subplots()
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c=pred_y)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red')
    st.pyplot(fig)

    # คำนวณค่า Silhouette score
    score = silhouette_score(df, pred_y)
    st.write('Silhouette Score:', score)
