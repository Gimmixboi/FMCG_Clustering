import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

def main():
    st.set_page_config(page_title="Clustering Model with K-Means", page_icon=":clipboard:", layout="wide")
    run_clustering()

if __name__ == "__main__":
    main()

def run_clustering():
    st.sidebar.title("Clustering Model with K-Means")
    # อัพโหลดไฟล์ csv
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    # อ่านไฟล์ csv และแสดงตัวอย่างข้อมูล
    if uploaded_file is not None:
        if uploaded_file.type == "application/vnd.ms-excel":
            df = pd.read_excel(uploaded_file, sheet_name=None)
            sheet_name = list(df.keys())[0]
            df = df[sheet_name]
        else:
            df = pd.read_csv(uploaded_file)
     
    selected_features = st.sidebar.multiselect("Select Features", df.columns)

    if len(selected_features) > 0:
        X = df[selected_features].values
        st.write("Data Shape:", X.shape)
    else:
        st.warning("Please select at least one feature.")

    if st.sidebar.button("Run Clustering"):
        if len(selected_features) == 0:
            st.warning("Please select at least one feature.")
        else:
            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
                kmeans.fit(X)
                wcss.append(kmeans.inertia_)

            st.set_option("deprecation.showPyplotGlobalUse", False)
            fig, ax = plt.subplots()
            ax.plot(range(1, 11), wcss)
            ax.set_title("Elbow Curve")
            ax.set_xlabel("Number of Clusters")
            ax.set_ylabel("WCSS")
            st.pyplot(fig)

            silhouette_scores = []
            for i in range(2, 11):
                kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
                kmeans.fit(X)
                score = silhouette_score(X, kmeans.labels_)
                silhouette_scores.append(score)

            fig, ax = plt.subplots()
            ax.plot(range(2, 11), silhouette_scores)
            ax.set_title("Silhouette Score")
            ax.set_xlabel("Number of Clusters")
            ax.set_ylabel("Score")
            st.pyplot(fig)

            k = st.number_input("Number of Clusters", min_value=2, max_value=10, value=3, step=1)
            kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
            y_kmeans = kmeans.fit_predict(X) 
            
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c="yellow", label="Centroids")
            plt.title("Clusters")
            plt.xlabel(selected_features[0])
            plt.ylabel(selected_features[1])
            plt.legend()
            st.pyplot()
