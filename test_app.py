import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# st.set_page_config(page_title="Clustering Model with K-Means", page_icon=":clipboard:", layout="wide")
# st.sidebar.title("Setting Plane")
st.title("Clustering Model with K-Means on Web-Application 💻")

# อัพโหลดไฟล์ csv
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])


# อ่านไฟล์ csv และแสดงตัวอย่างข้อมูล
if uploaded_file is None:
    st.write("Warning:")
    st.warning("Please upload a file.")
else: 
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    st.write("Uploaded file:")
    st.write(df.head())
    st.write(f"Data have {df.shape[0]} rows")
    #คลีนข้อมูล
    if st.button('Make data to clean🧹'):
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df.drop(['CustomerID','Document Date'], axis=1,inplace=True)
        le = LabelEncoder()
        df['Bussiness model (B2B,B2C)'] = le.fit_transform(df['Bussiness model (B2B,B2C)'])
        df['Channel'] = le.fit_transform(df['Channel'])
        df['SKU'] = le.fit_transform(df['SKU'])
        df['Product_type'] = le.fit_transform(df['Product_type'])
        df['Order Quantity (Item)'] = df['Order Quantity (Item)'].str.replace(',', '').astype(int)
        df['Total Value'] = df['Total Value'].str.replace(',', '').astype(float).round().astype(int)
        st.write("Cleaned Dataset:")
        st.write(df.head())
        st.write(f"Data have {df.shape[0]} rows")
#         st.write(df.dtypes)
#         if st.button('Step2 : Plotting WCSSS Graph'):
        st.write("Program is Running,Please wait.....")  
        wcss = []
        for i in range(1, 10):
            model = KMeans(n_clusters=i, init='k-means++', random_state=0)
            model.fit(df.values)
            wcss.append(model.inertia_)
        fig, ax = plt.subplots()
        st.write("WCSS Graph for find proper K and re-modeling")
        ax.plot(range(1, 10), wcss)
        ax.set_title('The Elbow Method')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)
#         else:
#             st.write("Warning:")
#             st.warning("Please find proper K and re-modeling")
    else: 
        st.write("Warning:")
        st.warning("Please Cleansing data first ")

