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
# clean = st.sidebar.checkbox("Make data to clean🧹")
# อัพโหลดไฟล์ csv
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])


# อ่านไฟล์ csv และแสดงตัวอย่างข้อมูล
if uploaded_file is None:
    st.write("Warning:")
    st.warning("Please upload a file.")
else: 
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    clean = st.button('Make data to clean🧹')
if clean:
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop(['CustomerID'], axis=1,inplace=True)
    st.write("Uploaded file:")
    le = LabelEncoder()
    df['Bussiness model (B2B,B2C)'] = le.fit_transform(df['Bussiness model (B2B,B2C)'])
    df['Channel'] = le.fit_transform(df['Channel'])
    df['SKU'] = le.fit_transform(df['SKU'])
    df['Product_type'] = le.fit_transform(df['Product_type'])
    st.write(df.head())
    st.write(df.dtypes)
else: 
    st.write("Warning:")
    st.warning("Please Cleansing data first ")
