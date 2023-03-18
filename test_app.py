import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score

# st.write("Clustering Model with K-Means on Web-Application")
st.set_page_config(page_title="Clustering Model with K-Means", page_icon=":clipboard:", layout="wide")
st.sidebar.title("Clustering Model with K-Means")
# อัพโหลดไฟล์ csv
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# อ่านไฟล์ csv และแสดงตัวอย่างข้อมูล
if uploaded_file is None:
    st.warning("Please upload a file.")
else: 
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1)
    st.write("Uploaded file:")
    st.dataframe(df)
    st.write(df.head())
