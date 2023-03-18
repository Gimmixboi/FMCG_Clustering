import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

#st.write("Clustering Model with K-Means on Web-Application")
st.set_page_config(page_title="Clustering Model with K-Means", page_icon=":clipboard:", layout="wide")
st.sidebar.title("Setting Plane")
# อัพโหลดไฟล์ csv
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# อ่านไฟล์ csv และแสดงตัวอย่างข้อมูล
if uploaded_file is None:
    st.write("Warning:")
    st.warning("Please upload a file.")
else: 
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
clean = st.sidebar.radio("Make data to clean?",
 ('No', 'Yes'))
if clean is "Yes":
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop(['CustomerID'], axis=1,inplace=True)
    st.write("Uploaded file:")
    st.write(df.head())
else: 
    st.write("Warning:")
    st.warning("Please Cleansing data first")
