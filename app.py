import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from streamlit_extras.stateful_button import button

st.title("🎆Clustering Model with K-Means on Web-Application💻")

def clean_data(df):
    with st.spinner("Data is processing,  ⏰ Please wait..."):
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
        cleaned_df = df
        return cleaned_df

def run_clustering(cleaned_df, n_clusters):
    st.subheader("📊 WCSS Graph for find proper K")
    with st.spinner("The data is clustering,  ⏰ Please wait..."):
        wcss = []
        for i in range(2, 10):
            model = KMeans(n_clusters=i, init='k-means++')
            model.fit(cleaned_df.values)
            wcss.append(model.inertia_)
        fig, ax = plt.subplots()
        plt.figure(figsize=(6, 4), dpi=150)
        ax.plot(range(2, 10), wcss)
        ax.set_title('The Elbow Method')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)
        return model,cleaned_df
    
def remodeling(cleaned_df, n_clusters):
    st.write("Select Number of Clusters first")
#     n_clusters = st.slider("",2, 10, 2)
    number = st.number_input('Insert a number')
    with st.spinner("Remodeling,  ⏰ Please wait..."):
        model2 = KMeans(n_clusters=number, init='k-means++')
        model2.fit(cleaned_df.values)
        score = silhouette_score(cleaned_df, model2.labels_)
        st.markdown(f'Silhouette Score: {score:.2f}','with proper K =',number)
        # สร้างตัวเลือก feature ที่เป็น checkbox
        features = st.multiselect('Select up to 2 features', options=cleaned_df.columns.tolist(), key='feature_selection', default=cleaned_df.columns.tolist()[:2])
        # กรองข้อมูลเฉพาะ feature ที่เลือก
        filtered_df = cleaned_df[features]
        # สร้างกราฟ
#         fig, ax = plt.subplots()
#         fig = plt.figure(figsize=(6, 4), dpi=150)
#         ax.scatter(filtered_df.iloc[:, 0], filtered_df.iloc[:, 1], c=model.labels_)
#         ax.set_xlabel(features[0])
#         ax.set_ylabel(features[1])
#         ax.set_title('Clusters')
#         st.pyplot(fig)
#         st.balloons()
 
def main():
    tab1, tab2, tab3 = st.tabs(["Upload file", "Result of Clustering", "Evaluation"])
    with tab1:
        # อัพโหลดไฟล์ csv
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        # อ่านไฟล์ csv และแสดงตัวอย่างข้อมูล
        if uploaded_file is None:
            st.write("Warning:")
            st.warning("Please upload a file.")

        else: 
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            st.write("Example uploaded file:")
            st.write(df.head(10))
            st.write(f"Data have total {df.shape[0]} rows")

    with tab2:
#         st.subheader("Cleaned Dataset:")
        if uploaded_file is not None: 
           cleaned_df = clean_data(df)
           n_clusters = 0
           model, _ = run_clustering(cleaned_df, n_clusters)
        else: 
           st.warning("Please upload data first.")
        
    with tab3: 
        st.subheader("Re-modeling")
        
        if uploaded_file is not None:
           st.markdown("**:blue[Remark : Choose K from the Elbow graph that results in the slowest decrease in SSE (elbow point)]** ")
           st.markdown("**:blue[This will be the appropriate number of K for clustering the data.]** ")
           st.divider()
           remodeling(cleaned_df, n_clusters)
        else: 
           st.warning("Please upload data first.")

            

if __name__ == '__main__':
    main()
