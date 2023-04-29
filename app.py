import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from streamlit_extras.stateful_button import button
from sklearn.decomposition import PCA

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
        st.write(df.head(10))
        st.write(f"Cleaned Data have total {df.shape[0]} rows")
        cleaned_df = df,
        return cleaned_df,le

def run_clustering(cleaned_df):
    st.divider()
    st.subheader("📊 WCSS Graph for find proper K")
    with st.spinner("The data is clustering,  ⏰ Please wait..."):
        wcss = []
        for i in range(2, 10):
            model = KMeans(n_clusters=i, init='k-means++')
            model.fit(cleaned_df[0].values)
            wcss.append(model.inertia_)
        fig, ax = plt.subplots()
        plt.figure(figsize=(6, 4), dpi=150)
        ax.plot(range(2, 10), wcss)
        ax.set_title('The Elbow Method')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)
    
def remodeling(cleaned_df):
    number = st.number_input("Select Proper Number of Clusters first to re-model",min_value=2,max_value=8,value=2)
    st.markdown("**:blue[Remark : Choose K from the Elbow graph that results in the slowest decrease in SSE (elbow point)]** ")
    st.markdown("**:blue[This will be the appropriate number of K for clustering the data.]** ")
    st.divider()
    with st.spinner("Remodeling,  ⏰ Please wait..."):
        cleaned_df, le = cleaned_df
        model2 = KMeans(n_clusters=number, init='k-means++')
        model2.fit(cleaned_df.values)
        score = silhouette_score(cleaned_df, model2.labels_)
        st.subheader("Evaluation")
        st.write(f'Silhouette Score: **:red[{score:.2f}]**','with proper K =',number)
    return cleaned_df,model2.labels_
        
def result(cleaned_df, cluster_labels,le):        
    labeldf = cleaned_df.assign(cluster_labels=cluster_labels)
    st.write(labeldf.sample(50))   

    # สร้างตัวเลือก feature ที่เป็น checkbox
    features = st.multiselect('Select up to 2 features', options=cleaned_df[0].columns.tolist(), key='feature_selection', default=cleaned_df[0].columns.tolist()[:2],max_selections=2)
    # กรองข้อมูลเฉพาะ feature ที่เลือก
    filtered_df = cleaned_df[0][features]

    # ลดมิติข้อมูลด้วย PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(filtered_df)

    # สร้างกราฟ
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # แสดง scatterplot ของ 2 features ที่เลือก
    x = le.inverse_transform(filtered_df[features[0]].values)
    y = le.inverse_transform(filtered_df[features[1]].values)
    sns.scatterplot(data=pca_data, x=x, y=y, hue=cluster_labels, ax=ax, palette='deep')
    # กำหนดชื่อแกน x, y และชื่อกราฟ
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title('Clustering Results')
    # แสดงกราฟ
    st.pyplot(fig)    

def main():
    tab1, tab2, tab3,tab4 = st.tabs(["Upload file", "Result of Clustering", "Evaluation","🔮Final Result"])
    with tab1:
        # อัพโหลดไฟล์ csv
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        # อ่านไฟล์ csv และแสดงตัวอย่างข้อมูล
        if uploaded_file is None:
            st.write("Warning:")
            st.warning("Please upload a file.")

        else: 
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            st.divider()
            st.write("Example uploaded file:")
            st.write(df.head(10))
            st.write(f"Data have total {df.shape[0]} rows")

    with tab2:
        st.subheader("Cleaned Dataset:")
        if uploaded_file is not None: 
           cleaned_df,le = clean_data(df)
           run_clustering(cleaned_df)
        else: 
           st.warning("Please upload data first.")
        
    with tab3: 
        st.subheader("Re-modeling")
        if uploaded_file is not None:
           cleaned_df, cluster_labels = remodeling(cleaned_df)
        else: 
           st.warning("Please upload data first.")
        
    with tab4: 
        st.subheader("Labeled Date frame : ")
        if uploaded_file is not None:
           result(cleaned_df, cluster_labels,le)
        else: 
           st.warning("Please upload data first.")         

if __name__ == '__main__':
    main()
