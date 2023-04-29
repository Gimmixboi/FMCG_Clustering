import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
        cleaned_df = df
        ss = StandardScaler()
        cleaned_df = pd.DataFrame(ss.fit_transform(cleaned_df))
        return cleaned_df,le,df

def run_clustering(cleaned_df):
    st.divider()
    st.subheader("📊 WCSS Graph for find proper K")
    with st.spinner("The data is clustering,  ⏰ Please wait..."):
        wcss = []
        for i in range(2, 10):
            model = KMeans(n_clusters=i, init='k-means++')
            model.fit(cleaned_df.values)
            wcss.append(model.inertia_)
        fig, ax1 = plt.subplots()
        plt.figure(figsize=(6, 4), dpi=150)
        ax1.plot(range(2, 10), wcss)
        ax1.set_title('The Elbow Method')
        ax1.set_xlabel('Number of clusters')
        ax1.set_ylabel('WCSS')
        pct_change = 100 * (np.diff(wcss) / wcss[:-1])
        for i, (x, y) in enumerate(zip(range(2, 8), wcss)):
            ax1.annotate(f"{i+2}\n({pct_change[i]:.1f}%)", xy=(x, y), xytext=(x+0.1, y+0.1), fontsize=8)
        st.pyplot(fig)
  
def remodeling(cleaned_df):
    number = st.number_input("Select Proper Number of Clusters first to re-model",min_value=2,max_value=8,value=2)
    st.markdown("**:blue[Remark : Choose K from the Elbow graph that results in the slowest decrease in SSE (elbow point)]** ")
    st.markdown("**:blue[This will be the appropriate number of K for clustering the data.]** ")
    st.divider()
    with st.spinner("Remodeling,  ⏰ Please wait..."):
        model2 = KMeans(n_clusters=number, init='k-means++')
        model2.fit(cleaned_df.values)
        score = silhouette_score(cleaned_df, model2.labels_)
        st.subheader("Evaluation")
        st.write(f'Silhouette Score: **:red[{score:.2f}]**','with proper K =',number)
        st.markdown(":blue[Remak: silhouette score of 0 means our model did not work very well. The worse could be -1, but the best can go upto 1] ")
    return cleaned_df,model2.labels_
        
def result(cleaned_df, le, df, cluster_labels):        
    labeldf = df.assign(cluster_labels=cluster_labels)
    st.write(labeldf.sample(30))   

    labeldf['cluster_labels'] = labeldf['cluster_labels'].astype(str)
    labeldf = labeldf.sort_values('cluster_labels')
    fig2, ax3 = plt.subplots(figsize=(14,10))
    sns.histplot(data=labeldf, x='cluster_labels', ax=ax3)
    ax3.set_title("Labeled Histogram")
    for i, v in enumerate(labeldf['cluster_labels'].unique()):
        count = labeldf[labeldf['cluster_labels']==v]['cluster_labels'].count()
        ax3.text(i, count, str(count), ha='center', fontsize=10, fontweight='bold')
    st.pyplot(fig2)
    st.divider()

    st.subheader('Clustering Results')
    features = st.multiselect('Select up to 2 features', options=df.columns.tolist(), key='feature_selection', default=df.columns.tolist()[:2],max_selections=2)
    filtered_df = df[features]
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(filtered_df)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    x = le.inverse_transform(filtered_df[features[0]])
    y = le.inverse_transform(filtered_df[features[1]])
    sns.scatterplot(data=pca_data, x=x, y=y, hue=cluster_labels, ax=ax, palette='deep')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])

    st.pyplot(fig)
    
def main():
    tab1, tab2, tab3,tab4 = st.tabs(["Upload file", "WCSS Graph", "Evaluation","🔮Final Result"])
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
           cleaned_df, le, df = clean_data(df)
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
           result(cleaned_df, le, df, cluster_labels)
        else: 
           st.warning("Please upload data first.")         

if __name__ == '__main__':
    main()
