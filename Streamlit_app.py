import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def main():
    st.set_page_config(page_title="Clustering Model with K-Means", page_icon=":clipboard:", layout="wide")
    run_clustering()

if __name__ == "__main__":
    main()
