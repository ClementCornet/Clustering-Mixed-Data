import prince
import pandas as pd
from sklearn.cluster import KMeans

import streamlit as st # FOR DEBUG

def process(df, k):

    famd = prince.FAMD(n_components=3)
    famd = famd.fit(df)
    reduced = famd.row_coordinates(df)

    km = KMeans(n_clusters=k)
    km.fit(reduced)
    df['cluster'] = km.labels_
    st.write('oui FMAD KMEANS ')
    return df