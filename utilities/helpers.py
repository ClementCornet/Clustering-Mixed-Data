from sklearn.manifold import SpectralEmbedding
import streamlit as st
import pandas as pd
from types import NoneType
import prince
import plotly.express as px


import algos.UMAP_HDBSCAN.umap_hdbscan as umap_hdbscan

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import silhouette_score
import gower
import umap

import utilities.generate_dataset

from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import numpy as np

#import sys
#from pandas_profiling import ProfileReport
#import streamlit_pandas_profiling as stpp



def choose_data():
    """ Choose between Sample Datasets or user-uploaded """
    data_choice = st.radio('Select Data', ['Existing Data', 'Upload Own Data', 'Generated Dataset'])
    df = pd.DataFrame()
    if data_choice == 'Upload Own Data':
        up = st.file_uploader('Upload File')
        st.session_state['truth'] = False
        if up:
            df = pd.read_csv(up)
            st.dataframe(df)

            
    if data_choice == 'Existing Data':
        select = st.selectbox('Choose Existing Dataset', [' --- Choose Dataset --- ',
                                                            'Heart Failure Short',
                                                            'Heart Failure Long',
                                                            'Contraceptive Method Choice',
                                                            'Churn',
                                                            'Penguins'])
        if select != ' --- Choose Dataset --- ':
            filename = '_'.join([word.lower() for word in select.split()])
            df = pd.read_csv(f'data/{filename}.csv',sep=None).dropna()
            try:
                st.dataframe(df)
            except:
                st.empty()

    
    if data_choice == 'Generated Dataset':
        
        gen_k = st.number_input('Length of data to generate', min_value=10, value=200, step=1)
        n_clusters = st.slider('Number of clusters to generate:',2,9)
        c1,c2 = st.columns([1,1])
        n_num = c1.slider('Numerical Features:',1,10,2)
        n_cat = c2.slider('Categorical Features:',1,10,2)
        d1,d2 = st.columns([1,1])
        cat_unq = d1.slider('Distinct values for categorical features',2,5,4)
        clust_std = d2.slider('Standard Deviation of the Clusters', 1.0,10.0,0.1)
        df = utilities.generate_dataset.generate(n_rows=gen_k,
                                                n_clusters=n_clusters,
                                                n_cat=n_cat,
                                                n_num=n_num,
                                                cat_unique=cat_unq,
                                                cluster_std=clust_std)
        st.dataframe(df)    

    if df.shape != (0,0):
        numerical_columns = df.select_dtypes('number').columns
        for col in numerical_columns:
            if df[col].nunique() < 5:
                df[col] = df[col].astype('object')

    #report = ProfileReport(df.reset_index())
    #stpp.st_profile_report(report)
    
    return df

def true_clusters(df):
    """Let the user define if a column represent 'true clusters'. 
    Auto for generated datasets"""
    if min(df.shape) > 0 :
        if 'truth' in df.columns:
            return df.pop('truth')
        opt = ['No']
        opt.extend(list(df.columns))
        choice = st.selectbox('Use a column as true clusters?', opt)
        if choice != 'No':
            st.write(f'Number of clusters : {df[choice].nunique()}')
            return df.pop(choice)
    return pd.DataFrame()

def select_columns(df):
    """Let the user define which columns to use or not"""
    if min(df.shape) > 0 :
        cols = st.multiselect('Use columns :',list(df.columns),list(df.columns))
        return df[cols]
    return df

def display_process(func, df):
    """Wrapper, to display the 'Process Data' section on an algorithm page

        Parameters:
            func (callable): function taking a DataFrame as an argument, returning it with a 'cluster' column
            df (pandas DataFrame): DataFrame to process clustering on
    """

    if type(df) == NoneType:
        st.empty()
        return
    
    # Let the user choose the number of cluster (cluster size for density based algorithms)
    k = 2
    if func != umap_hdbscan.process:
        k = st.select_slider('Choose number of clusters :', options=list(range(2,10)))
    else:
        k = st.select_slider('Minimum cluster size :', options=list(range(2,int(df.shape[0]/2))))
    
    func(df,k) # DATA PROCESSING HERE

    # Display each cluster's population
    # TO CLEAN
    dfcol = pd.DataFrame(df['cluster']
                    .value_counts()).transpose()
    st.dataframe(dfcol.reindex(sorted(dfcol.columns), axis=1))


    # Representing clusters
    #FAMD_Plot(df)
    #if st.button('UMAP'):
    #    UMAP_Plot(df)
    
    tab1, tab2, tab3 = st.tabs(["FAMD","UMAP","Laplacian Eigenmaps"])
    with tab1:
        FAMD_Plot(df)
    with tab2:
        UMAP_Plot(df)
    with tab3:
        Laplacian_Eigenmaps(df)

    # Univariate Data Exploration
    opt = [' --- Show Distribution of a variable --- ']
    opt.extend(list(df.columns))
    col = st.selectbox('',opt)
    if col in df.columns:
        fig = ''
        if df[col].dtype != 'object':
            fig = px.histogram(df, x=col, barmode='group',color='cluster', marginal="box")
        else:
            fig = px.histogram(df, x=col, color='cluster', barmode='group')
        fig.update_layout(title=f'Distribution of {col} in each cluster')
        st.plotly_chart(fig)


def FAMD_Plot(df):
    """ FAMD plot of clustered data, to diplay our results using dimensionality reduction.
    Each points cluser is represented by its color """

    if 'cluster' not in df.columns:
        return

    df['cluster'] = df['cluster'].astype(str) # To not have color gradient next to the plot
    famd = prince.FAMD(n_components=3) # 3 components = 3D, to plot
    famd = famd.fit(df.iloc[:,:-1]) # Last column is clusters, so it must not affect FAMD coordinates (just color)
    reduced = famd.row_coordinates(df.iloc[:,:-1]) # Get coordinates of each row
    reduced.columns = ['X','Y','Z']
    reduced['cluster'] = df['cluster']

    # Each axe's inertia
    labs = {
        "X" : f"Component 0 - ({round(100*famd.explained_inertia_[0],2)}% inertia)",
        "Y" : f"Component 1 - ({round(100*famd.explained_inertia_[1],2)}% inertia)",
        "Z" : f"Component 2 - ({round(100*famd.explained_inertia_[2],2)}% inertia)",
    }

    tot_inertia = f"{round(100*famd.explained_inertia_.sum(),2)}"
    st.write(f'FAMD Visualization of Clusters ({tot_inertia}%) :')
    fig = px.scatter_3d(reduced, 
                    x='X',y='Y',z='Z',
                    color='cluster',
                    labels=labs)
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    st.plotly_chart(fig)

def evaluation_indices(df, truth):
    """ Represent Evaluation indices for our clustered data.

    Internal Indices: Silouhette Score using Gower's Distance
    External Indices: Adjusted Rand Index (ARI) and Adjusted Mutual Info (AMI).
    Note that External Indices are processed only if a 'thruth' column ad been set."""
    if df.shape == (0,0):
        st.empty()
        return

    if 'cluster' not in df.columns:
        return

    g_mat = gower.gower_matrix(df.loc[:,df.columns != 'cluster'])
    silouhette = silhouette_score(
                 g_mat, 
                 df['cluster'],
                 metric="precomputed")
    #st.write(f"Silouhette Score (using Gower's Distance) : {silouhette}")
    d = {"Silouhette Score (using Gower's Distance)" : silouhette}

    if truth.shape != (0,0):
        #st.write('External Indices :')
        ARI = adjusted_rand_score(truth,df['cluster'])
        AMI = adjusted_mutual_info_score(truth,df['cluster'])
        #st.write(f"ARI : {ARI}")
        #st.write(f"AMI : {AMI}")
        d['ARI'] = ARI
        d['AMI'] = AMI

        indices = pd.DataFrame(d.items())
        indices.columns = ['Index', 'Value']
        st.dataframe(indices)

        df2 = df.copy()
        df2['cluster'] = truth

        #if st.button('See Real Clusters'):
        #    FAMD_Plot(df2)
        tab1, tab2, tab3 = st.tabs(["FAMD","UMAP","Laplacian Eigenmaps"])
        with tab1:
            FAMD_Plot(df2)
        with tab2:
            UMAP_Plot(df2)
        with tab3:
            Laplacian_Eigenmaps(df2)


def UMAP_Plot(df,n_components=3, intersection=False):
    """ UMAP plot of clustered data, to diplay our results using dimensionality reduction.
    Each points cluser is represented by its color """

    if df.shape == (0,0):
        st.empty()
        return

    if 'cluster' not in df.columns:
        return


    df2 = df.iloc[:,:-1] # Do not use clusters' column to compute UMAP coordinates

    # Scaling
    numerical = df2.select_dtypes(exclude='object')
    for c in numerical.columns:
        numerical[c] = (numerical[c] - numerical[c].mean())/numerical[c].std(ddof=0)
        
    ##preprocessing categorical
    categorical = df2.select_dtypes(include='object')
    categorical = pd.get_dummies(categorical)
    #Embedding numerical & categorical
    fit1 = umap.UMAP(random_state=12,
                    n_components=n_components).fit(numerical)
    fit2 = umap.UMAP(metric='dice', 
                    n_neighbors=250,
                    n_components=n_components).fit(categorical)
    # intersection will resemble the numerical embedding more.
    if intersection:
        embedding = fit1 * fit2

    # union will resemble the categorical embedding more.
    else:
        embedding = fit1 + fit2


    um = pd.DataFrame(embedding.embedding_) # Each points' UMAP coordinate 

    # Actual Plotting
    um.columns = ['X','Y','Z']
    um['cluster'] = df['cluster']
    fig = px.scatter_3d(um, 
                    x='X',y='Y',z='Z',
                    color='cluster')
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    st.write('UMAP plot of Clusters :')
    st.plotly_chart(fig)

def Laplacian_Eigenmaps(df):
    numerical = df.select_dtypes('number')
    categorical = df.select_dtypes('object')

    # Scaling
    scaler = StandardScaler()
    numerical = scaler.fit_transform(numerical)
    categorical = categorical.apply(lambda x: x.replace(x.unique(),list(range(1,1+len(x.unique())))))

    # Gamma parameter to compute pairwise distances
    gamma = np.mean(np.std(numerical))/2

    # Compute pairwise distance matrix
    distances = (cdist(numerical,numerical,'euclidean')) + cdist(categorical,categorical,'matching')*gamma
    distances = np.nan_to_num(distances)
    for i in range(len(distances[0])):
        for j in range(len(distances)):
            distances[i][j] = 1-distances[i][j]
    distances = np.nan_to_num(distances)

    

    ###### LAPLACIAN EMBEDDINGS
    lap = SpectralEmbedding(3,affinity="precomputed").fit_transform(np.interp(distances, (distances.min(), distances.max()), (0, +1)))
    #st.write(lap)
    lap_df = pd.DataFrame(lap)
    lap_df.columns = ['X','Y','Z']
    lap_df['cluster'] = df['cluster']

    fig = px.scatter_3d(lap_df,x='X',y='Y',z='Z',color='cluster')
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    st.write('Laplacian Eigenmaps Embeddings')
    st.plotly_chart(fig)