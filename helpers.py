import streamlit as st
import pandas as pd
from types import NoneType
import prince
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


import umap_hdbscan

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import silhouette_score
import gower
import umap


def choose_data():
    """ Choose between Sample Datasets or user-uploaded """
    data_choice = st.radio('Select Data', ['Existing Data', 'Upload Own Data'])
    df = pd.DataFrame()
    if data_choice == 'Upload Own Data':
        up = st.file_uploader('Upload File')
        st.session_state['truth'] = False
        if up:
            df = pd.read_csv(up)
            st.dataframe(df)
            #st.write(df.columns)
            #return df
            
    # TODO : update list of sample datasets with usable ones
    if data_choice == 'Existing Data':
        select = st.selectbox('Choose Existing Dataset', [' --- Choose Dataset --- ',
                                                            #'test', 
                                                            #'test again',
                                                            'Heart Failure Short',
                                                            'Heart Failure Long',
                                                            'Contraceptive Method Choice',
                                                            'Churn',
                                                            'Penguins'])
        if select != ' --- Choose Dataset --- ':
            filename = '_'.join([word.lower() for word in select.split()])
            df = pd.read_csv(f'{filename}.csv',sep=None).dropna()
            try:
                st.dataframe(df)
            except:
                st.empty()
            #st.write(df.columns)
            #truth_col = st.selectbox('Use a column as True Clusers?',)
            #return df

    if df.shape != (0,0):
        numerical_columns = df.select_dtypes('number').columns
        for col in numerical_columns:
            if df[col].nunique() < 5:
                df[col] = df[col].astype('object')
    return df

def true_clusters(df):
    if min(df.shape) > 0 :
        opt = ['No']
        opt.extend(list(df.columns))
        choice = st.selectbox('Use a column as true clusters?', opt)
        if choice != 'No':
            st.write(f'Number of clusters : {df[choice].nunique()}')
            return df.pop(choice)
    return pd.DataFrame()

def select_columns(df):
    if min(df.shape) > 0 :
        cols = st.multiselect('Use columns :',list(df.columns),list(df.columns))
        return df[cols]
    return df

def display_process(func, df):
    if type(df) == NoneType:
        st.empty()
        return
    k = 2
    if func != umap_hdbscan.process:
        k = st.select_slider('Choose number of clusters :', options=list(range(2,10)))
    else:
        k = st.select_slider('Minimum cluster size :', options=list(range(2,int(df.shape[0]/2))))
    #st.dataframe(func(df,k))
    func(df,k)

    dfcol = pd.DataFrame(df['cluster']
                    .value_counts()).transpose()

    
    st.dataframe(dfcol.reindex(sorted(dfcol.columns), axis=1))

    FAMD_Plot(df)

    if st.button('UMAP'):
        UMAP_Plot(df)
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
    """ FAMD plot of clustered data, to diplay our results using dimensionality reduction """

    if 'cluster' not in df.columns:
        return

    df['cluster'] = df['cluster'].astype(str)
    famd = prince.FAMD(n_components=3)
    famd = famd.fit(df.iloc[:,:-1])
    reduced = famd.row_coordinates(df.iloc[:,:-1])
    reduced.columns = ['X','Y','Z']
    reduced['cluster'] = df['cluster']

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
    if df.shape == (0,0):
        st.empty()
        return

    if 'cluster' not in df.columns:
        return

     
    #st.write("Internal Index : ")
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

        if st.button('See Real Clusters'):
            FAMD_Plot(df2)


def UMAP_Plot(df,n_components=3, intersection=False):

    if df.shape == (0,0):
        st.empty()
        return

    if 'cluster' not in df.columns:
        return


    df2 = df.iloc[:,:-1]

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


    um = pd.DataFrame(embedding.embedding_)

    um.columns = ['X','Y','Z']
    um['cluster'] = df['cluster']
    fig = px.scatter_3d(um, 
                    x='X',y='Y',z='Z',
                    color='cluster')
    fig.update_layout(showlegend=False)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    st.write('UMAP plot of Clusters :')
    st.plotly_chart(fig)
