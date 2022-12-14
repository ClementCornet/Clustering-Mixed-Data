import streamlit as st

import pandas as pd
import numpy as np
import plotly.express as px

import algos.KPrototype.kproto as kproto
import algos.Hierarchical_Gower.hierar_gower as hierar_gower
import algos.FAMD_KMeans.famd_kmeans as famd_kmeans
import algos.Kamila.kamila as kamila
import algos.ModhaSpangler.msclust as msclust
import algos.MixtComp.mixtcomp as mixtcomp
import algos.UMAP_HDBSCAN.umap_hdbscan as umap_hdbscan
import algos.Spectral.spectral as spectral

import utilities.comparisons as comparisons
import utilities.helpers as helpers

from types import NoneType


def go_to_page(algo_name):
    """App's navigation, go to user chosen page"""
    if algo_name == ' --- Choose Algorithm --- ':
        land_page()
    if algo_name == 'K-Prototype':
        algo_page(algo_name, kproto.process, 'algos\KPrototype\kproto.md')
    if algo_name == 'Hierarchical with Gower Distance':
        algo_page(algo_name, hierar_gower.process, 'algos\Hierarchical_Gower\hierar_gower.md')
    if algo_name == 'K-Means using FAMD':
        algo_page(algo_name, famd_kmeans.process, r'algos\FAMD_KMeans\famd_kmeans.md')
    if algo_name == 'Kamila':
        algo_page(algo_name, kamila.process, 'algos\Kamila\kamila.md')
    if algo_name == 'Modha-Spangler':
        algo_page(algo_name, msclust.process, 'algos\ModhaSpangler\msclust.md')
    if algo_name == 'MixtComp':
        algo_page(algo_name, mixtcomp.process, 'algos\MixtComp\mixtcomp.md')
    if algo_name == 'UMAP-HDBSCAN':
        algo_page(algo_name, umap_hdbscan.process, r'algos\UMAP_HDBSCAN\umap_hdbscan.md')
    if algo_name.startswith('Spectral'):
        algo_page(algo_name, spectral.process, 'algos\Spectral\spectral.md')
    if algo_name == 'Comparisons':
        comparisons.page()


def algo_page(algo_name, process_func, mdpath):
    """Wrapper, how each algorithm's page displays
    
        Parameters:
            algo_name: Algorithm's Name, to display a title on the page
            process_func: Function that processes Data
            mdpath: Path to a Markdown file, explaining the page's algorithm. (Relative path from app.py)
    """
    st.title(algo_name) # TITLE

    # Description Section with the Markdown file
    with st.expander('Description'):
        st.markdown(
            open(mdpath,'r').read(),
            unsafe_allow_html=True
        )
    
    # Let uer Choose Data to process
    with st.expander('Choose Data'):
        st.write('Choose between already existing and user uploaded data')
        df = helpers.choose_data()
        truth = helpers.true_clusters(df)
        df = helpers.select_columns(df)

    # Actual Data Processing
    with st.expander('Process data'):
        if df.shape != (0,0):
            helpers.display_process(process_func, df)
            if type(df) != NoneType:
                st.download_button("Download Results",
                                df.to_csv(index=False),
                                f"{'_'.join(algo_name.split())}.csv",
                                "text/csv", 
                                key="download-csv")
        else:
            st.empty()
    
    # Evaluation Indices
    with st.expander('Evaluation indices'):
        helpers.evaluation_indices(df, truth)




def land_page():
    """Content of App's Landing Page Describing the methodology used to compare different algorithms"""

    st.title('Mixed Clustering Algorithms') # Title

    # Explain visualizations (FAMD/UMAP)
    st.write('We implemented 8 mixed clustering algorithms to analyze and compare their results over differents datasets.')
    st.write("""To visualize the different clusters, we use dimensionality reduction algorithms such as 
                    FMAD and UMAP to plot our clusters in 3D.""")

    toy = pd.read_csv('data/heart_failure_short.csv')
    toy['cluster'] = toy['age'].apply(lambda _: np.random.randint(0,2)) # Note that clusters on landing page are random
    helpers.FAMD_Plot(toy)
    st.write("""FAMD (Factorial Analysis of Mixed Data) is a combination of PCA for numerical variables and
                MCA for categorical variables. We use it to depict as much inertia as possible on only 3 axes.""")
    helpers.UMAP_Plot(toy)
    st.write("""UMAP (Uniform Manifol Approximation and Projection) uses a graph layout algorithm (like t-SNE) 
                to arrange data in low dimensional spaces. It uses euclidean distance for numerical variables and
                Dice distance for categorical variables.""")


    # Univariate Analysis plot
    st.write("We then plot the distributions of the the variables for each cluster.")
    fig = px.histogram(toy, x='age',color='cluster',barmode='group',marginal="box")
    st.plotly_chart(fig)


    # Evaluation Indices Methodology
    st.write("""Finally, with use internal and external evaluation indices to evaluate and compare our 
                clustering Results""")
    st.write("""The Silouhette Score is, for each point the difference between average intra and extra cluster distances
                divided by the larger of those 2.
                Here, we use Gower's Distance.""")

    true_clust = pd.read_csv('data/heart_failure_short.csv')
    true_clust['cluster'] = true_clust['DEATH_EVENT']
    helpers.evaluation_indices(toy,true_clust['cluster'])

    st.write("""We also use the Adjusted Rand Index (ARI) and the Adjusted Mutual Index (AMI) to evaluate the similarity
                between 2 clustering results, or between 1 clustering result and an 'true clustering' series.
                Both calculation are based on the number of pairs of element grouped/separated in both clustering series.
                ARI should be used when clusters have similar sizes, and AMI when the clusters are unbalanced.""")
    
