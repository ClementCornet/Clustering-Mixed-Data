from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np

def generate(n_rows=100, n_clusters=None, n_cat=1, n_num=1, cat_unique=4, cluster_std=1):

    """
    Generate a Dataset From User input

    Parameters:
        n_rows: number of rows or individuals in the dataset
        n_clusters: number of clusters to generate
        n_cat: number of categorical variables
        n_num: number of numerical variables
        cat_unique: number of unique values for categorical variables
        cluster_std: Standard Deviation of the Clusters
        
    Returns:
        A DataFrame Generated with chosen parameters

    """

    X, y = make_blobs(n_samples=n_rows, centers=n_clusters, n_features=n_cat+n_num,cluster_std=cluster_std)
    
    X = np.absolute(X)
    df = pd.DataFrame(X)

    for i in range(n_cat):
        df.iloc[:,-i] = pd.qcut(df.iloc[:,-i],cat_unique,labels=False, duplicates='drop')

    df['truth'] = y
    
    return df