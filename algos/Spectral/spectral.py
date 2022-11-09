from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import SpectralClustering


def process(df, k):
    """Process Spectral Clustering
    
    Using K-prototype's distance (euclidean + gamma*matching) scaled as affinity matrix between 0 and 1"""

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

    # Perform Spectral Clustering using this distance
    clusters = SpectralClustering(n_clusters=k,affinity="precomputed").fit_predict(
        np.interp(distances, (distances.min(), distances.max()), (0, +1)))

    df['cluster'] = clusters
    
    return df