import prince
import pandas as pd
from sklearn.cluster import KMeans

def process(df, k):
    """Process K-Means of a Dataset's FAMD Coordinate"""

    # Get FAMD Coordinates
    famd = prince.FAMD(n_components=3) # Using 3 dimensions by default, could use more to maximize inertia
    famd = famd.fit(df)
    reduced = famd.row_coordinates(df)

    # Process Standard K-Means
    km = KMeans(n_clusters=k)
    km.fit(reduced)
    df['cluster'] = km.labels_
    
    return df