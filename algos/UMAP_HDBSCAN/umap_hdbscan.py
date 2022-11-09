from denseclus import DenseClus

def process(df, k):
    """Process HDBSCAN algorithm on dataset's UMAP coordinates.
    
    Based on Amazon-DenseClus python package"""
    
    clf = DenseClus(
        umap_combine_method="intersection_union_mapper",
        min_cluster_size=k
    )
    clf.fit(df)

    df['cluster'] = clf.score()

    
    return df