import pandas as pd
from denseclus import DenseClus
import numpy as np

import streamlit as st

def process(df, k):
    clf = DenseClus(
        umap_combine_method="intersection_union_mapper",
        min_cluster_size=k
    )
    clf.fit(df)

    df['cluster'] = clf.score()

    
    return df