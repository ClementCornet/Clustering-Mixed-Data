import pandas as pd
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
from kmodes.kprototypes import euclidean_dissim
import numpy as np

def process(df,k):
    numerical_columns = df.select_dtypes('number').columns
    categorical_columns = df.select_dtypes('object').columns
    categorical_indexes = []

    scaler = StandardScaler()

    for c in categorical_columns:
        categorical_indexes.append(df.columns.get_loc(c))

    if len(numerical_columns) == 0 or len(categorical_columns) == 0:
        return

    # create a copy of our data to be scaled
    df_scale = df.copy()

    # standard scale numerical features
    for c in numerical_columns:
        df_scale[c] = scaler.fit_transform(df[[c]])

    # we set the number of clusters to k
    kproto = KPrototypes(n_clusters=k,
                        num_dissim=euclidean_dissim,
                        random_state=0)

    kproto.fit_predict(df_scale, categorical= categorical_indexes)

    # add clusters to dataframe
    df["cluster"] = kproto.labels_#.astype(str)

    return df


def Manhattan_distance(point1, point2):
    return np.sum(np.absolute(point1 - point2))