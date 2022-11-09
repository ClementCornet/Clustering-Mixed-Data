from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import gower

def process(df, k):
    numerical_columns = df.select_dtypes('number').columns
    categorical_columns = df.select_dtypes('object').columns
    categorical_indexes = []

    scaler = StandardScaler()

    for c in categorical_columns:
        categorical_indexes.append(df.columns.get_loc(c))

    # create a copy of our data to be scaled
    df_scale = df.copy()

    # standard scale numerical features
    for c in numerical_columns:
        df_scale[c] = scaler.fit_transform(df[[c]])

    g_mat = gower.gower_matrix(df_scale)
    
    clusters = AgglomerativeClustering(k,affinity='precomputed',linkage='average').fit_predict(
        g_mat
    )
    df['cluster'] = clusters#.astype(str)
    return df