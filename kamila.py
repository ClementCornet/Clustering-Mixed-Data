import pandas as pd
import os
from subprocess import check_output

import streamlit as st
import helpers
import json

def process(df, k):

    numerical_columns = df.select_dtypes('number').columns
    categorical_columns = df.select_dtypes('object').columns
    df[categorical_columns].to_csv('temp_cat.csv',index=False)
    df[numerical_columns].to_csv('temp_continue.csv',index=False)
    json_data = { "n_clusters" : k}
    with open('k.json', 'w') as f:
        json.dump(json_data, f)

    check_output("Rscript kamila.R", shell=True).decode()

    df_out = pd.read_csv('temp_clustered.csv')

    df['cluster'] = df_out['cluster']
    #st.dataframe(df)
    #helpers.FAMD_Plot(df_out)

    os.remove("temp_cat.csv")
    os.remove("temp_continue.csv")
    os.remove("temp_clustered.csv")
    os.remove("k.json")

    return df