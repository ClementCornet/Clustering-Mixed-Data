import pandas as pd
import json
import os
from subprocess import check_output
import streamlit as st
import numpy as np

def process(df, k):
    #hf_long = pd.read_csv('heart_failure_long.csv')

    #hf_long.to_csv('mixtcomp_temp.csv',index=False)

    # df.select_dtypes('object').apply(lambda x: x.replace(x.unique(),list(range(1,1+len(x.unique())))))

    json_data = { "n_clusters" : k}
    with open('k.json', 'w') as f:
        json.dump(json_data, f)

    df2 = df.copy()
    df2.loc[:,df.dtypes == 'object']=df.loc[:,df.dtypes == 'object'].apply(
            lambda x: x.replace(x.unique(),list(range(1,1+len(x.unique())))))

    for col in df2.columns:
        if df2[col].dtype != 'object':
            if np.max(df2[col]) > 100000:
                df2[col] = df2[col]/100

    df2.columns = ["X"+str(i) for i in  range(df2.shape[1])]
    df2.to_csv('temp_data.csv', index=False)

    d = dict(zip(df2.columns,df.dtypes))

    for k in d.keys():
        t = str(d[k])
        if t == 'object':
            d[k] = 'Multinomial'
        elif t == 'int64':
            d[k] = 'Poisson'
        elif t == 'float64':
            d[k] = 'Gaussian'
    
    with open('model.json','w') as f:
        json.dump(d,f,indent=4)

    #st.dataframe(hf_long)

    ### R PROCESS

    check_output("Rscript mixtcomp2.R", shell=True).decode()

    clusters = pd.read_csv('mixtcomp_temp.csv')

    #st.dataframe(hf_long)

    df['cluster'] = clusters['x']
    os.remove('k.json')
    os.remove('model.json')
    os.remove('temp_data.csv')
    os.remove('mixtcomp_temp.csv')
    return df