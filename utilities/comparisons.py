import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def page():
    """Comparison page, displaying AMI/ARI heatmaps on our datasets"""
    st.title('Algorithm Comparisons')
    datasets = ['Heart Failure Short',
                'Heart Failure Long',
                'Contraceptive Method Choice',
                'Penguins'
                ]

    # Select Data and Index to display
    c1,c2 = st.columns([1,1])
    index = c1.selectbox('External Index : ',['ARI', 'AMI'])
    data = c2.selectbox('Choose Dataset : ', datasets)

    # Chosen Data's Caracteristics. Hard coded for now
    describe = {
        'Heart Failure Short' : '299 rows, 2 categorical and 3 numerical features',
        'Heart Failure Long' : "299 rows, 5 categorical and 7 numerical features",
        'Contraceptive Method Choice': '1472 rows, 7 categorical and 2 numerical features',
        'Penguins' : '344 rows, 3 categorical and 4 numerical features'
    }
    st.write(describe[data])

    # Read Locally stored heatmaps
    filepath = 'data/heatmaps/' +'_'.join(data.split()) + f'_{index}.csv'

    heat = pd.read_csv(filepath, index_col=0)
     
    # Plot heatmap
    fig,ax = plt.subplots()
    sns.heatmap(heat,annot=True,square=True,cbar=False,fmt=".2f")
    ax.set_title(data)
    sns.set(font_scale=1)

    st.pyplot(fig)

        