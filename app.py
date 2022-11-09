import streamlit as st
from utilities.template import go_to_page

st.set_page_config(page_title='Mixed Clustering',page_icon='favicon.png')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


with st.sidebar:
    st.title('Mixed Clustering')
    choice = st.selectbox("Algorithm : ", [" --- Choose Algorithm --- ",
                                            "K-Prototype",
                                            "Hierarchical with Gower Distance",
                                            "K-Means using FAMD",
                                            "Kamila",
                                            "Modha-Spangler",
                                            "MixtComp",
                                            "UMAP-HDBSCAN",
                                            "Spectral with K-Prototype\'s Distance",
                                            "Comparisons"
                                            ])



go_to_page(choice)