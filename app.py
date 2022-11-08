from cv2 import CHAIN_APPROX_NONE
import streamlit as st
from algo import go_to_page

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
                                            #"K-Prototype with new distance",
                                            #"Hierarchical with K-Prototype's distance",
                                            "Kamila",
                                            "Modha-Spangler",
                                            "MixtComp",
                                            "UMAP-HDBSCAN",
                                            "Spectral with K-Prototype\'s Distance",
                                            "Comparisons"
                                            ])



go_to_page(choice)