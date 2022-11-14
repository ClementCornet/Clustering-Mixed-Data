# Clustering-Mixed-Data
Dynamic Web App allowing to perform clustering over mixed data.

## Installation :

Install Python Dependencies :
```
pip install -r requirements.txt
```

Kamila, Modha-Spangler and MixtComp are computed through R. Thus you must have Rscript in your environement variable `PATH` to use them. By default, Rscript executable is located in `C:\Program Files\R\R-<version>\bin`. 

You should be able to use the other algorithms (K-Prototype, Spectral Clustering, Hierarchical Gower, FAMD-KMeans and UMAP-HDBSAN) even without R.

Execute `req.R` to install R dependencies.

## Run App :

To launch the app in a web browser :

```
streamlit run app.py engine=python
```