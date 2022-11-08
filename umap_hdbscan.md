The same way it is possible to perform K-Means over the coordinates given by FAMD, it is possible to perform another numerical clustering algorithm over the UMAP coordinates. This section relies on the Amazon-DensClus Python package, published in 2021 by AWS.  
 
After Reducing the dataset using UMAP, it performs the Hierarchical Density Based Spatial Clustering Algorithm with Noise (HDBSCAN) introduced by Campello,Moulavi and Sander in 2013.

- Transform the space according to the density/sparsity
- Build the minimum spanning tree of the distance weighted graph

<img src="https://cdn-images-1.medium.com/max/800/1*Afnvx3A3hPOM3eoT8l9dDg.png" width=700 height=200  style="display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;"/>

- Construct a cluster hierarchy of connected components
- Condense the cluster hierarchy based on minimum cluster size
- Extract the stable clusters from the condensed tree



Note that a group with a low number of elements is considered as noise