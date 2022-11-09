FAMD is often used to visualize mixed data. But since it allows us to translate mixed data into categorical coordinates, it is possible to perform numerical clustering over those coordinates. Here, we use the following K-Means algorithm.  
With k the desired number of clusters :

- Select k initial centroids
- Allocate each element from the dataset to a cluster whose centroid is the nearest to it, using Euclidean Distance
- Recompute the centroid for each cluster
- Iterate until no object changes, or the maximum number of iterations has been reached