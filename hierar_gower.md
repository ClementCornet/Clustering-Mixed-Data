Here, we use  agglomerative hierarchical clustering with Gower's Matrix. First, we compute Gower's Matrix, the pairwise distance matrix using Gower's Distance.
$$
    D_{Gower}(a,b) = 1 - \frac{\displaystyle\sum_{j=1}^{p}
    s_j(a,b)
    }{p} \\~\\


    s_j(a,b)
    \left\{
        \begin{array}{rcr}
            1-Manhattan(a,b)  \ for \ numerical \ variables \\
            1-Matching(a,b) \ for \ numerical \ variables
        \end{array}
    \right.
$$

We then use agglomerative clustering with this matrix.
- A cluster is created for each element of the dataset
- Each iteration, the 2 closest clusters are merged, until we obtain the desired number of clusters
