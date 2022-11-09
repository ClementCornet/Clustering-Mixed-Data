# Convex k-means

Algorithm created by Modha and Spangler in 2003.
The algorithm defines  distortion between two data objects as a weighted sum of suitable distortion measures on individual component feature vectors. Using a convex optimization formulation, the algorithm
generalizes the classical Euclidean k-means algorithm to employ the weighted distortion measure.

We are given N data objects, such that: $$x_i = (F_{(i,1)}^T, ..., F_{(i,l)}^T, ..., F_{(i,m)}^T)^T$$denote the **i**-th data vector with **m** different data types and $i = 1, 2, . . . , N$.

The objective is partitioning the dataset $(X_i)_{i=1}^n$ into **k** disjoint clusters $(\pi_u)_{u=1}^k$. Given a partitioning $(\pi_u)_{u=1}^k$, for each partition $\pi_u$, we write the corresponding generalized as : $$ c_u = (c_{(u,1)}, ..., c_{(u,l)}, ..., c_{(u,m)}) $$ 

## *Weighted distortion measure:*

The weighted distortion measure for mixed-type data is defined as : $$ D^\alpha(x_1,x_2) = \sum_{l=1}^m\alpha_lD_l(F_{(1,l)}, F_{(2,l)})$$ where :

-  Each of the **m** data types is assigned its own distance metric $D_l$ (For example : Euclidian or cosine distances) 
- Each of the **m** data types is assigned a corresponding weigth in $\alpha = (\alpha_1, ..., \alpha_l, ...,\alpha_m)$, with $\alpha_l \geq 0$.
   
## *Algorithm description:*

We measure the distortion of each individual cluster $\pi_u; u \in [1;k]$, as : $$ \sum_{x \in \pi_u} D^\alpha(x, c_u)$$

The quality of the entire partitioning $(\pi_u)_{u=1}^k$ as the combined distortion of all the **k** clusters is measured as: $$ \sum_{u=1}^k\sum_{x \in \pi_u} D^\alpha(x, c_u)$$
The proposed algorithm seeks to partition the data vectors into the k disjoint clusters $(\pi_u^*)_{u=1}^k$ such that the following is minimized : 

$$ \hspace{25mm} (\pi_u^*)_{u=1}^k = \argmin_{(\pi_u)_{u=1}^k}(\sum_{u=1}^k\sum_{x \in \pi_u} D^\alpha(x, c_u)) $$
 
The algorithm follows 4 steps:

1. Set the index of iteration $t = 0$ .Start with an arbitrary partitioning of the data objects, say $(\pi_u^{(0)})_{u=1}^k$. Let $(c_u^{(0)})_{u=1}^k$ denote the generalized centroids associated with the given partitioning. 

2. For each data object $x_i$, $1 ≤ i ≤ n$, find the generalized centroid that is closest to
$x_i$. Now, for all **k**, we compute the new partitioning $(\pi_u^{(t+1)})_{u=1}^k$ induced by the old generalized centroid $(c_u^{(t)})_{u=1}^k$ : $$ \pi_u^{(t+1)} = \{ x \in \{x_i\}_{i=1}^n : D^\alpha(x, c_u^{(t)}) \leq D^\alpha(x, c_v^{(t)}), 1 \leq v \leq k \}  $$
In words, $\pi_u^{(t+1)}$ is the set of all data objects that are closest to the generalized centroid
$c_u^{(t)}$. If some data object is simultaneously closest to more than one generalized centroid,
then it is randomly assigned to one of the clusters.

3. Given this new partitioning, we compute the new generalized centroids $(c_u^{(t+1)})_{u=1}^k$ ...

4. If the stopping criterion is met, then : $\pi_u^* = \pi_u^{(t+1)}$ and set $c_u^*$ = $c_u^{(t+1)}$. Otherwise, increment **t** by 1 and go back to step 2.
**Stopping criterion :** if the change in equation (3) between two successive iterations is less than some specified threshold.

## *The optimal feature weighting:*

We aim to select the optimal weigths $\alpha^*$. Modha and Spangler define the **average within-cluster distortion** for the **l**-th data type as: $$W_l(\alpha) = \sum_{u=1}^k \sum _{x \in \pi_u^*(\alpha)} D_l(F_l, c_{(u,l)}^*(\alpha)) $$ and the **average between-cluster distortion** as : $$ B_l(\alpha) = \sum _{i=1}^N D_l(F_l, \bar{c}_l(\alpha)) - W_l(\alpha)$$ where :
- $\bar{c}_l$ : the centroid of the **l**-th data type taken across all **N** data vectors.

We aim to minimize $W_l(\alpha)$ and maximize $B_l(\alpha)$, in other words, minimize the differences between the data objects within a cluster and maximize the differences between the data objects from the different clusters. To do so, we determine : $$ \alpha^* = \argmin_\alpha \{ \frac{W_1(\alpha)}{B_1(\alpha)} * \frac{W_2(\alpha)}{B_2(\alpha)}* ... * \frac{W_m(\alpha)}{B_m(\alpha)} \} $$

The weighting is identified through a brute-force search: the algorithm is run repeatedly for
a grid of the possible weightings $\alpha$ with selection as described above
