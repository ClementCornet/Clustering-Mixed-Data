
K-Prototype (introduced in 1997 by Huang) is a partional clustering algorithm for mixed data. It can be considered as a combination of K-Means ans K-Modes.  

- Select k initial prototypes from the dataset  
- Allocate each element from the dataset to a cluster whose prototype is the nearest to it. The euclidean and matching distances are respectively  used for numerical and categorical variables.  

$$  \\~\\  \hspace{10mm}  Distance(a,b) = Euclidean(a,b) + \gamma \times Matching(a,b) $$ 
  
$$   \hspace{50mm}   \gamma = \frac{\sum_{} std(X_i)}{N_{num}} $$

- Recompute the prototype (centroid) for each cluster
- Iterate until no object changes, or the maximum number of iterations has been reached