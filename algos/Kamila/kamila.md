KAMILA is a model based clustering algorithm for mixed data. 
 
We assume that our data set consists of **N** independent and identically distributed observations of a **(P + Q)** - dimensional vector of random variables **$(V^T,W^T)^T$** that follow a finite mixture distribution (see appendix) with **G** components.

I.e., $X_{1}, ..., X_{i}, ...,X_{N}$ the **N** observations i.i.d so that : $X_{i} = (V_i^T, W_i^T)^T$

With:
* $V_{i} = (V_{i1}, V_{i2}, ..., V_{ip}, ..., V_{iP})^T$ is a **P**-dimensional vector of **CONTINUOUS** random variables. $V_{i}$  is vector following a finite mixture of elliptical distributions, such that : $$ V_i \sim f_V(v) = \sum_{g=1}^G \pi_g h(v;µ_g)$$ where:
    - $\pi_g $: The prior probability of drawing an observation from the **g**-th population. It can be seen as the weigth of the distribution
    - h: The density function of the clusters.
$$ $$
* $W_{i} = (W_{i1}, W_{i2}, ..., W_{iq}, ..., W_{iQ})^T$ is a **Q**-dimensional vector of **CATEGORICAL** random variables with $q = 1,2, ...,Q$. The **q**-th elements of $W_{i}$ has $L_q$ categorical levels denoted $l = 1,2, ..., L_q$. In other words, $W_{iq} \in (1,2, ..., l, ...,L_q)$. $W_{i}$ is a mixture of multinomial random variables, i.e. each random variable follows a multinomial distribution, such that : $$ W_{i} \sim f_W(w) = \sum_{g = 1}^G \pi_g \prod_{q=1}^Q m(w_q;θ_{gq})$$ where :
    - $f_W(W)$ : The density function followed by $W_{i}$.
    - $m(w;θ)$ : The multinomial probability mass function, such that : $$m(w;θ) = \prod_{l=1}^{L_q} θ_l^{I(w=l)}$$ 
    - I{.} : The indicator function.
    - $θ_{gq}$ : The $L_q × 1$ parameter vector of the multinomial mass function corresponding to the **q**-th random variable drawn from the **g**-th cluster.

## *Algorithm description:*

At the **t**-th iteration we have:
* $\hat{µ}_g^{(t)}$ : the estimator of the centroid of a cluster **g**.
* $\hat{θ}_{gq}^{(t)}$ : the estimator of the parameters of the multinomial distribution corresponding to the **q**-th discrete random variable drawn from cluster **g**. 

### *Initialization:*
- Initializing $\hat{µ}_g^{(0)}$ for each **g** is done with random draws from the observed continuous data vectors.
- Initializing $\hat{θ}_{gq}^{(0)}$ for each **g** and each **q** is done with a draw from a Dirichlet distribution with shape parameters all equal to one.

### *Partition step:*

**Assigns each i observation of the N observations to one cluster g**

#### *For the continuous variables:*

1. At the **t**-th iteration, the Euclidean distance from observation **i** to each of the $\hat{µ}_g^{(t)}$ is calculated as: $$d_{ig}^{(t)} = \sqrt{\sum_{p = 1}^P [(v_{ip} - \hat{µ}_{gp}^{(t)} )]^2}$$

2. Next, the minimum distance is calculated for the **i**-th observation as : $$ r_i^{(t)} = \min_{g} (d_{ig}^{(t)}) $$ 

3. Then, the KD estimate (see appendix) of the minimum distances is constructed as : $$ \hat{f}_R^{(t)}(r) = \frac{1}{Nh^{(t)}}\sum_{i = 1}^N k(\frac{r - r_i^{(t)}}{h^{(t)}}) $$ where : 
    - k(.): The kernel function. (Gaussian kernel is used)
    - $ h = 0.9A^{\frac{1}{5}}$: The bandswith, with : $$A = \min(\hat{\sigma}, \frac{\hat{q}}{1.34}) $$ where:
        - $\hat{\sigma}$: The sample standard deviation.
        - $\hat{q}$: the sample interquartile range.

    
$$ $$
4. The function $\hat{f}_v^{(t)}$ is constructed as: $$\hat{f}_V^{(t)}(v) = \frac {\hat{f}_R^{(t)}(r)\Gamma(\frac{p}{2}+1)}{pr^{p-1}\pi^{\frac{p}{2}}} $$ where:
    - $r = \sqrt{v^Tv}$
    - $\Gamma$: The Gamma function.


#### *For the categorical variables:*

5. We calculate the probability of observing the **i**-th vector of categorical variables given population membership, in other words, that the **i**-th vector of categorical variables belongs to a cluster **g**, as : $$ c_{ig}^{(t)} = \prod_{q=1}^Q m(w_{iq};\hat{\theta}_{gq}^{(t)}) $$ where : 
    - $m(·;·)$ : The multinomial probability mass function.

#### *Then*:

We assign the **i**-th object to the cluster **g** that maximizes the function : $$ H_i^{(t)} = \log [\hat{f}_V^{(t)}(d_{ig}^{(t)})] + \log[c_{ig}^{(t)}] $$

### *Estimation step:*

6. In each iteration **t**, the latest partition of the **N** observations is used to calculate : 
$\hat{µ}_{gp}^{(t+1)}$ 
and 
$\hat{\theta}_{gq}^{(t+1)}$ 
for all **g**, **p** and **q**, as : 

$$ \hat{µ}_{g}^{(t+1)} = \frac{1}{|\Omega_g^{(t)}|} \sum_{i \in \Omega_g^{(t)}} v_i $$ 

$$ \hat{\theta}_{gql}^{(t+1)} = \frac{1}{|\Omega_g^{(t)}|}\sum_{i \in \Omega_g^{(t)}} I\{ w_{iq} = l\}$$ 

where :
    - $\Omega_g^{(t)}$ : The set of indices of observations assigned to population **g** at iteration **t**
    - $I\{.\}$: The indicator function.
    - $|.|$ : The cardinality of a set.

### *Stopping criterion:*

7. The algorithm's stopping criterion is when group membership remains unchanged from one iteration to the next. The algorithm uses the following quantities and stops when both are less than some chosen threshold(s) : 

$$ \epsilon_{con} = \sum_{g=1}^G \sum_{p=1}^P |\hat{µ}_{gp}^{(t)} - \hat{µ}_{gp}^{(t-1)}| $$

$$ \epsilon_{cat} = \sum_{g=1}^G \sum_{p=1}^P\sum_{l=1}^{L_q} |\hat{\theta}_{gql}^{(t)}-\hat{\theta}_{gql}^{(t-1)}| $$

### *Output:*
**In the paper**:
 
8. The algorithm selects the partitioning that maximizes over all runs:
$$ \sum_{i=1}^N \max_g\{H_i^{(final)}(g)\} $$
**In the package**:
 
8. The algorithm selects the partitioning that minimizes the product: $$ Q_{con} * [-\log c_{ig}^{(t)}]  $$ where :
    - $[-\log c_{ig}^{(t)}]$ :  The categorical negative log-likelihood
    - $Q_{con}$: The within- to between-cluster distance ratio of the continuous variables, with : $$ Q_{con} = \frac{W_{con}}{T_{con} - W_{con}} $$ where :
        - $W_{con}$ : a function that maps the observed data point index to its assigned cluster index, such that : $$W_{con} = \sum_{i=1}^N ||x_i - \hat{µ}_{z(i)}^{(final)}||_2$$ where :
            - $z : \{1, 2, ..., N\} \rightarrow \{1, 2, ..., G\} $
        $$ $$
        - $T_{con} = \sum_{i = 1}^N ||x_i - \hat{µ}||_2$, where:
            - $\hat{µ}$ : The sample mean taken across all continuous variables.



 





## *Appendix:*

### *Mixture distribution:* 

A mixture distribution is a probability distribution with density function **f**, such that **f** is a weighted summation of K distributions $\{g_1(x; \theta_1), ..., g_K(x; \theta_K)\}$ where every distribution has its own parameters $\theta_k$. Mathematicaly, the mixture distribution is formulated as : $$ f(x; \theta_1, ..., \theta_K) = \sum_{k=1}^K w_kg_k(x;\theta_k)$$ where:
- $w = \{w_1, ..., w_k, .., w_K\}$ : the weigths such that $\sum_{k=1}^K w_k = 1$

The distributions can be from different families, for example from beta and normal distributions, or from the same family.

In machine learning, many datasets can be clustered by mixture distribution, assuming that each cluster is modeled by one of the distribution.

### *Kernel Density Etimation (KDE):*

KDE is a non-parametric estimation method used to estimate a density function of a random variable.

Let $(x_1, x_2, ..., x_N)$ be independent and identically distributed samples drawn from some univariate distribution with an unknown density **f** at any given point **x**. The KD estimator is :

$$ \hat{f}_h(x) = \frac{1}{Nh}\sum_{i=1}^N K(\frac{x - x_i}{h}) $$ 
where :
- K: The kernel function.
- h: The bandswidth,  a smoothing parameter.

Most of the time, **K** is chosen as the density of a Gaussian, such that $K = \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}x^2}$.