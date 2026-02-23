# Distance Metrics

* Euclidean distance
    $$
    \sqrt{\sum_{i=1}^n (x_{1i} - x_{2i})^2}
    $$
* Manhattan Distance
    $$
    \sum_{i=1}^n |x_{1i} - x_{2i}|
    $$
* CosineSimilarity (https://docs.pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html#torch.nn.CosineSimilarity)

    $$
    \frac{x_1 \cdot x_2}{\max(\|x_1\|_2 \cdot \|x_2\|_2, \epsilon)}
    $$

* PairwiseDistance (https://docs.pytorch.org/docs/stable/generated/torch.nn.PairwiseDistance.html#torch.nn.PairwiseDistance)
    $$
    \sqrt{\sum_{i=1}^n (x_{1i} - x_{2i})^2}
    $$

    where, the $p$-norm (https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.norm.html) is defined as
    $$
    \|x\|_p = \left(\sum_{i=1}^n |x_{1i}|^p\right)^{1/p}
    $$
* KL-divergence
    $$
    D_{KL}(P \parallel Q) = \sum_{i=1}^n P(x_i) \log \frac{P(x_i)}{Q(x_i)}
    $$
* JS Divergence
    $$
    D_{JS}(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M)
    $$

