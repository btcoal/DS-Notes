# Clustering and Unsupervised Learning

## k-means Clustering

Objective: Partition data into `k` clusters by minimizing within-cluster variance. The loss function is given by

$$
J(\{C_i\}_{i=1}^k) = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
$$
where $C_i$ is the set of points in cluster $i$ and $\mu_i$ is the centroid of cluster $i$.

* Algorithm:
    1. Initialize `k` centroids randomly.
    2. Assign each point to the nearest centroid.
    3. Update centroids as the mean of assigned points.
    4. Repeat steps 2-3 until convergence.

* Limitations:
    - Sensitive to initial centroid placement.
    - Assumes spherical clusters of similar size.
    - Requires specifying `k` in advance.
* Variants:
    - k-medoids: Uses actual data points as centroids, more robust to outliers.
    - Mini-batch k-means: Uses small random samples for faster convergence on large datasets.

Python implementation from scratch:

```python
import numpy as np
def kmeans(X, k, max_iters=100, tol=1e-4):
    n_samples, n_features = X.shape
    # Randomly initialize centroids
    random_indices = np.random.choice(n_samples, size=k, replace=False)
    centroids = X[random_indices]

    for i in range(max_iters):
        # Assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return centroids, labels
```

Using scikit-learn:

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
```

## SVD and PCA

SVD is a matrix factorization technique that decomposes a data matrix `X` into three matrices:
$$
X = U \Sigma V^T
$$
where `U` and `V` are orthogonal matrices and `Σ` is a diagonal matrix of singular values.

PCA uses SVD to reduce dimensionality by projecting data onto the top `k` singular vectors corresponding to the largest singular values. This captures the directions of maximum variance in the data.

Python implementation of PCA using SVD:

```python
def pca(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    # Compute SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Project data onto top n_components
    X_reduced = U[:, :n_components] @ np.diag(S[:n_components])
    return X_reduced
```

Using scikit-learn:

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=k)
X_reduced = pca.fit_transform(X)
```

SVD in PyTorch:

```python
import torch
X_tensor = torch.tensor(X, dtype=torch.float32)
U, S, Vt = torch.svd(X_tensor)
```

SVD in NumPy:

```python
U, S, Vt = np.linalg.svd(X, full_matrices=False)
```
