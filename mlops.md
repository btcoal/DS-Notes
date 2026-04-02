# ML Systems and MLOps

* Metaflow (workflow orchestration).
* Titus (container execution).
* Data infrastructure (Iceberg, Spark, Arrow)
* Kubernetes, gRPC, CI/CD for ML.

## Feature Engineering and Management

### Data Quality

### Feature stores 
k/v stores (Redis, Cassandra).

## Model Serving and Deployment

gRPC, REST APIs, model versioning.


## Monitoring and Observability

https://careersatdoordash.com/blog/monitor-machine-learning-model-drift/

## Drift Detection

See [Gemaque et al (2020)](../readings/unsupervised-drift-detection.pdf) "An overview of unsupervised drift detection methods"

["The ravages of concept drift in stream learning applications and how to deal with it"](https://www.kdnuggets.com/2019/12/ravages-concept-drift-stream-learning-applications.html) (KDD)

["A survey on concept drift adaptation"](../readings/2523813.pdf)

Types of drift
* $P(Y \mid X)$ changes $\Rightarrow$ Concept drift: change in the underlying distribution of the data 
* $P(X)$ changes $\Rightarrow$ Feature/Data drift: change in the distribution of the input data.
* $P(Y)$ changes $\Rightarrow$ Label drift: change in the distribution of the labels.


## embedding stores.
Vector DBs (Pinecone, Weaviate, Milvus).

## References
https://careersatdoordash.com/blog/enabling-efficient-machine-learning-model-serving/

https://careersatdoordash.com/blog/building-a-gigascale-ml-feature-store-with-redis/

https://careersatdoordash.com/blog/building-a-declarative-real-time-feature-engineering-framework/

https://careersatdoordash.com/blog/using-cockroachdb-to-reduce-feature-store-costs-by-75/

https://careersatdoordash.com/blog/five-common-data-quality-gotchas-in-machine-learning-and-how-to-detect-them-quickly/

https://careersatdoordash.com/blog/how-we-applied-client-side-caching/

https://careersatdoordash.com/blog/ship-to-production-darkly-moving-fast-staying-safe-with-ml-deployments/

https://careersatdoordash.com/blog/introducing-fabricator-a-declarative-feature-engineering-framework/

https://eng.lyft.com/lyftlearn-evolution-rethinking-ml-platform-architecture-547de6c950e1