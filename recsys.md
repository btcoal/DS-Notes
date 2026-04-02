# Recommender Systems

## Two-Tower Models

### User Tower
* Input: user features (demographics, behavior history, preferences).
* Architecture: embedding layers for categorical features, dense layers for numerical features, followed by fully connected
    layers.
### Item Tower
* Input: item features (metadata, content embeddings).
* Architecture: similar to user tower.
### Interaction
* Compute dot product or cosine similarity between user and item tower outputs.
* Apply sigmoid or softmax for ranking or classification.
$$
\text{score}(u, i) = \frac{u \cdot i}{\|u\| \|i\|}
$$
* Train with pairwise ranking loss (e.g., BPR loss) or pointwise loss (e.g., binary cross-entropy).
## Multi-Task Rankers
* Jointly optimize for multiple objectives (e.g., click-through rate, conversion rate, dwell
    time).
* Shared bottom layers with task-specific heads.
* Use weighted sum of task losses for training.


## LLM-based Recommenders

## Offline vs Online Evaluation

* [Mastering Recommendation Systems Evaluation: An A/B Testing Approach with Insights from the Industry](https://www.youtube.com/watch?v=cQJfYtTfJQg&list=WL&t=804s)


## Resources

* https://netflixtechblog.com/recommending-for-long-term-member-satisfaction-at-netflix-ac15cada49ef
* https://netflixtechblog.medium.com/lessons-learnt-from-consolidating-ml-models-in-a-large-scale-recommendation-system-870c5ea5eb4a
* https://netflixtechblog.medium.com/recsysops-best-practices-for-operating-a-large-scale-recommender-system-95bbe195a841
* https://netflixtechblog.com/reinforcement-learning-for-budget-constrained-recommendations-6cbc5263a32a
* https://medium.com/tech-p7s1/video-recommendations-at-joyn-two-tower-or-not-to-tower-that-was-never-a-question-6c6f182ade7c
* https://developer.nvidia.com/merlin
* https://www.reddit.com/r/RedditEng/comments/1itlrtk/adding_exploration_in_ads_retrieval_ranking/
* YouTube Recommendations [1](../readings/45530.pdf),
* https://www.youtube.com/watch?v=2vlCqD6igVA&list=PLjN9RcbNfgy48cMbImxViYN2ZKOUOnw0Q&index=1&t=786s&pp=gAQBiAQB
* https://www.youtube.com/watch?v=AbZ4IYGbfpQ&list=WL&index=4&t=917s&ab_channel=AIEngineer
* https://www.youtube.com/watch?v=LxQsQ3vZDqo&t=888s&ab_channel=AIEngineer
* https://careersatdoordash.com/blog/homepage-recommendation-with-exploitation-and-exploration/
* https://careersatdoordash.com/blog/evolving-doordashs-substitution-recommendations-algorithm/