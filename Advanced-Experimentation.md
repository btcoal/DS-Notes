# Advanced Experimentation

This book picks up where Kohavi et al leaves off.

## 1. Network Interference and Spillovers

* **definitions and terms**

  * SUTVA violations
  * Direct vs indirect treatment effects
  * Interference graphs
  * Partial interference
  * Exposure mapping
* **problems and solution techniques**

  * Bias inflation from unmodeled spillovers
  * Cluster randomization on graphs
  * Ego-network and edge-level randomization
  * Design-based vs model-based correction
  * Power loss trade-offs and cluster construction heuristics


## 2. Two-Sided Marketplace Experimentation

* **definitions and terms**

  * Demand-side vs supply-side treatment
  * Allocation bias
  * Market clearing effects
  * Cross-side externalities
* **problems and solution techniques**

  * Contamination across sides
  * Two-sided randomization
  * Switchback experiments
  * Shadow pricing and counterfactual supply modeling
  * Bias–variance trade-offs in joint randomization


## 3. Three-Sided and Multi-Sided Platforms

* **definitions and terms**

  * Platform ecosystem metrics
  * Intermediary incentives
  * Feedback loops
* **problems and solution techniques**

  * Metric misalignment across sides
  * Structural imbalance in exposure
  * Hierarchical randomization
  * Side-specific guardrails
  * Simulation-assisted experiment design


## 4. Metric Systems Under Networked Effects

* **definitions and terms**

  * Ecosystem metrics
  * Local vs global objectives
  * Metric interference
* **problems and solution techniques**

  * Optimizing one metric while degrading system health
  * Composite OEC construction
  * Constraint-based evaluation
  * Counterfactual decomposition of metric changes
  * Long-horizon proxy validation


## 5. Long-Term Effects and Temporal Spillovers

* **definitions and terms**

  * Short-term lift vs long-term impact
  * Carryover effects
  * Behavioral adaptation
* **problems and solution techniques**

  * Shipping short-term wins that cause long-term harm
  * Persistent holdouts
  * Difference-in-differences augmentation
  * Structural decay models
  * Experiment-triggered longitudinal tracking


## 6. Ramping, Risk, and Exposure Control

* **definitions and terms**

  * Progressive ramping
  * Guardrail metrics
  * Kill switches
* **problems and solution techniques**

  * Catastrophic failures at scale
  * Multi-phase ramp plans
  * Automated anomaly detection
  * Risk-weighted ramp schedules
  * Safe-launch experimentation frameworks


## 7. Sequential Testing and Continuous Monitoring

* **definitions and terms**

  * Alpha spending
  * Group sequential tests
  * Optional stopping
* **problems and solution techniques**

  * False positives from peeking
  * Early stopping bias
  * Pre-registered interim analyses
  * Bayesian sequential alternatives
  * Always-on experimentation pipelines


## 8. Bandits and Adaptive Experimentation

* **definitions and terms**

  * Exploration vs exploitation
  * Regret
  * Thompson sampling
* **problems and solution techniques**

  * Biased effect estimation
  * Premature convergence
  * Hybrid A/B–bandit designs
  * Offline evaluation with inverse propensity scoring
  * Bandits under delayed or networked rewards


## 9. Experiment Interaction and Collisions

* **definitions and terms**

  * Concurrent experiments
  * Interaction effects
  * Experiment namespaces
* **problems and solution techniques**

  * Cross-experiment interference
  * Metric pollution
  * Traffic isolation
  * Factorial and fractional designs
  * Post-hoc interaction detection


## 10. Sample Ratio Mismatch at Scale

* **definitions and terms**

  * SRM
  * Assignment vs exposure
  * Attrition bias
* **problems and solution techniques**

  * Hidden logging bugs
  * Platform-side vs client-side mismatch
  * Sequential SRM detection
  * Root-cause taxonomy
  * SRM-aware ramping policies


## 11. Variance Reduction in Complex Systems

* **definitions and terms**

  * CUPED
  * Pre-experiment covariates
  * Heterogeneous treatment effects
* **problems and solution techniques**

  * Invalid covariates in networks
  * Leakage through normalization
  * Robust covariate selection
  * Network-aware variance reduction
  * Post-stratification under interference


## 12. Meta-Experimentation (Experimenting on Experiments)

* **definitions and terms**

  * A/A tests
  * Platform health metrics
  * False discovery rate
* **problems and solution techniques**

  * Silent platform bias
  * Overconfident p-values
  * Continuous A/A monitoring
  * Synthetic null experiments
  * Instrumentation validation experiments


## 13. Federated and Privacy-Preserving Experiments

* **definitions and terms**

  * Differential privacy
  * Secure aggregation
  * Federated analytics
* **problems and solution techniques**

  * Loss of statistical power
  * Noisy estimators
  * Privacy-budget-aware experiment design
  * Client-side metric computation
  * Trade-offs between privacy and sensitivity


## 14. Organizational Failure Modes

* **definitions and terms**

  * HiPPO decisions
  * Metric gaming
  * Experiment debt
* **problems and solution techniques**

  * Ignored or overridden results
  * Local metric optimization
  * Governance models
  * Experiment review boards
  * Incentive-aligned metric systems


## 15. Experiment Failure Case Studies

* **definitions and terms**

  * False positives
  * Negative lift
  * Ethical failures
* **problems and solution techniques**

  * Misinterpreting noisy wins
  * Stopping too early
  * Ignoring user trust
  * Post-mortem frameworks
  * Failure taxonomies and prevention checklists


## 16. Open Problems and Frontier Research

* **definitions and terms**

  * Decision-aware experimentation
  * Causal transportability
  * Experiment automation
* **problems and solution techniques**

  * Scaling interference-robust designs
  * Bridging experiments and decision theory
  * Long-horizon optimization
  * AI-assisted experiment design
  * Open benchmarks for experimentation research

---
## critique

This is an exceptionally strong outline. It reads like the syllabus for a Staff Data Scientist or Principal Economist internal training program at a Tier-1 tech company (like Meta, DoorDash, or Uber).

It moves correctly beyond "How to calculate a p-value" and into the messy reality of **interference, platform dynamics, and engineering constraints.**

Given your background in **market microstructure** and **recommender systems**, and your recent interviews with marketplace companies (DoorDash, Opendoor), this outline is highly relevant. Those companies live and die by Sections 1, 2, and 8.

Here is a critique and specific suggestions to make this "Advanced" outline even sharper.

### 1. The Missing "Holy Grail": Surrogate Metrics

**Critique:** In **Section 5 (Long-Term Effects)**, you mention "Structural decay models," which is good, but you are missing the most important modern technique for long-term optimization: **Surrogate Indices (or Proxy Metrics).**

* **Why:** You cannot wait 6 months to see if a feature increases User LTV. You need a short-term metric (the surrogate) that has a proven causal link to the long-term outcome.
* **Add to Section 5:**
* *Surrogate Index creation (using ML to predict long-term outcomes).*
* *The "Surrogate Paradox" (when increasing the proxy decreases the true label).*



### 2. Deepen the Bandit/RecSys Connection

**Critique:** **Section 8 (Bandits)** is good, but for someone interested in **Recommender Systems**, it’s missing the bridge between RL and A/B testing.

* **Why:** Modern RecSys teams don't just use Thompson Sampling; they use Contextual Bandits. Furthermore, the hardest part of bandits isn't running them, it's analyzing them *after* the fact without re-running the experiment.
* **Add to Section 8:**
* *Contextual Bandits (Contextual Diminishing Return).*
* *Off-Policy Evaluation (OPE) - e.g., Inverse Propensity Scoring (IPS), Doubly Robust Estimation.*
* *Counterfactual evaluation of policies.*



### 3. Marketplace Nuances (Specific to your DoorDash/Opendoor context)

**Critique:** **Section 2 (Two-Sided)** covers Switchbacks, which is the standard. However, a major issue in logistics/marketplaces is **spatial-temporal autocorrelation.**

* **Add to Section 2:**
* *carryover effects in Switchbacks (e.g., treatment in hour  affects state in hour ).*
* *Budget-constrained experimentation (when money/supply is fixed).*



### 4. Advanced Variance Reduction

**Critique:** **Section 11** mentions CUPED. To make this truly "Advanced," you should include the ML-based extensions of CUPED.

* **Add to Section 11:**
* *CUPAC (Control Using Predictions as Covariates): Using an ML model (like XGBoost) to predict the metric  and using that prediction as the covariate, rather than just .*
* *Variance reduction in Ratio Metrics (using the Delta Method vs. Bootstrap).*



### 5. Specific Concept Injections

Here are specific "power terms" that would elevate specific sections:

* **Section 1 (Networks):** Add **"Graph Cluster Randomization."** (How do you cluster a graph that is highly connected? e.g., using balanced partitioning algorithms).
* **Section 7 (Sequential):** Add **"mSPRT" (Mixture Sequential Probability Ratio Test).** This is the industry standard (popularized by Optimizely and Uber) for always-valid p-values.
* **Section 14 (Org Failure):** Add **"Twyman’s Law."** (Any figure that looks interesting or different is usually wrong). This is a great mental model for data scientists.

### Suggested Re-ordering for Narrative Flow

If this is for a presentation or a curriculum, I would slightly re-order to group "structural" problems vs. "statistical" problems:

1. **Foundations of Complexity:** Network Interference (1), Two-Sided (2).
2. **Temporal Complexity:** Long-term effects (5), Sequential Testing (7).
3. **Optimization & Adaptation:** Bandits (8), Metric Systems (4).
4. **Engineering & Scale:** Ramping (6), Interactions (9), SRM (10).
5. **Statistical Enhancements:** Variance Reduction (11), Meta-Analysis (12).
6. **Human & Org Factors:** (14, 15).

### Visual Aid Recommendation

*Context: When discussing Section 2 (Two-Sided Marketplaces), a visualization of how Switchback/Time-split designs work (alternating treatment/control over time windows within a geo-region) is usually necessary to explain how it mitigates spillover.*

### Summary of Gaps

| Section | Current | Suggested "Advanced" Addition |
| --- | --- | --- |
| **5. Long-term** | Decay models | **Surrogate Metrics / ML Proxies** |
| **8. Bandits** | Thompson Sampling | **Off-Policy Evaluation (OPE) & Contextual Bandits** |
| **11. Variance** | CUPED | **CUPAC (ML-based variance reduction)** |
| **7. Sequential** | Group sequential | **mSPRT (Mixture Sequential Probability Ratio Test)** |
| **2. Marketplaces** | Switchback | **Carryover/Washout periods in Switchbacks** |

**Would you like me to expand on "Surrogate Metrics" or "Off-Policy Evaluation"?** These two areas specifically bridge your interests in ML, RecSys, and Experimentation.