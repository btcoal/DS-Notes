# Waymo Interview Prep
**Planner Evaluation — Data Scientist / MLE track**

## Interview Questions

### Metrics Design & Validation
*Core to the role*

**How would you design a metric to measure the safety of an autonomous vehicle's planner?**

- Frame around: (1) defining "safe" operationally (following distance, yielding, collision avoidance), (2) leading vs lagging indicators — collisions are rare ground truth, proxy metrics correlate more frequently.
- Cover data sources (simulation vs real-world logs), edge case coverage (novel scenarios, new territories, new platforms).
- Desirable properties: sensitivity, robustness to noise, interpretability, and actionability — can you trace a metric change to a software change?
- Bonus: discuss distributional shift when expanding to new geographies.

---

**How do you validate that a metric is actually measuring what you think it's measuring?**

- Construct validity: does it correlate with ground truth outcomes?
- Sensitivity testing: inject known-bad behaviors in simulation — does the metric detect them?
- Face validity: do domain experts agree? Stability: does metric drift in a stationary environment?
- Discrimination: can it distinguish two software versions that differ in the relevant behavior?
- Watch for Goodhart's Law — metrics that become targets get gamed.

---

**A metric has been in use for 2 years. Engineers are questioning whether it still reflects real-world safety. Walk me through how you'd audit it.**

- (1) Gather use-case inventory — who uses it and for what decisions?
- (2) Compare values against known-good and known-bad software versions to check discrimination power.
- (3) Analyze over time — did it change with product launches or geography expansions?
- (4) Check correlation decay with downstream outcomes (incidents, disengagements).
- (5) Interview stakeholders on edge cases where metric didn't catch issues. Recommend: refresh, complement, or deprecate with migration plan.

---

### A/B Testing & Experimentation

**How would you design an A/B test to evaluate two versions of a planning algorithm? What are the specific challenges vs. a typical web A/B test?**

- Unit of randomization: trips/routes, not users — trips are not i.i.d. (same route, same environmental conditions).
- Rare events: safety-critical events are very low frequency; standard t-tests may be underpowered. Consider non-parametric or time-to-event analysis.
- Simulation vs real-world: simulation is scalable but has sim-to-real gap.
- Interference: vehicles on the same road influence each other (SUTVA violation). Handle with stratified randomization, clustered standard errors, or pre-registered metrics.

---

**How do you determine the minimum detectable effect, and how does that affect your test design?**

- (1) Define target effect size — what delta is operationally meaningful?
- (2) Estimate baseline variance from historical data — high variance means larger sample needed.
- (3) Use power analysis to compute required sample size at desired alpha and beta.
- (4) Rarer events need more data — may require longer runs or more simulation.
- (5) Multiple comparisons: with many safety metrics tested simultaneously, adjust alpha (Bonferroni or Benjamini-Hochberg). Consider sequential testing for early stopping.

---

**The A/B test is statistically significant but the effect size is very small. How do you advise stakeholders on whether to ship?**

- Distinguish statistical from practical significance.
- In safety, tiny improvements compound significantly at scale.
- Check consistency across subgroups (urban/highway, new territories, edge cases).
- Consider opportunity cost — does this version block better alternatives?
- Recommend shipping only if effect is consistent across segments and downside risk is bounded.

---

### Data & Coding

**Describe how you'd use simulation data to build a training dataset for a metric validation model.**

- Scenario generation: parameterize simulation to produce diverse, labeled behaviors (unsafe lane changes, unsafe following distances) at scale.
- Label quality: simulation labels are ground truth but may not generalize — discuss sim-to-real gap mitigation (domain randomization, learned sim calibration).
- Dataset balance: rare safety events need oversampling or synthetic augmentation.
- Validation strategy: hold out real-world examples as an OOD test set to catch sim-to-real failures.

---

**Walk me through how you'd write a SQL query to compute a safety metric aggregated by trip and scenario type.**

- Identify granularity of raw data (per-frame sensor readings).
- Compute per-trip minimum: `MIN(following_distance)` with `GROUP BY trip_id`.
- Join with a scenario metadata table for scenario type, then aggregate with `AVG` and `PERCENTILE_CONT`.
- Handle NULLs and filter for valid frames. Use window functions for rolling stats.
- Performance: partition log tables by date, use approximate quantile functions (`APPROX_QUANTILES`) for very large datasets.

---

### Stakeholder & Ambiguity

**Different teams interpret the same safety metric differently and are making conflicting decisions. How do you resolve this?**

- (1) Run a use-case inventory — understand each team's interpretation and decision context.
- (2) Identify the root cause: definition ambiguity, data pipeline difference, or genuine disagreement.
- (3) Propose a shared definition document with edge-case examples.
- (4) If stakeholders have legitimately different needs, advocate for purpose-built metrics rather than overloading one.
- (5) Get alignment in writing before making changes.

---

**You're asked to add support for a new geographic territory to an existing safety metric. What's your process?**

- Understand what's different: local traffic norms, road infrastructure, speed limits, regulatory definitions.
- Review whether existing metric definitions embed implicit assumptions (e.g. U.S. following distance norms).
- Collect exploratory data from the new territory, run the existing metric, and check for anomalies.
- Work with domain experts to validate whether anomalies reflect true differences or metric failures.
- Update metric logic with territory-conditional handling or a calibration layer. Re-run validation suite with territory-stratified examples.

---

## Technical Concepts

### Statistics & Experimentation

**Statistical Power & Sample Size**

Power = P(reject H₀ | H₁ true) = 1 − β. In AV: safety events are rare (low base rate), so you need large samples or high-frequency proxy metrics.

- MDE = smallest effect detectable at given α, β. For proportions: `n ≈ 2(z_α + z_β)² × p(1−p) / δ²`
- For rare events: use negative binomial or Poisson regression instead of t-tests.
- Sequential testing (SPRT) for early stopping without inflating type I error.

---

**Causal Inference & A/B Test Validity**

Requires: randomization, SUTVA (no spillover), and balance. In AV:

- Trips on same road may interfere (SUTVA violation) — use clustered standard errors.
- Stratify on route type, time-of-day, weather to reduce variance.
- Simpson's Paradox: aggregate metrics can reverse when conditioned on confounders (e.g. new territory).
- Difference-in-differences if you can't fully randomize (e.g. city-level rollout).

---

**Multiple Comparisons Correction**

When testing many metrics simultaneously, familywise error rate inflates.

- Bonferroni: α/m — conservative, controls FWER.
- Benjamini-Hochberg: controls FDR — less conservative, preferred with many hypotheses.
- Pre-registration: declare primary metric before seeing data to avoid p-hacking.
- Hierarchy: designate one primary safety metric; treat others as secondary/exploratory.

---

### Metrics & Measurement

**Metric Properties to Know Cold**

A good metric is: sensitive (detects real changes), specific (no false alarms), reliable (low noise), valid (measures what it claims), and actionable (traceable to root cause).

- Lagging indicators: collisions, disengagements — rare but high-stakes ground truth.
- Leading indicators: following distance, speed vs limit, yield compliance — frequent, proxy for safety.
- Goodhart's Law: when a measure becomes a target, it ceases to be a good measure.
- Distributional robustness: metric should behave consistently across scenario types and geographies.

---

**Simulation vs. Real-World Evaluation**

Simulation: fast, controllable, scalable, labeled — but suffers sim-to-real gap. Real-world: ground truth but expensive and safety-constrained.

- Validate sim metrics against real-world logs — do they correlate?
- Scenario mining: extract real-world clips and replay in simulation for faster iteration.
- Adversarial simulation: parametrically generate hard scenarios to stress-test the planner.
- Log replay — open-loop replays historical sensor data; closed-loop lets planner act and sim reacts.

---

### ML & Data Engineering

**Imbalanced Datasets & Rare Event Modeling**

Safety-critical events are rare. Techniques:

- Oversampling (SMOTE) or undersampling majority class.
- Class-weighted loss functions in training.
- Precision-recall AUC more informative than ROC-AUC for rare events.
- Threshold calibration: optimize on held-out set using F-beta (weight recall if false negatives are costly).
- Anomaly detection framing: train on normal behavior, flag deviations.

---

**Feature Engineering for Driving Behavior**

Raw data: per-frame states (position, velocity, heading, sensor readings). Useful features:

- Time-to-collision (TTC), time headway, post-encroachment time.
- Jerk (derivative of acceleration) — proxy for comfort/aggression.
- Trajectory smoothness (curvature, deviation from planned path).
- Contextual features: road type, speed limit, number of nearby agents.
- Rolling statistics: 5s / 30s windows of min/max/mean of above.

---

**SQL Patterns for Large-Scale Log Analysis**

Key patterns for AV data:

```sql
-- Rolling minimum over a 5-frame window
MIN(dist) OVER (
  PARTITION BY trip_id
  ORDER BY frame_ts
  ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
)

-- 5th percentile of a metric across trips
PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY metric)

-- Approximate quantiles in BigQuery for scale
APPROX_QUANTILES(val, 100)[OFFSET(5)]
```

- Sessionization: use `LAG` and timestamp deltas to define trip segments.
- Join optimization: partition logs by date, prune early.

---

## Strategy & Tips

> **Core theme:** This role sits at the intersection of rigorous statistics, product judgment, and stakeholder influence. Show that you can translate ambiguous business needs into precise metric definitions — and back again.

### How to Frame Answers

**For metrics questions** — Always start with "what decision is this metric meant to support?" This shows product thinking. Then: data source → definition → validation → stakeholder alignment → limitations. Mention scalability to new territories/platforms unprompted — it's in the JD.

**For A/B test questions** — Lead with the unique challenges of AV experimentation (rare events, interference, sim-to-real gap) before going to general statistical machinery. Interviewers will notice if you just recite web A/B methodology without adapting it.

**For ambiguous / stakeholder questions** — Show a structured process: (1) clarify the use case, (2) gather data / understand the disagreement, (3) propose a framework, (4) get alignment in writing. Don't jump to solutions — interviewers at this level test whether you slow down when things are uncertain.

---

### Things to Prepare in Advance

**2–3 STAR stories ready to go**
- Designing or owning a metric end-to-end (with concrete numbers).
- Running an experiment with a tricky methodological challenge.
- Influencing a cross-functional stakeholder decision with data.

**Know the Waymo product context**
- Waymo One (SF / Phoenix / Austin rides), Waymo Via (trucking), Jaguar I-PACE and Zeekr fleet.
- The Planner is the component responsible for motion planning — deciding how the car moves.
- Knowing this lets you speak concretely about what "evaluating the planner" actually means.

**Questions to ask your interviewers**
- How does the team prioritize which metrics to improve vs. sunset?
- How tightly coupled is metric development to the ML research team vs. the safety team?
- What does the feedback loop look like between a metric change and a planner software release?
- How does the team handle metric disagreements between city deployments?