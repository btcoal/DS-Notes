# Google LLM Eval Prep

## Additive Models vs Main Effects Models: The Core Tradeoff

When you have multiple factors affecting LLM performance, you're choosing between two fundamentally different questions:

**Additive model (main effects only):** "What is the best model *on average across tasks*? What is the best prompt template *on average across models*?"

**Saturated model (with interactions):** "What is the best specific combination of model × prompt × task?"

These can give contradictory answers, and neither is universally "correct"—it depends on your goal.

Suppose you're evaluating 2 models × 2 prompt templates × 2 task types, measuring accuracy:

| Model | Prompt | Task | Accuracy |
|-------|--------|------|----------|
| Gemini | Zero-shot | Summarization | 0.85 |
| Gemini | Zero-shot | Classification | 0.70 |
| Gemini | Few-shot | Summarization | 0.75 |
| Gemini | Few-shot | Classification | 0.90 |
| GPT | Zero-shot | Summarization | 0.80 |
| GPT | Zero-shot | Classification | 0.75 |
| GPT | Few-shot | Summarization | 0.78 |
| GPT | Few-shot | Classification | 0.77 |

**Raw cell means say:** The best combination is Gemini + Few-shot + Classification (0.90).

**But an additive model might say:**
- Best model: GPT (more consistent across conditions, higher average)
- Best prompt: Zero-shot (higher marginal mean)
- Best task: Classification (higher marginal mean)

So the additive model might recommend GPT + Zero-shot + Classification, which actually scores 0.75—much worse than the true best combination.

The discrepancy occurs because Gemini has a strong *interaction* with few-shot prompting specifically for classification tasks. The additive model misses this.

### Favor Additive Models When:

**1. You need generalizable insights.** If you're trying to answer "which model should we deploy across our entire product surface?" you want main effects. A model that's best on average, even if not best in every cell, may be the right strategic choice.

**2. You have sparse data.** With k factors having n₁, n₂, ..., nₖ levels, a saturated model requires estimating ∏nᵢ parameters. An additive model requires only ∑nᵢ. If you have 5 models × 10 prompt templates × 20 task types, that's 1,000 cells to estimate versus 35 parameters. Many cells may have few or no observations.

**3. You want statistical power.** By pooling information across cells, additive models give you tighter confidence intervals on the effects you care about.

**4. Interpretability matters.** "Gemini is 5% better than GPT" is easier to communicate than a 1,000-cell lookup table.

### Favor Saturated Models (or Interactions) When:

**1. You're optimizing a specific deployment.** If you know you're building a classification system and can choose any model-prompt combination, you want the actual best combination for that exact use case.

**2. You have strong priors that interactions exist.** LLMs are notorious for this—few-shot prompting helps some models dramatically and others barely at all. Chain-of-thought helps reasoning tasks but may hurt simple extraction tasks. These interactions are real and often large.

**3. You have sufficient data per cell.** If you can afford to evaluate each combination thoroughly, why throw away information by assuming additivity?

**4. The stakes are high for the specific decision.** If you're choosing the production configuration for a high-revenue product, you want the actual best combination, not the "best on average" one.


### The Middle Ground: Hierarchical/Mixed Models

In practice, the sophisticated approach is often a **hierarchical model** that estimates both main effects and interactions, but with regularization:

```
accuracy ~ model + prompt + task + (model × prompt) + (model × task) + (prompt × task) + (model × prompt × task)
```

With regularization (e.g., Bayesian priors or penalized regression), you get:
- Interaction estimates that shrink toward zero when data is sparse
- Interactions that remain large when the data strongly supports them
- Main effects that still inform your understanding of average behavior

This lets the data tell you how much interaction structure exists, rather than imposing an assumption either way.

### Connecting to LLM Evaluation Specifically

This tradeoff is particularly acute in LLM evaluation because:

**1. Interactions are often huge.** Unlike many domains where main effects dominate, LLM performance can vary wildly across prompt formulations, task types, and even specific examples. A model that's "best on average" may be worst on your specific use case.

**2. The factor space is enormous.** Models × prompt templates × system prompts × temperature settings × task types × domains × languages... you can't possibly evaluate every combination densely.

**3. Benchmark design choices embed these assumptions.** When a leaderboard reports "Model X is best on MMLU," that's implicitly an additive model—averaging across all subjects. But Model X might be terrible at the specific subject you care about.

**4. Production decisions are usually specific.** You're rarely choosing "the best model in general." You're choosing "the best model for customer service chatbots in Spanish with our specific prompt template."


> One of the key decisions in evaluation design is whether to model main effects only or include interactions. This isn't just a statistical nicety—it determines whether you're answering 'what's best on average' versus 'what's best for this specific configuration.' Given that LLMs show notoriously large interactions between models, prompts, and tasks, I'd advocate for either explicitly modeling interactions when data permits, or being very clear that benchmark rankings reflect average performance and may not transfer to specific deployments.

## Metrics

### ELo ratings for head-to-head model comparisons in a "Chatbot Arena" style eval.

Elo ratings are a way to rank models based on pairwise comparisons. Each model starts with a rating (e.g., 1500). When two models compete, the winner takes points from the loser based on their current ratings. The amount of points exchanged depends on the expected outcome:
$$ E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}} $$
$$ E_B = \frac{1}{1 + 10^{(R_A - R_B)/400}} $$
where $R_A$ and $R_B$ are the current ratings of models A and B, and $E_A$ and $E_B$ are the expected probabilities of winning. After a match, the ratings are updated:
$$ R_A' = R_A + K(S_A - E_A) $$
$$ R_B' = R_B + K(S_B - E_B) $$
where $S_A$ and $S_B$ are the actual scores (1 for win, 0 for loss, 0.5 for draw), and $K$ is a constant that determines how much the ratings change after each match.



Limitations
* Elo ratings assume that the outcome of each match is independent and identically distributed, which may not hold if there are systematic differences in the types of questions or prompts used in different matches.
* can be sensitive to the choice of the initial ratings and the $K$ factor, which can affect the stability and convergence of the ratings over time.
* do not account for the possibility of ties or draws, which can occur in LLM evaluations when two models perform equally well on a given question or prompt.
* designed for pairwise comparisons and may not scale well when evaluating a large number of models simultaneously, as the number of matches required grows quadratically with the number of models.
* do not provide information about the absolute performance of the models, only their relative rankings. A model with a high Elo rating may still perform poorly in an absolute sense if the overall level of performance in the evaluation is low.
* can be affected by the presence of outliers or anomalous matches, which can disproportionately influence the ratings and lead to inaccurate rankings. In such cases, it may be necessary to implement additional measures to mitigate the impact of outliers, such as using a more robust rating system or applying statistical techniques to identify and exclude anomalous matches from the rating calculations.
* assume that the skill level of each model is constant over time, which may not hold in practice as models can improve or degrade due to updates, changes in training data, or other factors. This can lead to ratings that do not accurately reflect the current performance of the models, especially if there are significant changes in their capabilities over the course of the evaluation period.
* do not account for the possibility of systematic biases in the evaluation process, such as certain types of questions or prompts favoring one model over another. If there are biases in the evaluation data, it can lead to skewed ratings that do not accurately reflect the true performance of the models. In such cases, it may be necessary to implement additional measures to control for biases, such as using a more balanced set of evaluation questions or applying statistical techniques to adjust for known biases in the data.
* do not provide a measure of uncertainty or confidence in the ratings, which can be important for interpreting the results of the evaluation and making informed decisions based on the ratings. In cases where there is significant variability in the outcomes of matches or when the number of matches is limited, it may be necessary to implement additional measures to estimate the uncertainty in the ratings, such as using bootstrapping techniques or Bayesian methods to derive confidence intervals for the ratings.
* assume that the performance of each model is independent of the specific questions or prompts used in the evaluation, which may not hold if certain models perform better on specific types of questions or prompts. This can lead to ratings that do not accurately reflect the true capabilities of the models across a diverse range of evaluation scenarios. In such cases, it may be necessary to implement additional measures to account for question-specific performance differences, such as using a more granular rating system that evaluates performance on different categories of questions or applying statistical techniques to adjust for question-specific effects in the rating calculations.
* do not account for the possibility of strategic behavior by the models or their developers, such as intentionally performing well on certain types of questions or prompts to boost their ratings. This can lead to ratings that do not accurately reflect the true performance of the models across a broad range of evaluation scenarios. In such cases, it may be necessary to implement additional measures to detect and mitigate strategic behavior, such as using a more diverse set of evaluation questions or applying statistical techniques to identify and adjust for patterns of strategic performance in the rating calculations.
* do not provide a direct measure of the absolute performance of the models, which can be important for understanding their capabilities and limitations in a practical context. While Elo ratings can indicate which model is better relative to others, they do not provide information about how well the models perform in an absolute sense, such as their accuracy, precision, recall, or other relevant metrics. In cases where absolute performance is important for decision-making, it may be necessary to complement Elo ratings with additional evaluation metrics that provide a more comprehensive assessment of the models' capabilities.
* do not provide a direct measure of the practical significance of the differences between models, which can be important for making informed decisions based on the ratings. While Elo ratings can indicate which model is better relative to others, they do not provide information about the magnitude of the performance differences or their practical implications in real-world applications. In cases where practical significance is important for decision-making, it may be necessary to complement Elo ratings with additional evaluation metrics that provide a more comprehensive assessment of the models' capabilities and their relevance to specific use cases.


### Cohen's $\kappa$ 
For inter-rater reliability when both raters classify items into mutually exclusive categories. It accounts for agreement occurring by chance.
$$ \kappa = \frac{P_o - P_e}{1 - P_e} $$
where
* $P_o$ is the observed agreement proportion (number of items both raters agree on divided by total items).
* $P_e$ is the expected agreement proportion by chance, calculated as:
* $P_e = \sum_{k=1}^K P_{A,k} P_{B,k}$, where $P_{A,k}$ and $P_{B,k}$ are the proportions of items that rater A and rater B classify into category $k$, respectively.
* $\kappa$ ranges from -1 to 1, where 1 indicates perfect agreement, 0 indicates agreement equivalent to chance, and negative values indicate less agreement than expected by chance.

Limitations of Cohen's $\kappa$:
* assumes that the raters are independent and that the categories are mutually exclusive and exhaustive. It also assumes that the raters have the same distribution of classifications, which may not hold in practice.
* If one rater is much more lenient or strict than the other, Cohen's $\kappa$ can be misleadingly low even if there is substantial agreement.
* does not account for the possibility of systematic bias between raters (e.g., one rater consistently rates higher than the other), which can also affect the reliability assessment.
* is sensitive to the prevalence of the categories. If one category is very common and the other is rare, it can lead to paradoxical situations where high agreement on the common category results in a low $\kappa$ value.
* is not suitable for evaluating agreement on ordinal or continuous data, as it treats all disagreements equally regardless of their magnitude. For ordinal data, weighted kappa can be used, and for continuous data, intraclass correlation coefficients (ICCs) are more appropriate.
* does not provide information about the nature of disagreements (e.g., whether they are systematic or random), which can be important for understanding the reliability of the ratings and for improving the rating process.
* can be affected by the number of categories and the distribution of ratings across those categories. In cases with many categories or skewed distributions, it may not accurately reflect the true level of agreement between raters.
* assumes that the ratings are independent, which may not hold if the raters influence each other or if there is a common bias affecting both raters. In such cases, alternative measures of agreement that account for dependence may be more appropriate.
* does not account for the possibility of raters having different levels of expertise or knowledge, which can affect the reliability of their ratings. If one rater is more knowledgeable than the other, it can lead to systematic disagreements that are not captured by $\kappa$.
* is not a measure of accuracy or validity; it only assesses the consistency of ratings between raters. High $\kappa$ does not necessarily imply that the ratings are correct or meaningful, and low $\kappa$ does not necessarily imply that the ratings are unreliable or invalid.
* can be difficult to interpret in terms of practical significance. A $\kappa$ value of 0.6, for example, may indicate moderate agreement, but whether this level of agreement is acceptable depends on the context and the consequences of disagreements in the specific application.
* can be affected by the sample size and the number of items being rated. With a small sample size, $\kappa$ may be unstable and not accurately reflect the true level of agreement between raters. Additionally, with a large number of items, even small disagreements can lead to a low $\kappa$ value, which may not be meaningful in practice.
* does not provide information about the direction of disagreements (e.g., whether one rater consistently rates higher or lower than the other), which can be important for understanding the nature of the disagreement and for improving the rating process.
* is not suitable for evaluating agreement on multi-rater data (i.e., more than two raters), as it only assesses agreement between two raters. For multi-rater data, other measures of agreement such as Fleiss' kappa or Krippendorff's alpha should be used.
* can be affected by the presence of missing data or incomplete ratings, which can lead to biased estimates of agreement. In such cases, methods for handling missing data (e.g., imputation) may be necessary to obtain accurate estimates of $\kappa$.
* is not a measure of reliability in the sense of consistency over time or across different raters; it only assesses agreement between two specific raters at a single point in time. For assessing reliability over time or across multiple raters, other measures such as test-retest reliability or inter-rater reliability with multiple raters should be used.


### Krippendorff's $\alpha$ for inter-rater reliability.

Krippendorff's $\alpha$ is a measure of inter-rater reliability that can be used for any number of raters, any level of measurement (nominal, ordinal, interval, ratio), and can handle missing data. It is defined as:
$$ \alpha = 1 - \frac{D_o}{D_e} $$
where
* $D_o$ is the observed disagreement, calculated as the average distance between ratings for the same item across all raters.
* $D_e$ is the expected disagreement by chance, calculated based on the distribution of ratings across all items and raters.
* $\alpha$ ranges from -1 to 1, where 1 indicates perfect agreement, 0 indicates agreement equivalent to chance, and negative values indicate less agreement than expected by chance.

Krippendorff's $\alpha$ is more flexible than Cohen's $\kappa$ because it can handle multiple raters, different types of data, and missing ratings. It also provides a more nuanced assessment of agreement by considering the magnitude of disagreements rather than treating all disagreements equally.
Limitations
* like Cohen's $\kappa$, Krippendorff's $\alpha$ can be affected by the prevalence of categories and the distribution of ratings, 
* does not provide information about the nature or direction of disagreements between raters.
  
In the context of LLM evaluation, Krippendorff's $\alpha$ can be particularly useful for assessing the reliability of human evaluations of model outputs, especially when there are multiple annotators and when the ratings are on an ordinal or interval scale (e.g., rating the helpfulness of a response on a 1-5 scale). It can help ensure that the human evaluation data is consistent and reliable before using it to draw conclusions about model performance.

##  


## topics
* Permutation tests for significance of differences between models.
* Simpson's paradox in multi-domain evals (e.g., model A better on domain 1, model B better on domain 2, but A overall better due to more questions in domain 1).
* Bias-variance tradeoff in eval design: more questions reduces variance but may introduce bias if questions are not representative.
* Power analysis for detecting small improvements in noisy LLM evals.
* Multiple comparisons correction when evaluating many metrics or subgroups.
* ELO ratings for head-to-head model comparisons in a "Chatbot Arena" style eval.


## Notes Docs

* [Applied LLMs](../../notes/applied-llms.md)
* [Notes on Miller (2024)](./miller-2024-notes.md)
* [Hypothesis Testing](../../notes/hypothesis-testing.md)
* [Statistics Questions](./Statistics_Questions.md)
* [Coding Questions doc](./coding-questions.md) and [Solutions Notebook](./coding-questions.ipynb)

## Gemini

### **Data Intuition (The "LLM Eval" Lens)**

Standard Google "Data Intuition" asks about ecosystem metrics (e.g., "How to measure search quality?"). For Applied AI, you must adapt this to conversational agents.

* **The Hierarchy of LLM Metrics:**
  * **N-gram based (Legacy):** BLEU, ROUGE (Know *why* these fail for chat—they measure overlap, not meaning).
  * **Embedding based:** BERTScore (Semantic similarity).
  * **Model-based (The Standard):** "LLM-as-a-judge" (Using a stronger model like Gemini Ultra to grade a smaller model's response).
  * **Human-aligned:** RLHF reward model scores, Elo ratings (Chatbot Arena style).
* **Specific Evaluation Dimensions (The "Business" side):**
  * **Factuality/Grounding:** Did the agent hallucinate? (Metric: Citation recall).
  * **Faithfulness:** Did the agent stick to the provided context (RAG)?
  * **Safety/Harm:** Jailbreak resistance, PII leakage.
  * **Utility:** Did the user actually solve their problem? (Metric: "Sessions with < 2 turns" might be good or bad depending on the task).

**Sample Staff-Level Prompt:**

> *"We are deploying a Gemini-based customer support agent for a car insurer. How do we know if it's 'good' enough to launch? How do we monitor it post-launch?"*
> **Staff-Level Answer Structure:**
> 1. **Deconstruct "Good":** It's a trade-off between *Resolution Rate* (Business value), *Hallucination Rate* (Risk), and *Latency/Cost* (Engineering constraints).
> 2. **Offline Eval Strategy:** Create a "Golden Dataset" of 500 hard historical queries. Run the new model. Use `LLM-as-a-judge` to grade answers against human-written gold answers.
> 3. **Online Experimentation:** You cannot purely A/B test "satisfaction" easily. Propose proxy metrics: "Sentiment analysis of user's final message" or "Rate of escalation to human agent."
> 4. **Long-term Metric:** "Cost per Successful Resolution" vs. "Customer Churn 3 months later."
> 
> 


### Methodology for Experimentation & ML

In Applied AI, standard A/B testing is difficult because LLM responses are non-deterministic and high-variance.


* **Variance Reduction:** LLMs are noisy. How do you detect a +1% improvement in "helpfulness" without running a 6-month experiment? (Answer: Use pre-experiment covariates, CUPED, or within-subject designs if possible).
* **Counterfactual Evaluation:** "How do we know what the user *would* have done if the model gave a different answer?"
* **Feedback Loops:**
  * **Explicit:** Thumbs up/down (Sparse, biased).
  * **Implicit:** Did the user copy-paste the code? Did they re-phrase the prompt (bad signal)? Did they say "Thanks" (good signal)?
  * See [The Field Guide to Non-Engagement Signals (Pinterest)](https://medium.com/pinterest-engineering/the-field-guide-to-non-engagement-signals-a4dd9089a176)
* **RAG (Retrieval Augmented Generation) Specifics:**
  * If the answer is bad, was it the **Retrieval** (found wrong doc) or the **Generation** (read doc wrong)?
  * *Metric:* Recall@K for retrieval vs. Faithfulness for generation.




### **Technical Refresher Checklist**

* **LLM Papers:** Read up on *Constitutional AI* (Anthropic) or *RLHF* basics (OpenAI/DeepMind). Know what "PPO" and "DPO" are at a high level.
* **Metrics:** Precision/Recall/F1 (standard), but also **RAGAS** (Retrieval Augmented Generation Assessment) framework concepts.
* **Stats:** Bootstrapping, Confidence Intervals, Power Analysis (specifically for ratio metrics).

### **Suggested Video Resource**

This video covers the nuance of "Product Sense" and "Metric Design" which is 80% of the "Data Intuition" rounds. While generic to DS, the *structure* of the answers (Clarify -> Framework -> Metric -> Trade-off) is exactly what Google looks for.

[Data Scientist Interview - Product Metric Question (Facebook / Google)](https://www.google.com/search?q=https://www.youtube.com/watch%3Fv%3DKzK8IuS88iU)

*Relevance: This video demonstrates the "Framework" approach to metric design (Clarifying the goal, defining user segments, and selecting counter-metrics), which is the critical skill for your "Data Intuition" rounds, even when applied to LLMs.*

## Claude

### Part 1: LLM Evaluation Fundamentals

**Reference-based metrics** — comparing model outputs to ground truth
- BLEU, ROUGE, METEOR for text similarity
- Exact match, F1 for extractive tasks
- Limitations: multiple valid answers exist, surface-form matching misses semantic equivalence

**Reference-free / Model-based evaluation**
- LLM-as-judge approaches (using a stronger model to score outputs)
- Reward models and preference modeling
- Human-AI agreement metrics (Cohen's kappa, Krippendorff's alpha)

**Human evaluation design**
- Rating scales (Likert, pairwise preference, best-worst scaling)
- Inter-annotator agreement and when it matters
- Cost-quality tradeoffs, when to use human eval vs. automated

**Dimensions to Evaluate in Conversational Agents.** Given they build customer service and ordering agents, expect questions about measuring:
- **Factual accuracy / groundedness** — does the agent hallucinate?
- **Task completion rate** — did the user accomplish their goal?
- **Conversation efficiency** — turns to resolution, handle time
- **Safety and policy compliance** — refusals, toxicity, PII handling
- **User satisfaction** — CSAT, NPS, or proxy metrics
- **Latency and cost** — especially for production systems

Sample Questions to Prepare
- "How would you design an evaluation framework for a customer service agent?"
- "A model scores well on automated metrics but users complain it's unhelpful. How do you investigate?"
- "How do you evaluate whether an agent is hallucinating vs. being appropriately uncertain?"
- "Design a metric for measuring whether an agent successfully completes a food order."


Statistics & Experimentation
* **Hypothesis testing**
    - Power analysis and sample size calculation
    - Multiple comparison corrections (Bonferroni, FDR)
    - When parametric vs. non-parametric tests apply
* **A/B testing and causal inference**
    - Randomization unit selection (user vs. session vs. query)
    - Network effects and interference
    - Regression discontinuity, difference-in-differences, synthetic control
    - Metric sensitivity and guardrail metrics
* **Bayesian approaches**
    - Bayesian A/B testing, credible intervals
    - When Bayesian methods are preferable (early stopping, continuous monitoring)

LLM-Specific Experimentation Challenges
- **High variance in outputs** — same prompt can yield different responses
- **Prompt sensitivity** — small changes cause large metric swings
- **Evaluation cost** — human eval doesn't scale, automated metrics are imperfect
- **Temporal drift** — model behavior may change with updates

Sample Questions
- "You're launching a new prompt template. Design an experiment to measure its impact on task completion."
- "Your A/B test shows a 2% improvement in CSAT but a 5% increase in handle time. How do you decide whether to launch?"
- "How would you set up a metric to detect model degradation over time?"

### ML Modeling & Data Intuition

ML Concepts Relevant to LLM Applications
- Fine-tuning vs. prompting vs. RAG tradeoffs
- Distillation (training smaller models from larger ones)
- Calibration and confidence estimation
- Active learning for labeling efficiency

Expect open-ended case questions like:
- "Agent resolution rate dropped 10% last week. Walk me through your investigation."
- "We have 1M unlabeled conversations. How would you prioritize what to label for evaluation?"
- "Users in one region have lower satisfaction scores. What hypotheses would you explore?"

Framework for these questions
1. Clarify the metric definition and data source
2. Segment the data (time, user type, agent version, conversation type)
3. Propose hypotheses ranked by likelihood
4. Suggest specific analyses or experiments to disambiguate



### Resources

- **Anthropic's work on constitutional AI and model evaluation** — relevant to safety/alignment eval
- **Google's HELM benchmark paper** — comprehensive LLM evaluation framework
- **"LLM-as-Judge" paper** (Zheng et al.) — directly relevant to automated evaluation
- **Trustworthy LLMs survey papers** — covers hallucination detection, calibration


## ChatGpt


### Data intuition + statistics (LLM-eval lens)

Typical shapes of questions:

* “We changed the system prompt and human ratings went up. How do you know this is real?”
* “Offline eval improved, online metrics regressed. What happened?”
* “Two annotators disagree 30% of the time. What do you do?”

You should be fluent discussing:
* Inter-rater reliability (Cohen’s κ, Krippendorff’s α) and when they fail.
* Why binomial assumptions break down with LLM outputs.
* When you’d prefer bootstrap or permutation tests over asymptotics.
* Simpson’s paradox in multi-domain LLM evaluations.


### Data intuition + business / product case


A canonical frame you should internalize:
* What is the **user goal**?
* What behavior is the model supposed to change?
* What observable signals correlate with that behavior?
* What are the known failure modes of those signals?

For example, if the agent is for call centers:
* Is success shorter calls, higher CSAT, fewer escalations, or revenue?
* Which of those can be causally attributed to the model?
* What happens when the model “sounds confident but is wrong”?

Expect to be asked to design:
* an evaluation framework,
* an experiment,
* and a post-launch monitoring plan.


### LLM evaluation fundamentals (must be fluent)

* Offline vs online evaluation and why correlation between them is fragile.
* Human-in-the-loop evaluation:
  * rubric design,
  * calibration,
  * anchoring bias,
  * drift over time.
* Reference-free evaluation (LLM-as-judge):
  * when it works,
  * when it catastrophically doesn’t,
  * how to validate it empirically.
* Distribution shift:
  * prompts,
  * domains,
  * user intent.

### Statistics & experimentation (LLM-specific framing)
A strong answer often includes: “I’d start simple, then test whether assumptions hold.”

You should be comfortable discussing:
* Non-IID data (conversation turns are correlated).
* Clustered standard errors at user or session level.
* Sequential testing and peeking.
* CUPED-style variance reduction and when it’s invalid for LLM outputs.
* Multiple testing in metric exploration.

### Metrics & measurement theory (this is the sleeper topic)

They care deeply about:
* construct validity (does the metric measure the thing?),
* sensitivity vs robustness,
* gaming and Goodhart’s Law.

Be ready to say:
* “This metric is useful short-term but dangerous long-term.”
* “This metric should be directional, not optimized.”



## Resources

### Statistics by Topic

#### Hypothesis Testing, Experimentatal Design, etc.
* [Casella and Berger - Statistical Inference](../../../Books/Casella_Berger_Statistical_Inference.pdf)
  * Chapter 7 - Point Estimation
  * Chapter 8 - Hypothesis Testing
  * Chapter 9 - Interval Estimation
* Larsen and Marx
  * Chapter 6 - Hypothesis Testing
  * Chapter 7 - The Normal Distribution
  * Chapter 9 - Two-Sample Problems
  * Chapter 12 - Analysis of Variance
  * Chapter 13 - Randomized Block Designs?
* Rice - Mathematical Statistics and Data Analysis
  * Chapter 11 - Comparing Two Samples
  * Chapter 12 - Analysis of Variance
  * Chapter 13 - The Analysis of Categorical Data

#### Regression Analysis

* Larsen and Marx, Chapter 11 - Regression

* Rice - Mathematical Statistics and Data Analysis, Chapter 14 - Linear Least Squares

* Efron and Hastie - Computer Age Statistical Inference, Chapter 8 - Generalized Linear Models and Regression Trees

* Kennedy - A Guide to Econometrics
  * 6 - Violating Assumption One: Wrong Regressors, Nonlinearities, and Parameter Inconstancy
  * 7 - Violating Assumption Two: Nonzero Expected Disturbance
  * 8 - Violating Assumption Three: Nonspherical Disturbances
  * 9 - Violating Assumption Four: Measurement Errors and Autoregression
  * 10 - Violating Assumption Four: Simultaneous Equations Models
  * 11 - Violating Assumption Five: Multicollinearity
  * 14 - Dummy Variables
  * 15 - Qualitative Dependent Variables
  * 16 - Limited Dependent Variables
  * 17 - Panel Data

### Statistics by Reference

#### [Casella and Berger - Statistical Inference](../../../Books/Casella_Berger_Statistical_Inference.pdf)
* Chapter 7 - Point Estimation
* Chapter 8 - Hypothesis Testing
* Chapter 9 - Interval Estimation

#### Larsen and Marx
* Chapter 6 - Hypothesis Testing
* Chapter 7 - The Normal Distribution
* Chapter 9 - Two-Sample Problems
* <span style="color: grey">*Chapter 8 - Categorizing Data*</span>
* <span style="color: grey">*Chapter 10 - Goodness-of-Fit Tests*</span>
* Chapter 11 - Regression
* Chapter 12 - Analysis of Variance
* Chapter 13 - Randomized Block Designs

#### Rice - Mathematical Statistics and Data Analysis
* Chapter 11 - Comparing Two Samples
* Chapter 12 - Analysis of Variance
* Chapter 13 - The Analysis of Categorical Data
* Chapter 14 - Linear Least Squares

#### Efron and Hastie - Computer Age Statistical Inference
* Chapter 8 - Generalized Linear Models and Regression Trees
* Chapter 10 - The Jackknife and the Bootstrap
* Chapter 11 - Boostrap Confidence Intervals
* Chapter 12 - Cross-Validation and $C_p$ Estimates of Prediction Error
* Chapter 15 - Large-Scale Hypothesis Testing and FDRs
* Chapter 17 - Random Forests and Boosting

#### Kennedy - A Guide to Econometrics
* 6 - Violating Assumption One: Wrong Regressors, Nonlinearities, and Parameter Inconstancy
* 7 - Violating Assumption Two: Nonzero Expected Disturbance
* 8 - Violating Assumption Three: Nonspherical Disturbances
* 9 - Violating Assumption Four: Measurement Errors and Autoregression
* 10 - Violating Assumption Four: Simultaneous Equations Models
* 11 - Violating Assumption Five: Multicollinearity
* 14 - Dummy Variables
* 15 - Qualitative Dependent Variables
* 16 - Limited Dependent Variables
* 17 - Panel Data

### LLM Evaluation Papers
* [paper](./2411.00640v1.pdf)
* [paper](./2503.01747v3.pdf)
* [paper](./2504.21303v1.pdf)
