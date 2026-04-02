# Data Intuition


## Regression fundamentals with a twist

Your confidence/prediction interval question is a perfect example of this category. The trap: people know the formulas but haven't thought about what they imply geometrically. Related questions you might get:

- What happens to R² as you add more predictors, regardless of whether they're useful? (It always increases or stays flat — adjusted R² exists for this reason.)
- You fit a linear regression on training data. A new observation has a very extreme value of x. How does this affect your prediction uncertainty? (Prediction interval widens — extrapolation.)
- You have a regression with two predictors that are very highly correlated. What breaks and what doesn't? (Inference — standard errors blow up, coefficients unstable. But *predictions* in the range of observed data may still be fine. This trips people up.)
- You add a variable to a regression and the coefficient on an existing variable flips sign. What does this mean? (Classic omitted variable / confounding — but can you articulate *why* without just saying "multicollinearity"?)


## P-values and hypothesis testing: conceptual precision

The blog's Q2, Q6, and Q10 all live here, and the post says these were among the most discriminating. The theme is that people have memorized the conclusion but not the mechanics.

- What does a p-value actually mean? What doesn't it mean? (A common wrong answer: probability the null is true.)
- You run an A/A test and 7% of your p-values are below 0.05. Is your platform broken? (No — slightly above 5% is within sampling noise for 100 tests. But this is a judgment call and the reasoning matters.)
- A result is statistically significant. A second, very similar result is not. Can you conclude these two results are different from each other? (The blog's Q10 — this is the Gelman/Stern point, "the difference between significant and not significant is not itself significant.")
- You run a test and get p = 0.051. Your colleague says "we just barely missed significance." What's wrong with this framing? (The threshold is arbitrary; the effect size and uncertainty are what matter.)
- When is a hypothesis test *overpowered*? What's the symptom and why is it a problem? (With huge n, trivially tiny effects become significant — the blog's Q6.)


## Simpson's paradox and aggregation traps

The blog tested this without naming it (Q9). Your restaurant question fits here too — the key insight being that what looks like a layout problem might actually be a selection problem that requires survey data to disentangle.

- UC Berkeley admissions: overall acceptance rate favors men, but within each department it favors women. How? (Simpson's paradox — department composition differs.)
- A drug appears effective overall but is harmful in both young and old patients separately. How is this possible?
- Average salary at a company increases, but average salary for every job title decreases. Is this possible? (Yes — if higher-paying roles are shrinking as a share of the workforce.)
- You split a metric by A and B (two exhaustive, mutually exclusive groups) and see it declining in both. Can the aggregate be increasing? (Yes — Simpson's paradox. The blog's Q9.)

## Measurement and study design (your restaurant example)

This is the category most over-looked by people who prep on modeling. The key question to ask yourself is always: "do we even have the right data, or do we need to collect it differently?"

- You want to know which version of a UI drives more purchases. You have clickstream data. What can and can't you conclude from it? (You can measure behavior, not preference or intent — survey gets at the latter.)
- A company wants to know if employees are happy with a new policy. They look at internal Slack messages. What's wrong with this? (Selection bias — who posts on Slack, sentiment analysis limitations, social desirability.)
- You want to measure the effect of a new restaurant menu layout on customer satisfaction. What are your options, and what are the tradeoffs of each? (Observational data, A/B test, survey — each with distinct assumptions.)
- You observe that people who eat breakfast tend to live longer. What study design would you need to make this causal? (RCT ideally, but why is observational analysis insufficient here?)


## Bias-variance and model complexity (without the jargon)

The blog's Q5 (the k-NN question) tested this without using the words "bias" or "variance." You should expect it framed in applied terms.

- You're building a model to predict house prices. Your training error is very low but test error is high. What are the possible causes and how would you diagnose them?
- A simple decision tree with depth 2 and a neural network with 10 layers are both fit to the same dataset. The neural net does better on training data. When would you prefer the tree?
- You're averaging predictions from k neighbors. As k increases, what happens to predictions? What's the tradeoff? (Smooths out noise but introduces bias by pulling in distant, dissimilar points.)


## Practical data intuition

- You have a dataset where 95% of values are "normal" and 5% are labeled "fraud." You build a classifier that predicts "not fraud" for everything. What's its accuracy? Why is this useless? (95% accurate — accuracy is the wrong metric for imbalanced classes.)
- You want to remove outliers by cutting anything beyond 3 standard deviations. What's the problem? (The blog's Q7 — SD is itself affected by outliers; use IQR-based methods instead.)
- You're analyzing survey responses on a 1–5 Likert scale and want to compare groups. Someone proposes a t-test. What concerns do you have? (Ordinal data, not continuous; but also: for large samples the CLT often makes this defensible — the answer is nuanced.)
- You have a very large dataset and everything is statistically significant. What do you do? (Focus on effect sizes, practical significance, and whether the effects are actionable.)


## Fundamental Statistics & Probability

**Q: In a simple linear regression, how does the width of the confidence interval for the predicted mean change as the predictor variable ($x$) moves further away from its sample mean ($\bar{x}$)?**
* **Answer:** The interval gets wider. This is because the uncertainty in the slope estimate ($\beta_1$) has a larger impact the further you move from the "anchor" point of the data ($\bar{x}, \bar{y}$). At the mean of $x$, the variance of the prediction is minimized.


**Q: You are testing a sensor that identifies pedestrians. If the probability of a "False Positive" is 1% and the probability of a "True Positive" is 99%, but pedestrians only appear in 0.1% of frames, what is the probability that a detection is actually a pedestrian?**
* **Answer:** This is a classic Bayes' Theorem problem (and a version of the "Base Rate Fallacy").
    * $P(\text{Pedestrian}) = 0.001$
    * $P(\text{Detection}|\text{Pedestrian}) = 0.99$
    * $P(\text{Detection}|\text{No Pedestrian}) = 0.01$
    * $P(\text{Pedestrian}|\text{Detection}) = \frac{0.99 \times 0.001}{(0.99 \times 0.001) + (0.01 \times 0.999)} \approx 9\%$
    * **Insight:** Even with a "99% accurate" sensor, most detections are false alarms because the event is so rare.

**Q: If you have two independent estimators for the same value, $\hat{\theta}_1$ and $\hat{\theta}_2$, with variances $\sigma_1^2$ and $\sigma_2^2$, how would you combine them to create a new estimator with the lowest possible variance?**
* **Answer:** Use a weighted average where the weights are inversely proportional to the variances: $w_i = \frac{1/\sigma_i^2}{\sum 1/\sigma_j^2}$. This is the foundation of sensor fusion (and the Kalman Filter).

## Developing "AV" Data Intuition (The "Simple" Case Study)

These questions focus on data collection and the "ground truth" problem, similar to your "restaurant survey" experience.

**Q: We want to know if our autonomous vehicles are perceived as "polite" by human drivers. We have all the vehicle's internal log data (speed, braking, distance to others). How should we measure this?**
* **Answer:** The key insight here is that "politeness" is a human perception, not a physics calculation. Before building a model, you need **ground truth from humans**.
    * **The "Simple" Solution:** Conduct a study where human drivers watch video clips of the AV's behavior and rate it on a "politeness" scale. 
    * **The Data Step:** Once you have those labels, you can then correlate them with your internal log data (e.g., "jerk" or "distance at lane change") to see which physical metrics actually map to human perception.

**Q: We are seeing a high rate of "disengagements" (human taking over) at a specific intersection. You have the video and the logs. How do you determine if the software is actually failing or if the human is just being over-cautious?**
* **Answer:** You need a **Counterfactual Comparison**. 
    * Take the exact state of the world at the moment of disengagement and run it through a high-fidelity simulator *without* the human intervention.
    * If the simulated car clears the intersection safely, the human was over-cautious. If the car gets stuck or has a safety violation, the software failed. This "Simulation vs. Reality" comparison is a foundational data intuition at Waymo.

**Q: You want to evaluate a new "Comfort Mode" for passengers. You decide to run an A/B test by enabling it on half the fleet in Phoenix. What is a potential pitfall of this design?**
* **Answer:** **Selection Bias/Interference.**
    * If the "Comfort Mode" makes the cars move more slowly or cautiously, it might change the traffic patterns for the "Control" cars in the same area. 
    * Alternatively, if certain users always get the "Comfort" car, their feedback might be skewed by the "Novelty Effect." 
    * **The "Simple" Solution:** Switchback testing (randomizing by time blocks rather than by car) or using a "Pre-Post" analysis on the same vehicles to control for individual car/route differences.


################################################################################################


**Q: In a simple linear regression ($Y = \beta_0 + \beta_1X + \epsilon$), what happens to the width of the prediction interval for $Y$ as $X$ gets very far from the mean of your training data ($\bar{x}$)?**
* **The "Undergrad" Answer:** It gets wider.
* **The "Insight" Waymo Wants:** The variance of the prediction has two components: the variance of the estimate of the mean and the variance of the error term. The first part depends on the distance $(X - \bar{x})^2$. **Why it matters for AV:** If we use a model trained on sunny-day data to predict behavior in a blizzard (a "far-out" covariate), our confidence intervals will explode, signaling that the model's prediction is statistically unreliable for that environment.

**Q: You run an A/B test on a new perception model. You calculate a p-value of 0.04. If the Null Hypothesis is actually true (the model is no better), what is the distribution of the p-value if you were to run this experiment 1,000 times?**
* **The Insight:** The p-value is **Uniformly distributed** $[0, 1]$ under the null.
* **Why it matters:** Many candidates think it would be "clustered near 1." Understanding it's uniform helps you realize that "fishing" for a 0.05 result is just a matter of time and multiple testing.

**Q: You want to reduce the Margin of Error of your estimate by half. How much do you need to increase your sample size?**
* **The Answer:** 4x (Quadruple).
* **The Insight:** The Margin of Error scales with $1/\sqrt{n}$. In AV, where "miles" are expensive to drive, you have to be honest about the diminishing returns of just "driving more" versus finding a more efficient sampling method or a more precise "surrogate" metric.


**Q: "Our passengers are reporting that the car feels 'jerky' during left turns, but our IMU (accelerometer) data shows the G-forces are within the 'Smooth' range. How do you investigate?"**
* **The Over-Prep Trap:** "I'd build a Time-Series Anomaly Detection model using LSTMs..."
* **The "Google" Insight:** **Go to the Source (Survey/Labeling).** * **Step 1:** Ask the passengers *when* they felt the jerk (was it the start, the middle, or the end of the turn?). 
    * **Step 2:** Have human experts watch the video of those specific turns. 
    * **The Discovery:** It might not be "physical jerk" (acceleration), but "visual jerk"—maybe the car is steering smoothly but the steering wheel is twitching, or the car is "inching" forward in a way that feels indecisive to a human. You can't see "indecisiveness" in an accelerometer; you need a human label.

**Q: "We want to know if our AVs are 'better' than human drivers at yielding to pedestrians. We have all our AV data, but we don't have data on every human driver in the world. How do you set up a fair comparison?"**
* **The Insight:** **Create a "Naturalistic" Control Group.** * Don't try to find a global "human average." Instead, find a specific intersection where Waymo cars drive. 
    * Set up a stationary camera (or use AV sensor data from a "shadow mode") to observe how humans behave at *that same intersection* at the same time of day. 
    * **The Key:** You are controlling for the "environment difficulty" so the only variable left is the "Driver" (AI vs. Human).

**Q: "We are considering changing the voice of the Waymo assistant. We want to know if it improves the 'Rider Experience.' What is the single most important metric, and how do you get it?"**
* **The Insight:** **Don't use proxy metrics for sentiment.** * Candidates often suggest "retention" or "ride frequency." But those are noisy and take months to move. 
    * **The Simple Solution:** A **post-ride survey** (NPS or "How much did you like the voice?"). If the user mentions the restaurant menu survey, this is exactly the same logic: **Subjective preferences require subjective data collection.**

## Summary Checklist for the Interview
* **Think like a Scientist, not a Scripter:** If the data is messy, don't ask "what library cleans this?" Ask "What is the physical process that created this noise?"
* **Ground Truth is King:** If you don't have a label for what you're measuring, your first step in any case study should be: **"How do I get a human to label this correctly?"**
* **Linearity & Basics:** Brush up on the **Gauss-Markov assumptions** for linear regression. They often ask "What happens if [Assumption X] is violated?" (e.g., Heteroscedasticity or Multicollinearity).
* **Regression Basics:** Know how $R^2$ changes with added variables, the difference between confidence and prediction intervals, and the impact of outliers (leverage).
* **Probability Basics:** Be ready for "balls in urns" or "coin flip" questions that can be mapped to sensor triggers or success/failure rates.
* **Experimental Logic:** If asked to measure something subjective, always suggest **Human Labeling/Surveys** as the first step for ground truth.
* **Metric Sanity:** If a metric (like "average speed") goes up, ask "is that actually good?" (e.g., faster isn't safer). Look for the trade-off.

## Takeaway

The throughline across all of these: the "right" answer usually requires you to hold two things in your head simultaneously that most people don't — e.g., statistical significance vs. practical significance, aggregate vs. subgroup trends, what a test *can* detect vs. what it *can't* conclude. The blog's authors are quite clear that this is what separates good candidates from strong ones.
