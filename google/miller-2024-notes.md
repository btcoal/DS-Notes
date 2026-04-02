# Notes on Miller (2024) "Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations"

Start with the hypothetical *super-population* of all possible questions we could pose to a LLM.[^1]

[^1]: ~~Doesn't this introduce another source of uncertainty? How to think about this?~~ This shows up in the "between-question" variance that Miller later calls $\mathrm{Var}(x)$, which is separate from the "within-question" variance $\mathrm{Var}(\varepsilon)$ that we can reduce by resampling.

An eval consists of $n$ independently drawn questions. ***(How many in a typical eval?)***

The score on question $i$ is $s_i$. Miller decomposes the score into a "mean" component and a zero-mean "random" component:
$$s_i = x_i + \varepsilon_i$$

**What does this mean?**

$x_i$ is the "true" score for question $i$, that is the expected score if we could ask that question infinitely many times (and $Var(X)$ is the dispersion of scores if we asked many different sets of questions.) $\varepsilon_i$ is the noise due to randomness in the model's response. What is (are) the source(s) of randomness in this score?

* $x_i = \mathbb{E}[s_i \mid \text{question } i]$ is the *conditional mean score*.
* $\varepsilon_i = s_i - x_i$ is a zero-mean residual capturing randomness conditional on that question, the *conditional variance*, the *within-question* variance.

The question $i$ is fixed. The randomness comes from how the model produces an answer and how we score that answer.[^2]

[^2]: See [Horace He, "Defeating Nondeterminism in LLM Inference"](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

1. **stochastic decoding**. Most evals sample from the model rather than taking a deterministic argmax. Temperature, top-p, top-k, nucleus sampling, etc., all introduce randomness. Even with the same prompt and model weights, repeated runs can yield different outputs, hence different scores. This is the dominant source in most modern LLM evals.
<!-- TODO: Find examples among frontier models of the decoding process for evals on benchmarks -->

2. **internal nondeterminism in inference**. GPU kernels, reduced-precision arithmetic, parallelism, and non-deterministic execution order can introduce tiny variations that sometimes flip tokens or downstream scoring outcomes. This is usually smaller than sampling noise, but it is real.
<!-- TODO: Can we quantify? How would we? -->

3. **scoring noise induced by discrete evaluation rules**. Many evals map a complex output to a coarse score: correct/incorrect, pass/fail, exact-match, or thresholded F1. Even if the model’s latent “competence” on the question is stable, small variations in wording or reasoning can push the output across a scoring boundary. This converts smooth uncertainty into variance in $s_i$.
<!-- TODO: Find examples among frontier models -->

4. **randomness inside the evaluation pipeline**. Some evals use stochastic graders (LLM-as-judge), randomized tie-breaking, or sampling inside a chain-of-thought or tool-use loop. Any randomness there also ends up in $\varepsilon_i$.
<!-- TODO: Find examples among frontier models -->


What is explicitly *not* in $\varepsilon_i$ is variation due to question difficulty. That lives in $x_i$. Easy questions have high $x_i$, hard ones have low $x_i$, and the spread of those $x_i$’s across questions is the “between-question” variance that Miller later calls $\mathrm{Var}(x)$.

This decomposition separates two levers:

* Variance from *which questions you sampled* (immutable unless you change the eval).
* Variance from *how noisily the model answers a fixed question* (reducible via resampling or next-token probabilities).

That distinction underpins Miller’s variance-reduction arguments later in the paper.

The unconditional versions are:
$$s = x + \varepsilon$$
where
* $s$ is the "true" eval score is the overall eval score,
* $x$ is the "mean" component of $s$ across all questions,
* $\varepsilon$ is noise that makes $s$ stochastic, i.e. vary around $x$.

Since $\mathbb{E}[\varepsilon] = 0$ by construction and independence of $x$ and $\varepsilon$:
$$\mathrm{Var}(s) = \mathrm{Var}(x) + \mathrm{Var}(\varepsilon)$$

Let $\mu = \mathbb{E}[x]$ be th (unobserved) mean score for the model across the entire super-population of questions such that $\mu = \mathbb{E}[x] = \mathbb{E}[s]$. 

We want to do *inference* on $s$ (confidence intervals, p-values, etc.) given observed scores $\{s_i\}_{i=1}^n$.[^3]

Let $\hat{\mu} \equiv s = \frac{1}{n} \sum_{i=1}^n s_i$ be the sample mean score across $n$ questions. From the (Weak or Strong) Law of Large Numbers, $\hat{\mu} \xrightarrow{a.s.} \mu = \mathbb{E}[s]$ as $n \to \infty$.

For SLLN, for any $\epsilon > 0$:

$$P\left( \lim_{n \to \infty} \vert \hat{\mu} - \mu \vert < \epsilon \right) = 1$$

and for the WLLN, for any $\epsilon > 0$:

$$\lim_{n \to \infty} P(\vert \hat{\mu} - \mu \vert < \epsilon) = 1$$

See Casella and Berger, Chapter 5.

By the CLT, if $\mathrm{Var}(s) < \infty$, then as $n \to \infty$:
$$\frac{\hat{\mu} - \mu}{\sqrt{\mathrm{Var}(s)/n}} \xrightarrow{d} N(0,1)$$


Letting $SE_{CLT} \equiv \sqrt{\mathrm{Var}(s)/n}$, then

$$SE_{CLT} = \sqrt{\frac{1}{n} \big( \frac{1}{n-1} \sum_{n}(s_i - \bar{s})^2 \big)}$$

If the evals consist of True/False questions, then $s_i$ is Bernoulli and $\mathrm{Var}(s) = p(1-p)$ where $p$ is the true probability of a correct answer, then 

$$SE_{CLT} = \sqrt{\frac{p(1-p)}{n}}$$

Since the variance $\varepsilon$ is unknown,

$$\frac{\hat{\mu} - \mu}{\sqrt{SE_{CLT}}} \sim t_{n-1}$$

and we can construct a $(1-\alpha)$% confidence interval for $\mu$ as:
$$\hat{\mu} \pm t_{n-1, 1-\alpha/2} \cdot SE_{CLT}$$

For $\alpha = 0.05$, the 95% confidence interval is:
$$\hat{\mu} \pm t_{n-1, 0.975} \cdot SE_{CLT}$$

In python
```python
from math import sqrt
from scipy import stats

def standard_error(scores):
    n = len(scores)
    mu_hat = sum(scores) / n
    return sqrt(sum((s - mu_hat)**2 for s in scores) / (n * (n - 1)))

def confidence_interval(scores, alpha=0.05):
    n = len(scores)
    mu_hat = sum(scores) / n
    se_clt = sqrt(sum((s - mu_hat)**2 for s in scores) / (n * (n - 1)))
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_of_error = t_crit * se_clt
    return mu_hat - margin_of_error, mu_hat + margin_of_error
```


[^3]: Does it matter if we work within a Fisherian or Neyman-Pearsonian framework here? Fisher would use p-values to reject nulls, Neyman-Pearson would construct CIs and accept/reject based on them.

## Variance Reduction

Below is a rewritten **Section 3 (Variance reduction)** with clearer exposition, expanded math/assumptions, and a more explicit worked example. 

---

## 3 Variance reduction (rewritten)

The standard error of $\hat\mu$ tells us how noisy our reported eval score is. If we want “tighter error bars” without changing the underlying task, we need to reduce
$$
\mathrm{Var}(\hat\mu),
\quad\text{where}\quad
\hat\mu \equiv \bar s = \frac{1}{n}\sum_{i=1}^n s_i .
$$

### 3.0 Setup and assumptions

We treat an eval as a sample of $n$ questions drawn from a large (conceptual) super-population.

For each sampled question $i$:

* The observed score is $s_i$.
* The model has a **conditional mean score** $x_i \equiv \mathbb{E}[s_i \mid i]$. Intuitively: if you could rerun the model many times on the *same* question (with the same prompt and sampling settings), $x_i$ is the average score you’d get.
* The randomness from decoding (sampling), tie-breakers, stochastic graders, etc. is captured by a zero-mean residual:
  $$
  s_i = x_i + \varepsilon_i,
  \quad \mathbb{E}[\varepsilon_i \mid i] = 0,
  \quad \mathrm{Var}(\varepsilon_i \mid i) = \sigma_i^2 .
  $$
* Across questions, $((x_i,\sigma_i^2))$ vary because some questions are intrinsically easier/harder or more/less sensitive to decoding randomness.

The key point: there are **two distinct sources of variance**.

1. **Question sampling variance**: different random draws of questions lead to different sets of $x_i$’s. This shows up as $\mathrm{Var}(x)$.
2. **Within-question (conditional) variance**: even for a fixed question $i$, repeated runs produce different $s_i$ due to sampling, etc. This shows up as $\mathbb{E}[\sigma_i^2]$.

Using the law of total variance,
$$
\mathrm{Var}(s) = \mathrm{Var}\big(\mathbb{E}[s\mid i]\big) + \mathbb{E}\big[\mathrm{Var}(s\mid i)\big]
= \mathrm{Var}(x) + \mathbb{E}[\sigma_i^2].
$$

Since $\hat\mu$ is the mean of $n$ question scores,
$$
\mathrm{Var}(\hat\mu) = \frac{\mathrm{Var}(s)}{n}
= \frac{\mathrm{Var}(x) + \mathbb{E}[\sigma_i^2]}{n}.
$$

**Implication:** increasing $n$ always helps, but only one part of the variance—$\mathbb{E}[\sigma_i^2]$—is reducible without changing the question set.

---

### 3.1 Resampling (multiple generations per question)

A straightforward way to reduce the within-question component ($\mathbb{E}[\sigma_i^2]$) is to answer each question multiple times and average the scores.

Suppose for each question $i$ we generate $K$ independent answers (with the same sampling settings), producing scores $(s_{i1},\dots,s_{iK})$. Define the per-question average:
$$
\bar s_i \equiv \frac{1}{K}\sum_{k=1}^K s_{ik}.
$$
We then estimate the overall eval score as
$$
\hat\mu_K \equiv \frac{1}{n}\sum_{i=1}^n \bar s_i.
$$

**What does resampling change?**
Conditioning on question $i$, the mean is still $x_i$, but the variance shrinks:
$$
\mathbb{E}[\bar s_i \mid i] = x_i,
\qquad
\mathrm{Var}(\bar s_i \mid i) = \frac{\sigma_i^2}{K}.
$$
So by the same variance decomposition,
$$
\mathrm{Var}(\hat\mu_K)
= \frac{\mathrm{Var}(x) + \mathbb{E}[\sigma_i^2]/K}{n}.
$$

This makes the diminishing-returns structure explicit:

* The **floor** you cannot beat with resampling is ($\mathrm{Var}(x)/n$), because that’s “question draw noise.”
* Resampling only attacks ($\mathbb{E}[\sigma_i^2]/(nK)$).

A practical rule is: choose (K) so that
$$
\frac{\mathbb{E}[\sigma_i^2]}{K} \ll \mathrm{Var}(x),
$$
because beyond that point you’re mostly paying extra tokens to fight a term that’s already small.

#### Worked example (expanded and cleaner)

Assume:

* Scores are binary: $s_{ik}\in\{0,1\}$.
* Question “difficulty” (i.e., expected correctness) is uniformly distributed:
  $$
  x \sim \mathrm{Uniform}(0,1).
  $$
* Given $x_i$, each sampled run is Bernoulli:
  $$
  s_{ik} \mid x_i \sim \mathrm{Bernoulli}(x_i).
  $$

Compute the two variance components:

1. **Variance of conditional means**
   $$
   \mathrm{Var}(x) = \mathrm{Var}(\mathrm{Uniform}(0,1)) = \frac{1}{12}.
   $$

2. **Expected conditional variance**
   For Bernoulli, ($\sigma_i^2 = \mathrm{Var}(s\mid x_i)=x_i(1-x_i)$). Therefore
   $$
   \mathbb{E}[\sigma_i^2] = \mathbb{E}[x(1-x)]
   = \mathbb{E}[x] - \mathbb{E}[x^2]
   = \frac{1}{2} - \frac{1}{3}
   = \frac{1}{6}.
   $$

Plugging into the formula:
$$
\mathrm{Var}(\hat\mu_K)
= \frac{\frac{1}{12} + \frac{1}{6K}}{n}
= \frac{1}{n}\left(\frac{1}{12} + \frac{1}{6K}\right).
$$

To compare with the (K=1) baseline:
$$
\mathrm{Var}(\hat\mu_1)=\frac{\frac{1}{12}+\frac{1}{6}}{n}=\frac{\frac{1}{4}}{n}.
$$
So the variance ratio is
$$
\frac{\mathrm{Var}(\hat\mu_K)}{\mathrm{Var}(\hat\mu_1)}
=======================================================

# \frac{\frac{1}{12} + \frac{1}{6K}}{\frac{1}{4}}

\frac{1 + 2/K}{3}.
$$

Now the numbers are easy to read:

* (K=1): ratio (=1).
* (K=2): ratio (=(1+1)/3=2/3) → **33% variance reduction**.
* (K=4): ratio (=(1+0.5)/3=1/2) → **50% reduction**.
* (K\to\infty): ratio (=1/3) → the **maximum** reduction from resampling here is **67%**, because the ($\mathrm{Var}(x)$) term remains.

#### Why you can’t just “pool” all (nK) samples

If you take all (nK) answers and pretend they are (nK) independent questions, you will understate uncertainty: the (K) samples within the same question share the same underlying ($x_i$), so they are not independent draws from the super-population. The right unit of independence is the **question**, not the **generation**. Operationally: compute the statistic per question (e.g., ($\bar s_i$)), then compute uncertainty across questions (and across clusters if questions are grouped, as in Section 2.2).

---

### 3.2 Next-token probabilities (eliminating conditional variance)

Resampling reduces ($\mathbb{E}[\sigma_i^2]$) by a factor of ($1/K$). In some eval formats, you can do even better: make ($\sigma_i^2 = 0$) by avoiding sampling altogether.

Consider a multiple-choice eval where:

* The model’s answer is determined by the probability of a particular token (or a small set of tokens) at a specified position.
* You can access the model’s next-token probabilities (logits / logprobs).

Let (p_i) be the model’s probability assigned to the correct option at the decision point. If the scoring rule is “1 for correct, 0 for incorrect,” then the *expected* score on that question is exactly (p_i). If we set
$$
s_i \equiv p_i,
$$
then there is no within-question randomness:
$$
x_i = p_i,\qquad \varepsilon_i = 0,\qquad \sigma_i^2=0.
$$
So
$$
\mathrm{Var}(\hat\mu)=\frac{\mathrm{Var}(p)}{n}.
$$

In the worked example above, moving from sampled 0/1 grading to probability-based scoring removes the ($\mathbb{E}[\sigma_i^2]$) term entirely—equivalent to taking ($K\to\infty$) in resampling—so the best-case variance reduction matches the same upper limit in that example.

Two caveats are doing real work here:

* This only applies when the eval can be expressed as a clean probability query (often multiple-choice, sometimes short-answer with constrained formats).
* If the eval requires multi-step generation (chain-of-thought, tool use, code execution, etc.), next-token probabilities at one position generally won’t represent the whole behavior you’re measuring.

---

### 3.3 Don’t touch the thermostat (why temperature-tuning is a trap)

Lowering sampling temperature can reduce output randomness, which sounds like “variance reduction.” But it changes the *data-generating process*—often changing the estimand, sometimes inflating variance in the part you can’t reduce, and sometimes introducing bias.

Formally, temperature changes the mapping from a question (i) to its conditional mean ($x_i(T)$). You are no longer estimating ($\mu = \mathbb{E}[x(T\neq 1)]$); you’re estimating ($\mathbb{E}[x(T\neq 0)]$) (or whatever temperature you picked). Those are different quantities unless you have a strong reason to believe they coincide.

It can also perversely increase the irreducible component ($\mathrm{Var}(x)$). A stylized illustration:

* Suppose at ($T=1$), question-level means are spread smoothly: ($x(T\neq 1)\sim \mathrm{Uniform}(0,1)$), so ($\mathrm{Var}(x)=1/12$).
* At ($T=0$), the model becomes essentially deterministic: it “rounds” behavior into always-correct vs always-wrong based on whether ($x(T\neq 1)$) is above 0.5:
  $$
  x(T\neq 0) = \mathbf{1}{x(T\neq 1) > 0.5},
  $$
  which yields a Bernoulli(0.5) distribution and ($\mathrm{Var}(x)=1/4$), **tripling** the minimum achievable variance.

Even worse, the temperature change can shift the mean, not just the variance, because the deterministic choice may not preserve expected correctness.

So: if you care about variance, attack it with (i) resampling or (ii) probability-based scoring, not by changing the sampling regime out from under the measurement.

### 3.4 Practical recommendation (how to pick a method)

* **If next-token probabilities are available and the eval can be expressed in that framework:** compute per-question expected scores (e.g., $p_i$), then compute uncertainty across questions (cluster-adjust if needed).
* **Otherwise:** resample each question $K$ times, average within question, and choose $K$ so that ($\mathbb{E}[\sigma_i^2]/K$) is small relative to ($\mathrm{Var}(x)$). Then compute uncertainty across question-level means (again, cluster-adjust if needed).
* **Do not** reduce temperature as a variance trick unless your scientific question is explicitly “How does the model perform at this temperature?”

---

### Expanded 3.2 Next-token probabilities (eliminating conditional variance) — expanded + implementation grounded
You’re right to push on this: “use next-token probabilities” is one of those phrases that sounds concrete until you try to implement it and realize there are a bunch of sharp edges (tokenization, prompt format, multi-token answers, etc.). Below is an expanded, implementation-grounded rewrite of **3.2** that makes the estimand and mechanics explicit, and shows what you actually do in code and in an eval harness. 

---

## 3.2 Next-token probabilities (eliminating conditional variance) — expanded + implementation grounded

The big idea is simple:

* In many evals, we score the model by **sampling one completion** and then grading it as correct/incorrect (or with partial credit). That introduces *within-question randomness* from decoding.
* If, instead, we compute the model’s **probability of each possible answer** at the decision point and score using those probabilities, then for a fixed question (i) the score becomes **deterministic** (given fixed model weights + prompt). That sets the conditional variance term ($\mathrm{Var}(s\mid i)$) to zero.

### A. What “conditional variance” is in practice

Suppose question (i) is multiple choice with one correct option. If we run the model once with temperature ($T>0$), we get a sampled answer ($A_i$) and a binary score ($s_i \in \{0,1\}$). If we repeat that run many times on the *same* question, we’ll observe a distribution of scores because sometimes the model samples the right option and sometimes it doesn’t. That variability is exactly:

$$
\sigma_i^2 \equiv \mathrm{Var}(s \mid i).
$$

If your reported metric is the mean score (\hat\mu = \frac{1}{n}\sum_i s_i), then the variance decomposition is:

$$
\mathrm{Var}(\hat\mu)
= \frac{1}{n}\mathrm{Var}(s)
= \frac{1}{n}\Big(\underbrace{\mathrm{Var}(x)}_{\text{question-to-question}} + \underbrace{\mathbb{E}[\sigma_i^2]}_{\text{within-question decoding noise}}\Big),
$$
where ($x_i = \mathbb{E}[s \mid i]$).

Resampling ($K$ completions per question) reduces ($\mathbb{E}[\sigma_i^2]$) by ($1/K$). Next-token probabilities go further: they aim to make ($\sigma_i^2 = 0$).

### B. When you can set ($\sigma_i^2 = 0$)

You can do this when the answer can be treated as a **small, discrete choice** at a specific point in the output. The cleanest case is *multiple choice* where you force the model to output exactly one of a small set of symbols.

Concretely, you design the prompt so that the model’s next token (or next few tokens) must be one of ${\text{"A"},\text{"B"},\text{"C"},\text{"D"}}$ (or “1/2/3/4”, etc.). Then you query the model for the probability it assigns to each option **without sampling**.

Let:

* ($p_{i,j}$) be the model’s probability that option ($j$) is the next output (given the prompt for question ($i$)).
* ($j^*$) be the correct option.

Then define the **probability-scored** per-question value:
$$
s_i \equiv p_{i,j^*}.
$$

This is not “the sampled correctness.” It is the model’s **predicted probability of being correct** under that forced response format. For fixed model + prompt, ($s_i$) is deterministic, so:
$$
\mathrm{Var}(s\mid i) = 0.
$$

Your overall eval estimator becomes:
$$
\hat\mu_{\text{prob}} = \frac{1}{n}\sum_{i=1}^n p_{i,j^*}.
$$

This is literally the mean of per-question correctness probabilities, which is the expected accuracy you would get if you *did* sample one option from the model’s implied distribution (under that forced format). But you no longer pay the sampling noise.

### C. How this changes uncertainty (explicitly)

With probability scoring, the only randomness left is **which questions you happened to sample** (and any clustering structure in how questions were built). The variance becomes:

$$
\mathrm{Var}(\hat\mu_{\text{prob}}) = \frac{\mathrm{Var}(p_{i,j^*})}{n},
$$
(or cluster-robust analogs if questions share passages/subjects).

Implementation consequence: your standard error is now computed across the *deterministic* values ($p_{i,j^*}$), not across stochastic 0/1 outcomes.

### D. Implementation recipe (what you actually do)

#### Step 1: Force a discrete “answer channel”

You want the answer to occur at a predictable position, e.g.:

> “Answer with exactly one letter: A, B, C, or D. Do not include punctuation or words. Output only the letter.”

Then add a literal prefix so the probability query is at a stable boundary:

> “Final answer: ”

Now the next token after `"Final answer: "` is (ideally) one of those letters.

#### Step 2: Query the model for next-token distribution

You run the model in a mode that returns:

* the log-probability (or probability) assigned to candidate tokens at that next position.

Many inference stacks support this:

* Local inference (e.g., HuggingFace / vLLM) can return logits for the next token.
* Hosted APIs often have a “return logprobs” option.

You do **not** sample a completion here. You just read the distribution.

#### Step 3: Convert logits/logprobs to probabilities over your answer set

Two important details:

1. **Tokenization mismatch.**
   “A” might be a different token than “ A” (leading space). If your prompt ends with `"Final answer:"` vs `"Final answer: "` you’ll get different token boundaries. Decide on one and test it.

2. **Renormalize to your candidate set.**
   Even if you ask for only A/B/C/D, the model will still assign some probability mass to other tokens. You have two options:

* **Absolute probability**: use the model’s ($P(\text{"A"})$) directly. This treats probability mass on “other junk tokens” as genuine uncertainty/format failure.
* **Conditional-on-format probability**: renormalize within (${A,B,C,D}$):
  $$
  \tilde p_{i,j} \equiv \frac{p_{i,j}}{\sum_{k\in{A,B,C,D}} p_{i,k}}.
  $$
  This answers: “given that the model outputs one of the allowed options, what’s the probability it picks each one?”

Which you want depends on what you’re measuring. If format adherence matters, don’t renormalize. If you want a clean measure of *knowledge conditional on obeying the output contract*, renormalize.

#### Step 4: Score and aggregate

For accuracy-style scoring:

* per question $s_i = p_{i,j^*}$ (or $\tilde p_{i,j^*}$ if renormalized)
* overall ($\hat\mu = \frac{1}{n}\sum_i s_i$)

For uncertainty:

* treat (${s_i}$) as your per-question “observations”
* compute standard error across questions:
  $$
  \widehat{\mathrm{SE}}(\hat\mu) = \sqrt{\frac{\widehat{\mathrm{Var}}(s)}{n}}.
  $$
* if you have clusters (passage-level, subject-level), compute a cluster-robust SE using cluster IDs (same mechanics as your Section 2.2 clustering discussion, just applied to (${s_i}$) values).

### E. A worked mini-example that actually mirrors implementation

Assume 3 questions, each with 4 choices A/B/C/D, and you’ve queried next-token probabilities after `"Final answer: "`.

You observe (already renormalized to A–D for simplicity):

* Q1 correct = B, probs: A 0.10, B 0.70, C 0.10, D 0.10 → (s_1 = 0.70)
* Q2 correct = D, probs: A 0.25, B 0.25, C 0.25, D 0.25 → (s_2 = 0.25)
* Q3 correct = A, probs: A 0.40, B 0.30, C 0.20, D 0.10 → (s_3 = 0.40)

Then:
[
\hat\mu_{\text{prob}} = (0.70+0.25+0.40)/3 = 1.35/3 = 0.45.
]

Interpretation: under the forced A–D output contract, the model’s expected accuracy on this question set is 45%.

Now compute uncertainty across questions (toy-size, but mechanics are real):

* Sample variance of (${s_i}$):

  * mean (=0.45)
  * deviations: (0.25, -0.20, -0.05)
  * squared: (0.0625, 0.0400, 0.0025), sum (0.105)
  * sample var (=0.105/(3-1)=0.0525)
* Standard error:
  $$
  \widehat{\mathrm{SE}}(\hat\mu) = \sqrt{0.0525/3} \approx \sqrt{0.0175} \approx 0.132.
  $$
Compare to sampling-based 0/1 scoring: you’d have additional within-question noise, especially on Q2/Q3. Here you’ve “collapsed” that noise into deterministic probabilities.

### F. Common failure modes (and how to guardrail them)

* **Multi-token options** (e.g., “(A)”, “A.”, “True”, “False”).
  If an option is more than one token, you need the probability of a *string*, not a single token. Practically you compute:
  $$
  P(\text{string}) = \prod_{t=1}^m P(\text{token}_t \mid \text{prompt + previous tokens}),
  $$
  summing over tokenizations if necessary (usually you pick the tokenizer’s canonical tokenization and live with it). This is implementable but more fiddly.

* **Whitespace/prefix token trap.**
  If the model tends to output `" A"` (space + letter) and you query the token `"A"` (no space), you’ll read the wrong probability. The fix is to standardize the prompt suffix and test what token actually appears in real completions.

* **Distributional mismatch vs “real” generation.**
  Probability scoring measures behavior under a constrained contract (“output exactly one letter right now”). If your real deployment involves free-form reasoning and then a final answer, this metric is more like a probe of *final-choice preference*, not full behavior. That’s not wrong, but it’s a different estimand. The nice thing is: you can make the estimand explicit.

* **Format noncompliance as a signal.**
  If the model often puts probability mass on non-A/B/C/D tokens, that’s meaningful. Decide whether to treat that as failure (don’t renormalize) or to study it separately (track the mass on invalid tokens as a “format adherence” metric).

### G. How to describe this crisply in the paper

If you want one clean sentence that is both mathematically honest and implementation-realistic:

> For evaluations where the response can be expressed as a discrete choice at a fixed output position (e.g., multiple choice), we replace sampled 0/1 scores with the model’s probability assigned to the correct option (from next-token logits). This makes the per-question score deterministic given the prompt and model, eliminating within-question decoding variance; uncertainty then comes only from the finite sample of questions (and any clustering structure).

