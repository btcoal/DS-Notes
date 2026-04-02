# Hypothesis Testing

## The Decision Rule

A hypothesis test is a formal procedure for deciding between two competing claims about a population parameter. The **null hypothesis** ($H_0$) represents the status quo or no-effect claim, while the **alternative hypothesis** ($H_1$ or $H_a$) represents what we're trying to find evidence for.

### Expressing Decision Rules in Terms of $Z$ Ratios

A **test statistic** is a function of the sample data that measures how far the observed data deviates from what we would expect under $H_0$. For testing a population mean with known variance, the test statistic is:

$$Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}}$$

where:
* $\bar{X}$ is the sample mean
* $\mu_0$ is the hypothesized population mean under $H_0$
* $\sigma$ is the known population standard deviation
* $n$ is the sample size

The **level of significance** ($\alpha$) is the probability of rejecting $H_0$ when it is actually true. Common choices are $\alpha = 0.05$, $0.01$, or $0.10$. The **critical value** is the threshold that determines the rejection region, chosen so that the probability of the test statistic falling in the rejection region equals $\alpha$ under $H_0$.

### One-Sided Versus Two-Sided Alternatives

**Two-sided (two-tailed) test:**
$$H_0: \mu = \mu_0 \quad \text{vs} \quad H_1: \mu \neq \mu_0$$

Reject $H_0$ if $|Z| > z_{\alpha/2}$, where $z_{\alpha/2}$ is the $(1 - \alpha/2)$ quantile of the standard normal.

**One-sided (one-tailed) tests:**

Right-tailed: $H_0: \mu \leq \mu_0$ vs $H_1: \mu > \mu_0$. Reject if $Z > z_\alpha$.

Left-tailed: $H_0: \mu \geq \mu_0$ vs $H_1: \mu < \mu_0$. Reject if $Z < -z_\alpha$.

### Testing $H_0: \mu = \mu_0$ ($\sigma$ known)

When the population standard deviation $\sigma$ is known, the test statistic under $H_0$ follows a standard normal distribution:

$$Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}} \sim N(0, 1)$$

**Procedure:**
1. State $H_0$ and $H_1$
2. Choose significance level $\alpha$
3. Compute the test statistic $Z$
4. Determine the critical value(s) or compute the p-value
5. Make a decision: reject $H_0$ if $Z$ falls in the rejection region (or if p-value $< \alpha$)

### Constructing a Confidence Interval for $\mu$

**When $\sigma$ is known:**
$$\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

**When $\sigma$ is unknown:**
$$\bar{X} \pm t_{\alpha/2, n-1} \cdot \frac{S}{\sqrt{n}}$$

**Duality between confidence intervals and hypothesis tests:**

A $(1-\alpha)$ confidence interval contains all values $\mu_0$ for which we would fail to reject $H_0: \mu = \mu_0$ at level $\alpha$.

Equivalently: Reject $H_0: \mu = \mu_0$ if and only if $\mu_0$ is not in the confidence interval.

## Testing $H_0: \mu = \mu_0$ (the one-sample $t$-test)

When $\sigma$ is unknown, we replace it with the sample standard deviation $S$:

**Test statistic:**
$$t = \frac{\bar{X} - \mu_0}{S / \sqrt{n}}$$

Under $H_0$ (assuming normality), $t \sim t_{n-1}$.

**Decision rules (at level $\alpha$):**
* $H_1: \mu \neq \mu_0$: Reject if $|t| > t_{\alpha/2, n-1}$
* $H_1: \mu > \mu_0$: Reject if $t > t_{\alpha, n-1}$
* $H_1: \mu < \mu_0$: Reject if $t < -t_{\alpha, n-1}$

**Assumptions:**
* Random sample
* Independence
* Normality (or large sample size for CLT)

### The $p$-Value

The **p-value** is the probability, assuming $H_0$ is true, of observing a test statistic at least as extreme as the one calculated from the sample data.

For a two-sided test: $p = 2 \cdot P(Z > |z_{obs}|)$

For a right-tailed test: $p = P(Z > z_{obs})$

For a left-tailed test: $p = P(Z < z_{obs})$

**Interpretation:** A small p-value (typically $\leq \alpha$) provides evidence against $H_0$. The p-value is NOT the probability that $H_0$ is true.

## Testing Binomial Data - $H_0: p = p_0$

### A Large-Sample Test for the Binomial Parameter $p$

When $n$ is large (rule of thumb: $np_0 \geq 10$ and $n(1-p_0) \geq 10$), the sampling distribution of $\hat{p}$ is approximately normal by the Central Limit Theorem.

**Test statistic:**
$$Z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1 - p_0)}{n}}}$$

where $\hat{p} = X/n$ is the sample proportion and $X$ is the number of successes.

Under $H_0$, $Z \sim N(0, 1)$ approximately.

### A Small-Sample Test for the Binomial Parameter $p$

When $n$ is small, the normal approximation is inadequate. We use the exact binomial distribution.

For testing $H_0: p = p_0$ vs $H_1: p \neq p_0$:

The p-value is computed directly from the binomial distribution:
$$p\text{-value} = 2 \cdot \min\left(P(X \leq x_{obs}), P(X \geq x_{obs})\right)$$

where $X \sim \text{Binomial}(n, p_0)$ under $H_0$.

Alternatively, for one-sided tests:
* $H_1: p > p_0$: p-value $= P(X \geq x_{obs})$
* $H_1: p < p_0$: p-value $= P(X \leq x_{obs})$

## Type I and Type II Errors

| | $H_0$ True | $H_0$ False |
|---|---|---|
| Reject $H_0$ | Type I Error ($\alpha$) | Correct Decision (Power = $1-\beta$) |
| Fail to Reject $H_0$ | Correct Decision | Type II Error ($\beta$) |

* **Type I Error:** Rejecting $H_0$ when it is true (false positive)
* **Type II Error:** Failing to reject $H_0$ when it is false (false negative)
* **Power:** The probability of correctly rejecting $H_0$ when it is false

### Computing the Probability of a Type I Error

The probability of a Type I error equals the significance level:
$$\alpha = P(\text{Reject } H_0 \mid H_0 \text{ is true})$$

For a two-sided Z-test at level $\alpha$:
$$\alpha = P(|Z| > z_{\alpha/2} \mid \mu = \mu_0) = \alpha$$

This is by construction—we choose the critical value to make this probability equal to $\alpha$.

### Computing the Probability of a Type II Error

The probability of a Type II error depends on the true parameter value $\mu_1 \neq \mu_0$:

$$\beta = P(\text{Fail to reject } H_0 \mid H_1 \text{ is true})$$

For a two-sided test of $H_0: \mu = \mu_0$ vs $H_1: \mu \neq \mu_0$:

$$\beta(\mu_1) = P\left(-z_{\alpha/2} < \frac{\bar{X} - \mu_0}{\sigma/\sqrt{n}} < z_{\alpha/2} \,\Big|\, \mu = \mu_1\right)$$

$$= \Phi\left(z_{\alpha/2} - \frac{\mu_1 - \mu_0}{\sigma/\sqrt{n}}\right) - \Phi\left(-z_{\alpha/2} - \frac{\mu_1 - \mu_0}{\sigma/\sqrt{n}}\right)$$

where $\Phi$ is the standard normal CDF.

### Power Curves

A **power curve** (or power function) plots the probability of rejecting $H_0$ as a function of the true parameter value.

$$\text{Power}(\mu) = 1 - \beta(\mu) = P(\text{Reject } H_0 \mid \mu)$$

Properties:
* At $\mu = \mu_0$: Power $= \alpha$ (by definition)
* As $|\mu - \mu_0|$ increases: Power approaches 1
* The power curve is symmetric around $\mu_0$ for two-sided tests

### Factors That Influence the Power of a Test

**The effect of $\alpha$ on $1-\beta$:**
* Increasing $\alpha$ increases power (wider rejection region)
* Trade-off: higher power comes at the cost of more Type I errors
* This is the fundamental tension in hypothesis testing

**The Effect of $\sigma$ on $1-\beta$:**
* Decreasing $\sigma$ increases power
* Smaller variance means the sampling distribution is more concentrated
* Easier to detect departures from $H_0$

**The Effect of $n$ on $1-\beta$:**
* Increasing sample size $n$ increases power
* Standard error $\sigma/\sqrt{n}$ decreases with $n$
* This is often the most practical way to increase power

**The Effect of Effect Size on $1-\beta$:**
* Larger effect size $|\mu_1 - \mu_0|$ increases power
* Easier to detect large departures from $H_0$

**Decision Rules for Non-Normal Data:**
* For large samples, the CLT provides approximate normality
* For small samples with non-normal data, consider:
  * Nonparametric tests (e.g., Wilcoxon, Mann-Whitney)
  * Permutation/randomization tests
  * Bootstrap methods
  * Transformations to achieve normality


## Testing $H_0: \sigma^2 = \sigma^2_0$

For a random sample $X_1, \ldots, X_n$ from $N(\mu, \sigma^2)$, to test $H_0: \sigma^2 = \sigma^2_0$:

**Test statistic:**
$$\chi^2 = \frac{(n-1)S^2}{\sigma^2_0}$$

where $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$ is the sample variance.

Under $H_0$, $\chi^2 \sim \chi^2_{n-1}$.

**Decision rules:**
* $H_1: \sigma^2 \neq \sigma^2_0$: Reject if $\chi^2 < \chi^2_{1-\alpha/2, n-1}$ or $\chi^2 > \chi^2_{\alpha/2, n-1}$
* $H_1: \sigma^2 > \sigma^2_0$: Reject if $\chi^2 > \chi^2_{\alpha, n-1}$
* $H_1: \sigma^2 < \sigma^2_0$: Reject if $\chi^2 < \chi^2_{1-\alpha, n-1}$

**Caveat:** This test is highly sensitive to departures from normality.



## Testing $H_0: \mu = \mu_0$ when the Normality Assumption is not met

When the normality assumption is violated:

**1. Large Sample Approach (CLT):**
For $n \geq 30$ (rule of thumb), the sampling distribution of $\bar{X}$ is approximately normal regardless of the population distribution. The t-test is robust to moderate departures from normality.

**2. Wilcoxon Signed-Rank Test:**
A nonparametric alternative that tests whether the median equals $\mu_0$:
* Compute $D_i = X_i - \mu_0$
* Rank the $|D_i|$ values
* Test statistic: $W^+ = \sum \text{rank}(|D_i|) \cdot \mathbf{1}(D_i > 0)$

**3. Sign Test:**
The simplest nonparametric alternative:
* Count $B = $ number of observations $> \mu_0$
* Under $H_0$, $B \sim \text{Binomial}(n, 0.5)$

**4. Bootstrap Test:**
* Resample with replacement from centered data
* Compute test statistic for each resample
* Estimate p-value from empirical distribution

**5. Permutation Test:**
* Under $H_0$, signs of $(X_i - \mu_0)$ are exchangeable
* Enumerate or sample all possible sign assignments
* Compare observed statistic to permutation distribution

## Testing $H_0: \mu_X = \mu_Y$ (the two-sample $t$-test)

### When $\sigma$ is known

$$z = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}}$$

where:
* $\bar{X}_1$ and $\bar{X}_2$ are the sample means of the two groups
* $\sigma_1^2$ and $\sigma_2^2$ are the population variances of the two groups
* $n_1$ and $n_2$ are the sample sizes of the two groups

Under $H_0$, $z \sim N(0, 1)$.

### When $\sigma$ is unknown

**Welch's t-test (unequal variances):**
$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

where:
* $\bar{X}_1$ and $\bar{X}_2$ are the sample means of the two groups
* $s_1^2$ and $s_2^2$ are the sample variances of the two groups:
  $$ s^2 = \frac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X})^2 $$
* $n_1$ and $n_2$ are the sample sizes of the two groups

The degrees of freedom are approximated by the Welch-Satterthwaite equation:
$$df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}$$

**Pooled t-test (equal variances assumed):**

If $\sigma_1^2 = \sigma_2^2 = \sigma^2$, use the pooled variance estimate:
$$s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}$$

$$t = \frac{\bar{X}_1 - \bar{X}_2}{s_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}$$

Under $H_0$, $t \sim t_{n_1 + n_2 - 2}$.

## Testing $H_0: \sigma^2_X = \sigma^2_Y$ (the $F$-test)

The F-test is used for comparing variances, and jointly comparing means across multiple groups (ANOVA).

### Comparing Two Variances

When comparing variances, the test statistic $F$ is given by:
$$ F = \frac{s_1^2}{s_2^2} $$

where $s_1^2$ and $s_2^2$ are the sample variances of the two groups. The degrees of freedom for the numerator and denominator are $n_1 - 1$ and $n_2 - 1$, respectively.

Under $H_0: \sigma_1^2 = \sigma_2^2$, $F \sim F_{n_1-1, n_2-1}$.

**Decision rules:**
* $H_1: \sigma_1^2 \neq \sigma_2^2$: Reject if $F < F_{1-\alpha/2}$ or $F > F_{\alpha/2}$
* Convention: Place the larger variance in the numerator to get $F > 1$

**Caveat:** The F-test for variances is highly sensitive to non-normality.

### One-Way ANOVA

When comparing means across $k$ groups (ANOVA), the test statistic $F$ is given by:
$$ F = \frac{MS_{between}}{MS_{within}} $$

where:
* $MS_{between}$ is the mean square between groups, calculated as:
$$ MS_{between} = \frac{SS_{between}}{df_{between}} $$
where $SS_{between} = \sum_{j=1}^k n_j(\bar{X}_j - \bar{X})^2$ is the sum of squares between groups and $df_{between} = k - 1$ is the degrees of freedom between groups.

* $MS_{within}$ is the mean square within groups, calculated as:
$$ MS_{within} = \frac{SS_{within}}{df_{within}} $$
where $SS_{within} = \sum_{j=1}^k \sum_{i=1}^{n_j} (X_{ij} - \bar{X}_j)^2$ is the sum of squares within groups and $df_{within} = N - k$ is the degrees of freedom within groups (total sample size minus number of groups).

Under $H_0: \mu_1 = \mu_2 = \cdots = \mu_k$, $F \sim F_{k-1, N-k}$.

## Binomial Data: Testing $H_0: p_X = p_Y$

Let $X$ and $Y$ be the number of successes in two independent binomial samples with parameters $(n_1, p_X)$ and $(n_2, p_Y)$, respectively. We want to test the null hypothesis $H_0: p_X = p_Y$ against the alternative hypothesis $H_1: p_X \neq p_Y$.

The test statistic for comparing two proportions is given by:
$$ z = \frac{\hat{p}_X - \hat{p}_Y}{\sqrt{\hat{p}(1 - \hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}} $$

where:
* $\hat{p}_X = \frac{X}{n_1}$ is the sample proportion of successes in the first group
* $\hat{p}_Y = \frac{Y}{n_2}$ is the sample proportion of successes in the second group
* $\hat{p} = \frac{X + Y}{n_1 + n_2}$ is the pooled sample proportion under the null hypothesis
* $n_1$ and $n_2$ are the sample sizes of the two groups
* $z$ follows a standard normal distribution under the null hypothesis, so we can calculate the p-value based on the observed value of $z$.
* The null hypothesis is that the two proportions are equal, and the alternative hypothesis is that they are not equal:
$$ H_0: p_X = p_Y $$
$$ H_1: p_X \neq p_Y $$

**Small sample alternative:** Fisher's exact test computes the exact p-value using the hypergeometric distribution.

## Confidence Intervals for the Two-Sample Problem

**For difference in means ($\sigma$ known):**
$$(\bar{X}_1 - \bar{X}_2) \pm z_{\alpha/2} \sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}$$

**For difference in means ($\sigma$ unknown, Welch):**
$$(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2, df} \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}$$

where $df$ is given by the Welch-Satterthwaite approximation.

**For difference in proportions (large sample):**
$$(\hat{p}_1 - \hat{p}_2) \pm z_{\alpha/2} \sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}$$

**For ratio of variances:**
$$\left(\frac{s_1^2}{s_2^2} \cdot \frac{1}{F_{\alpha/2, n_1-1, n_2-1}}, \frac{s_1^2}{s_2^2} \cdot \frac{1}{F_{1-\alpha/2, n_1-1, n_2-1}}\right)$$

## Power Calculations for the Two-Sample $t$-Test

To calculate power for detecting a difference $\delta = \mu_1 - \mu_2$:

**For known variance:**
$$\text{Power} = \Phi\left(\frac{|\delta|}{\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}} - z_{\alpha/2}\right) + \Phi\left(-\frac{|\delta|}{\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}} - z_{\alpha/2}\right)$$

The second term is usually negligible for reasonable power.

**For unknown variance (using effect size):**

Cohen's $d$ effect size: $d = \frac{\delta}{\sigma}$

Approximate power using noncentral t-distribution with noncentrality parameter:
$$\lambda = d \sqrt{\frac{n_1 n_2}{n_1 + n_2}}$$

**Sample size determination:**

For equal sample sizes $n_1 = n_2 = n$ and equal variances:
$$n \approx \frac{2(z_{\alpha/2} + z_\beta)^2 \sigma^2}{\delta^2}$$

where $1 - \beta$ is the desired power.

**Effect size conventions (Cohen):**
* Small: $d = 0.2$
* Medium: $d = 0.5$
* Large: $d = 0.8$


## Testing Independence

### Pearson's chi-squared test for independence in contingency tables

For two categorical variables $A$ and $B$ with $r$ and $c$ categories respectively, we can create a contingency table of observed frequencies $O_{ij}$ for each combination of categories.

The test statistic is given by:

$$ \chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}} $$

where:
* $O_{ij}$ is the observed frequency in the cell corresponding to row $i$ and column $j$ of the contingency table
* $E_{ij}$ is the expected frequency under the null hypothesis of independence, calculated as:

$$ E_{ij} = \frac{(\text{row total for row } i) \times (\text{column total for column } j)}{\text{grand total}} = \frac{n_{i\cdot} \cdot n_{\cdot j}}{n} $$

The degrees of freedom for the test are given by:

$$ df = (r - 1)(c - 1) $$

where $r$ is the number of rows and $c$ is the number of columns in the contingency table.

So under the null hypothesis of independence:

$$\chi^2 \sim \chi^2_{(r - 1)(c - 1)}$$

The null hypothesis is that the two categorical variables are independent, and the alternative hypothesis is that they are not independent:
$$ H_0: \text{The two categorical variables are independent} $$
$$ H_1: \text{The two categorical variables are not independent} $$

**Caveats:**
* The chi-squared test assumes that the expected frequencies in each cell are sufficiently large (usually at least 5) for the approximation to the chi-squared distribution to be valid. If this assumption is violated, the test may not be reliable, and alternative methods such as Fisher's exact test may be more appropriate.
* The chi-squared test does not provide information about the strength or direction of the association between the variables, only whether an association exists. To assess the strength of the association, measures such as Cramér's V or the contingency coefficient can be used.
* The chi-squared test is sensitive to sample size. With a large enough sample size, even small differences between observed and expected frequencies can lead to a statistically significant result, which may not be practically significant.
* Conversely, with a small sample size, the test may fail to detect a true association (Type II error). Therefore, it is important to consider both statistical significance and practical significance when interpreting the results of a chi-squared test for independence.

### Fisher's Exact Test

For $2 \times 2$ contingency tables when expected cell counts are small, Fisher's exact test provides an exact p-value using the hypergeometric distribution:

$$P = \frac{\binom{a+b}{a}\binom{c+d}{c}}{\binom{n}{a+c}}$$

where the table is:
|  | Col 1 | Col 2 |
|--|-------|-------|
| Row 1 | a | b |
| Row 2 | c | d |

The p-value is computed by summing probabilities of all tables at least as extreme as the observed table.

## Testing Distributional Fit

### Kolmogorov-Smirnov test for distributional equality

The KS test compares the CDFs of two samples (or a sample and a reference distribution) and calculates the maximum distance between them. We seek to identify the value of $x$ that maximizes the absolute difference between the two CDFs.

The test statistic is given by:
$$ D = \sup_x |F_1(x) - F_2(x)| $$

where $F_1(x)$ and $F_2(x)$ are the empirical CDFs of the two samples (or the sample and the reference distribution).

**Two-sample KS test:**
Under $H_0$ that both samples come from the same distribution:
$$\sqrt{\frac{n_1 n_2}{n_1 + n_2}} D \xrightarrow{d} K$$

where $K$ is the Kolmogorov distribution.

**One-sample KS test:**
Under $H_0$ that the sample comes from the reference distribution $F_0$:
$$\sqrt{n} D \xrightarrow{d} K$$

The p-value is calculated based on the distribution of the test statistic under the null hypothesis that the two samples come from the same distribution (or that the sample comes from the reference distribution).

**Properties:**
* Distribution-free (nonparametric)
* Most sensitive to differences near the center of the distribution
* Less powerful than Anderson-Darling for tail differences

### Anderson-Darling Test

A weighted version of the KS test that gives more weight to the tails:

$$A^2 = n \int_{-\infty}^{\infty} \frac{[F_n(x) - F_0(x)]^2}{F_0(x)(1-F_0(x))} dF_0(x)$$

More powerful than KS for detecting departures in the tails.

### Shapiro-Wilk Test for Normality

Tests whether a sample comes from a normal distribution:

$$W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}$$

where $x_{(i)}$ are the ordered sample values and $a_i$ are tabulated constants.

Generally considered the most powerful test for normality.

## Testing Quantities Other Than Means

**Levene's test for equal variances:**
More robust to non-normality than the F-test. Based on analyzing absolute deviations from group medians (or means).

$$W = \frac{(N-k)}{(k-1)} \cdot \frac{\sum_{j=1}^k n_j(\bar{Z}_j - \bar{Z})^2}{\sum_{j=1}^k \sum_{i=1}^{n_j}(Z_{ij} - \bar{Z}_j)^2}$$

where $Z_{ij} = |X_{ij} - \tilde{X}_j|$ and $\tilde{X}_j$ is the median of group $j$.

**Jarque-Bera test for normality:**
Based on sample skewness and kurtosis:

$$JB = \frac{n}{6}\left(S^2 + \frac{(K-3)^2}{4}\right)$$

where $S$ is sample skewness and $K$ is sample kurtosis. Under normality, $JB \sim \chi^2_2$ asymptotically.

**Testing quantiles (e.g., medians) via bootstrap or permutation tests:**
* Compute the test statistic on observed data
* Generate bootstrap/permutation distribution
* Calculate p-value as the proportion of resampled statistics more extreme than observed

## Non-parametric Tests

**Mann-Whitney $U$ test (Wilcoxon rank-sum test):**
Tests whether one distribution is stochastically greater than another. Compares ranks of observations from two independent samples:
$$U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1$$
where $R_1$ is the sum of ranks in sample 1.

**Wilcoxon signed-rank test:**
For paired samples, tests whether the distribution of differences is symmetric around zero:
$$W = \sum_{i: D_i > 0} \text{rank}(|D_i|)$$

**Kruskal-Wallis test:**
Nonparametric alternative to one-way ANOVA for comparing $k$ independent groups:
$$H = \frac{12}{N(N+1)} \sum_{j=1}^k \frac{R_j^2}{n_j} - 3(N+1)$$

**Friedman test:**
Nonparametric alternative to repeated-measures ANOVA for $k$ related samples.

**Permutation tests:**
For testing any statistic of interest without distributional assumptions. Under $H_0$, all permutations of the data are equally likely.

**Bootstrap tests:**
For testing any statistic of interest by resampling with replacement. Approximates the sampling distribution empirically.

**Randomization tests:**
For testing any statistic of interest by resampling without replacement.

**Sign test:**
For testing medians in paired samples. Counts the number of positive differences.

**Runs test:**
For testing randomness in a sequence. Counts the number of "runs" (consecutive sequences of the same type).

**Spearman's rank correlation test:**
For testing monotonic relationships between variables:
$$r_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$
where $d_i$ is the difference between ranks.

**Kendall's tau test:**
For testing ordinal associations between variables. Based on concordant and discordant pairs.

**Cochran's Q test:**
For testing differences in proportions across multiple related groups (extension of McNemar's test).

**McNemar's test:**
For testing differences in proportions in paired nominal data (2×2 matched pairs):
$$\chi^2 = \frac{(b-c)^2}{b+c}$$

**Jonckheere-Terpstra test:**
For testing ordered alternatives across multiple groups (when there's a hypothesized ordering).

**Mood's median test:**
For testing differences in medians across groups. Based on counting observations above/below the grand median.

## Multiple Testing Corrections

When performing multiple hypothesis tests, the probability of at least one Type I error increases. Several methods exist to control this:

1. **Bonferroni Correction**: Adjust the significance level by dividing it by the number of tests being performed. This controls the family-wise error rate but can be conservative.
2. **False Discovery Rate (FDR)**: Control the expected proportion of false positives among the rejected hypotheses, which can be more powerful than Bonferroni correction. The Benjamini-Hochberg procedure is a common method for controlling FDR.
3. **Pre-registration**: Pre-registering hypotheses and analysis plans can help reduce the risk of data dredging and p-hacking.
4. **Effect Size**: Consider reporting effect sizes along with p-values to provide more context about the practical significance of the findings.
5. **Replication**: Consider replicating significant findings in independent datasets to confirm their validity.

### Bonferroni Correction

Scale the significance level by the number of tests, $m$:

$$\alpha_{bonferroni} = \frac{\alpha}{m}$$

Equivalently, adjust p-values: $p_{adj} = \min(m \cdot p, 1)$

Controls the **Family-Wise Error Rate (FWER)**, the probability of making one or more false discoveries:
$$FWER = P(\text{at least one Type I error}) \leq \alpha$$

**Properties:**
* Simple and intuitive
* Too conservative (low power), especially for large $m$
* Valid for any dependency structure among tests
* Better for a small number of tests: as $m \to \infty$, the threshold $\alpha/m \to 0$

### Holm-Bonferroni Method (Step-Down)

A less conservative improvement over Bonferroni:

1. Order p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. For $i = 1, 2, \ldots, m$:
   * If $p_{(i)} > \frac{\alpha}{m - i + 1}$, stop and reject all $H_{(1)}, \ldots, H_{(i-1)}$
3. If all tests pass, reject all hypotheses

Still controls FWER but is uniformly more powerful than Bonferroni.

### Benjamini-Hochberg Procedure

Controls the **False Discovery Rate (FDR)**, the expected proportion of false discoveries among rejected hypotheses:
$$FDR = E\left[\frac{V}{R} \mid R > 0\right] \cdot P(R > 0)$$

where $V$ = number of false rejections, $R$ = total rejections.

**Procedure:**
1. Order p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. Find the largest $k$ such that $p_{(k)} \leq \frac{k}{m} \alpha$
3. Reject all $H_{(1)}, \ldots, H_{(k)}$

**Properties:**
* More powerful than FWER-controlling methods
* Valid under independence or positive regression dependency (PRDS)
* Better for a large number of tests
* Interpretation: If you reject 100 hypotheses at FDR = 0.05, expect about 5 to be false discoveries

**Why better for large $m$:** Bonferroni's threshold shrinks to zero as $m$ grows, while BH's threshold stays proportional to rank, maintaining reasonable power.

### Dunnett's Test

Dunnett's test is a multiple comparison procedure developed by Canadian statistician Charles Dunnett to compare each of a number of treatments with a single control. Multiple comparisons to a control are also referred to as many-to-one comparisons.

Unlike the Bonferroni correction, Dunnett's test is more powerful because it does not treat every pairwise comparison equally—only comparisons to the control group are tested. By accounting for the dependency between hypotheses (since all groups are compared to the control), Dunnett's correction offers a less conservative approach to controlling the FWER.

Dunnett's method is more complex than the Bonferroni correction, and its core innovation lies in comparing the test statistic to a more stringent critical value than the standard t-distribution. To achieve this, Dunnett developed an adjusted form of the t-distribution, which results in a stricter threshold than the regular distribution, but a more relaxed one compared to the Bonferroni correction.

**Test statistic:**
$$t_i = \frac{\bar{X}_i - \bar{X}_0}{s_p\sqrt{\frac{1}{n_i} + \frac{1}{n_0}}}$$

where $\bar{X}_0$ is the control group mean and $s_p$ is the pooled standard deviation.

### Tukey's Honest Significant Difference (HSD)

For all pairwise comparisons after ANOVA:

$$q = \frac{\bar{X}_i - \bar{X}_j}{SE}$$

where the studentized range distribution is used for inference.

### Šidák Correction

Slightly less conservative than Bonferroni:
$$\alpha_{Sidak} = 1 - (1 - \alpha)^{1/m}$$

Valid when tests are independent.

## Summary: Choosing the Right Test

| Scenario | Parametric Test | Nonparametric Alternative |
|----------|-----------------|---------------------------|
| One-sample mean | One-sample t-test | Wilcoxon signed-rank, Sign test |
| Two-sample means (independent) | Two-sample t-test | Mann-Whitney U |
| Two-sample means (paired) | Paired t-test | Wilcoxon signed-rank |
| Multiple group means | One-way ANOVA | Kruskal-Wallis |
| Two proportions | z-test for proportions | Fisher's exact |
| Multiple proportions | Chi-squared test | Fisher's exact |
| Two variances | F-test | Levene's test |
| Correlation | Pearson's r | Spearman's rho, Kendall's tau |
| Normality | — | Shapiro-Wilk, KS test |
| Independence (categorical) | Chi-squared | Fisher's exact |

**General guidelines:**
* Use parametric tests when assumptions are met (more powerful)
* Use nonparametric tests when: sample size is small, data is ordinal, distribution is clearly non-normal
* Always check assumptions before choosing a test
* Report effect sizes alongside p-values
* Consider multiple testing corrections when appropriate