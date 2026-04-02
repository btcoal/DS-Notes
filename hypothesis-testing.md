# Hypothesis Testing

## The Decision Rule
### Expressing Decison Rules in Terms of $Z$ Ratios
* test-statistic
* level of significance
### One-Sided Versus Two-Sided Alternatives
### Testing $H_0: \mu = \mu_0$ ($\sigma$ known)
### The $p$-Value

## Testing Binomial Data - $H_0: p = p_0$
### A Large-Sample Test for the Binomial Parameter $p$
### A Small-Sample Test for the Binomial Parameter $p$

## Type I and Type II Errors
### Computing the Probability of a Type I Error

$$Pr(\text{Type I Error}) = P(\text{Reject } H_0 \mid H_0 \text{ true}) = \alpha$$

### Computing the Probability of a Type II Error

$$Pr(\text{Type II Error}) = P(\text{Fail to Reject } H_0 \mid H_0 \text{ false}) = \beta$$

### Power Curves

$$Power = P(\text{Reject } H_0 \mid H_0 \text{ false}) = 1 - \beta$$

### Factors That Influence the Power of a Test
* The effect of $\alpha$ on $1-\beta$
* The Effect of $\sigma$ on $1-\beta$
* Decision Rules for Non-Normal Data

<!-- ## A Notion of Optimality: The Generalized Likelihood Ratio
* The Generalized Likelihood Ratio
* The Generalized Likelihood Ratio Test -->

## Testing $H_0: \sigma^2 = \sigma^2_0$

## Constructing a Confidence Interval for $\mu$ 

## Testing $H_0: \mu = \mu_0$ (the one-sample $t$-test)

## Testing $H_0: \mu = \mu_0$ when the Normality Assumption is not met

## Testing $H_0: \mu_X = \mu_Y$ (the two-sample $t$-test)

### When $\sigma$ is known

$$z = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}}$$
where
* $\bar{X}_1$ and $\bar{X}_2$ are the sample means of the two groups
* $\sigma_1^2$ and $\sigma_2^2$ are the population variances of the two groups
* $n_1$ and $n_2$ are the sample sizes of the two groups

### When $\sigma$ is unknown
$$
t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
$$
where
* $\bar{X}_1$ and $\bar{X}_2$ are the sample means of the two groups
* $s_1^2$ and $s_2^2$ are the sample variances of the two groups:
  $$ s^2 = \frac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X})^2 $$
* $n_1$ and $n_2$ are the sample sizes of the two groups

## Testing $H_0: \sigma^2_X = \sigma^2_Y$ (the $F$-test)

The F-test is used for comparing variances, and jointly comparing means across multiple groups (ANOVA)

When comparing variances the test statistic $F$ is given by:
$$ F = \frac{s_1^2}{s_2^2} $$
where $s_1^2$ and $s_2^2$ are the sample variances of the two groups. The degrees of freedom for the numerator and denominator are $n_1 - 1$ and $n_2 - 1$, respectively.

When comparing means across multiple groups (ANOVA), the test statistic $F$ is given by:
$$ F = \frac{MS_{between}}{MS_{within}} $$
where
* $MS_{between}$ is the mean square between groups, calculated as:
$$ MS_{between} = \frac{SS_{between}}{df_{between}} $$
where $SS_{between}$ is the sum of squares between groups and $df_{between}$ is the degrees of freedom between groups (number of groups - 1).
* $MS_{within}$ is the mean square within groups, calculated as:
$$ MS_{within} = \frac{SS_{within}}{df_{within}} $$
where $SS_{within}$ is the sum of squares within groups and $df_{within}$ is the degrees of freedom within groups (total sample size - number of groups).

## Binomial Data: Testing $H_0: p_X = p_Y$

Let $X$ and $Y$ be the number of successes in two independent binomial samples with parameters $(n_1, p_X)$ and $(n_2, p_Y)$, respectively. We want to test the null hypothesis $H_0: p_X = p_Y$ against the alternative hypothesis $H_1: p_X \neq p_Y$.
The test statistic for comparing two proportions is given by:
$$ z = \frac{\hat{p}_X - \hat{p}_Y}{\sqrt{\hat{p}(1 - \hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}} $$
where
* $\hat{p}_X = \frac{X}{n_1}$ is the sample proportion of successes in the first group
* $\hat{p}_Y = \frac{Y}{n_2}$ is the sample proportion of successes in the second group
* $\hat{p} = \frac{X + Y}{n_1 + n_2}$ is the pooled sample proportion under the null hypothesis
* $n_1$ and $n_2$ are the sample sizes of the two groups
* $z$ follows a standard normal distribution under the null hypothesis, so we can calculate the p-value based on the observed value of $z$.
* The null hypothesis is that the two proportions are equal, and the alternative hypothesis is that they are not equal:
$$ H_0: p_X = p_Y $$
$$ H_1: p_X \neq p_Y $$


## Confidence Intervals for the Two-Sample Problem

## Power Calculations for the Two-Sample $t$-Test



* $\chi^2$ test used for categorical data goodness-of-fit
* Mann-Whitney $U$ test for comparing distributions
* Wilcoxon signed-rank test for paired samples

## Testing Independence

### Pearson's chi-squared test for independence in contingency tables
For two categorical variables $A$ and $B$ with $r$ and $c$ categories respectively, we can create a contingency table of observed frequencies $O_{ij}$ for each combination of categories.

The test statistic is given by:

$$ \hat{\chi}^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}} $$

where
* $O_{ij}$ is the observed frequency in the cell corresponding to row $i$ and column $j$ of the contingency table
* $E_{ij}$ is the expected frequency under the null hypothesis of independence, calculated as:

$$ E_{ij} = \frac{(\text{row total for row } i) \times (\text{column total for column }j)}{\text{grand total}} $$

The degrees of freedom for the test are given by:

$$ df = (r - 1)(c - 1) $$

where $r$ is the number of rows and $c$ is the number of columns in the contingency table.

So under the null hypothesis of independence

$$\hat{\chi}^2 \sim \chi^2_{(r - 1)(c - 1)}$$

The null hypothesis is that the two categorical variables are independent, and the alternative hypothesis is that they are not independent:
$$ H_0: \text{The two categorical variables are independent} $$
$$ H_1: \text{The two categorical variables are not independent} $$

Caveats:
* The chi-squared test assumes that the expected frequencies in each cell are sufficiently large (usually at least 5) for the approximation to the chi-squared distribution to be valid. If this assumption is violated, the test may not be reliable, and alternative methods such as Fisher's exact test may be more appropriate.
* The chi-squared test does not provide information about the strength or direction of the association between the variables, only whether an association exists. To assess the strength of the association, measures such as Cramér's V or the contingency coefficient can be used.
* The chi-squared test is sensitive to sample size. With a large enough sample size, even small differences between observed and expected frequencies can lead to a statistically significant result, which may not be practically significant. 
* Conversely, with a small sample size, the test may fail to detect a true association (Type II error). Therefore, it is important to consider both statistical significance and practical significance when interpreting the results of a chi-squared test for independence.

## Testing Distributional Fit
### Kolmogorov-Smirnov test for distributional equality

The KS test compares the CDFs of two samples (or a sample and a reference distribution) and calculates the maximum distance between them. We seek to identify the value of $x$ that maximizes the absolute difference between the two CDFs.

The test statistic is given by:
$$ D = \sup_x |F_1(x) - F_2(x)| $$

where $F_1(x)$ and $F_2(x)$ are the empirical CDFs of the two samples (or the sample and the reference distribution). 

$$D \sim \sqrt{\frac{n_1 n_2}{n_1 + n_2}}$$

for two-sample KS test, and 

$$D \sim \sqrt{n}$$

for one-sample KS test.

The p-value is calculated based on the distribution of the test statistic under the null hypothesis that the two samples come from the same distribution (or that the sample comes from the reference distribution). 





## Testing Quantities Other Than Means
* Levene's test for equal variances
* Jarque-Bera test for normality
* Testing quantiles (e.g. medians) via bootstrap or permutation tests

## Non-parametric Tests
* Mann-Whitney $U$ test for comparing distributions
* Wilcoxon signed-rank test for paired samples
* Kruskal-Wallis test for comparing multiple groups
* Friedman test for comparing multiple paired groups
* Permutation tests for testing any statistic of interest without distributional assumptions
* Bootstrap tests for testing any statistic of interest by resampling with replacement
* Randomization tests for testing any statistic of interest by resampling without replacement
* Sign test for testing medians in paired samples
* Runs test for testing randomness in a sequence
* Spearman's rank correlation test for testing monotonic relationships between variables
* Kendall's tau test for testing ordinal associations between variables
* Cochran's Q test for testing differences in proportions across multiple related groups
* McNemar's test for testing differences in proportions in paired nominal data
* Jonckheere-Terpstra test for testing ordered alternatives across multiple groups
* Mood's median test for testing differences in medians across groups

![image](./kdnuggets-hypothesis-testing.jpg)

## Multiple Testing Corrections
1. **Bonferroni Correction**: Adjust the significance level by dividing it by the number of tests being performed. This controls the family-wise error rate but can be conservative.
2. **False Discovery Rate (FDR)**: Control the expected proportion of false positives among the rejected hypotheses, which can be more powerful than Bonferroni correction. The Benjamini-Hochberg procedure is a common method for controlling FDR.
3. **Pre-registration**: Pre-registering hypotheses and analysis plans can help reduce the risk of data dredging and p-hacking.
4. **Effect Size**: Consider reporting effect sizes along with p-values to provide more context about the practical significance of the findings.
5. **Replication**: Consider replicating significant findings in independent datasets to confirm their validity.


### Bonferroni Correction

Scale the p-value by the number of *independent* tests, $m$.

$$
\alpha_{bonferroni} = \frac{\alpha}{m}
$$

Controls the **Family-Wise Error Rate (FWER)**, the probability of making one or more false discoveries.

Simple, but too conservative (i.e. low power).

Betterr for a small number of tests: $lim_{m \to \infty} \frac{Bonferroni\;Threshold}{\alpha} = 0$ where m is the number of tests.

### Benjamini-Hochberg Procedure

Controls the **False Discovery Rate (FDR)**, the expected proportion of false discoveries among the rejected hypotheses.

FDR methods control the expected proportion of false positives among the rejected hypotheses, which *can be more powerful than Bonferroni correction*. The **Benjamini-Hochberg procedure** adjusts the p-value threshold based on the rank of each p-value: 
1. Rank the p-values from smallest to largest: $p_{(1)} \leq p_{(2)} \leq ... \leq p_{(m)}$.
2. Find the largest $k$ such that $p_{(k)} \leq \frac{k}{m} \cdot Q$, where $Q$ is the desired FDR level (e.g., 0.05).
3. Reject all hypotheses with p-values less than or equal to $p_{(k)}$.

Too complex in practice.

Better for a large number of tests $lim_{m \to \infty} \frac{BH\;Threshold}{\alpha} = 1$ where m is the number of tests.

### Dunnet's Test
Dunnett's test is a multiple comparison procedure[1] developed by Canadian statistician Charles Dunnett[2] to compare each of a number of treatments with a single control.[3][4] Multiple comparisons to a control are also referred to as many-to-one comparisons.

Unlike the Bonferroni correction, Dunnett’s test is more powerful because it does not treat every pairwise comparison equally, only comparisons to the control group are tested. By accounting for the dependency between hypotheses (since all groups are compared to the control), Dunnett’s correction offers a less conservative approach to controlling the FWER.

Dunnett's method is more complex than the Bonferroni correction, and its core innovation lies in comparing the test statistic to a more stringent critical value than the standard t-distribution. To achieve this, Dunnett developed an adjusted form of the t-distribution, which results in a stricter threshold than the regular distribution, but a more relaxed one compared to the Bonferroni correction.