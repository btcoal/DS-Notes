# Statistics Questions

## Describe the difference between Type I and Type II errors and the trade-offs involved in minimizing each.

## Testing a Coin for Fairness

***Say you flip a coin 10 times and get 1 head. What would be your null hypothesis and p-value for testing if the coin is fair?***

The null hypothesis:

$H_0: p = 0.5$, against

$H_1: p \neq 0.5$.

The $p\text{-value} = Pr(\text{number of heads} \leq 1 \text{ or } \geq 9 | H_0)$.

If $p=\frac{1}{2}$, then the number of heads $X$ follows a Binomial distribution with parameters $n=10$ and $p=\frac{1}{2}$. 

Thus, 

$p\text{-value} = Pr(X \leq 1) + Pr(X \geq 9) = Pr(X \leq 1) + Pr(X \leq 1) = 2Pr(X \leq 1)$.

Where

$Pr(X \leq 1) = Pr(X=0) + Pr(X=1)$

$= \binom{10}{0}(0.5)^{10} + \binom{10}{1}(0.5)^{10}$

$= \frac{1 + 10}{1024}$

$= \frac{11}{1024}$.

Therefore, the $p\text{-value}$ is $2 \cdot \frac{11}{1024} \approx 0.0215$.

## Multiple Testing

***Say you are testing hundreds of hypotheses, each with a $t$-test. What considerations would you take into account when doing this?***

When testing hundreds of hypotheses using $t$-tests, it is important to consider the issue of multiple comparisons, which can increase the likelihood of Type I errors (false positives). Here are some considerations:
1. **Bonferroni Correction**: This method adjusts the significance level by dividing it by the number of tests being performed. For example, if you are conducting 100 tests and want an overall significance level of 0.05, you would use a significance level of 0.0005 for each individual test. This is a conservative approach that helps control the **family-wise error rate**.

2. **False Discovery Rate (FDR)**: FDR methods control the expected proportion of false positives among the rejected hypotheses, which *can be more powerful than Bonferroni correction*. The **Benjamini-Hochberg procedure** adjusts the p-value threshold based on the rank of each p-value: 
   - Rank the p-values from smallest to largest: $p_{(1)} \leq p_{(2)} \leq ... \leq p_{(m)}$.
   - Find the largest $k$ such that $p_{(k)} \leq \frac{k}{m} \cdot Q$, where $Q$ is the desired FDR level (e.g., 0.05).
   - Reject all hypotheses with p-values less than or equal to $p_{(k)}$.

3. **Pre-registration**: Pre-registering hypotheses and analysis plans can help reduce the risk of data dredging and p-hacking.

4. **Effect Sizes matter**

5. **Replication**: Consider replicating significant findings in independent datasets to confirm their validity.


## How would you derive a confidence interval for the probability of flipping heads from a series of coin tosses?
### Bootstrap Method
In $n$ coin tosses, let $X$ be the number of heads observed. The sample proportion of heads is given by:
$$ \hat{p} = \frac{X}{n} $$

For $B$ bootstrap replications

1. Resample the original data with replacement to create a bootstrap sample of size $n$.
2. Calculate the sample proportion of heads for each bootstrap sample.
3. Use the distribution of the bootstrap sample proportions to estimate the confidence interval (e.g., using the percentile method).

### Use a Normal Approximation
In $n$ coin tosses, let $X$ be the number of heads observed. The sample proportion of heads is given by:

$$ \hat{p} = \frac{X}{n} $$

To derive a confidence interval for the true probability of flipping heads, we can use the normal approximation to the binomial distribution when $n$ is sufficiently large. The standard error of the sample proportion is given by:

$$ SE = \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}} $$

For a confidence level of $1 - \alpha$, the confidence interval can be calculated as:

$$ \hat{p} \pm z_{\alpha/2} \cdot SE $$

Where $z_{\alpha/2}$ is the critical value from the standard normal distribution corresponding to the desired confidence level. For example, for a 95% confidence interval, $z_{0.025} \approx 1.96$.

## What is the expected number of coin flips needed to get two consecutive heads?

Let $E$ be the expected number of flips needed to get two consecutive heads. We can break down the problem into states:
- State 0: No heads observed yet.
- State 1: One head observed (the last flip was a head).
- State 2: Two consecutive heads observed (the desired outcome).
- State 0: If we flip a tail, we stay in state 0. If we flip a head, we move to state 1.
- State 1: If we flip a tail, we go back to state 0. If we flip a head, we move to state 2.
- State 2: We have achieved our goal, so we stop.

We can set up the following equations based on the expected values:

$E_0 = 1 + \frac{1}{2}E_0 + \frac{1}{2}E_1$

$E_1 = 1 + \frac{1}{2}E_0 + \frac{1}{2}E_2$

$E_2 = 0$ (since we have achieved our goal)

Solving these equations gives us:
$E_0 = 6$ and $E_1 = 4$.

Thus, the expected number of flips needed to get two consecutive heads is $E_0 = 6$.

Intuitively, the expected number of flips needed to get the first head is 2 (since it's a geometric distribution with $p=0.5$). After getting the first head, the expected number of flips needed to get the second head is 4 (since we can either get a head and succeed, or get a tail and go back to the beginning). Therefore, the total expected number of flips is $2 + 4 = 6$.

## Bias vs Consistency

***What does it mean for an estimator to be unbiased? What about consistent? Give examples of an unbiased but inconsistent estimator, and a biased but consistent estimator.***

An example of an unbiased but inconsistent estimator is the $\hat{\beta}$ in a linear regression model with omitted variable bias is an unbiased estimator of the true coefficient $\beta$ if the omitted variable is uncorrelated with the included variables, but it is not consistent because it does not converge to the true coefficient as the sample size increases.

A biased estimator example is the sample variance calculated with $n$ in the denominator instead of $n-1$. This estimator is biased because it systematically underestimates the true population variance. However, it is consistent because as the sample size increases, the bias decreases and the estimator converges to the true population variance. Also in a LASSO regression, the coefficient estimates are biased towards zero due to the regularization term, but they can be consistent if the true coefficients are sparse and the regularization parameter is appropriately chosen.

## Assume that $\log(X) \sim N(0,1)$. What is the expectation of $X$?

Since $\log(X) \sim N(0,1)$, we can express $X$ as $X = e^{\log(X)}$. The expectation of $X$ is given by the moment generating function of a normal distribution:
$$
E[X] = E[e^{\log(X)}] = e^{\mu + \frac{\sigma^2}{2}} = e^{0 + \frac{1}{2}} = e^{\frac{1}{2}}.
$$

Where $\mu$ is the mean of the normal distribution (which is 0) and $\sigma^2$ is the variance of the normal distribution (which is 1). Therefore, the expectation of $X$ is $e^{\frac{1}{2}}$.

## Pooled Mean and Standard Deviation

***Say you have two samples for which you know ther means and standard deviations How do you calculate the pooled mean and standard deviation of the total dataset? Can you extend it to $K$ subsets?***

Calculate the sample means $\hat{\mu}_1$ and $\hat{\mu}_2$ for the two subsets.

Calculate each sample variance as 
$$ s_1^2 = \frac{1}{n_1 - 1} \sum_{j=1}^{n_1} (X_{1j} - \hat{\mu}_1)^2 $$
and
$$ s_2^2 = \frac{1}{n_2 - 1} \sum_{j=1}^{n_2} (X_{2j} - \hat{\mu}_2)^2 $$

Then, the pooled mean $\hat{\mu}$ can be calculated as:

$$ \hat{\mu} = \frac{n_1 \hat{\mu}_1 + n_2 \hat{\mu}_2}{n_1 + n_2} $$

The pooled standard deviation $\sigma$ can be calculated using the formula for the pooled variance:

$$ s^2 = \frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 1} $$
Then, the pooled standard deviation is $s = \sqrt{s^2}$.

## Detecting an Unfair Coin I

***A coin was flipped 1,000 times and 550 times it showed heads. Do you think the coin is biased towards heads?***

## Detecting an Unfair Coin II

***Say we have an unfair coin that lands on heads 60% of the time. How many coin flips are needed to detect that the coin is unfair?***

<!-- ## Say you draw $n$ samples from a uniform distribution $X \sim U(a, b)$. What are the MLEs of $a$ and $b$? -->

<!-- ## You are drawing from a normally distributed random varialbe $X \sim N(0, 1)$ once a day. What is the expected number of days until you draw a value greater than 2? -->

<!-- ## Say you are given a random Bernoulli trial generator. How would you generate values from a standard Normal distribution? -->

<!-- ## Say we have $X \sim U(0,1)$ and $Y \sim U(0, 1)$ and they are independent. What is the exptected value of $min(X, Y)$? -->

<!-- ## There are two games involving dice that you can play. In the first game, you roll two dice at once and receive a dollar amount equal to the product of the rolls. In the second game, you roll one die and get the dollar amount equivalent to the square of the roll. Which game has a higher expected payout? -->

<!-- ## Say you have $n$ numbers $1 \ldots n$ and you uniformly sample from this distribution with replacement $n$ times. What is the expected number of unique values you will draw? -->

<!-- ## What is the expected value of the maximum of two dice rolls? -->

## How many cards would you expect to draw from a standard 52-card deck before you get your first ace?
11 rolls.

Because there are 4 aces in a deck of 52 cards, the probability of drawing an ace on any given draw is $\frac{4}{52} = \frac{1}{13}$. 

The expected number of draws until the first success (drawing an ace) in a sequence of independent Bernoulli trials is given by the formula:
$E[X] = \frac{1}{p}$

where $p$ is the probability of success on each trial.
In this case, $p = \frac{1}{13}$, so the expected number of draws until the first ace is:
$E[X] = \frac{1}{\frac{1}{13}} = 13$.




## Estimating the Parameter of an Exponential Distribution

***Say you have $N$ iid draws from an exponential random variable. What is the best estimator for the rate parameter $\lambda$?***

The best estimator for the rate parameter $\lambda$ of an exponential distribution based on $N$ iid draws is the maximum likelihood estimator (MLE). The MLE for $\lambda$ can be derived as follows:
Given $N$ iid draws $X_1, X_2, ..., X_N$ from an exponential distribution with rate parameter $\lambda$, the likelihood function is:
$$ L(\lambda) = \prod_{i=1}^{N} \lambda e^{-\lambda X_i} = \lambda^N e^{-\lambda \sum_{i=1}^{N} X_i} $$
To find the MLE, we take the natural logarithm of the likelihood function:
$$ \ln L(\lambda) = N \ln \lambda - \lambda \sum_{i=1}^{N} X_i $$
Next, we take the derivative of $\ln L(\lambda)$ with respect to $\lambda$ and set it to zero to find the critical points:
$$ \frac{d}{d\lambda} \ln L(\lambda) = \frac{N}{\lambda} - \sum_{i=1}^{N} X_i = 0 $$
Solving for $\lambda$ gives us the MLE:
$$ \hat{\lambda} = \frac{N}{\sum_{i=1}^{N} X_i} $$
Thus, the best estimator for the rate parameter $\lambda$ of an exponential distribution based on $N$ iid draws is $\hat{\lambda} = \frac{N}{\sum_{i=1}^{N} X_i}$, which is the reciprocal of the sample mean of the observed data.

<!-- ## Noodle Loops

***There are 100 noodles in a bowl. At each step, you randomly select two noodle ends from the bowl and tie them together. What is the expectation on the number of loops formed?*** -->

<!-- ## Sequences of Uniform RVs

***Assume you are drawing from an infinite set of standard uniform iid random variables. You keep drawing as long as the sequence you are getting is monotonically increasing. What is the expected number of draws you will make?*** -->

<!-- ## Sampling Sums of Uniform RVs

***Say you continually sample from some iid standard uniform random variables until the sum exceeds 1. What is the expected number of samples you will draw?*** -->

<!-- ## How do you uniformly sample points at random from a circle with radius $r$? -->

## Say we have to random variables $X$ and $Y$. What does it mean for them to be independent? Uncorrelated? Give an example of two random variables that are uncorrelated but not independent.

$X \sim N(0, 1)$ and $Y = X^2$.

$\implies E[X \cdot Y]= E[X \cdot X^2] = E[X^3]= 0$ (since the odd moments of a standard normal distribution are zero), and thus 

$Cov(X, Y) = E[XY] - E[X]E[Y]= E[X^3] - E[X]E[Y]= 0 - 0 \cdot E[Y]= 0$

$\implies Cor(X, Y) = \frac{Cov(X, Y)}{\sqrt{Var(X)Var(Y)}} = 0$

<!-- ## Coupon Collector's Problem

***What is the expected number of rolls of a fair six-sided die until all six sides have appeared at least once?*** -->

<!-- ## Say you're rolling a fair, six-sided die. What is the expected number of rolls until you roll two consecutive fives? -->

## Question #1

Someone is fitting a linear regression model with a predictor (y) regressed on two variables (x1 and x2). They are trying to decide if they should also include an interaction between x1 and x2 in their model or not. What would be the most reasonable consideration in making this decision:

A. Whether or not x1 and x2 are independent.

B. Whether or not x1 and x2 are highly correlated.

C. Whether or not the interaction improves the fit of the predicted y values vs the actual y values on test data.

D. Whether or not the intercept is statistically significant in the model.

E. Whether or not the Kolmogorov-Smirnov test for normality is statistically significant for the residuals from the model.

### Answer

C. Whether or not the interaction improves the fit of the predicted y values vs the actual y values on test data.

## Question #2

***Someone is concerned that the p-values in their A/B experiment platform are not correct. In order to investigate they run 100 (unrelated, non-overlapping) experiments using that platform in which the test and control conditions are set to be the same. (These are sometimes called "A/A tests".) They use a significance level of alpha=.05. What should be true of the resulting 100 p-values?***

A. As the experiments run longer and longer, the p-values should get closer and closer to zero.

B. The p-values should all be near zero.

C. The p-values should all be near one.

D. The p-values should all be near 0.05.

E. Roughly 10% of the p-values should be below 0.10.

F. More than 5% of the p-values should be below 0.05.

G. Less than 5% of the p-values should be below 0.05.

H. The p-values will have a fairly symmetric and unimodal distribution with a peak near .50.

### Answer

E. Roughly 10% of the p-values should be below 0.10.

## Question #3

***Someone is fitting a linear regression model with two predictors x1 and x2. The x2 predictor is ordinal in nature taking the three values small, medium and large. They decide to encode this as small=1, medium=2 and large=3 and simply include it in the model as a linear term. This could be problematic for a number of reasons. Which of the following concerns would represent the strongest argument for NOT doing this.***

A. This model allows for predictions for x2 at values that are not 1,2 or 3. For example, it allows a prediction for x2=1.5 which is not meaningful.

B. This model allows for predictions for x2 at values outside of the range of 1 to 3. For example, it allows a prediction for x2=4 which is not meaningful.

C. If x1 is held constant, this model assumes that the expected response for x2=3 is three times the expected response for x2=1 which may not be true.

D. If x1 is held constant, this model assumes that the expected difference in the response for x2=3 vs x2=1 is twice that of x2=2 vs x2=1 which may not be true.

E. This model assumes that x2 has a roughly equal number of observations for x2=1 and x2=2 and x2=3 which may not be true.

### Answer

D. If x1 is held constant, this model assumes that the expected difference in the response for x2=3 vs x2=1 is twice that of x2=2 vs x2=1 which may not be true.

## Question #4

***Two data scientists are doing analysis of two categorical variables (country = USA/Canada/Mexico and phone type = iPhone/Android/other) as it relates to a numeric response variable (life expectancy in years). One data scientist simply analyzes the 9 means and the other fits a linear regression model. They arrive at very different conclusions regarding which combination of country and phone type has the highest life expectancy. What is the most likely reason?***

A. If the life expectancy is right skewed, the model assumptions may be violated.

B. If the model does not include an interaction term, it may give quite different results from the analysis of the 9 means.

C. The model will properly control for the right-censoring in the data, while the analysis of the 9 means ignores this.

D. The 'other' category may have a small sample size and may be removed from the model due to lack of statistical significance.

E. The proper model is a logistic regression model which would have been equivalent to the analysis of the 9 means.

### Answer

B. If the model does not include an interaction term, it may give quite different results from the analysis of the 9 means.

Often interview candidates struggle to understand what models are “doing under the hood” and instead view even simple models as a black box. With this question we wanted to test to see if respondents understood that the predictions from a model that includes all interactions are simply just the corresponding means in each cell.

This is a great question that tests understanding of linear models with categorical predictors.

A linear regression with two categorical variables but *no interaction* assumes the effect of country is the same regardless of phone type (and vice versa). This is called an **additive model**—it estimates main effects only.

With an additive model, the predicted mean for each combination is: $\mu + \alpha_{\text{country}} + \beta_{\text{phone}}$. The "best" combination would simply be the best country plus the best phone type.

But the raw 9-cell means capture the *actual* combination-specific averages, including any interaction effects. It's entirely possible that, say, (Canada, Android) has the highest raw mean even though Canada isn't the best country on average and Android isn't the best phone type on average.

This is a classic distinction between main-effects-only models and saturated models (or equivalently, models with interactions). For your Google interview context, this connects to a fundamental evaluation design principle: when comparing LLM performance across multiple factors (say, model version × prompt template × task type), you need to decide whether to model interactions or assume additivity—and that choice can lead to very different conclusions about which configuration is "best."


## Question #5

***A data scientist is trying to predict the future sale price of houses. For their predictions, they are considering using either the average sale price of the three (k=3) geographically closest houses that most recently sold or the average sale price of the ten (k=10) geographically closest houses that most recently sold. Which of the following statements is most correct?***

A. k=3 will always work best because k=3 is more similar to k=1 for which the median and the mean are the same and the median is more robust to outliers than the mean.

B. k=3 may work best because the other 7 houses may be quite different.

C. k=10 will always work best because it uses more of the data.

D. k=10 may work best because it will include a more diverse set of houses.

E. k=3 will work best because 3 is an odd number and 10 is even.

### Answer

B. k=3 may work best because the other 7 houses may be quite different.


## Question #6

***A data scientist has counts of people broken down by country = USA / Canada / Mexico and phone type = iPhone / Android / other. They want to do some modeling and analysis with this data but first want to determine if country and phone type are independent. They carry out a chi-squared hypothesis test for independence using this and conclude that they are indeed independent based on this. Going forward in the rest of the analysis they assume independence. What could be a problem with using the chi-squared hypothesis test for independence to conclude independence here?***

A. There may be missing data which needs to be accounted for or imputed.

B. The analysis described does not control for multiple testing.

C. If the data set is large, the hypothesis test may be overpowered.

D. The data may not be normally distributed so the hypothesis test is invalid.

E. A chi-squared hypothesis test for independence will almost always show independence if the sample size is sufficiently small.

### Answer

E. A chi-squared hypothesis test for independence will almost always show independence if the sample size is sufficiently small.

The null hypothesis of the chi-squared test for independence is that the two categorical variables are independent. If the sample size is small, the test may not have enough power to detect a true association between the variables, leading to a failure to reject the null hypothesis and an incorrect conclusion of independence. Therefore, it is important to ensure that the sample size is sufficiently large when using the chi-squared test for independence to draw conclusions about the relationship between categorical variables.

Choice C is correct in many situations, but in this particular situation because the question says “conclude that they are indeed independent” we know the null hypothesis was not rejected. Thus, being underpowered is a potential concern (choice E) but being overpowered (choice C) is irrelevant.

The question is perhaps made even more difficult since the language “conclude that they are indeed independent” reads as if the null hypothesis is determined to be true, which we know is commonly said but not technically correct; we should only say it is not determined to be untrue. Thus this question really forces the respondents to think and read carefully to differentiate being over/under powered and rejecting/not rejecting the null hypothesis.

## Question #7: 

***A data scientist wants to remove outliers in their dataset. They decide to remove anything more than 3 standard deviations above or below the mean. What could be problematic about this approach?***

A. They have not confirmed that the data is normally distributed. Many distributions have a very large fraction of data outside this range and they may be removing a majority (more than 50%) of the data if it isn't normal.

B. The standard deviation itself is impacted by outliers. It is better to use the interquartile range multiplied by a constant.

C. For large datasets, the standard deviation will be close to zero since it scales with the square root of the sample size.

D. The usual number to use is two standard deviations. It is very unusual to use three standard deviations without a strong reason.

### Answer

B. The standard deviation itself is impacted by outliers. It is better to use the interquartile range multiplied by a constant. *(How sensitive the standard deviation is to outliers depends on the distribution. For a normal distribution, it is not very sensitive, but for a heavy-tailed distribution, it can be very **(?)** sensitive.)*

A is ruled out by Chebyshev's inequality, which states that for any distribution, at least 1 - 1/k^2 of the data will be within k standard deviations of the mean. For k=3, this means that at least 8/9 (approximately 88.89%) of the data will be within 3 standard deviations of the mean, regardless of the distribution. Therefore, it is not possible for more than 50% of the data to be outside this range.

## Question #8: 

***The variable y is numeric, the variable x1 is numeric and the variable x2 is categorical taking on 5 unique values representing 5 colors (the five values of x2 are red, blue, yellow, green and purple.) A data scientist fits a linear regression model with y as the response and x1 and x2 as predictors (an intercept was fit as well but no interactions). A second data scientist decides instead to fit 5 separate regression models. In other words, they fit a regression model with y as the response and x1 as the predictor for each of the 5 colors separately (and each regression model includes an intercept). Both data scientists are fitting the models using the same (training) dataset. Which of the following is true regarding the predictions for the response variable y from these two approaches?***

A. These two approaches will give exactly the same predictions for all values of x1 and x2.

B. These two approaches will give exactly the same predictions for all values of x1 in the training data, but may give different predictions for test data with new values for x1.

C. These two approaches will give exactly the same predictions for all values of x1 and x2 in the training data, but may give different predictions for test data with new values for x1 and x2.

D. These two approaches may give different predictions. They would give exactly the same predictions if the first data scientist had included the interaction between x1 and x2.

E. These two approaches may give different predictions. They would give exactly the same predictions if x1 and x2 are statistically independent.

F. These two approaches may give different predictions. They would give exactly the same predictions if there are equal numbers of observations for the five colors red, blue, yellow, green and purple.

### Answer

D. These two approaches may give different predictions. They would give exactly the same predictions if the first data scientist had included the interaction between x1 and x2.

This is because the first data scientist's model assumes that the effect of x1 on y is the same across all colors (i.e., no interaction), while the second data scientist's approach allows for different effects of x1 on y for each color by fitting separate models. If the first data scientist had included an interaction term between x1 and x2, then the predictions from both approaches would be the same for all values of x1 and x2 in the training data, as both models would be able to capture the varying relationship between x1 and y across different colors.

Formally, the first data scientist's model can be expressed as:
$$\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x_1 + \hat{\beta}_{2} x_2$$
Since $x_2$ is categorical, we get 4 dummy variables (purple is omitted)
$$\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x_1 + \hat{\beta}_{2,red} \mathbb{I}\{x_2=red\} + \hat{\beta}_{2,blue} \mathbb{I}\{x_2=blue\} + \hat{\beta}_{2,yellow} \mathbb{I}\{x_2=yellow\} + \hat{\beta}_{2,green} \mathbb{I}\{x_2=green\}$$

The second data scientist's models can be expressed as:
$$\hat{y}_{red} = \hat{\beta}_{0,red} + \hat{\beta}_{1,red} x_1$$
$$\hat{y}_{blue} = \hat{\beta}_{0,blue} + \hat{\beta}_{1,blue} x_1$$
$$\hat{y}_{yellow} = \hat{\beta}_{0,yellow} + \hat{\beta}_{1,yellow} x_1$$
$$\hat{y}_{green} = \hat{\beta}_{0,green} + \hat{\beta}_{1,green} x_1$$
$$\hat{y}_{purple} = \hat{\beta}_{0,purple} + \hat{\beta}_{1,purple} x_1$$

Note that

$$y = [y_{red}, y_{blue}, y_{yellow}, y_{green}, y_{purple}]^T$$

If the first data scientist had included the interaction term, their model would be:
$$\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x_1 + \ldots + \hat{\beta}_{2,green} \mathbb{I}\{x_2=green\} + \hat{\beta}_{3,red} x_1 \cdot \mathbb{I}\{x_2=red\} + \ldots + \hat{\beta}_{3,green} x_1 \cdot \mathbb{I}\{x_2=green\}$$

The interaction terms would effectively limit the data in each model to each $y \in \{y_{red}, y_{blue}, y_{yellow}, y_{green}, y_{purple}\}$.

For example, the predictions for the red color would be:
$$\hat{y}_{red} = \hat{\beta}_0 + \hat{\beta}_1 x_1 + \hat{\beta}_{2,red} + \hat{\beta}_{3,red} x_1$$
and since the first data scientist's model with the interaction term would be fit using the same data as the second data scientist's model for red, we would have

$$\hat{\beta}_0 + \hat{\beta}_{2,red} = \hat{\beta}_{0,red}$$

and

$$(\hat{\beta}_1 + \hat{\beta}_{3,red})x_1 = \hat{\beta}_{1,red}x_1$$

thus giving the same predictions for red.

#### Factoring the likelihood into a product of five functions (one for each color) to show how the two methods are only equivalent if the interaction term is included

Assume the standard linear regression setup with Gaussian errors. Let's denote observations within color group $c$ as $(y_{ic}, x_{1,ic})$ for $i = 1, \ldots, n_c$.

*The Likelihood for the Separate Regressions (Approach 2)*

When we fit five separate regressions, we're assuming:

$$y_{ic} = \alpha_{0c} + \alpha_{1c} x_{1,ic} + \varepsilon_{ic}, \quad \varepsilon_{ic} \sim N(0, \sigma_c^2)$$

Each color has its own intercept $\alpha_{0c}$, its own slope $\alpha_{1c}$, and potentially its own variance $\sigma_c^2$.

The full likelihood factors naturally:

$$L(\theta) = \prod_{c=1}^{5} \prod_{i=1}^{n_c} \frac{1}{\sqrt{2\pi\sigma_c^2}} \exp\left(-\frac{(y_{ic} - \alpha_{0c} - \alpha_{1c}x_{1,ic})^2}{2\sigma_c^2}\right)$$

This separates cleanly into five independent optimization problems:

$$L(\theta) = \prod_{c=1}^{5} L_c(\alpha_{0c}, \alpha_{1c}, \sigma_c^2)$$

Because there are no shared parameters across groups, maximizing the full likelihood is equivalent to maximizing each $L_c$ separately. This is precisely what fitting five separate regressions does.

*The Likelihood for the Pooled Model Without Interaction (Approach 1)*

$$y_{ic} = \beta_0 + \beta_1 x_{1,ic} + \gamma_c + \varepsilon_{ic}, \quad \varepsilon_{ic} \sim N(0, \sigma^2)$$

where $\gamma_c$ are the color-specific intercept adjustments (with one reference category set to zero).

The likelihood is:

$$L(\theta) = \prod_{c=1}^{5} \prod_{i=1}^{n_c} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_{ic} - \beta_0 - \beta_1 x_{1,ic} - \gamma_c)^2}{2\sigma^2}\right)$$

We can write this as:

$$L(\theta) = \prod_{c=1}^{5} L_c(\beta_0, \beta_1, \gamma_c, \sigma^2)$$

**Here's the critical issue:** Even though the likelihood factors into a product over colors, the functions share parameters:
* $\beta_1$ appears in all five factors
* $\sigma^2$ appears in all five factors

When we maximize, we can't optimize each color group independently. The slope $\beta_1$ must balance the fit across all five groups simultaneously. The MLE for $\beta_1$ is a weighted combination of what would be optimal for each group separately.

This is why predictions differ from Approach 2.

*The Likelihood for the Pooled Model With Interaction*

$$y_{ic} = \beta_0 + \beta_{1c} x_{1,ic} + \gamma_c + \varepsilon_{ic}, \quad \varepsilon_{ic} \sim N(0, \sigma^2)$$

Notice that now each color has its own slope $\beta_{1c}$.

The likelihood becomes:

$$L(\theta) = \prod_{c=1}^{5} \prod_{i=1}^{n_c} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_{ic} - \beta_0 - \beta_{1c} x_{1,ic} - \gamma_c)^2}{2\sigma^2}\right)$$

Now examine how parameters appear in each factor $L_c$:

- $\beta_0 + \gamma_c$ only affects group $c$ (it's the effective intercept for that color)
- $\beta_{1c}$ only affects group $c$
- Only $\sigma^2$ is shared

For the intercept and slope parameters, the first-order conditions for each group decouple. The MLEs for $\beta_0 + \gamma_c$ and $\beta_{1c}$ depend only on data from group $c$.

Taking partial derivatives with respect to $\beta_{1c}$ for group $c$:

$$\frac{\partial \log L}{\partial \beta_{1c}} = \frac{1}{\sigma^2} \sum_{i=1}^{n_c} (y_{ic} - \beta_0 - \beta_{1c} x_{1,ic} - \gamma_c) x_{1,ic} = 0$$

This equation involves only group $c$ data and parameters. The solution for $\beta_{1c}$ is exactly what you'd get from regressing $y$ on $x_1$ within group $c$ alone.

***The One Remaining Difference: Error Variance***

You might notice that $\sigma^2$ is still shared in the interaction model. Doesn't this prevent full equivalence?

For **point predictions** (fitted values), this doesn't matter. The MLEs for slopes and intercepts are obtained by solving the normal equations, which don't depend on $\sigma^2$. The predicted values $\hat{y}_{ic} = \hat{\beta}_0 + \hat{\beta}_{1c} x_{1,ic} + \hat{\gamma}_c$ are identical to those from separate regressions.

Where it would matter:

- Standard errors of coefficients
- Prediction intervals
- Hypothesis tests

The pooled model with interaction assumes homoskedasticity across groups, while separate regressions allow heteroskedasticity. But for predictions alone—which is what the question asks about—the two approaches are equivalent.

*Summary*

| Model | Shared Parameters | Likelihood Factorization | Predictions |
|-------|-------------------|--------------------------|-------------|
| No interaction | $\beta_1$, $\sigma^2$ | Factors, but coupled optimization | Different from separate regressions |
| With interaction | $\sigma^2$ only | Factors with decoupled slope/intercept estimation | Identical to separate regressions |

The interaction term "liberates" each color group to find its own slope, making the likelihood optimization separable for the parameters that determine predictions.

## Question #9

***In a certain time period, an economist noted that average sale prices for cars had increased 10%. They split the data into gasoline cars and non-gasoline cars (two mutually exclusive categories) to analyze further. They observed that average sale prices for each were decreasing over this time period. Which of the following is true?***

A. This can not be possible. At least one of the two must have increased if the overall average sale price increased.

B. We know total gasoline car sales decreased over this time period.

C. We know total non-gasoline car sales increased over this time period.

D. We know gasoline cars and non-gasoline cars have different average sale prices.

### Answer

D. We know gasoline cars and non-gasoline cars have different average sale prices.

This is an example of Simpson's paradox, where a trend that appears in different groups of data can disappear or reverse when the groups are combined. In this case, it is possible for the overall average sale price to increase while the average sale prices for both gasoline and non-gasoline cars decrease if the proportion of sales shifts significantly between the two categories. For example, if there was a large increase in sales of higher-priced non-gasoline cars and a decrease in sales of lower-priced gasoline cars, the overall average could increase even though both categories individually show a decrease in average sale price. Therefore, we cannot conclude that either category must have increased or decreased based solely on the overall average sale price trend.

## Question #10

***A statistical model for market share showed that in June market share was not statistically significantly different from the target value of 50% market share. The same model using the same data showed that in July of the same year, the market share was indeed now statistically significantly different from the target value of 50% market share. Which of the following must be true?***

A. The difference between June and July market share is statistically significant.

B. The difference between June and July market share is not statistically significant.

C. The difference between June and July market share is statistically significant if using a significance level of $2\alpha$ where $\alpha$ is the original significance level.

D. The difference between June and July market share is statistically significant if using a significance level of $\alpha/2$ where $\alpha$ is the original significance level.

E. None of these.

### Answer
E. None of these.

The statistical significance of the market share in June and July being different from the target value of 50% does not necessarily imply that the difference between June and July market share is statistically significant. The statistical significance of each month being different from the target value is based on a comparison to a fixed value (50%), while the statistical significance of the difference between June and July market share would require a direct comparison between the two months. It is possible for both months to be statistically significantly different from 50% but not statistically significantly different from each other, or for one month to be statistically significantly different from 50% while the other is not, which would also affect the significance of their difference. Therefore, we cannot conclude any of the options A, B, C, or D without additional information about the data and the specific tests conducted.

This question evaluates a respondent’s understanding of the concept of statistical significance. The question is closely related to the problem discussed in the paper with the very descriptive title “The Difference Between “Significant” and “Not Significant” is not Itself Statistically Significant” by Andrew Gelman and Hal Stern (The American Statistician, Nov 2006, Vol. 60, No. 4). One of the authors of this blog post also encountered a high profile instance of this situation (and a large misunderstanding surrounding it) in their work at Google, which was part of the motivation for including the question.

One effective strategy for correctly answering this question is for a respondent to imagine situations which are consistent with the information provided in the question prompt but provide a counterexample to each incorrect answer choice. For instance for choice B one could imagine a situation where June is 50% +/- 5% and July is 90% +/- 5% and this large difference between June and July results in statistical significance, thus disproving choice B.


