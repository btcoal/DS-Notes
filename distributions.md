# Distributions

## Bernoulli

A bernoulli random variable takes value 1 with probability $p$ and value 0 with probability $1-p$.

Density function: 

$$
P(X=x) = p^x (1-p)^{1-x} \quad \text{for } x \in \{0, 1\}
$$

CDF:
$$
F(x) = \begin{cases}
0 & x < 0 \\
1 - p & 0 \leq x < 1 \\
1 & x \geq 1
\end{cases}
$$

Mean and Variance:

$$E[X] = p$$

$$Var(X) = p(1-p)$$

In numpy

`np.random.binomial(n=1, p=p)`, where `n=1` indicates a single trial and `p` is the success probability.

## Binomial

A binomial random variable represents the number of successes in $n$ independent Bernoulli trials, each with success probability $p$.

Density function:
$$
P(X=k) = \binom{n}{k} p^k (1-p)^{n-k} \quad \text{for } k = 0, 1, \ldots, n
$$

CDF:
$$
F(k) = \sum_{i=0}^{k} \binom{n}{i} p^i (1-p)^{n-i} \quad \text{for } k = 0, 1, \ldots, n
$$

Mean and Variance:
$$E[X] = np$$
$$Var(X) = np(1-p)$$

In numpy

`np.random.binomial(n, p)`, where `n` is the number of trials and `p` is the success probability.

## Multinomial

A multinomial random variable generalizes the binomial distribution to more than two categories. It represents the counts of each category in $n$ independent trials, where each trial results in one of $k$ categories with probabilities $p_1, p_2, \ldots, p_k$.

Density function:
$$
P(X_1 = x_1, X_2 = x_2, \ldots, X_k = x_k) = \frac{n!}{x_1! x_2! \ldots x_k!} p_1^{x_1} p_2^{x_2} \ldots p_k^{x_k}
$$
where $x_1 + x_2 + \ldots + x_k = n$.

The CDF for the multinomial distribution is complex and typically not expressed in closed form. It involves summing probabilities over all combinations of counts that satisfy the constraints.

Expanded Mean and Covariance:
$$E[X_i] = n p_i$$
$$Cov(X_i, X_j) = \begin{cases}
n p_i (1 - p_i) & \text{if } i = j \\
-n p_i p_j & \text{if } i \neq j
\end{cases}
$$

In numpy
`np.random.multinomial(n, pvals)`, where `n` is the number of trials and `pvals` is a list of category probabilities that sum to 1.

In pytorch
`torch.multinomial(input, num_samples, replacement=False, *, generator=None)`, where `input` is a 1D tensor of category probabilities, and `num_samples` is the number of samples to draw.

## Geometric

A geometric random variable represents the number of trials until the first success in a sequence of independent Bernoulli trials, each with success probability $p$.

## Discrete Uniform

A discrete uniform random variable takes on each of its $n$ possible values with equal probability.

Density function:
$$
P(X=x) = \frac{1}{n} \quad \text{for } x \in \{x_1, x_2, \ldots, x_n\}
$$

CDF:
$$
F(x) = \frac{k}{n} \quad \text{for } x_k \leq x < x_{k+1}, \; k = 0, 1, \ldots, n
$$


Mean and Variance:
$$E[X] = \frac{x_1 + x_2 + \ldots + x_n}{n}$$
alternatively, for integers from 1 to n:
$$E[X] = \frac{(x_n + x_1)}{2}$$
$$Var(X) = \frac{(x_n - x_1 + 1)^2 - 1}{12}$$

In numpy

`np.random.randint(low, high)` generates integers from `low` (inclusive) to `high` (exclusive), uniformly distributed.

## Discrete Poisson
A poisson random variable represents the number of events occurring in a fixed interval of time or space, given a known average rate $\lambda$.

Useful for modeling arrival processes on platforms like Uber, Airbnb, etc.

Density function:
$$
P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!} \quad \text{for } k = 0, 1, 2, \ldots
$$

CDF:
$$
F(k) = \sum_{i=0}^{k} \frac{\lambda^i e^{-\lambda}}{i!} \quad \text{for } k = 0, 1, 2, \ldots
$$

Mean and Variance:
$$E[X] = \lambda$$
$$Var(X) = \lambda$$

In numpy

`np.random.poisson(lam=lambda)`, where `lambda` is the average rate of occurrence.

*NB: Can alternatively be parameterized with rate parameter $\beta = 1/\lambda$*

## Continuous Uniform

A continuous uniform random variable takes on values in the interval $[a, b]$ with equal probability density.

Density function:
$$
f(x) = \frac{1}{b - a} \quad \text{for } x \in [a, b]
$$

CDF:
$$
F(x) = \begin{cases}
0 & x < a \\
\frac{x - a}{b - a} & a \leq x < b \\
1 & x \geq b
\end{cases}
$$

Mean and Variance:
$$E[X] = \frac{a + b}{2}$$
$$Var(X) = \frac{(b - a)^2}{12}$$

In numpy

`np.random.uniform(low=a, high=b)`, where `a` and `b` define the interval.

## Normal (Gaussian)

A normal random variable is characterized by its mean $\mu$ and variance $\sigma^2$. It can result from the sum of many independent random variables due to the Central Limit Theorem.

Density function:
$$
f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}} \quad \text{for } x \in (-\infty, \infty)
$$

CDF:
$$
F(x) = \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{x - \mu}{\sigma \sqrt{2}} \right) \right]
$$,
where erf is the error function: $ \text{erf}(z) = \frac{2}{\sqrt{\pi}} \int_0^z e^{-t^2} dt $

In numpy

`np.random.normal(loc=mu, scale=sigma)`, where `mu` is the mean and `sigma` is the standard deviation.

In pytorch

`torch.normal(mean=mu, std=sigma, size=(...))`, where `mu` is the mean, `sigma` is the standard deviation, and `size` defines the output shape.

### Standard Normal
A standard normal random variable has mean 0 and variance 1.

Density function:
$$
f(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}} \quad \text{for } z \in (-\infty, \infty)
$$

CDF:
$$
\Phi(z) = \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{z}{\sqrt{2}} \right) \right]
$$

## Exponential
A exponential random variable models the time between events in a Poisson process, characterized by the rate parameter $\lambda$.

Density function:
$$
f(x) = \lambda e^{-\lambda x} \quad \text{for } x \geq 0
$$

CDF:
$$
F(x) = 1 - e^{-\lambda x} \quad \text{for } x \geq 0
$$

Mean and Variance:
$$E[X] = \frac{1}{\lambda}$$
$$Var(X) = \frac{1}{\lambda^2}$$

In numpy

`np.random.exponential(scale=1/lambda)`, where `lambda` is the rate parameter.

## Log-Normal

A log-normal random variable is one whose logarithm is normally distributed. If $Y$ is log-normally distributed, then $X = \ln(Y)$ follows a normal distribution with parameters $\mu$ and $\sigma^2$.

Density function:
$$
f(y) = \frac{1}{y \sigma \sqrt{2\pi}} e^{-\frac{(\ln(y) - \mu)^2}{2\sigma^2}} \quad \text{for } y > 0
$$

CDF:
$$
F(y) = \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{\ln(y) - \mu}{\sigma \sqrt{2}} \right) \right] \quad \text{for } y > 0
$$

Mean and Variance:
$$E[Y] = e^{\mu + \frac{\sigma^2}{2}}$$
$$Var(Y) = (e^{\sigma^2} - 1) e^{2\mu + \sigma^2}$$

In numpy

`np.random.lognormal(mean=mu, sigma=sigma)`, where `mu` and `sigma` are the parameters of the underlying normal distribution.

In pytorch

`torch.distributions.LogNormal(loc=mu, scale=sigma).sample(sample_shape=(...))`, where `mu` and `sigma` are the parameters of the underlying normal distribution, and `sample_shape` defines the output shape.

## Beta

A beta random variable is defined on the interval [0, 1] and is characterized by two shape parameters, $\alpha$ and $\beta$. It is often used to model proportions and probabilities.

Density function:
$$
f(x) = \frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{B(\alpha, \beta)} \quad \text{for } x \in [0, 1]
$$

CDF:
$$
F(x) = \int_{0}^{x} f(t) \, dt
$$

Mean and Variance:
$$E[X] = \frac{\alpha}{\alpha + \beta}$$
$$Var(X) = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$$

In numpy

`np.random.beta(a=alpha, b=beta)`, where `alpha` and `beta` are the shape parameters.
## Gamma

## Chi-Squared

## Student's t

A $t$-distribution arises when estimating the mean of a normally distributed population in situations where the sample size is small and population standard deviation is unknown. It is characterized by its degrees of freedom $\nu$.

Density function:
$$
f(t) = \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\sqrt{\nu \pi} \Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{t^2}{\nu}\right)^{-\frac{\nu + 1}{2}} \quad \text{for } t \in (-\infty, \infty)
$$

CDF:
$$
F(t) = \int_{-\infty}^{t} f(x) \, dx
$$

Mean and Variance:
$$E[T] = 0 \quad \text{for } \nu > 1$$
$$Var(T) = \frac{\nu}{\nu - 2} \quad \text{for } \nu > 2$$

In numpy

`np.random.standard_t(df=nu)`, where `nu` is the degrees of freedom.

## F-Distribution

A $F$-distribution arises in the context of comparing variances and is characterized by two degrees of freedom parameters, $\nu_1$ and $\nu_2$.

Density function:
$$
f(x) = \frac{\sqrt{\frac{(\nu_1 x)^{\nu_1} \nu_2^{\nu_2}}{(\nu_1 x + \nu_2)^{\nu_1 + \nu_2}}}}{x B\left(\frac{\nu_1}{2}, \frac{\nu_2}{2}\right)} \quad \text{for } x > 0
$$

CDF:
$$
F(x) = \int_{0}^{x} f(t) \, dt
$$

Mean and Variance:
$$E[F] = \frac{\nu_2}{\nu_2 - 2} \quad \text{for } \nu_2 > 2$$
$$Var(F) = \frac{2 \nu_2^2 (\nu_1 + \nu_2 - 2)}{\nu_1 (\nu_2 - 2)^2 (\nu_2 - 4)} \quad \text{for } \nu_2 > 4$$

In numpy

`np.random.f(dfnum=nu1, dfden=nu2)`, where `nu1` and `nu2` are the degrees of freedom parameters.


