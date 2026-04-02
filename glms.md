# Generalized Linear Models

## OLS

### Assumptions

On one level, linear regression via minimization of mean squared error (MSE) is a purely algorithmic solution to the problem of estimating $\mathbb{E}[Y \mid X]$, *assuming* $Y = \beta_0 + \beta_1 X + \varepsilon$. Increasing structure and assumptions buys you increasing *inferential power*. 

You can think of linear regression as a ladder of assumptions layered on top of the basic objective "approximate $\mathbb{E}[Y \mid X]$." Each additional assumption converts a purely predictive procedure into something with stronger statistical guarantees and interpretability.

Start with the weakest structure and move upward.

**Linear Conditional Mean:** $\mathbb{E}[Y \mid X] = \beta_0 + \beta_1 X$. This is the core modeling assumption. It "buys" you interpretability and identifiability of a low-dimensional parameter. Without it, you are doing nonparametric regression. With it, you compress the problem into estimating a finite set of coefficients, and the solution becomes stable in small samples.

**Exogeneity:** $\mathbb{E}[\varepsilon \mid X] = 0$. This is the most important assumption for inference. It buys you unbiasedness and consistency of the OLS estimator. Without exogeneity, your coefficients absorb systematic bias, and the interpretation of ( \beta_1 ) as a causal or even descriptive slope of the conditional mean breaks down.

Then you typically assume i.i.d. sampling (or at least independence across observations).
This buys you standard convergence results like the law of large numbers and central limit theorem in their simplest form. It ensures that your estimator concentrates around the true parameter and that variance formulas behave as expected. If this fails, you move into clustered or time-series settings where corrections are required.

**Homoskedasticity.** $\mathrm{Var}(\varepsilon \mid X) = \sigma^2$. This does not affect unbiasedness, but it buys you efficiency and simple variance formulas. Under homoskedasticity, OLS is the best linear unbiased estimator (Gauss–Markov theorem). If it fails, OLS remains unbiased but no longer efficient, and standard errors must be adjusted (e.g., robust/White standard errors).

**No perfect multicollinearity.** Implies identifiability of coefficients. If predictors are perfectly collinear, the parameter vector is not uniquely defined and estimation becomes ill-posed.

**Normality of errors:** $\varepsilon \sim \mathcal{N}(0, \sigma^2)$. Strengthens things further. This buys you exact finite-sample inference. Test statistics (t, F) have exact distributions rather than relying on asymptotics. Without normality, inference is still valid asymptotically, but not exact in small samples.

If you further assume correct model specification beyond linearity (**no omitted variables**, **correct functional form**), you gain valid causal interpretation under appropriate conditions.

**no measurement error in $X$** Consistency of slope estimates. Measurement error in regressors biases coefficients toward zero *attenuation bias*, so ruling it out preserves interpretability.

So the progression looks like this in terms of what you gain:
* Linear conditional mean → tractable, interpretable model
* Exogeneity → unbiasedness and consistency
* Independence/i.i.d. → reliable convergence and variance estimation
* Homoskedasticity → efficiency and simple standard errors
* No multicollinearity → identifiability
* Normality → exact finite-sample inference
* Correct specification → causal interpretability

The key insight is that prediction requires very little structure, while inference—especially causal inference—requires a stack of increasingly strong assumptions.

It helps to separate the assumptions into three categories: those that affect **existence/identifiability**, those that affect **bias/consistency**, and those that affect **efficiency/inference**. Your question is about the second category, so I’ll focus tightly there and anchor everything to the behavior of ( \hat{\beta} ).

Start with the core condition:

If ( \mathbb{E}[\varepsilon \mid X] = 0 ) (exogeneity), then
( \hat{\beta} ) is **unbiased** and **consistent**.

This is the pivotal assumption. It implies that the regressors are uncorrelated with the error term in expectation. Under standard regularity conditions, this gives:

* ( \mathbb{E}[\hat{\beta}] = \beta ) (unbiasedness, finite sample)
* ( \hat{\beta} \xrightarrow{p} \beta ) (consistency)

If this condition fails, both properties generally fail. In particular:

If ( \mathbb{E}[\varepsilon \mid X] \neq 0 ), then
( \hat{\beta} ) is **biased** and **inconsistent**.

This is the omitted variable / endogeneity case. The estimator converges, but to the wrong quantity:
[
\hat{\beta} \xrightarrow{p} \beta + \text{bias term}
]
where the bias term depends on the correlation between ( X ) and the omitted structure in ( \varepsilon ).

Now consider the other assumptions in relation to bias/consistency:

Independence / i.i.d. sampling
This is not strictly required for unbiasedness if exogeneity holds, but it is typically required for **consistency via LLN/CLT**. If observations are dependent (e.g., time series), OLS can still be consistent, but you need weaker conditions like ergodicity or mixing. If those fail, consistency can break.

Homoskedasticity
( \mathrm{Var}(\varepsilon \mid X) = \sigma^2 ) has **no effect on bias or consistency**.
Even under heteroskedasticity:

* ( \hat{\beta} ) remains unbiased (given exogeneity)
* ( \hat{\beta} ) remains consistent
  What changes is efficiency and the correctness of standard errors.

No perfect multicollinearity
This is about **identifiability**, not bias per se. If ( X^\top X ) is not invertible, then:

* ( \hat{\beta} ) is not uniquely defined
  So unbiasedness and consistency are not even well-posed questions. When the matrix is invertible (full rank), then:
* unbiasedness and consistency follow from exogeneity

Normality of errors
No impact on bias or consistency. It only affects finite-sample distributions.

Measurement error in ( X )
This directly breaks exogeneity. Classical measurement error induces:

* **bias toward zero (attenuation bias)**
* **inconsistency**
  Here, ( X ) becomes correlated with the composite error term.

Putting this together in a compact mapping:

* Exogeneity → determines **both unbiasedness and consistency**
* Independence / weak dependence → supports **consistency**
* No multicollinearity → ensures **β is identifiable at all**
* Homoskedasticity → irrelevant for bias/consistency
* Normality → irrelevant for bias/consistency
* Measurement error / omitted variables → destroy **both**

The key structural result is that OLS is essentially a projection:
[
\hat{\beta} = (X^\top X)^{-1} X^\top Y
]
and substituting ( Y = X\beta + \varepsilon ) gives:
[
\hat{\beta} = \beta + (X^\top X)^{-1} X^\top \varepsilon
]

Everything reduces to the behavior of ( (X^\top X)^{-1} X^\top \varepsilon ).
Exogeneity ensures its expectation goes to zero and its sample analog vanishes asymptotically. When that fails, the term converges to a nonzero limit, which is exactly the source of bias and inconsistency.


## Colinearity

### Variance Inflation Factor (VIF)

In a linear model: $\hat{y} = g(X; \beta)$,

$$
VIF_j = \frac{1}{1 - R_j^2}
$$

where $R_j^2$ is the R-squared value obtained by regressing the j-th feature against all other features. In terms of Residual Sum of Squares (RSS) and Total Sum of Squares (TSS):
$$
VIF_j = \frac{TSS_j}{RSS_j}
$$

### Addressing Multicollinearity
* Remove highly correlated predictors
* Combine correlated predictors using techniques like PCA
* Regularization techniques (Ridge, Lasso)
* ***Keep 'em!*** (for prediction tasks)

## Causal Inference with Linear Models

* Fixed effects
* Random Effects Models
* Mixed Effects Models
* Bayesian Hierarchical Models

## Questions

* What role does the link function play in a GLM?

* How do you fit GLM to data?

* What is a type of GLM beyond logistic regression? When would you use it?

* What are the assumptions of GLMs?

* How do you choose the link function for a GLM?

Image of link functions 

![GLM Link Functions](./glm-link-functions.png)
