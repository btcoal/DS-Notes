# Poisson Regression

## Questions

* What is Poisson regression?

* When would you use Poisson regression?

* What are the assumptions of Poisson regression?

* How do you interpret the coefficients of a Poisson regression model?

* How do you assess the goodness-of-fit of a Poisson regression model?

* How do you handle overdispersion in Poisson regression?

* What are some alternatives to Poisson regression for count data?

## Tools and Libraries

* `R`: `glm()` function with `family = poisson()`

* `Python`: `statsmodels` library with `GLM` class and `family=sm.families.Poisson()`

## Example

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
# Generate some example data
np.random.seed(42)
n = 1000
X1 = np.random.normal(size=n)
X2 = np.random.normal(size=n)

# True coefficients
beta0 = 0.5
beta1 = 0.3
beta2 = -0.2

# Generate Poisson-distributed response variable
mu = np.exp(beta0 + beta1 * X1 + beta2 * X2)
Y = np.random.poisson(mu)
data = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2})
# Fit Poisson regression model
model = smf.glm(formula='Y ~ X1 + X2', data=data, family=sm.families.Poisson()).fit()
print(model.summary())
```

```
Generalized Linear Model Regression Results                  
==============================================================================
Dep. Variable:                      Y   No. Observations:                 1000
Model:                            GLM   Df Residuals:                      997
Model Family:                 Poisson   Df Model:                            2
Link Function:                    Log   Scale:                          1.0000
Method:                          IRLS   Log-Likelihood:                -1577.2
Date:                Sat, 04 Oct 2025   Deviance:                       1124.9
Time:                        13:38:53   Pearson chi2:                     993.
No. Iterations:                     5   Pseudo R-squ. (CS):             0.1632
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.4898      0.025     19.343      0.000       0.440       0.539
X1             0.2644      0.024     10.804      0.000       0.216       0.312
X2            -0.1836      0.024     -7.552      0.000      -0.231      -0.136
==============================================================================
```

```R
# Generate some example data
set.seed(42)
n <- 1000
X1 <- rnorm(n)
X2 <- rnorm(n)
# True coefficients
beta0 <- 0.5
beta1 <- 0.3
beta2 <- -0.2

# Generate Poisson-distributed response variable
mu <- exp(beta0 + beta1 * X1 + beta2 * X2)
Y <- rpois(n, mu)
data <- data.frame(Y = Y, X1 = X1, X2 = X2)
# Fit Poisson regression model
model <- glm(Y ~ X1 + X2, data = data, family = poisson())
summary(model)
```

## References