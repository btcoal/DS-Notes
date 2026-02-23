# Generalized Linear Models

## OLS

### Assumptions

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
