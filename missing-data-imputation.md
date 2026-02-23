# Missing Data Imputation

## Types of Missing Data
* MCAR (Missing Completely at Random): The probability of missingness is the same for all observations. Example: A survey respondent accidentally skips a question.
* MAR (Missing at Random): The probability of missingness is related to observed data but not the missing data itself. Example: Older individuals are less likely to report their income, but age is recorded.
* MNAR (Missing Not at Random): The probability of missingness is related to the missing data itself. Example: Individuals with higher incomes are less likely to report their income.

## Common Imputation Techniques
* Mean/Median/Mode Imputation: Replace missing values with the mean, median, or mode of the observed data.
* ffill
* bfill
* Interpolation: Estimate missing values using linear or polynomial interpolation based on surrounding data points.
* K-Nearest Neighbors (KNN) Imputation: Use the values of the nearest neighbors to impute missing values.
* Multiple Imputation: Create multiple datasets with different imputed values and combine results to account for uncertainty.
* Model-Based Imputation: Use regression or machine learning models to predict and fill in missing values.
* MICE (Multiple Imputation by Chained Equations): A sophisticated method that models each variable with missing data as a function of other variables in a round-robin fashion.