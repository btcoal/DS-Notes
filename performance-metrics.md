# Performance Metrics and Evaluation

### At-a-Glance Guide to Core Metrics and Loss Functions
| Topic                         | Key Discussion Point                                                     |
| ----------------------------- | ------------------------------------------------------------------------ |
| **Robustness**                | Compare MAE, MdAE, Huber for outlier resistance.                         |
| **Scale dependence**          | Which metrics allow cross-series comparisons? (MASE, MAPE, sMAPE)        |
| **Distributional evaluation** | When are CRPS and NLL needed vs RMSE?                                    |
| **Differentiability**         | Which metrics are non-smooth and how does that affect training?          |
| **Interpretability**          | Which metric communicates best to stakeholders (RMSE, MAPE)?             |
| **Business cost alignment**   | How to choose asymmetric or quantile losses for cost-sensitive problems? |

The following table provides a high-level schematic of the evaluation landscape. It serves as a conceptual map for the topics covered in this guide, connecting key metrics and loss functions to their primary domains, core concepts, and principal use cases.

| Metric / Loss Function | Primary Domain(s) | Core Concept / What it Measures | Primary Use Case / When to Prioritize |
|---|---|---|---|
| **Mean Squared Error (MSE)** | Regression, Time Series | Average squared difference; penalizes large errors heavily. Minimizing MSE predicts the conditional mean. | When large errors are disproportionately costly; when the business objective is to predict an aggregate or expected value.  |
| **Mean Absolute Error (MAE)** | Regression, Time Series | Average absolute difference; treats all errors linearly. Minimizing MAE predicts the conditional median. | When the dataset contains outliers that are considered noise; when predicting a "typical" user experience.  |
| **Huber Loss** | Regression | A hybrid of MSE for small errors and MAE for large errors, controlled by a delta parameter. | When seeking a balance between sensitivity to large errors and robustness to extreme outliers.  |
| **Cross-Entropy Loss** | Classification | Measures the dissimilarity between the predicted probability distribution and the true distribution. | The standard loss function for training classification models, particularly with probabilistic outputs (e.g., logistic regression, neural networks).  |
| **Recall (Sensitivity)** | Classification, Ranking | The fraction of all actual positives that were correctly identified by the model. | When the cost of a false negative (missing a positive case) is very high (e.g., disease diagnosis, fraud detection).  |
| **Precision** | Classification, Ranking | The fraction of positive predictions that were actually correct. | When the cost of a false positive (a false alarm) is very high (e.g., flagging a non-spam email as spam, flagging a valid transaction as fraud).  |
| **ROC AUC** | Classification | Area Under the Receiver Operating Characteristic curve. Measures the model's ability to discriminate between classes across all thresholds. | General-purpose classification evaluation for balanced datasets where both classes are of equal interest.  |
| **PR AUC** | Classification | Area Under the Precision-Recall curve. Measures the trade-off between precision and recall across all thresholds. | Evaluating classifiers on imbalanced datasets where the positive class is rare and of primary interest.  |
| **Expected Calibration Error (ECE)** | Classification | Measures the difference between a model's predicted probabilities (confidence) and the actual observed frequencies. | When the trustworthiness of the predicted probability scores is critical for decision-making (e.g., risk assessment, expected value calculations).  |
| **Brier Score / Log Loss** | Classification | Proper scoring rules that measure the accuracy of probabilistic predictions, penalizing miscalibrated probabilities. | Rigorous evaluation of a model's calibration. Log Loss is more sensitive to highly confident wrong predictions.  |
| **Mean Reciprocal Rank (MRR)** | Ranking/Retrieval | The average of the reciprocal ranks of the first relevant item for a set of queries. | When user success is defined by finding the *first* correct answer quickly (e.g., navigational search, question-answering systems).  |
| **Mean Average Precision (MAP)** | Ranking/Retrieval | The mean of the Average Precision scores over a set of queries. AP rewards ranking many relevant items high up. | When a user wants to see a set of relevant results and the density of relevant items at the top of the list is important (e.g., document search).  |
| **Normalized Discounted Cumulative Gain (NDCG)** | Ranking/Retrieval | Measures ranking quality by considering graded relevance and applying a logarithmic discount to items lower in the list. | When items have varying degrees of relevance and the position of highly relevant items is most important (e.g., e-commerce, media recommendations).  |
| **Mean Absolute Scaled Error (MASE)** | Time Series Forecasting | The MAE of the forecast scaled by the in-sample MAE of a naive (e.g., seasonal) forecast. | Comparing forecast accuracy across multiple time series with different scales; essential for intermittent demand forecasting.  |

* "How does the differentiability of these two loss functions impact the model training process, particularly when using gradient-based optimization?" (Probes understanding of optimization dynamics: MSE's smooth gradient vs. MAE's constant gradient and non-differentiability at zero ).

* "If our target variable has a highly skewed distribution, say for insurance claim amounts, how would that influence your choice between a model trained on MSE versus MAE?" (Probes the connection between the target distribution's mean/median and the desired model output ).

* A junior data scientist on your team states, 'We should always use MAE because it's more robust to outliers.' How do you respond? Go beyond the topic of outliers and explain the fundamental statistical quantity each loss function is trying to estimate. In what business scenario would you strongly advocate for MSE despite the presence of outliers?

    - provide a concrete business scenario for advocating MSE
        - Imagine we are building a model to forecast daily revenue for an e-commerce platform. The dataset will naturally have outliers—days like Black Friday or Prime Day where sales are 10x the normal volume. These are not noise; they are critical business events. If we use MAE, the model will learn to predict the *median* or 'typical' day's revenue, effectively ignoring these massive sales events. This would lead to systematically under-forecasting revenue and could cause disastrous inventory and staffing decisions. In this case, we *want* the outliers to disproportionately influence the model. MSE forces the model to learn the conditional mean, which accounts for these high-revenue days, providing a much more accurate forecast for total revenue over a period. The business objective is to predict the *expected value*, not the *typical value*, making MSE the correct choice."

    - **Stopping at Outliers:** The most common mistake is to only discuss outlier robustness without connecting it to the underlying statistical estimators (mean vs. median).
    - **Confusing RMSE and MSE:** While related, RMSE is an evaluation metric in the same units as the target, while MSE is the loss function often directly minimized. The core property of optimizing for the conditional mean applies to both.
    - **Believing MAE is always superior with noisy data:** Failing to recognize that sometimes "noise" is actually high-impact, business-critical information.

* "You are building two models. Model A predicts the delivery time for a food delivery service, which will be displayed to customers. Model B predicts the necessary inventory level for a critical, life-saving medical device in a hospital. Which loss function (MSE, MAE, Huber) would you primarily use to train each model, and why? Discuss the business implications of your choice in both cases."
    - asymmetric costs of prediction errors in each scenario.
    - Model A (Food Delivery Time): MAE or possibly Huber Loss.
        - The goal is to provide a customer with a reliable estimate of a *typical* delivery time. The business is harmed by being consistently wrong, but the cost of errors is relatively linear. Being 5 minutes late is bad, being 10 minutes late is roughly twice as bad. However, the system is subject to extreme outliers (e.g., a delivery caught in a sudden, massive traffic jam is 60 minutes late). We do not want this single rare event to dramatically skew the delivery estimates for all other customers.
        - Using MAE optimizes for the conditional median, providing a robust estimate that represents the 50th percentile of delivery times. This manages customer expectations effectively for the majority of orders. Using MSE would cause the model to predict longer delivery times for everyone to account for rare, extreme delays, making the service appear slower and less competitive for the average user. Huber loss could be a good compromise, treating small, common variations in delivery time quadratically while capping the penalty for extreme outliers linearly, making it less sensitive than MSE.
    - Model B (Medical Device Inventory): strongly advocate for MSE.
        - The cost of error is highly asymmetric and non-linear. The consequence of a small under-prediction (being short one device) is far more severe than a small over-prediction (having one extra device in stock). A large under-prediction (being short ten devices) is catastrophically worse than a small one. This is a scenario where large errors *must* be heavily penalized.
        - Using MSE directly addresses this business need. The squared error term means that an error of 10 units is penalized 100 times more than an error of 1 unit. This forces the model to be extremely conservative about under-prediction and to pay significant attention to any historical data points that suggested a spike in demand (the outliers). The model will learn to predict the conditional mean, which will be higher than the median if the demand distribution is skewed by rare high-demand events. This leads to higher safety stock but dramatically reduces the risk of a stockout, which in this context is the overriding business priority. Using MAE would be dangerously inappropriate, as it would predict the 'typical' demand and ignore the risk of catastrophic, high-demand events.

    - Huber loss as a sophisticated tool for balancing the properties of MSE and MAE.

    - MSE for "important things" -- mathematical link between the squared penalty and the non-linear cost of large errors.

    - For the Huber loss in the delivery time model, how would you go about tuning the delta hyperparameter? What trade-off are you managing?"
        - delta defines the threshold between treating an error as 'normal' vs. an 'outlier'

    - In the medical device scenario, what if the cost of overstocking is also very high (e.g., the device is perishable and expensive)? How might that change your approach? Would you still use MSE, or would you consider a different loss function entirely?
        - quantile loss
        - custom asymmetric loss functions

* After retraining your model, you observe that the Mean Absolute Error (MAE) has decreased by 10%, but the Root Mean Squared Error (RMSE) has increased by 5%. What does this tell you about the change in the model's prediction error distribution? Is the new model definitively better or worse? How would you investigate further?

    - The candidate must first explain the relationship between the two metrics. RMSE, by squaring errors before averaging, gives disproportionately more weight to large errors than MAE does.31 A key property is that for any set of errors, RMSE $\ge$ MAE The gap between RMSE and MAE is an indicator of the variance in the error distribution, and specifically the presence of large-magnitude errors.

    - A decrease in MAE suggests that for the *majority* of predictions, the model has improved. The "average" or "typical" error is smaller. The model is likely more accurate for the bulk of the data points that have small-to-moderate errors.

    - An increase in RMSE, despite the lower MAE, is a strong signal that the new model is making a few, very large errors. The improvements on the many small errors are being outweighed by the squared penalty on a few new, egregious mistakes. The tail of the error distribution has likely become heavier.

    - Is the Model Better or Worse? The candidate should state that it is not definitively better or worse—it depends entirely on the business context.

    - If the application is one where the average performance matters most and occasional large errors are tolerable (like the food delivery example), the new model might be considered an improvement due to the lower MAE.

    - If the application is one where large errors are unacceptable or catastrophic (like the medical device inventory example), the new model is significantly worse, as the increase in RMSE indicates a higher risk of extreme failure cases.

    - A senior candidate should propose a clear, systematic investigation plan:

        1. **Error Distribution Analysis:** Plot histograms of the prediction errors for both the old and new models. This will visually confirm if the new model's error distribution is more peaked around zero but has fatter tails.
        2. **Residual Analysis:** Examine the specific data points where the new model is making large errors. Are these points outliers in the feature space? Do they belong to a specific subgroup (e.g., a particular customer segment, a specific geographic region)? This helps identify where the model is failing.
        3. **Quantile Error Analysis:** Instead of just looking at the mean error, analyze the errors at different quantiles (e.g., 95th, 99th percentile). This will quantify the magnitude of the worst-case errors and confirm if they have indeed increased.
        4. **Business Impact Simulation:** If possible, run a simulation to estimate the financial or operational cost of the errors produced by each model. This translates the statistical metrics into a business-centric comparison, providing the ultimate answer as to which model is "better."

* After retraining your model, you observe that the Mean Absolute Error (MAE) has decreased by 10%, but the Root Mean Squared Error (RMSE) has increased by 5%. What does this tell you about the change in the model's prediction error distribution? Is the new model definitively better or worse? How would you investigate further? Could this pattern (lower MAE, higher RMSE) ever be a desirable outcome of a model change? If so, in what scenario?" (A creative thinking question. 
    - Perhaps in fraud detection, a model that is much better on non-fraud cases but flags one legitimate high-value customer as fraud could exhibit this pattern. The business might accept this if the overall fraud capture rate improves).

* A colleague presents a Receiver Operating Characteristic (ROC) curve with an Area Under the Curve (AUC) of 0.95 for a fraud detection model, where fraud instances are 0.1% of the data. They claim the model is excellent. Why might this be a dangerously optimistic assessment? Sketch or describe the likely shape of the corresponding Precision-Recall (PR) curve for this model and explain what it would reveal that the ROC curve hides. What is the fundamental mathematical reason for this discrepancy?

    - Part 1: The Flaw in ROC AUC for Imbalanced Data. An ROC AUC of 0.95 seems impressive, but it can be highly misleading on a dataset with extreme class imbalance, like 0.1% fraud. The ROC curve plots the True Positive Rate (TPR, or Recall) against the False Positive Rate (FPR)
        - TPR/Recall: $\frac{tp}{tp+fn}$ (The fraction of actual positives correctly identified)
        - FPR: $\frac{fp}{tp+fn}$ (The fraction of actual negatives incorrectly identified)

    - The problem lies with the FPR. In our case, the number of True Negatives (TN) is massive (99.9% of the data). A model can generate thousands of false positives (FP) without making a meaningful dent in the FPR. For example, if we have 1 million transactions, 1,000 are fraud (P) and 999,000 are not (N). If the model creates 1,000 false positives, the FPR is only $\frac{1,000}{999,000} \approx 0.1\%$. The ROC curve would barely move from the y-axis. The model could be creating a huge operational burden of false alarms, but the ROC AUC would still look fantastic because the vast number of true negatives in the denominator swamps the impact of the false positives."

    - The Precision-Recall (PR) curve provides a much more informative picture. It plots Precision against Recall (TPR)
        - Precision is: $\frac{tp}{tp+fp}$ (fraction of positive predictions that are correct)

    - Unlike FPR, Precision's denominator contains the number of False Positives (FP). This means Precision is directly sensitive to the number of false alarms.

    - For a model with a 0.95 ROC AUC on this dataset, the PR curve would likely start with high precision at very low recall (perhaps it correctly identifies a few very obvious fraud cases with high confidence). However, as we lower the threshold to increase recall and find more of the 1,000 fraud cases, the model will inevitably start making many more false positive predictions. Because the base rate of fraud is so low, even a good model will quickly see its precision plummet. The PR curve would likely show a sharp drop-off, looking much less impressive than the ROC curve. It might reveal that to achieve a recall of 50% (finding 500 fraud cases), our precision drops to 5%, meaning 95% of the alerts are false alarms. This is a critical insight for the fraud investigation team that the ROC curve completely hides.

    - The fundamental difference is the denominator. The ROC curve's x-axis (FPR) is normalized by the number of true negatives, while the PR curve's key metric (Precision) is normalized by the number of predicted positives. When the number of true negatives is enormous and the number of true positives is tiny, the ROC curve is insensitive to the trade-off that often matters most in business: the number of false alarms (FP) you must tolerate to capture a certain number of true positives (TP). The PR curve directly visualizes this trade-off, making it the superior tool for evaluating models on imbalanced data

* What is the baseline for a PR curve? How does it differ from the baseline for a ROC curve, and what does this tell you?
    - ROC baseline is the diagonal line at AUC=0.5
    - PR baseline is a horizontal line at the prevalence of the positive class --> harder to achieve good performance on a PR curve for rare events

* Is there ever a scenario with imbalanced data where you might still prefer to use ROC AUC?
    - e.g., both classes are of truly equal interest and the cost of a false positive and false negative are symmetric, even if the classes are imbalanced, ROC might be defensible.
    - rare in practice

* You have two models for predicting customer churn. Model A has a ROC AUC of 0.85 and an Expected Calibration Error (ECE) of 0.15. Model B has a ROC AUC of 0.82 and an ECE of 0.02. The business wants to use the model's output to offer targeted discounts, with the size of the discount proportional to the predicted churn probability. Which model would you recommend and why? 

    - Explain the concept of model calibration and the meaning of ECE to justify your choice.

    - Model calibration refers to how well a model's predicted probabilities align with the real-world frequencies of events. A perfectly calibrated model that predicts a 70% chance of churn for a group of customers will be correct for, on average, 70% of those customers.

    - ROC AUC, on the other hand, only measures discrimination—a model's ability to give a higher score to a churner than to a non-churner. It tells us about the ranking of predictions, but it tells us nothing about whether the probability values themselves are meaningful

    - A model can have a perfect ROC AUC of 1.0 while being terribly miscalibrated—for example, predicting 99% for all churners and 98% for all non-churners. The ranking is perfect, but the probabilities are useless.

    - Expected Calibration Error, or ECE, is a metric that quantifies this miscalibration. It works by grouping predictions into bins based on their confidence scores. For example, all predictions between 80-90% confidence go into one bin. Within each bin, it compares the average predicted probability (the 'confidence') to the actual fraction of positive cases (the 'accuracy'). The ECE is the weighted average of these differences across all bins. A lower ECE is better, with a perfectly calibrated model having an ECE of 0.

    - for this specific business application, calibration is more important than a small difference in discriminative power --> Model B.

        - **Model A** has slightly better ranking ability (AUC 0.85 vs 0.82), meaning it's marginally better at ordering customers from least to most likely to churn. However, its high ECE of 0.15 indicates it is poorly calibrated. A customer it predicts has an 80% churn risk might in reality only churn 65% of the time. If we base our discount strategy on this inflated 80% figure, we will be overspending on incentives and making decisions based on flawed financial calculations.
        
        - **Model B**, despite its slightly lower AUC, is extremely well-calibrated with an ECE of 0.02. When it predicts an 80% churn risk, we can be confident that the true churn rate for that group of customers is very close to 80%.

        - **Model B is the clear choice**. Its probability outputs can be used directly to drive decision-making. We can perform reliable expected value calculations, for instance: Expected_Profit = (1 - P_churn) * Margin - P_churn * Discount_Cost. This kind of quantitative, automated decision-making is only possible with a well-calibrated model. The small loss in ranking performance is a price well worth paying for trustworthy probabilities.

* What are some common reasons why a modern machine learning model, like a deep neural network or a gradient-boosted tree, might be poorly calibrated even if it has high accuracy and AUC?

    - certain optimization objectives, like minimizing cross-entropy, and model architectures can lead to over-confident predictions

* Besides ECE, what are some other ways to measure or visualize model calibration?

    - reliability diagrams (calibration plots)

    - proper scoring rules like Brier Score and log-loss

* You are building a forecasting system for a large retailer to predict daily sales for thousands of different products. Why would using Mean Absolute Error (MAE) as your primary metric to compare forecast accuracy across all products be flawed? A colleague suggests using Mean Absolute Percentage Error (MAPE) instead. What critical flaw does MAPE have, especially for slow-moving or niche products?

    - explain the distinct problems of scale-dependency for MAE and division-by-zero for MAPE.

    - The Flaw in MAE: Using MAE as the primary metric to compare forecast accuracy across thousands of different products is fundamentally flawed because MAE is scale-dependent. An MAE of 10 is excellent for a product that sells, on average, 10,000 units per day (an error of 0.1%). However, an MAE of 10 is terrible for a product that sells only 5 units per day (an error of 200%). If we were to average the MAE across all products, the metric would be completely dominated by high-volume products like milk and bread. The forecast accuracy for low-volume but high-margin products, like specialty cheeses or imported goods, would be completely invisible in the aggregate metric. We would have no way of knowing if our model was performing well on these items. We cannot meaningfully compare an error in 'number of cars sold' to an error in 'number of air fresheners sold' using a scale-dependent metric.

    - The Critical Flaw in MAPE: "The colleague's suggestion to use MAPE is a good first thought, as it is a percentage error and therefore scale-independent. 
    
        - The formula is $MAPE = \frac{1}{n} \sum_{t=1}^{n} \left| \frac{y_t - \hat{y}_t}{y_t} \right|$.
        - This allows us to compare the forecast for cars and air fresheners on an equal footing. 
        - However, MAPE has a critical, often fatal, flaw in a retail context: it is **undefined when the actual value () is zero.** Many products, especially slow-moving or niche items, will have days with zero sales. This is known as **intermittent demand**. When a zero-sales day occurs, the denominator in the MAPE calculation becomes zero, resulting in an infinite or undefined value. This makes it impossible to calculate the metric for these products. 
        - Furthermore, even when sales are not zero but are very small (e.g., 1 unit), MAPE can explode. A forecast of 2 units when the actual was 1 results in a 100% error. A forecast of 10 when the actual was 1 results in a 900% error. This extreme sensitivity to small denominators makes MAPE highly unstable and unreliable for products with low or intermittent sales patterns, which are very common in a large retail inventory.

* Besides the zero-value issue, MAPE is sometimes criticized for being asymmetric. What does this mean?
    - Probes understanding that MAPE's penalty for over-forecasting is unbounded, while the penalty for under-forecasting is capped at 100%, which can bias models toward under-forecasting

* "What is sMAPE (Symmetric MAPE), and does it solve problems of MAPE?
    - Problems are zero-value issues and explosion near zero
    - sMAPE, which uses the average of the actual and forecast in the denominator. The candidate should note that it helps with the asymmetry but can still be unstable if both actual and forecast are close to zero

* Explain the Mean Absolute Scaled Error (MASE) as you would to a business stakeholder. What is the 'naive forecast' it uses as a baseline? If a model has a MASE of 0.75, what does that mean in practical terms? Why is this metric particularly well-suited for forecasting problems involving intermittent demand?

    - Imagine we want to grade our new, complex forecasting model. To do that, we need a fair baseline to compare it against. The simplest possible forecast we could make is what we call a 'naive forecast.' For a non-seasonal product, this just means predicting that tomorrow's sales will be the same as today's sales. For a seasonal product, it means predicting that this Tuesday's sales will be the same as last Tuesday's sales. This is our 'common sense' baseline—it takes minimal effort and is a simple but often reasonable guess

    - **Mean Absolute Scaled Error, or MASE**, grades our sophisticated model by comparing its average error to the average error of that simple, naive forecast --> $\frac{\text{MAE of the model}}{\text{MAE of the naive forecast}}$. This gives us a single, powerful number.

* You are building a forecasting system for a large retailer to predict daily sales for thousands of different products. Interpret a MASE of 0.75

    - If MASE is **greater than 1**, it means our fancy model is doing *worse* than the simple naive forecast. It's a clear sign that our model is not adding value.
    
    - If MASE is **less than 1**, it means our model is doing *better* than the naive forecast.

    - So, a MASE of **0.75** is great news. It means that, on average, our model's prediction error is only 75% as large as the error we'd get from just using the simple 'last period's value' forecast. In other words, our model provides a **25% improvement in accuracy over the common-sense baseline**. This number is scale-free, so we can use it to fairly compare the 25% improvement on car sales to the 25% improvement on air freshener sales.

Part 3: Suitability for Intermittent Demand

* MASE is especially powerful for products with intermittent demand—those items that have many days of zero sales—for two key reasons:
    - **It handles zeros without a problem.** Unlike MAPE, which blows up when actual sales are zero, MASE's calculation doesn't involve dividing by the actual sales value at each point in time. As long as there is some variation in the historical sales data, the baseline error will be non-zero, and the metric is well-defined and stable.
    
    - **It provides a meaningful baseline.** For an intermittent product, a naive forecast will often predict zero (if yesterday's sales were zero). MASE evaluates how much better our model is at predicting the *timing* and *magnitude* of the non-zero sales spikes compared to that simple baseline.

    - MASE is the industry standard for large-scale forecasting because it's scale-independent, interpretable, and robust to the real-world problem of intermittent demand, which plagues other metrics like MAPE."

* The definition of the 'naive forecast' in the denominator of MASE is critical. What are the two main types of naive forecasts used for non-seasonal and seasonal data?
    - one-step naive forecast $\hat{y}_{t-1}$
    - the seasonal naive forecast $\hat{y}_{t-m}$

* If the entire historical sales data for a product is a flat line (e.g., 10 units sold every single day), what would happen to the MASE calculation, and what does that imply?
    - denominator—the in-sample MAE of the naive forecast—would be zero, making MASE undefined. This implies that MASE requires some historical variability to establish a meaningful baseline error by which to scale


The following table provides a practical, action-oriented framework for mapping common business goals to a defensible evaluation strategy. It can be used as a tool during interviews to pose realistic scenarios and assess a candidate's ability to reason from business needs to technical choices.

| Business Goal / Scenario | Implied Cost of Errors & User Intent | Primary Metric(s) to Optimize | Rationale / Why it Aligns |
|---|---|---|---|
| **"Capture every potential case of a rare disease from medical scans."** | Extremely high cost of False Negatives (missed cases); low cost of False Positives (extra review). | **Recall**, **PR-AUC** | Ensures no opportunity for early treatment is missed. PR-AUC is used because the positive class is rare, making it a more sensitive measure of performance than ROC-AUC.  |
| **"Ensure emails flagged as spam are almost certainly spam to maintain user trust."** | High cost of False Positives (losing an important email); low cost of False Negatives (seeing a spam email). | **Precision** | Prioritizes the correctness of positive predictions to avoid disrupting the user's workflow and eroding trust in the filter.  |
| **"Forecast total monthly revenue for the company to guide financial planning."** | Large errors are disproportionately bad. Outliers (e.g., holiday sales) are critical signals, not noise. | **MSE** (as loss function), **RMSE** (as metric) | Optimizes for the conditional mean, ensuring that high-impact, high-revenue events are properly weighted in the forecast to predict the correct expected value.  |
| **"Provide customers a reliable, 'typical' wait time for their taxi."** | Errors should be minimized on average. Extreme, rare delays (outliers) are noise and shouldn't skew the estimate for everyone. | **MAE** | Optimizes for the conditional median, providing a robust estimate of the central tendency that is not easily swayed by rare, unrepresentative events.  |
| **"A user asks a smart speaker for the capital of Australia."** | User wants the single correct answer immediately. Success is finding the first relevant result. | **MRR** | Directly measures the rank of the first correct answer, perfectly aligning with the navigational, known-item seeking intent of the user.  |
| **"Recommend a set of products on an e-commerce site for a user browsing 'laptops'."** | User wants a well-ordered list with the best matches first. Some laptops are better matches than others (graded relevance). | **NDCG** | Accounts for both the graded relevance of items (e.g., perfect match vs. good alternative) and the user's diminishing attention by discounting items lower in the list.  |
| **"Compare forecast performance for thousands of products, from fast-movers to items that sell once a month."** | Need a scale-free metric that is robust to zero-sales periods (intermittent demand). | **MASE** | Scales error relative to a simple, common-sense baseline (naive forecast), allowing for fair comparison across series of different scales and handling zero-value actuals gracefully.  |
| **"Decide how large a marketing discount to offer customers based on their predicted probability of churning."** | The action taken is proportional to the predicted probability. The probability score must be trustworthy and reflect real-world frequencies. | **ECE**, **Brier Score** (in addition to a discrimination metric like AUC) | Measures model calibration, ensuring that a predicted probability of X% corresponds to an actual event frequency of X%. This is essential for any application involving expected value calculations or direct use of probability scores.  |


## References

6. Classification: Accuracy, recall, precision, and related metrics | Machine Learning, accessed October 14, 2025, [https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall)

7. Precision vs. Recall in Cross-Sell Models - growth-onomics, accessed October 14, 2025, [https://growth-onomics.com/precision-vs-recall-in-cross-sell-models/](https://growth-onomics.com/precision-vs-recall-in-cross-sell-models/)

8. How to use classification threshold to balance precision and recall - Evidently AI, accessed October 14, 2025, [https://www.evidentlyai.com/classification-metrics/classification-threshold](https://www.evidentlyai.com/classification-metrics/classification-threshold)

11. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets - PMC, accessed October 14, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC4349800/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4349800/)

12. ROC and precision-recall with imbalanced datasets, accessed October 14, 2025, [https://classeval.wordpress.com/simulation-analysis/roc-and-precision-recall-with-imbalanced-datasets/](https://classeval.wordpress.com/simulation-analysis/roc-and-precision-recall-with-imbalanced-datasets/)

13. A gentle introduction and visual exploration of calibration and the expected calibration error (ECE) - arXiv, accessed October 14, 2025, [https://arxiv.org/html/2501.19047v2](https://arxiv.org/html/2501.19047v2)

15. Log Loss vs. Brier Score: A Comparison of Evaluation Metrics in Predictive Modeling | Glasp, accessed October 14, 2025, [https://glasp.co/hatch/M90PZowix3cxKZ1T1t47TbFsgUj1/p/Rs39X28xzW2L3cc0k2eP](https://glasp.co/hatch/M90PZowix3cxKZ1T1t47TbFsgUj1/p/Rs39X28xzW2L3cc0k2eP)

17. Performance Metrics for the Comparative Analysis of Clinical Risk Prediction Models Employing Machine Learning, accessed October 14, 2025, [https://www.ahajournals.org/doi/pdf/10.1161/CIRCOUTCOMES.120.007526](https://www.ahajournals.org/doi/pdf/10.1161/CIRCOUTCOMES.120.007526)

18. Evaluating recommendation systems (mAP, MMR, NDCG) | Shaped Blog, accessed October 14, 2025, [https://www.shaped.ai/blog/evaluating-recommendation-systems-map-mmr-ndcg](https://www.shaped.ai/blog/evaluating-recommendation-systems-map-mmr-ndcg)

19. MRR - Shaped Docs, accessed October 14, 2025, [https://docs.shaped.ai/docs/metrics/mrr/](https://docs.shaped.ai/docs/metrics/mrr/)

20. A complete guide to ranking and recommendations metrics - Evidently AI, accessed October 14, 2025, [https://www.evidentlyai.com/ranking-metrics](https://www.evidentlyai.com/ranking-metrics)

22. Normalized Discounted Cumulative Gain (NDCG) explained - Evidently AI, accessed October 14, 2025, [https://www.evidentlyai.com/ranking-metrics/ndcg-metric](https://www.evidentlyai.com/ranking-metrics/ndcg-metric)

23. Mean absolute scaled error - Wikipedia, accessed October 14, 2025, [https://en.wikipedia.org/wiki/Mean_absolute_scaled_error](https://en.wikipedia.org/wiki/Mean_absolute_scaled_error)

24. ANOTHER LOOK AT FORECAST-ACCURACY METRICS FOR INTERMITTENT DEMAND - Rob J Hyndman, accessed October 14, 2025, [https://robjhyndman.com/papers/foresight.pdf](https://robjhyndman.com/papers/foresight.pdf)

25. Mean squared error - Wikipedia, accessed October 14, 2025, [https://en.wikipedia.org/wiki/Mean_squared_error](https://en.wikipedia.org/wiki/Mean_squared_error)

31. Linear regression: Loss | Machine Learning - Google for Developers, accessed October 14, 2025, [https://developers.google.com/machine-learning/crash-course/linear-regression/loss](https://developers.google.com/machine-learning/crash-course/linear-regression/loss)

32. Evaluating Model Accuracy: MSE, RMSE, and MAE in Regression Analysis - CodeSignal, accessed October 14, 2025, [https://codesignal.com/learn/courses/regression-models-for-prediction/lessons/evaluating-model-accuracy-mse-rmse-and-mae-in-regression-analysis](https://codesignal.com/learn/courses/regression-models-for-prediction/lessons/evaluating-model-accuracy-mse-rmse-and-mae-in-regression-analysis)

34. The receiver operating characteristic curve accurately assesses imbalanced datasets - PMC, accessed October 14, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11240176/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11240176/)

41. 3.4 Evaluating forecast accuracy | Forecasting: Principles and Practice (2nd ed) - OTexts, accessed October 14, 2025, [https://otexts.com/fpp2/accuracy.html](https://otexts.com/fpp2/accuracy.html)

44. 5.8 Evaluating point forecast accuracy | Forecasting: Principles and Practice (3rd ed), accessed October 14, 2025, [https://otexts.com/fpp3/accuracy.html](https://otexts.com/fpp3/accuracy.html)

50. MeanAbsoluteScaledError — sktime documentation, accessed October 14, 2025, [https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.performance_metrics.forecasting.MeanAbsoluteScaledError.html](https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.performance_metrics.forecasting.MeanAbsoluteScaledError.html)

* [Scikit-learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
* [Scikit-learn Metrics API](https://scikit-learn.org/stable/api/sklearn.metrics.html)
* [PyTorch Lightning TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/)

## Questions

### How is NLL related to MSE when variance is fixed?
### What does a low RMSE but high NLL imply?
### Why is NLL better than RMSE for uncertainty-aware models?
### What happens as $\delta→0 or \delta→∞?
### Why is Huber loss more robust than MSE but smoother than MAE?
### How would you choose $\delta$ automatically?
### How does CRPS relate to MAE for deterministic forecasts?
### Why is CRPS a “proper scoring rule”?
### What does a low CRPS imply about calibration?
### What’s the effect of $\tau$=0.5 vs $\tau$=0.9?
### How is pinball loss used in prediction interval modeling?
### Why is it asymmetric?
### Why does RMSLE penalize underestimates more than overestimates?
### How do you handle zero or negative values?
### Why is MASE better for comparing across time series?
### What is the intuitive meaning of MASE < 1?
### Why is choice of baseline crucial?
### When would MdAE give a different conclusion than MAE?
### Why is MdAE rarely used as a training loss?
### Why does MSE amplify outliers?
### Show that minimizing MSE gives the conditional mean.
### When would MSE be a poor evaluation metric?
### Why is RMSE reported more often than MSE?
### What does a large difference between MAE and RMSE suggest?
### Why can MAPE explode near zero?
### How might you modify it to handle zeros?
### Why can MAPE be misleading for skewed targets?
### How does sMAPE correct MAPE’s asymmetry?
### What situations still break sMAPE?

### What is the relationship between sharpness and calibration for classifiers? Is there a tradeoff? Will improving one worsen the other?

This depends on the nature of the miscalibration, i.e., whether the model is over or under confident. Overconfidence occurs when the model is too close to the extremes — it predicts something with 99% when it actually happens at 80%. This is symmetric — it is also over confident when it predicts something at 1% when it actually happens at 20%. The ultimate overconfident model would just predict 0s or 1s as probabilities. The opposite problem occurs when the model is underconfident: the ultimate underconfident model might just predict 0.5 (or the global average) for each observation.

If the model is overconfident and too far into the tails, we lose sharpness to improve calibration. If models are under confident and not far enough into the tails, we can improve both calibration and sharpness. In principle, this means you can end up with either a lower or higher quadratic loss (or other loss functions) for finite samples after implementing the calibration methods we discuss below. In practice, we haven’t observed worse performance, in either the quadratic loss or log loss.

Other important losses we consider are accuracy (the proportion of correct classifications) and discrimination based metrics like AUC. These are less affected by calibration because they are only functions of the ordered probabilities and their labels (assuming you change your threshold for accuracy appropriately). We discuss below that we can choose calibration functions which keep accuracy and AUC unchanged.

This implies that if we care about AUC, but calibration also matters for our application, we can take the shortcut of just picking the best model according to AUC and applying a calibration fix on top of it. In fact, this is exactly our situation in the notifications case-study described in a later section.

### How to address calibration in classification tasks?
* See Gneiting (2007). Pick the best performing model amongst models that are approximately calibrated, where "approximately calibrated" is discussed in the next section.

* Fit a calibration function.

* Calibration functions are strictly monotonic, trained on independent data

* Calibration functions do not change the ordering of the predicted probabilities, so they do not change accuracy or AUC.

* Platt scaling (logistic regression on the log-odds scale) is a common choice, but may underfit

* Isotonic regression is another option, not monotonic

* I-splines with positive weights using a proper scoring rule such as log-loss or MSE

* Tensorflow Lattice

### How to check if your model is calibrated?

* Reliability diagram. Observed ~ Predicted probabilities. Visual inspection and hypothesis tests.


### Code a reliability diagram in Python, given a trained classifier and some data.

### What's the relationship between calibration and slices for classification tasks?

### What is a strictly proper scoring rule?
* Defined in Gneiting (2007). 
* scoring rule is a function S(p,y) that measures the quality of a predicted probability distribution p for an outcome y.
* It is proper if the expected score is maximized when p is the true distribution of y. 
* strictly proper if this maximum is unique
* Strictly proper scoring rules are loss functions such that the unique minimizer is the true probability distribution
* Log-loss and quadratic loss are two such examples. 
* ensures that with enough data our calibration function converges to the true probabilities:  $\hat{p} = Pr(Y|\hat{p})$

### Why might you prefer MAE over MSE?
### What does the squaring in MSE imply about the model’s sensitivity to outliers?
### Why do practitioners often report RMSE instead of MSE?
### What does “RMSE is in the same units as the target” practically mean?
### How does it differ from MAE in robustness and interpretability?
### When might MdAE provide a more truthful picture of model performance?
### What are the pitfalls of MAPE near zero-valued targets?
### How does sMAPE address some of these issues?
### Why might these metrics be misleading in skewed distributions or zero-heavy data?
### What makes MASE scale-independent?
### How does it use a naïve forecast as a baseline?
### Why is MASE common in time-series forecasting?
### What intuition motivates taking the log of predictions and targets?
### When is RMSLE preferable to RMSE?
### Why can RMSLE handle multiplicative rather than additive errors?
### What problem does Huber Loss try to solve relative to MSE and MAE?
### How does the $\delta$ parameter affect robustness?
### Where might you use Huber loss in practice (e.g., regression with some outliers)?
### What is the relationship between pinball loss and quantile regression?
### What does a 90th-percentile quantile loss penalize differently than a median loss?
### Why is pinball loss asymmetric?
### How is CRPS related to MAE for deterministic forecasts?
### Why is CRPS a “proper scoring rule”?
### What kind of models require CRPS to evaluate their predictions?
### Why is maximizing log-likelihood equivalent to minimizing NLL?
### How does NLL change if the predicted variance is wrong?
### Why is NLL preferred for probabilistic regression models over RMSE?
### Suppose your model’s MSE improves but MAE worsens — what might that indicate about your residuals?
### Your model has an excellent R² but high RMSE. What does that suggest?
### Why might MAPE be misleading when forecasting electricity demand in winter?
### How would you handle zero or negative values when computing RMSLE?
### If two models have identical RMSE but one has much lower NLL, what does that tell you about its uncertainty calibration?
### In a time-series forecasting setting, why might MASE be a fairer comparison than RMSE across multiple series?
### How could you tune $\delta$ in Huber loss automatically rather than picking it manually?
### Your probabilistic model has low CRPS but poor point forecast accuracy — how might you interpret that?
### When would you use a custom loss that combines RMSE and NLL components?
### How would you choose between pinball loss and CRPS for a probabilistic forecasting competition?
### Explain to a non-technical stakeholder the difference between R² = 0.9 and RMSE = *### Which is more informative?
### If prediction errors have asymmetric business costs (e.g., underprediction worse than overprediction), which loss or metric would you choose?
### In a high-stakes model (e.g., predicting equipment failure time), why might you prioritize a quantile loss or CRPS over RMSE?
### You’re asked to deploy a model with the lowest RMSE, but MASE shows it underperforms a naïve baseline. How do you respond?

## Regression
### Mean Absolute Error (MAE)

$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$

* Interpretation: Average absolute deviation of predictions from true values.
* Pros
   * Robust to outliers (linear penalty).
   * Easy to interpret in same units as target.
* Cons
   * Non-differentiable at zero.
   * Penalizes small and large errors equally.
   * Doesn’t capture direction of errors.
   * Not scale-free — can’t compare across datasets.
### Median Absolute Error (MdAE)

$\text{MdAE} = \text{median}(|y_i - \hat{y}_i|)$

* Interpretation: Median magnitude of errors — very robust summary.

* Pros
   * Immune to outliers.
   * Useful for skewed error distributions.
* Cons
   * Not smooth (harder to optimize).
   * Ignores tail behavior.
### Mean Squared Error (MSE)

$\text{MSE} = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$

* Interpretation: Penalizes squared deviations; emphasizes large errors.
* Pros
   * Differentiable; convenient for optimization.
   * Strong theoretical basis (minimizing → conditional mean).
* Cons
   * Sensitive to outliers.
   * In original squared units.
### Root Mean Squared Error (RMSE)

$\text{RMSE} = \sqrt{\text{MSE}}$
* Interpretation: Standard deviation of residuals; same units as target.
* Pros
   * More interpretable than MSE.
   * Highlights models that consistently make large errors.
* Cons
   * Same sensitivity to outliers as MSE.
   * Not scale-free.
### Mean Absolute Percentage Error (MAPE)

$\text{MAPE} = \frac{100}{n}\sum_i \left| \frac{y_i - \hat{y}_i}{y_i} \right|$

* Interpretation: Average percentage deviation.
* Pros
   * Unitless, interpretable as “% error.”
* Cons
   * Undefined when (y_i = 0).
   * Overweights small denominators.
   * Asymmetric for over- vs under-predictions.

### Symmetric MAPE (sMAPE)

$\text{sMAPE} = \frac{100}{n}\sum_i \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$

* Interpretation: Normalizes by mean magnitude of true and predicted values.

* Pros
   * Bounded between 0–200%.
   * Reduces asymmetry vs MAPE.
* Cons
   * Still unstable near zero.
   * Overly penalizes small values.
### Mean Absolute Scaled Error (MASE)

$\text{MASE} = \frac{\text{MAE(model)}}{\text{MAE(naïve baseline)}}$

* Interpretation: Compares model’s MAE to a baseline (e.g., random walk).
* Pros
   * Scale-independent.
   * Allows comparison across series.
* Cons
   * Needs well-defined baseline.
   * May be misleading for non-stationary data.
### Root Mean Squared Log Error (RMSLE)

$\text{RMSLE} = \sqrt{ \frac{1}{n}\sum_i \left( \log(1+\hat{y}_i) - \log(1+y_i) \right)^2 }$

* Interpretation: Penalizes relative, not absolute, differences.
* Pros
   * Handles skewed data and multiplicative errors.
   * Less punishing for large true values.

* Cons
   * Only for non-negative targets.
   * Sensitive to log(0) handling.
### Pinball (Quantile) Loss

For quantile $\tau$:

$
L_\tau(y, \hat{y}) =
\begin{cases}
\tau (y - \hat{y}) & \text{if } y > \hat{y} \\
(1-\tau)(\hat{y} - y) & \text{otherwise}
\end{cases}
$

* Interpretation: Asymmetric linear loss — encourages quantile-specific predictions.
* Pros
   * Enables quantile regression.
   * Captures uncertainty without full distribution.
* Cons
   * Nondifferentiable at zero.
   * Sensitive to mis-specified quantile.
### Continuous Ranked Probability Score (CRPS)

$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} [F(x) - \mathbf{1}(x \ge y)]^2 dx$

* Interpretation: Measures distance between predicted CDF and observed value.
* Pros
   * Generalizes MAE to probabilistic forecasts.
   * Proper scoring rule — incentivizes calibrated distributions.
* Cons
   * Computationally heavier.
   * Requires full predictive distribution.
### Huber Loss

$L_\delta(y, \hat{y}) =
\begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le \delta \\
\delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{o.w.}
\end{cases}$

* Interpretation: Combines quadratic (small errors) and linear (large errors) regions.
* Pros
   * Differentiable everywhere.
   * Robust to outliers.
* Cons
   * Requires tuning $\delta$.
   * Behavior depends on scaling.

### Log-Likelihood / Negative Log-Likelihood (NLL)

For Gaussian outputs:

$\text{NLL} = \frac{1}{2}\sum_i \left[ \log(2\pi\sigma_i^2) + \frac{(y_i - \mu_i)^2}{\sigma_i^2} \right]$

* Interpretation: Measures both accuracy and predicted uncertainty.
* Pros
   * Natural for probabilistic regression.
   * Penalizes both bias and miscalibration.
* Cons
   * Sensitive to variance estimation.
   * Harder to interpret in raw units.


$$ \text{sMAPE} = \frac{1}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2} $$

* Concordance correlation coefficient

$$ \rho_c = \frac{2 \rho_{xy} \sigma_x \sigma_y}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2} $$

## Ranking and Retrieval
* NDCG (Normalized Discounted Cumulative Gain)

## General Classification
* Precision, Recall, f1, AUROC

    * Handle division-by-zero

    * Binary vs micro vs macro results
    
    * Precision “Of all the things you said were positive, how many were actually positive” tp / (tp + fp)

    * Recall “Of all the actual positives in the data, how many did you capture” tp / (tp + fn)

    * F1: harmonic mean of precision and recall f1 = (2 * precision * recall)/(precision + recall)

* Top-k accuracy – for classification tasks.

* AUROC / AUPRC – imbalanced classification.

* Precision-Recall curve

* Log-loss / Cross-entropy loss

### Calibration

Calibration metrics (ECE, Brier score) – measuring probability reliability.

* Expected quadratic loss

$$ E[(\hat{p} - Y)^2] = E[(\hat{p} - \pi(\hat{p}))^2] + \bar{\pi}(1 - \bar{\pi}) $$

where $\pi(\hat{p})$ is $Pr(Y\vert\hat{p})$ and $\bar{\pi}$ is the base rate $Pr(Y)$.

* sharpness: $-E[(\pi(\hat{p}) - \bar{\pi})^2]$ is how much the predictions vary from the base rate.

* calibration: $E[(\hat{p} - \pi(\hat{p}))^2]$ is how much the predictions vary from the true probabilities.

* Proper scoring rules
* Brier Score

## Language Modeling
* BLEU / ROUGE / METEOR – sequence generation evaluation.
* Perplexity

## Computer Vision
* IoU
* FID and sFID
* mAP (mean average precision) – object detection & ranking.
* CLIPScore
* temporal consistency metrics (?)

## Clustering

## Distance Metrics
* Euclidean Distance
* Manhattan Distance
* Cosine Similarity
* Minkowski Distance
* KL Divergence
* JS Divergence
* Wasserstein Distance
* Shannon Entropy

