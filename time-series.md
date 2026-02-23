# Time Series

## Leveraging Causal Inference to Generate Accurate Forecasts

https://careersatdoordash.com/blog/leveraging-causal-inference-to-generate-accurate-forecasts/

June 14, 2022

Chad Akkoyun and Qiyun Pan

[Local](../readings/Leveraging%20Causal%20Inference%20to%20Generate%20Accurate%20Forecasts.mhtml)

## Increasing Operational Efficiency with Scalable Forecasting

https://careersatdoordash.com/blog/increasing-operational-efficiency-with-scalable-forecasting/

Ryan Schork

August 31, 2021

[Local](../readings/Increasing%20Operational%20Efficiency%20with%20Scalable%20Forecasting.mhtml)


## Stationarity

* Augmented Dickey-Fuller (ADF) Test
  - Null Hypothesis: The time series has a unit root (non-stationary).
  - Alternative Hypothesis: The time series is stationary.
  - p-value < 0.05 indicates rejection of the null hypothesis (stationary).
* Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
* Phillips-Perron (PP) Test

## Auto-Regressive Integrated Moving Average (ARIMA)
Model $y_t$ as a function of its own past values and past errors:
$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + ... + \phi_p y_{t-p} + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q} + \epsilon_t
$$

For $AR(p)$:
$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + ... + \phi_p y_{t-p} + \epsilon_t
$$

For $MA(q)$:
$$
y_t = c + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q} + \epsilon_t
$$

The "I" (Integrated) part involves differencing the time series to make it stationary. For example, first-order differencing is:
$$
y'_t = y_t - y_{t-1}
$$

An $ARIMA(2,1,2)$ model has 2 autoregressive terms, 1 differencing, and 2 moving average terms:
$$
y'_t = c + \phi_1 y'_{t-1} + \phi_2 y'_{t-2} + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \epsilon_t
$$

### Seasonal ARIMA (SARIMA)

### ARIMA with Exogenous Variables (ARIMAX)

## Exponential Smoothing (ETS)

Models that apply exponentially decreasing weights to past observations. The general form is:
$$
S_t = \alpha y_t + (1 - \alpha) S_{t-1}
$$
Where 
* $S_t$ is the smoothed value at time $t$,
* $y_t$ is the actual value, and
* $\alpha$ is the smoothing factor (0 < α < 1).
* At $t=0$, $S_0 = y_0$

## Holt-Winters (HW)
Extends exponential smoothing to capture seasonality and trends. The additive version is:
$$
L_t = \alpha \frac{y_t}{S_{t-m}} + (1 - \alpha)(L_{t-1} + T_{t-1})
$$
$$
T_t = \beta (L_t - L_{t-1}) + (1 - \beta) T_{t-1}
$$
$$
S_t = \gamma \frac{y_t}{L_t} + (1 - \gamma) S_{t-m}
$$
Where
* $L_t$ is the level,
* $T_t$ is the trend,
* $S_t$ is the seasonal component,
* $m$ is the seasonal period, and
* $\alpha$, $\beta$, and $\gamma$ are smoothing parameters

## Vector Autoregressive (VAR) and Extensions
### Vector Autoregressive Moving Average (VARMA)
### Vector Autoregressive Integrated Moving Average (VARIMA)
### Seasonal Vector Autoregressive (SVAR)
### Seasonal Vector Autoregressive Moving Average (SVMA)

## Gaussian Mixure Models (GMM)
## State Space Models (SSM)
## Bayesian Structural Time Series (BSTS)

## Modeling Libraries

* statsmodels
* prophet 
* pmdarima
* darts
* sktime
* tbats
* nixtla

## Metrics
* Mean Absolute Error (MAE)
* Mean Absolute Percentage Error (MAPE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Scaled Error (MASE)
* Symmetric Mean Absolute Percentage Error (sMAPE)
* Continuous Ranked Probability Score (CRPS)
* Prediction Interval Coverage Probability (PICP)
* Weighted Interval Score (Bracher et al., 2021)

## Forecasting with LLMs

### From News to Forecast Paper

*From News to Forecast: Integrating Event Analysis in LLM-Based Time Series Forecasting with Reflection*

[Wang et al (2024)](../readings/2409.17515v3.pdf)

## Time Series Foundation Models

* How is tokenization done for time series data?

* How is context length handled for time series with varying lengths?

* How does *patching* work for time series foundation models?

* How could you handle temporal or seasonal effects in time series foundation models?

* How could you handle irregular time series in time series foundation models?

### Case Studies

https://assets.amazon.science/90/c3/ba96f02341fc88b79f5585c48679/a-flexible-forecasting-stack.pdf


### References

* https://arxiv.org/abs/2506.03128

* https://arxiv.org/abs/2502.12944v1

* https://arxiv.org/abs/2310.10688

* https://arxiv.org/abs/2411.15743

* https://arxiv.org/abs/2402.07570

* https://arxiv.org/abs/2412.20810

* https://arxiv.org/abs/2412.05244

* https://arxiv.org/abs/2503.04118v1

* https://arxiv.org/abs/2503.07649

* https://arxiv.org/abs/2510.00742

* https://arxiv.org/abs/2310.07820

## Deep Learning for Time Series Forecasting

See [Overview from Amazon Science](./fcst-tutorial-ecml_2023.pdf)

* Recurrent Neural Networks (RNNs)
* Long Short-Term Memory (LSTMs)
* Gated Recurrent Units (GRUs)
* Convolutional Neural Networks (CNNs)
* Transformer Models
* Time-series Transformers
* DeepAR

### Case Studies

* https://www.uber.com/en-BE/blog/deepeta-how-uber-predicts-arrival-times/

* https://www.uber.com/en-BE/blog/orbit
    * https://www.youtube.com/watch?v=LXDpq_iwcWY
    * https://github.com/uber/orbit
    * https://arxiv.org/abs/2004.08492

* https://www.uber.com/en-BE/blog/neural-networks-uncertainty-estimation/?uclick_id=92964421-3bf2-497e-989a-b5436cadc193
* https://www.uber.com/en-BE/blog/neural-networks/?uclick_id=92964421-3bf2-497e-989a-b5436cadc193

## Interpretability



## Uncertainty

* Xu and Xie, [Conformal prediction for time series](../readings/2010.09107v15.pdf)


## Causality

### Causal Discovery

Identification of causal relationships in observational time series data.

### Causal Inference from Observational Data

Estimation of the causal effect of an event in the absence of randomized experiments.

### Causal Effect Estimation from Experimental Data

Estimation of the causal effect of a treatment or intervention on an outcome of interest

### Using Causal Models for Forecasting

* ARIMA with Exogenous Variables (ARIMAX)

### Libraries
* https://google.github.io/CausalImpact/CausalImpact.html
* https://github.com/lcastri/causalflow
* https://github.com/salesforce/causalai (archive Mar 1, 2025)
* https://github.com/Sanofi-Public/CImpact

## Time Series Anomaly Detection

* Twitter's AnomalyDetection
* [LinkedIn's Luminol](https://github.com/linkedin/luminol)
* [Netflix's RAD ("Robust Anomaly Detection")](https://netflixtechblog.com/rad-outlier-detection-on-big-data-d6b0494371cc)