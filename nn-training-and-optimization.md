# Neural Network Training

***Loss Functions, Optimization, and Learning Rate Schedules***

## Activation Functions

**ReLU (Rectified Linear Unit)**: $f(x) = \max(0, x)$
- Pros: Simple, efficient, mitigates vanishing gradient problem.
- Cons: Can suffer from "dying ReLU" problem where neurons become inactive.

**Sigmoid**: $f(x) = \frac{1}{1 + e^{-x}}$
- Pros: Smooth gradient, outputs between 0 and 1 (good for probabilities).
- Cons: Vanishing gradient for large positive/negative inputs, not zero-centered.

**Tanh (Hyperbolic Tangent)**: $f(x) = \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$
- Pros: Zero-centered outputs, stronger gradients than sigmoid.
- Cons: Still suffers from vanishing gradient for large inputs.

**Leaky ReLU**: $f(x) = \max(0.01x, x)$
- Pros: Mitigates dying ReLU problem by allowing a small gradient when inactive.
- Cons: Introduces a small slope in the negative region, which may not always be ideal.

**Swiglu**: $f(x) = x \cdot \text{sigmoid}(x)$
- Pros: Smooth, non-monotonic, combines benefits of ReLU and sigmoid.
- Cons: More computationally expensive than ReLU.

## Loss Functions

### When to use which loss function

| Task Type | Common Loss Functions |
|-----------|-----------------------|
| sequence prediction | Cross-Entropy, NLL |
| binary classification | Binary Cross-Entropy |
| multi-class classification | Softmax Cross-Entropy |
| regression | MSE, MAE, Huber Loss |
| ranking/retrieval | Contrastive Loss, Triplet Loss, InfoNCE |
| segmentation | Dice Loss, Jaccard Loss |
| metric learning | Contrastive Loss, Triplet Loss |
| image generation | Perceptual Loss, GAN Loss |
| language generation | Cross-Entropy, Perplexity |
| video generation | MSE, Perceptual Loss |

### Binary Cross-Entropy

* BCELoss – for binary classification.

$$
L = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$

where $y_i$ is the true label and $p_i$ is the predicted probability.

In PyTorch, use `nn.BCELoss` with sigmoid outputs.

### Negative Log Likelihood (NLL)
* Commonly used with softmax outputs for multi-class classification.
$$
L = -\sum_{i=1}^{N} \log(p_{i, y_i})
$$

where $p_{i, y_i}$ is the predicted probability for the true class $y_i$.

In PyTorch, use `nn.NLLLoss` with log-probabilities from `LogSoftmax`.

### Softmax Cross-Entropy

* Combines softmax activation and cross-entropy loss for multi-class classification.
$$
L = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log\left(\frac{\exp(z_{i,c})}{\sum_{k=1}^{C} \exp(z_{i,k})}\right)
$$

where $z_{i,c}$ is the logit for class $c$ of sample $i$, and $y_{i,c}$ is the one-hot encoded true label.

In PyTorch, use `nn.CrossEntropyLoss`, which combines `LogSoftmax` and `NLLLoss`.

### Contrastive loss – metric learning, siamese networks.

* Contrastive Loss is used to learn embeddings where similar samples are pulled together and dissimilar samples are pushed apart. It is commonly used in siamese networks for tasks like face verification.
$$
L = \frac{1}{2N} \sum_{i=1}^{N} \left[ y_i d_i^2 + (1 - y_i) \max(0, m - d_i)^2 \right]
$$

where $d_i$ is the distance between the pair of samples, $y_i$ is 1 if they are similar and 0 if dissimilar, and $m$ is a margin.

Implemented in PyTorch as a custom loss function, e.g.:
```python
import torch
import torch.nn as nn
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            label * torch.pow(euclidean_distance, 2) +
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive
```

### InfoNCE / NT-Xent loss – foundation of contrastive self-supervised learning. See [Lilian Wang's post](https://lilianweng.github.io/posts/2021-05-31-contrastive/).

* A popular loss function for contrastive learning, particularly in self-supervised learning frameworks like SimCLR. It encourages similar samples to have higher similarity scores while dissimilar samples have lower scores.

$
L = -\log \frac{\exp(\text{sim}(x_i, x_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(x_i, x_k)/\tau)}
$

where $\text{sim}(x_i, x_j)$ is the similarity (e.g., cosine similarity) between samples $x_i$ and $x_j$, $\tau$ is a temperature parameter, and $N$ is the batch size.

### Kullback-Leibler Divergence

* Measures how one probability distribution diverges from a second, expected probability distribution. Commonly used in variational autoencoders and knowledge distillation.
$$
D_{KL}(P || Q) = \sum_{i} P(i) \log\left(\frac{P(i)}{Q(i)}\right)
$$

where $P$ is the true distribution and $Q$ is the predicted distribution.

Note that KL divergence is not symmetric: $D_{KL}(P || Q) \neq D_{KL}(Q || P)$.

In PyTorch, use `nn.KLDivLoss` with log-probabilities.

```python
from torch.nn import KLDivLoss
criterion = KLDivLoss(reduction='batchmean')
```

### Regression Metrics
* Mean Absolute Error (MAE)
$MAE(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$

* Mean Squared Error (MSE)
$MSE(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$

* Huber loss – regression robust to outliers.
$$
L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \leq \delta \\
\delta (|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$

### Focal loss
Focal loss combats class imbalance by focusing on hard examples.

### Dice loss/Jaccard loss

Dice loss/Jaccard loss are for segmentation.

### Triplet loss – enforces relative embedding distances.

Triplet loss enforces relative embedding distances.

## Batch size, learning rate, and epochs

* **Batch Size:** Larger batch sizes can lead to more stable gradient estimates, but may require adjustments to the learning rate. Smaller batch sizes introduce more noise in gradients, which can help escape local minima but may slow convergence.

* **Learning Rate:** A higher learning rate can speed up training but risks overshooting minima, while a lower learning rate provides more precise convergence but may require more epochs.

* **Epochs:** More epochs allow the model to learn more from the data, but excessive epochs can lead to overfitting. The optimal number of epochs depends on the dataset size and complexity.

A common strategy is to start with a moderate batch size and learning rate, then adjust based on validation performance. Techniques like learning rate scheduling and early stopping can help manage the trade-offs between these hyperparameters.

## Optimizers

### Newton's Method, BFGS, LBFGS

The core iterative algorithm is:

$$
\text{While not converged:}
$$
$$
\theta_t \leftarrow \theta_{t-1} - H^{-1} \times \nabla_\theta J(\theta_{t-1})
$$

where $H$ is the Hessian matrix of second derivatives. And convergence is typically defined as when the change in loss or parameters falls below a threshold:
$$
|\theta_t - \theta_{t-1}| < \epsilon \quad \text{or} \quad |J(\theta_t) - J(\theta_{t-1})| < \epsilon
$$

**BFGS (Broyden-Fletcher-Goldfarb-Shanno)** and its **limited-memory variant LBFGS** are *quasi-Newton* methods that approximate the Hessian matrix to perform optimization without computing second derivatives directly. They build up an approximation of the inverse Hessian using gradient evaluations from previous iterations.

### SGD

Stochastic Gradient Descent (SGD) is the most basic optimizer.

$$\theta_t \leftarrow \theta_{t-1} - \alpha \times \nabla_\theta J(\theta_t)$$

### Momentum

**Momentum**: Exponential moving average of gradients (first moment)

$$EMA(g) \equiv m_t = \beta_1 × m_{t-1} + (1-\beta_1) × \nabla_\theta J(\theta_t)$$

$$\theta_t \leftarrow \theta_{t-1} - \alpha \times m_t$$

Helps smooth out noisy gradients and maintain direction.


### AdaGrad (2011)

AdaGrad was designed to solve a key problem: different parameters need different learning rates. Some parameters are updated frequently (like common words in NLP), while others are updated rarely (like rare words).

* Accumulates the squared gradients: 
    $$\nabla J(\theta_t) = \nabla J(\theta_{t-1}) + \nabla J(\theta_{t-1})^2$$
* Updates parameters: 
    $$\theta_t \leftarrow \theta_{t-1} - \frac{\alpha}{\sqrt{\nabla J(\theta_{t-1}) + \varepsilon}} \times \nabla_\theta J(\theta_t)$$

* Parameters with large accumulated gradients get smaller learning rates, while parameters with small accumulated gradients get larger learning rates.

* **The problem:** $G_t$ (the accumulated squared gradients) only grows (never shrinks), so learning rates become vanishingly small over time. Training effectively stops after a while because the denominator becomes too large.

### RMSProp (2012)

RMSProp fixed AdaGrad's main weakness by using an exponential moving average instead of accumulating all past gradients.

* Exponential moving average of squared gradients: 

$v_t = \beta_2 × v_{t-1} + (1-\beta_2) × \nabla_\theta J(\theta_t)^2$

where $v_t$ is the EMA of squared gradients at time step t, and $\beta_2$ (typically 0.9) controls how much history to remember.

* Updates parameters: 
    $$\theta_t \leftarrow \theta_{t-1} - \frac{\alpha}{\sqrt{v_t + \varepsilon}} \times \nabla_\theta J(\theta_t)$$

**Key improvement:** The exponential moving average "forgets" old gradients exponentially, preventing the learning rate from shrinking to zero. The β parameter (typically 0.9) controls how much history to remember.

**What it still lacks:** No momentum term to help navigate through local minima and maintain direction.

### Adam

Adam combines RMSProp's adaptive learning rates with momentum, creating a more robust optimizer.

**What Adam takes from RMSProp:**
- Exponential moving average of squared gradients (second moment)

    $$v_t = \beta_2 × v_{t-1} + (1-\beta_2) × \nabla_\theta J(\theta_t)^2$$

- Adaptive learning rate scaling: dividing by $$\sqrt{v_t}$$

**What Adam adds:**
- **Momentum**
- **Bias correction**: Both moments are corrected for initialization bias

**AdaGrad** said: "Let's give different learning rates to different parameters based on their history"
- Problem: History accumulates forever, killing learning rates

**RMSProp** said: "Let's use recent history instead of all history"
- Solution: Exponential moving average keeps learning rates alive
- Missing: No momentum to maintain direction

**Adam** said: "Let's combine RMSProp's adaptive rates with momentum"
- Takes RMSProp's v_t for adaptive scaling
- Adds momentum term m_t for directional consistency
- Includes bias correction for better early training

### AdamW

AdamW is a variant of Adam that adds **weight decay** (L2 regularization) to the optimizer.

$
\theta_t \leftarrow \theta_{t-1} - \frac{\alpha}{\sqrt{g^2_t + \varepsilon}} \times m_t - \frac{\alpha}{\beta} \times \theta_{t-1}
$

where $g^2_t$ is the EMA of squared gradients, $m_t$ is the EMA of gradients, and $\beta$ is the weight decay coefficient. $\varepsilon$ is a small constant for numerical stability.

### Summary
Below, let $g = \nabla_\theta J(\theta_t)$ be the gradient at time step t.

**AdaGrad**

$\theta_t \leftarrow \theta_{t-1} - \frac{\alpha}{\sqrt{\sum g^2}} \times g$

**RMSProp**

$
\theta_t \leftarrow \theta_{t-1} - \frac{\alpha}{\sqrt{EMA(g^2)}} \times g
$

**Adam**

$
\theta_t \leftarrow \theta_{t-1} - \frac{\alpha}{\sqrt{EMA(g^2)}} \times EMA(g)
$

**AdamW**

$
\theta_t \leftarrow \theta_{t-1} - \frac{\alpha}{\sqrt{EMA(g^2)}} \times EMA(g) - \frac{\alpha}{\beta} \times \theta_{t-1}
$


### Extra bits

Lookahead optimizer
* slow/fast weight updates for stability.

Learning rate warmup
* avoids instability at the start.

## Learning Rate Schedules
**Step decay:** reduce LR by factor every N epochs.

$LR_t = LR_0 \times \text{drop}^{\lfloor t / \text{epochs\_drop} \rfloor}$

**Exponential decay:** multiply LR by constant factor each epoch.
    
$LR_t = LR_0 \times e^{-\lambda t}$

**Linear decay:** decrease LR linearly to zero.
$LR_t = LR_0 \times (1 - \frac{t}{T})$

**Staircase decay:** reduce LR in discrete steps.

$LR_t = LR_0 \times \text{drop}^{\lfloor t / \text{epochs\_drop} \rfloor}$

**Cyclical LR:** vary LR between bounds cyclically.
    
$LR_t = LR_{min} + \frac{1}{2}(LR_{max} - LR_{min})(1 + \sin(\frac{2 \pi t}{T}))$

**Cosine decay schedule:** widely used in vision & transformers.
$LR_t = LR_{min} + \frac{1}{2}(LR_{max} - LR_{min})(1 + \cos(\frac{t \pi}{T}))$

### In PyTorch
Below is a simple logistic regression training loop with an exponential learning rate scheduler:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.datasets import make_classification

# Dummy dataset
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# DataLoader
dataset = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Simple logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = LogisticRegression(input_dim=10)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = ExponentialLR(optimizer, gamma=0.9)  # Decay LR by 10% every epoch

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()  # Update learning rate
    print(f'Epoch {epoch+1}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}')
```

### One-cycle policy

* cyclical LR with momentum scheduling.

## Stability and Regularization

### Gradient clipping

* Avoid exploding gradients.

* Adaptive gradient clipping scales clipping by norm.

### Batch Normalization and Layer Normalization

* Techniques to stabilize and accelerate training by normalizing activations.

### Dropout
* Regularization technique to prevent overfitting by randomly dropping units during training.

### Early Stopping

* Stop training when validation performance degrades to prevent overfitting.


## References

* https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
* https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
* https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
* https://docs.pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
* https://docs.pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss
* https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html#torch.nn.HuberLoss
