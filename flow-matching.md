Here’s a clean and precise outline of the **Flow Matching training recipe** from the screenshot, written in Markdown with LaTeX support:

---

## 🌀 Flow Matching – Training Procedure

### **Step 1: Sample from Data and Noise**

Take a training data sample $\mathbf{X}_1$, a noise sample $\mathbf{X}_0 \sim \mathcal{N}(0, I)$, and sample a timestep $t \in [0, 1]$.

### **Step 2: Construct Interpolated Sample**

Construct the interpolated sample $\mathbf{X}_t$ via linear interpolation:

$$
\mathbf{X}_t = t \mathbf{X}_1 + \left(1 - (1 - \sigma_{\min})t\right) \mathbf{X}_0
$$

This defines a path between the noise $\mathbf{X}_0$ and data $\mathbf{X}_1$, with controllable variance via $\sigma_{\min}$.

### **Step 3: Compute Ground Truth Velocity**

Compute the ground truth velocity $\mathbf{v}_t$ as the time derivative of $\mathbf{X}_t$:

$$
\mathbf{v}_t = \frac{d\mathbf{X}_t}{dt} = \mathbf{X}_1 - (1 - \sigma_{\min}) \mathbf{X}_0
$$

This is the velocity field that moves the interpolated sample toward the data sample.

### **Step 4: Train the Model to Predict Velocity**

Train a neural network $u(\mathbf{X}_t, \mathbf{P}, t; \theta)$ to predict the velocity:

$$
\mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1, \mathbf{P}} \left\| u(\mathbf{X}_t, \mathbf{P}, t; \theta) - \mathbf{v}_t \right\|^2
$$

Here:

* $\theta$: model parameters,
* $\mathbf{P}$: optional conditioning (e.g. text prompt),
* The loss is the **mean squared error** between predicted and ground truth velocity.

---

This corresponds to the **Optimal Transport (OT)** path variant of Flow Matching introduced in [Flow Matching for Generative Modeling (Lipman et al., 2022)](https://arxiv.org/abs/2210.02747). The interpolation and velocity match exactly what's presented in Section 4.1 (Equation 23).

Let me know if you want the code or connection to CNFs explained next.
