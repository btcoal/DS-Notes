Alright — here’s what I’d expect you to get on **diffusion models** at Runway / Pika / YouTube / Netflix, broken into the same three tiers I used for VAEs:

---

## **1. Core Theory & Math** (whiteboard-level)

They’ll test that you understand *where the math comes from* — not just the high-level story.

**Likely questions:**

* Derive the forward noising process:

  $$
  q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t} \, x_{t-1}, \beta_t I)
  $$

  and the closed form for $q(x_t \mid x_0)$.
* Explain the *reverse process* and why we can parameterize it as $\epsilon_\theta(x_t, t)$.
* What’s the role of the noise schedule ($\beta_t$)? How do linear, cosine, and learned schedules differ?
* Why does the diffusion loss often reduce to a simple MSE between predicted and true noise?
* Compare *score-based generative models* and *denoising diffusion probabilistic models (DDPM)* — what’s the connection?
* What’s classifier-free guidance and how is it implemented? Derive the equation.
* Why is diffusion more mode-covering than GANs?

---

## **2. Applied / Modern Usage in Industry**

Expect questions about scaling, hybridization, and adapting diffusion to video/multimodal.

**Likely questions:**

* Why operate in *latent* space instead of pixel space? Trade-offs in fidelity, compute, and training complexity.
* How would you adapt an image diffusion model to generate videos with temporal consistency?
* How do you integrate conditioning signals (text, segmentation maps, reference images) into the U-Net?
* How would you fine-tune a large diffusion model for a new domain with limited data?
* Explain how inpainting/outpainting works in the diffusion framework.
* How does the VAE component interact with the diffusion process in Stable Diffusion?
* For a product with *real-time preview*, how would you design the sampling process to balance speed and quality?

---

## **3. Debugging, Optimization, and Scaling**

These probe your ability to make diffusion models work at production scale.

**Likely questions:**

* Sampling is too slow — list ways to speed it up without killing quality (e.g., DDIM, DPM-Solver, distillation).
* Why might outputs have “washed out” colors? Potential causes at training vs inference time.
* How do you detect and mitigate overfitting in large diffusion models?
* How would you profile and reduce VRAM usage during training and inference?
* How do you scale training to multi-node, multi-GPU while keeping the forward/backward pass numerically stable?
* You see repetitive textures or artifacts in outputs — how do you debug the cause?
* Explain “posterior collapse” analogues in diffusion (when the network ignores conditioning).

---

## **Curveball Research-Engineer Style**

These come up if the interviewer wants to see creative, research-level thinking:

* If you replace Gaussian noise with another distribution in the forward process, what changes?
* Can you train a diffusion model without any KL-regularized autoencoder? When is that a bad idea?
* How would you measure whether the learned denoiser has *calibrated uncertainty estimates*?
* Propose a way to integrate attention-based conditioning for both spatial and temporal guidance in video diffusion.
* Would a fully transformer-based denoiser outperform a U-Net for high-res video? Why/why not?