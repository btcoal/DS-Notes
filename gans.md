# GANs

Here’s the **GAN interview prep equivalent** to the VAE, Diffusion, and ViT lists we did earlier — tuned for a **Runway / Pika Labs / YouTube / Netflix**–style MLE or research engineer role.

## **1. Core Theory & Math** (whiteboard-level)

They’ll want to see you know the original GAN setup *and* why it evolved.

**Likely questions:**

* Write the original GAN objective from Goodfellow et al. (2014):

  $$
  \min_G \max_D \; \mathbb{E}_{x \sim p_\text{data}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]
  $$

  Explain what $G$ and $D$ are optimizing for.
* Show the connection between GANs and **Jensen–Shannon divergence** minimization.
* Why does the original objective lead to vanishing gradients for $G$? What’s the “non-saturating” trick?
* Compare **WGAN** loss with original GAN loss. Why does WGAN use weight clipping or gradient penalty?
* Explain **mode collapse** — what causes it and how to mitigate it.
* How does a **conditional GAN (cGAN)** differ from an unconditional one? Show the loss change.
* Differences between GANs, VAEs, and diffusion models in terms of likelihood estimation, sample fidelity, and mode coverage.

## **2. Applied / Modern Usage in Industry**

Expect application questions that blend GAN knowledge with production constraints.

**Likely questions:**

* What are GANs still good for in 2025 when diffusion dominates? (e.g., super-resolution, frame interpolation, domain translation).
* How would you integrate a GAN loss into a diffusion or VAE pipeline to sharpen outputs?
* How do you design a GAN for **video**? (temporal discriminators, 3D convolutions, spatio-temporal attention).
* If you needed real-time GAN inference in a browser, how would you approach model compression?
* How would you adapt a pretrained GAN for a new artistic style without retraining from scratch? (style transfer, fine-tuning, adapters).
* How can you leverage GAN discriminators as *feature extractors* for downstream retrieval tasks?

## **3. Debugging, Optimization, and Scaling**

GAN training is notoriously fragile — they’ll test that you can diagnose and fix issues.

**Likely questions:**

* Your GAN outputs look fine in early training but deteriorate after a few epochs — possible causes?
* What are **spectral normalization** and **instance noise**? How do they help stability?
* How do you choose the ratio of $D$ to $G$ updates per step?
* How do you monitor overfitting in a GAN, given there’s no explicit likelihood?
* How do you measure GAN output quality? Pros/cons of FID, IS, and precision/recall metrics.
* How would you parallelize GAN training across multiple GPUs without destabilizing learning?
* What are the trade-offs between using **PatchGAN** vs full-image discriminators?

## **4. Curveball Research-Engineer Style**

These test whether you can go beyond cookbook recipes.

* Can GANs be interpreted as *implicit energy-based models*? How?
* How would you modify the GAN objective for **multi-modal conditioning** (e.g., video generation from both text and reference footage)?
* Could you make a GAN that outputs directly into a latent space (like SD’s VAE space) instead of pixels? Why might that help?
* Would replacing the discriminator with a **contrastive loss** be equivalent? What’s lost?
* What’s the trade-off between adversarial loss and perceptual loss when fine-tuning a generative pipeline?
* Is there a theoretical reason GANs tend to produce sharper textures than VAEs?