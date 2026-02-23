# VAEs

If you were interviewing at **Runway ML**, **Pika Labs**, **YouTube**, or **Netflix**—all companies that touch large-scale video/image generation—you could expect VAE-related questions to fall into three broad categories:

---

## **1. Core VAE Theory and Math** (Whiteboard-level)

They’ll want to see that you understand *why* VAEs exist, not just how to code them.

**Likely questions:**

* Derive the **ELBO** starting from the log-likelihood $\log p_\theta(x)$.
* Explain the **reparameterization trick** and why it’s needed.
* What is the **role of the KL term** in the VAE objective?
* How does the KL term interact with the reconstruction term in practice? What happens if you crank β up or down?
* Why does a vanilla VAE often produce blurry reconstructions?
* Compare VAEs vs. GANs vs. diffusion in terms of *mode coverage* and *sample fidelity*.

**Follow-up “research engineer” variant:**

* Show how you’d modify the VAE objective to use a **perceptual loss** instead of pixel MSE.
* Discuss how the choice of latent dimensionality affects both training stability and generation quality.

---

## **2. Applied / Modern VAE Usage in Industry**

Companies like Runway or Pika Labs are more interested in VAEs as *components* inside larger systems (e.g., Latent Diffusion, video pipelines).

**Likely questions:**

* In Stable Diffusion, what does the VAE do? Why not operate in pixel space?
* How would you design a VAE to balance **compression ratio** and **visual fidelity** for a video-generation pipeline?
* How would you adapt a 2D-image VAE to handle video frames while preserving temporal consistency?
* If you want to *edit* an image/video in latent space, how do you ensure edits map back cleanly to pixel space?
* How might you reduce **VAE-induced artifacts** that harm downstream diffusion model outputs?

**Follow-up “production” variant:**

* Given GPU memory constraints, how do you decide the downsampling factor in the VAE encoder?
* How would you fine-tune an existing pretrained VAE for a new domain (e.g., medical, anime) without catastrophic forgetting?

---

## **3. Debugging, Optimization, and Scaling Questions**

They’ll want to know you can *make VAEs work at scale* in production.

**Likely questions:**

* You notice your VAE decoder outputs have color shifts or ringing artifacts. How do you debug?
* Training loss is fine, but reconstructions are poor — list possible causes.
* What’s the trade-off between *KL annealing*, *free bits*, and *β-VAE* in large-scale training?
* How do you efficiently train a VAE on multi-node GPU clusters for 4K video frames?
* How would you profile a VAE to find bottlenecks in encoding/decoding during inference?
* Explain how you’d deploy the VAE in a high-throughput serving pipeline for an interactive video-editing tool.

---

### **Examples of “curveball” interview questions**

These pop up at research-y product companies:

* Can a VAE be used *without* a stochastic latent variable? What would you lose?
* How would you integrate quantization (as in VQ-VAE) into an LDM pipeline, and why might you prefer it?
* How would you measure whether the VAE’s latent space is *good* for downstream generative tasks?
* Given unlimited data, would you still regularize the latent distribution toward a fixed Gaussian? Why/why not?

### **How to prepare for this class of interviews**

1. **Derivations** — Be able to cleanly derive ELBO and reparameterization.
2. **Implementation-level knowledge** — Know PyTorch implementation of a VAE from scratch.
3. **Modern hybrids** — Understand how VAEs plug into diffusion models, video models, etc.
4. **Scaling & optimization** — Have talking points for large-batch, distributed training, mixed precision, etc.
5. **Failure modes** — Be able to reason about artifacts, posterior collapse, and latent space quality.