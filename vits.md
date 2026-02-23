# Vision Transformers (ViTs)

## DINOv3

* Self-supervised learning for vision transformers using knowledge distillation without labels.
* Uses a teacher-student framework where the student learns to match the teacher's output on augmented views of the same image.
* Achieves strong performance on image classification, segmentation, and detection tasks.

***Comparison to CLIP:*** *DINOv3 focuses solely on visual representation learning without text, while CLIP learns joint vision-language embeddings.*

### Architecture
* Gram-Anchoring Loss
    * Encourages the student to produce similar feature correlations as the teacher.
    * Helps learn rich representations without explicit labels.
* Backbone: Vision Transformer (ViT) architecture.
* Multi-crop strategy: Uses multiple crops of different sizes during training to improve robustness.
* No need for negative samples, unlike contrastive learning methods.

### Applications
* Image classification
* Object detection
* Semantic segmentation
* Few-shot learning
* depth estimation


## **1. Core Theory & Math** (whiteboard-level)

* Explain the key difference between CNNs and ViTs in terms of inductive bias.
* How does splitting an image into patches work? Give the formula for flattening & projecting to embeddings.
* Derive the input sequence shape for a ViT given image resolution, patch size, and embedding dimension.
* What is positional encoding? Compare **absolute** vs **relative** positional embeddings.
* Write the equations for scaled dot-product attention and explain why the scaling factor is $\frac{1}{\sqrt{d_k}}$.
* Compare the parameter count and computational complexity of a ViT vs a CNN for the same input size.
* How does the \[CLS] token work in classification? How is this different from average pooling?


## **2. Applied / Modern Usage in Industry**

* How are ViTs used in Stable Diffusion and video diffusion models (e.g., as CLIP text/image encoders)?
* Explain why large-scale pretraining is critical for ViTs compared to CNNs.
* How would you adapt a ViT for video? Discuss **TimeSformer**, **Video Swin**, or temporal attention schemes.
* How do you handle very high-res images without blowing up memory? (e.g., windowed attention, hierarchical ViTs like Swin Transformer)
* How would you fine-tune a pretrained ViT for a small dataset? Pros/cons of freezing early layers.
* What’s the role of attention maps in interpretability, and how might you use them in an editing product?
* How do you integrate multimodal input (image + text) into a transformer backbone?


## **3. Debugging, Optimization, and Scaling**

* Why might a ViT underperform a ResNet on a small dataset?
* How would you improve training stability in a ViT from scratch?
* You hit an out-of-memory error on 4K images — what’s your strategy? (gradient checkpointing, patch size increase, attention downsampling)
* How do you profile ViT inference and find the main bottlenecks?
* How can you speed up ViT inference for interactive applications? (quantization, distillation, pruning, FlashAttention)
* Discuss trade-offs between *global* vs *local* attention for high-res images.
* What are the precision/memory trade-offs when using mixed precision (FP16/BF16) in attention-heavy models?


## **4. Curveball Research-Engineer Style**

* Could you hybridize a ViT and a CNN to get the best of both worlds? Give an example.
* How would you design a ViT specifically optimized for temporal consistency in generative video models?
* What are the pros/cons of replacing fixed patch embedding with a learnable CNN stem?
* How might you design a ViT architecture for **real-time style transfer** in a video editing app?
* Is it possible to make positional embeddings *fully learned* and resolution-agnostic? How?
* If you replace attention with a low-rank approximation, how does that affect model performance and complexity?