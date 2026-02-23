# Vision-Language Models

* Most use a decoder-only LLM (e.g., GPT) as the backbone.
* cross-attention layers connect vision and language modalities.
    * cross-attention considers two sequences of tokens—*the tokens from the encoder and the tokens from the decoder*—and computes attention between these two sequences. 
    * By doing this, we allow the decoder to consider the representations produced by the encoder when generating its output

## CLIP (Contrastive Language-Image Pre-Training)

![CLIP training example](./clip-training-example.jpg)

### Architecture
* Image encoder: Vision Transformer (ViT) or ResNet.
* Text encoder: Transformer-based model (like GPT).
* Both encoders project inputs into a shared embedding space.

### Training via Contrastive Learning
* Learn joint embedding space for images and text by maximizing similarity of matching pairs and minimizing for non-matching pairs.

* Large-scale dataset of image-caption pairs (e.g., 400M pairs).

* Train with batch size of thousands to millions for effective contrastive learning.
More specifically, CLIP is trained using the simple task of classifying the correct caption for an image among a group of candidate captions (i.e., all other captions within a training batch). Practically, this objective is implemented by:

* Passing a group of images and textual captions through their respective encoders (i.e., the ViT for images and the LLM for text).

* Maximizing the cosine similarity between image and text embeddings (obtained from the encoders) of the true image-caption pairs.

* Minimizing the cosine similarity between all other image-caption pairs.

* InfoNCE loss

$$
L = -\log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(v_i, t_j)/\tau)}
$$

where $v_i$ and $t_i$ are image and text embeddings, $\text{sim}$ is cosine similarity, $\tau$ is temperature. See [PyTorch implementation](https://github.com/RElbers/info-nce-pytorch).

### Applications
* Zero-shot image classification by comparing image embeddings to text embeddings of class names.
* Image-text retrieval: find images given text queries and vice versa.

## Benchmarks

* MTEB (Massive Text Embedding Benchmark)

## Fine-Tuning VLMs


## References
* https://lilianweng.github.io/posts/2021-05-31-contrastive/
* ***(Needs updating)*** https://lilianweng.github.io/posts/2022-06-09-vlm/ 
* https://cameronrwolfe.substack.com/p/vision-llms
* https://cameronrwolfe.substack.com/p/using-clip-to-classify-images-without-any-labels-b255bb7205de
* https://magazine.sebastianraschka.com/p/understanding-multimodal-llms