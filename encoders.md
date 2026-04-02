# Encoders

https://huggingface.co/transformers/v3.0.2/multilingual.html

## BERT

### RoBERTa
RoBERTa improves BERT with new pretraining objectives, demonstrating BERT was undertrained and training design is important. The pretraining objectives include dynamic masking, sentence packing, larger batches and a byte-level BPE tokenizer.

#### DistilBERT
DistilBERT is pretrained by knowledge distillation to create a smaller model with faster inference and requires less compute to train. Through a triple loss objective during pretraining, language modeling loss, distillation loss, cosine-distance loss, DistilBERT demonstrates similar performance to a larger transformer language model.
### ALBERT

### ColBERT

ColBERT (Contextualized Late Interaction over BERT) is a retrieval model designed to improve the balance between accuracy and efficiency in tasks like document retrieval or question answering. Unlike standard bi-encoder approaches, which encode queries and documents into fixed-dimensional vectors and compute a single similarity score, ColBERT introduces a “late interaction” mechanism. This allows the model to compare individual token-level embeddings from the query and document, capturing finer-grained semantic relationships. The key difference lies in how interactions between queries and documents are handled: bi-encoders compute similarity once after encoding, while ColBERT delays interaction until after encoding, enabling more nuanced comparisons without excessive computational cost.

Standard bi-encoders, such as those using BERT, process queries and documents independently. For example, a query like “How to bake a cake” and a document titled “Easy dessert recipes” would each be converted into a single vector. The similarity between these vectors (e.g., via dot product) determines relevance. While efficient—since document embeddings can be precomputed—this approach risks losing context. Specific terms like “bake” or “cake” might not align well if the document focuses on “oven temperatures” or “frosting techniques” but uses different phrasing. Bi-encoders rely on the encoder to compress all semantic information into one vector, which can oversimplify complex relationships. ColBERT addresses this by generating multiple embeddings per token. Each token in the query (e.g., “bake,” “cake”) is compared to every token in the document, and the model aggregates the maximum similarities across these token pairs. This allows ColBERT to recognize that “bake” aligns with “oven” and “cake” with “recipe,” even if the document doesn’t use the exact query terms.

ColBERT maintains efficiency by leveraging precomputed document token embeddings, similar to bi-encoders, but processes queries dynamically. For instance, during retrieval, a document’s token embeddings are stored in advance, while the query’s embeddings are generated on the fly. The late interaction step—calculating per-token similarities—adds computational overhead compared to bi-encoders but remains far cheaper than cross-encoders (which process query-document pairs jointly). This makes ColBERT suitable for large-scale applications where bi-encoders might miss nuanced matches. For example, a search for “car maintenance” could retrieve a document discussing “automobile care” by matching “car” to “automobile” and “maintenance” to “care” at the token level, even if the overall document vector isn’t close to the query vector in a bi-encoder setup. By balancing granularity and scalability, ColBERT offers a middle ground for developers needing higher accuracy without sacrificing precomputation benefits.

### SBERT (a.k.a. Sentence Transformers)
[SentenceTransformers 🤗](https://huggingface.co/sentence-transformers) and [Documentation](https://www.sbert.net/)

Sentence Transformers (a.k.a. SBERT) is the go-to Python module for accessing, using, and training state-of-the-art embedding and reranker models. It can be used to compute embeddings using Sentence Transformer models, to calculate similarity scores using Cross-Encoder (a.k.a. reranker) models, or to generate sparse embeddings using Sparse Encoder models. This unlocks a wide range of applications, including semantic search, semantic textual similarity, and paraphrase mining.

### Usage

```python
from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])
```

## Multilingual Encoders

### XLM
XLM demonstrates cross-lingual pretraining with two approaches, unsupervised training on a single language and supervised training on more than one language with a cross-lingual language model objective. The XLM model supports the causal language modeling objective, masked language modeling, and translation language modeling (an extension of the BERT) masked language modeling objective to multiple language inputs).

### `bert-base-multilingual-cased` and `bert-base-multilingual-uncased`

## Fine-Tuning
