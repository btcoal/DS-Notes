# Training Stability & Scaling

## Questions



## Gradient accumulation – effective large batch training on small GPUs.

## Learning Rates

### Warmup
**Learning Rate Warm-up** is a scheduling strategy that gradually increases the learning rate from a low initial value (often zero) to a target peak value during the early stages of training.

* **Pioneering Research**: The practice of linear warmup was popularized by **Goyal et al. (2017)** in the paper [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677). They introduced it to overcome optimization challenges when using large batch sizes.
* **Transformer Standard**: The procedure became a standard component for training Transformer architectures following the landmark paper **Vaswani et al. (2017)**, [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

* **Theoretical Mechanisms**: Recent research, such as the 2024 paper [Why Warmup the Learning Rate? Underlying Mechanisms and Improvements](https://arxiv.org/abs/2406.09405) by Kalra and Barkeshli, explains that warmup's primary benefit is allowing the network to tolerate larger target learning rates by forcing it into "more well-conditioned areas of the loss landscape".

* **Variance Reduction**: Another academic perspective suggests that warmup acts as a variance reduction technique, particularly for adaptive optimizers like **Adam**, by preventing large, unstable weight updates when gradient statistics are still unreliable.

* **Performance Stability**: Studies show that while not using warmup might result in faster initial progress, it often leads to a "permanent gap" in performance or even training divergence that cannot be recovered later.

In PyTorch with Huggingface using the `Trainer` API in `TrainerArguments`, you can enable learning rate warmup by setting the `warmup_steps` or `warmup_ratio` parameters. Here's a brief example:

```python
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',
    ...
    warmup_steps=500,  # Number of warmup steps
    learning_rate=5e-5,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
```

## Mixed precision training – FP16/BF16 for speed/memory.

## LoRA

* Low-Rank Adaptation (LoRA) injects low-rank matrices into transformer layers to adapt pre-trained models with minimal parameter updates.

* **Parameter-Efficient Fine-Tuning (PEFT)** techniques like LoRA and adapters allow fine-tuning with fewer trainable parameters, reducing computational 
costs.

* LoRA makes fine-tuning more efficient by drastically reducing the number of trainable parameters.

* The original pre-trained weights are kept frozen, which means you can have multiple lightweight and portable LoRA models for various downstream tasks built on top of them.

* LoRA is orthogonal to many other parameter-efficient methods and can be combined with many of them.

* Performance of models fine-tuned using LoRA is comparable to the performance of fully fine-tuned models.

* LoRA does not add any inference latency because adapter weights can be merged with the base model.

**NB U:** *While LoRA is significantly smaller and faster to train, you may encounter latency issues during inference due to separately loading the base model and the LoRA model. To eliminate latency, use the [`merge_and_unload()`](https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraModel.merge_and_unload) function to merge the adapter weights with the base model which allows you to effectively use the newly merged model as a standalone model.
*
**NB D:** *When quantizing the base model, e.g. for QLoRA training, consider using the [LoftQ initialization](https://arxiv.org/abs/2310.08659), which has been shown to improve the performance with quantization. The idea is that the LoRA weights are initialized such that the quantization error is minimized. To use this option, do not quantize the base model.*

The short answer is: **primarily training efficiency.**

While LoRA (Low-Rank Adaptation) is designed to make the *training* process significantly faster and less resource-heavy, its impact on *inference* is more about **flexibility and memory management** rather than raw speed optimization.

Here is a breakdown of how LoRA impacts both stages.

### LoRA’s biggest advantage is reducing the "cost of entry" for fine-tuning large models.

* **Memory Savings:** By freezing the original weights () and only training two small low-rank matrices ( and ), you reduce the number of trainable parameters by up to **10,000x**. This allows you to fine-tune a 70B parameter model on consumer-grade hardware that would otherwise require a massive server cluster.
* **Reduced Optimizer State:** In full fine-tuning, the optimizer must store "momentum" for every single parameter. LoRA only requires this for the tiny  and  matrices, drastically lowering VRAM usage.
* **Faster Checkpointing:** Instead of saving a new 140GB model file for every experiment, you only save the LoRA weights, which are typically only **10MB to 100MB**.

### Connection to Inference Optimization

LoRA does not inherently make an LLM "run faster" (generate tokens more quickly) than the base model, but it offers unique operational efficiencies:

* **No Latency Overhead (via Merging):** You can mathematically "merge" the trained LoRA weights back into the original weights before deployment. This means the final model has the exact same architecture as the original, resulting in **zero additional latency** during inference.
* **Multi-Tenant Serving:** This is the biggest "inference" win. A single server can keep one "base" model in memory and swap in dozens of different LoRA adapters (e.g., one for coding, one for medical chat, one for creative writing) on the fly. This is much more efficient than loading 10 separate full models.
* **Reduced Storage:** Because the adapters are so small, you can deploy personalized AI to edge devices or mobile apps without needing gigabytes of storage for every new task.

### Summary Comparison

| Feature | Full Fine-Tuning | LoRA |
| --- | --- | --- |
| **Trainable Params** | 100% | < 1% |
| **GPU Memory (Training)** | Very High | Very Low |
| **Inference Latency** | Baseline | Baseline (if merged) |
| **Model Switching** | Slow (Reload entire model) | Fast (Swap small adapters) |
| **Storage Cost** | Huge (GBs per task) | Tiny (MBs per task) |

## QLoRA
If LoRA is a "lite" version of fine-tuning, **QLoRA (Quantized LoRA)** is the "ultra-lite" version.

QLoRA takes the efficiency of LoRA and pushes it further by compressing the base model itself. While LoRA focuses on reducing **trainable parameters**, QLoRA focuses on reducing the **memory footprint of the frozen base model**.

### How QLoRA Works

The "Q" stands for **Quantization**. In a standard LoRA setup, the base model is usually loaded in 16-bit precision ( or ). QLoRA uses three main innovations to shrink that:

* **4-bit NormalFloat (NF4):** It compresses the base model weights from 16-bit down to 4-bit. This reduces the VRAM required to hold the model by **~75%**.
* **Double Quantization:** It quantizes the "quantization constants" themselves, saving an extra bit or two of memory that usually goes to management overhead.
* **Paged Optimizers:** It uses "paged" memory (similar to virtual memory on a PC) to handle sudden spikes in memory usage, preventing "Out of Memory" (OOM) crashes during training.


### Training vs. Inference in QLoRA

Just like standard LoRA, the primary benefit is **training efficiency**, but there is a slight "tax" on inference speed.

QLoRA is the gold standard for training on a budget. It allows you to fine-tune a **70B parameter model** (which usually requires 140GB+ of VRAM) on a single **48GB GPU** (like an A6000 or RTX 6000).

* **Cost:** Makes fine-tuning massive models accessible to anyone with a high-end consumer GPU.
* **Accuracy:** Despite the heavy compression, QLoRA typically reaches **99% of the performance** of a full 16-bit fine-tune.


Inference with QLoRA is slightly different from standard LoRA:

* **Computation Overhead:** Since the base weights are in 4-bit, but the math needs to happen in 16-bit, the system must "de-quantize" the weights on the fly during every forward pass. This makes inference **slightly slower** (roughly 5–10%) than an unquantized model.
* **Merging Limitations:** Unlike standard LoRA, you cannot simply "merge" QLoRA adapters back into a high-precision base model without losing the memory benefits. You typically serve the model in its 4-bit state and load the 16-bit adapter on top.

### Summary Comparison: LoRA vs. QLoRA

| Feature | LoRA | QLoRA |
| --- | --- | --- |
| **Base Model Precision** | 16-bit () | 4-bit () |
| **Memory for 7B Model** | ~14–16 GB VRAM | ~5–8 GB VRAM |
| **Memory for 70B Model** | ~140+ GB (Multi-GPU) | ~48 GB (Single GPU) |
| **Inference Speed** | Fast (Baseline) | Slightly Slower (Dequantization overhead) |
| **Primary Use Case** | Fast experimentation on mid-range hardware. | Training massive models on limited hardware. |

## Zero Redundancy Optimizer (ZeRO) – memory-efficient distributed training.

## Sharded DDP / FSDP – model parallelism strategies.

## Checkpoint averaging / EMA of weights – smoother convergence.


## Fine-Tuning and Adapters

### SFT
* Supervised Fine-Tuning (SFT) uses labeled datasets to fine-tune LLM

* 🤗 `transformer` [AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSequenceClassification) to load a pre-trained model for sequence classification tasks

```python
from transformers import AutoModelForSequenceClassification

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "roberta-large"

awt_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_categories,
    id2label=id2label,
    label2id=label2id).to(device)
```

* Use `Trainer` with a dataset of input-output pairs.

```python
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    ...  # other training args
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

* For inference in 🤗, use the fine-tuned model to predict labels.

```python
from transformers import pipeline

classifier = pipeline('text-classification', model=model_path, truncation=True, padding=True, device = device, batch_size = batch_size)

def return_top_predictions_udf(texts: pd.Series) -> pd.Series:
 predictions = [prediction['label'] for prediction in classifier(texts.to_list())]
 return pd.Series(predictions)

texts = pd.Series(["Sample text 1", "Sample text 2"])
predicted_labels = return_top_predictions_udf(texts)
print(predicted_labels)
```



### DPO

* Direct Preference Optimization (DPO) fine-tunes LLMs directly on preference data without requiring a reward model. 

* Optimizes the model to increase the likelihood of preferred outputs over non-preferred ones based on human feedback.

**Training Recipe for DPO**
1. Collect preference data: pairs of preferred and non-preferred outputs for given inputs.

2. Define the DPO loss function:

    $L_{DPO} = -\log \sigma\left(\frac{1}{\tau}(s_{\theta}(x, y^+) - s_{\theta}(x, y^-))\right)$, where

* $s_{\theta}(x, y)$ is the model score for 
* input-output pair $(x, y)$
* $y^+$ is the preferred output
* $y^-$ is the non-preferred output
* $\sigma$ is the sigmoid function
* $\tau$ is a temperature hyperparameter.

3. Fine-tune the model using gradient descent to minimize the DPO loss over the preference dataset.

4. Evaluate the fine-tuned model on held-out preference data to ensure it aligns with human preferences.




## Libraries and Frameworks

* [DeepSpeed](https://www.deepspeed.ai/) – large model training optimizations.
* [PyTorch Lightning](https://www.pytorchlightning.ai/) – high-level training framework.
* FSDP in PyTorch – built-in model sharding.
* Unsloth – efficient large model training.