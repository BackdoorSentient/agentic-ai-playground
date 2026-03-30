## 6. Fine-tuning

### Q1: What is fine-tuning and when should you use it vs. prompt engineering or RAG?

**Answer:**

**Fine-tuning** updates a pre-trained model's weights on a smaller, curated dataset to adapt it to a specific task or domain.

**When to use what:**

| Approach | Best For | Limitations |
|---|---|---|
| Prompt engineering | Quick iteration, no training data, general tasks | Token cost on every call, limited by context window |
| RAG | Dynamic/fresh knowledge, large knowledge bases, citations needed | Retrieval quality bottleneck, added latency |
| Fine-tuning | Style adaptation, consistent format, confidential data not sent to API, reduce token cost | Needs curated data (500–10K examples), training cost, re-training when data changes |
| Fine-tuning + RAG | Best of both: tuned model + dynamic knowledge | Most complex, highest cost |

**Key rule of thumb:** If you need the model to *know more facts*, use RAG. If you need it to *behave differently or consistently*, use fine-tuning.

**Real-world example:** 
- A legal firm uses RAG to ground answers in their document library (dynamic, confidential).
- They also fine-tune to ensure the model always formats case citations in their house style (behavioral).

---

### Q2: What are LoRA and QLoRA, and why are they used for fine-tuning?

**Answer:**

Full fine-tuning updates every parameter in the model. For a 70B-parameter model, that requires enormous GPU memory and compute. **Parameter-Efficient Fine-Tuning (PEFT)** techniques solve this.

**LoRA (Low-Rank Adaptation):**

Instead of updating the full weight matrix W, LoRA decomposes the update into two small matrices:

```
ΔW = A × B    where A is (d × r) and B is (r × d), with r << d
```

Only A and B are trained. Typical rank `r = 4 to 64`. If the original weight matrix is (4096 × 4096) = 16M parameters, with r=8 you train only 2 × 4096 × 8 = 65K parameters per matrix — a 245x reduction.

**QLoRA (Quantized LoRA):**

Combines LoRA with 4-bit quantization of the base model weights. Enables fine-tuning 70B models on a single A100 GPU (80GB VRAM) that would otherwise require 8+ GPUs for full fine-tuning.

**Numbers:**
- Full fine-tune Llama 3 70B: ~8× A100 80GB GPUs, weeks of training
- QLoRA Llama 3 70B: 1× A100 80GB, hours to days
- LoRA adds <1% overhead at inference since A×B can be merged back into W

**When to use which:**
- Small model (<7B), GPU-rich: Full fine-tune
- Medium model (7B–70B), budget-constrained: LoRA
- Large model (>70B), single GPU: QLoRA

---

### Q3: How much data do you need for fine-tuning?

**Answer:**

It depends heavily on the task:

| Task | Minimum | Recommended |
|---|---|---|
| Style/tone adaptation | 50–200 examples | 500–2,000 |
| Domain adaptation (e.g., medical QA) | 500–1,000 | 5,000–50,000 |
| New skill (e.g., code generation in a niche framework) | 1,000–5,000 | 10,000+ |
| Function calling schemas | 200–500 | 1,000–5,000 |

**Quality >> Quantity.** OpenAI's own fine-tuning documentation notes that 50 high-quality examples often outperform 1,000 noisy ones.

**Data format (OpenAI fine-tuning JSONL example):**
```jsonl
{"messages": [{"role": "system", "content": "You are a legal assistant."}, 
               {"role": "user", "content": "Summarize this contract clause: ..."}, 
               {"role": "assistant", "content": "This clause states..."}]}
```

**Common pitfalls:**
- **Data leakage:** Test set examples appearing in training set — inflated evaluation scores.
- **Overfitting:** Fine-tuning on too few examples on too many epochs causes the model to memorize rather than generalize. Monitor validation loss.
- **Distribution shift:** Training on neat examples but deploying on messy real-world inputs.

---
