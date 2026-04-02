## 2. Temperature

### Q1: What is a probability distribution?

**Answer:**

A **probability distribution** is simply a list of all possible outcomes and how likely each one is — where all the probabilities add up to 100%.

**Simple example — a dice:**

| Outcome | Probability |
|---------|-------------|
| 1       | 17%         |
| 2       | 17%         |
| 3       | 17%         |
| 4       | 17%         |
| 5       | 17%         |
| 6       | 17%         |
| **Total** | **100%**  |

That table *is* the probability distribution — every possible outcome, with its likelihood.

**In LLMs:**

At every single step, the model produces a probability distribution over its entire vocabulary (~100,000 tokens):

| Token   | Probability |
|---------|-------------|
| "the"   | 40%         |
| "a"     | 25%         |
| "an"    | 15%         |
| "this"  | 10%         |
| ...     | ...         |
| **Total** | **100%**  |

Then `temperature`, `top-k`, and `top-p` all work by **reshaping or filtering** this distribution before the model picks the next token.

**One line summary:**

> A probability distribution tells you — *for every possible outcome, how likely is it?* — and they all must add up to 100%.

---

### Q2: What does temperature control in an LLM, and what is happening mathematically?

**Answer:**

Temperature controls the **sharpness or flatness of the probability distribution** over the vocabulary at each token generation step.

**Math:**

At each step, the model produces raw scores called **logits** for every token in the vocabulary (e.g., 100,000 tokens for GPT-4). These are converted to probabilities using **softmax**:

```
P(token_i) = exp(logit_i / T) / sum(exp(logit_j / T) for all j)
```

Where `T` is temperature.

- When **T → 0**: The distribution becomes a spike on the highest-logit token. The model always picks the most probable token (greedy decoding). Output is deterministic and repetitive.
- When **T = 1.0**: The raw softmax probabilities are used as-is. This is the model's "natural" distribution.
- When **T > 1.0**: The distribution flattens. Lower-probability tokens get more chance. Output becomes more random and creative — but also more likely to be incoherent.

**Intuition:** Temperature is like a volume knob on randomness. Low = focused and predictable. High = creative and chaotic.

---

### Q3: What temperature values should you use for different use cases, and why?

**Answer:**

| Use Case | Temperature Range | Reason |
|---|---|---|
| Tool calling / function invocation | 0.0 – 0.2 | Must pick the right function name and arguments deterministically |
| Code generation | 0.1 – 0.3 | Syntax must be correct; slight variation for alternatives |
| Question answering / factual | 0.0 – 0.3 | Reproducibility and accuracy |
| Summarization | 0.3 – 0.5 | Some variation is fine, but should stay grounded |
| Chatbot / conversation | 0.5 – 0.8 | Natural, varied responses without going off-rails |
| Creative writing / brainstorming | 0.8 – 1.2 | Diversity and novelty desired |
| Poetry / experimental | 1.0 – 1.5 | High creativity, some incoherence acceptable |

**Real-world numbers from Anthropic and OpenAI documentation:**
- OpenAI recommends `temperature=0` for deterministic outputs in production systems.
- Claude's default temperature is `1.0` in the API; for agentic workflows, Anthropic recommends `0` to `0.3`.

**Key trade-off:** Higher temperature = more diverse outputs = harder to evaluate automatically. If your evaluation pipeline expects consistent format, high temperature breaks your parsers.

---

### Q4: What is top-k sampling?

**Answer:**

**Top-k sampling** means — only consider the **k most likely tokens** when picking the next word, ignore everything else.

**Simple example — k=3:**

The model's full probability distribution:

| Token   | Probability |
|---------|-------------|
| "the"   | 40%         |
| "a"     | 25%         |
| "an"    | 15%         |
| "this"  | 10%         |
| "every" | 5%          |
| "some"  | 3%          |
| ...     | ...         |

With `top-k=3`, you **cut the list to top 3** and redistribute probabilities among them:

| Token | Original | After top-k=3 |
|-------|----------|----------------|
| "the" | 40%      | 50%            |
| "a"   | 25%      | 31%            |
| "an"  | 15%      | 19%            |
| ~~"this"~~ | ~~10%~~ | ❌ removed |
| ~~"every"~~ | ~~5%~~ | ❌ removed |

The model now only samples from those 3 tokens.

**Code:**
```python
import torch

logits = torch.tensor([4.0, 3.0, 2.5, 1.0, 0.5])  # raw model outputs
k = 3

# Zero out everything outside top-k
top_k_values, top_k_indices = torch.topk(logits, k)
filtered = torch.full_like(logits, float('-inf'))
filtered[top_k_indices] = top_k_values

# Convert to probabilities
probs = torch.softmax(filtered, dim=-1)
print(probs)  # only top 3 have non-zero probability
```

**The problem with top-k:**

`k` is a fixed number — it doesn't adapt to the model's confidence:

- Model is **very confident** → top 3 tokens cover 97% probability. With `k=50` you still include 47 near-zero tokens unnecessarily.
- Model is **uncertain** → 200 tokens each have ~0.5% probability. With `k=50` you cut out 150 valid options.

This is exactly why **top-p (nucleus sampling)** was introduced — it adapts the cutoff based on probability mass instead of a fixed count.

**Typical values:**
- `top_k = 50` is a common default
- `top_k = 1` = always pick the single most likely token (greedy decoding)

---

### Q5: What is the difference between temperature and top-k?

**Answer:**

Both modify how the next token is sampled, but they operate differently:

- **Temperature** reshapes the entire probability distribution (soft control).
- **Top-k** hard-cuts the vocabulary to only the `k` most probable tokens before sampling. All tokens outside the top-k get probability 0.

They are often used **together**: first apply temperature to reshape probabilities, then apply top-k (or top-p) to limit sampling to plausible tokens.

**Example:** At temperature=0.9 and top-k=50, you get varied outputs but only from the 50 most likely tokens — preventing truly garbage tokens from being selected.

---