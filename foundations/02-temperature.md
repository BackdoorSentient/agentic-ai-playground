## 2. Temperature

### Q1: What does temperature control in an LLM, and what is happening mathematically?

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

### Q2: What temperature values should you use for different use cases, and why?

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

### Q3: What is the difference between temperature and top-k?

**Answer:**

Both modify how the next token is sampled, but they operate differently:

- **Temperature** reshapes the entire probability distribution (soft control).
- **Top-k** hard-cuts the vocabulary to only the `k` most probable tokens before sampling. All tokens outside the top-k get probability 0.

They are often used **together**: first apply temperature to reshape probabilities, then apply top-k (or top-p) to limit sampling to plausible tokens.

**Example:** At temperature=0.9 and top-k=50, you get varied outputs but only from the 50 most likely tokens — preventing truly garbage tokens from being selected.

---