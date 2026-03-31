## 1. Zero-Shot & Few-Shot Prompting

### Q1: What is zero-shot prompting and when does it work?

**Answer:**

**Zero-shot prompting** is giving the model a task with no examples — just a direct instruction. The model must rely entirely on knowledge from pre-training and fine-tuning (RLHF/SFT) to understand what you want.

```
Prompt: "Classify the sentiment of this review as positive, negative, or neutral:
'The product arrived on time but the packaging was damaged.'"

Response: "Neutral"
```

**Why it works:** Instruction-tuned models (GPT-4, Claude, Gemini) were fine-tuned on millions of instruction-response pairs. They have seen enough "classify this", "summarize this", "translate this" patterns to generalize without examples.

**When it works well:**
- Tasks the model has clearly seen in training: translation, summarization, sentiment, basic QA
- Simple, well-defined tasks with unambiguous output format
- When latency/token cost is a constraint (no examples = shorter prompt)

**When it breaks:**
- Niche or domain-specific tasks the model hasn't seen at scale
- Tasks requiring a very specific output format the model doesn't know to infer
- Edge cases and corner cases — without examples, the model defaults to the "average" behavior

**Real-world example:** Stripe uses zero-shot prompts for basic payment dispute classification — the task is common enough that GPT-4 generalizes without examples, saving tokens at millions-of-requests scale.

---

### Q2: What is few-shot prompting and how do you design good examples?

**Answer:**

**Few-shot prompting** provides 2–8 examples (input → output pairs) before the actual task. The model learns the pattern from the examples and applies it.

```
Prompt:
Review: "Absolutely loved it!" → Sentiment: Positive
Review: "Terrible quality, fell apart in a week." → Sentiment: Negative
Review: "It's okay, nothing special." → Sentiment: Neutral

Review: "Fast shipping but product is mediocre." → Sentiment:
```

**Why it works:** In-context learning — the transformer's attention mechanism can identify the input→output pattern from the examples and apply it to the new input, without any weight updates.

**How to design good few-shot examples:**

| Principle | Bad Example | Good Example |
|---|---|---|
| **Diversity** | All positive reviews | Mix of all classes |
| **Difficulty** | Only obvious cases | Include edge cases and ambiguous ones |
| **Format consistency** | Varied formatting | Identical format in every example |
| **Relevance** | Random examples | Examples similar in domain to test input |
| **Length balance** | All short, then one long | Similar length inputs across examples |

**Key numbers:**
- 2–3 examples: Usually sufficient for format learning
- 4–8 examples: Better for complex tasks or unusual domains
- 8+ examples: Diminishing returns; consider fine-tuning instead
- Each example costs tokens — at GPT-4o pricing ($2.50/1M input tokens), 5 examples of 100 tokens each = $0.00125 overhead per call. At 1M calls/day, that's $1,250/day just from examples.

**Real-world example:** Customer support ticket classification systems often need 5–6 examples because the label taxonomy is company-specific (not general "positive/negative"). Without examples, GPT-4 maps tickets to generic categories that don't match the internal schema.

---

### Q3: What is the difference between zero-shot and few-shot CoT, and when does each apply?

**Answer:**

**Zero-shot CoT:** Append "Let's think step by step" to the prompt. This single phrase activates the model's chain-of-thought behavior without any examples. Discovered in the 2022 paper "Large Language Models are Zero-Shot Reasoners" (Kojima et al.).

```
Prompt: "A store buys 15 apples at $0.80 each and sells them at $1.20 each.
If 3 apples spoil, what is the profit? Let's think step by step."
```

**Few-shot CoT:** Provide complete reasoning traces as examples. The model mimics the reasoning structure.

```
Example:
Q: "A store buys 10 apples at $0.50. Sells 8 at $1.00. Profit?"
A: "Cost = 10 × $0.50 = $5.00. Revenue = 8 × $1.00 = $8.00. Profit = $8.00 - $5.00 = $3.00."

Q: "A store buys 15 apples at $0.80..."
A: [model continues in same format]
```

**Performance comparison (from Wei et al., 2022 — original CoT paper):**

| Task | Standard Prompting | Zero-shot CoT | Few-shot CoT |
|---|---|---|---|
| GSM8K (math word problems) | 17.9% | 40.7% | 58.1% |
| AQuA (algebra) | 22.4% | 33.5% | 35.8% |
| SVAMP (arithmetic) | 69.9% | 74.4% | 83.7% |

**When to use which:**
- **Zero-shot CoT:** Quick wins, no example data available, general reasoning tasks
- **Few-shot CoT:** Specialized domains, consistent reasoning format required, high-stakes tasks needing reliability

**Senior engineer tip:** For agentic tasks, always use some form of CoT. An agent deciding which tool to call benefits massively from reasoning through the decision rather than jumping to an answer.

---

### Q4: What are the failure modes of few-shot prompting in production?

**Answer:**

**1. Recency bias:** The model overweights the last example in the prompt. Always shuffle examples or put the hardest/most representative example last.

**2. Label imbalance:** If 4 out of 5 examples are "Positive", the model skews toward predicting Positive. Balance classes in your examples.

**3. Spurious correlations:** If all your "Negative" examples happen to be short, the model may learn "short = Negative" rather than the semantic content.

**4. Format brittleness:** The model imitates example format exactly — including any typos or inconsistencies. One malformed example can corrupt all subsequent outputs.

**5. Example-query mismatch:** If examples are all short reviews but the test input is a 3-paragraph essay, the model may truncate or misformat the output.

**6. Token blowup at scale:** 5 examples × 200 tokens each = 1,000 tokens overhead per call. At 10M calls/month, that's 10B extra input tokens — on GPT-4o, ~$25,000/month extra.

**Mitigation strategies:**
- Dynamically select examples using semantic similarity to the current query (dynamic few-shot / example retrieval)
- Store examples in a vector DB, retrieve the top-3 most similar to the current input
- This cuts average example count from 5 fixed to 2–3 dynamic, with better accuracy

---