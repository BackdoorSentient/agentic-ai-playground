## Greedy Decoding

### Q1: What is greedy decoding?

**Answer:**

**Greedy decoding** is a text generation strategy where, at every step, the model picks the **single most probable next token** from the vocabulary — the "greedy" choice — without considering what comes after.

Formally, at each position:
```
token_t = argmax P(token | previous tokens)
```

This repeats until an end-of-sequence (EOS) token is produced or a max length limit is hit.

**Why it matters:**

Greedy decoding is the default strategy in most production LLM inference pipelines because it's fast, deterministic, and cheap. When you hear "the model always gives the same answer for the same input" — that's greedy decoding at work.

---

### Q2: What actually happens under the hood during greedy decoding?

**Answer:**

At each generation step, the model:

1. **Runs a forward pass** through the full transformer, producing a logit vector of shape `[vocab_size]` (e.g., 50,257 for GPT-2, ~32,000 for LLaMA 3)
2. **Applies softmax** to convert logits → probability distribution
3. **Takes argmax** of the distribution → selects the highest-probability token
4. **Appends** that token to the sequence and repeats

```python
# Simplified greedy decoding loop
input_ids = tokenizer.encode("The capital of France is")

for _ in range(max_new_tokens):
    logits = model(input_ids).logits[:, -1, :]   # last token's logits
    next_token = logits.argmax(dim=-1)            # greedy pick
    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
    if next_token == tokenizer.eos_token_id:
        break
```

**Temperature interaction:** Before argmax, logits can be divided by a temperature value `T`. With greedy decoding, `T` is effectively irrelevant — argmax is scale-invariant. Temperature only matters when you're sampling.

---

### Q3: What is the KV cache and how does greedy decoding benefit from it?

**Answer:**

The **KV (Key-Value) cache** stores the intermediate attention key and value tensors from previously processed tokens so they don't have to be recomputed on each new step.

Without KV cache: every new token requires recomputing attention over the entire sequence → O(n²) cost per step.

With KV cache: only the **new token's** K and V are computed and appended → effectively O(n) per step.

```
Step 1: Process "The capital of France"  → compute + cache K,V for all 5 tokens
Step 2: Generate "is"                    → compute K,V for "is" only, attend over cached K,V
Step 3: Generate "Paris"                 → compute K,V for "Paris" only, attend over cached K,V
```

**Why greedy decoding benefits most:** Since greedy decoding always produces a single sequence (batch size = 1 per request), the KV cache is always fully utilized. Sampling methods that generate multiple candidates in parallel require separate caches per beam/sample, multiplying memory cost.

**Practical impact:** KV cache can reduce token generation latency by **5–10x** for long sequences. This is why inference frameworks like vLLM and TGI are heavily optimized around KV cache management.

---

### Q4: What is the exposure bias problem with greedy decoding?

**Answer:**

**Exposure bias** is the training/inference mismatch that affects all autoregressive decoding, including greedy.

- **During training:** The model is fed the **ground-truth previous tokens** at each step (teacher forcing). It learns `P(token_t | ground_truth_t-1)`.
- **During inference (greedy):** The model is fed **its own previously generated tokens**. It must condition on `P(token_t | its_own_output_t-1)`.

If the model makes a slightly wrong prediction early on, the error **compounds** — future tokens are conditioned on a distribution the model was never trained on.

**Concrete example:**
```
Ground truth:  "The Eiffel Tower is located in Paris, France"
Greedy output: "The Eiffel Tower is located in Paris, which is..."
                                                   ↑ slightly off
                                     All future tokens now conditioned on this
```

**Why it matters for senior engineers:**
- Exposure bias is a core reason why long-form greedy outputs degrade in quality
- Techniques like **scheduled sampling** and **RLHF** partially address this by training the model on its own outputs
- This is also why beam search can sometimes outperform greedy — it hedges against early errors

---

### Q5: What is the repetition problem in greedy decoding and how is it fixed?

**Answer:**

Greedy decoding is notorious for **repetition loops** — once a token or phrase is generated, it often becomes the highest-probability continuation of itself:

```
"The best way to learn is to learn and to learn and to learn and to learn..."
```

This happens because the model's training distribution assigns high probability to n-gram continuations it has seen frequently.

**Fix 1 — Repetition penalty (multiplicative):**
```python
# Penalize tokens already in the sequence by dividing their logits
for token_id in set(generated_ids):
    logits[token_id] /= repetition_penalty  # e.g., 1.3
```
A penalty > 1.0 reduces the probability of repeating tokens.

**Fix 2 — No-repeat n-gram blocking:**
```python
# In HuggingFace Transformers:
model.generate(input_ids, no_repeat_ngram_size=3)
```
This hard-blocks any 3-gram from appearing twice, overriding greedy selection if needed.

**Fix 3 — Min-new-tokens:** Force the model to generate at least N tokens before allowing EOS — prevents premature short repetitive outputs.

**Tradeoff:** Repetition penalty can hurt factual outputs. If "Paris" is the correct answer three times in a row, penalizing it degrades quality. Tune carefully.

---

### Q6: How does greedy decoding compare to beam search, and why did beam search fall out of favor for LLMs?

**Answer:**

| Strategy | How it works | Compute | Quality |
|---|---|---|---|
| **Greedy** | Keep top-1 sequence at each step | O(n · V) | Locally optimal |
| **Beam search (k=5)** | Keep top-k sequences at each step | O(k · n · V) | Better global optima |

Where `n` = sequence length, `V` = vocab size.

**Why beam search fell out of favor for LLMs:**

1. **Modern LLMs are large enough** that greedy outputs are already high quality — the marginal gain from beam search is smaller than it was for seq2seq models in 2018.

2. **Memory cost:** Beam search with k=5 requires maintaining 5 separate KV caches simultaneously → 5× memory overhead per request.

3. **The "beam search curse":** Research (Stahlberg & Byrne, 2019) showed beam search can actually **hurt** quality on open-ended generation — it finds high-probability but generic/boring text. For summarization and translation it still helps.

4. **Latency:** Running k forward passes per step multiplies inference time by roughly k.

**When beam search still wins:** Machine translation, summarization, and constrained generation tasks where output quality is measured by BLEU/ROUGE and diversity is not needed.

---

### Q7: What is constrained decoding and how does it build on greedy decoding?

**Answer:**

**Constrained decoding** modifies the greedy argmax step to enforce structural constraints on the output — most commonly to guarantee valid JSON, SQL, or other structured formats.

**The problem without it:**
```python
response = model.generate("Return a JSON with name and age")
# Output: "Sure! Here's the JSON: {name: John, age: 30}"  ← invalid JSON
```

**How it works:**
At each step, instead of argmax over the full vocabulary, we:
1. Determine which tokens are **valid continuations** given the current partial output and the target grammar
2. **Mask** all invalid tokens (set their logits to -∞)
3. Apply argmax over the remaining valid tokens

```python
# Conceptual example with Outlines library
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B")

schema = '{"name": "string", "age": "integer"}'
generator = outlines.generate.json(model, schema)

result = generator("Generate a person")
# Guaranteed valid JSON output: {"name": "Alice", "age": 28}
```

**Libraries:** Outlines, Guidance, LM-Format-Enforcer, LMQL — all implement this pattern.

**Why senior engineers care:** In production agentic systems where LLM outputs are parsed by downstream code, constrained decoding eliminates an entire class of runtime errors without prompt engineering hacks.

---

### Q8: How does greedy decoding perform vs. sampling in production, and when do you choose each?

**Answer:**

**Use greedy decoding when:**
- Output correctness is well-defined (code gen, SQL, structured extraction, classification)
- You need **reproducibility** (same input → same output, critical for debugging and testing)
- **Latency is critical** — no overhead from multiple samples
- The model is large enough that greedy outputs are already high quality

**Use sampling when:**
- Tasks require **creativity or diversity** (story generation, brainstorming, dialogue)
- You're running **best-of-N** inference (generate N samples, rank by a reward model, return best)
- You want to **avoid repetition** without explicit penalty mechanisms

**Real production defaults:**
- GitHub Copilot / code completion: greedy (deterministic completions)
- ChatGPT conversational responses: sampling with low temperature (~0.7)
- Claude default API: sampling with temperature=1.0
- Structured data extraction pipelines: greedy or constrained decoding

**The nuance:** A 70B model with greedy decoding often outperforms a 7B model with beam search. Model quality dominates decoding strategy. Decoding strategy is a second-order optimization.

---

### Q9: What is the "lost in the middle" problem and does it affect greedy decoding?

**Answer:**

The **lost in the middle** problem (Liu et al., Stanford 2023) refers to LLMs performing worst on information placed in the **middle** of a long context — not at the start or end.

This is a **model-level issue**, not a decoding strategy issue — it affects greedy, sampling, and beam search equally. However, it directly impacts **when greedy decoding appears to fail** in practice.

**Why it happens:**
Transformers attend to all tokens but the attention gradient signal is stronger for tokens at the beginning (primacy effect from positional encoding) and end (recency effect) of the context.

**Practical impact for engineers:**

```
Context: [System prompt] [Document part 1] [Document part 2] [Document part 3] [User query]
                                              ↑ model struggles most here
```

**Mitigations:**
- **Reorder context** — put the most relevant chunk at the start or end, not the middle
- **Use RAG** to pre-filter context below 10–30K tokens rather than stuffing full documents
- **Map-reduce patterns** — process chunks independently, then aggregate results

**Key numbers:** Research showed accuracy dropped from ~80% (beginning/end) to ~55% (middle) on multi-document QA tasks at 32K context.

---

### Q10: What are the key performance characteristics of greedy decoding at inference time?

**Answer:**

**Latency breakdown:**

| Phase | Metric | Scales with |
|---|---|---|
| **Prefill** (processing input tokens) | Time-to-first-token (TTFT) | Input length (O(n²) attention) |
| **Decode** (generating output tokens) | Tokens-per-second (TPS) | Output length, KV cache size |

**Why greedy is fastest for decode:**
- Single beam → full KV cache utilization
- No reranking or scoring overhead
- Modern hardware (A100/H100) optimized for exactly this access pattern

**Rough numbers on an A100 (80GB):**
- LLaMA 3 70B, greedy, batch=1: ~30–40 tokens/sec
- Same model, beam search k=4: ~8–12 tokens/sec
- Same model, top-p sampling: ~28–38 tokens/sec (minor overhead vs greedy)

**Memory cost of greedy vs. alternatives:**

```
KV cache memory = 2 × num_layers × num_heads × head_dim × seq_len × bytes_per_param

For LLaMA 3 70B (80 layers, fp16):
- Greedy (1 sequence):    ~1.5 GB at 4K context
- Beam search k=4:        ~6 GB at 4K context (4× multiplier)
```

**Practical guidance:** For latency-sensitive APIs, greedy + KV cache is the baseline. Only add sampling overhead when the use case genuinely requires it.