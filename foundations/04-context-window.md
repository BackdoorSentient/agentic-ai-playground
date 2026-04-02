## 4. Context Window

### Q1: What is an inference call?

**Answer:**

An **inference call** is simply when you **send a prompt to an LLM and get a response back**.

"Inference" = the model doing its job — taking your input and predicting/generating output.

So every time your code does this:
```python
response = client.messages.create(
    model="claude-sonnet-4-5",
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
```

That's one **inference call** — you're calling the model to run inference on your input.

**Why it matters:**

Every inference call costs tokens and adds latency. So when you hear:

- "This agent makes **3 inference calls** per user query" → it hits the LLM 3 times to complete the task
- More calls = higher cost + higher latency

This is why senior engineers always ask: *"How many inference calls does this agent make per request?"*

### Q2: What is a context window, and what counts towards it?

**Answer:**

The **context window** is the maximum number of tokens an LLM can "see" in a single inference call — it includes:
- The system prompt
- All conversation history (prior turns)
- Retrieved documents (in RAG)
- Tool definitions (for function calling)
- The current user message
- The model's response being generated

Everything in the window is attended to simultaneously via the transformer's attention mechanism. Nothing outside the window is visible to the model.

**Current context windows (as of early 2025):**

| Model | Context Window |
|---|---|
| Gemini 1.5 Pro | 1,000,000 tokens (~750,000 words) |
| Gemini 1.5 Flash | 1,000,000 tokens |
| Claude 3.5 Sonnet | 200,000 tokens (~150,000 words) |
| GPT-4o | 128,000 tokens |
| GPT-4o mini | 128,000 tokens |
| Llama 3.1 405B | 128,000 tokens |
| Mistral Large | 128,000 tokens |

**Token approximation:** 1 token ≈ 0.75 English words. 1,000 tokens ≈ 750 words ≈ 1.5 pages.

---

### Q3: What are the practical limitations of large context windows?

**Answer:**

Large context ≠ free lunch. Key issues:

**1. Cost:** Tokens are billed on input + output. A 1M-token context request costs significantly more than a 10K-token one.

**2. Latency:** Time-to-first-token (TTFT) scales with context length. Processing 1M tokens takes longer than 10K even on fast hardware.

**3. Lost in the middle problem:** Research (Stanford, 2023) showed that LLMs perform **worst on information placed in the middle** of a long context. Performance is best for information at the start and end. This means naively stuffing documents into a 200K context doesn't guarantee the model will "find" the relevant part.

**4. Attention cost (quadratic scaling):** The attention mechanism in transformers is O(n²) with respect to sequence length. Modern models use techniques like FlashAttention, sliding window attention, and ring attention to mitigate this — but it remains a fundamental bottleneck.

**5. Quality degradation:** Models are not equally good across their full claimed context length. A model "trained" for 200K may perform much better on 50K inputs.

**Practical guidance:** Use RAG to pre-filter context below 10–30K tokens whenever possible. Reserve large context windows for tasks that genuinely need it (e.g., analyzing an entire codebase).

---

### Q4: How do you manage a conversation that exceeds the context window limit?

**Answer:**

Several strategies, often combined:

**1. Sliding window:** Drop the oldest turns from conversation history. Simple but loses early context.

**2. Summarization:** Periodically summarize the conversation so far into a compressed block. LangChain's `ConversationSummaryMemory` does this automatically.

```python
from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm, max_token_limit=2000)
```

**3. Retrieval-augmented memory:** Store all conversation turns in a vector DB. Retrieve only the most relevant past turns for each new query. This is how long-running agents scale to weeks of conversation.

**4. Hierarchical summarization:** Summarize in chunks, then summarize the summaries (a tree structure). Useful for very long documents.

**5. KV cache:** At the API layer, models can cache the KV (key-value) computations for static prefix tokens (like system prompts). Anthropic's prompt caching reduces cost by up to 90% and latency by up to 85% for repeated prefixes.

**Key numbers to remember:** Start worrying about context management above 80% of the context window limit. At 95%+ you risk truncation errors or degraded performance.

---