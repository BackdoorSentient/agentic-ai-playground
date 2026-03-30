# Day 1: Understanding How LLMs Work — Senior Engineer Q&A Notes

> **Format:** Each topic has a set of questions a senior engineer should be able to answer, with deep explanations, trade-offs, real-world examples, and key numbers.

---

## Table of Contents

1. [Product AI Patterns — Assist vs Automate](#1-product-ai-patterns)
2. [Temperature](#2-temperature)
3. [Top-p (Nucleus Sampling)](#3-top-p-nucleus-sampling)
4. [Context Window](#4-context-window)
5. [Pre-training](#5-pre-training)
6. [Fine-tuning](#6-fine-tuning)
7. [RLHF](#7-rlhf)
8. [Structured Outputs](#8-structured-outputs)
9. [RAG Architecture & Evaluation](#9-rag-architecture--evaluation)
10. [FastAPI Framework](#10-fastapi-framework)
11. [Hands-On Exercises](#11-hands-on-exercises)

---

## 1. Product AI Patterns

### Q1: What is the difference between "Assist" and "Automate" AI patterns, and when do you choose one over the other?

**Answer:**

The **Assist** pattern keeps a human in the loop for every significant decision. The AI drafts, suggests, or flags — but a human reviews before anything is committed. Think GitHub Copilot autocompleting code: you accept or reject each suggestion.

The **Automate** pattern lets the AI execute end-to-end with no human touch per transaction. Think of an AI agent that triages support tickets, routes them, and auto-responds to a class of simple queries.

**Decision framework:**

| Dimension | Assist | Automate |
|-----------|--------|----------|
| Error cost | High (medical, legal, financial) | Low (low-stakes routing, classification) |
| User trust | Users want control | Users want speed |
| Edge case frequency | High/unpredictable | Low and known |
| Latency tolerance | Human-speed (seconds to minutes) | Sub-second preferred |
| Reversibility | Irreversible actions (payments, deletions) | Easily undoable |

**Real-world examples:**

- **Assist:** Notion AI writes a draft — the user edits before publishing. Gmail Smart Compose suggests — the user accepts word by word.
- **Automate:** Intercom's Fin resolves 40–60% of chat tickets autonomously. AWS Lambda functions auto-provisioned by AI ops tools.

**Key trade-off:** Automation maximizes throughput and cost savings but amplifies errors at scale. A bug in an assisted system affects one user per mistake; a bug in an automated system can fire 10,000 wrong emails before anyone notices.

**Senior engineer tip:** Always design the Automate path with a "confidence threshold" gate — if the model's confidence is below 0.80 (or whatever your calibration shows), fall back to Assist. This is called **graceful degradation**.

---

### Q2: What are the main UX flows in AI-powered products, and what failure modes must you design for?

**Answer:**

**Common UX flows:**

1. **Inline suggestion** (Copilot-style): AI completes as you type. Latency must be < 200ms for it to feel responsive.
2. **Chat/turn-based**: User sends a query, AI responds. The round-trip latency budget is typically < 3 seconds for user satisfaction.
3. **Background agent**: AI runs autonomously and surfaces results. User sees output, not process.
4. **Approval-gated**: AI proposes, human approves before execution (e.g., before sending an email, before a database write).

**Failure modes and how to handle them:**

| Failure Mode | Description | Mitigation |
|---|---|---|
| Hallucination | Model states false information confidently | RAG grounding, citation requirements, factuality checks |
| Latency spikes | Model slow under load | Streaming responses, P95 SLA monitoring, fallback to smaller model |
| Prompt injection | User manipulates system prompt | Input sanitization, XML tag separation, instruction hierarchy |
| Overconfident wrong answers | Model says "definitely X" when it's wrong | Calibrated confidence scoring, hedging language in prompts |
| Context length overflow | Conversation exceeds window limit | Sliding window, summarization, retrieval |
| AI demo syndrome | Works in demos, breaks on real data | Regression test suites, golden datasets, production sampling |

**AI demo syndrome** deserves special attention: models often perform well on clean, structured demo inputs but fail on real-world messy data. Senior engineers build evaluation pipelines before shipping — not after.

---

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

## 3. Top-p (Nucleus Sampling)

### Q1: What is nucleus sampling and how does it differ from top-k?

**Answer:**

**Top-k** selects from a fixed number of top tokens. **Top-p (nucleus sampling)** selects from the smallest set of tokens whose **cumulative probability** reaches threshold `p`.

**Algorithm:**
1. Sort tokens by probability, descending.
2. Accumulate probabilities until the running sum ≥ p.
3. Only sample from this "nucleus" set.

**Why this is smarter than top-k:**

Imagine two situations:
- The model is very confident: the top 3 tokens cover 97% probability mass. With top-k=50, you'd include 47 tokens with near-zero probability. With top-p=0.95, you'd only sample from the top 3.
- The model is uncertain: 200 tokens each have ~0.5% probability. With top-k=50, you still only sample 50. With top-p=0.95, you sample from all 200, preserving the model's genuine uncertainty.

Top-p **adapts** to the model's confidence at each step. Top-k does not.

**Typical values:**
- `top_p = 0.9` is a common default.
- OpenAI recommends: "We generally recommend altering `top_p` or `temperature` but not both."

---

### Q2: When would you set top-p = 1.0 vs top-p = 0.5?

**Answer:**

- **top-p = 1.0**: No nucleus cutoff. The model can sample from any token in the vocabulary (full distribution). Maximizes diversity. Use for creative tasks where even rare tokens are desirable.
- **top-p = 0.9**: Cuts off the bottom 10% of the probability mass. A safe default for most applications.
- **top-p = 0.5**: Very conservative. Only samples from the highest-probability tokens that together cover 50% of the mass. Tight, predictable output. Good for structured/factual tasks.

**Real-world example:** For a legal document summarizer, you might use `temperature=0.2, top_p=0.7` — you want grounded, consistent language, not creative paraphrasing that might misrepresent the original.

---

### Q3: What happens if you set both temperature and top-p to non-default values simultaneously?

**Answer:**

You can, but it's confusing and often unnecessary. The typical guidance (from OpenAI and Anthropic):

> Modify one parameter at a time. Adjusting both temperature and top-p simultaneously creates compounding effects that are difficult to reason about and tune.

**Order of operations** (OpenAI API):
1. Temperature is applied first to reshape logit probabilities.
2. Top-p is applied after, on the reshaped distribution.
3. Top-k (if set) is applied after top-p.
4. A token is sampled from the final set.

So `temperature=2.0, top_p=0.1` would: first flatten the distribution wildly, then cut it back to a tight nucleus — an incoherent combination.

**Best practice:** Set temperature for overall creativity level. Leave top-p at 0.9–1.0 unless you have a specific reason to constrain the nucleus.

---

## 4. Context Window

### Q1: What is a context window, and what counts towards it?

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

### Q2: What are the practical limitations of large context windows?

**Answer:**

Large context ≠ free lunch. Key issues:

**1. Cost:** Tokens are billed on input + output. A 1M-token context request costs significantly more than a 10K-token one.

**2. Latency:** Time-to-first-token (TTFT) scales with context length. Processing 1M tokens takes longer than 10K even on fast hardware.

**3. Lost in the middle problem:** Research (Stanford, 2023) showed that LLMs perform **worst on information placed in the middle** of a long context. Performance is best for information at the start and end. This means naively stuffing documents into a 200K context doesn't guarantee the model will "find" the relevant part.

**4. Attention cost (quadratic scaling):** The attention mechanism in transformers is O(n²) with respect to sequence length. Modern models use techniques like FlashAttention, sliding window attention, and ring attention to mitigate this — but it remains a fundamental bottleneck.

**5. Quality degradation:** Models are not equally good across their full claimed context length. A model "trained" for 200K may perform much better on 50K inputs.

**Practical guidance:** Use RAG to pre-filter context below 10–30K tokens whenever possible. Reserve large context windows for tasks that genuinely need it (e.g., analyzing an entire codebase).

---

### Q3: How do you manage a conversation that exceeds the context window limit?

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

## 5. Pre-training

### Q1: What happens during LLM pre-training, and what does the model actually learn?

**Answer:**

Pre-training is the first and most expensive phase of building an LLM. The model learns to predict the next token given a sequence of tokens, trained on massive text corpora.

**Objective:** Next-token prediction (autoregressive language modeling). Given tokens [t1, t2, ..., t_n], predict t_{n+1}.

**Loss function:** Cross-entropy loss between predicted probability distribution and the actual next token.

**What the model learns:**
- Grammar, syntax, style across many languages
- World knowledge embedded in training data (facts, relationships, reasoning patterns)
- Code patterns across programming languages
- Long-range dependencies via attention mechanisms

**Scale of pre-training data:**
- GPT-3: 300 billion tokens (~570GB of text)
- GPT-4: Estimated 1–13 trillion tokens (undisclosed)
- Llama 3: 15 trillion tokens
- Common data sources: CommonCrawl, Wikipedia, GitHub, books, academic papers

**Compute cost (approximate):**
- GPT-3 (175B params): ~$4–12M in compute at 2020 prices
- GPT-4: Estimated $50–100M+
- Llama 3 70B: ~$1.5M on Meta infrastructure

**Result:** A **base model** that is an excellent text predictor but doesn't know how to be helpful, follow instructions, or avoid harmful outputs. That comes from fine-tuning and RLHF.

---

### Q2: What is the training lifecycle — pre-training → fine-tuning → RLHF — and what does each stage add?

**Answer:**

| Stage | Data | Objective | Result |
|---|---|---|---|
| Pre-training | Web-scale unlabeled text (trillions of tokens) | Next-token prediction | Base model: knows language and facts |
| Supervised Fine-Tuning (SFT) | High-quality instruction-response pairs (thousands to millions) | Imitate desired responses | Instruction-following model |
| RLHF / DPO | Human preference rankings between responses | Maximize human preference scores | Aligned, helpful, safe model |

**Analogy:** Pre-training is like a child reading every book ever written. SFT is like teaching them how to answer questions in a professional context. RLHF is like giving them feedback on which answers were helpful vs. annoying until they internalize the preference.

---

### Q3: What is tokenization and why does it matter for agents?

**Answer:**

**Tokenization** converts raw text into the integer token IDs the model processes. There is no single universal mapping — each model family has its own tokenizer.

**Common tokenization algorithms:**
- **BPE (Byte-Pair Encoding):** Iteratively merges the most common byte pairs. Used by GPT-2/3/4, Llama.
- **SentencePiece:** Language-agnostic subword tokenization. Used by T5, BERT, Gemma.
- **tiktoken:** OpenAI's fast BPE implementation.

**Why it matters for agents:**

1. **Cost:** You're billed per token, not per word. "ChatGPT" is 1 token. A UUID like "a4f2b7c1-9e3d" might be 8 tokens.

2. **Context limits:** Your 10,000-word document might be 13,000 tokens — you can't assume 1:1 mapping.

3. **Multilingual gotchas:** Non-English languages (especially CJK, Arabic) often tokenize very inefficiently. A 100-word Chinese sentence might be 300 tokens in a model not optimized for Chinese.

4. **Code tokenization:** Different models tokenize code differently. Python indentation spaces each count as tokens.

**Tool:** OpenAI's [Tokenizer](https://platform.openai.com/tokenizer) or the `tiktoken` library to count tokens before sending.

---

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

## 7. RLHF

### Q1: What is RLHF and how does it work step by step?

**Answer:**

**RLHF (Reinforcement Learning from Human Feedback)** is the process of aligning a language model's outputs to human preferences — making it more helpful, honest, and harmless.

**Step-by-step process:**

**Step 1: Supervised Fine-Tuning (SFT)**
- Start with a pre-trained base model.
- Fine-tune it on a dataset of high-quality human-written responses to prompts.
- Output: An SFT model that can follow instructions.

**Step 2: Reward Model Training**
- Collect **comparison data**: for the same prompt, show annotators two model responses and ask which is better.
- Train a separate **reward model (RM)** to predict human preference scores.
- The RM takes (prompt, response) as input and outputs a scalar reward score.

**Step 3: PPO Fine-tuning**
- Use **Proximal Policy Optimization (PPO)** — a reinforcement learning algorithm — to fine-tune the SFT model.
- For each prompt, the policy (language model) generates a response, the RM scores it, and PPO updates the model weights to maximize reward.
- A **KL divergence penalty** prevents the model from drifting too far from the SFT model (otherwise it learns to "game" the reward model with incoherent outputs that score high).

**Formula:**
```
Objective = E[reward_model(response)] - β × KL(policy || SFT_policy)
```

Where β controls how much the model is allowed to deviate from SFT behavior.

---

### Q2: What is DPO and how does it differ from RLHF?

**Answer:**

**DPO (Direct Preference Optimization)** achieves the same alignment goal as RLHF but without training a separate reward model or using RL.

**Key insight:** The optimal RLHF policy has a closed-form solution. DPO reparameterizes the RLHF objective so you can train directly on preference pairs using a simple supervised loss.

**DPO loss:**
```
L_DPO = -log σ(β × log(π(y_w | x) / π_ref(y_w | x)) - β × log(π(y_l | x) / π_ref(y_l | x)))
```

Where:
- `y_w` = preferred (winning) response
- `y_l` = rejected (losing) response
- `π` = current policy
- `π_ref` = reference (SFT) policy

**RLHF vs DPO:**

| Aspect | RLHF | DPO |
|---|---|---|
| Complexity | High (3 stages: SFT, RM, PPO) | Low (1 stage on preference pairs) |
| Training stability | Fragile (PPO is sensitive to hyperparameters) | More stable (supervised loss) |
| Reward model | Required (separate model) | Not needed |
| Performance | State of art (GPT-4, Claude) | Competitive, simpler to implement |
| Data format | Preference rankings | Same (chosen vs rejected pairs) |

**Real-world usage:** Llama 2, Mistral, and many open-source models use DPO or variants (IPO, KTO) because of its simplicity. GPT-4 and Claude 3 reportedly use RLHF.

---

### Q3: What is Constitutional AI and how does Anthropic use it?

**Answer:**

**Constitutional AI (CAI)** is Anthropic's approach to alignment, introduced in their 2022 paper. Instead of relying on human feedback for every output, it uses a set of written principles (a "constitution") to guide AI-generated feedback.

**Process:**

1. **Critique generation:** The model is prompted to critique its own output against constitutional principles (e.g., "Is this response harmful? Is it honest?").
2. **Revision:** The model revises its output based on its own critique.
3. **AI Feedback (RLAIF):** A preference model is trained on AI-generated preference data (not just human labels), using the constitution as the rubric.

**Why it matters:**
- Scales alignment without proportional human labeling costs.
- More transparent: principles are explicit and auditable.
- Reduces sycophancy: the model learns to prioritize honesty over making users feel good.

**Example constitutional principles:**
- "Choose the response that is less likely to contain false or misleading information."
- "Choose the response that is less harmful and more ethical."
- "Choose the response that is less racist, sexist, or otherwise discriminatory."

---

## 8. Structured Outputs

### Q1: What are structured outputs and why are they critical for agentic systems?

**Answer:**

**Structured outputs** constrain the model to generate responses that conform to a predefined schema — typically JSON, but also YAML, XML, or custom formats.

**Why agents need structured outputs:**

Agents don't read model responses as humans do. An agent needs to:
- Parse the response programmatically
- Extract tool call parameters
- Route decisions based on field values
- Compose responses into downstream systems

If the model outputs: "I think you should call the get_weather function with city=London" in free text, parsing this reliably is brittle. Structured output produces:
```json
{
  "tool": "get_weather",
  "parameters": {"city": "London"}
}
```

Which is directly parseable and type-safe.

---

### Q2: How do you implement structured outputs in the OpenAI and Anthropic APIs?

**Answer:**

**OpenAI — JSON mode (basic):**
```python
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[{"role": "user", "content": "Extract name and age from: John is 30 years old. Return JSON."}]
)
```
⚠️ JSON mode only guarantees valid JSON, not a specific schema.

**OpenAI — Structured Outputs with Pydantic (strict):**
```python
from pydantic import BaseModel
from openai import OpenAI

class PersonExtraction(BaseModel):
    name: str
    age: int
    email: str | None = None

response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": "Extract: John Smith, 30, john@example.com"}],
    response_format=PersonExtraction
)
person = response.choices[0].message.parsed  # Already a PersonExtraction object
```

**Anthropic — Tool use for structured outputs:**
```python
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[{
        "name": "extract_person",
        "description": "Extract person details from text",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
    }],
    messages=[{"role": "user", "content": "Extract from: John is 30 years old."}]
)
```

---

### Q3: What is Pydantic and why is it the standard for validation in AI pipelines?

**Answer:**

**Pydantic** is a Python data validation library that uses type annotations to define schemas and validate data at runtime.

**Why it's standard for AI pipelines:**
1. Schema definition is just Python type hints — no separate schema files.
2. Automatically validates types, required fields, and constraints.
3. Raises clear errors when validation fails — critical for catching LLM hallucinations that produce invalid JSON.
4. Integrates directly with FastAPI (request/response validation) and LangChain.

**Example: Validating LLM output:**
```python
from pydantic import BaseModel, field_validator
from typing import Literal

class TicketClassification(BaseModel):
    category: Literal["billing", "technical", "refund", "other"]
    priority: int  # Must be 1–5
    summary: str
    
    @field_validator("priority")
    def priority_must_be_valid(cls, v):
        if not 1 <= v <= 5:
            raise ValueError("Priority must be between 1 and 5")
        return v

# Parse and validate LLM JSON response
try:
    result = TicketClassification.model_validate_json(llm_response)
except ValidationError as e:
    # Retry with the error feedback in the prompt
    print(f"Validation failed: {e}")
```

**What to retry:** When validation fails, include the error message in the next prompt: "Your previous response failed validation: {error}. Please correct and try again." This is called **self-healing structured output**.

---

### Q4: What is the TOON format and how does it compare to JSON for AI use?

**Answer:**

**TOON (Token-Optimized Object Notation)** is a compact alternative to JSON designed to reduce token usage when passing structured data to/from LLMs.

**JSON (verbose):**
```json
{"name": "John", "age": 30, "city": "London"}
```
Approx. 42 characters / ~12 tokens.

**TOON (compact):**
```
name=John|age=30|city=London
```
Approx. 28 characters / ~8 tokens — ~33% fewer tokens.

**Trade-offs:**

| Aspect | JSON | TOON |
|---|---|---|
| Token efficiency | Lower | Higher (~20–40% savings) |
| Ecosystem support | Universal (all parsers) | Custom implementation needed |
| Nested structures | Native support | Awkward / less readable |
| LLM reliability | Models trained extensively on JSON | Less training data, lower reliability |
| Debugging | Easy | Harder |

**Senior engineer verdict:** TOON is interesting for high-volume, simple flat structures where token costs matter. For complex nested schemas (which agents typically need), JSON with strict schema validation is more reliable. Do not use TOON for schemas with deeply nested objects.

---

## 9. RAG Architecture & Evaluation

### Q1: What is RAG and what problem does it solve?

**Answer:**

**RAG (Retrieval-Augmented Generation)** is the pattern of retrieving relevant documents from an external knowledge store and injecting them into the LLM's context window at inference time, before generating a response.

**Problem it solves:**

LLMs have two fundamental limitations:
1. **Knowledge cutoff:** Training data has a cutoff date. The model doesn't know about events after that date.
2. **Hallucination:** Models fabricate plausible-sounding but incorrect information when they lack knowledge.

**RAG addresses both** by grounding the model's response in retrieved, verifiable documents.

**RAG pipeline components:**

```
User Query
    ↓
[Query Embedding]  ← Embed query into vector space
    ↓
[Vector Search]  ← Find top-k similar document chunks
    ↓
[Reranker (optional)]  ← Re-score retrieved chunks for relevance
    ↓
[Context Assembly]  ← Inject chunks into prompt
    ↓
[LLM Generation]  ← Generate response grounded in context
    ↓
[Response + Citations]
```

---

### Q2: How do embeddings work and how do you choose an embedding model?

**Answer:**

**Embeddings** convert text into dense numerical vectors that capture semantic meaning. Similar texts have vectors that are close in the embedding space (measured by cosine similarity or dot product).

**How they work:** An embedding model (usually a smaller transformer like BERT or a dedicated embedding model) processes text and outputs a fixed-size vector (e.g., 768, 1536, or 3072 dimensions). The training objective ensures that semantically similar texts produce similar vectors.

**Embedding model comparison:**

| Model | Dimensions | Max Tokens | Performance (MTEB) | Cost |
|---|---|---|---|---|
| OpenAI text-embedding-3-small | 1536 | 8191 | ~62.3% | $0.02/1M tokens |
| OpenAI text-embedding-3-large | 3072 | 8191 | ~64.6% | $0.13/1M tokens |
| Cohere embed-v3 | 1024 | 512 | ~64.5% | $0.10/1M tokens |
| BGE-large-en-v1.5 (OSS) | 1024 | 512 | ~63.5% | Free (self-hosted) |
| E5-mistral-7b (OSS) | 4096 | 4096 | ~66.6% | Free (self-hosted) |

**MTEB (Massive Text Embedding Benchmark)** is the standard benchmark. Higher = better retrieval.

**Key decision factors:**
- **Latency sensitive:** Use smaller models (text-embedding-3-small)
- **Quality critical:** Use larger models or E5-mistral
- **Data privacy:** Self-host OSS models (BGE, E5)
- **Multilingual:** Use Cohere embed-v3 or multilingual-E5

---

### Q3: What is reranking and when is it necessary?

**Answer:**

**Vector search** (approximate nearest neighbor) is fast but imprecise — it finds semantically similar chunks, not necessarily the most relevant ones for answering the specific query.

**Reranking** is a second-stage scoring pass that takes the top-k retrieved chunks (e.g., k=20) and rescores them using a more expensive but more accurate model, returning the top-n (e.g., n=5) for the prompt.

**How rerankers work:** A cross-encoder model takes (query, document) as a pair and outputs a relevance score. Unlike bi-encoders (used for embeddings), cross-encoders can model the interaction between query and document — much more accurate.

**When to use reranking:**
- When retrieval quality is the bottleneck in your RAG pipeline
- When you have a large document corpus (>100K chunks)
- When queries are complex or require reasoning to determine relevance

**Reranker options:**
- **Cohere Rerank:** $2/1K searches. Easy API integration.
- **BGE-reranker-large:** Free, self-hosted, ~SOTA open-source
- **ColBERT:** Token-level late interaction, excellent for long documents

**Numbers:** In practice, adding a reranker increases RAG answer accuracy by 10–20% on benchmarks like BEIR. The cost is added latency (~100–300ms) and API cost.

---

### Q4: What is a golden dataset and how do you evaluate RAG quality?

**Answer:**

A **golden dataset** is a curated set of (question, expected answer, source document) triples that represents the ground truth for evaluating your RAG system.

**How to build one:**
1. Sample 50–200 representative queries from production (or synthetic generation)
2. Have domain experts write ideal answers and cite the relevant source document
3. Include edge cases: ambiguous queries, queries with no good answer, multi-hop reasoning queries

**RAG evaluation metrics:**

| Metric | What it Measures | Tool |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? | Ragas, TruLens |
| **Answer Relevance** | Does the answer address the question? | Ragas |
| **Context Recall** | Did retrieval find the right documents? | Ragas, custom |
| **Context Precision** | Are retrieved docs relevant (no noise)? | Ragas |
| **Answer Correctness** | Is the answer factually correct vs. golden? | ROUGE, BERTScore, LLM-judge |

**Ragas RAG triad (the essential three):**
```
Faithfulness: answer is grounded in context  (0–1)
Answer Relevance: answer addresses the question (0–1)
Context Recall: right documents were retrieved (0–1)
```

**Regression testing:** Every time you change the embedding model, chunk size, retrieval k, or prompt, run the golden dataset and compare scores. A PR that drops faithfulness by >5% should be blocked.

**Chunk size impact (empirical benchmarks):**
- Chunk size 256 tokens: High precision, may miss context
- Chunk size 512 tokens: Best balance for most domains
- Chunk size 1024 tokens: Captures more context, more noise in retrieval

---

## 10. FastAPI Framework

### Q1: What is FastAPI and why is it the standard choice for AI backend services?

**Answer:**

**FastAPI** is a modern Python web framework for building APIs, built on top of Starlette (ASGI) and Pydantic.

**Why it dominates AI backends:**

1. **Native Pydantic integration:** Request and response schemas are Pydantic models — the same library used for LLM structured outputs. Zero friction.

2. **Async-first:** Built on ASGI (Asynchronous Server Gateway Interface). LLM API calls are I/O-bound — async lets you handle 100+ concurrent requests on a single server thread while awaiting model responses.

3. **Auto-generated docs:** FastAPI automatically generates OpenAPI (Swagger) and ReDoc documentation from your type hints. Critical for team collaboration.

4. **Performance:** Comparable to Node.js and Go for I/O-bound tasks (the bottleneck for LLM services is always the model latency, not the framework).

5. **Type safety end-to-end:** From HTTP request → Python business logic → LLM schema → response, everything is typed and validated.

---

### Q2: What does a production-grade FastAPI LLM service look like?

**Answer:**

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from openai import AsyncOpenAI
import asyncio

app = FastAPI(title="AI Classification Service")
client = AsyncOpenAI()

# Request schema
class ClassificationRequest(BaseModel):
    text: str
    max_tokens: int = 500

# Response schema (structured output)
class ClassificationResult(BaseModel):
    category: str
    confidence: float
    reasoning: str

@app.post("/classify", response_model=ClassificationResult)
async def classify_text(request: ClassificationRequest):
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Classify the text. Return JSON with category, confidence (0-1), reasoning."},
                {"role": "user", "content": request.text}
            ],
            max_tokens=request.max_tokens
        )
        import json
        result = json.loads(response.choices[0].message.content)
        return ClassificationResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check — always needed for Kubernetes/load balancer probes
@app.get("/health")
async def health():
    return {"status": "ok"}
```

**Key production considerations:**

| Concern | Solution |
|---|---|
| Rate limiting | `slowapi` or API gateway (Kong, nginx) |
| Authentication | OAuth2 / API keys via `Depends()` |
| Timeouts | `asyncio.wait_for(llm_call(), timeout=30)` |
| Retries | `tenacity` library with exponential backoff |
| Streaming | `StreamingResponse` + SSE for real-time output |
| Observability | OpenTelemetry / Langfuse middleware |

---

### Q3: How do you implement streaming responses in FastAPI for LLM outputs?

**Answer:**

Streaming is critical for user experience — users see text appear word-by-word instead of waiting 10+ seconds for the full response.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import asyncio

app = FastAPI()
client = AsyncOpenAI()

async def generate_stream(prompt: str):
    """Async generator that yields SSE-formatted chunks"""
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            # Server-Sent Events format
            yield f"data: {delta}\n\n"
    yield "data: [DONE]\n\n"

@app.get("/stream")
async def stream_response(prompt: str):
    return StreamingResponse(
        generate_stream(prompt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

**Time-to-First-Token (TTFT):** The key UX metric for streaming. Users perceive streaming as fast even if total generation time is the same, because they see output start in <1 second.

**TTFT benchmarks (approximate, vary by load):**
- GPT-4o: ~300–600ms TTFT
- Claude Sonnet: ~400–800ms TTFT
- Gemini 1.5 Pro: ~500–1000ms TTFT

---

## 11. Hands-On Exercises

### Exercise 1: Temperature Comparison

**Task:** Run the same prompt at temperatures 0.1, 0.5, and 1.0 five times each. Document output variance.

**Code:**

```python
from openai import OpenAI
import json

client = OpenAI()
prompt = "Write a one-sentence description of how machine learning works."

results = {}

for temp in [0.1, 0.5, 1.0]:
    outputs = []
    for run in range(5):
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=temp,
            messages=[{"role": "user", "content": prompt}]
        )
        outputs.append(response.choices[0].message.content)
    results[temp] = outputs

# Analysis
for temp, outputs in results.items():
    unique = len(set(outputs))
    print(f"\nTemp={temp}: {unique}/5 unique responses")
    for i, out in enumerate(outputs):
        print(f"  Run {i+1}: {out[:100]}...")
```

**Expected observations:**
- **T=0.1:** Nearly identical across all 5 runs. Same phrasing, same structure.
- **T=0.5:** Moderate variation in word choice. Same core meaning, different expressions.
- **T=1.0:** Noticeably different sentence structures across runs. Occasionally a run goes off in an unexpected direction.

**What to document:** Character count variance, semantic similarity (use cosine similarity of embeddings for rigorous comparison), and any factual inconsistencies at high temperature.

---

### Exercise 2: Token Economics Calculator

**Pricing as of early 2025 (verify at time of use — prices change):**

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| GPT-4o | $2.50 | $10.00 |
| GPT-4o mini | $0.15 | $0.60 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Claude 3 Haiku | $0.25 | $1.25 |
| Gemini 1.5 Pro | $1.25 (≤128K) | $5.00 (≤128K) |
| Gemini 1.5 Flash | $0.075 | $0.30 |

**Calculator code:**

```python
import tiktoken

PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
}

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Estimate tokens using tiktoken (GPT tokenizer as approximation)"""
    enc = tiktoken.encoding_for_model("gpt-4o")
    return len(enc.encode(text))

def estimate_cost(prompt: str, response: str, model: str) -> dict:
    input_tokens = count_tokens(prompt)
    output_tokens = count_tokens(response)
    
    pricing = PRICING[model]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
        "cost_per_1000_calls": round((input_cost + output_cost) * 1000, 4)
    }

# Example usage
sample_prompt = """You are a helpful assistant. 
User: Explain the difference between supervised and unsupervised learning in 3 sentences."""

sample_response = """Supervised learning trains models on labeled data where the correct output is known, 
learning to map inputs to outputs. Unsupervised learning finds patterns in unlabeled data without 
predefined correct answers, discovering hidden structure. The key difference is that supervised learning 
requires labeled training examples while unsupervised learning works with raw, unlabeled data."""

for model in PRICING:
    result = estimate_cost(sample_prompt, sample_response, model)
    print(f"{model}: ${result['total_cost_usd']:.6f} per call | ${result['cost_per_1000_calls']:.4f} per 1K calls")
```

**Expected output (approximate):**
```
gpt-4o: $0.000058 per call | $0.0578 per 1K calls
gpt-4o-mini: $0.000004 per call | $0.0036 per 1K calls
claude-3-5-sonnet: $0.000069 per call | $0.0690 per 1K calls
claude-3-haiku: $0.000006 per call | $0.0058 per 1K calls
gemini-1.5-pro: $0.000029 per call | $0.0290 per 1K calls
gemini-1.5-flash: $0.0000018 per call | $0.0018 per 1K calls
```

**Key insight:** For high-volume applications (millions of calls/day), the difference between GPT-4o ($2.50/1M) and Gemini Flash ($0.075/1M) is a **33x cost difference**. Model selection is an economic decision, not just a performance one.

---

## Quick Reference: Numbers to Memorize

| Fact | Value |
|---|---|
| Tokens to words ratio | ~1 token = 0.75 words |
| Claude 3.5 Sonnet context | 200K tokens |
| GPT-4o context | 128K tokens |
| Gemini 1.5 Pro context | 1M tokens |
| Temperature for tool use | 0.0–0.2 |
| Temperature for creativity | 0.8–1.2 |
| Top-p default | 0.9 |
| LoRA typical rank | 8–64 |
| Fine-tuning minimum examples | 50–200 (style), 500+ (domain) |
| GPT-4o input price | $2.50/1M tokens |
| Gemini Flash input price | $0.075/1M tokens |
| Reranker latency overhead | 100–300ms |
| Optimal RAG chunk size | 256–512 tokens |
| Golden dataset minimum size | 50 examples |

---
