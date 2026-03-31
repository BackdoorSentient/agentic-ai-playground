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

## 1. Product AI Patterns

> 📄 Full notes: [`../foundations/01-product-ai-patterns.md`](../foundations/01-product-ai-patterns.md)

**Core concept:** Every AI product decision starts with one question — should the AI **assist** (human reviews every output) or **automate** (AI executes end-to-end)? Getting this wrong is the #1 cause of AI demo syndrome.

**Key questions a senior engineer must answer:**
- What is the difference between Assist and Automate patterns, and how do you choose?
- What are the main UX flows in AI products and what failure modes must you design for?
- What is AI demo syndrome and how do you prevent it in production?

**Critical trade-off:** Automation maximizes throughput but amplifies errors at scale. A bug in an Assist system affects one user per mistake; a bug in an Automate system can fire 10,000 wrong emails before anyone notices. Always design with a confidence threshold gate for graceful degradation.

---

## 2. Temperature

> 📄 Full notes: [`../foundations/02-temperature.md`](../foundations/02-temperature.md)

**Core concept:** Temperature reshapes the probability distribution over the vocabulary at each token step. Low = deterministic and focused. High = creative and chaotic. It's a volume knob on randomness.

**Key questions a senior engineer must answer:**
- What is happening mathematically when you change temperature?
- What temperature values should you use for tool use vs. creative writing?
- What is the difference between temperature and top-k?

**Critical trade-off:** Higher temperature = more diverse outputs = harder to evaluate automatically. If your evaluation pipeline expects consistent format, high temperature breaks your parsers. For agentic tool use, always use 0.0–0.2.

---

## 3. Top-p (Nucleus Sampling)

> 📄 Full notes: [`../foundations/03-top-p.md`](../foundations/03-top-p.md)

**Core concept:** Instead of a fixed top-k tokens, top-p samples from the smallest set of tokens whose cumulative probability reaches threshold `p`. It adapts to the model's confidence at each step — top-k does not.

**Key questions a senior engineer must answer:**
- What is nucleus sampling and how does it differ from top-k?
- When would you set top-p = 1.0 vs top-p = 0.5?
- What happens if you set both temperature and top-p to non-default values simultaneously?

**Critical trade-off:** OpenAI and Anthropic both recommend adjusting temperature OR top-p — not both simultaneously. Compounding effects are hard to reason about and tune. Default: leave top-p at 0.9 and tune temperature only.

---

## 4. Context Window

> 📄 Full notes: [`../foundations/04-context-window.md`](../foundations/04-context-window.md)

**Core concept:** The context window is the maximum tokens an LLM can process in one inference call — it includes the system prompt, conversation history, retrieved documents, tool definitions, and the response being generated. Everything outside the window is invisible to the model.

**Key questions a senior engineer must answer:**
- What counts toward the context window limit?
- What are the practical limitations of large context windows (cost, latency, lost-in-the-middle)?
- How do you manage conversations that exceed the context window?

**Critical trade-off:** Large context ≠ free lunch. The "lost in the middle" problem (Stanford, 2023) shows models perform worst on information placed in the middle of a long context. Use RAG to pre-filter context below 10–30K tokens whenever possible.

---

## 5. Pre-training

> 📄 Full notes: [`../foundations/05-pretraining.md`](../foundations/05-pretraining.md)

**Core concept:** Pre-training teaches the model next-token prediction on web-scale corpora. The result is a base model that knows language and facts — but doesn't know how to be helpful, follow instructions, or avoid harmful outputs. That comes from fine-tuning and RLHF.

**Key questions a senior engineer must answer:**
- What does the model actually learn during pre-training?
- What is the full training lifecycle: pre-training → SFT → RLHF?
- What is tokenization (BPE, SentencePiece) and why does it matter for agents?

**Critical trade-off:** Pre-training is the most expensive phase by far (GPT-4 estimated $50–100M+). Fine-tuning and RLHF are cheap by comparison. You never pre-train from scratch — you always start from an existing base model.

---

## 6. Fine-tuning

> 📄 Full notes: [`../foundations/06-fine-tuning.md`](../foundations/06-fine-tuning.md)

**Core concept:** Fine-tuning updates model weights on a curated dataset to adapt behavior or style. Key rule: if you need the model to *know more facts*, use RAG. If you need it to *behave differently and consistently*, use fine-tuning.

**Key questions a senior engineer must answer:**
- When do you use fine-tuning vs. prompt engineering vs. RAG?
- What are LoRA and QLoRA, and why are they used?
- How much data do you need, and what are the common pitfalls?

**Critical trade-off:** Fine-tuning on too few examples (or too many epochs) causes overfitting — the model memorizes rather than generalizes. Quality >> quantity: 50 high-quality examples often outperform 1,000 noisy ones (OpenAI docs).

---

## 7. RLHF

> 📄 Full notes: [`../foundations/07-RLHF.md`](../foundations/07-RLHF.md)

**Core concept:** RLHF (Reinforcement Learning from Human Feedback) aligns a model to human preferences through three stages: SFT → Reward Model training → PPO fine-tuning. DPO is a simpler alternative that skips the reward model entirely.

**Key questions a senior engineer must answer:**
- What are the three steps of RLHF and what does each add?
- What is DPO and how does it differ from RLHF?
- What is Constitutional AI and how does Anthropic use it?

**Critical trade-off:** RLHF (used by GPT-4, Claude) achieves state-of-the-art alignment but is complex and fragile — PPO is sensitive to hyperparameters. DPO is simpler and more stable but slightly less performant. Most open-source models (Llama 2, Mistral) use DPO for this reason.

---

## 8. Structured Outputs

> 📄 Full notes: [`../foundations/08-structured-outputs.md`](../foundations/08-structured-outputs.md)

**Core concept:** Structured outputs constrain the model to generate responses conforming to a predefined schema (JSON, YAML). Agents parse responses programmatically — free text breaks pipelines. Schema enforcement + Pydantic validation = reliable, type-safe agents.

**Key questions a senior engineer must answer:**
- Why do agents need structured outputs and what breaks without them?
- How do you implement structured outputs in OpenAI vs. Anthropic APIs?
- What is Pydantic and how do you use it for self-healing validation?

**Critical trade-off:** Strict structured outputs (OpenAI `response_format`, Anthropic tool use) give 99%+ reliability but only work on supported models. Prompt-based JSON works everywhere but has 85–90% reliability. Always use schema-enforced outputs in production where the model supports it.

---

## 9. RAG Architecture & Evaluation

> 📄 Full notes: [`../foundations/09-rag-architecture-and-evaluation.md`](../foundations/09-rag-architecture-and-evaluation.md)

**Core concept:** RAG (Retrieval-Augmented Generation) grounds model responses in retrieved, verifiable documents — solving knowledge cutoff and hallucination problems. Pipeline: embed query → vector search → rerank → inject into prompt → generate.

**Key questions a senior engineer must answer:**
- What is the full RAG pipeline and what does each stage do?
- How do you choose an embedding model (MTEB benchmark, dimensions, cost)?
- What is reranking and when is it necessary?
- What is a golden dataset and how do you evaluate RAG quality (Faithfulness, Answer Relevance, Context Recall)?

**Critical trade-off:** Adding a reranker improves RAG accuracy by 10–20% but adds 100–300ms latency and API cost. Optimal chunk size is 256–512 tokens for most domains — larger chunks capture more context but introduce more noise in retrieval.

---

## 10. FastAPI Framework

> 📄 Full notes: [`../foundations/10-fastapi-framework.md`](../foundations/10-fastapi-framework.md)

**Core concept:** FastAPI is the standard Python framework for building AI backends — async by default, automatic OpenAPI docs, native Pydantic integration, and high performance. It's what you wrap your LLM calls in to serve agents as APIs.

**Key questions a senior engineer must answer:**
- What makes FastAPI better than Flask for AI workloads?
- How do you implement streaming LLM responses with FastAPI?
- How do you structure a production FastAPI app for an agent backend?

**Critical trade-off:** FastAPI's async model is a double-edged sword — it handles high concurrency well, but blocking calls (synchronous DB queries, non-async LLM SDKs) inside async routes will block the event loop and kill performance. Always use async-compatible libraries.

---

## 11. Hands-On Exercises

### Exercise 1: Temperature Comparison
Run the same prompt at temperatures 0.1, 0.5, and 1.0. Run each 5 times and document variance.

**What to observe:** At 0.1 — nearly identical outputs every run. At 1.0 — significant variation in wording, structure, and sometimes factual claims. Document which temperature produced the most useful output for your task.

---

### Exercise 2: Token Economics Calculator
Pick a realistic prompt + response pair (e.g., a support ticket classification). Estimate cost across:
- GPT-4o: $2.50 input / $10.00 output per 1M tokens
- Claude 3.5 Sonnet: $3.00 input / $15.00 output per 1M tokens
- Gemini 2.5 Pro: $1.25 input / $5.00 output per 1M tokens
- Gemini Flash: $0.075 input / $0.30 output per 1M tokens

**What to observe:** At 1M calls/month, the difference between the cheapest and most expensive model can be $50,000+/month. Use `tiktoken` to count tokens accurately before estimating.

---