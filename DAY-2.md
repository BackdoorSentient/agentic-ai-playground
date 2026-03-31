# Day 2: Prompt Engineering for Agents — Senior Engineer Q&A Notes

> **Format:** Each topic has a set of questions a senior engineer should be able to answer, with deep explanations, trade-offs, real-world examples, and key numbers.

---

## Table of Contents

1. [Zero-Shot & Few-Shot Prompting](#1-zero-shot--few-shot-prompting)
2. [Chain-of-Thought (CoT)](#2-chain-of-thought-cot)
3. [ReAct & Reflexion Patterns](#3-react--reflexion-patterns)
4. [System Prompt Design for Agents](#4-system-prompt-design-for-agents)
5. [Structured Outputs, JSON / YAML / TOON & Pydantic Validation](#5-structured-outputs-json--yaml--toon--pydantic-validation)
6. [Hands-On Exercises](#6-hands-on-exercises)

---

## Quick Reference: Numbers to Memorize

| Fact | Value |
|---|---|
| Few-shot examples sweet spot | 2–8 examples |
| Few-shot beyond which to fine-tune | 8+ examples |
| Self-consistency sample count | 5–10 chains, majority vote |
| CoT improvement on GSM8K (PaLM 540B) | 56.5% → 74.4% with self-consistency |
| Tree of Thought on Game of 24 | 4% (CoT) → 74% (ToT) |
| Reflexion on AlfWorld benchmark | 71% (ReAct) → 97% (Reflexion) |
| Prompt injection success rate (naive agent) | 60–80% |
| Prompt injection with full defenses | <5% |
| JSON vs TOON token savings | ~40–50% fewer tokens with TOON |
| Self-healing validation retry success rate | 70–80% on first retry |
| Streaming time-to-first-token | 200–500ms vs 2–5s non-streaming |
| ReAct max iterations (production) | 10–15 |
| ReAct per-iteration latency | 500ms–3s LLM call + tool time |
| TOON token cost saving at 10M calls/month vs JSON | ~$500/month on GPT-4o |

---

## 1. Zero-Shot & Few-Shot Prompting

> 📄 Full notes: [`prompting/01-zero-shot-and-few-shot.md`](prompting/01-zero-shot-and-few-shot.md)

**Core concept:** Zero-shot = direct instruction, no examples. Few-shot = 2–8 examples before the task. Both rely on the model's in-context learning ability.

**Key questions a senior engineer must answer:**
- When does zero-shot work vs. fail?
- How do you design diverse, balanced few-shot examples?
- What is the difference between zero-shot CoT and few-shot CoT?
- What are the production failure modes of few-shot prompting?

**Critical trade-off:** More examples = better accuracy but higher token cost at scale. Dynamic few-shot (retrieving examples from a vector DB based on similarity to the current query) is the production solution.

---

## 2. Chain-of-Thought (CoT)

> 📄 Full notes: [`prompting/02-chain-of-thought.md`](prompting/02-chain-of-thought.md)

**Core concept:** Force the model to generate intermediate reasoning steps before the final answer. Reasoning tokens condition the output, acting as working memory.

**Key questions a senior engineer must answer:**
- What is zero-shot CoT vs. few-shot CoT vs. self-consistency vs. Tree of Thought?
- What is the scratchpad pattern and how does it work in production?
- When does CoT hurt rather than help?
- How do you evaluate whether CoT is adding value?

**Critical trade-off:** CoT adds 200–1000ms latency and 2–5× token cost. Use it only when the task has more than 2 logical steps or when wrong answers are costly.

---

## 3. ReAct & Reflexion Patterns

> 📄 Full notes: [`prompting/03-react-and-reflexion.md`](prompting/03-react-and-reflexion.md)

**Core concept:** ReAct interleaves Thought → Action → Observation in a loop. Reflexion adds self-critique across multiple attempts, storing verbal reflections to improve on the next try.

**Key questions a senior engineer must answer:**
- What is the Thought → Action → Observation loop and why does it outperform pure CoT?
- How do you implement ReAct with max_iterations, error handling, and context management?
- How does Reflexion extend ReAct, and when is the added cost worth it?
- What are prompt injection attacks in agent systems and how do you defend against them?

**Critical trade-off:** Each ReAct iteration = 1 LLM call + tool latency. A 5-iteration agent can take 5–15 seconds. Reflexion multiplies this by the number of attempts.

---

## 4. System Prompt Design for Agents

> 📄 Full notes: [`prompting/04-system-prompt-design.md`](prompting/04-system-prompt-design.md)

**Core concept:** The system prompt is the agent's "constitution" — it defines identity, scope, tools, reasoning style, output format, and safety constraints. It is the single highest-leverage engineering decision in an agent system.

**Key questions a senior engineer must answer:**
- What are the 6 layers every production system prompt must cover?
- What are the 6 most common system prompt mistakes that break agents in production?
- How do you defend against prompt injection in the system prompt itself?
- When do you use few-shot examples inside the system prompt vs. a standalone system prompt?

**Critical trade-off:** The system prompt is paid on every single API call. A 2,000-token system prompt at GPT-4o pricing across 10M calls/month = $50,000/month. Write concisely.

---

## 5. Structured Outputs, JSON / YAML / TOON & Pydantic Validation

> 📄 Full notes: [`prompting/05-structured-outputs-and-validation.md`](prompting/05-structured-outputs-and-validation.md)

**Core concept:** Agents parse responses programmatically — free text breaks pipelines. Structured outputs (JSON schema enforcement) + Pydantic validation create reliable, type-safe agent pipelines.

**Key questions a senior engineer must answer:**
- What are the 3 structured output approaches (prompt-based, JSON mode, strict schema) and when do you use each?
- How do you implement self-healing validation with retry loops?
- When should you use JSON vs. YAML vs. TOON?
- How do you handle streaming structured outputs?

**Critical trade-off:** Strict structured outputs are most reliable but require supported models and add schema overhead. Prompt-based JSON is fragile but universally supported. In production, always use schema-enforced outputs where the model supports it.

---

## 6. Hands-On Exercises

### Exercise 1: ReAct Agent Simulation

Implement a prompt that forces the LLM to follow Thought → Action → Observation to solve a multi-step problem.

**Task:** "Find the current weather in Mumbai and recommend appropriate clothing."

**What to build:**
- A system prompt enforcing the ReAct format
- A mock tool registry: `search(query)`, `get_weather(city)`
- A parsing loop that extracts Action → executes tool → feeds Observation back
- A `max_iterations=10` guard

**What to document:**
- Does the agent correctly multi-hop (find city → get weather → recommend clothing)?
- What happens at the iteration limit?
- How does the agent handle tool errors?

---

### Exercise 2: Structured Output Validator

Build a prompt that reliably extracts structured data. Test at scale.

**Task:** Extract `{name, age, email, sentiment}` from 20 customer messages.

**What to build:**
```python
class CustomerExtraction(BaseModel):
    name: str
    age: int | None
    email: str | None
    sentiment: Literal["positive", "negative", "neutral"]

# Test with 20 varied inputs
# Measure: success rate, validation errors, retry effectiveness
```

**What to document:**
- Baseline success rate (no schema enforcement) vs. strict structured outputs
- Which inputs cause validation failures?
- Does the self-healing retry loop fix them?

---

### Exercise 3: TOON Experiment

Convert a 10-row dataset to JSON, YAML, and TOON. Compare token counts.

**What to build:**
```python
import tiktoken

data = [{"name": "Alice", "age": 30, "city": "Mumbai", "score": 87.5} for _ in range(10)]

# Encode in JSON, YAML, TOON
# Count tokens using tiktoken (cl100k_base)
# Measure: token count, LLM reliability (does the model parse each correctly?)
```

**What to document:**
- Token counts for each format
- % savings TOON vs JSON
- Whether the model reliably outputs each format without extra prompting

---

### Deliverable: Prompt Library

Add `prompting/prompt-library.md` to your repo with at least 5 reusable prompt templates covering:

1. Zero-shot classifier
2. Few-shot extractor with dynamic examples
3. ReAct agent system prompt
4. Structured output extractor with schema
5. Self-healing retry prompt template

---