# Day 4: Tool Calling & Function Integration — Senior Engineer Q&A Notes

> **Format:** Each topic has a set of questions a senior engineer should be able to answer, with deep explanations, trade-offs, real-world examples, and key numbers. Start here for the full-day overview, then dive into individual topic files for complete coverage.

---

## Table of Contents

1. [Tool Schema Design](#1-tool-schema-design)
2. [Tool Calling Mechanics](#2-tool-calling-mechanics)
3. [Tool Selection Strategies](#3-tool-selection-strategies)
4. [Error Handling & Retry Logic](#4-error-handling--retry-logic)
5. [Hands-On Exercises](#5-hands-on-exercises)

---

## Quick Reference: Numbers to Memorize

| Fact | Value |
|---|---|
| Max tool name length (OpenAI) | 64 characters |
| Tool selection accuracy: 1–5 tools | ~99% |
| Tool selection accuracy: 10–20 tools | ~88–94% |
| Tool selection accuracy: 50+ tools in context | <70–72% |
| Accuracy after RAG or hierarchical routing | ~93–99% |
| All-in-context sweet spot | ≤10 tools |
| RAG retrieval overhead | 25–65ms |
| Optimal top-K for tool retrieval | 3–5 |
| text-embedding-3-small cost | $0.02 / 1M tokens |
| Classifier cost (gpt-4o-mini) | ~$0.00002 per call |
| Minimum API round trips per tool call | 2 |
| Parallel tool latency savings (3 independent tools) | ~63% |
| Self-correction rate on structured error feedback | 70–80% |
| Recommended max retries (transient failures) | 3 |
| Exponential backoff base delay | 1.0s |
| Simple REST API timeout | 3–5s |
| P95 user-acceptable wait | 15s |
| Circuit breaker failure threshold | 5 failures in 60s |
| Accuracy improvement from description rewrite alone | 72% → 94% (Stripe) |

---

## 1. Tool Schema Design

> 📄 Full notes: [`../tool-calling/01-tool-schema-design.md`](../tool-calling/01-tool-schema-design.md)

**Core concept:** A tool schema is a JSON contract between your code and the LLM — the model never sees your Python functions, only these definitions. The `description` field is the single most critical component: it drives tool selection accuracy. The `parameters` block defines what the model can pass. Key differences exist between OpenAI (`parameters`), Claude (`input_schema`), and LangChain (`@tool` decorator with auto-generated schema from docstrings).

**Key questions a senior engineer must answer:**
- What are the mandatory components of a tool schema across OpenAI, Claude, and open-source models?
- Why is the description the most critical field, and what does a production-quality description contain?
- How do you use enums, nested objects, arrays, and anyOf — and when does each matter?
- How do you write provider-agnostic tool definitions that work across multiple LLM providers?
- What are the 8 most common tool schema mistakes that cause production failures?

**Critical trade-off:** A rich, detailed description increases token cost per request (more tokens in the tool schema) but directly improves tool selection accuracy. At ≤10 tools, description quality is the dominant accuracy factor — a 200-token description upgrade costs ~$0.0005/1000 requests but can improve accuracy from 72% to 94%. Always invest in description quality before reaching for routing infrastructure.

---

## 2. Tool Calling Mechanics

> 📄 Full notes: [`../tool-calling/02-tool-calling-mechanics.md`](../tool-calling/02-tool-calling-mechanics.md)

**Core concept:** Tool calling is a minimum 2-turn protocol: (1) User → LLM: model returns a `tool_use` block instead of text. (2) Your code executes the tool. (3) Your code → LLM: returns tool result as a `tool_result` message. (4) LLM generates its final response. Parallel tool calling lets the model request multiple independent tools in one turn, cutting latency by up to 63% for multi-tool tasks. The `tool_use_id` must be echoed back exactly — mismatch causes silent failures or API errors.

**Key questions a senior engineer must answer:**
- What is the complete 4-step request/response cycle for a single tool call?
- What are all `stop_reason` / `finish_reason` values you must handle (and what happens if you don't)?
- How does parallel tool calling work, and how do you implement concurrent tool execution?
- How does LangGraph's `ToolNode` pattern simplify the execution loop?
- Why does `tool_use_id` mismatch cause silent failures, and what are the common causes?

**Critical trade-off:** Implementing the raw tool call loop (parsing blocks, matching IDs, building result messages) is error-prone. LangGraph's `ToolNode` abstracts this correctly and handles error formatting. Use `ToolNode` unless you need custom execution logic — the abstraction cost is zero and the bug-prevention value is high.

---

## 3. Tool Selection Strategies

> 📄 Full notes: [`../tool-calling/03-tool-selection-strategies.md`](../tool-calling/03-tool-selection-strategies.md)

**Core concept:** Tool selection accuracy collapses above ~20 tools in context (from ~99% to <70% at 50+). The solution scales with toolset size: all-in-context for <10 tools, categorized routing for 10–50, RAG-based semantic retrieval for 50–200, and hierarchical specialist agents for 200+. Each strategy adds a small routing overhead but recovers near-perfect selection accuracy. The "when NOT to use" clause in tool descriptions is the cheapest and most effective disambiguation technique.

**Key questions a senior engineer must answer:**
- Why does all-in-context selection fail at scale (attention dilution, token cost, semantic collision)?
- How do you implement categorized routing with a fast classifier model?
- How does RAG tool retrieval work, and what are the key parameters (embedding model, top-K, threshold)?
- When do you choose hierarchical agents over RAG routing?
- How do you handle cross-category queries and overlapping tool descriptions?

**Critical trade-off:** RAG retrieval adds 25–65ms overhead per request but reduces token cost by ~80–95% and recovers ~93–96% accuracy on large toolsets. For toolsets >20, RAG routing pays for itself in cost savings within days. For toolsets <10, pure description quality is a better ROI than routing infrastructure.

---

## 4. Error Handling & Retry Logic

> 📄 Full notes: [`../tool-calling/04-error-handling-retry-logic.md`](../tool-calling/04-error-handling-retry-logic.md)

**Core concept:** Tool failures fall into four categories — transient (retry with backoff), input validation (return structured error for LLM self-correction), authorization (escalate, don't retry), and business logic (pass result to LLM to reason about). Exponential backoff with full jitter prevents thundering herd on retry storms. Structured error responses (with `error_type`, `invalid_field`, and `expected_format`) enable 70–80% self-correction on first retry. Every tool must have an explicit timeout. The fallback hierarchy defines what the agent does when a tool is completely unavailable — always ending in a user-friendly message, never a raw exception.

**Key questions a senior engineer must answer:**
- What are the four categories of tool failure and why is each handled differently?
- How do you implement exponential backoff with full jitter, and why does jitter matter at scale?
- How do you format structured error responses that allow the LLM to self-correct?
- What is the three-level fallback hierarchy (primary → backup → cache → graceful fail)?
- How do you set tool timeouts, and what values are appropriate per tool type?
- What is the circuit breaker pattern and when do you use it?

**Critical trade-off:** Structured error handling requires more code than a simple `try/except` but is the difference between an agent that recovers from tool failures automatically and one that crashes or loops infinitely. The 70–80% self-correction rate means 3 in 4 tool failures resolve without user impact — invisible reliability that only structured errors enable.

---

## 5. Hands-On Exercises

### Exercise 1: Multi-Tool Agent

Build an agent with ≥3 tools that correctly selects the right tool per query.

**What to build:**
```python
# Tools to implement:
# 1. calculator(expression: str) → evaluate math safely with no dangerous builtins
# 2. get_weather(city: str, units: str = "celsius") → mock/real weather API
# 3. search_web(query: str, max_results: int = 3) → mock web search results

# Requirements:
# - Works with both Claude and OpenAI (provider-agnostic schema)
# - LangGraph ToolNode for execution
# - Parallel tool calling for independent queries
# - Tool call logging (name, args, result, latency_ms)

# Test queries:
# "What's the weather in Mumbai?"                         → get_weather
# "Calculate 15% tip on a ₹2,450 restaurant bill"       → calculator
# "Search for the latest news on agentic AI"            → search_web
# "What's 23 * 47 and the weather in Delhi?"           → parallel: calculator + weather
```

**What to document:**
- Tool selection accuracy across 10 test queries
- Latency: single tool vs. parallel tool calls
- Does the agent correctly avoid calling tools when the query doesn't need one?
- Token cost comparison: query with tools in context vs. without

---

### Exercise 2: Error Handling Lab

Implement retry logic, structured errors, and graceful degradation.

**What to build:**
```python
# Simulate failure modes:
# 1. Flaky tool: fails 50% of the time (random) → test retry with backoff
# 2. Invalid input: always fails with bad params → test structured error + LLM self-correction
# 3. Timeout: always hangs for 30s → test timeout handling
# 4. Unavailable: always fails → test fallback hierarchy

# Requirements:
# - retry_with_backoff decorator (3 retries, exponential, full jitter)
# - ToolError dataclass with error_type, field, expected_format
# - is_error: true in tool_result for Claude
# - User-facing error message at the end of fallback chain
# - Circuit breaker that opens after 5 consecutive failures

# Measure:
# - Self-correction rate: how often does structured error → correct retry call?
# - Retry latency distribution over 20 test runs
# - What % of failures are invisible to the user?
```

**What to document:**
- Self-correction rate for validation errors with vs. without structured error format
- Retry attempt distribution (how often does it succeed on attempt 1 vs. 2 vs. 3?)
- User message quality comparison: raw exception vs. graceful fallback message

---

### Exercise 3: Tool Tracing

Log all tool invocations and build a simple dashboard view.

**What to build:**
```python
# Trace schema to capture per tool call:
@dataclass
class ToolTrace:
    trace_id: str          # UUID
    session_id: str        # conversation ID
    timestamp: str         # ISO 8601
    tool_name: str
    tool_args: dict
    result: dict | None
    error: str | None
    latency_ms: float
    success: bool
    retry_count: int

# Requirements:
# - Wrap every tool call in the trace decorator
# - Persist traces to SQLite (simple, no infra)
# - Dashboard (terminal table or simple HTML):
#   - Total calls per tool
#   - Success rate per tool
#   - P50 / P95 latency per tool
#   - Error type distribution

# Dashboard output example:
# Tool              | Calls | Success | P50 (ms) | P95 (ms) | Top Error
# get_weather       |   42  |  95.2%  |   312    |   891    | TimeoutError
# calculator        |   28  |  100%   |    12    |    18    | —
# search_web        |   19  |  84.2%  |   748    |  2,341   | ConnectionError
```

**What to document:**
- Which tool has the highest failure rate in your test run?
- What is the P95 latency vs. P50 — is there high variance?
- At what tool error rate would you add a circuit breaker?

---

### Deliverable: Multi-Tool Agent with Error Handling and Logging

Combine all three exercises into a single production-ready agent:

**Architecture:**
```
User query
    ↓
[Tool Retrieval] — categorized routing if >5 tools
    ↓
[LangGraph Agent Loop]
    Agent node → ToolNode (with retry + timeout) → Agent node → ...
    ↓
[Tool Tracer] — logs every call with latency, success, error
    ↓
[Fallback Handler] — graceful degradation on repeated failures
    ↓
User response
```

**Success criteria:**
- Correctly routes 9/10 test queries to the right tool
- All tool failures are handled — no raw exceptions reach the user
- Tool traces stored and queryable
- Dashboard shows per-tool latency and success rate
- Agent handles parallel tool calls for independent sub-tasks