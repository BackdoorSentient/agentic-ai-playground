# Day 3: Memory & State Management — Senior Engineer Q&A Notes

> **Format:** Each topic has a set of questions a senior engineer should be able to answer, with deep explanations, trade-offs, real-world examples, and key numbers. Start here for the full-day overview, then dive into individual topic files for complete coverage.

---

## Table of Contents

1. [Memory Taxonomy](#1-memory-taxonomy)
2. [Conversation Memory with Summarization](#2-conversation-memory-with-summarization)
3. [State Schemas for Complex Agent Workflows](#3-state-schemas-for-complex-agent-workflows)
4. [Checkpointing for Durable Agent Execution](#4-checkpointing-for-durable-agent-execution)
5. [Hands-On Exercises](#5-hands-on-exercises)

---

## Quick Reference: Numbers to Memorize

| Fact | Value |
|---|---|
| GPT-4o context window | 128K tokens |
| Claude 3.7 Sonnet context window | 200K tokens |
| Gemini 1.5 Pro context window | 1M tokens |
| Lost-in-the-Middle accuracy drop | ~40% for facts in the middle of long contexts |
| Summarization trigger threshold (practical) | 2,000 tokens |
| Summarization compression ratio | 5–10× |
| Summarization call latency | 200–800ms |
| `text-embedding-3-small` cost | $0.02 / 1M tokens |
| ChromaDB practical limit | ~1M vectors |
| Memory injection overhead | 200–500 tokens / session |
| Pydantic validation overhead vs. TypedDict | ~10–30% |
| Recommended max_iterations for agent loop | 20–50 |
| Average LangGraph checkpoint size | 5–50KB |
| SqliteSaver max concurrent writers | 1 |
| Checkpoint data growth (1K sessions/day) | ~200MB/day |
| Words to tokens ratio (English) | ×1.3 |
| Top-K episodes to retrieve per session | 3–5 |
| Fine-tuning cost range | $100–$10,000 |

---

## 1. Memory Taxonomy

> 📄 Full notes: [`../memory/01-memory-taxonomy.md`](../memory/01-memory-taxonomy.md)

**Core concept:** AI agent memory has five types — short-term (context window), long-term (external store), episodic (event-indexed), semantic (concept-indexed), and procedural (behavioral/weights). Each has different storage, lifetime, and access patterns. Production agents combine all five.

**Key questions a senior engineer must answer:**
- What are the five memory types and how do they map to cognitive science?
- Why is the context window alone insufficient for multi-session agents?
- How does episodic memory differ from semantic memory architecturally?
- What is procedural memory and why is it the hardest to modify?
- How do you design a unified memory architecture for a production multi-session agent?

**Critical trade-off:** Short-term memory (context window) is zero-latency and perfectly accurate but expensive and ephemeral. Long-term memory (vector DB) is persistent and scales to infinite history but requires retrieval latency (~100–300ms) and an infrastructure dependency. The production answer is always a hybrid: recent turns in-context, older history in external storage retrieved on demand.

---

## 2. Conversation Memory with Summarization

> 📄 Full notes: [`../memory/02-conversation-memory-summarization.md`](../memory/02-conversation-memory-summarization.md)

**Core concept:** When conversation history exceeds a token threshold, you must compress it rather than truncate it. Summarization uses an LLM to produce a dense, lossless-ish representation of older turns. The production pattern: summarize turns beyond 2,000 tokens into a running summary, keep the last 2–4 turns verbatim. LangChain's `ConversationSummaryBufferMemory` implements this, but LangGraph's explicit state is better for complex agents.

**Key questions a senior engineer must answer:**
- What are the three strategies for handling context overflow, and when do you use each?
- How do you implement a summarization trigger with accurate token counting?
- What makes a good summarization prompt vs. a bad one?
- What are the production failure modes of summarization (loss, latency, drift)?
- How does LangChain's `ConversationSummaryBufferMemory` work, and what are its limits?

**Critical trade-off:** Summarization adds an LLM API call (200–800ms, ~$0.0001–0.0005) every time the threshold is crossed. Asynchronous summarization (trigger after delivering the response, apply to the next turn) eliminates user-visible latency entirely. Always run summarization async in production.

---

## 3. State Schemas for Complex Agent Workflows

> 📄 Full notes: [`../memory/03-state-schemas-agent-workflows.md`](../memory/03-state-schemas-agent-workflows.md)

**Core concept:** Agent state must be explicitly typed using `TypedDict` (LangGraph native) or Pydantic models. Untyped dicts cause silent failures, untraceable mutations, and no IDE support. LangGraph's `Annotated[type, reducer]` pattern controls how state fields merge across node updates — critical for message history (append) vs. stage flags (replace).

**Key questions a senior engineer must answer:**
- What goes wrong without a typed state schema?
- How do you design a `TypedDict` state schema for LangGraph with reducers?
- When do you use TypedDict vs. Pydantic, and what is the hybrid pattern?
- How do you design state for multi-agent (supervisor + subagent) workflows?
- How do you handle schema evolution when checkpointed state uses an old schema?

**Critical trade-off:** TypedDict is zero-overhead and LangGraph-native but provides no runtime validation — a wrong type silently flows through the graph. Pydantic adds 10–30% validation overhead but catches errors at the boundary. Use Pydantic for state fields populated from external data (user input, API responses); TypedDict for internal coordination fields. Additive-only schema changes are the safest migration strategy.

---

## 4. Checkpointing for Durable Agent Execution

> 📄 Full notes: [`../memory/04-checkpointing-durable-execution.md`](../memory/04-checkpointing-durable-execution.md)

**Core concept:** Checkpointing serializes full agent state to durable storage after every node execution. LangGraph's built-in checkpointers (`MemorySaver` → `SqliteSaver` → `PostgresSaver`) provide pause/resume, fault tolerance, HITL interrupts, and time-travel debugging with the same API. The `thread_id` is the key that scopes checkpoints to a conversation; resuming requires only the `thread_id` — LangGraph fetches the latest checkpoint automatically.

**Key questions a senior engineer must answer:**
- What does checkpointing enable, and what is the real cost of agents without it?
- What is the structure of a LangGraph checkpoint object?
- How do you implement SqliteSaver checkpointing with resume in production code?
- How do you implement Human-in-the-Loop (HITL) with `interrupt_before`/`interrupt_after`?
- When do you use SqliteSaver vs. PostgresSaver, and what are the scaling limits?
- How do you use `get_state_history` for time-travel debugging?

**Critical trade-off:** SqliteSaver is simple (single file, no infrastructure) but has a single-writer lock — not suitable for multiple concurrent agent processes. PostgresSaver requires a running PostgreSQL instance but handles arbitrary concurrency and is the right choice for any production deployment with >1 server process. Checkpoint overhead is 5–50ms per node — negligible against the 200ms–5s LLM call latency.

---

## 5. Hands-On Exercises

### Exercise 1: Conversation Summarizer

Build a chatbot that auto-summarizes history when it exceeds 2,000 tokens.

**What to build:**
```python
# Key components:
# 1. Token counter using tiktoken (accurate, model-specific)
# 2. ConversationSummaryBufferMemory OR manual summarization node
# 3. A trigger that fires when total tokens > 2,000
# 4. A summarization prompt that preserves: preferences, decisions, facts
# 5. Async summarization so the user sees no latency increase
```

**What to document:**
- Actual token counts before and after summarization (use tiktoken)
- Compression ratio achieved on 3 different conversation styles (technical, casual, multi-topic)
- Does the model retain user preferences stated before the summarization boundary?
- Latency comparison: synchronous vs. asynchronous summarization

---

### Exercise 2: Vector Memory Integration

Build a long-term memory system using ChromaDB that stores and retrieves user facts.

**What to build:**
```python
# Key components:
# 1. ChromaDB or FAISS collection for storing user facts
# 2. Embedding model: text-embedding-3-small (cheap, good)
# 3. Fact extraction LLM call: "Extract any user facts from this message"
# 4. Semantic retrieval: given new message, find top-3 relevant past facts
# 5. Memory injection: prepend retrieved facts to system prompt

# Example facts to store:
# "User prefers Python over JavaScript"
# "User works on backend systems, not frontend"
# "User has used FastAPI and Django"
# "User's name is Aniket"
```

**What to document:**
- Embedding costs for storing 100 user facts (use text-embedding-3-small pricing)
- Retrieval accuracy: does the right fact surface for the right query?
- Memory injection token overhead per session
- How you handle conflicting facts (e.g., user updates a preference)

---

### Exercise 3: State Checkpoint Demo

Build a LangGraph workflow with SqliteSaver that demonstrates interrupt + resume with full state recovery.

**What to build:**
```python
# Key components:
# 1. A multi-step agent graph (plan → execute → review → done)
# 2. SqliteSaver attached to persist all checkpoints
# 3. interrupt_before=["execute"] for HITL approval
# 4. A resume flow that works after process restart (kill the process, restart, resume)
# 5. get_state_history to display all checkpoints for a thread

# Demonstrate:
# - Agent starts, runs "plan" node, pauses before "execute"
# - Kill the process (simulating a crash)
# - Restart the process — load from checkpoint — resume execution
# - Show time-travel: list all checkpoints, load one from history
```

**What to document:**
- Checkpoint file size before and after 10 steps
- Time to restore from checkpoint (should be <100ms for SQLite)
- What happens to state when you call `app.update_state()` before resuming
- Schema for the SQLite checkpoints table (run `.schema` in sqlite3)

---

### Deliverable: Memory-Enabled Chatbot

Build a chatbot that persists user preferences across sessions using all techniques from Day 3:

**Architecture:**
```
Session Start:
  1. Load user profile from ChromaDB (semantic memory)
  2. Retrieve top-3 relevant past episodes from ChromaDB (episodic memory)
  3. Inject both into system prompt context
  4. Restore LangGraph checkpoint for this user_id (conversation state)

During Session:
  5. Manage context with ConversationSummaryBufferMemory (2,000 token threshold)
  6. Checkpoint after every turn via SqliteSaver

Session End:
  7. Extract new facts from the conversation (LLM extraction call)
  8. Update ChromaDB with new facts
  9. Embed and store the session summary as a new episode
```

**What to demonstrate:**
- Start a conversation, tell the chatbot your name and 3 preferences
- End the session (kill the process)
- Start a new session — the chatbot should remember your name and preferences without being told again
- Have a 20-turn conversation that triggers summarization
- Show the summarization boundary: what is preserved, what is compressed

---