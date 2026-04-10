# Day 6: Build Your First Complete Agent — Design & Setup

**Theme:** Before writing a single line of agent logic, design the full system architecture on paper. A well-structured agent is cheap to extend; a poorly structured one is expensive to fix.

---

## Quick-Reference Numbers

| Parameter | Value |
|---|---|
| Short-term memory limit (trigger summarization) | 2000 tokens |
| Long-term memory retrieval | Top-3 facts per turn |
| Embedding model | `text-embedding-3-small` |
| Vector DB (local) | ChromaDB or FAISS |
| Checkpointer | SqliteSaver |
| Log fields per LLM call | timestamp, prompt_tokens, completion_tokens, latency_ms, tool_invoked |
| Tool count (Day 6 agent) | 3 — web search, note-taking, calendar lookup |

---

## Q&A Notes

---

### Q1. What is the full architecture of a production-grade single agent?

**A:** A production-grade single agent has six layers. Designing these before writing code prevents the most common failure mode: a working prototype that cannot be extended or debugged.

```
┌─────────────────────────────────────────────────────┐
│                   USER INTERFACE                    │  CLI / Gradio / API
├─────────────────────────────────────────────────────┤
│                  AGENT ORCHESTRATOR                 │  LangGraph graph
│   Planning → Tool Selection → Execution → Response  │
├──────────────┬──────────────────────────────────────┤
│  TOOL        │  MEMORY LAYER                        │
│  REGISTRY    │  Short-term: conversation history    │
│  web_search  │  Long-term: ChromaDB vector store    │
│  note_take   │  State: TypedDict + SqliteSaver       │
│  cal_lookup  │                                      │
├──────────────┴──────────────────────────────────────┤
│                  HITL CHECKPOINT                    │  interrupt() on note-saving
├─────────────────────────────────────────────────────┤
│               OBSERVABILITY LAYER                   │  Logging every LLM call + tool
└─────────────────────────────────────────────────────┘
```

**The six layers and their responsibilities:**

| Layer | Responsibility | What breaks without it |
|---|---|---|
| Interface | Accepts user input, displays output | Cannot interact with the agent |
| Orchestrator | Routes between nodes, manages graph flow | Agent has no control flow |
| Tool Registry | Exposes callable functions with schemas | LLM cannot invoke external actions |
| Memory Layer | Persists context short-term and long-term | Agent forgets everything each turn |
| HITL Checkpoint | Pauses for human approval on sensitive actions | Agent acts autonomously on risky ops |
| Observability | Logs all decisions, tokens, latency | Cannot debug or cost-optimize |

---

### Q2. How do you design the component map before writing code?

**A:** Draw a data-flow diagram that answers three questions: what data flows in, what transforms it, and what flows out at each step.

```
User message
    │
    ▼
[Memory Retrieval] ← ChromaDB (embed query, fetch top-3 facts)
    │ facts injected into system prompt
    ▼
[LLM Call — ReAct] ← system prompt + history + facts + tools
    │ returns: Thought + Action (tool name + args) OR final answer
    ▼
[Tool Router] ─── web_search? ──► [Web Search Tool]
               ├── note_take?  ──► [HITL Gate] ──► [Note Tool]
               └── cal_lookup? ──► [Calendar Tool]
    │ tool result (Observation)
    ▼
[LLM Call — Synthesize] ← history + tool result
    │ final response
    ▼
[Memory Write] ──► ChromaDB (extract facts from this turn)
    │
    ▼
[Logging] ──► logs/agent.jsonl
    │
    ▼
[Feedback Collection] ──► feedback.db
    │
    ▼
User response
```

**Why draw this first:** Every arrow in this diagram is a function call or a data transformation you need to implement. Drawing it upfront reveals dependencies — e.g., memory retrieval must happen before the LLM call, not after.

---

### Q3. What should the state schema look like and why?

**A:** The state schema is the single source of truth for everything the agent knows at any point in the graph. Define it with TypedDict so LangGraph can validate it and the checkpointer can serialize it.

```python
from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Core conversation
    messages: Annotated[list, add_messages]   # Full message history
    session_id: str                            # Unique per conversation
    user_id: str                               # Links to long-term memory

    # Memory
    retrieved_facts: list[str]                 # Top-3 facts from ChromaDB this turn
    conversation_summary: Optional[str]        # Set when context > 2000 tokens

    # Tool execution
    pending_tool_call: Optional[dict]          # Tool name + args before execution
    tool_result: Optional[str]                 # Result after tool execution

    # HITL
    pending_note: Optional[str]                # Note awaiting human approval
    hitl_decision: Optional[str]               # "approved" | "rejected" | "edited"

    # Observability
    turn_log: list[dict]                       # Log entries for this turn
    total_tokens_used: int                     # Running total for cost tracking
```

**Design rules:**

| Rule | Why |
|---|---|
| Use `Annotated[list, add_messages]` for messages | LangGraph merges message lists correctly instead of overwriting |
| Keep `pending_*` fields for HITL | Makes the approval gate stateless — it just reads pending_note |
| Store `session_id` and `user_id` separately | session_id is per-conversation, user_id links to long-term memory across sessions |
| Include `turn_log` in state | Logs travel with the checkpoint — useful for debugging interrupted graphs |

---

### Q4. What is the correct project folder structure for a single agent?

**A:**

```
personal-research-assistant/
│
├── main.py                    # Entry point — run the agent
├── requirements.txt
├── .env                       # API keys — never commit
│
├── agents/
│   └── research_agent.py      # LangGraph graph definition
│
├── tools/
│   ├── __init__.py
│   ├── web_search.py          # web_search tool
│   ├── note_taking.py         # note_taking tool
│   └── calendar_lookup.py     # calendar_lookup tool
│
├── memory/
│   ├── __init__.py
│   ├── short_term.py          # Summarization logic
│   ├── long_term.py           # ChromaDB read/write
│   └── fact_extraction.py     # LLM-based fact extractor
│
├── prompts/
│   ├── system_prompt.py       # ReAct system prompt template
│   └── fact_extraction.py     # Fact extraction prompt
│
├── observability/
│   └── logger.py              # LLM call logger
│
├── logs/
│   └── .gitkeep               # Log files written here at runtime
│
├── data/
│   ├── notes.json             # Note storage (flat file for Day 6)
│   └── chroma_db/             # ChromaDB persistence directory
│
└── tests/
    ├── test_tools.py
    ├── test_memory.py
    └── test_agent_e2e.py
```

**Why this structure:** Each folder maps to exactly one architectural layer. When a bug occurs in memory retrieval, you open `memory/`. When a tool misbehaves, you open `tools/`. No layer bleeds into another.

---

### Q5. How do you implement the three core tools with proper schemas?

**A:** Each tool needs: a Python function, a schema (for OpenAI/Claude tool calling), and a description specific enough for the LLM to choose it correctly.

**Tool 1: Web Search (mock)**

```python
# tools/web_search.py
from langchain_core.tools import tool

MOCK_SEARCH_DB = {
    "python async": "Python async/await allows non-blocking I/O. Use asyncio.run() to start the event loop.",
    "langchain memory": "LangChain supports buffer memory, summary memory, and vector store memory.",
    "openai pricing": "GPT-4o costs $2.50 per 1M input tokens and $10.00 per 1M output tokens as of 2024.",
}

@tool
def web_search(query: str) -> str:
    """Search the web for current information on any topic.
    Use this when the user asks about facts, news, how-to guides,
    or anything that requires up-to-date information.
    Input: a search query string.
    Output: a text summary of the search results."""

    query_lower = query.lower()
    for key, result in MOCK_SEARCH_DB.items():
        if key in query_lower:
            return f"Search result for '{query}': {result}"
    return f"Search result for '{query}': No specific results found. General web search would return relevant articles."
```

**Tool 2: Note Taking**

```python
# tools/note_taking.py
import json
import os
from datetime import datetime
from langchain_core.tools import tool

NOTES_FILE = "data/notes.json"

@tool
def save_note(content: str, title: str = "") -> str:
    """Save an important note or piece of information for the user.
    Use this when the user explicitly asks to save, remember, or note something.
    Always get human approval before calling this tool.
    Input: content (the note text) and optional title.
    Output: confirmation with the note ID."""

    os.makedirs("data", exist_ok=True)
    notes = []
    if os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, "r") as f:
            notes = json.load(f)

    note = {
        "id": f"note-{len(notes)+1:04d}",
        "title": title or f"Note {len(notes)+1}",
        "content": content,
        "created_at": datetime.utcnow().isoformat()
    }
    notes.append(note)

    with open(NOTES_FILE, "w") as f:
        json.dump(notes, f, indent=2)

    return f"Note saved successfully. ID: {note['id']}, Title: '{note['title']}'"
```

**Tool 3: Calendar Lookup**

```python
# tools/calendar_lookup.py
from langchain_core.tools import tool
from datetime import datetime

MOCK_CALENDAR = {
    "today": [
        {"time": "10:00", "title": "Team standup", "duration": "30 min"},
        {"time": "14:00", "title": "Architecture review", "duration": "60 min"},
    ],
    "tomorrow": [
        {"time": "09:00", "title": "1:1 with manager", "duration": "30 min"},
        {"time": "15:00", "title": "Sprint planning", "duration": "90 min"},
    ],
    "this week": [
        {"day": "Wednesday", "time": "11:00", "title": "Product demo"},
        {"day": "Friday", "time": "16:00", "title": "Team retrospective"},
    ]
}

@tool
def calendar_lookup(time_period: str) -> str:
    """Look up the user's calendar events for a given time period.
    Use this when the user asks about their schedule, meetings, or upcoming events.
    Input: time_period — one of 'today', 'tomorrow', or 'this week'.
    Output: list of events with times and titles."""

    period = time_period.lower().strip()
    events = MOCK_CALENDAR.get(period, MOCK_CALENDAR.get("today"))

    if not events:
        return f"No events found for '{time_period}'."

    lines = [f"Events for {period}:"]
    for event in events:
        time = event.get("time", "")
        day = event.get("day", "")
        prefix = f"{day} {time}".strip() if day else time
        lines.append(f"  • {prefix} — {event['title']} ({event.get('duration', '')})")

    return "\n".join(lines)
```

**The schema quality rule:** The `"""docstring"""` IS the tool schema description. Write it as if you are instructing a junior engineer who has never seen your codebase. The LLM uses this to decide when to invoke the tool.

---

### Q6. How do you implement the ReAct prompting pattern in the system prompt?

**A:** ReAct (Reason + Act) structures the LLM's output as an explicit loop: Thought (reasoning) → Action (tool invocation) → Observation (tool result) → repeat until final answer.

```python
# prompts/system_prompt.py

REACT_SYSTEM_PROMPT = """You are a Personal Research Assistant. You help users with research, note-taking, and schedule management.

You have access to three tools:
- web_search: search the web for information
- save_note: save important information the user wants to keep
- calendar_lookup: check the user's calendar

## Reasoning Pattern

Always follow this pattern before responding:

Thought: [Reason about what the user needs and whether a tool is required]
Action: [Tool name and arguments, or "none" if no tool needed]
Observation: [Tool result — filled in automatically]
Thought: [Reason about the observation and what to do next]
Final Answer: [Your response to the user]

## Rules

1. Always start with a Thought before taking any Action.
2. Never call save_note without first confirming the content with the user.
3. If you don't know something, use web_search rather than guessing.
4. When calendar_lookup results are available, cite specific times and titles.
5. Keep Final Answer concise and directly useful.

## User Context (from memory)

{retrieved_facts}

## Current Date

{current_date}
"""

def build_system_prompt(retrieved_facts: list[str]) -> str:
    facts_text = "\n".join(f"- {fact}" for fact in retrieved_facts) if retrieved_facts else "No facts retrieved yet."
    return REACT_SYSTEM_PROMPT.format(
        retrieved_facts=facts_text,
        current_date=__import__("datetime").datetime.now().strftime("%A, %B %d, %Y")
    )
```

**Why include `retrieved_facts` in the system prompt, not the user message:** System prompt facts are treated as the agent's background knowledge. User message facts are treated as new input. Mixing them confuses the model's attention.

---

### Q7. How do you implement long-term memory with ChromaDB?

**A:** Long-term memory has two operations: write (extract facts from this turn and upsert) and read (embed the current query and retrieve top-k relevant facts).

```python
# memory/long_term.py
import chromadb
from chromadb.config import Settings
from openai import OpenAI

client = OpenAI()
chroma_client = chromadb.PersistentClient(path="data/chroma_db")

def get_collection(user_id: str):
    return chroma_client.get_or_create_collection(
        name=f"user_{user_id}",
        metadata={"hnsw:space": "cosine"}
    )

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def retrieve_facts(user_id: str, query: str, top_k: int = 3) -> list[str]:
    """Retrieve top-k relevant facts for the current query."""
    collection = get_collection(user_id)

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[embed(query)],
        n_results=min(top_k, collection.count())
    )

    return results["documents"][0] if results["documents"] else []

def upsert_facts(user_id: str, facts: list[dict]):
    """Write extracted facts to ChromaDB. Upsert by fact_id to avoid duplicates."""
    if not facts:
        return

    collection = get_collection(user_id)
    collection.upsert(
        ids=[f["id"] for f in facts],
        documents=[f["text"] for f in facts],
        embeddings=[embed(f["text"]) for f in facts],
        metadatas=[{"source": f.get("source", "conversation")} for f in facts]
    )
```

**The upsert pattern:** Using `upsert` instead of `add` means re-running the agent on the same conversation won't create duplicate facts. The `id` for each fact should be deterministic — e.g., a hash of the fact text.

---

### Q8. How do you write the fact extraction prompt?

**A:** Fact extraction is a separate LLM call that runs after every turn. It reads the latest user message and extracts durable facts — things that would still be true in future conversations.

```python
# memory/fact_extraction.py
import json
import hashlib
from openai import OpenAI

client = OpenAI()

FACT_EXTRACTION_PROMPT = """You are a fact extractor. Given a user message, extract any persistent facts about the user that would be useful to remember in future conversations.

Only extract facts that are:
- Durable (still true weeks from now): name, location, job, preferences, goals
- Specific (not vague): "user is a Python developer" not "user likes coding"
- About the user (not general knowledge)

Do NOT extract:
- Questions the user asked
- Temporary states ("user is tired today")
- General knowledge or opinions about the world

Respond ONLY with a JSON array. If no facts found, return [].

Format:
[
  {"id": "<hash>", "text": "<fact as a complete sentence about the user>"}
]

User message: {message}"""

def extract_facts(message: str) -> list[dict]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # Use cheaper model for extraction
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": FACT_EXTRACTION_PROMPT.format(message=message)
        }]
    )

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
        facts = data if isinstance(data, list) else data.get("facts", [])
    except json.JSONDecodeError:
        return []

    # Generate deterministic IDs from fact text
    for fact in facts:
        if "id" not in fact or not fact["id"]:
            fact["id"] = hashlib.md5(fact["text"].encode()).hexdigest()[:12]

    return facts
```

**Why use `gpt-4o-mini` for extraction:** Fact extraction is a simple classification task. Using the full GPT-4o would cost 10–15× more per call for no meaningful quality improvement on this task.

---

### Q9. How do you set up the observability logger?

**A:**

```python
# observability/logger.py
import json
import time
import os
from datetime import datetime
from functools import wraps

LOG_FILE = "logs/agent.jsonl"

def log_llm_call(prompt_tokens: int, completion_tokens: int,
                 latency_ms: float, tool_invoked: str = None,
                 model: str = "gpt-4o", node: str = ""):

    os.makedirs("logs", exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "model": model,
        "node": node,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_ms": round(latency_ms, 2),
        "tool_invoked": tool_invoked,
        "estimated_cost_usd": estimate_cost(model, prompt_tokens, completion_tokens)
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        "gpt-4o":       {"input": 2.50,  "output": 10.00},
        "gpt-4o-mini":  {"input": 0.15,  "output": 0.60},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    }
    rates = PRICING.get(model, {"input": 2.50, "output": 10.00})
    return round(
        (prompt_tokens / 1_000_000) * rates["input"] +
        (completion_tokens / 1_000_000) * rates["output"],
        6
    )

def timed_llm_call(func):
    """Decorator: wraps any LLM call function to auto-log timing and tokens."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        latency_ms = (time.time() - start) * 1000
        # Caller is responsible for passing token counts to log_llm_call
        # This decorator just measures wall-clock time
        result._latency_ms = latency_ms
        return result
    return wrapper
```

**Log format (JSONL — one JSON object per line):**

```json
{"timestamp":"2026-04-10T09:14:22.411Z","model":"gpt-4o","node":"planning","prompt_tokens":412,"completion_tokens":87,"total_tokens":499,"latency_ms":843.2,"tool_invoked":"web_search","estimated_cost_usd":0.001903}
```

JSONL is the right format for logs: each line is independently parseable, easy to append to, and can be streamed into analytics tools like Langfuse or BigQuery.

---

## Hands-On Deliverable Checklist

- [ ] Project structure created: agents/, tools/, memory/, observability/, logs/, data/, tests/
- [ ] All three tools implemented with proper `@tool` docstring schemas
- [ ] `AgentState` TypedDict defined with all required fields
- [ ] ReAct system prompt template written with `{retrieved_facts}` injection
- [ ] ChromaDB initialized with `PersistentClient` pointing to `data/chroma_db/`
- [ ] `retrieve_facts()` and `upsert_facts()` implemented and tested independently
- [ ] Fact extraction prompt tested on 5 sample messages — confirm it returns valid JSON
- [ ] Logger writing to `logs/agent.jsonl` with all 6 fields
- [ ] End-to-end test: run 3 turns, mention your name and location, restart the agent, ask "what do you know about me?" — facts should be recalled from ChromaDB

---

## Resources

- [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [ChromaDB Quickstart](https://docs.trychroma.com/getting-started)
- [LangChain Tools](https://python.langchain.com/docs/concepts/tools/)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)

---

> 📄 Deep dives: [`agent-design/`](../agent-design/)