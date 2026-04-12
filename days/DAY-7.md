# Day 7 — Build Your First Complete Agent: HITL, Observability & Polish

> **Theme:** Wire everything together — Human-in-the-Loop approval gates, conversation summarization, feedback collection, full observability, and an optional Gradio UI.

---

## Q1. What is the complete LangGraph graph topology for a production agent with HITL?

**Answer:**

A production agent graph has **9 nodes** connected by conditional edges. The key insight is that the approval gate is a _conditional branch_, not a node every call passes through.

```
[START]
   │
   ▼
[planner_node]          ← decides what to do next
   │
   ▼
[tool_selection_node]   ← picks which tool(s) to call
   │
   ├─ tool == note_tool ──► [approval_gate_node]  ← interrupt() fires here
   │                               │
   │                        approve / reject / edit
   │                               │
   └─ other tools ──────────────────┘
                                   │
                                   ▼
                         [tool_execution_node]   ← actually runs the tool
                                   │
                                   ▼
                         [response_node]         ← formats final answer
                                   │
                                   ▼
                         [feedback_node]         ← interrupt() for thumbs up/down
                                   │
                                   ▼
                               [END]
```

**Conditional edge logic (pseudo-Python):**

```python
def route_after_tool_selection(state: AgentState) -> str:
    pending = state.get("pending_tool")
    if pending and pending["name"] == "save_note":
        return "approval_gate"
    return "tool_execution"

graph.add_conditional_edges(
    "tool_selection",
    route_after_tool_selection,
    {"approval_gate": "approval_gate", "tool_execution": "tool_execution"}
)
```

**Trade-offs:**

| Design | Pro | Con |
|---|---|---|
| Approval gate as a node | Clean separation, easy to unit-test | Extra graph complexity |
| Inline approval in tool executor | Simpler graph | Mixes concerns, hard to re-route |
| Approval for all tools | Maximum safety | Friction — hurts UX |
| Approval only for destructive tools | Good UX | Must maintain a "dangerous tools" list |

**Key numbers:** `interrupt()` suspends the graph at that node and serializes state to the checkpointer. Resume requires calling `graph.invoke(None, config)` with the same `thread_id`.

---

## Q2. How do you implement the HITL approval node correctly in LangGraph?

**Answer:**

The approval node uses `interrupt()` to pause execution. It must: (1) show the user exactly what will be saved, (2) accept a structured decision, and (3) either proceed, abort, or mutate the pending tool call.

```python
from langgraph.types import interrupt

def approval_gate_node(state: AgentState) -> AgentState:
    pending = state["pending_tool"]

    # Show user exactly what will be saved
    decision = interrupt({
        "type": "approval_request",
        "tool": pending["name"],
        "content_preview": pending["args"].get("content", ""),
        "title_preview": pending["args"].get("title", "untitled"),
        "message": "About to save a note. Approve, reject, or provide edited content."
    })

    # decision comes back as: {"action": "approve" | "reject" | "edit", "edited_content": "..."}
    action = decision.get("action", "reject")

    if action == "reject":
        return {**state, "pending_tool": None, "approval_status": "rejected"}
    elif action == "edit":
        edited = decision.get("edited_content", pending["args"]["content"])
        updated_tool = {**pending, "args": {**pending["args"], "content": edited}}
        return {**state, "pending_tool": updated_tool, "approval_status": "approved_edited"}
    else:
        return {**state, "approval_status": "approved"}
```

**State schema additions needed:**

```python
from typing import TypedDict, Annotated, Optional, Literal
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    pending_tool: Optional[dict]           # tool call waiting for approval
    approval_status: Optional[Literal["approved", "rejected", "approved_edited"]]
    session_id: str
    token_count: int
    summary: Optional[str]
```

**Resuming after interrupt:**

```python
# Client side — after user clicks approve/reject
config = {"configurable": {"thread_id": session_id}}
graph.invoke(
    Command(resume={"action": "approve"}),  # or "reject" or "edit"
    config=config
)
```

**Real-world pitfall:** Never pass the full message history in the `interrupt()` payload — it bloats the checkpointer. Pass only the preview data the user needs to make a decision.

---

## Q3. How does conversation summarization work with tiktoken and what are the exact thresholds?

**Answer:**

Summarization prevents context overflow. The flow: count tokens → if over threshold → summarize → replace history with summary + tail messages.

```python
import tiktoken

SUMMARIZATION_THRESHOLD = 2000   # tokens
TAIL_MESSAGES_TO_KEEP = 4        # always keep last N messages raw

def count_tokens(messages: list, model: str = "gpt-4o") -> int:
    enc = tiktoken.encoding_for_model(model)
    total = 0
    for msg in messages:
        # 4 tokens overhead per message (role, separators)
        total += 4
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(enc.encode(content))
        elif isinstance(content, list):   # multi-modal messages
            for block in content:
                if block.get("type") == "text":
                    total += len(enc.encode(block["text"]))
    return total

def maybe_summarize(state: AgentState, llm) -> AgentState:
    messages = state["messages"]
    token_count = count_tokens(messages)

    if token_count <= SUMMARIZATION_THRESHOLD:
        return {**state, "token_count": token_count}

    # Keep tail raw; summarize everything before
    tail = messages[-TAIL_MESSAGES_TO_KEEP:]
    to_summarize = messages[:-TAIL_MESSAGES_TO_KEEP]

    existing_summary = state.get("summary", "")
    summary_prompt = f"""
    Existing summary (if any): {existing_summary}
    
    New conversation to add to summary:
    {format_messages_for_summary(to_summarize)}
    
    Write a concise factual summary preserving: key facts learned, decisions made, 
    user preferences, and tool results. Max 200 words.
    """

    response = llm.invoke([{"role": "user", "content": summary_prompt}])
    new_summary = response.content

    # Inject summary as a system message at position 0
    summary_message = {
        "role": "system",
        "content": f"[Conversation summary up to this point]: {new_summary}"
    }
    compressed_messages = [summary_message] + tail

    return {
        **state,
        "messages": compressed_messages,
        "summary": new_summary,
        "token_count": count_tokens(compressed_messages)
    }
```

**Why 2000 tokens?**

| Threshold | Trade-off |
|---|---|
| 500 tokens | Summarizes too aggressively — loses detail |
| 2000 tokens | Good balance for most chats — ~10–15 turns |
| 4000 tokens | Leaves room but gets expensive fast on GPT-4 |
| Never summarize | Context blows up; $$ cost + degraded performance |

**Key number:** tiktoken counts ~4 tokens of overhead per message plus content tokens. A typical user message is 20–50 tokens. 2000 tokens ≈ 30–50 messages.

---

## Q4. How do you implement the feedback collection node with SQLite persistence?

**Answer:**

The feedback node fires at the end of every turn. It uses `interrupt()` to capture a thumbs-up/down, then writes to SQLite with full context for later analysis.

```python
import sqlite3
import json
from datetime import datetime

# SQLite schema
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    query       TEXT NOT NULL,
    response    TEXT NOT NULL,
    rating      INTEGER NOT NULL,     -- 1 = thumbs up, -1 = thumbs down
    tool_used   TEXT,
    token_count INTEGER,
    latency_ms  INTEGER
);
"""

def feedback_node(state: AgentState) -> AgentState:
    # Interrupt to collect rating
    rating_input = interrupt({
        "type": "feedback_request",
        "message": "Was this response helpful? (👍 / 👎)",
        "options": ["thumbs_up", "thumbs_down", "skip"]
    })

    rating_action = rating_input.get("action", "skip")
    if rating_action == "skip":
        return state

    rating = 1 if rating_action == "thumbs_up" else -1

    # Extract last query and response from messages
    messages = state["messages"]
    last_user_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    last_ai_msg = next(
        (m["content"] for m in reversed(messages) if m["role"] == "assistant"), ""
    )

    # Persist to SQLite
    conn = sqlite3.connect("agent_feedback.db")
    conn.execute(CREATE_TABLE_SQL)
    conn.execute(
        """INSERT INTO feedback 
           (session_id, timestamp, query, response, rating, tool_used, token_count)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            state["session_id"],
            datetime.utcnow().isoformat(),
            last_user_msg,
            last_ai_msg,
            rating,
            state.get("last_tool_used"),
            state.get("token_count", 0),
        )
    )
    conn.commit()
    conn.close()

    return {**state, "last_rating": rating}
```

**Why SQLite and not a flat file?**

- Queryable: `SELECT AVG(rating), tool_used FROM feedback GROUP BY tool_used`
- Concurrent-safe (WAL mode)
- Zero-dependency — ships with Python
- Easy migration to Postgres later

**Closing the loop:** After collecting 50+ ratings, you can run:

```sql
SELECT tool_used, AVG(rating) as avg_score, COUNT(*) as n
FROM feedback
GROUP BY tool_used
ORDER BY avg_score ASC;
```

This tells you which tools produce the worst responses and need improvement.

---

## Q5. How do you build the observability decorator for LLM calls?

**Answer:**

A decorator wraps every LLM call transparently, capturing timing + token usage + cost without changing call sites.

```python
import time
import json
import functools
from datetime import datetime

# Per-token pricing (USD) — update when OpenAI/Anthropic change prices
PRICING = {
    "gpt-4o":            {"input": 2.50 / 1_000_000,  "output": 10.00 / 1_000_000},
    "gpt-4o-mini":       {"input": 0.15 / 1_000_000,  "output": 0.60  / 1_000_000},
    "claude-sonnet-4-5": {"input": 3.00 / 1_000_000,  "output": 15.00 / 1_000_000},
}

LOG_FILE = "logs/agent_calls.jsonl"

def observe_llm(model: str, tool_name: str = None):
    """Decorator factory for LLM call observability."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            error = None
            response = None

            try:
                response = func(*args, **kwargs)
            except Exception as e:
                error = str(e)
                raise
            finally:
                latency_ms = (time.perf_counter() - start) * 1000
                usage = getattr(response, "usage", None) if response else None

                input_tokens  = getattr(usage, "prompt_tokens",     0) if usage else 0
                output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                price = PRICING.get(model, {"input": 0, "output": 0})
                cost_usd = (input_tokens * price["input"]) + (output_tokens * price["output"])

                log_entry = {
                    "timestamp":     datetime.utcnow().isoformat(),
                    "model":         model,
                    "tool_name":     tool_name,
                    "input_tokens":  input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens":  input_tokens + output_tokens,
                    "cost_usd":      round(cost_usd, 8),
                    "latency_ms":    round(latency_ms, 2),
                    "error":         error,
                }

                with open(LOG_FILE, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            return response
        return wrapper
    return decorator

# Usage:
@observe_llm(model="gpt-4o-mini", tool_name="summarizer")
def call_summarizer_llm(messages):
    return openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    )
```

**Trade-offs:**

| Approach | Pro | Con |
|---|---|---|
| Decorator | Zero call-site changes, clean | Doesn't capture streaming tokens easily |
| Middleware/callback | Works with LangChain callbacks | Tightly coupled to framework |
| External (Langfuse, LangSmith) | Rich UI, team dashboards | External dependency, PII risk |
| Manual wrapping | Full control | Boilerplate everywhere |

**Why JSONL?** One JSON object per line. Append-only. `grep`, `jq`, and pandas all handle it natively. No database needed for logs.

---

## Q6. How do you build the log viewer that summarizes cost and latency?

**Answer:**

The log viewer reads the JSONL file and produces a human-readable summary across all sessions.

```python
import json
from collections import defaultdict
from pathlib import Path

def view_logs(log_file: str = "logs/agent_calls.jsonl"):
    entries = []
    path = Path(log_file)
    if not path.exists():
        print("No logs found.")
        return

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        print("Log file is empty.")
        return

    # Aggregate
    total_calls   = len(entries)
    total_tokens  = sum(e.get("total_tokens", 0) for e in entries)
    total_cost    = sum(e.get("cost_usd", 0.0) for e in entries)
    error_count   = sum(1 for e in entries if e.get("error"))

    # Per-tool breakdown
    by_tool = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0, "latencies": []})
    for e in entries:
        tool = e.get("tool_name") or "unknown"
        by_tool[tool]["calls"]    += 1
        by_tool[tool]["tokens"]   += e.get("total_tokens", 0)
        by_tool[tool]["cost"]     += e.get("cost_usd", 0.0)
        by_tool[tool]["latencies"].append(e.get("latency_ms", 0))

    print(f"\n{'='*50}")
    print(f"  AGENT OBSERVABILITY REPORT")
    print(f"{'='*50}")
    print(f"  Total LLM calls  : {total_calls}")
    print(f"  Total tokens     : {total_tokens:,}")
    print(f"  Total cost       : ${total_cost:.4f} USD")
    print(f"  Error count      : {error_count}")
    print(f"\n  Per-tool breakdown:")
    print(f"  {'Tool':<25} {'Calls':>6} {'Tokens':>8} {'Cost':>10} {'Avg Lat':>10}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*10} {'-'*10}")

    for tool, stats in sorted(by_tool.items()):
        avg_lat = sum(stats["latencies"]) / len(stats["latencies"])
        print(f"  {tool:<25} {stats['calls']:>6} {stats['tokens']:>8,} "
              f"${stats['cost']:>9.4f} {avg_lat:>8.1f}ms")

    print(f"{'='*50}\n")
```

**Sample output:**

```
==================================================
  AGENT OBSERVABILITY REPORT
==================================================
  Total LLM calls  : 23
  Total tokens     : 14,822
  Total cost       : $0.0089 USD
  Error count      : 0

  Per-tool breakdown:
  Tool                      Calls   Tokens       Cost    Avg Lat
  ------------------------- ------ -------- ---------- ----------
  planner                       8     4200   $0.0025   1842.3ms
  summarizer                    2      980   $0.0001    623.1ms
  responder                     8     5100   $0.0031   2103.4ms
  tool_router                   5     4542   $0.0032   1944.7ms
==================================================
```

---

## Q7. What are the 5 end-to-end test scenarios every agent must pass?

**Answer:**

These 5 scenarios exercise every major code path:

**Turn 1 — Web search:**
```
User: "What is LangGraph and when was it released?"
Expected: tool_selection routes to web_search, result incorporated, no HITL fires
Verify: logs show 1 planner call + 1 tool execution call
```

**Turn 2 — Note saving with HITL:**
```
User: "Save a note: LangGraph is a graph-based agent framework by LangChain"
Expected: approval_gate fires, user sees preview, approval → note saved
         rejection → agent says "Note not saved per your request"
Verify: approval_status in state = "approved", note appears in notes.json
```

**Turn 3 — Calendar lookup (no HITL):**
```
User: "What meetings do I have tomorrow?"
Expected: routes to calendar_tool directly, no interrupt
Verify: HITL checkpoint NOT triggered, response is fast
```

**Turn 4 — Memory recall:**
```
User: "What did you save about LangGraph earlier?"
Expected: long-term memory (ChromaDB) query returns the note from Turn 2
Verify: response mentions "graph-based agent framework" without re-calling web_search
```

**Turn 5 — Low confidence + feedback:**
```
User: "What will the weather be in 2035?"
Expected: agent flags uncertainty, gives honest answer, feedback node fires
         user gives thumbs down, rating -1 stored in SQLite
Verify: feedback table has 1 row with rating = -1
```

**Evaluation criteria checklist:**

| Criterion | How to verify |
|---|---|
| Tool routing correct by intent | Check `state["last_tool_used"]` matches intent |
| Memory persists across turns | ChromaDB returns result from earlier turn |
| HITL fires only for notes | `approval_status` only set on Turn 2 |
| Logs capture all decisions | `agent_calls.jsonl` has entry for every LLM call |
| Feedback stored in SQLite | `SELECT * FROM feedback` returns Turn 5 row |
| Summarization triggers at 2000 tokens | Force long conversation, check `state["summary"]` |

---

## Q8. How do you wrap the agent in a Gradio interface with dynamic HITL buttons?

**Answer:**

The Gradio UI needs to handle two async states: normal chat, and a paused state where approve/reject buttons appear.

```python
import gradio as gr
import threading

# Global state for pending HITL
pending_hitl = {"active": False, "thread_id": None, "preview": None}

def chat_handler(user_msg: str, history: list, state: dict):
    thread_id = state.get("thread_id", "session-1")
    config = {"configurable": {"thread_id": thread_id}}

    response_text = ""
    hitl_visible = False
    preview_text = ""

    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_msg}]},
        config=config,
        stream_mode="values"
    ):
        # Check if we hit an interrupt
        if "__interrupt__" in event:
            interrupt_data = event["__interrupt__"][0].value
            if interrupt_data.get("type") == "approval_request":
                pending_hitl["active"] = True
                pending_hitl["thread_id"] = thread_id
                pending_hitl["preview"] = interrupt_data
                preview_text = f"📝 Save note?\n\nTitle: {interrupt_data['title_preview']}\n\nContent:\n{interrupt_data['content_preview']}"
                hitl_visible = True
                response_text = "⏸ Waiting for your approval to save this note..."
                break

        # Extract latest assistant message
        msgs = event.get("messages", [])
        if msgs:
            last = msgs[-1]
            if hasattr(last, "content") and last.type == "ai":
                response_text = last.content

    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": response_text})

    return history, gr.update(visible=hitl_visible), preview_text, state

def approve_handler(state):
    if not pending_hitl["active"]:
        return [], gr.update(visible=False), ""
    config = {"configurable": {"thread_id": pending_hitl["thread_id"]}}
    graph.invoke(Command(resume={"action": "approve"}), config=config)
    pending_hitl["active"] = False
    return gr.update(visible=False), "✅ Note saved!"

def reject_handler(state):
    if not pending_hitl["active"]:
        return gr.update(visible=False), ""
    config = {"configurable": {"thread_id": pending_hitl["thread_id"]}}
    graph.invoke(Command(resume={"action": "reject"}), config=config)
    pending_hitl["active"] = False
    return gr.update(visible=False), "❌ Note rejected."

with gr.Blocks(title="Personal Research Assistant") as demo:
    gr.Markdown("## 🤖 Personal Research Assistant")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", height=500)
            msg_input = gr.Textbox(placeholder="Ask me anything...", label="Your message")
            send_btn = gr.Button("Send", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 📚 Retrieved Memory Facts")
            memory_display = gr.Markdown("_No facts retrieved yet_")

    # HITL panel — hidden by default
    with gr.Group(visible=False) as hitl_panel:
        gr.Markdown("### ⚠️ Approval Required")
        hitl_preview = gr.Markdown("")
        with gr.Row():
            approve_btn = gr.Button("✅ Approve", variant="primary")
            reject_btn  = gr.Button("❌ Reject",  variant="stop")

    session_state = gr.State({"thread_id": "session-1"})

    send_btn.click(chat_handler, [msg_input, chatbot, session_state],
                   [chatbot, hitl_panel, hitl_preview, session_state])
    approve_btn.click(approve_handler, [session_state], [hitl_panel, hitl_preview])
    reject_btn.click(reject_handler,  [session_state], [hitl_panel, hitl_preview])

demo.launch()
```

**Trade-offs:**

| UI choice | Pro | Con |
|---|---|---|
| Gradio Blocks | Fast to build, Python-native | Limited customization |
| Streamlit | Great for data apps | Doesn't handle async HITL well |
| FastAPI + React | Production-grade | 10x more code |
| Terminal (rich) | Zero dependencies | No good for approval flows |

---

## Q9. How do you add streaming so responses appear token by token?

**Answer:**

Streaming requires using `graph.stream()` with `stream_mode="messages"` and yielding chunks to the UI.

```python
# LangGraph streaming — yields message deltas
def stream_agent_response(user_msg: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    for msg_chunk, metadata in graph.stream(
        {"messages": [{"role": "user", "content": user_msg}]},
        config=config,
        stream_mode="messages"   # yields (AIMessageChunk, metadata) tuples
    ):
        if isinstance(msg_chunk, AIMessageChunk):
            yield msg_chunk.content   # partial text

# Gradio streaming integration:
def streaming_chat(user_msg, history):
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": ""})

    for chunk in stream_agent_response(user_msg, "session-1"):
        history[-1]["content"] += chunk
        yield history, ""   # yield updated history after each token

# In Gradio:
send_btn.click(streaming_chat, [msg_input, chatbot], [chatbot, msg_input])
```

**Key implementation detail:** When streaming is active and the graph hits an `interrupt()`, streaming stops. You must detect this and switch the UI to the HITL approval mode. The pattern is to check for `"__interrupt__"` in the metadata from `stream_mode="messages"`.

**Numbers:**
- Time-to-first-token (TTFT) with streaming: ~300–500ms (vs 2–4s for full response)
- Perceived latency improvement: ~80% better UX even at same total latency
- GPT-4o supports streaming natively; all major LLM providers do

---

## Q10. What does a complete project README for the Personal Research Assistant look like?

**Answer:**

A production-quality README answers four questions: what it does, how to install it, how to run it, and how it works.

```markdown
# 🤖 Personal Research Assistant

A fully agentic AI assistant with memory, tool use, HITL approval, 
conversation summarization, feedback collection, and full observability.

## Features
- **3 tools**: web search (Tavily), note saving, calendar lookup
- **HITL**: approval gate for note saving — you see exactly what will be saved
- **Memory**: ChromaDB long-term + tiktoken-based summarization (2000 token threshold)
- **Feedback**: thumbs up/down per turn, stored in SQLite
- **Observability**: JSONL logs with token counts, cost estimates, latency per call
- **UI**: Gradio chat interface with approve/reject buttons and memory sidebar

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/AniketWaichal/agentic-ai-playground
cd agent-design
pip install -r requirements.txt

# 2. Set environment variables
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...   # optional — mock search works without it

# 3. Run the agent (terminal mode)
python agent.py

# 4. Run with Gradio UI
python ui.py
# Open http://localhost:7860
```

## Architecture

```
User Input
    │
    ▼
Planner Node        ← GPT-4o-mini decides intent
    │
    ▼
Tool Selector       ← picks tool from [web_search, save_note, calendar]
    │
    ├─ save_note ──► Approval Gate (interrupt) ──► Tool Executor
    └─ other ──────────────────────────────────► Tool Executor
                                                       │
                                                  Response Node
                                                       │
                                               Feedback Node (interrupt)
```

## Running Tests

```bash
python tests/test_end_to_end.py   # runs all 5 scenarios
python logs/viewer.py             # view cost + latency summary
```

## Cost Estimate
~$0.01–0.05 per 10-turn conversation using gpt-4o-mini.
```

**What makes a README production-quality:**
1. Feature list answers "why should I use this"
2. Quick start works copy-paste in <2 minutes
3. Architecture section so contributors understand the code
4. Cost estimate sets expectations
5. No wall of text — use headers and code blocks

---

## Key Numbers for Day 7

| Metric | Value |
|---|---|
| Summarization threshold | 2000 tokens |
| Tail messages kept after summarization | 4 |
| Token overhead per message (tiktoken) | ~4 tokens |
| Typical 10-turn conversation cost (gpt-4o-mini) | ~$0.002 |
| Time-to-first-token improvement with streaming | ~80% perceived latency reduction |
| HITL nodes that use interrupt() | 2 (approval gate + feedback) |
| SQLite feedback columns | 8 (id, session, ts, query, response, rating, tool, tokens) |
| LangGraph resume call | `graph.invoke(Command(resume=decision), config)` |