# HITL Approval Workflows — Complete Agent Integration

> Deep dive: Day 7 | Topic: Human-in-the-Loop approval gate for note saving

---

## Q1. What is the exact LangGraph `interrupt()` contract and how does resume work?

**Answer:**

`interrupt()` is a first-class LangGraph primitive that **suspends a graph mid-node**, serializes the full state to the checkpointer, and returns control to the caller. It is not an exception — it is a structured pause.

**The full lifecycle:**

```
graph.invoke(input, config)
    │
    ▼ node runs until interrupt() is called
    │
interrupt(value) ─────────────────────────────────► Returns to caller
    │                                                   │
    │                                              User sees payload
    │                                                   │
    ▼                                             User makes decision
State serialized to checkpointer                        │
    │                                                   │
    ◄───────────────────────────────────────────────────┘
    │
graph.invoke(Command(resume=decision), config)
    │
    ▼ node continues from AFTER the interrupt() call
```

**Code contract:**

```python
from langgraph.types import interrupt, Command

# Inside a node:
def approval_gate_node(state):
    # Everything before interrupt() runs before the pause
    preview = build_preview(state["pending_tool"])
    
    # interrupt() returns whatever the client passes to Command(resume=...)
    decision = interrupt({"preview": preview, "type": "approval_request"})
    
    # Everything after interrupt() runs AFTER resume
    # decision = {"action": "approve"} or {"action": "reject"} etc.
    return process_decision(state, decision)

# Client — initial call
result = graph.invoke({"messages": [...]}, config)
# result will be a dict with "__interrupt__" key if paused

# Client — resume after user decides
result = graph.invoke(
    Command(resume={"action": "approve"}),
    config  # MUST use the same thread_id
)
```

**Critical rule:** The `config` dict **must** contain the same `thread_id` on both the initial call and the resume call. If the thread_id changes, LangGraph creates a new thread and loses the suspended state.

---

## Q2. What should the approval payload show the user, and why does the content matter?

**Answer:**

The approval payload should give the user **exactly enough information to make a confident yes/no/edit decision** — no more, no less.

**Anatomy of a good approval payload:**

```python
interrupt({
    "type":            "approval_request",
    "tool":            "save_note",
    "title":           state["pending_tool"]["args"]["title"],
    "content":         state["pending_tool"]["args"]["content"],
    "tags":            state["pending_tool"]["args"].get("tags", []),
    "word_count":      len(state["pending_tool"]["args"]["content"].split()),
    "message":         "About to save the above note. Approve, reject, or edit.",
    "options":         ["approve", "reject", "edit"],
    "timestamp":       datetime.utcnow().isoformat(),
})
```

**What NOT to include in the payload:**

| Don't include | Why |
|---|---|
| Full message history | Bloats checkpointer, user doesn't need it |
| Internal state fields | Leaks implementation details |
| Tool call IDs | Not human-readable |
| Raw LLM prompt | Confusing and irrelevant |

**Why the content matters deeply:**

If the agent is about to save "LangGraph is a graph-based framework" but the user asked about something different, the approval preview catches the hallucination before it gets persisted. This is the core value of HITL — catching errors at the moment of consequence, not after.

---

## Q3. How do you handle all three approval outcomes: approve, reject, and edit?

**Answer:**

```python
def process_approval_decision(state: AgentState, decision: dict) -> AgentState:
    action = decision.get("action", "reject")
    pending = state["pending_tool"]

    if action == "approve":
        # Proceed with original tool call unchanged
        return {
            **state,
            "approval_status": "approved",
            # pending_tool stays as-is for tool_execution_node
        }

    elif action == "reject":
        # Clear the pending tool, add an assistant message explaining why
        rejection_msg = {
            "role": "assistant",
            "content": "Understood — I won't save that note. Let me know if you'd like to rephrase it or save something different."
        }
        return {
            **state,
            "pending_tool":    None,
            "approval_status": "rejected",
            "messages":        state["messages"] + [rejection_msg],
        }

    elif action == "edit":
        # User provided corrected content
        edited_content = decision.get("edited_content")
        edited_title   = decision.get("edited_title", pending["args"].get("title", "untitled"))

        if not edited_content:
            # Edit submitted empty — treat as reject
            return {**state, "pending_tool": None, "approval_status": "rejected"}

        updated_args = {
            **pending["args"],
            "content": edited_content,
            "title":   edited_title,
        }
        updated_tool = {**pending, "args": updated_args}

        return {
            **state,
            "pending_tool":    updated_tool,
            "approval_status": "approved_edited",
        }

    else:
        # Unknown action — safe default is reject
        return {**state, "pending_tool": None, "approval_status": "rejected"}
```

**Post-approval routing:**

```python
def route_after_approval(state: AgentState) -> str:
    status = state.get("approval_status")
    if status in ("approved", "approved_edited"):
        return "tool_execution"
    else:
        # rejected — skip tool execution, go straight to response
        return "response_node"

graph.add_conditional_edges(
    "approval_gate",
    route_after_approval,
    {"tool_execution": "tool_execution", "response_node": "response_node"}
)
```

---

## Q4. How do you wire the conditional edges so HITL only fires for the note tool?

**Answer:**

The routing logic lives in a single function that inspects `state["pending_tool"]`:

```python
def route_after_tool_selection(state: AgentState) -> str:
    """
    Route to approval gate for destructive/persistent tools.
    Route directly to execution for read-only tools.
    """
    pending = state.get("pending_tool")

    if pending is None:
        # No tool selected — go to response (agent decided to answer directly)
        return "response_node"

    # List of tools requiring human approval
    APPROVAL_REQUIRED_TOOLS = {"save_note", "delete_note", "send_email", "book_meeting"}

    if pending["name"] in APPROVAL_REQUIRED_TOOLS:
        return "approval_gate"

    return "tool_execution"

# Wire it:
graph.add_conditional_edges(
    "tool_selection",
    route_after_tool_selection,
    {
        "approval_gate": "approval_gate",
        "tool_execution": "tool_execution",
        "response_node":  "response_node",
    }
)
```

**Design principle: approval gates protect _writes_, not _reads_.**

| Tool | Needs approval? | Reason |
|---|---|---|
| web_search | No | Read-only, reversible |
| calendar_lookup | No | Read-only |
| save_note | **Yes** | Persistent write |
| delete_note | **Yes** | Irreversible |
| send_email | **Yes** | Irreversible, external effect |
| summarize_text | No | Stateless computation |

---

## Q5. How do you implement the full audit trail for HITL decisions?

**Answer:**

Every approval decision should be logged with enough context to answer: "What was about to be saved? Who approved it? When?"

```python
import json
import sqlite3
from datetime import datetime

CREATE_HITL_AUDIT_TABLE = """
CREATE TABLE IF NOT EXISTS hitl_audit (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL,
    timestamp      TEXT NOT NULL,
    tool_name      TEXT NOT NULL,
    action         TEXT NOT NULL,        -- approve / reject / approved_edited
    original_content TEXT,
    final_content  TEXT,                 -- may differ if edited
    user_id        TEXT,
    latency_ms     INTEGER               -- time user took to decide
);
"""

def log_hitl_decision(
    session_id: str,
    tool_name: str,
    action: str,
    original_content: str,
    final_content: str,
    latency_ms: int
):
    conn = sqlite3.connect("agent_feedback.db")
    conn.execute(CREATE_HITL_AUDIT_TABLE)
    conn.execute(
        """INSERT INTO hitl_audit 
           (session_id, timestamp, tool_name, action, original_content, final_content, latency_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (session_id, datetime.utcnow().isoformat(), tool_name,
         action, original_content, final_content, latency_ms)
    )
    conn.commit()
    conn.close()
```

**Useful audit queries:**

```sql
-- What fraction of notes get approved?
SELECT action, COUNT(*) as n, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) as pct
FROM hitl_audit
WHERE tool_name = 'save_note'
GROUP BY action;

-- How long do users take to approve? (proxy for trust/clarity)
SELECT AVG(latency_ms) / 1000.0 as avg_seconds
FROM hitl_audit WHERE action = 'approve';

-- Notes that got edited (agent generated wrong content)
SELECT original_content, final_content, timestamp
FROM hitl_audit WHERE action = 'approved_edited'
ORDER BY timestamp DESC LIMIT 10;
```

**The last query is gold** — it tells you exactly where the LLM is generating content the user disagrees with. Feed those pairs back into your few-shot examples or fine-tuning dataset.

---

## Key Numbers

| Parameter | Value | Why |
|---|---|---|
| Tools requiring HITL approval | Writes, irreversible, external effects | Protects user data |
| LangGraph resume call | `Command(resume=decision)` | Not `graph.invoke(decision, ...)` |
| Checkpointer required | Yes — MemorySaver or SqliteSaver | Suspend/resume impossible without it |
| thread_id must match | Strictly required | Different thread = new execution, state lost |
| Audit table columns | 9 | Enough context for full replay |