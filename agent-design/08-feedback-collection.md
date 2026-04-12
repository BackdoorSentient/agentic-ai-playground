# Feedback Collection — Closing the Improvement Loop

> Deep dive: Day 7 | Topic: Explicit feedback collection, SQLite persistence, loop closure

---

## Q1. What is the taxonomy of agent feedback signals and which should you collect?

**Answer:**

Feedback signals fall into two categories: **explicit** (user deliberately rates) and **implicit** (user behavior reveals satisfaction).

**Explicit signals:**
| Signal | Mechanism | Reliability | Cost |
|---|---|---|---|
| Thumbs up/down | Button or interrupt() | High | Adds friction |
| 1–5 star rating | Widget | Medium | More friction |
| Free-text comment | Text input | Very high | High friction |
| Edit correction | User edits AI output | Very high | Zero friction |

**Implicit signals:**
| Signal | Mechanism | Reliability | Cost |
|---|---|---|---|
| Retry (re-sends same query) | Detect duplicate queries | Medium | Zero friction |
| Copy response | Track clipboard events | Medium | Requires UI hooks |
| Follow-up clarification | User asks "what do you mean?" | Medium | Requires NLP to detect |
| Session abandonment | User leaves mid-session | Low | Confounders (phone rang) |
| Tool approval rate | % of HITL approvals vs rejections | High | Already collected |

**For a Day 7 agent, collect:**
1. Explicit thumbs up/down per turn (via `interrupt()`)
2. HITL approval/rejection (already in the approval gate)
3. Retry detection (count duplicate queries in SQLite)

---

## Q2. How do you implement the full feedback node with SQLite?

**Answer:**

```python
import sqlite3
import json
from datetime import datetime, timezone

DB_PATH = "agent_feedback.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT    NOT NULL,
    turn_number   INTEGER NOT NULL,
    timestamp     TEXT    NOT NULL,
    query         TEXT    NOT NULL,
    response      TEXT    NOT NULL,
    rating        INTEGER NOT NULL,   -- 1=thumbs_up, -1=thumbs_down, 0=skipped
    tool_used     TEXT,               -- last tool called in this turn
    token_count   INTEGER,
    latency_ms    REAL,
    model         TEXT,
    extra_context TEXT                -- JSON blob for anything else
);

CREATE TABLE IF NOT EXISTS implicit_signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    signal_type TEXT    NOT NULL,   -- retry, abandonment, edit, etc.
    query       TEXT,
    details     TEXT                -- JSON blob
);
"""

def init_feedback_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()

def feedback_node(state: dict) -> dict:
    from langgraph.types import interrupt

    # Ask user for rating
    feedback_input = interrupt({
        "type":    "feedback_request",
        "message": "Was this response helpful?",
        "options": ["thumbs_up", "thumbs_down", "skip"],
    })

    action = feedback_input.get("action", "skip")
    rating_map = {"thumbs_up": 1, "thumbs_down": -1, "skip": 0}
    rating = rating_map.get(action, 0)

    if action != "skip":
        _persist_feedback(state, rating)

    # Implicit: detect if this is a retry
    _check_for_retry(state)

    return {**state, "last_rating": rating}

def _get_last_exchange(state: dict) -> tuple[str, str]:
    """Extract last user query and AI response from messages."""
    messages = state.get("messages", [])

    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"
         and not m.get("content", "").startswith("[Conversation summary")),
        ""
    )
    last_ai = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "assistant"),
        ""
    )
    return last_user, last_ai

def _persist_feedback(state: dict, rating: int):
    init_feedback_db()
    query, response = _get_last_exchange(state)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO feedback
           (session_id, turn_number, timestamp, query, response, rating,
            tool_used, token_count, model)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            state.get("session_id", "unknown"),
            state.get("turn_number", 0),
            datetime.now(timezone.utc).isoformat(),
            query,
            response,
            rating,
            state.get("last_tool_used"),
            state.get("token_count", 0),
            state.get("last_model_used", "unknown"),
        )
    )
    conn.commit()
    conn.close()

def _check_for_retry(state: dict):
    """Detect if the current query is a retry of the previous one."""
    query, _ = _get_last_exchange(state)
    session_id = state.get("session_id", "unknown")

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT query FROM feedback
           WHERE session_id = ? ORDER BY id DESC LIMIT 3""",
        (session_id,)
    ).fetchall()
    conn.close()

    recent_queries = [r[0] for r in rows]
    if query in recent_queries:
        _log_implicit_signal(session_id, "retry", query, {"recent": recent_queries})

def _log_implicit_signal(session_id: str, signal_type: str, query: str, details: dict):
    init_feedback_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO implicit_signals (session_id, timestamp, signal_type, query, details)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, datetime.now(timezone.utc).isoformat(),
         signal_type, query, json.dumps(details))
    )
    conn.commit()
    conn.close()
```

---

## Q3. How do you analyze feedback data to improve the agent?

**Answer:**

The feedback loop is only closed when data drives concrete changes. Here are the key queries and what to do with each result.

**Query 1: Per-tool satisfaction**

```sql
SELECT 
    tool_used,
    ROUND(AVG(rating), 3) AS avg_rating,
    COUNT(*) AS n,
    SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS thumbs_down
FROM feedback
GROUP BY tool_used
ORDER BY avg_rating ASC;
```

*Action:* Tool with lowest avg_rating needs its system prompt, output formatting, or error handling improved.

**Query 2: Which queries get retried?**

```sql
SELECT query, COUNT(*) AS retry_count
FROM implicit_signals
WHERE signal_type = 'retry'
GROUP BY query
ORDER BY retry_count DESC
LIMIT 20;
```

*Action:* These are your hardest queries. Add them to your few-shot examples or create a special handling path.

**Query 3: HITL edit rate (proxy for hallucination rate)**

```sql
SELECT 
    ROUND(100.0 * SUM(CASE WHEN action = 'approved_edited' THEN 1 ELSE 0 END) / COUNT(*), 1) AS edit_pct,
    ROUND(100.0 * SUM(CASE WHEN action = 'rejected' THEN 1 ELSE 0 END) / COUNT(*), 1) AS reject_pct
FROM hitl_audit;
```

*Action:* Edit rate > 20% means the LLM is generating notes the user disagrees with. Review the note-generation prompt.

**Query 4: Satisfaction over time (is the agent improving?)**

```sql
SELECT DATE(timestamp) AS day, AVG(rating) AS daily_avg
FROM feedback
GROUP BY DATE(timestamp)
ORDER BY day;
```

*Action:* Plot this. A flat or declining line after a prompt change means the change didn't help.

---

## Q4. How should you handle the friction of the feedback interrupt() in UX?

**Answer:**

The feedback `interrupt()` adds a step at the end of every turn, which can annoy users if not handled carefully.

**Mitigation strategies:**

**1. Make skip easy and the default:**
```python
interrupt({
    "type":    "feedback_request",
    "message": "Helpful? (press Enter to skip)",
    "options": ["👍", "👎", "skip"],
    "default": "skip",       # UI hint: default action on Enter
    "timeout_seconds": 8,    # Auto-skip after 8 seconds
})
```

**2. Collect feedback asynchronously (Gradio approach):**
Don't block the agent turn on feedback. Show thumbs buttons *below* the response, let the user rate at their leisure. Store with the session_id and turn_number so you can match later.

```python
# In Gradio: feedback buttons always visible, non-blocking
thumbs_up_btn.click(
    fn=lambda sid, turn: record_feedback(sid, turn, 1),
    inputs=[session_state, turn_counter],
)
```

**3. Sample feedback (don't ask every turn):**
```python
import random

def feedback_node(state):
    # Only ask for feedback 30% of turns
    if random.random() > 0.30:
        return state
    # ... rest of feedback logic
```

**Trade-off table:**

| Approach | Feedback volume | User friction | Implementation |
|---|---|---|---|
| interrupt() every turn | High | High | Simple |
| interrupt() 30% sample | Medium | Low | Simple |
| Async buttons (Gradio) | Medium | Zero | Requires UI |
| Implicit only | Low | Zero | Complex detection |

**Recommended for Day 7:** async buttons in Gradio. Zero friction, collects data when users care enough to click.

---

## Key Numbers

| Parameter | Value |
|---|---|
| Minimum ratings to draw conclusions | 50 per tool/node |
| Feedback ask frequency (sampling) | 30% of turns |
| SQLite WAL mode write throughput | ~5,000 writes/second |
| Useful edit_rate threshold | > 20% → fix the prompt |
| Retry rate threshold | > 15% of queries → address those cases |
| Auto-skip timeout (UX) | 8 seconds |