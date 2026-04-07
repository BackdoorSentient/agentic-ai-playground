# 01 — Approval Workflows for Sensitive Agent Actions

---

## Q1. What makes an action "sensitive" and worth interrupting for?

**A:** An action is sensitive when the cost of being wrong exceeds the cost of human review. There are four categories:

| Category | Examples | Why sensitive |
|---|---|---|
| **Irreversible** | Delete record, send email, post to social | Cannot undo without external action |
| **Financial** | Process payment, issue refund, apply discount | Direct monetary loss if wrong |
| **Privacy** | Export PII, share data with third party | Legal/compliance risk |
| **Escalatory** | Escalate ticket, page on-call engineer | Triggers costly downstream processes |

**The rule of thumb:** If you'd want a human's signature on a physical form before doing this in the real world, you need an interrupt in the agent.

---

## Q2. How do you implement a robust approval workflow node in LangGraph?

**A:** A well-designed approval node has three responsibilities: risk classification, interrupt with structured context, and state update based on the human's decision.

```python
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Literal
from datetime import datetime
import uuid

class AgentState(TypedDict):
    messages: list
    pending_action: Optional[dict]
    approval_decision: Optional[Literal["approved", "rejected", "modified"]]
    audit_log: list

# --- Risk classifier ---
def is_high_risk(action: dict) -> bool:
    HIGH_RISK_TYPES = {"delete", "payment", "refund", "data_export", "bulk_update"}
    if action.get("type") in HIGH_RISK_TYPES:
        return True
    if action.get("type") == "refund" and action.get("amount", 0) > 50:
        return True
    return False

# --- Approval gate node ---
def approval_gate(state: AgentState) -> AgentState:
    action = state["pending_action"]

    if not is_high_risk(action):
        # Auto-approve low-risk actions — no interrupt
        return {
            "approval_decision": "approved",
            "audit_log": state["audit_log"] + [{
                "event": "auto_approved",
                "action": action,
                "timestamp": datetime.utcnow().isoformat()
            }]
        }

    # Present to human with full context
    human_response = interrupt({
        "type": "approval_request",
        "action": action,
        "risk_reason": classify_risk_reason(action),
        "options": ["approve", "reject", "modify"],
        "context": {
            "session_id": state["messages"][0].get("session_id"),
            "requested_at": datetime.utcnow().isoformat()
        }
    })

    decision = human_response.get("choice")
    modified_action = human_response.get("modified_action")

    log_entry = {
        "event": f"human_{decision}",
        "action": action,
        "operator_id": human_response.get("operator_id"),
        "timestamp": datetime.utcnow().isoformat()
    }

    if decision == "modify" and modified_action:
        return {
            "pending_action": modified_action,
            "approval_decision": "approved",  # Modified = approved with changes
            "audit_log": state["audit_log"] + [log_entry]
        }

    return {
        "approval_decision": decision,
        "audit_log": state["audit_log"] + [log_entry]
    }

def classify_risk_reason(action: dict) -> str:
    if action.get("type") == "delete":
        return "Irreversible data deletion"
    if action.get("type") in {"payment", "refund"}:
        return f"Financial action — ${action.get('amount', 0)}"
    return "High-risk action"

# --- Execution node ---
def execute_action(state: AgentState) -> AgentState:
    if state["approval_decision"] == "approved":
        # Do the actual work
        result = f"Action executed: {state['pending_action']}"
        return {"messages": state["messages"] + [{"role": "assistant", "content": result}]}
    else:
        return {"messages": state["messages"] + [{
            "role": "assistant",
            "content": "Action was rejected by the operator."
        }]}
```

---

## Q3. What should you show the human during an approval interrupt?

**A:** Show everything needed to make an informed decision — no more, no less.

**Required:**
- What action will be taken (type, target)
- The exact parameters (amount, user ID, record ID)
- Why the agent wants to do it (brief reasoning)
- What the options are (approve / reject / modify)

**Avoid:**
- Vague summaries ("the agent wants to do something")
- Technical jargon the operator won't understand
- Missing parameters that require the operator to look things up

**Example of a good interrupt payload:**

```python
interrupt({
    "type": "approval_request",
    "summary": "Process refund of $120.00 for order ORD-8821",
    "details": {
        "action": "process_refund",
        "order_id": "ORD-8821",
        "amount": 120.00,
        "currency": "USD",
        "customer_id": "CUST-4481",
        "reason": "Item arrived damaged — customer provided photo evidence"
    },
    "risk_level": "medium",
    "options": ["approve", "reject", "modify_amount"]
})
```

---

## Q4. How do you handle the case where no human responds (timeout)?

**A:** LangGraph interrupts have no built-in timeout — the graph stays suspended indefinitely until resumed. You need to build timeout handling at the orchestration layer.

**Pattern: scheduled timeout check**

```python
import asyncio
from datetime import datetime, timedelta

async def monitor_pending_approvals(graph, checkpointer, timeout_minutes=30):
    """Runs as a background task. Checks for stale interrupted threads."""
    while True:
        await asyncio.sleep(60)  # check every minute
        pending = checkpointer.list_interrupted_threads()

        for thread in pending:
            age = datetime.utcnow() - thread["interrupted_at"]
            if age > timedelta(minutes=timeout_minutes):
                # Auto-reject on timeout
                graph.invoke(
                    None,
                    config={"configurable": {"thread_id": thread["thread_id"]}},
                    command=Command(resume={"choice": "reject", "reason": "timeout"})
                )
```

**Alternative: use a dead-letter queue** — move timed-out approval requests to a separate queue for async human review, and continue the agent with a safe fallback.

---

## Q5. How do you build a full audit trail for compliance?

**A:** Every approval/rejection must be persisted with:

```python
import sqlite3
from datetime import datetime

def init_audit_db():
    conn = sqlite3.connect("audit.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS approval_audit (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            action_type TEXT,
            action_payload TEXT,
            decision TEXT,
            operator_id TEXT,
            operator_comment TEXT,
            requested_at TEXT,
            decided_at TEXT,
            latency_seconds REAL
        )
    """)
    conn.commit()
    return conn

def log_approval(conn, session_id, action, decision, operator_id,
                 requested_at, comment=None):
    import json
    conn.execute("""
        INSERT INTO approval_audit VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(uuid.uuid4()),
        session_id,
        action["type"],
        json.dumps(action),
        decision,
        operator_id,
        comment,
        requested_at.isoformat(),
        datetime.utcnow().isoformat(),
        (datetime.utcnow() - requested_at).total_seconds()
    ))
    conn.commit()
```

**What to track:**

| Field | Why |
|---|---|
| `operator_id` | Who approved — accountability |
| `latency_seconds` | How long humans take — SLA monitoring |
| `action_payload` | Exact params — for replay/audit |
| `decision` | What was decided |
| `requested_at` | When agent asked |
| `decided_at` | When human responded |

---

## Key Numbers

| Metric | Target |
|---|---|
| Approval latency (SLA) | < 5 minutes for P95 |
| Auto-approval rate | > 80% (means HITL is targeted correctly) |
| Timeout threshold | 30 min (domain dependent) |
| Audit log retention | 7 years for financial actions (varies by jurisdiction) |