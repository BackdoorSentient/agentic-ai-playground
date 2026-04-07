# Day 5: Human-in-the-Loop (HITL) & Interrupts

**Theme:** Keeping humans in the decision loop for sensitive, uncertain, or high-stakes agent actions.

---

## Quick-Reference Numbers

| Threshold | Rule of Thumb |
|---|---|
| Confidence < 0.7 | Escalate to human |
| Refund > $50 | Require approval |
| Data deletion | Always require approval |
| External API side-effects | Review before execution |
| Feedback sample rate | 100% for MVP, 10–20% in prod |

---

## Q&A Notes

---

### Q1. What is Human-in-the-Loop (HITL) and why do agents need it?

**A:** HITL is the design pattern where an autonomous agent pauses its execution and hands control back to a human before proceeding with a specific action. The agent doesn't stop permanently — it suspends state, waits for input, then resumes.

**Why agents need it:**

- LLMs are probabilistic. Even at temperature 0, they can hallucinate or misclassify.
- Some actions are **irreversible** — deleting data, sending emails, processing payments.
- Legal/compliance requirements mandate a human in certain decision chains (finance, healthcare).
- It is far cheaper to interrupt once than to recover from a wrong action.

**The core trade-off:**

| Without HITL | With HITL |
|---|---|
| Fully autonomous, fast | Slower, introduces latency |
| Higher risk on edge cases | Lower risk, higher trust |
| Harder to audit | Full audit trail |
| Better UX for simple tasks | Better for high-stakes tasks |

**Real-world example:** Stripe's fraud review pipeline uses automated scoring but routes transactions above a risk threshold to a human reviewer queue before the charge is declined. Pure automation would block legitimate transactions; pure human review wouldn't scale.

---

### Q2. What are the five core HITL patterns and when do you use each?

**A:**

**1. Approve / Reject**
- **Trigger:** Before irreversible or sensitive actions — payments, deletions, emails to customers.
- **Implementation:** `interrupt()` with a binary choice presented to the operator.
- **Example:** "Agent wants to process a $120 refund. Approve or Reject?"

**2. Edit State**
- **Trigger:** Agent output is partially correct but needs correction before it proceeds.
- **Implementation:** `interrupt()` returns the current state; human modifies it; agent resumes with the corrected state.
- **Example:** Agent drafts an email with wrong tone — human edits the draft, agent sends the corrected version.

**3. Review Tool Calls**
- **Trigger:** Before any external API call that has side effects (write, delete, send).
- **Implementation:** Agent shows the tool name + parameters; human approves before execution.
- **Example:** "About to call `delete_user(user_id=4821)`. Proceed?"

**4. Confidence Escalation**
- **Trigger:** When model uncertainty is high (confidence score < threshold, typically 0.7).
- **Implementation:** Classify the query, score confidence, route to human queue if below threshold.
- **Example:** Customer intent is ambiguous between billing and cancellation — route to agent instead of guessing.

**5. Feedback Collection**
- **Trigger:** After task completion.
- **Implementation:** Thumbs up/down + optional free text, stored for offline analysis and model improvement.
- **Example:** "Was this response helpful? 👍 👎"

---

### Q3. How does LangGraph's `interrupt()` work under the hood?

**A:** `interrupt()` is a special function in LangGraph that raises an internal exception (`GraphInterrupt`) at the exact line it's called. The graph execution is suspended, the full state is serialized and saved to the checkpointer (e.g., SqliteSaver or PostgresSaver), and control is returned to the caller.

When the human provides input, the graph is resumed by calling `.invoke()` again with a `Command(resume=<human_response>)`. LangGraph replays up to the interrupted node, injects the human's response as the return value of `interrupt()`, and continues execution.

```python
from langgraph.types import interrupt
from langgraph.graph import StateGraph

def sensitive_action_node(state):
    action = state["pending_action"]

    # Execution STOPS here — state is saved
    human_response = interrupt({
        "question": f"Approve this action?",
        "action": action,
        "options": ["approve", "reject", "modify"]
    })

    if human_response["choice"] == "approve":
        return execute_action(action)
    elif human_response["choice"] == "modify":
        return {"pending_action": human_response["modified_action"]}
    else:
        return {"status": "rejected"}
```

**Key mechanics:**

| Detail | Value |
|---|---|
| State persistence | Requires a checkpointer (SqliteSaver, PostgresSaver) |
| Resume mechanism | `graph.invoke(None, config, command=Command(resume=data))` |
| Thread isolation | Each conversation is a separate `thread_id` |
| Time-travel | You can re-run from any checkpoint |

**What you must NOT do:** Call `interrupt()` inside a tool — it must be inside a graph node. Tools are synchronous and don't support suspension.

---

### Q4. How do you design an approval workflow for sensitive agent actions?

**A:** Approval workflows follow a 4-step pattern: **detect → suspend → present → resume**.

```python
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

class AgentState(TypedDict):
    messages: list
    pending_action: Optional[dict]
    action_approved: Optional[bool]

def planning_node(state: AgentState) -> AgentState:
    # Agent decides what to do
    action = {"type": "refund", "amount": 120, "user_id": "U-4821"}
    return {"pending_action": action}

def approval_gate_node(state: AgentState) -> AgentState:
    action = state["pending_action"]

    # Only interrupt for high-risk actions
    if action["type"] == "refund" and action["amount"] > 50:
        decision = interrupt({
            "message": f"Approve ${action['amount']} refund for user {action['user_id']}?",
            "action": action
        })
        return {"action_approved": decision["approved"]}

    # Auto-approve low-risk actions
    return {"action_approved": True}

def execution_node(state: AgentState) -> AgentState:
    if state["action_approved"]:
        # execute the action
        return {"messages": [{"role": "assistant", "content": "Refund processed."}]}
    else:
        return {"messages": [{"role": "assistant", "content": "Action cancelled."}]}

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("plan", planning_node)
workflow.add_node("approval_gate", approval_gate_node)
workflow.add_node("execute", execution_node)
workflow.set_entry_point("plan")
workflow.add_edge("plan", "approval_gate")
workflow.add_edge("approval_gate", "execute")
workflow.add_edge("execute", END)
```

**Design principles:**

- Be selective — only interrupt when the action is irreversible or above a risk threshold. Every interrupt adds latency and human cost.
- Always show the exact parameters of the action, not a vague summary.
- Provide a modify option, not just approve/reject — humans often want to adjust, not block.
- Log every approval/rejection with timestamp and operator ID for compliance.

---

### Q5. How do you build confidence-based escalation?

**A:** Confidence escalation routes ambiguous agent decisions to a human queue rather than guessing. You need a confidence signal — this can come from the model's logprobs, a separate classifier, or a structured self-assessment.

**Pattern 1: Model self-assessment (simplest)**

```python
import json
from openai import OpenAI

client = OpenAI()

def classify_with_confidence(user_query: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{
            "role": "system",
            "content": """Classify the user intent. Respond with JSON only:
            {
              "intent": "billing | technical | refund | other",
              "confidence": 0.0-1.0,
              "reasoning": "brief explanation"
            }"""
        }, {
            "role": "user",
            "content": user_query
        }]
    )
    return json.loads(response.choices[0].message.content)

def confidence_router(state):
    query = state["messages"][-1]["content"]
    result = classify_with_confidence(query)

    if result["confidence"] < 0.7:
        # Escalate to human
        return {"route": "human_queue", "classification": result}
    else:
        return {"route": result["intent"], "classification": result}
```

**Pattern 2: Logprobs-based (more precise)**

```python
response = client.chat.completions.create(
    model="gpt-4o",
    logprobs=True,
    top_logprobs=5,
    messages=[...]
)

import math
# Get probability of the top token
top_logprob = response.choices[0].logprobs.content[0].logprob
confidence = math.exp(top_logprob)  # convert log-probability to probability
```

**Thresholds by use case:**

| Use Case | Escalate Below |
|---|---|
| Customer support routing | 0.75 |
| Medical triage | 0.95 |
| Legal document classification | 0.90 |
| E-commerce intent | 0.65 |

**Trade-off:** A lower threshold catches more edge cases but creates more human work. Calibrate using your golden test dataset — measure what % of escalations were actually ambiguous.

---

### Q6. How do you build a feedback loop that improves agent performance?

**A:** A feedback loop is a closed cycle: collect → store → analyze → improve → deploy → repeat.

**Step 1: Collect feedback in-line**

```python
def feedback_collection_node(state):
    # After the agent responds, ask for feedback
    feedback = interrupt({
        "type": "feedback_request",
        "response_id": state["last_response_id"],
        "question": "Was this response helpful?",
        "options": ["thumbs_up", "thumbs_down"],
        "optional_text": True
    })
    return {"feedback": feedback}
```

**Step 2: Store with full context**

```python
import sqlite3
from datetime import datetime

def store_feedback(session_id, query, response, rating, comment=None):
    conn = sqlite3.connect("feedback.db")
    conn.execute("""
        INSERT INTO feedback
        (session_id, timestamp, query, response, rating, comment)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, datetime.utcnow(), query, response, rating, comment))
    conn.commit()
```

**Step 3: Analyze offline**

- Cluster negative feedback by topic (embedding similarity).
- Identify prompts that consistently score low — these are your prompt improvement targets.
- Run the LLM-as-judge (Day 10) on negative examples to get structured failure categories.

**Step 4: Improve**

| Signal | Action |
|---|---|
| Low confidence + thumbs down | Add few-shot examples for this query type |
| Consistent misrouting | Adjust intent classifier prompt |
| Wrong tool selected | Improve tool descriptions |
| Correct answer, wrong format | Tighten output schema |

**The flywheel:** More users → more feedback → better prompts → better responses → more users.

**Real numbers:** Anthropic's Constitutional AI paper showed that RLHF from 10K high-quality preference pairs significantly outperforms larger datasets of lower quality. Prefer quality over volume in your feedback collection.

---

### Q7. What are the key trade-offs when deciding where to place HITL checkpoints?

**A:**

| Factor | Low HITL | High HITL |
|---|---|---|
| Latency | Fast (ms) | Slow (seconds to minutes for human) |
| Cost | Cheap per transaction | Expensive (human time) |
| Risk | Higher error rate on edge cases | Lower error rate |
| Trust | Lower user trust for high-stakes | Higher user trust |
| Scalability | Scales infinitely | Bottlenecked by human availability |

**Decision framework — interrupt when:**
1. The action is **irreversible** (delete, send, pay).
2. The action **cost > human review cost** if wrong.
3. **Confidence < threshold** (calibrate per domain).
4. The action involves **PII, money, or legal consequences**.
5. The agent has **never seen this pattern** before (cold-start).

**Don't interrupt when:**
- Action is read-only (queries, lookups).
- Confidence is consistently high on this query type in your eval data.
- Recovery is trivially easy (e.g., adding a to-do list item).

---

## Hands-On Deliverable Checklist

- [ ] Approval Workflow: Agent pauses before any action where `amount > $50` or `type == "deletion"`.
- [ ] Confidence Router: Classify intent, escalate if `confidence < 0.7`, log escalation reasons.
- [ ] Feedback Loop: Collect thumbs up/down + optional comment after each agent response, store to DB.
- [ ] At least **2 HITL checkpoints** in a single agent graph.
- [ ] Feedback stored with full context: query, response, rating, timestamp, session ID.

---

## Resources

- [LangGraph Human-in-the-Loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [LangGraph interrupt() reference](https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.interrupt)
- [Designing HITL AI Systems — Chip Huyen](https://huyenchip.com/2024/07/01/human-in-the-loop.html)

---

> 📄 Deep dives: [`hitl/`](../hitl/)