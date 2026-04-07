# 03 — Mastering LangGraph's interrupt() for HITL Patterns

---

## Q1. What exactly is interrupt() and how does it differ from just pausing a loop?

**A:** `interrupt()` is not just a sleep or a polling loop. It is a first-class graph primitive that serializes the entire agent state, saves it to the checkpointer, raises a `GraphInterrupt` exception to halt execution, and returns control to the caller — all atomically.

**What makes it different from a polling loop:**

| Polling Loop | interrupt() |
|---|---|
| Thread stays alive, consuming resources | Thread is freed — no resource usage while waiting |
| State lives in memory — lost if process crashes | State persisted to disk/DB — survives crashes |
| No time-travel or replay | Full checkpoint — can replay from any point |
| Hard to scale to many concurrent waits | Scales to thousands of concurrent interrupted threads |
| Requires complex coordination code | Single function call |

**The execution model:**

```
graph.invoke() →
  node_A() →
  node_B() →
  approval_node() →
    interrupt(payload)         ← GraphInterrupt raised here
    ← state saved to checkpointer
  ← graph.invoke() returns {"__interrupt__": payload}

[human reviews and responds]

graph.invoke(None, config, Command(resume=response)) →
  approval_node() →            ← resumes at interrupt(), gets human_response
    human_response = <value>   ← continues from this line
  node_C() →
  END
```

---

## Q2. What are all the ways to use interrupt() — show me the full patterns?

**A:**

**Pattern 1: Binary Approve/Reject**

```python
from langgraph.types import interrupt

def approve_reject_node(state):
    decision = interrupt({
        "question": "Should I delete this record?",
        "record": state["target_record"],
        "options": ["approve", "reject"]
    })
    if decision["choice"] == "approve":
        return delete_record(state["target_record"])
    return {"status": "cancelled"}
```

**Pattern 2: Edit State**

```python
def edit_state_node(state):
    draft = generate_draft(state["input"])

    # Human can edit the draft before it's sent
    edited = interrupt({
        "type": "edit_request",
        "draft": draft,
        "instruction": "Review and edit this email draft. Return the final version."
    })

    final_draft = edited.get("final_draft", draft)  # use human version if provided
    return {"output": final_draft}
```

**Pattern 3: Review Tool Call Before Execution**

```python
def tool_review_node(state):
    # Agent has decided which tool to call — show it before executing
    pending_call = state["pending_tool_call"]

    approval = interrupt({
        "type": "tool_review",
        "tool_name": pending_call["name"],
        "parameters": pending_call["parameters"],
        "preview": f"Will call {pending_call['name']} with {pending_call['parameters']}"
    })

    if approval["proceed"]:
        result = execute_tool(pending_call)
        return {"tool_result": result}
    return {"tool_result": None, "status": "tool_call_rejected"}
```

**Pattern 4: Multi-step Wizard**

```python
def wizard_step_1(state):
    step1 = interrupt({"step": 1, "question": "What is the customer's name?"})
    return {"customer_name": step1["answer"]}

def wizard_step_2(state):
    step2 = interrupt({
        "step": 2,
        "question": f"Confirm order for {state['customer_name']}?",
        "order": state["pending_order"]
    })
    return {"confirmed": step2["confirmed"]}
```

---

## Q3. How do you set up the checkpointer — what are the options?

**A:** `interrupt()` requires a checkpointer. Without one, LangGraph cannot save state and will raise an error.

**Option 1: MemorySaver (development only)**

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
```

Lost on process restart. Use only for local testing.

---

**Option 2: SqliteSaver (local persistence)**

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = workflow.compile(checkpointer=checkpointer)
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Process refund"}]},
        config={"configurable": {"thread_id": "session-001"}}
    )
```

Persists across restarts. Good for single-server deployments.

---

**Option 3: PostgresSaver (production)**

```python
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg

conn_string = "postgresql://user:pass@host:5432/dbname"
with PostgresSaver.from_conn_string(conn_string) as checkpointer:
    checkpointer.setup()  # Creates tables on first run
    graph = workflow.compile(checkpointer=checkpointer)
```

Scales horizontally. Use for any production deployment.

---

## Q4. How do you resume a graph after an interrupt? Show the full invoke-resume cycle.

**A:**

```python
from langgraph.types import Command
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = workflow.compile(checkpointer=checkpointer)

    thread_config = {"configurable": {"thread_id": "thread-abc-123"}}

    # --- FIRST INVOKE ---
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "Refund $120 for order ORD-8821"}]},
        config=thread_config
    )

    # result will contain __interrupt__ if the graph paused
    if "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        print("Agent is waiting for approval:")
        print(interrupt_payload)

        # --- HUMAN REVIEWS ---
        # (In a real app, this would be a UI interaction)
        human_decision = {"choice": "approve", "operator_id": "OP-42"}

        # --- RESUME ---
        final_result = graph.invoke(
            None,  # No new input — just resuming
            config=thread_config,
            command=Command(resume=human_decision)
        )
        print("Final result:", final_result)
```

**Key detail:** The `thread_id` is what ties the resume to the right suspended graph. Every conversation must have a unique, stable `thread_id`.

---

## Q5. How do you handle multiple sequential interrupts in one graph run?

**A:** Each call to `interrupt()` creates a separate suspension point. The graph resumes from where it left off each time.

```python
def multi_checkpoint_agent(state):
    # First checkpoint
    step1 = interrupt({"step": 1, "question": "Confirm customer identity?"})
    if not step1["confirmed"]:
        return {"status": "identity_check_failed"}

    # Second checkpoint (only reached if first was approved)
    step2 = interrupt({
        "step": 2,
        "question": "Approve the $200 refund?",
        "customer": step1["customer_name"]
    })
    if not step2["approved"]:
        return {"status": "refund_rejected"}

    return process_refund(state)
```

**The resume cycle for two interrupts:**

```
invoke() → interrupt(step1) → suspend
resume(step1_decision) → interrupt(step2) → suspend
resume(step2_decision) → execute → END
```

Each resume call picks up exactly where the last interrupt left off.

---

## Q6. How do you inspect and debug interrupted graphs?

**A:**

```python
# List all interrupted threads
interrupted_threads = checkpointer.list(
    filter={"status": "interrupted"},
    config={"configurable": {}}
)

for thread in interrupted_threads:
    print(f"Thread: {thread.config['configurable']['thread_id']}")
    print(f"Interrupted at: {thread.metadata.get('created_at')}")

# Get state of a specific interrupted thread
state = graph.get_state(config={"configurable": {"thread_id": "thread-abc-123"}})
print("Current state:", state.values)
print("Next nodes:", state.next)  # Shows which node will run on resume
print("Interrupts:", state.tasks)  # Shows the interrupt payload

# Get full history (time-travel)
history = list(graph.get_state_history(
    config={"configurable": {"thread_id": "thread-abc-123"}}
))
for checkpoint in history:
    print(f"Step {checkpoint.metadata.get('step')}: {checkpoint.next}")
```

---

## Q7. What are common mistakes with interrupt() and how do you avoid them?

**A:**

| Mistake | Problem | Fix |
|---|---|---|
| Calling interrupt() inside a tool | Tools don't support suspension | Move interrupt to a node, not a tool |
| No checkpointer configured | Runtime error | Always compile with a checkpointer |
| Reusing thread_id across sessions | State from old session bleeds in | Use UUID-based thread_ids per session |
| Resuming with wrong thread_id | Graph not found or wrong state loaded | Store thread_id in your session layer |
| Not handling `__interrupt__` in the response | Caller doesn't know graph is waiting | Always check for `__interrupt__` key |
| interrupt() inside a conditional branch with no checkpointer | Silently skipped | Test interrupt paths explicitly |

---

## Key Numbers

| Detail | Value |
|---|---|
| Max pending interrupted threads | Checkpointer DB capacity (effectively unlimited) |
| State size per checkpoint | Typically 1–50KB (depends on message history) |
| Checkpoint write latency (Sqlite) | < 5ms |
| Checkpoint write latency (Postgres) | 5–20ms |
| Recommended thread_id format | `f"{user_id}-{session_uuid}"` |