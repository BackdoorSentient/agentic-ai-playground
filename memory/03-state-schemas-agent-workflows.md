## 3. State Schemas for Complex Agent Workflows — Designing Typed State for Agents

### Q1: What is agent state, why does it need a schema, and what goes wrong without one?

**Answer:**

**Agent state** is the complete snapshot of everything an agent needs to continue its work: the conversation history, intermediate results, tool outputs, user preferences, workflow stage, and error flags.

Without a schema, agent state is typically a raw Python dict passed between functions:

```python
# Anti-pattern: untyped state dict
state = {
    "messages": [...],
    "user_name": "Aniket",
    "task_status": "in_progress",
    # ...any key can be added/removed silently
}
```

**What goes wrong without a schema:**

1. **Silent key errors:** A downstream function expects `state["user_preferences"]` but upstream forgot to set it. KeyError at runtime, 3 hours into a long-running agent task.

2. **Type mismatches:** One function sets `state["score"] = "87"` (string), another expects an integer and crashes on `state["score"] + 10`.

3. **Untraceable state mutations:** Any function can overwrite any key. Debugging who corrupted the state requires tracing the entire execution graph.

4. **No IDE support:** No autocomplete, no type checking, no documentation of what fields exist.

5. **No checkpoint schema:** If you want to serialize state for fault recovery, untyped dicts require custom serialization logic per agent.

**The solution:** typed state schemas using Python TypedDict (LangGraph's approach) or Pydantic models.

---

### Q2: How do you design a typed state schema with TypedDict for LangGraph?

**Answer:**

LangGraph uses `TypedDict` as its state schema. Every node in the graph receives the full state and returns a dict of fields to update:

```python
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    # Conversation history — uses operator.add as reducer
    # (new messages are appended, not replaced)
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # Current workflow stage
    stage: Literal["planning", "executing", "reviewing", "done"]
    
    # Tool call results
    tool_outputs: list[dict]
    
    # User preferences (persisted across summarization)
    user_preferences: dict[str, str]
    
    # Error tracking
    error_count: int
    last_error: str | None
    
    # Iteration guard
    iteration: int
```

**The `Annotated` pattern for reducers:** By default, LangGraph replaces a state field when a node returns a new value. The `Annotated[type, reducer]` pattern lets you specify *how* updates merge:

```python
# Default: replace
stage: str  # node returns {"stage": "done"} → state["stage"] = "done"

# Append (for message history)
messages: Annotated[list, operator.add]  # new messages appended to existing list

# Custom reducer (e.g., deduplicate tool outputs)
def merge_tool_outputs(existing: list, new: list) -> list:
    existing_ids = {o["id"] for o in existing}
    return existing + [o for o in new if o["id"] not in existing_ids]

tool_outputs: Annotated[list, merge_tool_outputs]
```

---

### Q3: When should you use TypedDict vs. Pydantic for state schemas, and what are the trade-offs?

**Answer:**

| Dimension | TypedDict | Pydantic BaseModel |
|---|---|---|
| Runtime validation | ❌ None — type hints only | ✅ Full validation on assignment |
| Performance | ✅ Zero overhead (just a dict) | ~10–30% overhead for validation |
| LangGraph compatibility | ✅ Native — LangGraph is built on it | ⚠️ Needs adapter or `model_dump()` |
| Nested model validation | ❌ No | ✅ Yes |
| Default values | ✅ Via `total=False` | ✅ Via `Field(default=...)` |
| Serialization | ✅ JSON-native | ✅ `.model_dump()` |
| IDE autocomplete | ✅ Yes | ✅ Yes |

**Use TypedDict when:**
- Working with LangGraph (native schema type)
- State is flat or shallowly nested
- Performance matters (hot path with many state updates)
- You trust your node implementations to set correct types

**Use Pydantic when:**
- State fields come from external inputs (user data, API responses)
- You need automatic coercion (e.g., `"87"` → `87` for an `int` field)
- Nested schemas with their own validation
- You want `.model_validate()` to fail loudly on bad data

**Hybrid pattern (recommended for complex agents):**

```python
from pydantic import BaseModel
from typing import TypedDict

class UserProfile(BaseModel):
    """Validated external data — use Pydantic."""
    name: str
    email: str
    tier: Literal["free", "pro", "enterprise"]
    preferences: dict[str, str] = {}

class AgentState(TypedDict):
    """Internal workflow state — use TypedDict."""
    messages: Annotated[list[BaseMessage], operator.add]
    user_profile: UserProfile  # Pydantic model embedded in TypedDict
    stage: str
    iterations: int
```

---

### Q4: How do you design state schemas for multi-agent workflows (supervisor + subagent pattern)?

**Answer:**

In multi-agent systems, each agent has its own local state, but there is a shared global state that coordinates them. The design challenge is: what lives in shared state vs. local state?

```python
from typing import TypedDict, Annotated
import operator

# Shared state — visible to all agents
class SharedWorkflowState(TypedDict):
    # The full conversation (all agents see this)
    messages: Annotated[list[BaseMessage], operator.add]
    
    # Task assignment (supervisor writes, subagents read)
    current_task: str
    assigned_agent: Literal["researcher", "writer", "critic", "supervisor"]
    
    # Outputs from each subagent (append-only)
    research_results: Annotated[list[str], operator.add]
    draft_outputs: Annotated[list[str], operator.add]
    critique_notes: Annotated[list[str], operator.add]
    
    # Workflow control
    is_complete: bool
    final_output: str | None

# Local state for the researcher subagent
class ResearcherState(TypedDict):
    query: str
    search_results: list[dict]
    sources_checked: int
    max_sources: int
```

**Key design principles:**

1. **Shared state for coordination, local state for execution:** The supervisor reads/writes `current_task` and `assigned_agent`. The researcher only reads `current_task` and writes `research_results`.

2. **Append-only for outputs:** Use `Annotated[list, operator.add]` for outputs from subagents — never let one agent overwrite another's results.

3. **Explicit completion signal:** Always have an `is_complete: bool` field that the supervisor sets. This prevents infinite loops.

4. **Iteration counter:** Add `iteration: int` to the shared state and enforce `max_iterations = 20` at the graph level to prevent runaway loops.

---

### Q5: How do you handle schema evolution when your agent state needs new fields in production?

**Answer:**

Schema evolution — adding, removing, or renaming fields — is the hardest operational challenge for stateful agents in production. Checkpointed state (serialized to a database) uses the old schema; live agents use the new schema.

**Three approaches:**

**1. Additive-only changes (safest):**
Only ever *add* new optional fields with defaults. Never remove or rename.

```python
class AgentStateV2(TypedDict, total=False):
    # V1 fields — required
    messages: list[BaseMessage]
    stage: str
    
    # V2 fields — optional with defaults (won't break V1 checkpoints)
    user_preferences: dict  # new in V2
    error_count: int        # new in V2
```

**2. Migration scripts:**
When loading a checkpoint, detect the schema version and run a migration:

```python
def migrate_state(raw_state: dict) -> AgentState:
    version = raw_state.get("schema_version", 1)
    if version == 1:
        raw_state["user_preferences"] = {}
        raw_state["error_count"] = 0
        raw_state["schema_version"] = 2
    return raw_state
```

**3. Schema versioning in the checkpoint key:**
Store checkpoints under `checkpoint:{session_id}:v{version}`. Load the latest version, migrate if needed.

**Production recommendation:** Use additive-only for the first 3–6 months of a new agent. Only introduce breaking changes during planned maintenance windows with a migration script and a checkpoint backfill job.

---

### Key Numbers to Memorize

| Metric | Value |
|---|---|
| Pydantic validation overhead vs. TypedDict | ~10–30% |
| LangGraph max recommended state size | <1MB (serialization bottleneck) |
| Recommended max_iterations for any agent loop | 20–50 |
| Schema migration risk window | Any time you have persisted checkpoints |
| Subagent count in typical multi-agent system | 3–7 |