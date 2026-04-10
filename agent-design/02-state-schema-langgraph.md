# 02 — State Schema Design & LangGraph Setup

---

## Q1. Why is the state schema the most important design decision in a LangGraph agent?

**A:** In LangGraph, every node reads from state and writes back to state. State is the only communication channel between nodes — there are no direct function calls, no shared globals. If your state schema is wrong, every node that depends on it is wrong.

**The three failure modes of a bad state schema:**

| Failure | Example | Cost |
|---|---|---|
| Missing field | Tool result not in state — next node can't see it | Refactor all nodes that need it |
| Wrong type | `messages: list` instead of `Annotated[list, add_messages]` — messages overwritten not appended | Silent data loss, hard to debug |
| Overly coupled | One giant field holding everything — nodes read/write the same field | Race conditions, merge conflicts |

**The principle:** State fields should be owned by exactly one node (writer) and read by any node that needs them. If two nodes write the same field, you have a design problem.

---

## Q2. What is `Annotated[list, add_messages]` and why is it required for messages?

**A:** By default, when a LangGraph node returns a state update, it **replaces** the existing field value. For most fields this is correct. For `messages`, it is catastrophically wrong — you'd lose your conversation history on every turn.

`Annotated[list, add_messages]` tells LangGraph to use a custom reducer: instead of replacing the messages list, it **merges** the new messages into the existing list, deduplicating by message ID.

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

class AgentState(TypedDict):
    # CORRECT — messages are merged, not replaced
    messages: Annotated[list, add_messages]

    # WRONG — every node return would wipe the history
    # messages: list

# Demonstration
state_v1 = {"messages": [HumanMessage(content="Hello")]}
state_update = {"messages": [AIMessage(content="Hi there!")]}

# With add_messages reducer:
# Result: [HumanMessage("Hello"), AIMessage("Hi there!")]

# Without reducer:
# Result: [AIMessage("Hi there!")]  ← history lost
```

**The `add_messages` contract:**
- New messages are appended to existing ones
- If a new message has the same `id` as an existing one, it replaces it (update semantics)
- Accepts both single messages and lists

---

## Q3. How do you design state fields for optional vs. required data?

**A:** Use `Optional[T]` for fields that are only populated during certain graph paths. Never use a sentinel value like `""` or `[]` for "not set" — use `None` and check explicitly.

```python
from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # ── Always present ──────────────────────────────────────
    messages: Annotated[list, add_messages]  # Full conversation history
    session_id: str                           # Set at graph entry, never changes
    user_id: str                              # Set at graph entry, never changes

    # ── Set by memory retrieval node ────────────────────────
    retrieved_facts: list[str]               # Top-3 facts; empty list if none

    # ── Set when context is long ────────────────────────────
    conversation_summary: Optional[str]      # None until summarization triggers

    # ── Set by LLM call node ────────────────────────────────
    pending_tool_call: Optional[dict]        # {"name": "...", "args": {...}}
    tool_result: Optional[str]               # None until tool executes

    # ── Set during HITL flow ────────────────────────────────
    pending_note: Optional[str]             # Note awaiting approval
    hitl_decision: Optional[str]            # "approved" | "rejected" | "edited"
    edited_note: Optional[str]              # Set only if human edited the note

    # ── Observability ───────────────────────────────────────
    turn_log: list[dict]                    # Log entries — reset each turn
    total_tokens_used: int                  # Cumulative across all turns
```

**Initialization — always provide a complete initial state:**

```python
def initial_state(session_id: str, user_id: str, user_message: str) -> AgentState:
    from langchain_core.messages import HumanMessage
    return {
        "messages": [HumanMessage(content=user_message)],
        "session_id": session_id,
        "user_id": user_id,
        "retrieved_facts": [],
        "conversation_summary": None,
        "pending_tool_call": None,
        "tool_result": None,
        "pending_note": None,
        "hitl_decision": None,
        "edited_note": None,
        "turn_log": [],
        "total_tokens_used": 0
    }
```

**Why initialize all fields:** LangGraph will raise `KeyError` if a node tries to read a field that doesn't exist in state. Providing all fields at initialization prevents this class of error entirely.

---

## Q4. How do you implement the routing function for conditional edges?

**A:** The routing function reads the current state and returns a string that matches one of the keys in the `conditional_edges` mapping. Keep it pure — no side effects, no LLM calls.

```python
from typing import Literal

def route_to_tool(state: AgentState) -> Literal["web_search", "calendar_lookup", "save_note", "no_tool"]:
    """
    Reads the pending_tool_call field set by the LLM node.
    Returns the name of the next node to execute.
    """
    tool_call = state.get("pending_tool_call")

    if tool_call is None:
        return "no_tool"

    tool_name = tool_call.get("name", "")

    if tool_name == "web_search":
        return "web_search"
    elif tool_name == "calendar_lookup":
        return "calendar_lookup"
    elif tool_name == "save_note":
        return "save_note"
    else:
        # Unknown tool — treat as no tool to avoid hanging
        return "no_tool"

def should_continue_after_tool(state: AgentState) -> Literal["llm_call", "write_memory"]:
    """
    After a tool executes, decide whether to call LLM again
    (to synthesize the result) or end the turn.
    """
    # If there's a tool result, synthesize it into a response
    if state.get("tool_result") is not None:
        return "llm_call"
    return "write_memory"
```

**Type the return value with `Literal`:** This gives you IDE autocomplete and catches typos at definition time rather than at runtime when the graph fails to route.

---

## Q5. How do you implement the LLM call node with proper tool extraction?

**A:**

```python
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from observability.logger import log_llm_call
from prompts.system_prompt import build_system_prompt

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def llm_call_node(state: AgentState) -> dict:
    start = time.time()

    # Build system prompt with retrieved facts
    system_msg = SystemMessage(content=build_system_prompt(state["retrieved_facts"]))

    # Build message list: system + history
    # If we have a tool result, append it as a tool message
    messages = [system_msg] + state["messages"]

    # Bind tools
    from tools.web_search import web_search
    from tools.note_taking import save_note
    from tools.calendar_lookup import calendar_lookup

    llm_with_tools = llm.bind_tools([web_search, save_note, calendar_lookup])

    # Call LLM
    response = llm_with_tools.invoke(messages)

    latency_ms = (time.time() - start) * 1000

    # Extract tool call if present
    pending_tool_call = None
    if response.tool_calls:
        tc = response.tool_calls[0]  # Handle first tool call
        pending_tool_call = {"name": tc["name"], "args": tc["args"]}

    # Log the call
    usage = response.usage_metadata or {}
    log_entry = log_llm_call(
        prompt_tokens=usage.get("input_tokens", 0),
        completion_tokens=usage.get("output_tokens", 0),
        latency_ms=latency_ms,
        tool_invoked=pending_tool_call["name"] if pending_tool_call else None,
        node="llm_call"
    )

    return {
        "messages": [response],              # add_messages will append this
        "pending_tool_call": pending_tool_call,
        "tool_result": None,                 # Reset tool result for this turn
        "turn_log": state["turn_log"] + [log_entry],
        "total_tokens_used": state["total_tokens_used"] + usage.get("total_tokens", 0)
    }
```

---

## Q6. How does the graph handle tool result injection back into the LLM?

**A:** After a tool executes, its result needs to be added to the message history as a `ToolMessage` so the LLM can see it in the next call. LangGraph handles this through the message add_messages reducer.

```python
from langchain_core.messages import ToolMessage

def web_search_node(state: AgentState) -> dict:
    tool_call = state["pending_tool_call"]
    query = tool_call["args"]["query"]

    # Execute the tool
    result = web_search.invoke({"query": query})

    # Create a ToolMessage so the LLM can see the result
    tool_message = ToolMessage(
        content=result,
        tool_call_id=tool_call.get("id", "tool-call-1")
    )

    return {
        "messages": [tool_message],   # Will be appended via add_messages
        "tool_result": result,
        "pending_tool_call": None     # Clear the pending call
    }
```

**The message flow for a tool-using turn:**

```
Turn start:
messages = [HumanMessage("what is langchain memory?")]

After LLM call:
messages = [HumanMessage(...), AIMessage(tool_calls=[{name:"web_search", args:{query:"langchain memory"}}])]

After tool execution:
messages = [HumanMessage(...), AIMessage(tool_calls=[...]), ToolMessage(content="LangChain supports...")]

After synthesis LLM call:
messages = [..., AIMessage(content="LangChain supports three memory types: ...")]
```

---

## Key Numbers

| Parameter | Value |
|---|---|
| State fields (Day 6 agent) | 13 |
| Required fields (always set) | 3 — messages, session_id, user_id |
| Optional fields | 5 — summary, tool_call, tool_result, note, hitl_decision |
| Message format for tool result | `ToolMessage` with `tool_call_id` |
| LangGraph node return type | `dict` — partial state update |
| Routing function return type | `Literal[...]` — string key |