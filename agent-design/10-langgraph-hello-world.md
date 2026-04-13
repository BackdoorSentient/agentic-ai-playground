# LangGraph Hello-World Agent

## Overview

LangGraph models an agent as a **directed, potentially cyclic graph**. State flows through nodes; edges decide where to route next. This gives you full control — and full responsibility.

---

## Mental Model

```
[User Input]
     ↓
 [call_model] ←────────────────────────────────┐
     ↓                                          │
Does response have tool calls?         [execute_tools]
  YES ──────────────────────────────────────────┘
  NO
     ↓
 [END]
```

This is the canonical ReAct loop in graph form.

---

## Minimal Setup

```python
# requirements: langgraph langchain-openai python-dotenv
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
import json

# ── 1. State Schema ──────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # append-only via reducer

# ── 2. Model + Tools ─────────────────────────────────────────────────────────
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Return mock weather for a city."""
    return f"The weather in {city} is 22°C and sunny."

@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

tools = [get_weather, calculate]
tool_map = {t.name: t for t in tools}

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

# ── 3. Nodes ─────────────────────────────────────────────────────────────────
def call_model(state: AgentState) -> AgentState:
    """Call the LLM with current messages."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def execute_tools(state: AgentState) -> AgentState:
    """Execute all tool calls from the last AI message."""
    last_message = state["messages"][-1]
    tool_messages = []
    for tc in last_message.tool_calls:
        fn = tool_map[tc["name"]]
        result = fn.invoke(tc["args"])
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )
    return {"messages": tool_messages}

# ── 4. Routing ────────────────────────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    """Route: if last message has tool calls → tools, else → END."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "execute_tools"
    return END

# ── 5. Graph Assembly ─────────────────────────────────────────────────────────
graph = StateGraph(AgentState)
graph.add_node("call_model", call_model)
graph.add_node("execute_tools", execute_tools)

graph.set_entry_point("call_model")
graph.add_conditional_edges("call_model", should_continue)
graph.add_edge("execute_tools", "call_model")  # ← The cycle

agent = graph.compile()

# ── 6. Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in Mumbai and what is 42 * 17?")]
    })
    print(result["messages"][-1].content)
```

**Expected output:**
```
The weather in Mumbai is 22°C and sunny. Additionally, 42 × 17 = 714.
```

---

## Adding a Checkpointer (Persistence + HITL)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("agent.db")
agent = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "session-001"}}

# First turn
result = agent.invoke(
    {"messages": [HumanMessage(content="What is 10 + 5?")]},
    config=config
)

# Second turn — state is automatically persisted and loaded
result = agent.invoke(
    {"messages": [HumanMessage(content="Now multiply that by 3")]},
    config=config
)
# Model sees full conversation history from thread-001
```

---

## Annotated State Fields

```python
from typing import Annotated
from operator import add

class RicherState(TypedDict):
    messages: Annotated[list, add_messages]  # reducer: append new messages
    tool_calls_made: Annotated[int, add]     # reducer: sum increments
    user_id: str                              # no reducer: overwrite
    session_notes: Annotated[list, add]      # reducer: accumulate notes
```

**Rule:** Any field that multiple nodes write concurrently needs a reducer. Without it, the last writer wins (silently drops earlier writes in parallel branches).

---

## Visualising the Graph

```python
from IPython.display import Image
Image(agent.get_graph().draw_mermaid_png())
```

Or print as Mermaid ASCII:
```python
print(agent.get_graph().draw_mermaid())
```

---

## LangGraph Developer Experience Assessment

| Dimension | Score | Notes |
|---|---|---|
| Setup complexity | Medium | ~50 lines for hello-world |
| Documentation | Good | Official docs + many tutorials |
| Debugging | Excellent | LangSmith traces, graph visualization |
| State control | Excellent | Full TypedDict, reducers, checkpoints |
| HITL support | Excellent | interrupt() + SqliteSaver |
| Learning curve | Steep | Graph mental model takes time |
| Community | Large | LangChain ecosystem |
| Production maturity | High | Used in production widely |

---

## Common Pitfalls

**1. Forgetting `add_messages` reducer**
```python
# BAD — each node overwrites messages
class State(TypedDict):
    messages: list

# GOOD — reducer appends
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

**2. Infinite loops**
Always ensure your routing function has a terminal condition. A `should_continue` that always returns `"execute_tools"` loops forever. Add a `max_iterations` counter in state if needed.

**3. Parallel node writes without reducers**
If you use `add_edge` to multiple nodes that run concurrently, all writing the same field, you need a reducer or you'll get non-deterministic state.

---

## Full ReAct Agent in ~60 Lines

The code above IS a full ReAct agent:
- **Thought:** LLM call in `call_model`
- **Action:** Tool selection in `tool_calls`
- **Observation:** Tool result in `execute_tools` → `ToolMessage`
- **Repeat:** Edge back to `call_model`

No extra prompting needed — the model's tool-use training handles the ReAct pattern automatically when tools are bound.