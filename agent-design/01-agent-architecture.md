# 01 — Agent Architecture & Component Map

---

## Q1. What is a component map and why must you design it before coding?

**A:** A component map is a one-page diagram that shows every component in your agent system, what data each component receives, what it produces, and which other components it depends on. It is the architectural contract for your codebase.

**Why before coding:** The most expensive bugs in agent systems are not logic bugs — they are structural bugs. Wrong data flowing into the wrong component, memory writes happening after the LLM call instead of before, logs missing because they were added as an afterthought. A component map makes these visible on paper in 20 minutes instead of in production after 20 hours.

**The six components of a single agent:**

```
┌──────────────────────────────────────────────────────────────────┐
│  INPUT LAYER                                                     │
│  Accepts: raw user message string                                │
│  Produces: normalized message + session_id + user_id            │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│  MEMORY RETRIEVAL                                                │
│  Accepts: user_id + current query                                │
│  Produces: list of top-3 relevant facts from ChromaDB           │
│  Depends on: ChromaDB, embedding model                          │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│  LLM ORCHESTRATOR (ReAct)                                        │
│  Accepts: system prompt (with facts) + message history + tools   │
│  Produces: tool call OR final answer                             │
│  Depends on: LLM API, tool schemas, system prompt template      │
└──────────┬───────────────┬──────────────┬────────────────────────┘
           │               │              │
┌──────────▼──┐  ┌─────────▼──┐  ┌───────▼────────────────────────┐
│ web_search  │  │cal_lookup  │  │ HITL GATE                      │
│             │  │            │  │ Accepts: pending note content   │
│ Accepts:    │  │ Accepts:   │  │ Produces: approved/rejected/    │
│ query str   │  │ time_period│  │          edited note           │
│ Produces:   │  │ Produces:  │  │ Depends on: interrupt()        │
│ result str  │  │ events str │  └───────────┬────────────────────┘
└──────┬──────┘  └─────┬──────┘              │
       │               │              ┌──────▼───────────┐
       └───────┬────────┘              │  save_note tool  │
               │                      └──────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────────┐
│  MEMORY WRITE                                                    │
│  Accepts: latest user message                                    │
│  Produces: extracted facts upserted to ChromaDB                 │
│  Depends on: fact_extraction LLM call, ChromaDB                 │
└──────────────────────────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────────┐
│  OBSERVABILITY                                                   │
│  Accepts: every LLM call result + tool call result              │
│  Produces: JSONL log entry                                       │
│  Depends on: log file, timing wrapper                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## Q2. What are the most common architectural mistakes in first-agent builds?

**A:**

| Mistake | Consequence | Correct Pattern |
|---|---|---|
| Memory write before LLM call | Agent writes facts from the previous turn, not the current one | Write AFTER generating response |
| Memory retrieval after LLM call | LLM never sees relevant facts | Retrieve BEFORE building prompt |
| Tool schemas in the node code | Schema changes require touching multiple files | Colocate schema with tool function |
| Logging added after the fact | Gaps in logs, inconsistent fields | Design log schema before building anything |
| Single monolithic graph node | Cannot test or reuse individual components | One node per responsibility |
| No checkpointer configured | Cannot interrupt or resume — HITL is impossible | Always compile with checkpointer |

---

## Q3. How do LangGraph nodes map to architectural components?

**A:** Each architectural component becomes one LangGraph node. Nodes communicate only through the shared `AgentState` — no direct function calls between nodes.

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

def build_agent_graph():
    workflow = StateGraph(AgentState)

    # Add one node per architectural component
    workflow.add_node("retrieve_memory", retrieve_memory_node)
    workflow.add_node("llm_call", llm_call_node)
    workflow.add_node("tool_router", tool_router_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("calendar_lookup", calendar_lookup_node)
    workflow.add_node("hitl_gate", hitl_gate_node)
    workflow.add_node("save_note", save_note_node)
    workflow.add_node("write_memory", write_memory_node)
    workflow.add_node("feedback", feedback_node)

    # Set entry point
    workflow.set_entry_point("retrieve_memory")

    # Linear edges
    workflow.add_edge("retrieve_memory", "llm_call")
    workflow.add_edge("llm_call", "tool_router")

    # Conditional edges from tool_router
    workflow.add_conditional_edges(
        "tool_router",
        route_to_tool,  # returns node name as string
        {
            "web_search": "web_search",
            "calendar_lookup": "calendar_lookup",
            "save_note": "hitl_gate",   # Note saving goes through HITL first
            "no_tool": "write_memory",  # No tool needed — go straight to memory write
        }
    )

    # Tool results converge back
    workflow.add_edge("web_search", "llm_call")        # Re-run LLM with observation
    workflow.add_edge("calendar_lookup", "llm_call")
    workflow.add_edge("hitl_gate", "save_note")
    workflow.add_edge("save_note", "write_memory")

    # End of turn
    workflow.add_edge("write_memory", "feedback")
    workflow.add_edge("feedback", END)

    with SqliteSaver.from_conn_string("data/checkpoints.db") as checkpointer:
        return workflow.compile(checkpointer=checkpointer)
```

**The key insight:** `add_conditional_edges` takes a routing function that reads the current state and returns a string key. This is where all branching logic lives — not inside the nodes themselves.

---

## Q4. How do you choose between OpenAI and Anthropic for this agent?

**A:**

| Factor | OpenAI GPT-4o | Anthropic Claude 3.5 Sonnet |
|---|---|---|
| Tool calling | Mature, well-documented | Strong, supports MCP natively |
| Structured output | `response_format={"type":"json_object"}` | XML tags or `tool_use` |
| Context window | 128K tokens | 200K tokens |
| ReAct pattern | Works well | Works well |
| Cost (input/output per 1M) | $2.50 / $10.00 | $3.00 / $15.00 |
| Latency (P50) | ~800ms | ~900ms |
| LangChain integration | `ChatOpenAI` | `ChatAnthropic` |

**For Day 6:** Use OpenAI GPT-4o. The documentation and LangChain integration are more mature, and most learning resources use it. Switch to Claude when you need the longer context window or MCP tool integration.

```python
# LLM configuration
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,         # 0 for deterministic tool selection
    max_tokens=1000,
    timeout=30,
    max_retries=2
)

# Bind tools to the LLM
llm_with_tools = llm.bind_tools([web_search, save_note, calendar_lookup])
```

**Temperature 0 for agents:** Agents use tools to get information — they don't need creative variation. Temperature 0 gives the most reliable tool selection. Reserve higher temperatures for the final answer synthesis step if you want more natural-sounding responses.

---

## Q5. How do you configure SqliteSaver and what are its limitations?

**A:**

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Context manager approach (recommended — handles connection lifecycle)
with SqliteSaver.from_conn_string("data/checkpoints.db") as checkpointer:
    graph = workflow.compile(checkpointer=checkpointer)
    result = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": "session-abc-001"}}
    )

# Alternative: keep connection open for the app lifetime
checkpointer = SqliteSaver.from_conn_string("data/checkpoints.db")
checkpointer.__enter__()
graph = workflow.compile(checkpointer=checkpointer)
```

**SqliteSaver limitations and when to upgrade:**

| Limitation | Impact | Upgrade Path |
|---|---|---|
| Single writer at a time | Cannot run two agent turns concurrently for the same thread | PostgresSaver with connection pool |
| File-based — tied to one machine | Cannot deploy multiple replicas | PostgresSaver with shared DB |
| No built-in TTL / cleanup | Checkpoint file grows indefinitely | Add a cleanup job, or use Redis with TTL |
| No observability UI | Cannot browse checkpoints visually | LangSmith or Langfuse |

**For Day 6:** SqliteSaver is perfectly sufficient. The limitations only matter at production scale.

---

## Key Numbers

| Parameter | Value |
|---|---|
| Nodes in Day 6 agent graph | 9 |
| Tool count | 3 |
| ChromaDB similarity metric | cosine |
| Embedding dimensions (text-embedding-3-small) | 1536 |
| Checkpoint DB location | data/checkpoints.db |
| Log file format | JSONL (one object per line) |
| LLM temperature for tool selection | 0 |
| LLM temperature for synthesis | 0.3 |