# Supervisor / Orchestrator Pattern

> **Core Idea:** A central orchestrator receives every task, decides which worker agent handles it, dispatches the work, and aggregates results. Workers never talk to each other — all communication flows through the supervisor.

---

## 1. Concept

The supervisor is the **single point of intelligence** for routing and synthesis. Workers are narrow specialists that do one thing well.

```
User ──► Supervisor ──► Worker A (coding)
                    ──► Worker B (research)
                    ──► Worker C (math)
                         │
                         └── results aggregated by supervisor ──► User
```

The supervisor can be:
- **LLM-based**: asks the model "which worker should handle this?" — flexible but costs tokens
- **Rule-based**: regex or classifier decides route — fast, deterministic, zero LLM cost
- **Hybrid**: classifier for known intents, LLM fallback for ambiguous ones

---

## 2. Mechanics

### Step-by-step execution

1. User sends a task to the supervisor.
2. Supervisor classifies the task (intent detection, topic tagging, or LLM reasoning).
3. Supervisor dispatches to the appropriate worker with a **focused sub-prompt** — not the full conversation history.
4. Worker executes with its own tools, memory, and system prompt.
5. Worker returns a structured response to the supervisor.
6. Supervisor optionally routes to additional workers (sequential or parallel).
7. Supervisor synthesizes a final answer and returns it to the user.

### Key design choices

| Decision | Option A | Option B |
|---|---|---|
| Routing logic | LLM "intent router" | Rule-based classifier |
| Worker invocation | Sequential | Parallel (async) |
| State location | Supervisor holds all state | Workers hold sub-state |
| Result synthesis | LLM summarizer | Template concatenation |

---

## 3. Implementation Skeleton (LangGraph)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage
import operator

# --- State ---
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_worker: str        # "coding" | "research" | "END"
    worker_result: str

# --- Supervisor node ---
def supervisor_node(state: AgentState) -> AgentState:
    """Routes to correct worker based on message content."""
    last_msg = state["messages"][-1].content.lower()
    
    if any(k in last_msg for k in ["code", "python", "function", "bug", "script"]):
        next_worker = "coding"
    elif any(k in last_msg for k in ["research", "explain", "what is", "summarize"]):
        next_worker = "research"
    else:
        next_worker = "research"   # default fallback
    
    return {"next_worker": next_worker}

# --- Worker nodes ---
def coding_agent(state: AgentState) -> AgentState:
    """Specialist: writes and explains code."""
    task = state["messages"][-1].content
    # In production: call LLM with coding-specific system prompt + tools
    result = f"[CodingAgent] Here is the solution to: {task}"
    return {
        "worker_result": result,
        "messages": [AIMessage(content=result, name="coding_agent")]
    }

def research_agent(state: AgentState) -> AgentState:
    """Specialist: retrieves facts, summarizes sources."""
    task = state["messages"][-1].content
    result = f"[ResearchAgent] Here is what I found about: {task}"
    return {
        "worker_result": result,
        "messages": [AIMessage(content=result, name="research_agent")]
    }

# --- Routing function ---
def route_to_worker(state: AgentState) -> str:
    return state["next_worker"]

# --- Build graph ---
graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("coding", coding_agent)
graph.add_node("research", research_agent)

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", route_to_worker, {
    "coding": "coding",
    "research": "research"
})
graph.add_edge("coding", END)
graph.add_edge("research", END)

app = graph.compile()

# --- Run ---
result = app.invoke({
    "messages": [HumanMessage(content="Write a Python function to reverse a linked list")],
    "next_worker": "",
    "worker_result": ""
})
print(result["worker_result"])
```

---

## 4. LLM-Based Supervisor (More Flexible)

When task boundaries are fuzzy, use the LLM to classify:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

SUPERVISOR_PROMPT = """You are a routing supervisor. 
Given a user task, decide which specialist handles it.

Specialists:
- coding: code writing, debugging, algorithms, data structures
- research: explanations, facts, summaries, analysis

Respond with ONLY the specialist name: coding or research
Task: {task}"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def llm_supervisor_node(state: AgentState) -> AgentState:
    task = state["messages"][-1].content
    prompt = SUPERVISOR_PROMPT.format(task=task)
    response = llm.invoke([HumanMessage(content=prompt)])
    next_worker = response.content.strip().lower()
    
    # Safety: default to research if unexpected output
    if next_worker not in ("coding", "research"):
        next_worker = "research"
    
    return {"next_worker": next_worker}
```

**Trade-off:** LLM router costs ~300–500 tokens per call but handles edge cases better. Rule-based is zero-cost but breaks on novelty.

---

## 5. Parallel Worker Execution

For tasks that benefit from multiple specialists simultaneously:

```python
import asyncio
from langgraph.graph import StateGraph

async def parallel_supervisor(state: AgentState) -> AgentState:
    """Fan out to both agents, merge results."""
    task = state["messages"][-1].content
    
    # Dispatch both in parallel
    coding_result, research_result = await asyncio.gather(
        asyncio.to_thread(coding_agent, state),
        asyncio.to_thread(research_agent, state)
    )
    
    merged = f"Code perspective:\n{coding_result['worker_result']}\n\nResearch perspective:\n{research_result['worker_result']}"
    return {"worker_result": merged}
```

Real-world latency gain: if each worker takes 2s, parallel execution brings total to ~2.5s vs 4s sequential.

---

## 6. Trade-offs

### Advantages
- **Auditable**: every routing decision passes through one node — easy to log and debug
- **Predictable cost**: supervisor knows total agent count; can enforce budget limits
- **Separation of concerns**: workers have zero knowledge of each other
- **Easy to add workers**: add a node and update routing logic — no other changes needed

### Disadvantages
- **Single point of failure**: supervisor bug breaks everything
- **Bottleneck**: all traffic flows through one LLM call (if LLM-based routing)
- **Context loss**: supervisor must summarize worker outputs; nuance can be lost in synthesis
- **Latency multiplier**: sequential dispatch means N workers = N × latency

---

## 7. Production Checklist

```python
# 1. Always set max iterations
MAX_SUPERVISOR_ITERATIONS = 5

# 2. Log every routing decision
def supervisor_node(state):
    decision = classify(state)
    logger.info(f"supervisor_routed|task={task[:50]}|to={decision}|ts={time.time()}")
    return {"next_worker": decision}

# 3. Timeout per worker
import signal
def with_timeout(fn, timeout_seconds=30):
    signal.alarm(timeout_seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)

# 4. Fallback on worker failure
def safe_coding_agent(state):
    try:
        return coding_agent(state)
    except Exception as e:
        return {"worker_result": f"Error in coding agent: {e}. Falling back to research."}
```

---

## 8. Real-World Examples

| Company / System | Supervisor Pattern Usage |
|---|---|
| **AutoGPT** | Master agent plans, sub-agents execute tasks |
| **LangGraph's built-in supervisor** | `create_supervisor()` helper in `langgraph-supervisor` |
| **OpenAI Swarm (early)** | Central triage agent before handoffs |
| **Salesforce Agentforce** | Planner agent routes to Action agents |
| **Microsoft Copilot Studio** | Orchestrator dispatches to skill bots |

---

## 9. Key Numbers

| Metric | Typical Value |
|---|---|
| Supervisor routing latency (rule-based) | < 1ms |
| Supervisor routing latency (LLM-based) | 300–800ms |
| Tokens for LLM routing call | 300–600 |
| Worker execution time (simple task) | 1–3s |
| Worker execution time (tool-calling) | 3–10s |
| Recommended max worker count | 5–10 (beyond this, use hierarchical supervisors) |