# Agent Framework Comparison: LangGraph vs CrewAI
## With Code Samples and Production Trade-offs

---

## 1. Philosophy

### LangGraph
> "Give the developer complete control over every node, edge, and state transition."

LangGraph treats an agent as a **directed graph** you assemble from scratch. Nothing is hidden. You define nodes as Python functions, edges as routing logic, and state as a TypedDict. This means more boilerplate but zero magic.

### CrewAI
> "Model the agent system as a team of specialised humans."

CrewAI hides the ReAct loop inside `Agent` and `Task` objects. You think in terms of roles, goals, and task outputs — not nodes and edges. This means less boilerplate but less control.

---

## 2. Side-by-Side: The Same Agent in Both Frameworks

**Task:** A two-step agent that (1) looks up information and (2) writes a summary.

### LangGraph Version (~70 lines)

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool

# ── State ─────────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
def lookup_info(topic: str) -> str:
    """Look up information about a topic (mock)."""
    return f"Key facts about {topic}: [Fact 1], [Fact 2], [Fact 3]"

@tool
def write_summary(facts: str) -> str:
    """Write a structured summary from facts (mock)."""
    return f"Executive Summary\n\nBased on research: {facts}"

tools = [lookup_info, write_summary]
tool_map = {t.name: t for t in tools}
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

# ── Nodes ─────────────────────────────────────────────────────────────────────
def call_model(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}

def execute_tools(state: State) -> State:
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        result = tool_map[tc["name"]].invoke(tc["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}

def route(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

# ── Graph ─────────────────────────────────────────────────────────────────────
g = StateGraph(State)
g.add_node("model", call_model)
g.add_node("tools", execute_tools)
g.set_entry_point("model")
g.add_conditional_edges("model", route)
g.add_edge("tools", "model")
agent = g.compile()

result = agent.invoke({"messages": [HumanMessage(content="Research and summarise LLM agents")]})
print(result["messages"][-1].content)
```

### CrewAI Version (~35 lines)

```python
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

# ── Tools ─────────────────────────────────────────────────────────────────────
@tool("InfoLookup")
def lookup_info(topic: str) -> str:
    """Look up information about a topic (mock)."""
    return f"Key facts about {topic}: [Fact 1], [Fact 2], [Fact 3]"

# ── Agents ────────────────────────────────────────────────────────────────────
researcher = Agent(
    role="Research Analyst",
    goal="Find key facts about the given topic",
    backstory="A meticulous analyst with years of research experience.",
    tools=[lookup_info],
    llm="gpt-4o-mini",
    verbose=True,
)

writer = Agent(
    role="Technical Writer",
    goal="Write a clear summary from research findings",
    backstory="Expert at turning dense research into readable summaries.",
    llm="gpt-4o-mini",
    verbose=True,
)

# ── Tasks ─────────────────────────────────────────────────────────────────────
research_task = Task(
    description="Research and collect key facts about: {topic}",
    expected_output="A list of 3-5 key facts.",
    agent=researcher,
)

summary_task = Task(
    description="Write a 2-paragraph executive summary from the research.",
    expected_output="Two clear paragraphs summarising the findings.",
    agent=writer,
    context=[research_task],
)

# ── Crew ──────────────────────────────────────────────────────────────────────
crew = Crew(agents=[researcher, writer], tasks=[research_task, summary_task],
            process=Process.sequential, verbose=True)
result = crew.kickoff(inputs={"topic": "LLM agents"})
print(result.raw)
```

**Observation:** CrewAI version is roughly half the lines. The trade-off is that the ReAct loop, tool dispatch, and state routing are all hidden inside `Agent.kickoff()`.

---

## 3. Concurrency and Parallelism

### LangGraph — Parallel branches

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

class State(TypedDict):
    results: Annotated[list, add]  # reducer accumulates results

def research_a(state: State) -> State:
    return {"results": ["Result from source A"]}

def research_b(state: State) -> State:
    return {"results": ["Result from source B"]}

def merge(state: State) -> State:
    combined = "\n".join(state["results"])
    return {"results": [f"Merged: {combined}"]}

g = StateGraph(State)
g.add_node("research_a", research_a)
g.add_node("research_b", research_b)
g.add_node("merge", merge)

# Fan-out: entry → both research nodes in parallel
g.set_entry_point("research_a")  # or use a START node with parallel edges
g.add_edge("research_a", "merge")
g.add_edge("research_b", "merge")
```

### CrewAI — Parallel tasks (limited)

```python
# CrewAI doesn't have first-class parallel execution
# Workaround: use Process.hierarchical and let manager delegate concurrently
# Or: run multiple kickoffs in asyncio.gather() for true parallelism
import asyncio

async def run_parallel():
    results = await asyncio.gather(
        crew_a.kickoff_async(inputs={"source": "A"}),
        crew_b.kickoff_async(inputs={"source": "B"}),
    )
    return results
```

---

## 4. Error Handling and Retry

### LangGraph — Explicit retry node

```python
import time

class State(TypedDict):
    messages: Annotated[list, add_messages]
    retry_count: int
    error: str | None

def execute_tools_with_retry(state: State) -> State:
    last = state["messages"][-1]
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            results = []
            for tc in last.tool_calls:
                result = tool_map[tc["name"]].invoke(tc["args"])
                results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            return {"messages": results, "retry_count": 0, "error": None}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                return {
                    "messages": [ToolMessage(
                        content=f"Tool failed after {max_retries} attempts: {e}",
                        tool_call_id=last.tool_calls[0]["id"]
                    )],
                    "retry_count": attempt + 1,
                    "error": str(e)
                }
```

### CrewAI — Built-in retry

```python
agent = Agent(
    role="...",
    goal="...",
    backstory="...",
    max_iter=7,           # max ReAct steps; also serves as retry budget
    max_retries=3,        # retry on tool error
    llm="gpt-4o-mini",
)
```

CrewAI handles retries internally. LangGraph requires you to write the logic — but you can customise the backoff strategy, fallback tool, and error message format exactly.

---

## 5. Production Checklist Comparison

| Production Concern | LangGraph | CrewAI |
|---|---|---|
| **Crash recovery** | ✅ SqliteSaver checkpoints | ❌ Restart from scratch |
| **Cost budgeting** | Manual token tracking | `max_rpm` per agent |
| **Rate limiting** | Manual (tenacity) | `max_rpm` built-in |
| **Timeout per step** | Manual asyncio.wait_for | `max_iter` (indirect) |
| **Secrets management** | BYO (env vars) | BYO (env vars) |
| **Audit logging** | ✅ Full state snapshots | Step-level logs only |
| **Reproducibility** | ✅ Replay any checkpoint | ❌ No replay |
| **CI/CD testability** | ✅ Mock state injection | Task-level mocking |

---

## 6. When Each Framework Falls Short

### LangGraph pain points
- **Boilerplate overhead:** 3× more code than CrewAI for simple tasks
- **Learning curve:** Graph mental model takes ~1 week to internalise
- **No built-in memory:** You wire ChromaDB or Postgres yourself
- **Debugging parallel branches:** State merging bugs are subtle

### CrewAI pain points
- **State opacity:** Can't inspect intermediate state between tasks
- **No native HITL:** `human_input=True` blocks the thread; not suitable for async web apps
- **Sequential by default:** Parallelism needs workarounds
- **Less control over prompt:** Backstory/goal system prompts are merged by CrewAI, not directly editable

---

## 7. Final Recommendation

| Scenario | Winner | Why |
|---|---|---|
| Production agent needing crash recovery | LangGraph | Checkpointing |
| HITL approval before irreversible action | LangGraph | interrupt() |
| Internal hackathon / 2-day POC | CrewAI | Speed |
| Content pipeline (research → write → edit) | CrewAI | Role metaphor fits |
| Compliance-critical, auditable workflow | LangGraph | Full state audit trail |
| Multi-provider model routing | LangGraph | Model-agnostic |
| Team familiar with LangChain | LangGraph | Ecosystem fit |
| Team with no LLM framework experience | CrewAI | Gentler onboarding |

**Rule of thumb:**
- If you'd be comfortable calling it a "workflow" → LangGraph
- If you'd call it "a team tackling a brief" → CrewAI
- If neither fits, combine them: CrewAI for high-level crew orchestration, LangGraph as the execution engine per agent