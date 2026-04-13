# CrewAI Hello-World Agent

## Overview

CrewAI lets you define a **crew** of AI agents, each with a role, goal, and backstory — like a real team. Tasks are assigned to agents; the crew runs them in sequence or hierarchically. The abstraction hides most of the ReAct loop from you.

---

## Mental Model

```
Crew
 ├── Agent: Researcher  (role, goal, backstory, tools)
 ├── Agent: Writer      (role, goal, backstory, tools)
 │
 ├── Task: "Research topic X"   → assigned to Researcher
 └── Task: "Write report on X"  → assigned to Writer
        ↑ receives output of Task 1 as context
```

The `Crew` orchestrates execution. With `Process.sequential`, Task 1 runs, its output feeds Task 2, and so on.

---

## Minimal Setup

```python
# requirements: crewai crewai-tools python-dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool  # web search (needs SERPER_API_KEY)
import os

# ── 1. Tools ──────────────────────────────────────────────────────────────────
# CrewAI tools can be crewai_tools or custom @tool decorated functions
from crewai.tools import tool

@tool("Calculator")
def calculate(expression: str) -> str:
    """Evaluate a simple math expression safely."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

# ── 2. Agents ─────────────────────────────────────────────────────────────────
researcher = Agent(
    role="Research Analyst",
    goal="Find accurate, factual information on the given topic",
    backstory=(
        "You are a meticulous research analyst who always cites sources "
        "and separates facts from speculation."
    ),
    tools=[calculate],          # list of tools this agent can use
    llm="gpt-4o-mini",          # or omit to use OPENAI_API_KEY default
    verbose=True,               # prints Thought/Action/Observation
    allow_delegation=False,     # prevent recursive delegation
    max_iter=5,                 # max ReAct iterations per task
)

writer = Agent(
    role="Technical Writer",
    goal="Produce clear, structured summaries from research findings",
    backstory=(
        "You are an expert technical writer who turns complex findings "
        "into concise, readable reports."
    ),
    llm="gpt-4o-mini",
    verbose=True,
    allow_delegation=False,
)

# ── 3. Tasks ──────────────────────────────────────────────────────────────────
research_task = Task(
    description=(
        "Research the following topic and produce a bullet-point summary "
        "of the 5 most important facts: {topic}"
    ),
    expected_output="A numbered list of 5 key facts with brief explanations.",
    agent=researcher,
)

write_task = Task(
    description=(
        "Using the research findings provided, write a concise 3-paragraph "
        "executive summary suitable for a non-technical audience."
    ),
    expected_output="Three clear paragraphs: overview, key findings, implications.",
    agent=writer,
    context=[research_task],   # ← output of research_task feeds here automatically
)

# ── 4. Crew ───────────────────────────────────────────────────────────────────
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,   # sequential | hierarchical
    verbose=True,
)

# ── 5. Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = crew.kickoff(inputs={"topic": "LLM agent frameworks in 2025"})
    print("\n=== FINAL OUTPUT ===")
    print(result.raw)
```

---

## Hierarchical Process

```python
from crewai import Agent, Task, Crew, Process

manager = Agent(
    role="Project Manager",
    goal="Delegate tasks efficiently and synthesise final output",
    backstory="Experienced PM who coordinates specialist teams.",
    llm="gpt-4o",        # manager should be the most capable model
    allow_delegation=True,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.hierarchical,
    manager_agent=manager,   # explicitly set the manager
    verbose=True,
)
```

With `hierarchical`, the manager LLM decides task order, can re-delegate failed tasks, and synthesises output — no hardcoded sequence.

---

## Custom Tool with Schema

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description="City name to get weather for")

class WeatherTool(BaseTool):
    name: str = "WeatherChecker"
    description: str = "Fetch current weather for a city"
    args_schema: type[BaseModel] = WeatherInput

    def _run(self, city: str) -> str:
        # Replace with real API call
        return f"Weather in {city}: 22°C, Sunny"

weather_tool = WeatherTool()
```

---

## Memory in CrewAI

```python
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    memory=True,           # enables short-term + long-term + entity memory
    embedder={             # default uses OpenAI; swap here
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"}
    },
    verbose=True,
)
```

**Memory types CrewAI manages:**
| Type | What it stores | Backend |
|---|---|---|
| Short-term | Recent context window | In-memory |
| Long-term | Historical task outcomes | SQLite by default |
| Entity | People, places, facts | ChromaDB by default |
| User | Per-user preferences | SQLite |

---

## CrewAI Developer Experience Assessment

| Dimension | Score | Notes |
|---|---|---|
| Setup complexity | Low | ~30 lines for a working crew |
| Documentation | Good | Official docs, many examples |
| Debugging | Medium | verbose=True helps, less granular than LangGraph |
| State control | Low | CrewAI manages state internally |
| HITL support | Limited | No native interrupt(); needs custom callbacks |
| Learning curve | Shallow | Role/task metaphor is intuitive |
| Community | Very large | Most GitHub stars of any agent framework |
| Production maturity | Medium | Improving quickly, less battle-tested than LangGraph |

---

## LangGraph vs CrewAI: Developer Experience Comparison

| Aspect | LangGraph | CrewAI |
|---|---|---|
| Lines for hello-world | ~60 | ~30 |
| State management | Explicit TypedDict + reducers | Internal, opaque |
| Debugging granularity | Node-level traces, graph viz | Task-level logs |
| HITL | First-class interrupt() | Not native |
| Multi-agent | Manual routing in graph | Built-in crew/delegation |
| Tool integration | LangChain tools + custom | crewai_tools + custom |
| Flexibility | Very high | Medium |
| Recommended for | Production complex systems | Rapid POCs, team-metaphor tasks |

---

## Common Pitfalls

**1. allow_delegation=True on all agents**
Creates infinite delegation loops. Only enable on manager agents. Default to `False` for workers.

**2. Vague task descriptions**
CrewAI passes your `description` directly to the LLM. Vague tasks produce vague outputs. Be specific: include expected format, length, constraints.

**3. Too many agents**
Each agent adds LLM calls. Start with 2 agents. Add a third only when the first two clearly have different, non-overlapping responsibilities.

**4. Max iterations**
Default `max_iter` is 25. For production, set it to 5–10 to cap runaway reasoning loops.

```python
agent = Agent(
    ...
    max_iter=7,
    max_rpm=10,   # rate limit API calls per minute
)
```