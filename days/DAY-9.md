# Day 9 — Multi-Agent Architecture Patterns

> **Theme:** Design, implement, and reason about multi-agent systems at the senior engineering level.

---

## Key Concepts

| Pattern | Control Style | Best For | Failure Mode |
|---|---|---|---|
| **Supervisor / Orchestrator** | Top-down, central | Clear task delegation, auditable pipelines | Single point of failure; supervisor bottleneck |
| **Agents-as-Tools** | Hierarchical function calls | Expertise routing, composable sub-agents | Deep nesting adds latency; hard to debug |
| **Swarm / Peer-to-Peer** | Autonomous handoffs | Flexible routing, ambiguous task domains | No global state; hard to guarantee completion |
| **Graph (LangGraph)** | State machine, typed edges | Complex multi-step workflows, HITL | Graph design overhead; state schema coupling |
| **Debate / Consensus** | Multi-round critique | Quality improvement, reducing hallucination | Token cost explosion; deadlock risk |

---

## Q&A Highlights

**Q: When should I use supervisor vs swarm?**
Supervisor → you need auditability and predictable cost. Swarm → you need dynamic routing and don't know the path ahead of time.

**Q: How does agents-as-tools differ from a supervisor?**
In supervisor, the orchestrator *orchestrates*. In agents-as-tools, sub-agents are *called* like functions — the orchestrator is just an LLM using tool calls. Same topology, different abstraction.

**Q: What's the biggest production risk in multi-agent systems?**
Unbounded loops. Always enforce `max_iterations` and budget caps per agent.

**Q: How do I handle state across agents?**
Shared typed state (TypedDict in LangGraph), or message passing (agent-to-agent structured JSON). Never rely on implicit prompt history — it's invisible to other agents.

**Q: What's a realistic token cost for a 3-agent debate?**
~8,000–15,000 tokens per full round trip (2 debate agents + judge). At GPT-4o pricing (~$5/M), that's ~$0.04–0.075 per debate. Budget accordingly.

---

## Numbers to Know

| Metric | Value |
|---|---|
| Supervisor overhead (routing LLM call) | ~500–800 tokens |
| Typical sub-agent call (tool + response) | ~1,000–3,000 tokens |
| Swarm handoff payload | ~200–400 tokens (context slice) |
| Debate round (2 agents + judge) | ~8,000–15,000 tokens |
| LangGraph node transition overhead | ~0 tokens (pure Python state machine) |
| Recommended max_iterations guard | 10–20 per agent |

---

## Deep Dives

| File | Topic |
|---|---|
| [`agent-design/14-supervisor-pattern.md`](../agent-design/14-supervisor-pattern.md) | Supervisor / Orchestrator — full mechanics |
| [`agent-design/15-agents-as-tools.md`](../agent-design/15-agents-as-tools.md) | Agents-as-Tools delegation pattern |
| [`agent-design/16-swarm-pattern.md`](../agent-design/16-swarm-pattern.md) | Swarm & peer-to-peer handoffs |
| [`agent-design/17-graph-pattern.md`](../agent-design/17-graph-pattern.md) | LangGraph graph-based workflows |
| [`agent-design/18-debate-consensus.md`](../agent-design/18-debate-consensus.md) | Debate / Consensus pattern |
| [`agent-design/19-multi-agent-hands-on.md`](../agent-design/19-multi-agent-hands-on.md) | Hands-on builds (supervisor, debate, swarm) |

---

## Hands-On Deliverables

- **Supervisor system** routing to a Coding Agent and Research Agent
- **Debate system** with two debater agents and a Judge agent
- **Swarm handoff** chain across three agents (intake → specialist → finalizer)

---

## Mental Model

```
Supervisor     →  "Boss assigns tasks, workers report back"
Agents-as-Tools → "Boss calls sub-agents like API endpoints"
Swarm          →  "Hot-potato: each agent does their bit, passes it on"
Graph          →  "State machine: edges decide who runs next"
Debate         →  "Red team / Blue team with a referee"
```