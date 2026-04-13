# Day 8 — Agent Framework Landscape

> **Theme:** Understand the major agent frameworks, their trade-offs, and when to use each. Build hello-world agents in two frameworks and produce a comparison matrix.

---

## Key Numbers to Remember

| Framework | Stars (approx) | Abstraction | Best For |
|---|---|---|---|
| LangGraph | ~10k+ | Low–Medium (graph) | Complex, cyclic, HITL workflows |
| CrewAI | ~20k+ | High (role-based) | Team-style, task-delegation agents |
| OpenAI Agents SDK | Official | Medium (handoff) | OpenAI-native stacks |
| Claude SDK | Official | Low–Medium (tool use) | Anthropic-native, safety-critical |
| Google ADK | Official | Medium (workflow) | GCP / Gemini deployments |
| AWS Strands | Official | Medium (model-driven) | AWS-native production workloads |

---

## Q&A Notes

### Q1: What is the difference between an "agentic workflow" and a "true agent"?

**Workflow:** A pre-defined sequence of LLM calls with deterministic routing. The developer controls every branch. Predictable, auditable, easier to test.

**Agent:** The LLM itself decides which tools to call, in what order, and when to stop. The developer provides tools and a goal; the model plans the path.

**Trade-off table:**

| Dimension | Workflow | Agent |
|---|---|---|
| Predictability | High | Low |
| Flexibility | Low | High |
| Debugging difficulty | Low | High |
| Token cost | Lower (targeted) | Higher (planning loop) |
| Failure modes | Deterministic | Emergent |

**Real-world rule:** Start with a workflow. Reach for an agent only when the set of needed steps is genuinely unknown at design time (e.g., open-ended research, dynamic code generation).

---

### Q2: What is LangGraph and what makes it different?

LangGraph models your agent as a **directed graph** (nodes = functions, edges = routing logic). Cycles are first-class — you can loop back to a node without recursion depth limits.

**Core primitives:**
- `StateGraph(TypedDict)` — typed shared state flows through every node
- `add_node(name, fn)` — any Python callable
- `add_edge / add_conditional_edges` — static or dynamic routing
- `interrupt()` — suspend graph for human approval, then resume
- **Checkpointers** — SqliteSaver, PostgresSaver, MemorySaver; enable pause/resume, time-travel debugging, and HITL

**When to choose LangGraph:**
- You need explicit cycles (ReAct, retry loops)
- Human-in-the-loop approval gates
- Complex branching (10+ conditional paths)
- You want full control over state shape and routing logic
- Production systems that need observability and auditability

**When NOT to choose LangGraph:**
- Simple linear chains — overkill
- Rapid prototyping — too much boilerplate
- Teams without Python graph-thinking experience

**Code verbosity:** High. You write every node, every edge. You own everything.

---

### Q3: What is CrewAI and what makes it different?

CrewAI organises agents as a **crew** of role-playing workers with a shared goal. You define `Agent` (role, goal, backstory, tools), `Task` (description, expected output, assigned agent), and `Crew` (agents + tasks + process type).

**Process types:**
- `Process.sequential` — tasks run in order, each output feeds next
- `Process.hierarchical` — a manager LLM delegates, reviews, and re-delegates

**When to choose CrewAI:**
- The problem maps naturally to human-team metaphors (researcher → writer → editor)
- You want high-level abstractions with minimal boilerplate
- Quick POCs or demos
- Role-based task delegation patterns

**When NOT to choose CrewAI:**
- You need fine-grained control over state
- HITL or complex approval workflows (limited native support)
- Heavy production reliability requirements (less mature checkpointing)
- You need exact token control or cost auditing per step

**Code verbosity:** Low. A working multi-agent crew in ~30 lines.

---

### Q4: What is the OpenAI Agents SDK and what makes it different?

The OpenAI Agents SDK is OpenAI's official Python framework for building agents. It uses a **handoff** model: one agent can hand off to another, creating a network of specialised agents.

**Core primitives:**
- `Agent(name, instructions, tools, handoffs)` — the unit of execution
- `handoff(agent)` — transfer control to another agent with context preserved
- `Runner.run(agent, input)` — starts execution, follows handoffs
- Built-in tracing and guardrails integration

**Key differentiators:**
- Native streaming with `Runner.run_streamed()`
- First-class tracing baked in (no extra setup)
- Model context protocol (MCP) support
- Designed for OpenAI model family — best performance/cost with GPT-4o

**When to choose:**
- Your stack is 100% OpenAI
- You want handoff-based specialist routing
- You want minimal setup with strong official support
- Streaming UX is important

**When NOT to choose:**
- Multi-provider model routing
- Complex cyclic workflows needing checkpoints
- You need deep HITL control

---

### Q5: What is the Claude SDK (Anthropic) approach to agents?

Anthropic does not ship a graph-style agent framework. Instead the **Anthropic Python SDK** + **tool use** + **Model Context Protocol (MCP)** form the building blocks.

**Building blocks:**
- `anthropic.messages.create(tools=[...])` — model decides which tool to call
- Tool loop: call model → execute tool → append `tool_result` → call model again
- **Extended thinking** (`thinking: {type: "enabled", budget_tokens: N}`) — model produces a private chain-of-thought before answering
- **MCP** — standardised protocol for exposing tools/resources to any MCP-compatible model

**When to choose:**
- Safety-critical applications (constitutional AI training, content moderation)
- You need extended thinking for hard reasoning tasks
- Anthropic model is required (regulated industries, internal policy)
- MCP-first tool architecture

**When NOT to choose:**
- You need high-level orchestration abstractions (use LangGraph on top)
- Large multi-agent crews with complex delegation

---

### Q6: What is Google ADK and what makes it different?

Google Agent Development Kit (ADK) is Google's official Python framework for building agents optimised for the **Gemini** model family and **Google Cloud** deployment.

**Key features:**
- **Workflow agents**: `SequentialAgent`, `ParallelAgent`, `LoopAgent` — composable without writing graph code
- **LlmAgent**: reasoning + tool use, similar to ReAct
- **Multi-agent by composition**: agents are tools of other agents
- Native Vertex AI deployment, Cloud Run, Agent Engine integration
- Built-in evaluation framework
- Streaming support via Server-Sent Events

**When to choose:**
- GCP is your cloud (IAM, logging, billing already there)
- Gemini models (cost, latency, grounding via Google Search)
- You want managed agent hosting (Agent Engine removes infra burden)
- Parallel sub-task decomposition patterns

**When NOT to choose:**
- Multi-cloud or AWS/Azure shops
- OpenAI or Anthropic model requirements
- Ecosystem is still maturing (fewer community resources than LangGraph)

---

### Q7: What is AWS Strands and what makes it different?

AWS Strands Agents is Amazon's model-driven agent SDK (open-sourced May 2025). It follows a **model-in-the-loop** philosophy: the model drives everything; the framework provides minimal scaffolding.

**Key features:**
- `@tool` decorator — any Python function becomes a tool instantly
- `Agent(model, tools, system_prompt)` — minimal setup
- Supports Bedrock models (Claude, Llama, Titan, etc.) and OpenAI via provider abstraction
- Built-in multi-agent: `agents_as_tools` pattern
- Native AWS integrations: Bedrock, CloudWatch, S3
- Streaming out of the box

**When to choose:**
- AWS is your cloud (Bedrock, IAM, CloudWatch)
- You want minimum framework surface area
- Multi-model routing within Bedrock
- Production maturity is needed (AWS enterprise support)

**When NOT to choose:**
- Complex cyclic graph workflows (no native checkpointing like LangGraph)
- You need rich HITL primitives
- Non-AWS deployments

---

### Q8: How do you select a framework for a given use case?

Use this decision tree:

```
Is the workflow steps known at design time?
  YES → Workflow pattern (LangGraph graph, ADK SequentialAgent)
  NO  → Agent pattern (ReAct loop in any framework)
        ↓
Does it need HITL approval gates?
  YES → LangGraph (interrupt() + checkpointers)
  NO  ↓
Is the team metaphor natural? (researcher/writer/reviewer)
  YES → CrewAI
  NO  ↓
What cloud/model is locked in?
  OpenAI   → OpenAI Agents SDK
  Anthropic → Claude SDK (+LangGraph for orchestration)
  GCP      → Google ADK
  AWS      → AWS Strands
  None     → LangGraph (most flexible)
```

**Complexity vs abstraction trade-off:**

```
Low abstraction ←————————————————→ High abstraction
LangGraph   Claude SDK   Strands   ADK   OpenAI SDK   CrewAI
(most control)                               (least control)
```

---

### Q9: What is the Model Context Protocol (MCP) and why does it matter?

MCP is an **open standard** (from Anthropic, now multi-vendor) for connecting AI models to tools and data sources. Think USB-C for AI tools.

**Before MCP:** Every framework had its own tool schema. A tool written for OpenAI needed rewriting for Claude, LangGraph, etc.

**With MCP:** Write a tool once as an MCP server. Any MCP-compatible model or framework consumes it via a standard JSON-RPC protocol.

**Components:**
- **MCP Server**: Exposes tools/resources/prompts
- **MCP Client**: The model/framework consuming them
- **Transport**: stdio (local) or HTTP/SSE (remote)

**Real-world impact:** Claude Desktop, Cursor, Zed, and dozens of tools now support MCP. Write one server; it works everywhere.

---

### Q10: What are the key dimensions for comparing agent frameworks?

| Dimension | Why it matters |
|---|---|
| State management | Can state survive crashes, restarts, HITL pauses? |
| Tool calling | Native schema, parallel calls, error recovery? |
| Memory | Built-in short/long-term or BYO? |
| Streaming | Token-level, tool-call-level, or response-level? |
| Multi-agent | Native handoff/crew support or manual? |
| Observability | Built-in tracing or third-party required? |
| Deployment | Managed hosting or self-hosted only? |
| Model agnosticism | One provider or multi-provider? |
| HITL | Suspend/resume or polling only? |
| Community/docs | Stack Overflow coverage, GitHub issues, tutorials |

---

## Hello-World Agents

See deep dives:
- [`agent-design/10-langgraph-hello-world.md`](../agent-design/10-langgraph-hello-world.md)
- [`agent-design/11-crewai-hello-world.md`](../agent-design/11-crewai-hello-world.md)
- [`agent-design/12-framework-feature-matrix.md`](../agent-design/12-framework-feature-matrix.md)
- [`agent-design/13-framework-comparison-doc.md`](../agent-design/13-framework-comparison-doc.md)

---

## Summary: When to Use What

| Use Case | Recommended Framework |
|---|---|
| Complex HITL workflow with state | LangGraph |
| Team-style multi-agent POC | CrewAI |
| OpenAI-native production agent | OpenAI Agents SDK |
| Safety-critical / extended thinking | Claude SDK + LangGraph |
| GCP / Gemini deployment | Google ADK |
| AWS Bedrock production | AWS Strands |
| Research / multi-provider | LangGraph |