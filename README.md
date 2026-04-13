# 🤖 Agentic AI Playground

<p align="center">
  <b>Author:</b> Aniket Waichal<br>
  <b>A Hands-on, Structured Learning Hub for Agentic AI & LLM Engineering</b><br>
  Built to go beyond theory — covering concepts, experiments, and real-world AI system building.
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/Focus-Agentic%20AI-blue" />
  <img src="https://img.shields.io/badge/Domain-LLM%20Engineering-orange" />
  <img src="https://img.shields.io/badge/Content-Notes%20%2B%20Experiments-green" />
  <img src="https://img.shields.io/badge/Format-Structured%20Learning-purple" />
  <img src="https://img.shields.io/badge/Status-In%20Progress-yellow" />
  <img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey" />
</p>

---

## 📌 About This Repository

This repository documents my journey into **Agentic AI and LLM Engineering** through structured learning, deep-dive notes, and hands-on experiments.

The goal is to:
- Build a strong foundation in how LLMs work
- Understand how to design AI-powered products
- Move from theory → practical implementation
- Develop production-ready AI systems

---

## 📁 Repository Structure

```text
agentic-ai-playground/
│
├── README.md
├── LICENSE                           # CC BY-NC 4.0 — read before reusing
│
├── days/                             # Daily Q&A summary notes (entry point per day)
│   ├── DAY-1.md                      # LLM Foundations ✅
│   ├── DAY-2.md                      # Prompt Engineering ✅
│   ├── DAY-3.md                      # Memory & State Management ✅
│   ├── DAY-4.md                      # Tool Calling & Function Integration ✅
│   ├── DAY-5.md                      # Human-in-the-Loop & Interrupts ✅
│   ├── DAY-6.md                      # Build Your First Agent — Design & Setup ✅
│   ├── DAY-7.md                      # Complete Agent — HITL, Observability & Polish ✅
│   └── DAY-8.md                      # Agent Framework Landscape ✅ NEW
│
├── foundations/                      # Core LLM and AI fundamentals (Day 1)
│   ├── 01-product-ai-patterns.md
│   ├── 02-temperature.md
│   ├── 03-top-p.md
│   ├── 04-context-window.md
│   ├── 05-pretraining.md
│   ├── 06-fine-tuning.md
│   ├── 07-RLHF.md
│   ├── 08-structured-outputs.md
│   ├── 09-rag-architecture-and-evaluation.md
│   └── 10-fastapi-framework.md
│
├── prompting/                        # Prompt engineering techniques (Day 2)
│   ├── 01-zero-shot-and-few-shot.md
│   ├── 02-chain-of-thought.md
│   ├── 03-react-and-reflexion.md
│   ├── 04-system-prompt-design.md
│   └── 05-structured-outputs-and-validation.md
│
├── memory/                           # Memory & State Management (Day 3)
│   ├── 01-memory-taxonomy.md
│   ├── 02-conversation-memory-summarization.md
│   ├── 03-state-schemas-agent-workflows.md
│   └── 04-checkpointing-durable-execution.md
│
├── tool-calling/                     # Tool Calling & Function Integration (Day 4)
│   ├── 01-tool-schema-design.md
│   ├── 02-tool-calling-mechanics.md
│   ├── 03-tool-selection-strategies.md
│   └── 04-error-handling-retry-logic.md
│
├── hitl/                             # Human-in-the-Loop & Interrupts (Day 5)
│   ├── 01-approval-workflows.md
│   ├── 02-confidence-escalation.md
│   ├── 03-langgraph-interrupt.md
│   └── 04-feedback-loops.md
│
├── agent-design/                     # Agent Design & Complete Build (Days 6–8) ✅
│   ├── 01-agent-architecture.md
│   ├── 02-state-schema-langgraph.md
│   ├── 03-tools-react-memory.md
│   ├── 04-observability-logging.md
│   ├── 05-hitl-approval-integration.md
│   ├── 06-conversation-summarization.md
│   ├── 07-observability-logging.md
│   ├── 08-feedback-collection.md
│   ├── 09-e2e-testing-gradio.md
│   ├── 10-langgraph-hello-world.md       ← NEW Day 8
│   ├── 11-crewai-hello-world.md          ← NEW Day 8
│   ├── 12-framework-feature-matrix.md   ← NEW Day 8
│   └── 13-framework-comparison-doc.md   ← NEW Day 8
│
├── agents/                           # Agentic workflows and systems (upcoming)
│   ├── tool-calling.md
│   ├── memory.md
│   └── multi-agent-systems.md
│
├── rag/                              # Retrieval-Augmented Generation (upcoming)
│   ├── basics.md
│   ├── embeddings.md
│   └── vector-databases.md
│
└── experiments/                      # Hands-on implementations (upcoming)
    ├── mini-projects/
    └── prototypes/
```

---

## 📚 What I'm Learning

### 🧠 LLM Foundations — Day 1 ✅
- Product AI Patterns (Assist vs Automate)
- Temperature & Top-p (Nucleus Sampling)
- Context Window & Tokenization
- Pre-training, Fine-tuning, RLHF and model behavior
- Structured Outputs & RAG Architecture
- FastAPI Framework

> 📄 Summary: [`days/DAY-1.md`](days/DAY-1.md) | Deep dives: [`foundations/`](foundations/)

---

### ✍️ Prompt Engineering — Day 2 ✅
- Zero-shot & Few-shot prompting
- Chain-of-Thought (CoT, Self-consistency, Tree of Thought)
- ReAct & Reflexion agent patterns
- System prompt design for production agents
- Structured outputs: JSON / YAML / TOON
- Pydantic validation & self-healing retry loops

> 📄 Summary: [`days/DAY-2.md`](days/DAY-2.md) | Deep dives: [`prompting/`](prompting/)

---

### 🧠 Memory & State Management — Day 3 ✅
- Memory taxonomy: short-term, long-term, episodic, semantic, procedural
- Conversation memory with LLM-based summarization (context overflow management)
- Typed state schemas with TypedDict and Pydantic for complex agent workflows
- Checkpointing for durable execution: pause, resume, HITL, time-travel debugging
- Hands-on: Conversation summarizer, ChromaDB vector memory, LangGraph SqliteSaver

> 📄 Summary: [`days/DAY-3.md`](days/DAY-3.md) | Deep dives: [`memory/`](memory/)

---

### 🔧 Tool Calling & Function Integration — Day 4 ✅
- Tool schema design: writing definitions LLMs can reliably invoke across OpenAI, Claude, and open-source models
- Tool calling mechanics: the full request/response cycle, parallel tool calls, ToolNode pattern
- Tool selection strategies: all-in-context (<10), categorized routing (10–50), RAG retrieval (50+), hierarchical agents (100+)
- Error handling & retry logic: exponential backoff, structured errors, fallback hierarchy, circuit breakers
- Hands-on: Multi-tool agent, error handling lab, tool tracing dashboard

> 📄 Summary: [`days/DAY-4.md`](days/DAY-4.md) | Deep dives: [`tool-calling/`](tool-calling/)

---

### 🛑 Human-in-the-Loop & Interrupts — Day 5 ✅
- Approval workflows: when to interrupt, what to show, how to audit decisions
- Confidence-based escalation: logprobs, self-assessment, ensemble voting, threshold calibration
- LangGraph `interrupt()` mechanics: full suspend/resume cycle, checkpointers (Memory, Sqlite, Postgres)
- Feedback loops: explicit (thumbs up/down), implicit (retry detection), offline analysis, closing the loop
- Hands-on: Approval workflow with audit trail, confidence router at 0.7 threshold, feedback collection node

> 📄 Summary: [`days/DAY-5.md`](days/DAY-5.md) | Deep dives: [`hitl/`](hitl/)

---

### 🏗️ Build Your First Agent — Design & Setup — Day 6 ✅
- Full production agent architecture: 6 layers, component map, data-flow design before coding
- State schema design: TypedDict with `Annotated[list, add_messages]`, optional fields, initialization
- Project structure: agents/, tools/, memory/, prompts/, observability/, logs/, data/, tests/
- LangGraph graph: 9 nodes, conditional edges, SqliteSaver checkpointer
- Three core tools: web_search (mock + real via Tavily), save_note (JSON file), calendar_lookup (mock)
- ReAct system prompt: Thought → Action → Observation → Final Answer with retrieved facts injected
- Short-term memory: token counting with tiktoken, LLM-based summarization at 2000 tokens
- Long-term memory: ChromaDB PersistentClient, embed-upsert-retrieve cycle, fact extraction with gpt-4o-mini
- Observability: JSONL logger, cost estimation per model, session report generator, log-based debugging

> 📄 Summary: [`days/DAY-6.md`](days/DAY-6.md) | Deep dives: [`agent-design/`](agent-design/)

---

### 🚀 Complete Agent — HITL, Observability & Polish — Day 7 ✅
- **Full graph topology**: 9-node LangGraph with conditional edges — approval gate fires only for `save_note`
- **HITL approval node**: `interrupt()` shows exact note content to user, handles approve / reject / edit outcomes
- **Conversation summarization**: tiktoken counting at 2000-token threshold, LLM-based compression, keeps last 4 messages raw
- **Feedback collection node**: `interrupt()` for thumbs up/down at end of every turn, persisted to SQLite
- **Observability decorator**: `@observe_llm(model, node_name)` wraps every LLM call — logs timestamp, tokens, cost, latency_ms, errors to JSONL
- **Log viewer**: reads JSONL, prints total calls/tokens/cost, per-node latency and cost breakdown
- **E2E test suite**: 5 scenarios covering web search, note HITL, calendar, memory recall, feedback
- **Gradio UI** *(stretch)*: chat window + dynamic approve/reject panel + memory sidebar
- **Streaming** *(stretch)*: token-by-token response with generator-based Gradio handler

> 📄 Summary: [`days/DAY-7.md`](days/DAY-7.md) | Deep dives: [`agent-design/05`](agent-design/05-hitl-approval-integration.md) · [`06`](agent-design/06-conversation-summarization.md) · [`07`](agent-design/07-observability-logging.md) · [`08`](agent-design/08-feedback-collection.md) · [`09`](agent-design/09-e2e-testing-gradio.md)

---

### 🗺️ Agent Framework Landscape — Day 8 ✅ NEW
- **Framework survey**: LangGraph, CrewAI, OpenAI Agents SDK, Claude SDK, Google ADK, AWS Strands — philosophy, primitives, and trade-offs
- **Agentic concepts**: workflows vs agents, autonomy spectrum, orchestration patterns
- **Hello-world in LangGraph**: full ReAct loop with tool calling, checkpointer, state schema (~60 lines)
- **Hello-world in CrewAI**: researcher + writer crew, sequential process, custom tools (~35 lines)
- **Developer experience comparison**: lines of code, learning curve, debugging, state control, HITL, community
- **Feature matrix**: state management, tool calling, memory, streaming, multi-agent, HITL, observability, deployment, model agnosticism across all 6 frameworks
- **Side-by-side code**: same two-step agent built in both LangGraph and CrewAI
- **Model Context Protocol (MCP)**: open standard for tool interoperability
- **Selection criteria**: decision tree for choosing the right framework per use case
- **Deliverable**: Framework comparison document with code samples, production checklist, and recommendation table

> 📄 Summary: [`days/DAY-8.md`](days/DAY-8.md) | Deep dives: [`agent-design/10`](agent-design/10-langgraph-hello-world.md) · [`11`](agent-design/11-crewai-hello-world.md) · [`12`](agent-design/12-framework-feature-matrix.md) · [`13`](agent-design/13-framework-comparison-doc.md)

---

### 🤖 Agentic AI Systems (upcoming)
- Multi-agent architecture patterns: Supervisor, Swarm, Debate, Graph-based (Day 9)
- Evaluation & metrics: LLM-as-judge, golden datasets, DeepEval (Day 10)
- Observability & production debugging: Langfuse, LangSmith, cost tracking (Day 11)
- Responsible AI & guardrails: prompt injection defense, PII masking, NeMo Guardrails (Day 12)
- Capstone: Multi-Agent Customer Support System (Days 13–14)

---

### 🔎 Retrieval-Augmented Generation (upcoming)
- Embeddings and semantic search
- Vector databases
- Chunking and retrieval strategies
- Improving response accuracy

---

## 🚀 How to Use This Repo

Each file is written in a **clear, structured format** to maximize understanding and retention.

**Suggested learning path:**
```text
days/DAY-X.md        ← start here each day (summary + key numbers)
      ↓
foundations/         ← Day 1 deep dives
      ↓
prompting/           ← Day 2 deep dives
      ↓
memory/              ← Day 3 deep dives
      ↓
tool-calling/        ← Day 4 deep dives
      ↓
hitl/                ← Day 5 deep dives
      ↓
agent-design/        ← Days 6–8 deep dives (01–13)
      ↓
rag/                 ← upcoming
      ↓
agents/              ← upcoming
      ↓
experiments/         ← hands-on builds
```

---

## 🧪 Learning Approach

- Focus on **first principles**
- Break down concepts into **simple mental models**
- Reinforce learning through **experiments**
- Connect theory with **real-world use cases**
- Senior engineer Q&A format — concept + trade-offs + numbers + real examples

---

## 🗺️ Roadmap

**Foundations — Day 1** ✅
**Prompt Engineering — Day 2** ✅
**Memory & State Management — Day 3** ✅
**Tool Calling & Function Integration — Day 4** ✅
**Human-in-the-Loop & Interrupts — Day 5** ✅
**Build Your First Agent — Design & Setup — Day 6** ✅
**Complete Agent — HITL, Observability & Polish — Day 7** ✅
**Agent Framework Landscape — Day 8** ✅ NEW
- [x] Framework survey: LangGraph, CrewAI, OpenAI Agents SDK, Claude SDK, Google ADK, AWS Strands
- [x] Workflows vs agents — autonomy spectrum and decision criteria
- [x] LangGraph hello-world: ReAct loop, tool calling, checkpointer
- [x] CrewAI hello-world: researcher/writer crew, sequential + hierarchical process
- [x] Developer experience comparison (lines, learning curve, debugging)
- [x] Full feature matrix: state, tools, memory, streaming, multi-agent, HITL, observability, deployment
- [x] Side-by-side code: same agent in LangGraph and CrewAI
- [x] Model Context Protocol (MCP) explainer
- [x] Framework selection decision tree
- [x] Deliverable: Comparison document with code samples

**Upcoming — Week 2**
- [ ] Day 9: Multi-Agent Architecture Patterns (Supervisor, Swarm, Debate, Graph)
- [ ] Day 10: Evaluation & Metrics (LLM-as-judge, golden datasets, DeepEval)
- [ ] Day 11: Observability & Production Debugging (Langfuse, cost dashboards)
- [ ] Day 12: Responsible AI & Guardrails (injection defense, PII masking, NeMo)
- [ ] Days 13–14: Capstone — Multi-Agent Customer Support System

**Projects**
- [ ] Personal Research Assistant (Days 6–7) ← in progress
- [ ] Implement RAG pipeline
- [ ] Multi-agent customer support system (capstone)

---

## 👤 Who This Is For

- Engineers getting started with **LLMs and Agentic AI**
- Developers transitioning into **AI Engineering roles**
- Anyone who wants a **hands-on, practical approach** to learning AI systems

---

## ⚠️ License & Attribution

**© 2026 Aniket Waichal. All rights reserved under CC BY-NC 4.0.**

This repository is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

**You are free to:**
- Read, study, and learn from this material
- Share it with proper attribution

**You are NOT allowed to:**
- Republish or redistribute this content as your own
- Use this material for commercial purposes
- Remove attribution and claim authorship

If you use or reference this work, you **must** credit:
```
Aniket Waichal — https://github.com/AniketWaichal/agentic-ai-playground
Licensed under CC BY-NC 4.0
```

See the [`LICENSE`](LICENSE) file for full terms.

---

## 🤝 Contributing

If you have suggestions, improvements, or ideas, feel free to open an issue or contribute — with attribution intact.