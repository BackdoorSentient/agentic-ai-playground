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
│   ├── DAY-8.md                      # Agent Framework Landscape ✅
│   └── DAY-9.md                      # Multi-Agent Architecture Patterns ✅ NEW
│
├── agent-design/                     # Agent Design & Complete Build (Days 6–9) ✅
│   ├── 01-agent-architecture.md
│   ├── 02-state-schema-langgraph.md
│   ├── 03-tools-react-memory.md
│   ├── 04-observability-logging.md
│   ├── 05-hitl-approval-integration.md
│   ├── 06-conversation-summarization.md
│   ├── 07-observability-logging.md
│   ├── 08-feedback-collection.md
│   ├── 09-e2e-testing-gradio.md
│   ├── 10-langgraph-hello-world.md
│   ├── 11-crewai-hello-world.md
│   ├── 12-framework-feature-matrix.md
│   ├── 13-framework-comparison-doc.md
│   ├── 14-supervisor-pattern.md          ← NEW Day 9
│   ├── 15-agents-as-tools.md             ← NEW Day 9
│   ├── 16-swarm-pattern.md               ← NEW Day 9
│   ├── 17-graph-pattern.md               ← NEW Day 9
│   ├── 18-debate-consensus.md            ← NEW Day 9
│   └── 19-multi-agent-hands-on.md        ← NEW Day 9
```

---

## 📚 What I'm Learning

### 🧠 LLM Foundations — Day 1 ✅
> 📄 Summary: [`days/DAY-1.md`](days/DAY-1.md) | Deep dives: [`foundations/`](foundations/)

### ✍️ Prompt Engineering — Day 2 ✅
> 📄 Summary: [`days/DAY-2.md`](days/DAY-2.md) | Deep dives: [`prompting/`](prompting/)

### 🧠 Memory & State Management — Day 3 ✅
> 📄 Summary: [`days/DAY-3.md`](days/DAY-3.md) | Deep dives: [`memory/`](memory/)

### 🔧 Tool Calling & Function Integration — Day 4 ✅
> 📄 Summary: [`days/DAY-4.md`](days/DAY-4.md) | Deep dives: [`tool-calling/`](tool-calling/)

### 🛑 Human-in-the-Loop & Interrupts — Day 5 ✅
> 📄 Summary: [`days/DAY-5.md`](days/DAY-5.md) | Deep dives: [`hitl/`](hitl/)

### 🏗️ Build Your First Agent — Day 6 ✅
> 📄 Summary: [`days/DAY-6.md`](days/DAY-6.md) | Deep dives: [`agent-design/01–09`](agent-design/)

### 🚀 Complete Agent — HITL, Observability & Polish — Day 7 ✅
> 📄 Summary: [`days/DAY-7.md`](days/DAY-7.md) | Deep dives: [`agent-design/05–09`](agent-design/)

### 🗺️ Agent Framework Landscape — Day 8 ✅
> 📄 Summary: [`days/DAY-8.md`](days/DAY-8.md) | Deep dives: [`agent-design/10–13`](agent-design/)

---

### 🤝 Multi-Agent Architecture Patterns — Day 9 ✅ NEW
- **Supervisor / Orchestrator pattern**: central LLM routes tasks to specialist workers, top-down control
- **Agents-as-Tools pattern**: sub-agents wrapped as callable functions — composable, swappable
- **Swarm / Peer-to-Peer pattern**: autonomous agents hand off to each other via structured HANDOFF tokens
- **Graph pattern (LangGraph)**: typed state machine with nodes/edges/checkpointers for complex workflows
- **Debate / Consensus pattern**: multi-agent critique + judge → 15–25% fewer factual errors
- **Hands-on #1**: Supervisor system routing to Coding Agent and Research Agent (LangGraph graph)
- **Hands-on #2**: Two debater agents + judge with 2-round structured debate and JSON verdict
- **Hands-on #3**: Three-agent swarm chain: intake → specialist → quality agent
- **Deliverable**: Full working multi-agent system (supervisor + swarm + debate) with logging and guards

> 📄 Summary: [`days/DAY-9.md`](days/DAY-9.md) | Deep dives: [`agent-design/14`](agent-design/14-supervisor-pattern.md) · [`15`](agent-design/15-agents-as-tools.md) · [`16`](agent-design/16-swarm-pattern.md) · [`17`](agent-design/17-graph-pattern.md) · [`18`](agent-design/18-debate-consensus.md) · [`19`](agent-design/19-multi-agent-hands-on.md)

---

### 🤖 Agentic AI Systems (upcoming)
- Evaluation & metrics: LLM-as-judge, golden datasets, DeepEval (Day 10)
- Observability & production debugging: Langfuse, LangSmith, cost tracking (Day 11)
- Responsible AI & guardrails: prompt injection defense, PII masking, NeMo Guardrails (Day 12)
- Capstone: Multi-Agent Customer Support System (Days 13–14)

---

## 🗺️ Roadmap

**Foundations — Day 1** ✅
**Prompt Engineering — Day 2** ✅
**Memory & State Management — Day 3** ✅
**Tool Calling & Function Integration — Day 4** ✅
**Human-in-the-Loop & Interrupts — Day 5** ✅
**Build Your First Agent — Design & Setup — Day 6** ✅
**Complete Agent — HITL, Observability & Polish — Day 7** ✅
**Agent Framework Landscape — Day 8** ✅
**Multi-Agent Architecture Patterns — Day 9** ✅ NEW
- [x] Supervisor / Orchestrator pattern — mechanics, routing, parallel workers
- [x] Agents-as-Tools — composable sub-agent delegation via function calls
- [x] Swarm / Peer-to-Peer — autonomous handoffs with HANDOFF token protocol
- [x] Graph pattern (LangGraph) — typed state machine, fan-out, HITL nodes
- [x] Debate / Consensus — adversarial debate, voting, constitutional self-critique
- [x] Hands-on: Supervisor system (coding + research agents)
- [x] Hands-on: Debate system (2 debaters + judge, JSON verdict)
- [x] Hands-on: Swarm chain (intake → specialist → quality agent)
- [x] Deliverable: Complete multi-agent system with guards + logging

**Upcoming — Week 2**
- [ ] Day 10: Evaluation & Metrics (LLM-as-judge, golden datasets, DeepEval)
- [ ] Day 11: Observability & Production Debugging (Langfuse, cost dashboards)
- [ ] Day 12: Responsible AI & Guardrails (injection defense, PII masking, NeMo)
- [ ] Days 13–14: Capstone — Multi-Agent Customer Support System

---

## 🧪 Learning Approach

- Focus on **first principles**
- Break down concepts into **simple mental models**
- Reinforce learning through **experiments**
- Connect theory with **real-world use cases**
- Senior engineer Q&A format — concept + trade-offs + numbers + real examples

---

## ⚠️ License & Attribution

**© 2026 Aniket Waichal. All rights reserved under CC BY-NC 4.0.**

Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
If you use or reference this work, credit: `Aniket Waichal — https://github.com/AniketWaichal/agentic-ai-playground`