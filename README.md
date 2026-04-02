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
│   ├── DAY-1.md                      # LLM Foundations
│   ├── DAY-2.md                      # Prompt Engineering
│   ├── DAY-3.md                      # Memory & State Management ✅
│   └── DAY-4.md                      # Tool Calling & Function Integration ✅ NEW
│
├── foundations/                      # Core LLM and AI fundamentals
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
├── prompting/                        # Prompt engineering techniques
│   ├── 01-zero-shot-and-few-shot.md
│   ├── 02-chain-of-thought.md
│   ├── 03-react-and-reflexion.md
│   ├── 04-system-prompt-design.md
│   └── 05-structured-outputs-and-validation.md
│
├── memory/                           # Memory & State Management (Day 3) ✅
│   ├── 01-memory-taxonomy.md
│   ├── 02-conversation-memory-summarization.md
│   ├── 03-state-schemas-agent-workflows.md
│   └── 04-checkpointing-durable-execution.md
│
├── tool-calling/                     # Tool Calling & Function Integration (Day 4) ✅ NEW
│   ├── 01-tool-schema-design.md
│   ├── 02-tool-calling-mechanics.md
│   ├── 03-tool-selection-strategies.md
│   └── 04-error-handling-retry-logic.md
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

### 🔧 Tool Calling & Function Integration — Day 4 ✅ NEW
- Tool schema design: writing definitions LLMs can reliably invoke across OpenAI, Claude, and open-source models
- Tool calling mechanics: the full request/response cycle, parallel tool calls, ToolNode pattern
- Tool selection strategies: all-in-context (<10), categorized routing (10–50), RAG retrieval (50+), hierarchical agents (100+)
- Error handling & retry logic: exponential backoff, structured errors, fallback hierarchy, circuit breakers
- Hands-on: Multi-tool agent, error handling lab, tool tracing dashboard

> 📄 Summary: [`days/DAY-4.md`](days/DAY-4.md) | Deep dives: [`tool-calling/`](tool-calling/)

---

### 🤖 Agentic AI Systems (upcoming)
- Human-in-the-Loop (HITL) patterns
- Planning and reasoning workflows
- Multi-agent architectures

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
foundations/         ← Week 1 deep dives
      ↓
prompting/           ← Week 1 deep dives
      ↓
memory/              ← Day 3 deep dives
      ↓
tool-calling/        ← Day 4 deep dives
      ↓
rag/                 ← upcoming
      ↓
agents/              ← upcoming
      ↓
experiments/         ← hands-on builds
```

Start with the `days/DAY-X.md` summary file for each day to get the full picture and key numbers, then go deep into the individual topic files in each folder.

---

## 🧪 Learning Approach

- Focus on **first principles**
- Break down concepts into **simple mental models**
- Reinforce learning through **experiments**
- Connect theory with **real-world use cases**
- Senior engineer Q&A format — concept + trade-offs + numbers + real examples

---

## 🗺️ Roadmap

**Foundations — Day 1**
- [x] Product AI Patterns (Assist vs Automate)
- [x] Temperature & Top-p
- [x] Context Window
- [x] Pre-training
- [x] Fine-tuning & LoRA / QLoRA
- [x] RLHF & DPO
- [x] Structured Outputs & Pydantic
- [x] RAG Architecture & Evaluation
- [x] FastAPI Framework

**Prompt Engineering — Day 2**
- [x] Zero-Shot & Few-Shot Prompting
- [x] Chain-of-Thought (CoT, Self-consistency, ToT)
- [x] ReAct & Reflexion Patterns
- [x] System Prompt Design for Agents
- [x] Structured Outputs: JSON / YAML / TOON
- [x] Pydantic Validation & Self-healing Retry

**Memory & State Management — Day 3**
- [x] Memory Taxonomy (short-term, long-term, episodic, semantic, procedural)
- [x] Conversation Memory with Summarization (context overflow)
- [x] State Schemas for Complex Agent Workflows (TypedDict + Pydantic)
- [x] Checkpointing for Durable Agent Execution (LangGraph SqliteSaver + HITL)
- [ ] Deliverable: Memory-enabled chatbot with persistent user preferences

**Tool Calling & Function Integration — Day 4**
- [x] Tool Schema Design (OpenAI, Claude, LangChain — provider-agnostic)
- [x] Tool Calling Mechanics (request/response cycle, parallel calls, ToolNode)
- [x] Tool Selection Strategies (all-in-context → routing → RAG → hierarchical)
- [x] Error Handling & Retry Logic (backoff, structured errors, circuit breaker)
- [ ] Deliverable: Multi-tool agent with error handling and tracing

**Agentic Systems — Day 5+**
- [ ] Human-in-the-Loop (HITL) & Interrupts
- [ ] Multi-agent workflows
- [ ] Agent evaluation and observability

**Projects**
- [ ] Build a basic AI assistant
- [ ] Implement RAG pipeline
- [ ] Create an agent with tools
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