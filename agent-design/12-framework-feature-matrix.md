# Agent Framework Feature Matrix

> A detailed comparison of LangGraph, CrewAI, OpenAI Agents SDK, Claude SDK, Google ADK, and AWS Strands across every dimension that matters for production systems.

---

## At a Glance

| Framework | Author | Language | Stars (≈) | Abstraction Level | Primary Strength |
|---|---|---|---|---|---|
| LangGraph | LangChain | Python, JS/TS | 10k+ | Low–Medium | Graph-based control flow, HITL |
| CrewAI | CrewAI Inc | Python | 25k+ | High | Role-based teams, rapid POC |
| OpenAI Agents SDK | OpenAI | Python | Official | Medium | Handoff routing, native OpenAI |
| Claude SDK | Anthropic | Python, JS | Official | Low | Tool use, safety, MCP, thinking |
| Google ADK | Google | Python | Official | Medium | GCP/Gemini, workflow composition |
| AWS Strands | AWS | Python | Official | Low–Medium | Bedrock multi-model, AWS-native |

---

## 1. State Management

| Framework | State Primitive | Persistence | Time-Travel | Notes |
|---|---|---|---|---|
| **LangGraph** | `TypedDict` + reducers | SqliteSaver, PostgresSaver, MemorySaver | ✅ Yes | Full checkpoint replay, thread_id scoping |
| **CrewAI** | Internal dict | SQLite (long-term memory) | ❌ No | State opaque to developer |
| **OpenAI Agents SDK** | Message list (RunContext) | In-memory (BYO persistence) | ❌ No | Lightweight; add your own DB |
| **Claude SDK** | Raw `messages` array | None built-in | ❌ No | Developer owns all state |
| **Google ADK** | `Session` (InMemorySessionService / DB) | Pluggable session services | ❌ No | Vertex AI managed storage |
| **AWS Strands** | `AgentContext` dict | In-memory (BYO DynamoDB/S3) | ❌ No | Integrates with AWS services |

**Winner for state control:** LangGraph (most expressive, best persistence).

---

## 2. Tool Calling

| Framework | Tool Definition | Parallel Calls | Error Handling | Schema Validation |
|---|---|---|---|---|
| **LangGraph** | LangChain `@tool` or `StructuredTool` | ✅ Native (ToolNode) | Manual in node | Pydantic via args_schema |
| **CrewAI** | `@tool`, `BaseTool`, crewai_tools | ✅ (per agent turn) | Auto-retry via max_iter | Pydantic BaseModel |
| **OpenAI Agents SDK** | `@function_tool`, `FunctionTool` | ✅ Native | Built-in try/catch wrapper | Pydantic / JSON Schema |
| **Claude SDK** | JSON schema dict | ✅ Yes | Manual tool_result error field | JSON Schema |
| **Google ADK** | `@tool`, `FunctionTool`, built-in tools | ✅ Yes | Error propagation in tool_result | Pydantic |
| **AWS Strands** | `@tool` decorator | ✅ Yes | Exception → error message | Pydantic / docstring |

**Notable:** AWS Strands' `@tool` is the simplest DX — just decorate any function. Claude SDK is the most verbose (raw JSON schema).

---

## 3. Memory

| Framework | Short-Term | Long-Term | Semantic/Vector | Entity Memory |
|---|---|---|---|---|
| **LangGraph** | State `messages` | BYO (LangChain memory, ChromaDB) | BYO | BYO |
| **CrewAI** | Context window | SQLite (built-in) | ChromaDB (built-in) | ✅ Built-in |
| **OpenAI Agents SDK** | Message list | BYO | BYO | BYO |
| **Claude SDK** | `messages` array | BYO | BYO | BYO |
| **Google ADK** | Session context | Cloud Firestore (via service) | Vertex Matching Engine | BYO |
| **AWS Strands** | `AgentContext` | BYO (DynamoDB recommended) | BYO (OpenSearch) | BYO |

**Winner for built-in memory:** CrewAI (most batteries included). **Most flexible:** LangGraph (BYO everything, full control).

---

## 4. Streaming

| Framework | Streaming Type | Granularity | Notes |
|---|---|---|---|
| **LangGraph** | `agent.astream()`, `astream_events()` | Token + event-level | Full event taxonomy: `on_llm_stream`, `on_tool_start`, `on_chain_end` |
| **CrewAI** | `step_callback` + partial streaming | Step-level | Token streaming via callback |
| **OpenAI Agents SDK** | `Runner.run_streamed()` | Token + event-level | `StreamEvent` taxonomy, SSE-ready |
| **Claude SDK** | `client.messages.stream()` | Token-level | `with client.messages.stream() as s:` context manager |
| **Google ADK** | SSE via `run_sse()` | Token + event-level | Built for web streaming |
| **AWS Strands** | Generator via `agent.stream_async()` | Token-level | Bedrock streaming passthrough |

---

## 5. Multi-Agent Support

| Framework | Pattern | Handoff | Shared State | Parallel Agents |
|---|---|---|---|---|
| **LangGraph** | Subgraph nodes, `Command` routing | Manual graph edges | ✅ Shared state | ✅ Via parallel branches |
| **CrewAI** | Crew with delegation | `allow_delegation=True` | Crew memory | Limited (sequential/hierarchical) |
| **OpenAI Agents SDK** | `handoff(agent)` | ✅ First-class | RunContext passed on handoff | ❌ Sequential |
| **Claude SDK** | BYO orchestration | Manual | BYO | BYO |
| **Google ADK** | `agent_as_tool`, `ParallelAgent` | Via tool call | Session shared | ✅ ParallelAgent |
| **AWS Strands** | `agents_as_tools` | Via tool call | BYO | ❌ Sequential |

---

## 6. Human-in-the-Loop (HITL)

| Framework | Native HITL | Mechanism | Audit Trail |
|---|---|---|---|
| **LangGraph** | ✅ First-class | `interrupt()` + checkpointer | ✅ Full graph state snapshot |
| **CrewAI** | ❌ Limited | `human_input=True` on task (blocks) | ❌ None built-in |
| **OpenAI Agents SDK** | ❌ Not native | Lifecycle hooks (`on_tool_call`) | Partial |
| **Claude SDK** | ❌ Not native | BYO | BYO |
| **Google ADK** | ❌ Limited | Callback hooks | Partial |
| **AWS Strands** | ❌ Not native | BYO | BYO |

**Winner:** LangGraph is the only framework with production-grade HITL.

---

## 7. Observability / Tracing

| Framework | Built-in Tracing | Third-Party | Cost Tracking |
|---|---|---|---|
| **LangGraph** | LangSmith (native) | Langfuse via callbacks | ✅ Via LangSmith |
| **CrewAI** | Basic verbose logs | Langfuse, Helicone | Limited |
| **OpenAI Agents SDK** | ✅ First-class tracing (trace ID, spans) | OpenTelemetry export | ✅ Via dashboard |
| **Claude SDK** | ❌ None | BYO (Langfuse, custom JSONL) | Manual |
| **Google ADK** | Cloud Trace (GCP) | OpenTelemetry | Cloud Billing |
| **AWS Strands** | CloudWatch (BYO setup) | OpenTelemetry | AWS Cost Explorer |

---

## 8. Deployment Options

| Framework | Managed Hosting | Containerise | Serverless | Cloud-Native |
|---|---|---|---|---|
| **LangGraph** | LangGraph Cloud | ✅ Docker | ✅ Lambda/Cloud Run | Cloud-agnostic |
| **CrewAI** | CrewAI Enterprise | ✅ Docker | ✅ Any | Cloud-agnostic |
| **OpenAI Agents SDK** | ❌ Self-host only | ✅ | ✅ | OpenAI-centric |
| **Claude SDK** | ❌ Self-host only | ✅ | ✅ | Cloud-agnostic |
| **Google ADK** | ✅ Agent Engine (Vertex AI) | ✅ Cloud Run | ✅ Cloud Functions | GCP-native |
| **AWS Strands** | ✅ Bedrock Agents (partial) | ✅ ECS/Lambda | ✅ Lambda | AWS-native |

---

## 9. Model Agnosticism

| Framework | Primary Model | Other Models | Notes |
|---|---|---|---|
| **LangGraph** | Any (LangChain) | ✅ All | Model-agnostic by design |
| **CrewAI** | OpenAI default | ✅ Ollama, Groq, Anthropic, etc | Via LiteLLM integration |
| **OpenAI Agents SDK** | GPT-4o | ✅ Any OpenAI-compatible | Best with OpenAI |
| **Claude SDK** | Claude 3.x | ❌ Anthropic only | Single-provider |
| **Google ADK** | Gemini | Partial (LiteLLM) | Best with Gemini |
| **AWS Strands** | Bedrock (Claude, Llama, etc) | ✅ OpenAI via provider | Multi-model by design |

---

## 10. Developer Experience Summary

| Framework | Lines (hello-world) | Learning Curve | Testing | Docs Quality |
|---|---|---|---|---|
| **LangGraph** | ~60 | Steep | Excellent (mock state injection) | Good |
| **CrewAI** | ~30 | Shallow | Medium (task-level mocks) | Good |
| **OpenAI Agents SDK** | ~25 | Low | Good (Runner.run_sync in tests) | Excellent |
| **Claude SDK** | ~20 | Low | Good (tool mocking) | Excellent |
| **Google ADK** | ~35 | Medium | Good (InMemory services) | Good |
| **AWS Strands** | ~20 | Low | Good (mock Bedrock) | Good |

---

## Quick Selection Guide

```
Complex state + HITL + cycles?            → LangGraph
Team-style POC, fast iteration?           → CrewAI
OpenAI stack, handoff routing?            → OpenAI Agents SDK
Anthropic Claude, safety-critical?        → Claude SDK + LangGraph orchestration
GCP / Gemini, managed hosting?            → Google ADK
AWS / Bedrock, multi-model production?    → AWS Strands
Research / model comparison / multi-cloud? → LangGraph (most flexible)
```