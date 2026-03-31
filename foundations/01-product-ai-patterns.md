## 1. Product AI Patterns

### Q1: What is the difference between "Assist" and "Automate" AI patterns, and when do you choose one over the other?

**Answer:**

The **Assist** pattern keeps a human in the loop for every significant decision. The AI drafts, suggests, or flags — but a human reviews before anything is committed. Think GitHub Copilot autocompleting code: you accept or reject each suggestion.

The **Automate** pattern lets the AI execute end-to-end with no human touch per transaction. Think of an AI agent that triages support tickets, routes them, and auto-responds to a class of simple queries.

**Decision framework:**

| Dimension | Assist | Automate |
|-----------|--------|----------|
| Error cost | High (medical, legal, financial) | Low (low-stakes routing, classification) |
| User trust | Users want control | Users want speed |
| Edge case frequency | High/unpredictable | Low and known |
| Latency tolerance | Human-speed (seconds to minutes) | Sub-second preferred |
| Reversibility | Irreversible actions (payments, deletions) | Easily undoable |

**Real-world examples:**

- **Assist:** Notion AI writes a draft — the user edits before publishing. Gmail Smart Compose suggests — the user accepts word by word.
- **Automate:** Intercom's Fin resolves 40–60% of chat tickets autonomously. AWS Lambda functions auto-provisioned by AI ops tools.

**Key trade-off:** Automation maximizes throughput and cost savings but amplifies errors at scale. A bug in an assisted system affects one user per mistake; a bug in an automated system can fire 10,000 wrong emails before anyone notices.

**Senior engineer tip:** Always design the Automate path with a "confidence threshold" gate — if the model's confidence is below 0.80 (or whatever your calibration shows), fall back to Assist. This is called **graceful degradation**.

---

### Q2: What are the main UX flows in AI-powered products, and what failure modes must you design for?

**Answer:**

**Common UX flows:**

1. **Inline suggestion** (Copilot-style): AI completes as you type. Latency must be < 200ms for it to feel responsive.
2. **Chat/turn-based**: User sends a query, AI responds. The round-trip latency budget is typically < 3 seconds for user satisfaction.
3. **Background agent**: AI runs autonomously and surfaces results. User sees output, not process.
4. **Approval-gated**: AI proposes, human approves before execution (e.g., before sending an email, before a database write).

**Failure modes and how to handle them:**

| Failure Mode | Description | Mitigation |
|---|---|---|
| Hallucination | Model states false information confidently | RAG grounding, citation requirements, factuality checks |
| Latency spikes | Model slow under load | Streaming responses, P95 SLA monitoring, fallback to smaller model |
| Prompt injection | User manipulates system prompt | Input sanitization, XML tag separation, instruction hierarchy |
| Overconfident wrong answers | Model says "definitely X" when it's wrong | Calibrated confidence scoring, hedging language in prompts |
| Context length overflow | Conversation exceeds window limit | Sliding window, summarization, retrieval |
| AI demo syndrome | Works in demos, breaks on real data | Regression test suites, golden datasets, production sampling |

**AI demo syndrome** deserves special attention: models often perform well on clean, structured demo inputs but fail on real-world messy data. Senior engineers build evaluation pipelines before shipping — not after.

---