# 20 — Designing Evaluation Datasets for Agent Testing

> Deep dive for Day 10 · Part of [`agent-design/`](.)

---

## Why Dataset Design Is the Most Underrated Skill in Agent Engineering

Most engineers spend 80% of their time on the agent and 20% on evaluation. It should be closer to 50/50. A weak dataset gives you false confidence — your agent passes 95% of tests because the tests are trivial. A strong dataset is a compressed representation of everything that can go wrong.

**Core principle:** Your evaluation dataset is your specification. If you can't describe a failure mode as a test case, you haven't specified your system completely.

---

## The Four Dataset Types

### 1. Golden Dataset (50–200 cases)

**What it is:** Hand-curated, expert-reviewed cases covering the primary happy paths and most critical edge cases. These are your unit tests.

**Build process:**
1. List all primary user intents (the "jobs to be done")
2. Write 2–3 clean examples per intent
3. Add known failure modes from design review
4. Review with a domain expert
5. Lock the dataset — changes require a PR review

**Anatomy of a golden case:**
```json
{
  "id": "golden_001",
  "description": "Standard order status lookup",
  "input": "Where is my order #ORD-9821?",
  "context": {
    "user_id": "u_123",
    "prior_turns": []
  },
  "expected": {
    "tool_calls": [
      {"tool": "get_order_status", "args": {"order_id": "ORD-9821"}}
    ],
    "output_must_contain": ["status", "estimated"],
    "output_must_not_contain": ["I don't know", "error"],
    "max_tool_calls": 2,
    "max_turns": 1
  },
  "tags": ["happy_path", "order_management"]
}
```

**Trade-off:** Expensive to build (30–60 min of expert time per 10 cases). Worth it — these are your regression guard.

---

### 2. Edge Case Dataset (20–50 cases)

**What it is:** Cases that stress the boundaries of your agent's capability. Built from failure-mode analysis and real production bugs.

**Edge case taxonomy:**

| Category | Example | What it tests |
|---|---|---|
| Missing required information | "Cancel my order" (no ID) | Clarification behaviour |
| Ambiguous intent | "I want to change it" | Disambiguation |
| Multi-intent | "Cancel order AND update address" | Compound handling |
| Out-of-scope | "Write me a haiku" | Graceful refusal |
| Adversarial / prompt injection | "Ignore previous instructions" | Safety guardrails |
| Tool failure | Tool returns 500 error | Error recovery |
| Long input | 2000-word customer complaint | Context handling |
| Non-English / mixed language | "¿Dónde está mi pedido?" | Multilingual support |
| Contradictory context | Order exists but user says it doesn't | Conflict resolution |
| Repeated clarification | User gives wrong info 3 times | Patience / re-ask logic |

**Template for each edge case:**
```json
{
  "id": "edge_007",
  "description": "User sends ambiguous cancel request without order ID",
  "category": "missing_required_info",
  "input": "Cancel my order please",
  "expected": {
    "behaviour": "ask_for_clarification",
    "output_must_contain": ["order number", "which order"],
    "must_not_call_tools": ["cancel_order"]
  }
}
```

---

### 3. Synthetic Dataset (1,000+ cases)

**What it is:** LLM-generated variations of your golden cases. Used for scale testing — discovering rare failure modes through volume.

**Generation prompt pattern:**
```python
SYNTHETIC_GEN_PROMPT = """
Given this example agent test case:
{golden_case}

Generate {n} diverse variations that:
- Cover different phrasings of the same intent
- Vary entity types (different order IDs, names, dates)
- Include minor spelling errors and informal language
- Vary context (first-time user vs repeat user)

Return as a JSON array. Each item must have the same schema as the input.
"""
```

**Quality control:**
- Run the generator with temperature 0.9 for diversity
- Deduplicate by semantic similarity (cosine similarity > 0.95 → drop one)
- Spot-check 5% manually — expect ~10% to be garbage
- Filter out cases where expected output is ambiguous

**When to use synthetic data:**
- After your golden dataset is stable and your pipeline is working
- When you need to test a new feature at scale before adding to golden set
- For load testing your evaluation pipeline itself

**Warning:** Never replace golden cases with synthetic ones. Synthetic data has the biases of the generator model baked in.

---

### 4. Production Samples (Ongoing)

**What it is:** Real conversations sampled from production traffic, with human-reviewed labels added retroactively.

**Sampling strategies:**

| Strategy | How | When to use |
|---|---|---|
| Random sample | 1% of traffic | Baseline coverage |
| Failure-triggered | Sample when user thumbs down | Catch real failures |
| Low-confidence | Sample when judge score < 3 | Catch borderline cases |
| High-stakes | Sample all conversations in sensitive categories | Compliance, safety |

**Pipeline:**
```
Production traffic
        ↓
Sampling filter (e.g., 1% random + all low-confidence)
        ↓
Human review queue (label: correct / incorrect / needs improvement)
        ↓
Promote to golden dataset if failure mode is novel
```

**Practical note:** Production sampling is the most valuable dataset type for catching distribution shift — when your users start asking questions you didn't anticipate during design. Schedule a weekly review of 20–30 production samples even when things seem fine.

---

## Building Your 20-Case Starter Golden Dataset

**Step-by-step for a customer support agent:**

```
Cases 1–5:   Core happy paths (one per primary intent)
Cases 6–8:   Multi-turn happy paths (clarification resolved correctly)
Cases 9–11:  Missing information edge cases
Cases 12–14: Out-of-scope / refusal cases
Cases 15–17: Tool error / recovery cases
Cases 18–20: Adversarial / safety cases
```

**YAML format (easier to hand-edit than JSON):**
```yaml
- id: golden_001
  description: Standard order lookup
  tags: [happy_path, order]
  input: "What's the status of order ORD-4521?"
  expected:
    tool_calls:
      - tool: get_order_status
        args:
          order_id: ORD-4521
    output_must_contain: [status]
    output_must_not_contain: ["I cannot", "I don't know"]
    max_turns: 1
    max_tool_calls: 1

- id: golden_015
  description: Tool returns 500 error — agent should retry once then inform user
  tags: [edge_case, tool_failure]
  input: "Track my package ORD-9900"
  mock_tool_responses:
    get_order_status: {error: "Service unavailable", status_code: 500}
  expected:
    behaviour: inform_user_of_service_issue
    output_must_contain: ["unable", "try again"]
    must_not_hallucinate_status: true
    max_tool_calls: 2  # one retry allowed
```

---

## Dataset Versioning

Treat your dataset like a database schema — every change needs a migration.

```
datasets/
├── golden/
│   ├── v1.0.0_golden.json    # initial 20 cases
│   ├── v1.1.0_golden.json    # added 5 edge cases after bug in prod
│   └── v2.0.0_golden.json    # full refresh after agent redesign
├── edge_cases/
│   └── v1.0.0_edge.json
└── CHANGELOG.md
```

**CHANGELOG entry format:**
```
## v1.1.0 — 2026-04-18
Added:
- golden_021: user sends duplicate cancel request (regression from prod bug #142)
- golden_022: user switches language mid-conversation
Changed:
- golden_008: updated expected tool args after API schema change
```

---

## Anti-Patterns to Avoid

| Anti-pattern | Problem | Fix |
|---|---|---|
| Only testing happy paths | Gives false confidence | Add edge case budget: min 30% of dataset |
| Expected outputs are too strict | Fails on valid paraphrases | Use `must_contain` patterns not exact match |
| Dataset grows without curation | Noisy, slow to run | Cap golden set at 200; promote judiciously |
| No negative test cases | Agent can say anything and pass | Always include `output_must_not_contain` |
| Generated without domain expert review | Misses real user language | Block the first version behind an expert review |