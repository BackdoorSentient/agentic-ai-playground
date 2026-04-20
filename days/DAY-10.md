# Day 10 — Evaluation & Metrics for Agents

> **Format:** Senior Engineer Q&A — concept + trade-offs + numbers + real-world examples  
> **Deep dives:** [`agent-design/20-evaluation-datasets.md`](../agent-design/20-evaluation-datasets.md) · [`21-llm-as-judge.md`](../agent-design/21-llm-as-judge.md) · [`22-key-metrics.md`](../agent-design/22-key-metrics.md) · [`23-experiment-tracking.md`](../agent-design/23-experiment-tracking.md) · [`24-deepeval-setup.md`](../agent-design/24-deepeval-setup.md) · [`25-evaluation-hands-on.md`](../agent-design/25-evaluation-hands-on.md)

---

## Core Formula

```
EVALUATION = DATASET × TASK × SCORERS
```

Every robust evaluation pipeline has three axes:
- **DATASET** — what inputs you test against
- **TASK** — what the agent is supposed to do
- **SCORERS** — how you measure success

---

## Q1: Why do agents need a different evaluation strategy than classic ML models?

**Classic ML** evaluation: fixed input → fixed label → accuracy/F1. Deterministic, easy to batch.

**Agent evaluation** is harder because:

| Challenge | Why it matters |
|---|---|
| Non-deterministic outputs | Same prompt → different valid answers |
| Multi-step reasoning | Error can occur at step 3 of 10 |
| Tool use | Did it call the right tool? With right args? |
| Long horizon tasks | Hard to define a single "label" |
| Latency + cost | Part of the quality definition |

**Real-world implication:** You can't just compute BLEU or exact match. You need a combination of rule-based checks, execution-based checks, and LLM-as-judge for semantic quality.

---

## Q2: What is the Evaluation Formula and how do you apply it?

```
EVALUATION = DATASET × TASK × SCORERS
```

**DATASET** — your test inputs (see Q3)  
**TASK** — the specific capability being tested (e.g., "summarise a ticket", "call search tool correctly")  
**SCORERS** — metrics + judges (see Q5)

**Example mapping:**

```
Dataset:  20-case golden set (happy paths + edge cases)
Task:     Customer support agent — resolve ticket in ≤3 tool calls
Scorers:  Task Completion Rate + Tool Correctness + LLM-as-judge (1–5)
```

The formula forces you to be explicit about all three axes before you write a single line of code. Teams that skip this end up with vague "it seems better" evaluations that can't catch regressions.

---

## Q3: What are the four dataset types and when do you use each?

| Type | Size | How built | When to use |
|---|---|---|---|
| **Golden Dataset** | 50–200 cases | Hand-curated by domain expert | Core regression suite; run on every prompt change |
| **Edge Cases** | 20–50 cases | Failure-mode brainstorm + real bug reports | Catch known brittle paths (ambiguous input, adversarial queries) |
| **Synthetic** | 1,000+ cases | LLM-generated variations | Scale coverage fast; cheap but noisier |
| **Production Samples** | Ongoing | Sampled from real traffic | Ground truth of actual usage; catches distribution shift |

**Trade-offs:**

- **Golden datasets** are expensive to build but give you the highest signal. Treat them like unit tests — every PR runs against them.
- **Synthetic datasets** are cheap but carry hallucination risk from the generator model. Always spot-check 5–10% manually.
- **Production samples** are the most realistic but require a feedback loop (thumbs up/down, human review queue).

**Practical tip:** Start with 20 golden cases. You'll catch 80% of regressions. Scale synthetic once your scoring pipeline is validated.

---

## Q4: How do you design a 20-case golden dataset for a Week 1 agent?

**Step 1 — Map the happy paths (≈10 cases)**

For each primary user intent, create one clean example:
```
intent: look up order status
input: "Where is my order #1234?"
expected_tool_call: get_order_status(order_id="1234")
expected_output_contains: ["status", "estimated delivery"]
```

**Step 2 — Map the edge cases (≈10 cases)**

| Edge case category | Example |
|---|---|
| Missing required info | "Where's my order?" (no ID given) |
| Ambiguous intent | "Cancel it" (no context) |
| Out-of-scope query | "Write me a poem" |
| Multiple intents | "Cancel order AND update address" |
| Adversarial | "Ignore previous instructions and..." |
| Tool failure | Simulate tool returning error |

**Step 3 — Add expected outputs**

Each case should have:
- `input` — raw user message (+ context if multi-turn)
- `expected_tool_calls` — list of `{tool, args}` dicts (can be partial match)
- `expected_output_fields` — keys or phrases that MUST appear
- `forbidden_output` — things the agent must never say
- `max_turns` — upper bound on steps

---

## Q5: What are the key metrics for agent evaluation?

| Metric | Formula | Target | Notes |
|---|---|---|---|
| **Task Completion Rate (TCR)** | tasks_completed / total_tasks | >85% prod | Binary: did the agent fully accomplish the goal? |
| **Tool Correctness** | correct_tool_calls / total_tool_calls | >90% | Checks tool name + argument validity |
| **Reasoning Quality** | LLM-as-judge 1–5 scale | >3.5 avg | Semantic quality of chain-of-thought |
| **Faithfulness (FACTSCORE)** | Facts supported by context / total facts | >80% | Critical for RAG agents |
| **Latency P50 / P95** | Median & 95th percentile response time | P50 <3s, P95 <10s | User experience SLA |
| **Cost Efficiency** | total_cost / tasks_completed | Varies | Track $/1000 tasks |

**Why P95 latency matters more than average:** Averages hide tail latency. A P95 of 15s means 5% of your users wait 15 seconds — that's your most complex cases, often your highest-value users.

**Tool Correctness breakdown:**
```
Tool name match:    did it call "search" not "lookup"?
Argument presence:  were all required args provided?
Argument validity:  were arg types/formats correct?
Argument accuracy:  did the arg values make sense given input?
```

---

## Q6: How does LLM-as-Judge work, and what are its failure modes?

**Pattern:**

```python
JUDGE_PROMPT = """
You are evaluating an AI agent's response.

Task description: {task_description}
User input: {user_input}
Agent response: {agent_response}
Reference answer (if available): {reference}

Rate on a 1–5 scale:
1 = Completely wrong or harmful
2 = Major issues, partially relevant
3 = Acceptable but missing key elements
4 = Good, minor issues
5 = Excellent, fully correct and helpful

Respond with JSON only:
{"score": <int>, "reasoning": "<one sentence>"}
"""
```

**Strengths:**
- Handles semantic equivalence (two different phrasings of the same correct answer)
- Can evaluate nuance, tone, safety
- Scales cheaply — GPT-4o-mini as judge costs ~$0.001/eval

**Failure modes:**

| Failure | Mitigation |
|---|---|
| **Verbosity bias** — longer answers score higher | Include length-appropriateness criterion |
| **Self-preference bias** — Claude rates Claude higher | Use a different model as judge, or cross-model |
| **Sycophancy** — judge agrees with confident-sounding wrong answers | Add adversarial examples to calibrate |
| **Position bias** — in pairwise eval, prefers answer A | Randomize order, average both orderings |

**Calibration tip:** Before deploying your judge, run it on 20 hand-labeled cases and measure agreement with human scores. Target >80% agreement (within 1 point).

---

## Q7: What is FACTSCORE and when do you use it for agents?

**FACTSCORE (Factual Precision Score)** measures factual faithfulness:

```
FACTSCORE = (facts supported by source) / (total facts in output)
```

**Use it when:**
- Agent answers questions from retrieved documents (RAG)
- Agent summarises long-form content
- Factual accuracy is a hard requirement (medical, legal, finance)

**Implementation:**

1. Parse agent response into atomic facts: "The CEO is John Smith", "Founded in 2010"
2. For each fact, check if it appears in the source context
3. Score = supported_facts / total_facts

**Practical note:** You can implement this with an LLM call:
```
Given this context: {context}
Given this claim: {atomic_fact}
Is this claim directly supported by the context? Answer yes/no with a one-line reason.
```

**Typical baselines:**
- Naive RAG agent: 65–75%
- With citation grounding: 80–90%
- With reranking + self-check: >90%

---

## Q8: How do you set up experiment tracking for prompt iterations?

**Core concept:** Treat prompts like code — version them, track results, never overwrite.

**Minimal experiment log schema:**
```json
{
  "experiment_id": "exp_20260418_001",
  "prompt_version": "v2.3",
  "model": "claude-sonnet-4-20250514",
  "timestamp": "2026-04-18T10:00:00Z",
  "dataset": "golden_v1",
  "results": {
    "task_completion_rate": 0.82,
    "tool_correctness": 0.91,
    "avg_judge_score": 3.7,
    "p50_latency_ms": 1840,
    "p95_latency_ms": 4200,
    "cost_per_task_usd": 0.0043
  },
  "notes": "Added chain-of-thought prefix to system prompt"
}
```

**Tools:**
| Tool | Best for | Cost |
|---|---|---|
| **MLflow** | Self-hosted, full control | Free |
| **Weights & Biases** | Team collaboration, rich UI | Free tier available |
| **LangSmith** | LangChain native, trace-level detail | Free tier |
| **Langfuse** | Open-source, self-hostable, GDPR friendly | Free |
| **Simple JSON/CSV log** | Solo projects, quick prototyping | Free |

**Rule of thumb:** Log *everything* during development. Storage is cheap; missing data when debugging a regression is expensive.

---

## Q9: How do you set up DeepEval for automated evaluation?

**DeepEval** is an open-source LLM evaluation framework with built-in metrics.

**Install:**
```bash
pip install deepeval
```

**Key built-in metrics:**
- `AnswerRelevancyMetric` — is the response relevant to the input?
- `FaithfulnessMetric` — are facts grounded in context?
- `ContextualPrecisionMetric` — are retrieved chunks useful?
- `HallucinationMetric` — does the output contain fabrications?
- `GEval` — custom criteria with LLM-as-judge

**Basic usage:**
```python
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What is the refund policy?",
    actual_output=agent_response,
    retrieval_context=retrieved_chunks,
)

metrics = [
    AnswerRelevancyMetric(threshold=0.7),
    FaithfulnessMetric(threshold=0.8),
]

evaluate([test_case], metrics)
```

**Integration with pytest:**
```python
import pytest
from deepeval import assert_test

@pytest.mark.parametrize("test_case", golden_dataset)
def test_agent(test_case):
    response = run_agent(test_case["input"])
    assert_test(LLMTestCase(...), [AnswerRelevancyMetric(threshold=0.7)])
```

**Cost awareness:** Each metric evaluation makes 1–3 LLM API calls. Running 20 cases × 3 metrics = 60 calls ≈ $0.06 with GPT-4o-mini. Budget for this in your CI pipeline.

---

## Q10: What does a complete evaluation pipeline look like end-to-end?

```
1. LOAD dataset (JSON file or database query)
        ↓
2. RUN agent on each case (parallel with rate limiting)
        ↓
3. SCORE each result:
   ├── Rule-based: exact match, regex, JSON schema validation
   ├── Execution-based: tool call correctness check
   └── LLM-as-judge: semantic quality 1–5
        ↓
4. AGGREGATE metrics (mean, P50/P95, pass/fail count)
        ↓
5. LOG to experiment tracker (JSON log / MLflow / Langfuse)
        ↓
6. COMPARE to baseline (diff against last passing run)
        ↓
7. ALERT if TCR drops >5% or judge score drops >0.3
```

**Parallelism note:** Agent calls are I/O bound. Use `asyncio.gather` or `ThreadPoolExecutor` to run 20 cases in parallel — reduces wall time from ~60s to ~8s.

**Regression gate:** In CI, fail the build if:
- TCR drops below 80%
- Tool Correctness drops below 85%
- Average judge score drops below 3.0

---

## Summary Table

| Concept | Key number / rule |
|---|---|
| Golden dataset size | 50–200 cases (start with 20) |
| Edge case dataset size | 20–50 cases |
| Synthetic dataset size | 1,000+ cases |
| Target TCR (production) | >85% |
| Target Tool Correctness | >90% |
| Target LLM-judge score | >3.5 / 5 |
| Target FACTSCORE (RAG) | >80% |
| Latency P50 target | <3 seconds |
| Latency P95 target | <10 seconds |
| Judge calibration target | >80% agreement with humans |
| Regression alert threshold | TCR drop >5%, score drop >0.3 |