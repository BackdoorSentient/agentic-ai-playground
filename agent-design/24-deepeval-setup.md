# 24 — DeepEval: Automated Evaluation with Built-in Metrics

> Deep dive for Day 10 · Part of [`agent-design/`](.)

---

## What is DeepEval

DeepEval is an open-source Python framework for LLM evaluation. It provides:
- 14+ built-in evaluation metrics
- pytest integration for CI/CD
- Dataset management
- A dashboard (Confident AI) for tracking results over time

**Install:**
```bash
pip install deepeval
```

**Set up your LLM evaluator:**
```bash
# DeepEval uses OpenAI by default, but you can use any provider
export OPENAI_API_KEY="your-key"

# Or configure Anthropic:
deepeval set-anthropic-api-key "your-anthropic-key"
```

---

## Core Concepts

**LLMTestCase** — the atomic unit of evaluation:
```python
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What is the refund policy for digital products?",
    actual_output=agent_response,           # what your agent said
    expected_output="30-day money back...", # optional reference
    retrieval_context=retrieved_chunks,     # for RAG metrics
    context=["policy doc 1", "policy doc 2"] # broader context
)
```

**Metric** — a scorer with a threshold:
```python
from deepeval.metrics import AnswerRelevancyMetric

metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini")
metric.measure(test_case)
print(metric.score)    # 0.85
print(metric.passed)   # True
print(metric.reason)   # "Response directly addresses the user's question about refunds"
```

---

## Built-in Metrics Reference

### For general agents

| Metric | What it measures | When to use |
|---|---|---|
| `AnswerRelevancyMetric` | Is the response on-topic? | All agents |
| `FaithfulnessMetric` | Are claims grounded in context? | RAG agents |
| `HallucinationMetric` | Does output contradict context? | RAG agents |
| `BiasMetric` | Does output show demographic bias? | Customer-facing agents |
| `ToxicityMetric` | Is output harmful/offensive? | Public-facing agents |

### For RAG agents specifically

| Metric | What it measures |
|---|---|
| `ContextualPrecisionMetric` | Are retrieved chunks ranked correctly? |
| `ContextualRecallMetric` | Does retrieval cover all needed info? |
| `ContextualRelevancyMetric` | Are retrieved chunks relevant to the query? |

### For task-completion agents

| Metric | What it measures |
|---|---|
| `TaskCompletionMetric` | Did agent complete the described task? |
| `ToolCorrectnessMetric` | Were tools called correctly? |
| `GEval` | Custom criteria (define your own rubric) |

---

## Using GEval for Custom Criteria

GEval is the most flexible metric — you define the evaluation criteria:

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

reasoning_quality = GEval(
    name="Reasoning Quality",
    criteria="""
    Evaluate the agent's step-by-step reasoning quality.
    
    A high score (4-5) means:
    - The reasoning is logically sound
    - Each step follows from the previous
    - The agent identifies the correct approach
    - No logical leaps or unsupported assumptions
    
    A low score (1-2) means:
    - Circular reasoning or non-sequiturs
    - Wrong approach to the problem
    - Missing critical reasoning steps
    """,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    threshold=0.7,
)
```

---

## Full Evaluation Suite for a Support Agent

```python
# eval/test_support_agent.py
import pytest
import json
from pathlib import Path
from deepeval import assert_test, evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    GEval,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall
from your_agent import run_support_agent

# Load golden dataset
GOLDEN_DATASET = json.loads(Path("datasets/golden/v1.0.0_golden.json").read_text())

# Define metrics
answer_relevancy = AnswerRelevancyMetric(threshold=0.7)
faithfulness = FaithfulnessMetric(threshold=0.75)
hallucination = HallucinationMetric(threshold=0.1)  # max 10% hallucination
tool_correctness = ToolCorrectnessMetric(threshold=0.8)

helpfulness = GEval(
    name="Helpfulness",
    criteria="Is the response actionable and helpful for the user's specific situation?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
)

# Parametrize across all golden cases
@pytest.mark.parametrize("case", GOLDEN_DATASET)
def test_support_agent(case):
    # Run the agent
    result = run_support_agent(
        user_input=case["input"],
        context=case.get("context", {})
    )
    
    # Build test case
    test_case = LLMTestCase(
        input=case["input"],
        actual_output=result["response"],
        expected_output=case.get("expected", {}).get("reference_output"),
        retrieval_context=result.get("retrieved_chunks", []),
        # Map agent tool calls to DeepEval format
        tools_called=[
            ToolCall(name=tc["tool"], input_parameters=tc["args"])
            for tc in result.get("tool_calls", [])
        ],
        expected_tools=[
            ToolCall(name=tc["tool"], input_parameters=tc["args"])
            for tc in case.get("expected", {}).get("tool_calls", [])
        ],
    )
    
    # Choose metrics based on case type
    metrics = [answer_relevancy, helpfulness]
    if case.get("expected", {}).get("tool_calls"):
        metrics.append(tool_correctness)
    if result.get("retrieved_chunks"):
        metrics.extend([faithfulness, hallucination])
    
    assert_test(test_case, metrics)
```

**Run with:**
```bash
pytest eval/test_support_agent.py -v --tb=short

# Or use DeepEval's native runner (shows nicer output + uploads to dashboard)
deepeval test run eval/test_support_agent.py
```

---

## Batch Evaluation (without pytest)

For running evals programmatically (e.g., from a script, not test runner):

```python
from deepeval import evaluate

test_cases = [
    LLMTestCase(
        input=case["input"],
        actual_output=run_agent(case["input"]),
    )
    for case in golden_dataset
]

metrics = [
    AnswerRelevancyMetric(threshold=0.7),
    GEval(name="Quality", criteria="...", threshold=0.7,
          evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]),
]

results = evaluate(test_cases, metrics)

# Results include per-case scores and overall pass rate
print(f"Pass rate: {results.confident_score:.1%}")
```

---

## Confident AI Dashboard

DeepEval integrates with Confident AI (their hosted platform) for tracking evaluations over time.

```bash
deepeval login  # Creates a free account
deepeval test run eval/ --confident-ai  # Sends results to dashboard
```

**Dashboard features:**
- Time-series charts of metric scores
- Case-level drill-down (which specific cases are failing)
- Comparison between runs
- Slack/email alerts on regressions

**Self-hosted alternative:** DeepEval also supports exporting results to JSON for your own tracking:
```python
results = evaluate(test_cases, metrics, write_cache=True)
# Results cached in .deepeval/ directory
```

---

## Cost Optimisation for DeepEval

Each metric makes 1–3 LLM API calls. For 20 cases × 4 metrics = 80–240 API calls.

| Scenario | Approximate cost |
|---|---|
| 20 cases, 3 metrics, GPT-4o-mini | ~$0.05 |
| 20 cases, 3 metrics, GPT-4o | ~$0.80 |
| 200 cases, 5 metrics, GPT-4o-mini | ~$0.50 |
| 1000 cases, 5 metrics, GPT-4o-mini | ~$2.50 |

**Cost tip:** Use GPT-4o-mini (or Claude Haiku) as your judge model. The marginal quality difference vs GPT-4o for evaluation scoring is small, and the cost difference is 10–20×.

```python
# Configure cheaper model for all DeepEval metrics
from deepeval.models import GPTModel

cheap_judge = GPTModel(model="gpt-4o-mini")

AnswerRelevancyMetric(threshold=0.7, model=cheap_judge)
FaithfulnessMetric(threshold=0.8, model=cheap_judge)
```

---

## DeepEval vs Custom Pipeline: When to Use Each

| Scenario | Use DeepEval | Use custom pipeline |
|---|---|---|
| Standard metrics (relevancy, faithfulness) | ✅ | |
| Custom domain-specific criteria | Use GEval | or DIY |
| Tool call correctness | ✅ (ToolCorrectnessMetric) | Both work |
| Complex multi-step scoring logic | | ✅ |
| Integration with non-OpenAI judges | Limited support | ✅ |
| Pytest CI integration | ✅ | Both work |
| Need full control over prompts | | ✅ |

**Recommendation:** Start with DeepEval for built-in metrics and add custom pipeline components for anything it doesn't cover.