# 22 — Key Metrics: Tool Correctness, Task Completion, Reasoning Quality

> Deep dive for Day 10 · Part of [`agent-design/`](.)

---

## The Metrics Stack

Agent evaluation needs metrics at three levels:

```
Level 3: Business outcomes     (e.g., customer satisfaction, resolution rate)
Level 2: Agent-level metrics   (task completion, cost, latency)
Level 1: Step-level metrics    (tool correctness, reasoning quality per step)
```

Most teams only measure Level 2. The best teams measure all three and trace regressions from Level 3 down to Level 1.

---

## 1. Task Completion Rate (TCR)

**Definition:** Fraction of tasks where the agent fully accomplished the user's goal.

```
TCR = tasks_fully_completed / total_tasks
```

**Binary classification — a task is complete when:**
- All expected tool calls were made correctly
- Output contains all required information
- Output contains no forbidden content
- Completed within the step limit

**Implementation:**
```python
def compute_tcr(results: list[dict]) -> float:
    completed = sum(1 for r in results if r["task_complete"])
    return completed / len(results)
```

**Nuance — partial completion:**  
Some tasks have multiple sub-goals. Track partial completion rate separately:

```python
@dataclass
class TaskResult:
    case_id: str
    sub_goals: list[str]
    sub_goals_completed: list[str]
    
    @property
    def completion_fraction(self) -> float:
        return len(self.sub_goals_completed) / len(self.sub_goals)
    
    @property 
    def fully_complete(self) -> bool:
        return self.completion_fraction == 1.0
```

**Target benchmarks:**
- Development: >70% on golden set before shipping to staging
- Staging: >80% on full evaluation suite
- Production: >85% sustained, alert at <80%

**Common failure modes:**
- Agent asks for clarification on a clear question (over-cautious)
- Agent answers confidently with wrong information (hallucination)
- Agent completes task but adds unnecessary caveats that confuse users

---

## 2. Tool Correctness

**Definition:** Measures whether the agent called the right tool with the right arguments.

**Four sub-dimensions:**

```
Tool Name Match:    Did the agent call "search_orders" not "query_database"?
Argument Presence:  Were all required args provided?
Argument Validity:  Were arg types/formats correct?
Argument Accuracy:  Did the arg values make contextual sense?
```

**Full implementation with partial scoring:**
```python
def tool_correctness_score(expected_calls: list[dict], actual_calls: list[dict]) -> dict:
    """
    Compute tool correctness across a sequence of expected tool calls.
    Handles: missing calls, extra calls, wrong order.
    """
    if not expected_calls:
        return {"score": 1.0, "details": "No tool calls expected — matched"}
    
    if not actual_calls:
        return {"score": 0.0, "details": "Expected tool calls but agent made none"}
    
    scores = []
    details = []
    
    for i, expected in enumerate(expected_calls):
        # Find best matching actual call (greedy match by tool name)
        match = next(
            (a for a in actual_calls if a.get("tool") == expected["tool"]),
            None
        )
        
        if match is None:
            scores.append(0.0)
            details.append(f"Tool '{expected['tool']}' never called")
            continue
        
        # Score the match
        expected_args = expected.get("args", {})
        actual_args = match.get("args", {})
        
        arg_checks = []
        for key, expected_val in expected_args.items():
            actual_val = actual_args.get(key)
            if actual_val is None:
                arg_checks.append(0.0)  # missing arg
            elif type(actual_val) != type(expected_val):
                arg_checks.append(0.3)  # wrong type
            elif str(actual_val).lower() == str(expected_val).lower():
                arg_checks.append(1.0)  # exact match
            else:
                arg_checks.append(0.5)  # present but wrong value
        
        call_score = sum(arg_checks) / len(arg_checks) if arg_checks else 1.0
        scores.append(call_score)
        details.append(f"Tool '{expected['tool']}': score {call_score:.2f}")
    
    return {
        "score": sum(scores) / len(scores),
        "per_call_scores": scores,
        "details": details
    }
```

**Penalty for extra tool calls:**

Extra tool calls waste latency and cost. Track them separately:
```python
extra_calls = max(0, len(actual_calls) - len(expected_calls))
efficiency_penalty = 0.1 * extra_calls  # -10% per unnecessary call
adjusted_score = max(0, base_tool_score - efficiency_penalty)
```

---

## 3. Reasoning Quality (LLM-as-Judge 1–5 Scale)

**Definition:** A holistic semantic score for the quality of the agent's chain-of-thought and final response.

**Rubric (anchor each score with examples):**

| Score | Label | Description | Example signal |
|---|---|---|---|
| **1** | Harmful / Wrong | Factually incorrect or harmful | Hallucinates order details, gives dangerous advice |
| **2** | Major Issues | Partially relevant but significant gaps | Answers adjacent question, misses critical info |
| **3** | Acceptable | Correct intent, incomplete execution | Right answer but missing confirmation, awkward phrasing |
| **4** | Good | Correct, complete, minor issues | Correct answer, slightly verbose |
| **5** | Excellent | Accurate, complete, concise, helpful | Correct answer in 2 sentences with clear next step |

**Score calibration examples for a support agent:**

```
User: "Where is my order ORD-452?"

Score 1: "Your order will arrive tomorrow" (no tool called, hallucinated)
Score 2: "I found your account. You have 3 orders." (irrelevant info, didn't answer)
Score 3: "Your order ORD-452 is in transit." (correct but no ETA, no tracking link)
Score 4: "Your order ORD-452 is in transit and expected to arrive by Thursday, April 21." 
Score 5: "Your order ORD-452 is in transit, arriving Thursday April 21. Track it here: [link]"
```

---

## 4. Faithfulness (FACTSCORE)

**Definition:** Fraction of output claims that are directly supported by the source context.

```
FACTSCORE = supported_atomic_facts / total_atomic_facts
```

**When it matters most:**
- RAG (retrieval-augmented) agents
- Agents summarising documents
- Any setting where the agent has access to ground-truth context

**Implementation using LLM decomposition + verification:**
```python
ATOMIZE_PROMPT = """
Extract every distinct factual claim from this text as a bullet list.
Each claim should be a single, verifiable statement.
Only include claims about the real world, not about formatting or presentation.

Text: {text}

Return as a JSON array of strings.
"""

VERIFY_CLAIM_PROMPT = """
Context: {context}
Claim: {claim}

Is this claim directly and explicitly supported by the context above?
Answer with JSON only: {{"supported": true/false, "reason": "<one sentence>"}}
"""

async def compute_factscore(output: str, context: str) -> dict:
    # Step 1: Extract atomic facts
    facts_response = await llm_call(ATOMIZE_PROMPT.format(text=output))
    facts = json.loads(facts_response)
    
    # Step 2: Verify each fact
    verifications = await asyncio.gather(*[
        llm_call(VERIFY_CLAIM_PROMPT.format(context=context, claim=fact))
        for fact in facts
    ])
    
    results = [json.loads(v) for v in verifications]
    supported = sum(1 for r in results if r["supported"])
    
    return {
        "factscore": supported / len(facts) if facts else 1.0,
        "total_facts": len(facts),
        "supported_facts": supported,
        "unsupported": [
            {"fact": facts[i], "reason": results[i]["reason"]}
            for i in range(len(facts))
            if not results[i]["supported"]
        ]
    }
```

---

## 5. Latency Metrics (P50 / P95)

**Why percentiles, not averages:**

Averages hide tail latency. If your P50 is 1.5s and P95 is 12s, 5% of users wait 12 seconds. That's your complex cases — often your highest-value users.

```python
import numpy as np

def compute_latency_stats(latencies_ms: list[float]) -> dict:
    arr = np.array(latencies_ms)
    return {
        "p50_ms": np.percentile(arr, 50),
        "p75_ms": np.percentile(arr, 75),
        "p90_ms": np.percentile(arr, 90),
        "p95_ms": np.percentile(arr, 95),
        "p99_ms": np.percentile(arr, 99),
        "mean_ms": arr.mean(),
        "max_ms": arr.max(),
    }
```

**Target SLAs by use case:**

| Use case | P50 target | P95 target |
|---|---|---|
| Real-time chat | <2s | <6s |
| Background task (async) | <10s | <30s |
| Batch processing | <60s | <300s |

**Latency breakdown — where time goes:**
```
Total latency = LLM call time + tool call time + orchestration overhead

Typical breakdown for a 2-turn agent:
  LLM call 1 (planning):     800ms
  Tool call (API):            200ms
  LLM call 2 (responding):   700ms
  Overhead:                   100ms
  Total:                    1,800ms
```

---

## 6. Cost Efficiency

**Definition:** Total API cost divided by number of successfully completed tasks.

```
Cost per successful task = total_api_cost / tasks_completed
```

**Tracking cost in code:**
```python
def extract_cost(response: anthropic.types.Message) -> float:
    """Estimate cost from Anthropic usage stats."""
    # Prices as of 2026 — verify at anthropic.com/pricing
    PRICES = {
        "claude-haiku-4-5-20251001": {"input": 0.00025, "output": 0.00125},     # per 1K tokens
        "claude-sonnet-4-20250514":  {"input": 0.003,   "output": 0.015},
        "claude-opus-4-20250514":    {"input": 0.015,   "output": 0.075},
    }
    
    model = response.model
    usage = response.usage
    price = PRICES.get(model, {"input": 0.003, "output": 0.015})
    
    input_cost  = (usage.input_tokens  / 1000) * price["input"]
    output_cost = (usage.output_tokens / 1000) * price["output"]
    return input_cost + output_cost

# Aggregate across a full eval run
total_cost = sum(r["cost_usd"] for r in results)
completed = sum(1 for r in results if r["task_complete"])
cost_per_success = total_cost / completed if completed > 0 else float("inf")
```

**Cost optimisation levers:**
1. Use Haiku for simple tasks, Sonnet for complex
2. Compress system prompts (remove redundant instructions)
3. Limit tool call depth (max 5 iterations)
4. Cache tool results for repeated calls
5. Use prompt caching for long stable system prompts

---

## Aggregated Metrics Dashboard

```python
def print_eval_summary(results: list[dict]) -> None:
    n = len(results)
    
    tcr = sum(r["task_complete"] for r in results) / n
    avg_tool = sum(r.get("tool_score", 0) or 0 for r in results) / n
    avg_judge = sum(r["judge_score"] for r in results) / n
    avg_cost = sum(r.get("cost_usd", 0) for r in results) / n
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    
    print(f"""
╔══════════════════════════════════════════╗
║         EVALUATION SUMMARY               ║
╠══════════════════════════════════════════╣
║  Cases evaluated:      {n:>4}             ║
║  Task Completion:      {tcr:>6.1%}          ║
║  Tool Correctness:     {avg_tool:>6.1%}          ║
║  Avg Judge Score:      {avg_judge:>5.2f} / 5.0     ║
║  Avg Cost/Task:        ${avg_cost:.4f}          ║
║  P50 Latency:          {np.percentile(latencies, 50):>6.0f}ms         ║
║  P95 Latency:          {np.percentile(latencies, 95):>6.0f}ms         ║
╚══════════════════════════════════════════╝
""")
```