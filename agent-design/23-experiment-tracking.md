# 23 — Experiment Tracking for Prompt Iterations

> Deep dive for Day 10 · Part of [`agent-design/`](.)

---

## Why Experiment Tracking Matters

Prompt engineering without tracking is archaeology — you dig through Git diffs to figure out why the agent got worse after "that change last Tuesday."

Experiment tracking gives you:
- A record of every prompt version and its scores
- The ability to roll back to a known-good configuration
- Data to justify prompt changes to stakeholders ("v2.3 improved TCR by 7%")
- Debugging surface when production degrades

**Analogy:** Treat prompts like model weights. You wouldn't train a neural network without logging loss curves. Don't iterate on prompts without logging scores.

---

## What to Track Per Experiment

```python
@dataclass
class ExperimentRun:
    # Identity
    experiment_id: str          # "exp_20260418_001"
    prompt_version: str         # "v2.3"
    dataset_version: str        # "golden_v1.1"
    
    # Configuration
    model: str                  # "claude-sonnet-4-20250514"
    temperature: float          # 0.0
    max_tokens: int             # 1024
    system_prompt_hash: str     # SHA256 of system prompt
    
    # Results
    task_completion_rate: float
    avg_tool_correctness: float
    avg_judge_score: float
    factscore: float | None
    p50_latency_ms: float
    p95_latency_ms: float
    total_cost_usd: float
    cost_per_success_usd: float
    
    # Meta
    timestamp: str
    notes: str                  # "Added CoT prefix to system prompt"
    git_commit: str             # git rev-parse HEAD
    triggered_by: str           # "manual" | "ci" | "scheduled"
```

---

## Minimal JSON Logger (No External Dependency)

For solo projects or early stages:

```python
import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

EXPERIMENTS_FILE = "experiments/log.jsonl"

def log_experiment(config: dict, results: dict, notes: str = "") -> str:
    """Append one experiment to the JSONL log. Returns experiment_id."""
    
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Get current git commit
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode().strip()
    except Exception:
        git_commit = "unknown"
    
    record = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
        "notes": notes,
        **config,
        **results,
    }
    
    Path(EXPERIMENTS_FILE).parent.mkdir(exist_ok=True)
    with open(EXPERIMENTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    
    print(f"[Logged] {experiment_id} | TCR={results['task_completion_rate']:.1%} "
          f"| Judge={results['avg_judge_score']:.2f} | ${results['total_cost_usd']:.3f}")
    
    return experiment_id

def compare_experiments(exp_a: str, exp_b: str) -> None:
    """Print a diff of two experiment runs."""
    logs = {}
    with open(EXPERIMENTS_FILE) as f:
        for line in f:
            rec = json.loads(line)
            if rec["experiment_id"] in (exp_a, exp_b):
                logs[rec["experiment_id"]] = rec
    
    a, b = logs[exp_a], logs[exp_b]
    metrics = ["task_completion_rate", "avg_tool_correctness", "avg_judge_score",
               "p50_latency_ms", "cost_per_success_usd"]
    
    print(f"\n{'Metric':<30} {exp_a:<20} {exp_b:<20} {'Δ':<10}")
    print("-" * 80)
    for m in metrics:
        va, vb = a.get(m, 0), b.get(m, 0)
        delta = vb - va
        sign = "▲" if delta > 0 else "▼" if delta < 0 else "="
        print(f"{m:<30} {va:<20.3f} {vb:<20.3f} {sign}{abs(delta):.3f}")
```

---

## Langfuse Integration (Open-Source, Self-Hostable)

Langfuse gives you trace-level visibility — see exactly what happened in each agent step.

```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse(
    public_key="your-public-key",
    secret_key="your-secret-key",
    host="https://cloud.langfuse.com"  # or your self-hosted URL
)

# Wrap your agent with the @observe decorator
@observe()
async def run_agent(user_input: str, config: dict) -> dict:
    # Langfuse automatically captures:
    # - Input/output
    # - Token usage
    # - Latency
    # - Cost estimate
    langfuse_context.update_current_trace(
        name="agent_run",
        metadata={"prompt_version": config["prompt_version"]},
        tags=["evaluation", config.get("dataset", "unknown")]
    )
    
    # ... your agent code ...
    return result

# Log scores back to the trace
def log_scores_to_langfuse(trace_id: str, scores: dict) -> None:
    for metric_name, value in scores.items():
        langfuse.score(
            trace_id=trace_id,
            name=metric_name,
            value=value,
        )
```

**Langfuse features useful for evaluation:**
- Dataset management (upload golden sets, run evals from UI)
- Prompt versioning with automatic diff view
- Score time-series charts
- Filtering by prompt version, model, date

---

## MLflow Integration (Best for Team Environments)

```python
import mlflow

def run_eval_with_mlflow(config: dict, dataset: list, agent_fn) -> dict:
    
    with mlflow.start_run(run_name=f"eval_{config['prompt_version']}"):
        # Log config
        mlflow.log_params({
            "model": config["model"],
            "prompt_version": config["prompt_version"],
            "dataset_version": config["dataset_version"],
            "temperature": config.get("temperature", 0.0),
        })
        
        # Run evaluation
        results = run_evaluation_pipeline(dataset, agent_fn)
        
        # Log metrics
        mlflow.log_metrics({
            "task_completion_rate": results["task_completion_rate"],
            "avg_tool_correctness": results["avg_tool_correctness"],
            "avg_judge_score": results["avg_judge_score"],
            "p50_latency_ms": results["p50_latency_ms"],
            "p95_latency_ms": results["p95_latency_ms"],
            "cost_per_success_usd": results["cost_per_success_usd"],
        })
        
        # Save detailed results as artifact
        mlflow.log_dict(results, "detailed_results.json")
        
        return results
```

---

## Regression Detection

Run this after every evaluation to catch regressions automatically:

```python
REGRESSION_THRESHOLDS = {
    "task_completion_rate": {"min_drop": 0.05, "absolute_min": 0.75},
    "avg_judge_score":      {"min_drop": 0.30, "absolute_min": 3.0},
    "avg_tool_correctness": {"min_drop": 0.05, "absolute_min": 0.80},
}

def check_regression(baseline: dict, current: dict) -> list[str]:
    """Returns list of regression warnings. Empty = no regressions."""
    warnings = []
    
    for metric, thresholds in REGRESSION_THRESHOLDS.items():
        baseline_val = baseline.get(metric, 0)
        current_val = current.get(metric, 0)
        drop = baseline_val - current_val
        
        if drop > thresholds["min_drop"]:
            warnings.append(
                f"REGRESSION: {metric} dropped by {drop:.2%} "
                f"({baseline_val:.2f} → {current_val:.2f})"
            )
        
        if current_val < thresholds["absolute_min"]:
            warnings.append(
                f"BELOW THRESHOLD: {metric} = {current_val:.2f} "
                f"(minimum: {thresholds['absolute_min']:.2f})"
            )
    
    return warnings
```

---

## Prompt Versioning Convention

```
prompts/
├── system/
│   ├── v1.0.0.txt    # Initial version
│   ├── v1.1.0.txt    # Added CoT prefix
│   ├── v2.0.0.txt    # Major rewrite: tool descriptions
│   └── current -> v2.0.0.txt   # Symlink to active version
└── CHANGELOG.md
```

**Semantic versioning for prompts:**
- **MAJOR** (X.0.0): Complete rewrite or change in agent behaviour spec
- **MINOR** (X.Y.0): New capability or significant instruction change
- **PATCH** (X.Y.Z): Typo fix, minor clarification, formatting

**Git tag each version:**
```bash
git tag "prompt-v2.0.0" -m "Major rewrite: improved tool descriptions and CoT"
git push origin --tags
```

---

## CI Integration (GitHub Actions)

```yaml
# .github/workflows/eval.yml
name: Agent Evaluation

on:
  pull_request:
    paths:
      - 'prompts/**'
      - 'agent/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run evaluation suite
        run: python scripts/run_eval.py --dataset golden_v1 --output eval_results.json
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      
      - name: Check for regressions
        run: python scripts/check_regression.py --current eval_results.json --baseline baselines/main.json
      
      - name: Comment results on PR
        uses: actions/github-script@v7
        with:
          script: |
            const results = JSON.parse(fs.readFileSync('eval_results.json'));
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              body: `## Eval Results\n- TCR: ${(results.task_completion_rate * 100).toFixed(1)}%\n- Judge: ${results.avg_judge_score.toFixed(2)}/5\n- Cost/task: $${results.cost_per_success_usd.toFixed(4)}`
            });
```