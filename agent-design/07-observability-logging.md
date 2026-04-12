# Observability & Logging — Production Agent Instrumentation

> Deep dive: Day 7 | Topic: Decorator-based LLM call logging, cost estimation, log viewer

---

## Q1. What is the full observability contract for a production agent?

**Answer:**

Observability answers four questions about every LLM call: **What was called? How long did it take? How much did it cost? Did it succeed?**

**The four pillars:**

| Pillar | What to capture | Why |
|---|---|---|
| **Tracing** | Call graph, node order, tool name | Debug wrong tool selection |
| **Metrics** | Token counts, latency, cost | Control spend, find bottlenecks |
| **Logging** | Inputs, outputs, errors | Reproduce issues, audit decisions |
| **Feedback** | User ratings, implicit signals | Improve quality over time |

**Minimum viable logging fields (JSONL per call):**

```json
{
  "timestamp":     "2026-04-12T10:23:44.123Z",
  "session_id":    "sess-abc123",
  "node_name":     "planner",
  "model":         "gpt-4o-mini",
  "tool_name":     null,
  "input_tokens":  312,
  "output_tokens": 87,
  "total_tokens":  399,
  "cost_usd":      0.00006435,
  "latency_ms":    1843.2,
  "error":         null,
  "trace_id":      "trace-xyz789"
}
```

**Why JSONL (not JSON array, not CSV)?**

- **Append-only** — no need to read and rewrite the whole file
- **Stream-parseable** — `grep`, `jq`, `pandas.read_json(lines=True)` all work
- **Crash-safe** — each line is independent; a crash mid-write only corrupts one entry
- **Human-readable** — open in any text editor

---

## Q2. How do you implement the observability decorator with full error handling?

**Answer:**

```python
import time
import json
import functools
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

# --- Pricing table (USD per token) ---
# Update when providers change pricing
PRICING = {
    "gpt-4o":              {"input": 2.50e-6,  "output": 10.00e-6},
    "gpt-4o-mini":         {"input": 0.15e-6,  "output": 0.60e-6},
    "gpt-4-turbo":         {"input": 10.00e-6, "output": 30.00e-6},
    "claude-opus-4-5":     {"input": 15.00e-6, "output": 75.00e-6},
    "claude-sonnet-4-5":   {"input": 3.00e-6,  "output": 15.00e-6},
    "claude-haiku-4-5":    {"input": 0.80e-6,  "output": 4.00e-6},
}

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "agent_calls.jsonl"

def ensure_log_dir():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def _write_log(entry: dict):
    ensure_log_dir()
    with open(LOG_FILE, "a", buffering=1) as f:  # buffering=1 = line-buffered
        f.write(json.dumps(entry, default=str) + "\n")

def _extract_usage(response) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from various response formats."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0

    # OpenAI format
    if hasattr(usage, "prompt_tokens"):
        return usage.prompt_tokens, usage.completion_tokens

    # Anthropic format
    if hasattr(usage, "input_tokens"):
        return usage.input_tokens, usage.output_tokens

    # Dict format
    if isinstance(usage, dict):
        return (
            usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            usage.get("completion_tokens", usage.get("output_tokens", 0)),
        )
    return 0, 0

def observe_llm(
    model: str,
    node_name: str = "unknown",
    tool_name: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """
    Decorator factory for transparent LLM call observability.
    
    Usage:
        @observe_llm(model="gpt-4o-mini", node_name="planner")
        def call_planner(messages):
            return openai_client.chat.completions.create(...)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            trace_id = str(uuid.uuid4())[:8]
            start_ns = time.perf_counter_ns()
            response = None
            error    = None

            try:
                response = func(*args, **kwargs)
                return response
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)}"
                raise
            finally:
                latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

                input_tok, output_tok = _extract_usage(response) if response else (0, 0)

                price    = PRICING.get(model, {"input": 0.0, "output": 0.0})
                cost_usd = (input_tok * price["input"]) + (output_tok * price["output"])

                _write_log({
                    "timestamp":     datetime.now(timezone.utc).isoformat(),
                    "trace_id":      trace_id,
                    "session_id":    session_id,
                    "node_name":     node_name,
                    "model":         model,
                    "tool_name":     tool_name,
                    "input_tokens":  input_tok,
                    "output_tokens": output_tok,
                    "total_tokens":  input_tok + output_tok,
                    "cost_usd":      round(cost_usd, 8),
                    "latency_ms":    round(latency_ms, 2),
                    "error":         error,
                })

        return wrapper
    return decorator
```

**Variant: context manager (for non-decorator use):**

```python
from contextlib import contextmanager

@contextmanager
def observe_call(model: str, node_name: str, tool_name: str = None):
    start_ns = time.perf_counter_ns()
    error = None
    result_holder = {}

    try:
        yield result_holder   # caller puts response in result_holder["response"]
    except Exception as e:
        error = str(e)
        raise
    finally:
        latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        response = result_holder.get("response")
        input_tok, output_tok = _extract_usage(response) if response else (0, 0)
        # ... rest of logging ...

# Usage:
with observe_call("gpt-4o-mini", "tool_router", tool_name="web_search") as ctx:
    response = openai_client.chat.completions.create(...)
    ctx["response"] = response
```

---

## Q3. How do you build the log viewer with per-tool cost and latency breakdown?

**Answer:**

```python
import json
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime

def load_logs(log_file: str = "logs/agent_calls.jsonl") -> list[dict]:
    path = Path(log_file)
    if not path.exists():
        return []
    entries = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: corrupt log entry at line {line_num}: {e}")
    return entries

def view_logs(log_file: str = "logs/agent_calls.jsonl", last_n: int = None):
    entries = load_logs(log_file)

    if not entries:
        print("No log entries found.")
        return

    if last_n:
        entries = entries[-last_n:]

    # Global stats
    total_calls   = len(entries)
    total_tokens  = sum(e.get("total_tokens", 0) for e in entries)
    total_cost    = sum(e.get("cost_usd",     0.0) for e in entries)
    error_count   = sum(1 for e in entries if e.get("error"))
    all_latencies = [e.get("latency_ms", 0) for e in entries if e.get("latency_ms")]
    avg_latency   = sum(all_latencies) / len(all_latencies) if all_latencies else 0

    # Per-node breakdown
    by_node = defaultdict(lambda: {
        "calls": 0, "tokens": 0, "cost": 0.0,
        "latencies": [], "errors": 0
    })
    by_model = defaultdict(lambda: {"calls": 0, "tokens": 0, "cost": 0.0})

    for e in entries:
        node  = e.get("node_name") or "unknown"
        model = e.get("model")     or "unknown"

        by_node[node]["calls"]    += 1
        by_node[node]["tokens"]   += e.get("total_tokens", 0)
        by_node[node]["cost"]     += e.get("cost_usd",     0.0)
        by_node[node]["latencies"].append(e.get("latency_ms", 0))
        if e.get("error"):
            by_node[node]["errors"] += 1

        by_model[model]["calls"]  += 1
        by_model[model]["tokens"] += e.get("total_tokens", 0)
        by_model[model]["cost"]   += e.get("cost_usd",     0.0)

    # --- Print report ---
    W = 60
    print(f"\n{'═'*W}")
    print(f"  AGENT OBSERVABILITY REPORT")
    if last_n:
        print(f"  (last {last_n} calls)")
    print(f"{'═'*W}")
    print(f"  Total calls      : {total_calls:,}")
    print(f"  Total tokens     : {total_tokens:,}")
    print(f"  Total cost       : ${total_cost:.6f} USD  (~${total_cost*100:.4f} cents)")
    print(f"  Errors           : {error_count} ({100*error_count/total_calls:.1f}%)")
    print(f"  Avg latency      : {avg_latency:.1f} ms")

    print(f"\n  {'─'*W}")
    print(f"  BY NODE")
    print(f"  {'─'*W}")
    print(f"  {'Node':<22} {'Calls':>6} {'Tokens':>8} {'Cost ($)':>10} {'AvgLat':>8} {'Errors':>7}")
    print(f"  {'─'*22} {'─'*6} {'─'*8} {'─'*10} {'─'*8} {'─'*7}")

    for node, s in sorted(by_node.items(), key=lambda x: x[1]["cost"], reverse=True):
        avg_lat = sum(s["latencies"]) / len(s["latencies"]) if s["latencies"] else 0
        print(f"  {node:<22} {s['calls']:>6,} {s['tokens']:>8,} "
              f"{s['cost']:>10.6f} {avg_lat:>7.1f}ms {s['errors']:>7}")

    print(f"\n  {'─'*W}")
    print(f"  BY MODEL")
    print(f"  {'─'*W}")
    for model, s in sorted(by_model.items(), key=lambda x: x[1]["cost"], reverse=True):
        print(f"  {model:<30} calls={s['calls']:>4,}  "
              f"tokens={s['tokens']:>8,}  cost=${s['cost']:.6f}")

    print(f"{'═'*W}\n")

if __name__ == "__main__":
    last_n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    view_logs(last_n=last_n)
```

**Sample output:**

```
════════════════════════════════════════════════════════════
  AGENT OBSERVABILITY REPORT
════════════════════════════════════════════════════════════
  Total calls      : 31
  Total tokens     : 18,432
  Total cost       : $0.002764 USD  (~$0.2764 cents)
  Errors           : 0 (0.0%)
  Avg latency      : 1,623.4 ms

  ────────────────────────────────────────────────────────
  BY NODE
  ────────────────────────────────────────────────────────
  Node                   Calls   Tokens    Cost ($)   AvgLat  Errors
  ───────────────────── ────── ──────── ────────── ──────── ───────
  planner                    8    4,200   0.000630  1842.3ms       0
  responder                  8    6,400   0.000960  2103.4ms       0
  tool_router                7    4,942   0.000741  1944.7ms       0
  summarizer                 5    1,890   0.000284   623.1ms       0
  tool_execution_web         3    1,000   0.000150   890.2ms       0

  ────────────────────────────────────────────────────────
  BY MODEL
  ────────────────────────────────────────────────────────
  gpt-4o-mini                    calls=  28  tokens=  17,432  cost=$0.002614
  gpt-4o                         calls=   3  tokens=   1,000  cost=$0.000150
════════════════════════════════════════════════════════════
```

---

## Q4. How do you estimate cost accurately when usage metadata is missing?

**Answer:**

When a streaming call or certain proxy APIs don't return `usage`, you need to estimate from the inputs:

```python
import tiktoken

def estimate_tokens_from_messages(messages: list[dict], model: str) -> int:
    """Estimate input token count when usage is not returned."""
    return count_message_tokens(messages, model)

def estimate_output_tokens(response_text: str, model: str) -> int:
    enc = tiktoken.encoding_for_model(model) 
    return len(enc.encode(response_text))

def estimate_cost(
    messages: list[dict],
    response_text: str,
    model: str
) -> float:
    input_tok  = estimate_tokens_from_messages(messages, model)
    output_tok = estimate_output_tokens(response_text, model)
    price      = PRICING.get(model, {"input": 0, "output": 0})
    return (input_tok * price["input"]) + (output_tok * price["output"])
```

**Accuracy note:** tiktoken estimates are exact for OpenAI models. For Claude, use the Anthropic token counting API or accept ~5% variance with a comparable tokenizer.

---

## Q5. What are the most actionable insights you can extract from agent logs?

**Answer:**

| Question | Query | Action if bad |
|---|---|---|
| Which node is slowest? | `AVG(latency_ms) GROUP BY node_name` | Cache, parallelise, or reduce prompt size |
| Which tool costs most? | `SUM(cost_usd) GROUP BY tool_name` | Switch to cheaper model for that tool |
| What's the error rate? | `COUNT(*) WHERE error IS NOT NULL` | Add retry logic, improve error handling |
| Is cost growing? | `SUM(cost_usd) GROUP BY DATE(timestamp)` | Add summarization or smaller model |
| Slowest sessions? | `SUM(latency_ms) GROUP BY session_id ORDER BY SUM DESC LIMIT 10` | Investigate those sessions specifically |

**Pro tip:** Set a cost alert threshold. If daily spend > $X, send yourself an email:

```python
def check_daily_cost_alert(threshold_usd: float = 1.00):
    entries = load_logs()
    today = datetime.now().date().isoformat()
    today_cost = sum(
        e.get("cost_usd", 0) for e in entries
        if e.get("timestamp", "").startswith(today)
    )
    if today_cost > threshold_usd:
        send_alert(f"Agent cost today: ${today_cost:.4f} — exceeds ${threshold_usd} threshold")
```

---

## Key Numbers

| Parameter | Value |
|---|---|
| JSONL write overhead | <0.1ms per entry |
| Log entry size | ~300–500 bytes per call |
| 10,000 calls log file size | ~4 MB |
| tiktoken accuracy vs actual API billing | Exact for OpenAI models |
| Recommended cost alert threshold (dev) | $1.00/day |
| gpt-4o-mini cost per 1M tokens | $0.15 input / $0.60 output |
| gpt-4o cost per 1M tokens | $2.50 input / $10.00 output |
| claude-sonnet-4-5 cost per 1M tokens | $3.00 input / $15.00 output |