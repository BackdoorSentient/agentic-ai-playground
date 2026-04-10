# 04 — Observability & Logging for Agent Systems

---

## Q1. Why is observability the first thing to build, not the last?

**A:** Most engineers add logging as an afterthought. In agent systems, this is a critical mistake because agent failures are non-deterministic — the same input can produce different outputs on different runs, and the only way to understand why is to have a full trace of every decision the agent made.

**What you cannot debug without logs:**

| Problem | Without logs | With logs |
|---|---|---|
| Agent called wrong tool | "It just didn't work" | See exact LLM output, tool name selected, args passed |
| Costs higher than expected | No idea which component is expensive | Cost per node, cost per tool, cost per session |
| Slow responses | No idea where the time goes | Latency per LLM call, per tool, per node |
| Memory not working | Cannot tell if facts were retrieved | See retrieved_facts injected into prompt |
| HITL not firing | Cannot tell if interrupt() was reached | See node execution sequence |

**The rule:** Build the observability layer on Day 1, use it on Day 2. Every other component you build will be easier to debug because of it.

---

## Q2. What is JSONL and why is it the right format for agent logs?

**A:** JSONL (JSON Lines) is a text file where each line is a complete, independently parseable JSON object. No commas between records, no surrounding array brackets.

```jsonl
{"timestamp":"2026-04-10T09:14:22Z","node":"retrieve_memory","latency_ms":142,"facts_found":2}
{"timestamp":"2026-04-10T09:14:23Z","node":"llm_call","prompt_tokens":412,"completion_tokens":87,"latency_ms":843,"tool_invoked":"web_search","cost_usd":0.001903}
{"timestamp":"2026-04-10T09:14:24Z","node":"web_search","query":"langchain memory","latency_ms":12}
{"timestamp":"2026-04-10T09:14:25Z","node":"llm_call","prompt_tokens":520,"completion_tokens":145,"latency_ms":921,"tool_invoked":null,"cost_usd":0.002750}
```

**Why JSONL over plain JSON array:**

| Property | JSON Array | JSONL |
|---|---|---|
| Append new record | Re-read, parse, append, re-write entire file | `file.write(json.dumps(entry) + "\n")` |
| Read last N records | Load entire file | `tail -n 100 agent.jsonl \| jq .` |
| Corrupt one record | Entire file unparseable | Only that line fails — others readable |
| Stream to analytics | Cannot stream partial JSON | Each line is a complete event |
| Grep / filter | Requires JSON parser | `grep "web_search" agent.jsonl` |

---

## Q3. How do you implement the full observability logger?

**A:**

```python
# observability/logger.py
import json
import os
import time
from datetime import datetime
from functools import wraps
from typing import Optional

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "agent.jsonl")

# Token pricing per 1M tokens (update as pricing changes)
MODEL_PRICING = {
    "gpt-4o":                {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":           {"input": 0.15,  "output": 0.60},
    "claude-3-5-sonnet-20241022": {"input": 3.00,  "output": 15.00},
    "claude-3-haiku":        {"input": 0.25,  "output": 1.25},
    "text-embedding-3-small":{"input": 0.02,  "output": 0.0},
}

def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = MODEL_PRICING.get(model, {"input": 2.50, "output": 10.00})
    return round(
        (input_tokens / 1_000_000) * rates["input"] +
        (output_tokens / 1_000_000) * rates["output"],
        8
    )

def _write_log(entry: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def log_llm_call(
    node: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    tool_invoked: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict:
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type": "llm_call",
        "session_id": session_id,
        "node": node,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_ms": round(latency_ms, 2),
        "tool_invoked": tool_invoked,
        "cost_usd": _estimate_cost(model, prompt_tokens, completion_tokens),
    }
    _write_log(entry)
    return entry

def log_tool_call(
    node: str,
    tool_name: str,
    args: dict,
    result_length: int,
    latency_ms: float,
    success: bool,
    session_id: Optional[str] = None,
) -> dict:
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type": "tool_call",
        "session_id": session_id,
        "node": node,
        "tool_name": tool_name,
        "args": args,
        "result_length_chars": result_length,
        "latency_ms": round(latency_ms, 2),
        "success": success,
    }
    _write_log(entry)
    return entry

def log_memory_event(
    event_type: str,  # "retrieve" or "write"
    user_id: str,
    facts_count: int,
    latency_ms: float,
    session_id: Optional[str] = None,
) -> dict:
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type": f"memory_{event_type}",
        "session_id": session_id,
        "user_id": user_id,
        "facts_count": facts_count,
        "latency_ms": round(latency_ms, 2),
    }
    _write_log(entry)
    return entry
```

---

## Q4. How do you build a simple log summary report?

**A:**

```python
# observability/report.py
import json
from collections import defaultdict

def load_logs(log_file: str = "logs/agent.jsonl") -> list[dict]:
    entries = []
    try:
        with open(log_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return entries

def print_session_report(session_id: str = None):
    entries = load_logs()
    if session_id:
        entries = [e for e in entries if e.get("session_id") == session_id]

    llm_calls = [e for e in entries if e["type"] == "llm_call"]
    tool_calls = [e for e in entries if e["type"] == "tool_call"]
    memory_events = [e for e in entries if e["type"].startswith("memory_")]

    total_cost = sum(e.get("cost_usd", 0) for e in llm_calls)
    total_tokens = sum(e.get("total_tokens", 0) for e in llm_calls)
    avg_latency = sum(e.get("latency_ms", 0) for e in llm_calls) / len(llm_calls) if llm_calls else 0

    tool_counts = defaultdict(int)
    for tc in tool_calls:
        tool_counts[tc["tool_name"]] += 1

    print(f"\n{'='*50}")
    print(f"  Agent Session Report{' — ' + session_id if session_id else ''}")
    print(f"{'='*50}")
    print(f"  LLM Calls:        {len(llm_calls)}")
    print(f"  Total Tokens:     {total_tokens:,}")
    print(f"  Total Cost:       ${total_cost:.6f}")
    print(f"  Avg LLM Latency:  {avg_latency:.0f}ms")
    print(f"\n  Tool Usage:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"    {tool:<20} {count} calls")
    print(f"\n  Memory Events:    {len(memory_events)}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    print_session_report()
```

**Sample output:**

```
==================================================
  Agent Session Report
==================================================
  LLM Calls:        6
  Total Tokens:     3,421
  Total Cost:       $0.021403
  Avg LLM Latency:  876ms

  Tool Usage:
    web_search           3 calls
    calendar_lookup      1 calls
    save_note            1 calls

  Memory Events:    8
==================================================
```

---

## Q5. How do you use logs to debug the most common agent failures?

**A:**

**Failure 1: Wrong tool selected**

```bash
# Filter logs to see all tool selections
grep '"type": "llm_call"' logs/agent.jsonl | jq '.tool_invoked'
# If you see "null" when a tool was expected, the LLM decided no tool was needed
# Check: is the tool description precise enough? Is the query ambiguous?
```

**Failure 2: Tool not returning useful results**

```bash
# Find tool calls with short results (may indicate empty/error responses)
grep '"type": "tool_call"' logs/agent.jsonl | jq 'select(.result_length_chars < 50)'
```

**Failure 3: High cost per turn**

```bash
# Find the most expensive LLM calls
grep '"type": "llm_call"' logs/agent.jsonl | jq -s 'sort_by(.cost_usd) | reverse | .[0:5]'
# If prompt_tokens is very high → memory is not being compressed
# If completion_tokens is very high → LLM is being too verbose, add length constraint
```

**Failure 4: Memory not being retrieved**

```bash
# Check memory retrieval events
grep '"type": "memory_retrieve"' logs/agent.jsonl | jq '.facts_count'
# If consistently 0 → ChromaDB is empty or embedding mismatch
# Run: python -c "from memory.long_term import _get_collection; c = _get_collection('user1'); print(c.count())"
```

---

## Key Numbers

| Metric | Good | Needs attention |
|---|---|---|
| LLM call latency P50 | < 1000ms | > 2000ms |
| Cost per conversation turn | < $0.005 | > $0.02 |
| prompt_tokens per call | < 1000 | > 3000 (memory not compressing) |
| Tool success rate | > 95% | < 90% |
| Memory facts retrieved | 1–3 | 0 (memory broken) |
| Log file growth rate | ~500 bytes/turn | > 5KB/turn (something is very verbose) |