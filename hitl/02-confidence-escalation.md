# 02 — Confidence-Based Escalation to Human Operators

---

## Q1. What is confidence-based escalation and why does it matter?

**A:** Confidence-based escalation is the pattern where an agent measures how certain it is about a decision and routes to a human when certainty falls below a threshold. It is the automated equivalent of an employee saying "I'm not sure about this one — let me check with my manager."

Without it, an agent will guess on ambiguous inputs and often be wrong. With it, the agent handles the 80% of clear cases automatically and only escalates the 20% where it genuinely needs help.

**The core insight:** A wrong confident answer is worse than an honest "I'm not sure." An agent that always gives an answer trains users to over-trust it. An agent that escalates appropriately trains users to trust it correctly.

**Real example:** Google's Smart Reply on Gmail only suggests replies when the model is confident. When confidence is low, no suggestions appear — rather than suggesting something wrong.

---

## Q2. What are the main ways to get a confidence score from an LLM?

**A:** Three main methods, in order of accuracy vs. implementation cost:

**Method 1: Model self-assessment (simplest, least calibrated)**

Ask the model to rate its own confidence as part of the structured output.

```python
import json
from openai import OpenAI

client = OpenAI()

def classify_with_self_assessment(query: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{
            "role": "system",
            "content": """You are an intent classifier for a customer support system.
Classify the user's intent and assess your confidence.

Respond ONLY with JSON:
{
  "intent": "billing | technical | refund | account | other",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<one sentence>",
  "ambiguity_reason": "<why you're uncertain, or null if confident>"
}

Confidence guide:
- 0.9+: Clear, unambiguous query matching one intent
- 0.7–0.9: Likely correct but has some ambiguity
- 0.5–0.7: Two intents are plausible
- <0.5: Cannot determine intent"""
        }, {
            "role": "user",
            "content": query
        }]
    )
    return json.loads(response.choices[0].message.content)
```

**Limitation:** LLMs are poorly calibrated — a model that says 0.9 confidence is not necessarily right 90% of the time. Calibrate against your golden dataset.

---

**Method 2: Logprobs (more precise)**

Use the log-probability of the model's output tokens as a proxy for confidence.

```python
import math

def classify_with_logprobs(query: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        logprobs=True,
        top_logprobs=5,
        max_tokens=20,
        messages=[{
            "role": "system",
            "content": "Classify the user intent. Reply with exactly one word: billing, technical, refund, account, or other."
        }, {
            "role": "user",
            "content": query
        }]
    )

    content = response.choices[0].logprobs.content
    top_token = content[0]

    # Convert log-probability to probability
    confidence = math.exp(top_token.logprob)
    intent = top_token.token.strip().lower()

    # Look at the top alternatives to understand competition
    alternatives = [
        {"token": t.token, "prob": math.exp(t.logprob)}
        for t in top_token.top_logprobs
    ]

    return {
        "intent": intent,
        "confidence": confidence,
        "alternatives": alternatives
    }
```

**When to use:** When you need calibrated probabilities, especially for routing decisions. Logprobs reflect the model's actual token distribution.

---

**Method 3: Ensemble voting (most robust, most expensive)**

Run the same classification prompt 3–5 times and use agreement rate as confidence.

```python
from collections import Counter

def classify_with_ensemble(query: str, runs: int = 5) -> dict:
    intents = []

    for _ in range(runs):
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,  # Some variance to surface disagreement
            max_tokens=10,
            messages=[{
                "role": "system",
                "content": "Reply with exactly one word: billing, technical, refund, account, or other."
            }, {
                "role": "user",
                "content": query
            }]
        )
        intents.append(response.choices[0].message.content.strip().lower())

    counts = Counter(intents)
    top_intent, top_count = counts.most_common(1)[0]
    confidence = top_count / runs

    return {
        "intent": top_intent,
        "confidence": confidence,  # 1.0 = unanimous, 0.4 = 2/5 agreement
        "distribution": dict(counts)
    }
```

**Cost:** 5× API calls per classification. Use sparingly — only for genuinely high-stakes routing.

---

## Q3. How do you build the confidence router node in LangGraph?

**A:**

```python
from langgraph.types import interrupt
from typing import Literal

CONFIDENCE_THRESHOLD = 0.70
HUMAN_QUEUE_THRESHOLD = 0.50  # Below this, escalate immediately without even trying

def confidence_router_node(state: AgentState) -> AgentState:
    query = state["messages"][-1]["content"]
    classification = classify_with_self_assessment(query)

    confidence = classification["confidence"]
    intent = classification["intent"]

    # Below floor threshold — definitely escalate
    if confidence < HUMAN_QUEUE_THRESHOLD:
        escalation_data = interrupt({
            "type": "confidence_escalation",
            "query": query,
            "classification": classification,
            "message": f"Agent confidence too low ({confidence:.0%}). Please handle this query.",
            "suggested_intent": intent
        })
        # Human provides the correct intent
        return {
            "route": escalation_data["intent"],
            "confidence": 1.0,  # Human decision is certain
            "escalated": True
        }

    # Between thresholds — try but flag for review
    if confidence < CONFIDENCE_THRESHOLD:
        # Log for async review but proceed (soft escalation)
        log_low_confidence_case(query, classification)
        return {
            "route": intent,
            "confidence": confidence,
            "flagged_for_review": True
        }

    # Above threshold — route automatically
    return {
        "route": intent,
        "confidence": confidence,
        "escalated": False
    }

def log_low_confidence_case(query, classification):
    # Store for offline analysis — don't block the user
    import sqlite3
    conn = sqlite3.connect("low_confidence.db")
    conn.execute(
        "INSERT INTO cases (query, intent, confidence, timestamp) VALUES (?, ?, ?, datetime('now'))",
        (query, classification["intent"], classification["confidence"])
    )
    conn.commit()
```

---

## Q4. How do you calibrate your confidence threshold?

**A:** Don't pick 0.7 arbitrarily — calibrate it against a golden test dataset.

**Calibration process:**

```python
# Step 1: Run classifier on 100–200 labeled examples
results = []
for example in golden_dataset:
    classification = classify_with_self_assessment(example["query"])
    correct = classification["intent"] == example["true_intent"]
    results.append({
        "confidence": classification["confidence"],
        "correct": correct
    })

# Step 2: Bin by confidence and measure accuracy
import pandas as pd
df = pd.DataFrame(results)
df["confidence_bin"] = pd.cut(df["confidence"], bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

calibration = df.groupby("confidence_bin")["correct"].agg(["mean", "count"])
print(calibration)
# confidence_bin | mean (actual accuracy) | count
# (0.5, 0.6]    | 0.45                   | 23
# (0.6, 0.7]    | 0.61                   | 31
# (0.7, 0.8]    | 0.79                   | 44
# (0.8, 0.9]    | 0.91                   | 52
# (0.9, 1.0]    | 0.97                   | 50
```

**Set your threshold** where the accuracy in your target bucket meets your SLA. If you need 85%+ accuracy, set the threshold at ~0.85 based on the calibration table above.

**The escalation rate trade-off:**

| Threshold | Escalation Rate | Error Rate |
|---|---|---|
| 0.50 | ~5% | ~15% |
| 0.70 | ~20% | ~7% |
| 0.85 | ~40% | ~3% |
| 0.95 | ~65% | ~1% |

Choose based on your tolerance for errors vs. human workload.

---

## Q5. What should you do with escalated cases after they're resolved?

**A:** Every escalation is a training signal. Store it and close the loop.

```python
def store_escalation_resolution(session_id, query, agent_classification,
                                 human_decision, resolution_time_seconds):
    """Store resolved escalations for later analysis and model improvement."""
    conn = sqlite3.connect("escalations.db")
    conn.execute("""
        INSERT INTO escalation_resolutions
        (session_id, query, agent_intent, agent_confidence,
         human_intent, resolution_seconds, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        session_id,
        query,
        agent_classification["intent"],
        agent_classification["confidence"],
        human_decision["intent"],
        resolution_time_seconds
    ))
    conn.commit()

# Analyze: where is the agent most wrong?
# SELECT agent_intent, human_intent, COUNT(*) as errors
# FROM escalation_resolutions
# WHERE agent_intent != human_intent
# GROUP BY agent_intent, human_intent
# ORDER BY errors DESC
```

**What to do with this data:**
- Queries where agent was wrong → add as few-shot examples to the classifier prompt.
- Query types with consistently high escalation rates → create a dedicated sub-classifier.
- Queries the human also struggled with → improve your taxonomy (your intent categories may be wrong).

---

## Key Numbers

| Metric | Value |
|---|---|
| Default confidence threshold | 0.70 |
| Medical/legal threshold | 0.90–0.95 |
| Target escalation rate (mature system) | 5–15% |
| Escalation resolution SLA | < 5 min (support), < 1 hr (async) |
| Golden dataset size for calibration | 100–200 labeled examples |