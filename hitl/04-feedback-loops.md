# 04 — Building Feedback Loops That Improve Agent Performance

---

## Q1. What is a feedback loop in the context of AI agents and why does it matter?

**A:** A feedback loop is a system that captures signals about agent quality, stores them, analyzes them, and feeds the insights back into the agent to improve it. Without a feedback loop, your agent is static — it cannot improve from real-world usage.

**The flywheel:**

```
Users interact with agent
       ↓
Feedback collected (explicit or implicit)
       ↓
Stored with full context
       ↓
Analyzed offline (clustering, LLM-as-judge)
       ↓
Prompt improved / few-shot examples added
       ↓
Agent performs better → more user trust
       ↓
(back to top)
```

**Why it matters at scale:** A 1% improvement in task completion from feedback analysis, compounded over weeks, compounds dramatically. At 10,000 daily interactions, a 5% improvement in routing accuracy saves 500 human escalations per day.

---

## Q2. What are the types of feedback signals and which should you collect?

**A:**

| Type | Signal | Pros | Cons |
|---|---|---|---|
| **Explicit: thumbs up/down** | User rates response | Clear signal, easy to implement | Low response rate (5–15%) |
| **Explicit: rating (1–5)** | Numeric score | More granular | Even lower response rate |
| **Explicit: free text** | User comment | Rich signal, explains why | Hard to analyze at scale |
| **Implicit: retry** | User rephrases and asks again | High volume, no friction | Ambiguous — could be clarification |
| **Implicit: copy/paste** | User copies the response | Strong positive signal | Hard to detect reliably |
| **Implicit: follow-up** | "That's not what I meant" | Clear negative signal | Requires NLP to detect |
| **System: escalation rate** | % routed to human | Objective | Lags behind root cause |
| **System: task completion** | Did the downstream action succeed? | Objective outcome | Not always available |

**Recommended minimum:** Collect thumbs up/down + optional free text comment. It's low friction and gives you enough signal to start improving.

---

## Q3. How do you implement feedback collection with LangGraph's interrupt()?

**A:**

```python
from langgraph.types import interrupt
from typing import TypedDict, Optional
import sqlite3
from datetime import datetime
import uuid

class AgentState(TypedDict):
    messages: list
    session_id: str
    last_response: Optional[str]
    feedback: Optional[dict]

def response_node(state: AgentState) -> AgentState:
    """Agent generates a response."""
    # ... generate response ...
    response = "Here is your answer: ..."
    return {
        "last_response": response,
        "messages": state["messages"] + [{"role": "assistant", "content": response}]
    }

def feedback_collection_node(state: AgentState) -> AgentState:
    """Pause and ask the user for feedback after responding."""
    feedback = interrupt({
        "type": "feedback_request",
        "response": state["last_response"],
        "question": "Was this response helpful?",
        "options": ["thumbs_up", "thumbs_down"],
        "optional": True,          # User can skip this
        "timeout_seconds": 30      # Hint to the UI layer — not enforced by LangGraph
    })

    # Store feedback regardless of whether the user skips
    if feedback and feedback.get("rating"):
        store_feedback(
            session_id=state["session_id"],
            query=state["messages"][-2]["content"],  # The user's question
            response=state["last_response"],
            rating=feedback["rating"],
            comment=feedback.get("comment")
        )
        return {"feedback": feedback}

    return {"feedback": None}  # User skipped — that's fine

def store_feedback(session_id, query, response, rating, comment=None):
    conn = sqlite3.connect("feedback.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            query TEXT,
            response TEXT,
            rating TEXT,
            comment TEXT,
            timestamp TEXT
        )
    """)
    conn.execute("""
        INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        str(uuid.uuid4()),
        session_id,
        query,
        response,
        rating,
        comment,
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()
```

---

## Q4. How do you analyze feedback to find actionable improvements?

**A:** Raw thumbs-up/down data tells you something is wrong but not what. You need to dig deeper.

**Step 1: Find the bad responses**

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("feedback.db")
df = pd.read_sql("SELECT * FROM feedback", conn)

# Focus on thumbs-down
negatives = df[df["rating"] == "thumbs_down"].copy()
print(f"Negative rate: {len(negatives)/len(df):.1%}")
print(f"Total negatives: {len(negatives)}")
```

**Step 2: Cluster by topic (embedding similarity)**

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Embed all negative queries
negatives["embedding"] = negatives["query"].apply(get_embedding)

# Cluster with k-means
from sklearn.cluster import KMeans
X = np.array(negatives["embedding"].tolist())
kmeans = KMeans(n_clusters=5, random_state=42)
negatives["cluster"] = kmeans.fit_predict(X)

# See what each cluster is about
for cluster_id in range(5):
    cluster_items = negatives[negatives["cluster"] == cluster_id]["query"].head(3).tolist()
    print(f"\n--- Cluster {cluster_id} ---")
    for item in cluster_items:
        print(f"  • {item}")
```

**Step 3: LLM-as-judge on negatives**

```python
def analyze_failure(query, response):
    analysis = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{
            "role": "system",
            "content": """Analyze why this agent response received negative feedback.
Return JSON:
{
  "failure_category": "wrong_answer | wrong_format | incomplete | hallucination | off_topic | tone",
  "root_cause": "brief explanation",
  "fix_suggestion": "what should change in the prompt or logic"
}"""
        }, {
            "role": "user",
            "content": f"Query: {query}\nResponse: {response}"
        }]
    ).choices[0].message.content
    return json.loads(analysis)

# Run on all negatives
negatives["analysis"] = negatives.apply(
    lambda row: analyze_failure(row["query"], row["response"]), axis=1
)

# Aggregate failure categories
from collections import Counter
categories = Counter([a["failure_category"] for a in negatives["analysis"]])
print(categories)
# Counter({'wrong_format': 23, 'incomplete': 18, 'hallucination': 7, ...})
```

---

## Q5. How do you close the loop — turning feedback insights into agent improvements?

**A:** Map failure categories to specific fixes:

| Failure Category | Fix |
|---|---|
| `wrong_answer` | Add few-shot examples of correct answers for this query type |
| `wrong_format` | Tighten the output schema / add format examples to system prompt |
| `incomplete` | Add instruction: "Always include X when answering Y type questions" |
| `hallucination` | Add RAG grounding; add instruction to say "I don't know" when uncertain |
| `off_topic` | Add topic boundary instructions; improve intent classifier |
| `tone` | Add tone guidelines to system prompt with examples |

**Implementing the fix:**

```python
# Example: Add few-shot examples for a mishandled query type
# Before (original system prompt):
original_prompt = """You are a customer support agent for Acme SaaS.
Answer billing, technical, and account questions helpfully."""

# After (with few-shot fix based on feedback analysis):
improved_prompt = """You are a customer support agent for Acme SaaS.
Answer billing, technical, and account questions helpfully.

When a customer asks about changing their plan:
Q: "How do I downgrade my plan?"
A: "To downgrade: Settings → Billing → Change Plan → select the plan you want → Confirm.
The change takes effect at your next billing cycle. You won't be charged the difference."

Q: "Can I switch from annual to monthly?"
A: "Yes. Settings → Billing → Change Plan → toggle to Monthly. Note that switching from
annual to monthly means losing the annual discount (typically 20%). Would you like to proceed?" """
```

**Track improvement over time:**

```python
# Compare thumbs-down rate before and after prompt change
before = df[df["timestamp"] < "2026-03-01"]["rating"].value_counts(normalize=True)
after = df[df["timestamp"] >= "2026-03-01"]["rating"].value_counts(normalize=True)

print(f"Negative rate before: {before.get('thumbs_down', 0):.1%}")
print(f"Negative rate after:  {after.get('thumbs_down', 0):.1%}")
```

---

## Q6. What is implicit feedback and when should you use it instead of explicit?

**A:** Implicit feedback is collected from user behavior without asking them directly. It has higher volume (every interaction generates signal) but is noisier.

**Retry detection (strong negative signal):**

```python
def detect_retry(messages: list) -> bool:
    """Detects if user rephrased the same question — implicit negative signal."""
    if len(messages) < 4:
        return False

    last_user = messages[-1]["content"]
    prev_user = messages[-3]["content"]  # Two turns back

    # Check semantic similarity
    embeddings = client.embeddings.create(
        model="text-embedding-3-small",
        input=[last_user, prev_user]
    ).data

    similarity = cosine_similarity(embeddings[0].embedding, embeddings[1].embedding)

    # High similarity + user asking again = likely retry
    return similarity > 0.85

def cosine_similarity(a, b):
    import numpy as np
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

**Session abandonment (negative signal):**

```python
# In your session analytics layer:
# If user asks a question and then the session ends within 30 seconds
# without a positive action (copy, click, follow-up) → flag as negative
```

**Use implicit when:** You have many users and low explicit feedback rates. At scale, even noisy implicit signals provide enough volume to be statistically meaningful.

---

## Key Numbers

| Metric | Value |
|---|---|
| Typical explicit feedback rate | 5–15% of interactions |
| Thumbs-down rate (baseline, good agent) | < 10% |
| Thumbs-down rate (needs improvement) | > 20% |
| Minimum negatives to analyze a cluster | 20+ examples |
| Golden dataset for regression testing | 50–200 examples |
| Embedding model for clustering | `text-embedding-3-small` (fastest/cheapest) |
| K for k-means clustering (starting point) | 5–10 clusters |