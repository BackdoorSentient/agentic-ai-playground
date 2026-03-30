## 10. FastAPI Framework

### Q1: What is FastAPI and why is it the standard choice for AI backend services?

**Answer:**

**FastAPI** is a modern Python web framework for building APIs, built on top of Starlette (ASGI) and Pydantic.

**Why it dominates AI backends:**

1. **Native Pydantic integration:** Request and response schemas are Pydantic models — the same library used for LLM structured outputs. Zero friction.

2. **Async-first:** Built on ASGI (Asynchronous Server Gateway Interface). LLM API calls are I/O-bound — async lets you handle 100+ concurrent requests on a single server thread while awaiting model responses.

3. **Auto-generated docs:** FastAPI automatically generates OpenAPI (Swagger) and ReDoc documentation from your type hints. Critical for team collaboration.

4. **Performance:** Comparable to Node.js and Go for I/O-bound tasks (the bottleneck for LLM services is always the model latency, not the framework).

5. **Type safety end-to-end:** From HTTP request → Python business logic → LLM schema → response, everything is typed and validated.

---

### Q2: What does a production-grade FastAPI LLM service look like?

**Answer:**

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from openai import AsyncOpenAI
import asyncio

app = FastAPI(title="AI Classification Service")
client = AsyncOpenAI()

# Request schema
class ClassificationRequest(BaseModel):
    text: str
    max_tokens: int = 500

# Response schema (structured output)
class ClassificationResult(BaseModel):
    category: str
    confidence: float
    reasoning: str

@app.post("/classify", response_model=ClassificationResult)
async def classify_text(request: ClassificationRequest):
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Classify the text. Return JSON with category, confidence (0-1), reasoning."},
                {"role": "user", "content": request.text}
            ],
            max_tokens=request.max_tokens
        )
        import json
        result = json.loads(response.choices[0].message.content)
        return ClassificationResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check — always needed for Kubernetes/load balancer probes
@app.get("/health")
async def health():
    return {"status": "ok"}
```

**Key production considerations:**

| Concern | Solution |
|---|---|
| Rate limiting | `slowapi` or API gateway (Kong, nginx) |
| Authentication | OAuth2 / API keys via `Depends()` |
| Timeouts | `asyncio.wait_for(llm_call(), timeout=30)` |
| Retries | `tenacity` library with exponential backoff |
| Streaming | `StreamingResponse` + SSE for real-time output |
| Observability | OpenTelemetry / Langfuse middleware |

---

### Q3: How do you implement streaming responses in FastAPI for LLM outputs?

**Answer:**

Streaming is critical for user experience — users see text appear word-by-word instead of waiting 10+ seconds for the full response.

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import asyncio

app = FastAPI()
client = AsyncOpenAI()

async def generate_stream(prompt: str):
    """Async generator that yields SSE-formatted chunks"""
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            # Server-Sent Events format
            yield f"data: {delta}\n\n"
    yield "data: [DONE]\n\n"

@app.get("/stream")
async def stream_response(prompt: str):
    return StreamingResponse(
        generate_stream(prompt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

**Time-to-First-Token (TTFT):** The key UX metric for streaming. Users perceive streaming as fast even if total generation time is the same, because they see output start in <1 second.

**TTFT benchmarks (approximate, vary by load):**
- GPT-4o: ~300–600ms TTFT
- Claude Sonnet: ~400–800ms TTFT
- Gemini 1.5 Pro: ~500–1000ms TTFT

---

## 11. Hands-On Exercises

### Exercise 1: Temperature Comparison

**Task:** Run the same prompt at temperatures 0.1, 0.5, and 1.0 five times each. Document output variance.

**Code:**

```python
from openai import OpenAI
import json

client = OpenAI()
prompt = "Write a one-sentence description of how machine learning works."

results = {}

for temp in [0.1, 0.5, 1.0]:
    outputs = []
    for run in range(5):
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=temp,
            messages=[{"role": "user", "content": prompt}]
        )
        outputs.append(response.choices[0].message.content)
    results[temp] = outputs

# Analysis
for temp, outputs in results.items():
    unique = len(set(outputs))
    print(f"\nTemp={temp}: {unique}/5 unique responses")
    for i, out in enumerate(outputs):
        print(f"  Run {i+1}: {out[:100]}...")
```

**Expected observations:**
- **T=0.1:** Nearly identical across all 5 runs. Same phrasing, same structure.
- **T=0.5:** Moderate variation in word choice. Same core meaning, different expressions.
- **T=1.0:** Noticeably different sentence structures across runs. Occasionally a run goes off in an unexpected direction.

**What to document:** Character count variance, semantic similarity (use cosine similarity of embeddings for rigorous comparison), and any factual inconsistencies at high temperature.

---

### Exercise 2: Token Economics Calculator

**Pricing as of early 2025 (verify at time of use — prices change):**

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| GPT-4o | $2.50 | $10.00 |
| GPT-4o mini | $0.15 | $0.60 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Claude 3 Haiku | $0.25 | $1.25 |
| Gemini 1.5 Pro | $1.25 (≤128K) | $5.00 (≤128K) |
| Gemini 1.5 Flash | $0.075 | $0.30 |

**Calculator code:**

```python
import tiktoken

PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
}

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Estimate tokens using tiktoken (GPT tokenizer as approximation)"""
    enc = tiktoken.encoding_for_model("gpt-4o")
    return len(enc.encode(text))

def estimate_cost(prompt: str, response: str, model: str) -> dict:
    input_tokens = count_tokens(prompt)
    output_tokens = count_tokens(response)
    
    pricing = PRICING[model]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
        "cost_per_1000_calls": round((input_cost + output_cost) * 1000, 4)
    }

# Example usage
sample_prompt = """You are a helpful assistant. 
User: Explain the difference between supervised and unsupervised learning in 3 sentences."""

sample_response = """Supervised learning trains models on labeled data where the correct output is known, 
learning to map inputs to outputs. Unsupervised learning finds patterns in unlabeled data without 
predefined correct answers, discovering hidden structure. The key difference is that supervised learning 
requires labeled training examples while unsupervised learning works with raw, unlabeled data."""

for model in PRICING:
    result = estimate_cost(sample_prompt, sample_response, model)
    print(f"{model}: ${result['total_cost_usd']:.6f} per call | ${result['cost_per_1000_calls']:.4f} per 1K calls")
```

**Expected output (approximate):**
```
gpt-4o: $0.000058 per call | $0.0578 per 1K calls
gpt-4o-mini: $0.000004 per call | $0.0036 per 1K calls
claude-3-5-sonnet: $0.000069 per call | $0.0690 per 1K calls
claude-3-haiku: $0.000006 per call | $0.0058 per 1K calls
gemini-1.5-pro: $0.000029 per call | $0.0290 per 1K calls
gemini-1.5-flash: $0.0000018 per call | $0.0018 per 1K calls
```

**Key insight:** For high-volume applications (millions of calls/day), the difference between GPT-4o ($2.50/1M) and Gemini Flash ($0.075/1M) is a **33x cost difference**. Model selection is an economic decision, not just a performance one.

---
