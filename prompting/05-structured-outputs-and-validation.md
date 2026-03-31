## 5. Structured Outputs, JSON / YAML / TOON & Pydantic Validation

### Q1: Why do agents need structured outputs and what breaks without them?

**Answer:**

Agents are software systems — they parse model responses programmatically. Free-text responses break agent pipelines because:

1. **No reliable parsing:** "I think you should call the weather API with city=London" requires fragile regex or NLP to extract `city=London`. Structured output gives you `{"tool": "get_weather", "city": "London"}` directly.

2. **No type safety:** "The confidence is pretty high" can't be compared to a threshold. `{"confidence": 0.87}` can.

3. **Downstream failures cascade:** If the LLM outputs "```json\n{...}\n```" with markdown fences but your parser expects raw JSON, your entire agent crashes.

4. **Non-deterministic schemas:** Without a schema, the model may include different keys on different runs — some calls return `{"name": "..."}`, others return `{"full_name": "..."}`.

**What breaks in practice:**
```python
# Brittle — breaks on format variation
response = llm("Extract the name and age from: John Smith is 30 years old")
# Response sometimes: '{"name": "John Smith", "age": 30}'
# Response sometimes: 'Name: John Smith\nAge: 30'
# Response sometimes: 'The person is John Smith, aged 30.'

# All three need different parsing logic — this is a production bug
```

---

### Q2: What are the three main structured output approaches and when do you use each?

**Answer:**

**Approach 1: Prompt-based JSON (cheapest, least reliable)**

Ask the model to return JSON in the prompt. No API enforcement.

```python
prompt = """
Extract person details from this text and return ONLY valid JSON with no markdown.
Format: {"name": string, "age": integer, "email": string or null}

Text: "John Smith is 30 years old. Contact: john@example.com"
"""
```

- ✅ Works on any model, zero setup
- ❌ ~85–90% reliability — model sometimes adds preamble, markdown fences, or wrong keys
- Use for: prototyping, non-critical workflows, models without native structured output

---

**Approach 2: JSON mode / response_format (better reliability)**

API-level enforcement that the output is valid JSON. Does NOT enforce a specific schema.

```python
# OpenAI JSON mode
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[{"role": "user", "content": "Extract: John is 30. Return JSON."}]
)
# Guaranteed valid JSON, but schema may vary
```

- ✅ Always valid JSON (no parse errors)
- ❌ Schema not enforced — key names may differ across runs
- Use for: when you need valid JSON but can tolerate flexible schemas

---

**Approach 3: Strict structured outputs (most reliable)**

API enforces exact schema adherence using Pydantic or JSON Schema. Model cannot deviate.

```python
from pydantic import BaseModel
from openai import OpenAI

class PersonExtraction(BaseModel):
    name: str
    age: int
    email: str | None = None

client = OpenAI()
response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",  # Must use a model that supports structured outputs
    messages=[{"role": "user", "content": "Extract: John Smith, 30, john@example.com"}],
    response_format=PersonExtraction
)

person = response.choices[0].message.parsed  # Already a PersonExtraction object
print(person.name)   # "John Smith"
print(person.age)    # 30
print(person.email)  # "john@example.com"
```

**Anthropic equivalent — tool use as structured output:**
```python
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[{
        "name": "extract_person",
        "description": "Extract person details from text",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full name"},
                "age": {"type": "integer", "description": "Age in years"},
                "email": {"type": "string", "description": "Email address"}
            },
            "required": ["name", "age"]
        }
    }],
    tool_choice={"type": "tool", "name": "extract_person"},  # Force tool use
    messages=[{"role": "user", "content": "Extract: John Smith, 30, john@example.com"}]
)

tool_input = response.content[0].input  # Always structured per schema
```

- ✅ Schema strictly enforced — 99%+ reliability
- ✅ Returned as typed Python objects
- ❌ Only on supported models (GPT-4o-2024-08-06+, Claude via tool use)
- Use for: production agents, data extraction pipelines, any system that parses output

---

### Q3: What is Pydantic and how do you use it for LLM output validation?

**Answer:**

**Pydantic** is the de facto Python library for data validation using type annotations. It validates data at runtime and raises clear, structured errors when validation fails.

**Basic usage:**
```python
from pydantic import BaseModel, field_validator, model_validator
from typing import Literal, Optional
from datetime import datetime

class TicketClassification(BaseModel):
    category: Literal["billing", "technical", "refund", "general"]
    priority: int  # 1 (low) to 5 (critical)
    summary: str
    requires_human: bool
    estimated_resolution_hours: Optional[float] = None
    
    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v):
        if not 1 <= v <= 5:
            raise ValueError(f"Priority must be 1-5, got {v}")
        return v
    
    @field_validator("summary")
    @classmethod
    def summary_not_empty(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Summary must be at least 10 characters")
        return v.strip()
    
    @model_validator(mode="after")
    def high_priority_needs_human(self):
        if self.priority >= 4 and not self.requires_human:
            raise ValueError("Priority 4+ tickets must require human review")
        return self
```

**Self-healing validation loop (critical production pattern):**
```python
import json
from pydantic import ValidationError

def extract_with_retry(text: str, schema: type[BaseModel], max_retries: int = 3) -> BaseModel:
    prompt = f"Extract structured data from this text. Return JSON only.\nText: {text}"
    
    for attempt in range(max_retries):
        response = llm(prompt)
        
        try:
            # Strip markdown fences if present
            clean = response.strip().removeprefix("```json").removesuffix("```").strip()
            data = json.loads(clean)
            return schema.model_validate(data)
        
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == max_retries - 1:
                raise
            
            # Feed error back into the next prompt
            prompt = f"""
Your previous response had this error: {str(e)}

Please fix it and return valid JSON matching this schema:
{schema.model_json_schema()}

Original text: {text}
"""
    
    raise RuntimeError("Exhausted retries")
```

**Why self-healing works:** The model can parse its own ValidationError messages and correct the issue 70–80% of the time on the first retry.

---

### Q4: What is JSON vs YAML vs TOON and which should you use?

**Answer:**

**JSON (JavaScript Object Notation):**
```json
{
  "customer": {
    "name": "John Smith",
    "age": 30,
    "orders": ["ORD-001", "ORD-002"]
  }
}
```
- ✅ Universal — every language, every tool, every LLM has native JSON support
- ✅ LLMs have massive training exposure to JSON
- ✅ Strict spec — easy to validate
- ❌ Verbose — quotes on every key, nested brackets
- ❌ No comments allowed
- **Token count (above example): ~35 tokens**

---

**YAML (YAML Ain't Markup Language):**
```yaml
customer:
  name: John Smith
  age: 30
  orders:
    - ORD-001
    - ORD-002
```
- ✅ More readable for humans
- ✅ Supports comments (`# this is a comment`)
- ✅ ~20–30% fewer tokens than equivalent JSON
- ❌ Indentation-sensitive — easy to corrupt in LLM outputs
- ❌ Less common in AI pipelines — more parsing edge cases
- **Token count (above example): ~25 tokens**

---

**TOON (Token-Optimized Object Notation):**
```
customer.name=John Smith|customer.age=30|customer.orders=ORD-001,ORD-002
```
- ✅ Most token-efficient — ~40–60% fewer tokens than JSON
- ❌ No standard spec — you must write your own parser
- ❌ LLMs have very little training data on TOON — reliability is lower
- ❌ Nested structures are awkward
- **Token count (above example): ~15 tokens**

**Comparison table:**

| Format | Token efficiency | LLM reliability | Nested support | Ecosystem |
|---|---|---|---|---|
| JSON | Low (baseline) | Highest (extensive training) | Excellent | Universal |
| YAML | Medium (~25% less) | High | Good | Good |
| TOON | High (~50% less) | Lower (less training data) | Poor | None (custom) |

**Senior engineer decision framework:**
- **Default:** Use JSON. Period. The token savings from YAML/TOON rarely justify the added fragility.
- **Use YAML** if: the structured data is being shown to humans who will read/edit it (config files, prompt templates)
- **Use TOON** only if: you have a high-volume, flat-schema use case, you have the engineering bandwidth to maintain a custom parser, and you've empirically validated it works reliably with your specific model
- **Never use TOON in production without** a JSON fallback path and a reliability test

**Real-world token math:** If you're processing 10M structured outputs per month:
- JSON → TOON = ~20 fewer tokens/call = 200M tokens/month saved
- At GPT-4o pricing: 200M × $2.50/1M = $500/month saved
- vs. engineering cost of maintaining TOON parser + handling failures: likely not worth it unless scale is 10× higher

---

### Q5: How do you handle streaming structured outputs in production?

**Answer:**

**The problem:** Streaming returns tokens one at a time. JSON is not valid until the closing `}` arrives — you can't parse it mid-stream.

**Solution 1: Stream display, parse at end**
```python
buffer = ""
for chunk in client.chat.completions.create(..., stream=True):
    delta = chunk.choices[0].delta.content or ""
    buffer += delta
    print(delta, end="", flush=True)  # Stream to UI

# Parse only after stream completes
result = PersonExtraction.model_validate_json(buffer)
```

**Solution 2: Streaming with partial JSON (advanced)**
Use `jsonpatch` or streaming JSON parsers like `ijson` to process partial JSON as it arrives:
```python
import ijson

def stream_and_parse(stream):
    buffer = b""
    for chunk in stream:
        buffer += chunk.encode()
        try:
            # Try to parse incrementally
            parser = ijson.items(buffer, "")
            for item in parser:
                yield item  # Yield complete objects as they become available
        except:
            continue  # Not complete yet, keep buffering
```

**Solution 3: OpenAI streaming with structured outputs**
OpenAI's `.stream()` method handles this automatically, buffering until each structured field is complete:
```python
with client.beta.chat.completions.stream(
    model="gpt-4o",
    messages=[...],
    response_format=PersonExtraction
) as stream:
    for event in stream:
        if event.type == "content.delta":
            print(event.delta, end="")  # Stream text
    
    result = stream.get_final_completion().choices[0].message.parsed
```

**Key numbers:** Streaming reduces time-to-first-token from 2–5s (full generation) to 200–500ms. For user-facing UIs, always stream. For backend pipelines, batch without streaming for simpler code.

---