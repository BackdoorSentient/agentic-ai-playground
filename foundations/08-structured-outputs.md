## 8. Structured Outputs

### Q1: What are structured outputs and why are they critical for agentic systems?

**Answer:**

**Structured outputs** constrain the model to generate responses that conform to a predefined schema — typically JSON, but also YAML, XML, or custom formats.

**Why agents need structured outputs:**

Agents don't read model responses as humans do. An agent needs to:
- Parse the response programmatically
- Extract tool call parameters
- Route decisions based on field values
- Compose responses into downstream systems

If the model outputs: "I think you should call the get_weather function with city=London" in free text, parsing this reliably is brittle. Structured output produces:
```json
{
  "tool": "get_weather",
  "parameters": {"city": "London"}
}
```

Which is directly parseable and type-safe.

---

### Q2: How do you implement structured outputs in the OpenAI and Anthropic APIs?

**Answer:**

**OpenAI — JSON mode (basic):**
```python
response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[{"role": "user", "content": "Extract name and age from: John is 30 years old. Return JSON."}]
)
```
⚠️ JSON mode only guarantees valid JSON, not a specific schema.

**OpenAI — Structured Outputs with Pydantic (strict):**
```python
from pydantic import BaseModel
from openai import OpenAI

class PersonExtraction(BaseModel):
    name: str
    age: int
    email: str | None = None

response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": "Extract: John Smith, 30, john@example.com"}],
    response_format=PersonExtraction
)
person = response.choices[0].message.parsed  # Already a PersonExtraction object
```

**Anthropic — Tool use for structured outputs:**
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
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
    }],
    messages=[{"role": "user", "content": "Extract from: John is 30 years old."}]
)
```

---

### Q3: What is Pydantic and why is it the standard for validation in AI pipelines?

**Answer:**

**Pydantic** is a Python data validation library that uses type annotations to define schemas and validate data at runtime.

**Why it's standard for AI pipelines:**
1. Schema definition is just Python type hints — no separate schema files.
2. Automatically validates types, required fields, and constraints.
3. Raises clear errors when validation fails — critical for catching LLM hallucinations that produce invalid JSON.
4. Integrates directly with FastAPI (request/response validation) and LangChain.

**Example: Validating LLM output:**
```python
from pydantic import BaseModel, field_validator
from typing import Literal

class TicketClassification(BaseModel):
    category: Literal["billing", "technical", "refund", "other"]
    priority: int  # Must be 1–5
    summary: str
    
    @field_validator("priority")
    def priority_must_be_valid(cls, v):
        if not 1 <= v <= 5:
            raise ValueError("Priority must be between 1 and 5")
        return v

# Parse and validate LLM JSON response
try:
    result = TicketClassification.model_validate_json(llm_response)
except ValidationError as e:
    # Retry with the error feedback in the prompt
    print(f"Validation failed: {e}")
```

**What to retry:** When validation fails, include the error message in the next prompt: "Your previous response failed validation: {error}. Please correct and try again." This is called **self-healing structured output**.

---

### Q4: What is the TOON format and how does it compare to JSON for AI use?

**Answer:**

**TOON (Token-Optimized Object Notation)** is a compact alternative to JSON designed to reduce token usage when passing structured data to/from LLMs.

**JSON (verbose):**
```json
{"name": "John", "age": 30, "city": "London"}
```
Approx. 42 characters / ~12 tokens.

**TOON (compact):**
```
name=John|age=30|city=London
```
Approx. 28 characters / ~8 tokens — ~33% fewer tokens.

**Trade-offs:**

| Aspect | JSON | TOON |
|---|---|---|
| Token efficiency | Lower | Higher (~20–40% savings) |
| Ecosystem support | Universal (all parsers) | Custom implementation needed |
| Nested structures | Native support | Awkward / less readable |
| LLM reliability | Models trained extensively on JSON | Less training data, lower reliability |
| Debugging | Easy | Harder |

**Senior engineer verdict:** TOON is interesting for high-volume, simple flat structures where token costs matter. For complex nested schemas (which agents typically need), JSON with strict schema validation is more reliable. Do not use TOON for schemas with deeply nested objects.

---