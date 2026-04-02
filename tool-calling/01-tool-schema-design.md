## 1. Tool Schema Design — Writing Tool Definitions LLMs Can Reliably Invoke

### Q1: What is a tool schema, and what are its mandatory components across OpenAI, Claude, and open-source models?

**Answer:**

A **tool schema** is a structured JSON definition that tells the LLM: what a tool is called, what it does, and exactly what arguments it expects. The model never sees your Python code — it only sees this JSON contract. If the schema is ambiguous, the model invokes the tool incorrectly or not at all.

**OpenAI / GPT-4o format:**
```json
{
  "type": "function",
  "function": {
    "name": "get_order_status",
    "description": "Retrieves the current status of a customer order including shipping carrier, tracking number, estimated delivery date, and any delay alerts. Use this when a user asks where their order is or when it will arrive.",
    "parameters": {
      "type": "object",
      "properties": {
        "order_id": {
          "type": "string",
          "description": "Order ID in format ORD-XXXXXX (e.g., ORD-123456). Found on the confirmation email."
        },
        "include_history": {
          "type": "boolean",
          "description": "If true, includes full shipping event history. Default false.",
          "default": false
        }
      },
      "required": ["order_id"],
      "additionalProperties": false
    },
    "strict": true
  }
}
```

**Anthropic / Claude format:**
```python
{
    "name": "get_order_status",
    "description": "Retrieves the current status of a customer order...",
    "input_schema": {          # Claude uses input_schema, not parameters
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "Order ID in format ORD-XXXXXX"
            }
        },
        "required": ["order_id"]
    }
}
```

**Key structural differences:**

| Field | OpenAI | Claude | LangChain (open-source) |
|---|---|---|---|
| Wrapper key | `function.parameters` | `input_schema` | `args_schema` (Pydantic) |
| Strict mode | `"strict": true` | Not supported — prompt-enforced | N/A |
| Tool type declaration | `"type": "function"` | Not required | Not required |
| Multiple tool names | `tools` array | `tools` array | `tools` list |

**Mandatory components in every schema:**
1. `name` — snake_case, no spaces, ≤64 characters
2. `description` — the most important field (see Q2)
3. `parameters`/`input_schema` — JSON Schema object with typed properties
4. `required` array — explicitly list non-optional fields

---

### Q2: Why is the description field the most critical part of a tool schema, and what makes a good vs. bad description?

**Answer:**

The model's tool selection decision is almost entirely driven by the description. The model reads descriptions, reasons about which tool matches the user's intent, and then fills in parameters. A bad description causes wrong tool selection or hallucinated parameters — bugs that don't throw exceptions.

**Bad description:**
```json
"description": "Gets order status"
```
Problems: What does "status" include? When should I call this vs. `get_shipment_details`? What format is the order ID?

**Good description:**
```json
"description": "Retrieves the current delivery status of a customer order — including carrier name, tracking number, estimated delivery date, and any delay alerts. Call this when the user asks 'where is my order', 'when will it arrive', or 'is my order delayed'. Do NOT use for returns or refunds — use process_return instead. Requires the ORD-XXXXXX format order ID from their confirmation email."
```

**What a good description must include:**
1. **What the tool returns** — not what it's called, but what data comes back
2. **When to call it** — explicit trigger phrases or conditions
3. **When NOT to call it** — disambiguation from similar tools (the most commonly skipped)
4. **Parameter format hints** — "in format ORD-XXXXXX" prevents the model from guessing

**Parameter-level descriptions follow the same rules:**
```json
"order_id": {
  "type": "string",
  "description": "The order identifier in ORD-XXXXXX format (6 digits). Found in the confirmation email subject line. Example: ORD-482910"
}
```
The `Example:` pattern in parameter descriptions is the single highest-leverage addition for reducing parameter hallucination.

**Real-world impact:** Stripe's internal tooling teams report that tool call accuracy improved from ~72% to ~94% simply by rewriting descriptions to include "when NOT to use" clauses and format examples. No code changes — description only.

---

### Q3: How do you use JSON Schema features (enums, nested objects, arrays, anyOf) in tool parameters, and when does each matter?

**Answer:**

The model honors JSON Schema constraints during generation (especially with `"strict": true` in OpenAI). Using the right schema type prevents the model from passing garbage values.

**Enums — constrain to a fixed set of values:**
```json
"status_filter": {
  "type": "string",
  "enum": ["pending", "shipped", "delivered", "cancelled"],
  "description": "Filter orders by status. Use 'shipped' for in-transit orders."
}
```
Without enum: the model might pass `"in-transit"` or `"processing"` — strings your API rejects. With enum: it can only pick from your list.

**Nested objects — group related parameters:**
```json
"date_range": {
  "type": "object",
  "description": "Optional date range filter for the search",
  "properties": {
    "start": {"type": "string", "description": "ISO 8601 date, e.g. 2026-01-01"},
    "end":   {"type": "string", "description": "ISO 8601 date, e.g. 2026-12-31"}
  },
  "required": ["start", "end"]
}
```

**Arrays — when the tool accepts multiple values:**
```json
"product_ids": {
  "type": "array",
  "items": {"type": "string"},
  "description": "List of product IDs to look up. Max 20 items.",
  "maxItems": 20
}
```

**anyOf — optional typed union (use sparingly):**
```json
"identifier": {
  "anyOf": [
    {"type": "string", "description": "Order ID in ORD-XXXXXX format"},
    {"type": "integer", "description": "Numeric customer ID"}
  ]
}
```
Warning: `anyOf` increases schema complexity and can confuse models. Prefer separate parameters with a mutual-exclusivity note in the description.

**Strict mode (`"strict": true`) in OpenAI:**
- Forces the model to output only fields defined in the schema
- Disallows additional properties
- Requires all `required` fields to be present
- **Caveat:** Strict mode requires all nested objects to also declare `additionalProperties: false` and all properties to be `required` or listed under an `optional` pattern. This can make complex schemas verbose.

---

### Q4: How do tool schemas differ across providers, and how do you write provider-agnostic tool definitions?

**Answer:**

When your agent must run on multiple LLM providers (dev on Claude, prod on GPT-4o, fallback to a local model), maintaining separate tool schemas per provider is a maintenance nightmare. The solution: write in a canonical format and translate at call time.

**Provider schema translation:**

```python
from typing import Any

def canonical_tool(name: str, description: str, properties: dict, required: list) -> dict:
    """Single source of truth for a tool definition."""
    return {
        "name": name,
        "description": description,
        "properties": properties,
        "required": required
    }

def to_openai(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": {
                "type": "object",
                "properties": tool["properties"],
                "required": tool["required"],
                "additionalProperties": False
            },
            "strict": True
        }
    }

def to_claude(tool: dict) -> dict:
    return {
        "name": tool["name"],
        "description": tool["description"],
        "input_schema": {
            "type": "object",
            "properties": tool["properties"],
            "required": tool["required"]
        }
    }

def to_langchain(tool: dict):
    """Use Pydantic model generation for LangChain."""
    from pydantic import create_model, Field
    fields = {
        k: (str, Field(description=v.get("description", "")))
        for k, v in tool["properties"].items()
    }
    return create_model(tool["name"], **fields)
```

**LangChain `@tool` decorator — the cleanest abstraction:**
```python
from langchain_core.tools import tool

@tool
def get_order_status(order_id: str, include_history: bool = False) -> str:
    """Retrieves delivery status for a customer order.
    
    Args:
        order_id: Order ID in ORD-XXXXXX format (e.g., ORD-123456)
        include_history: If True, includes full shipping event history
    
    Returns:
        JSON string with status, carrier, tracking number, ETA
    """
    # implementation
    return f"Order {order_id}: In transit, arriving 2026-04-05"
```
LangChain auto-generates the JSON schema from the docstring and type hints. This is the recommended approach when you don't need fine-grained schema control — it keeps the schema co-located with the implementation.

---

### Q5: What are the most common tool schema mistakes that cause production failures, and how do you prevent them?

**Answer:**

| Mistake | Symptom | Fix |
|---|---|---|
| Missing `required` array | Model omits non-optional params | Always declare every non-optional field in `required` |
| Vague description | Wrong tool selected | Add "when to use / when NOT to use" |
| No parameter format example | Model invents format (`"order123"` instead of `"ORD-123456"`) | Add `Example:` to every string parameter |
| Overlapping tool names | Model picks wrong tool | Add disambiguation section to each description |
| Too many tools (>20 in context) | Tool selection accuracy drops sharply | Use categorized routing or RAG retrieval |
| Mutable side effects not flagged | Model calls write tools when read intended | Add "WARNING: This action modifies data" to destructive tools |
| No `additionalProperties: false` | Model hallucinates extra fields | Always set this in strict mode |
| Snake_case inconsistency | Some models struggle with camelCase | Standardize on snake_case for all parameter names |

**Tool count vs. selection accuracy (empirical):**

| Tools in context | Typical selection accuracy |
|---|---|
| 1–5 | ~99% |
| 6–10 | ~95–97% |
| 11–20 | ~88–93% |
| 21–50 | ~75–85% |
| 50+ | <70% — routing required |

These numbers are from LangChain and Anthropic benchmarking on GPT-4o and Claude 3.5 Sonnet. Accuracy degrades faster with semantically similar tools than with dissimilar ones.

---

### Key Numbers to Memorize

| Metric | Value |
|---|---|
| Max tool name length (OpenAI) | 64 characters |
| Tool selection accuracy drop threshold | >20 tools in context |
| Accuracy with 1–5 tools | ~99% |
| Accuracy with 50+ tools in context | <70% |
| Accuracy improvement from description rewrite | 72% → 94% (Stripe internal) |
| Recommended tools per context (practical) | ≤10 for best accuracy |
| `strict: true` support | OpenAI GPT-4o-2024-08-06+ only |