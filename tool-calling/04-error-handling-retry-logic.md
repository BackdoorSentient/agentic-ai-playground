## 4. Error Handling & Retry Logic — Graceful Degradation When Tools Fail

### Q1: What are the four categories of tool failure, and why must each be handled differently?

**Answer:**

Tools fail in fundamentally different ways, and a one-size-fits-all `try/except` is not a production error handling strategy. Each failure category has a different cause, retry profile, and user impact.

**Category 1: Transient failures (retry immediately)**
- Network timeouts, temporary API unavailability, rate limiting (429)
- The tool will succeed on retry — the failure is environmental, not logical
- Retry with exponential backoff: 1s → 2s → 4s → give up after 3 attempts

**Category 2: Input validation failures (do not retry — fix the input)**
- Malformed parameters: wrong format, missing required field, out-of-range value
- Retrying with the same input produces the same failure
- Return a structured error to the LLM so it can self-correct its call

**Category 3: Authorization / permission failures (do not retry — escalate)**
- API key invalid, user lacks permission, resource is private
- Retrying is pointless — this requires a human or configuration fix
- Return an informative error; do not retry; notify appropriately

**Category 4: Semantic / business logic failures (depends)**
- The tool ran successfully but returned "not found", "no results", "order cancelled"
- Not a code error — the tool worked; the result is just unhelpful
- Do not retry; pass the result to the LLM so it can reason about next steps

**Classification in code:**

```python
import httpx
from enum import Enum

class ToolFailureCategory(Enum):
    TRANSIENT = "transient"          # retry
    INPUT_ERROR = "input_error"      # self-correct
    AUTH_ERROR = "auth_error"        # escalate
    BUSINESS_LOGIC = "business"      # pass to LLM

def classify_error(error: Exception) -> ToolFailureCategory:
    if isinstance(error, (TimeoutError, ConnectionError)):
        return ToolFailureCategory.TRANSIENT
    if isinstance(error, httpx.TimeoutException):
        return ToolFailureCategory.TRANSIENT
    if isinstance(error, httpx.HTTPStatusError):
        if error.response.status_code == 429:
            return ToolFailureCategory.TRANSIENT   # rate limit — retry after backoff
        if error.response.status_code in (401, 403):
            return ToolFailureCategory.AUTH_ERROR
        if error.response.status_code == 422:
            return ToolFailureCategory.INPUT_ERROR # unprocessable entity — bad params
    if isinstance(error, ValueError):
        return ToolFailureCategory.INPUT_ERROR
    return ToolFailureCategory.TRANSIENT  # default: assume transient, try once more
```

---

### Q2: How do you implement exponential backoff with jitter for transient tool failures?

**Answer:**

**Exponential backoff** doubles the wait time on each retry. **Jitter** adds randomness to prevent the "thundering herd" problem — where hundreds of agents all retry at the same second and overwhelm the downstream service.

```python
import asyncio
import random
import time
import logging
from functools import wraps
from typing import Callable, Any, Type

logger = logging.getLogger(__name__)

def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (TimeoutError, ConnectionError)
):
    """Decorator for automatic retry with exponential backoff + jitter."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break  # exhausted retries — raise below
                    
                    # Calculate delay: base * 2^attempt
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    
                    # Full jitter: random value in [0, delay]
                    # (better than multiplicative jitter for high-concurrency systems)
                    if jitter:
                        delay = random.uniform(0, delay)
                    
                    logger.warning(
                        f"Tool call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    if jitter:
                        delay = random.uniform(0, delay)
                    time.sleep(delay)
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Usage
@retry_with_backoff(max_retries=3, base_delay=1.0, retryable_exceptions=(TimeoutError, ConnectionError))
async def get_weather(city: str) -> dict:
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"https://api.weather.com/v1/{city}")
        response.raise_for_status()
        return response.json()
```

**Retry schedule with base_delay=1.0, factor=2.0, jitter=True:**

| Attempt | Nominal delay | With full jitter |
|---|---|---|
| 1 | 1.0s | 0–1.0s |
| 2 | 2.0s | 0–2.0s |
| 3 | 4.0s | 0–4.0s |
| Give up | — | Total: 0–7s worst case |

**Why full jitter beats no jitter:** At 1,000 concurrent agents all hitting a rate limit simultaneously, no-jitter causes all 1,000 to retry at t+1s, t+3s, t+7s — stampeding the service in waves. Full jitter spreads retries uniformly across [0, delay], reducing peak load by ~50%.

---

### Q3: How do you return structured errors to the LLM so it can self-correct bad tool calls?

**Answer:**

When a tool fails due to bad input, the LLM made the mistake. Returning a generic `"error"` string gives it nothing to work with. A structured error with the exact issue + guidance allows the model to self-correct its next call.

**Structured error format:**
```python
from dataclasses import dataclass
from typing import Any
import json

@dataclass
class ToolError:
    error_type: str          # "validation", "not_found", "rate_limit", "server_error"
    message: str             # human-readable error description
    field: str | None        # which parameter caused the issue (for validation errors)
    expected_format: str | None  # what the correct format is
    retry_after: int | None  # seconds to wait (for rate limits)
    
    def to_tool_result(self, tool_use_id: str) -> dict:
        """Format as a tool_result block for the LLM."""
        content = {
            "error": True,
            "error_type": self.error_type,
            "message": self.message,
        }
        if self.field:
            content["invalid_field"] = self.field
        if self.expected_format:
            content["expected_format"] = self.expected_format
        if self.retry_after:
            content["retry_after_seconds"] = self.retry_after
        
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "is_error": True,       # Claude-specific: marks this as an error result
            "content": json.dumps(content)
        }

# Examples:
# Validation error:
ToolError(
    error_type="validation",
    message="Invalid order_id format",
    field="order_id",
    expected_format="ORD-XXXXXX where X is a digit (e.g., ORD-123456). The value 'order-abc' is invalid.",
    retry_after=None
)

# Rate limit:
ToolError(
    error_type="rate_limit",
    message="Weather API rate limit exceeded",
    field=None,
    expected_format=None,
    retry_after=30
)

# Not found:
ToolError(
    error_type="not_found",
    message="Order ORD-999999 does not exist in the system",
    field=None,
    expected_format=None,
    retry_after=None
)
```

**What the LLM does with structured errors:**

When you return a validation error with `expected_format`, the model reads it and corrects its next tool call. Empirically, structured error feedback causes successful self-correction on the next attempt in 70–80% of cases (same finding as self-healing Pydantic retry from Day 2).

**Claude's `is_error: true`:** This field tells Claude the tool_result contains an error. Claude handles this gracefully — it acknowledges the error in its next response rather than treating the error JSON as a successful result.

---

### Q4: What is the fallback hierarchy, and how do you design graceful degradation for end users?

**Answer:**

A **fallback hierarchy** defines what the agent does when a tool is unavailable. The goal: never surface a raw error to the user. Every failure has a planned degradation path.

**Three-level fallback pattern:**

```python
from typing import Optional

async def get_weather_with_fallback(city: str) -> dict:
    """
    Level 1: Try primary weather API
    Level 2: Try backup weather API
    Level 3: Return cached/stale data with staleness warning
    Level 4: Apologize gracefully with helpful context
    """
    
    # Level 1: Primary
    try:
        return await primary_weather_api(city)
    except (TimeoutError, httpx.HTTPStatusError) as e:
        logger.warning(f"Primary weather API failed: {e}")
    
    # Level 2: Backup
    try:
        return await backup_weather_api(city)
    except Exception as e:
        logger.warning(f"Backup weather API failed: {e}")
    
    # Level 3: Cache (stale data is better than nothing)
    cached = await cache.get(f"weather:{city}")
    if cached:
        return {**cached, "stale": True, "cached_at": cached["timestamp"]}
    
    # Level 4: Graceful failure (user-friendly, not a stack trace)
    return {
        "error": True,
        "user_message": f"I'm unable to fetch weather data for {city} right now — "
                        f"the weather service appears to be temporarily unavailable. "
                        f"Please try again in a few minutes or check weather.com directly.",
        "city": city
    }
```

**User-facing error message design principles:**

| Principle | Bad example | Good example |
|---|---|---|
| No raw errors | `"HTTPStatusError: 503 Service Unavailable"` | `"Weather service temporarily unavailable"` |
| Actionable | `"Error fetching data"` | `"Try again in 2 minutes or check weather.com"` |
| Honest | `"I don't know the weather"` | `"I'm unable to fetch real-time weather right now"` |
| Preserves trust | `"Something went wrong"` | `"The flight data API is temporarily down — I can still help with other questions"` |

**Agent-level fallback — when the whole tool call loop fails:**

```python
MAX_TOOL_ERRORS = 3

def agent_loop_with_fallback(messages: list) -> str:
    error_count = 0
    
    while True:
        response = llm.invoke(messages)
        
        if response.stop_reason == "end_turn":
            return extract_text(response)
        
        if response.stop_reason == "tool_use":
            results = execute_tools(response)
            error_count += sum(1 for r in results if r.get("is_error"))
            
            if error_count >= MAX_TOOL_ERRORS:
                # Too many tool failures — bail out gracefully
                return (
                    "I'm having trouble accessing the tools needed to answer your question right now. "
                    "Here's what I know without real-time data: [provide best-effort response from model knowledge]"
                )
            
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": results})
```

---

### Q5: How do you implement tool call timeout handling, and what are the right timeout values?

**Answer:**

Every external tool call must have an explicit timeout. Without one, a hanging API call blocks your agent indefinitely — and in a multi-user system, it blocks a thread for all users.

```python
import asyncio
import httpx

async def call_tool_with_timeout(
    tool_func,
    tool_args: dict,
    timeout_seconds: float = 10.0
) -> dict:
    """Execute a tool with a hard timeout."""
    try:
        result = await asyncio.wait_for(
            tool_func(**tool_args),
            timeout=timeout_seconds
        )
        return {"success": True, "result": result}
    
    except asyncio.TimeoutError:
        return {
            "success": False,
            "error_type": "timeout",
            "message": f"Tool did not respond within {timeout_seconds}s. "
                       f"The service may be slow or unavailable.",
            "retry_suggestion": "Try again in 30 seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "error_type": "execution_error",
            "message": str(e)
        }

# Tool-specific timeout recommendations:
TOOL_TIMEOUTS = {
    "get_weather": 5.0,          # Simple API call — 5s is generous
    "search_web": 10.0,          # Web search can be slow
    "run_code": 30.0,            # Code execution — may take time
    "send_email": 8.0,           # Email APIs — occasionally slow
    "database_query": 15.0,      # DB queries — allow for complex queries
    "llm_subagent": 60.0,        # Sub-agent call — allow for full reasoning
}
```

**Timeout value guidelines:**

| Tool type | Recommended timeout | Why |
|---|---|---|
| Simple REST API | 3–5s | Should respond in <1s; 5s allows for slow networks |
| Search / retrieval | 8–12s | Index queries can be slow under load |
| LLM sub-agent | 30–60s | Full reasoning loop takes time |
| File processing | 20–30s | Depends on file size |
| Database query | 10–20s | Complex joins can run long |
| P95 user-acceptable wait | 15s | Beyond this, most users assume something is broken |

**Circuit breaker pattern (advanced):** If a tool fails >5 times in 60 seconds, mark it as "open circuit" and stop calling it for 30 seconds. This prevents an unhealthy dependency from slowing your entire agent:

```python
from datetime import datetime, timedelta

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = timedelta(seconds=recovery_timeout)
        self.last_failure_time = None
        self.state = "closed"  # closed=normal, open=blocking calls
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def record_success(self):
        self.failure_count = 0
        self.state = "closed"
    
    def is_open(self) -> bool:
        if self.state == "open":
            if datetime.now() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"  # try one request to test recovery
                return False
            return True
        return False
```

---

### Key Numbers to Memorize

| Metric | Value |
|---|---|
| Self-correction success rate on structured error feedback | 70–80% on first retry |
| Recommended max retries for transient failures | 3 |
| Exponential backoff base delay | 1.0s |
| Thundering herd reduction with full jitter | ~50% peak load reduction |
| Max tool errors before agent fallback | 3 |
| Simple REST API timeout | 3–5s |
| P95 user-acceptable wait | 15s |
| Circuit breaker failure threshold (recommended) | 5 failures in 60s |
| Circuit breaker recovery timeout | 30s |
| Rate limit retry (429) standard header | `Retry-After` seconds |