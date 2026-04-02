## 2. Tool Calling Mechanics — The Full Request/Response Cycle

### Q1: What is the complete request/response cycle for a tool call, and what happens at each step?

**Answer:**

Tool calling is a multi-turn protocol. A single "tool call" is actually a minimum of two API round trips. Understanding every step is essential for debugging failures and designing correct retry logic.

**The 4-step cycle:**

```
Step 1: USER → LLM
  User sends a message. You include tool schemas in the request.
  LLM decides to call a tool → returns a tool_use block (NOT a text response).

Step 2: YOUR CODE executes the tool
  Parse the tool_use block → extract tool name + arguments.
  Call your actual function with those arguments.
  Capture result (or error).

Step 3: YOUR CODE → LLM (second request)
  Send the full conversation history + the tool result as a tool_result message.
  LLM reads the result → continues reasoning.

Step 4: LLM → USER
  LLM generates the final text response using the tool result.
  (May call another tool → repeat from Step 1. This is the agentic loop.)
```

**Full implementation — Claude:**
```python
import anthropic
import json

client = anthropic.Anthropic()

tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city. Returns temperature in Celsius, humidity, and conditions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Mumbai' or 'New York'"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
        },
        "required": ["city"]
    }
}]

def get_weather(city: str, units: str = "celsius") -> dict:
    """Your actual tool implementation."""
    # In production: call a real weather API
    return {"city": city, "temp": 32, "humidity": 78, "condition": "Partly cloudy"}

def run_agent(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        # Step 1: Call LLM
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        # Step 4: No tool call — model is done
        if response.stop_reason == "end_turn":
            return next(b.text for b in response.content if b.type == "text")
        
        # Step 2: Tool call detected
        if response.stop_reason == "tool_use":
            # Add assistant's response (with tool_use block) to history
            messages.append({"role": "assistant", "content": response.content})
            
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # Execute the tool
                    if block.name == "get_weather":
                        result = get_weather(**block.input)
                    else:
                        result = {"error": f"Unknown tool: {block.name}"}
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,   # CRITICAL: must match the tool_use id
                        "content": json.dumps(result)
                    })
            
            # Step 3: Return tool results to LLM
            messages.append({"role": "user", "content": tool_results})
            # Loop continues → LLM reads result → may call another tool or finish
```

**Full implementation — OpenAI:**
```python
from openai import OpenAI
import json

client = OpenAI()

def run_agent_openai(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            tools=tools,  # OpenAI format tools
            messages=messages
        )
        
        choice = response.choices[0]
        
        # Step 4: Done
        if choice.finish_reason == "stop":
            return choice.message.content
        
        # Step 2: Tool call
        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)  # add assistant message with tool_calls
            
            for tool_call in choice.message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = get_weather(**args)  # execute
                
                # Step 3: Return result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,  # must match
                    "content": json.dumps(result)
                })
```

---

### Q2: What is `stop_reason` / `finish_reason`, and what are all the values you must handle?

**Answer:**

The stop reason tells you *why* the model stopped generating. You must handle all possible values — unhandled stop reasons cause silent infinite loops or dropped tool calls.

**Claude stop reasons:**

| `stop_reason` | Meaning | Your action |
|---|---|---|
| `"end_turn"` | Model finished its response normally | Extract text, return to user |
| `"tool_use"` | Model wants to call one or more tools | Execute tools, return results, continue loop |
| `"max_tokens"` | Hit `max_tokens` limit mid-response | Increase `max_tokens`, or handle partial response |
| `"stop_sequence"` | Hit a custom stop sequence | Extract text up to that point |

**OpenAI finish reasons:**

| `finish_reason` | Meaning | Your action |
|---|---|---|
| `"stop"` | Normal completion | Return response |
| `"tool_calls"` | Tool call requested | Execute tools, continue |
| `"length"` | Hit `max_tokens` | Increase limit or handle truncation |
| `"content_filter"` | Content policy triggered | Log, return safe fallback message |
| `"function_call"` | Legacy (pre-`tool_calls`) | Same as `tool_calls` |

**Production-safe stop reason handler:**
```python
def handle_response(response, messages: list) -> tuple[bool, list]:
    """Returns (is_done: bool, updated_messages: list)"""
    
    stop = response.stop_reason  # Claude
    # stop = response.choices[0].finish_reason  # OpenAI
    
    if stop == "end_turn":
        return True, messages
    
    elif stop == "tool_use":
        # handle tool calls (see Q1)
        return False, messages  # continue loop
    
    elif stop == "max_tokens":
        # Handle partial response — either increase limit or return what we have
        raise ValueError(f"Response truncated at max_tokens. Increase max_tokens or chunk the task.")
    
    else:
        # Unexpected stop reason — fail loudly, don't silently continue
        raise RuntimeError(f"Unhandled stop_reason: {stop}. Full response: {response}")
```

---

### Q3: How does parallel tool calling work, and when does the model call multiple tools in one turn?

**Answer:**

Modern models (GPT-4o, Claude 3+ series) can request multiple tool calls in a single response when they determine the calls are independent of each other. This is called **parallel tool calling** and dramatically reduces latency for multi-tool tasks.

**Example:** "What's the weather in Mumbai and London right now?"

Without parallel tool calling: 2 sequential LLM round trips → 2× latency.
With parallel tool calling: 1 LLM turn requests both → you execute in parallel → 1 follow-up turn.

**How to detect and handle multiple tool calls:**
```python
import asyncio

async def execute_tool(tool_name: str, tool_args: dict) -> dict:
    """Execute a single tool asynchronously."""
    if tool_name == "get_weather":
        return get_weather(**tool_args)
    elif tool_name == "calculate":
        return calculate(**tool_args)
    else:
        return {"error": f"Unknown tool: {tool_name}"}

async def handle_tool_calls_parallel(response_content: list) -> list:
    """Execute all tool calls in parallel."""
    tool_use_blocks = [b for b in response_content if b.type == "tool_use"]
    
    # Launch all tools concurrently
    tasks = [
        execute_tool(block.name, block.input)
        for block in tool_use_blocks
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    tool_results = []
    for block, result in zip(tool_use_blocks, results):
        if isinstance(result, Exception):
            result = {"error": str(result)}
        
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": json.dumps(result)
        })
    
    return tool_results
```

**Latency impact of parallel tool calling:**

| Scenario | Sequential | Parallel | Savings |
|---|---|---|---|
| 2 independent tools (500ms each) | ~1000ms | ~500ms + overhead | ~45% |
| 3 independent tools (500ms each) | ~1500ms | ~500ms + overhead | ~63% |
| 3 tools, first feeds second | ~1500ms | ~1500ms (must be sequential) | 0% |

**When to force sequential:** If tool B depends on the output of tool A, you cannot parallelize. Design tool descriptions to help the model understand dependencies: "Call get_customer_id first, then use the returned ID to call get_order_history."

---

### Q4: How does tool calling work inside LangGraph's agentic loop (the ToolNode pattern)?

**Answer:**

LangGraph provides a pre-built `ToolNode` that handles the entire tool execution step as a graph node, plugging directly into the ReAct agent pattern:

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
import operator

# 1. Define tools using @tool decorator
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Returns temperature and conditions."""
    return f"{city}: 32°C, Partly cloudy, Humidity 78%"

@tool  
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '2 + 2 * 10'"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

@tool
def search_web(query: str) -> str:
    """Search the web for current information about a topic."""
    return f"[Mock search results for: {query}]"

tools = [get_weather, calculate, search_web]

# 2. Bind tools to LLM
llm = ChatAnthropic(model="claude-sonnet-4-5")
llm_with_tools = llm.bind_tools(tools)

# 3. Define state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 4. Define agent node (calls LLM)
def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 5. Build graph with ToolNode
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))   # ToolNode handles execution + error catching

graph.set_entry_point("agent")

# tools_condition: returns "tools" if last message has tool calls, else END
graph.add_conditional_edges("agent", tools_condition, {
    "tools": "tools",
    END: END
})
graph.add_edge("tools", "agent")  # after tools, go back to agent

app = graph.compile()

# 6. Run
from langchain_core.messages import HumanMessage
result = app.invoke({"messages": [HumanMessage(content="What's the weather in Mumbai and 15 * 23?")]})
```

**What `ToolNode` handles automatically:**
- Parsing tool call blocks from the LLM response
- Executing the correct tool function by name
- Catching tool exceptions and returning them as error tool results (not crashing the graph)
- Formatting results back into the correct message format for the next LLM call

---

### Q5: What is the `tool_use_id` and why does mismatching it cause silent failures?

**Answer:**

The `tool_use_id` is a unique identifier generated by the model for each tool call it requests. When you return tool results, you must reference the exact same `tool_use_id` — this is how the model correlates "I asked for tool X with these args" with "here is the result."

**What happens on mismatch:**
- Claude: Returns an API error: `"tool_result tool_use_id does not match any pending tool use"`
- OpenAI: Returns an API error: `"Invalid parameter: tool_call_id does not match any tool call"`
- If you swallow these errors: The model receives no tool result → it may hallucinate a result or loop infinitely

**Common mismatch scenarios:**

```python
# BUG 1: Using a hardcoded ID instead of the one from the response
tool_results.append({
    "tool_use_id": "my-custom-id",  # WRONG — model doesn't recognize this
    "content": "..."
})

# BUG 2: Parallel tools — returning results in wrong order without ID tracking
# If model requests tools [A, B, C], you execute and get results, 
# but return them as [C_result, A_result, B_result] matched to [A_id, B_id, C_id]
# → C_result is attributed to A, etc.

# CORRECT: Always use block.id from the response
for block in response.content:
    if block.type == "tool_use":
        result = execute_tool(block.name, block.input)
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,  # ← exact ID from the model's response
            "content": json.dumps(result)
        })
```

---

### Key Numbers to Memorize

| Metric | Value |
|---|---|
| Minimum API round trips per tool call | 2 |
| Parallel tool latency savings (3 independent tools) | ~63% |
| Claude stop reason for tool call | `"tool_use"` |
| OpenAI finish reason for tool call | `"tool_calls"` |
| LangGraph node for tool execution | `ToolNode` |
| LangGraph routing function | `tools_condition` |
| Tool result content type | Must be a JSON string or text string |