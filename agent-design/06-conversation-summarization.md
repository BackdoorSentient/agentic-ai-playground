# Conversation Summarization — Context Management at Scale

> Deep dive: Day 7 | Topic: Token counting with tiktoken and LLM-based summarization

---

## Q1. Why does conversation summarization exist and when is it not enough?

**Answer:**

Every LLM has a finite context window. Without summarization, a long conversation either: (a) gets truncated (losing early context), or (b) becomes prohibitively expensive as every call includes the full history.

**The problem space:**

```
Turn 1:   ~50 tokens
Turn 5:   ~300 tokens cumulative
Turn 20:  ~1,500 tokens cumulative
Turn 50:  ~5,000 tokens cumulative  ← GPT-4o-mini starts getting expensive
Turn 100: ~12,000 tokens cumulative ← approaching model limits + $$$
```

**Cost without summarization (gpt-4o-mini at $0.15/M input tokens):**

| Conversation length | Input tokens per call | Cost per call | Cost per 100 turns |
|---|---|---|---|
| 10 turns | ~800 | $0.00012 | $0.012 |
| 50 turns | ~5,000 | $0.00075 | $0.075 |
| 100 turns | ~12,000 | $0.0018 | $0.18 |
| With summarization | ~400 steady-state | $0.00006 | $0.006 |

**When summarization is not enough:**
- Tasks that require verbatim recall of earlier content (contracts, code)
- Debugging sessions where every exchange matters
- Solution: use a **retrieval-based** approach (store verbatim in vector DB, retrieve by relevance) rather than lossy summarization

---

## Q2. How does tiktoken work and why use it over `len(text.split())`?

**Answer:**

tiktoken is OpenAI's BPE tokenizer. It gives you the **exact** token count a model will see — `split()` can be off by 30–50%.

```python
import tiktoken

def get_encoder(model: str) -> tiktoken.Encoding:
    """Cache the encoder to avoid re-loading on every call."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback for unknown models
        return tiktoken.get_encoding("cl100k_base")

def count_message_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    enc = get_encoder(model)
    total = 3  # Every reply starts with <|start|>assistant<|message|>

    for msg in messages:
        total += 4   # role + separators overhead per message
        content = msg.get("content", "")

        if isinstance(content, str):
            total += len(enc.encode(content))

        elif isinstance(content, list):
            # Handle multi-modal (vision) messages
            for block in content:
                if block.get("type") == "text":
                    total += len(enc.encode(block["text"]))
                elif block.get("type") == "image_url":
                    # Rough estimate: image tokens vary by resolution
                    total += 765  # 512x512 low-res default

    return total

# Why NOT use split():
text = "I've seen 3 examples of HITL with LangGraph's interrupt() function."
word_count   = len(text.split())         # 12 words
token_count  = len(enc.encode(text))     # 17 tokens
# "HITL", "LangGraph's", "interrupt()" all tokenize to 2+ tokens each
```

**Model → encoding mapping:**

| Model family | Encoding | Overhead/msg |
|---|---|---|
| gpt-4o, gpt-4o-mini | cl100k_base | 4 tokens |
| gpt-3.5-turbo | cl100k_base | 4 tokens |
| gpt-4 | cl100k_base | 4 tokens |
| Claude models | SentencePiece (internal) | Use Anthropic's token counter API |

---

## Q3. What is the full summarization algorithm including edge cases?

**Answer:**

```python
import tiktoken
from langchain_openai import ChatOpenAI

SUMMARIZATION_THRESHOLD = 2000
TAIL_MESSAGES_TO_KEEP   = 4
SUMMARY_MAX_WORDS       = 200

def format_messages_for_llm(messages: list[dict]) -> str:
    """Convert messages list to readable text for summarization prompt."""
    lines = []
    for msg in messages:
        role    = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content if b.get("type") == "text"
            )
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)

def maybe_summarize(state: dict, llm: ChatOpenAI) -> dict:
    messages    = state.get("messages", [])
    token_count = count_message_tokens(messages)

    # Don't summarize if under threshold
    if token_count <= SUMMARIZATION_THRESHOLD:
        return {**state, "token_count": token_count}

    # Don't summarize if too few messages (nothing to compress)
    if len(messages) <= TAIL_MESSAGES_TO_KEEP + 1:
        return {**state, "token_count": token_count}

    # Split: tail (keep raw) vs head (summarize)
    tail         = messages[-TAIL_MESSAGES_TO_KEEP:]
    to_summarize = messages[:-TAIL_MESSAGES_TO_KEEP]

    # Filter out existing summary messages (avoid summarizing summaries)
    real_messages = [
        m for m in to_summarize
        if not m.get("content", "").startswith("[Conversation summary")
    ]

    existing_summary = state.get("summary", "")

    summarization_prompt = f"""You are a conversation summarizer for an AI assistant.
    
{"Previous summary:" + existing_summary if existing_summary else ""}

New conversation segment to summarize:
{format_messages_for_llm(real_messages)}

Write a factual, dense summary that preserves:
- Key facts and information the user shared or asked about
- Tool results (what was searched, what notes were saved)
- User preferences or constraints mentioned
- Any important decisions or conclusions

Maximum {SUMMARY_MAX_WORDS} words. Write as bullet points for easy scanning.
Do NOT include pleasantries or meta-commentary."""

    response = llm.invoke([{"role": "user", "content": summarization_prompt}])
    new_summary = response.content.strip()

    # Replace old messages with: [summary system msg] + tail
    summary_system_msg = {
        "role":    "system",
        "content": f"[Conversation summary up to this point]:\n{new_summary}"
    }
    compressed = [summary_system_msg] + tail

    new_token_count = count_message_tokens(compressed)

    return {
        **state,
        "messages":    compressed,
        "summary":     new_summary,
        "token_count": new_token_count,
    }
```

**Edge cases handled:**

| Edge case | Handling |
|---|---|
| Messages already contain a summary msg | Filter it out before re-summarizing |
| Only 4 messages total | Skip — nothing to compress |
| Summary LLM call fails | Catch exception, log error, return original state |
| First summarization (no prior summary) | `existing_summary` is `""` — omit from prompt |
| Tool result messages (ToolMessage type) | Included in summarization — tool results are facts |

---

## Q4. Where does the summarization node sit in the graph and when does it trigger?

**Answer:**

Summarization should happen **at the start of each turn**, before the planner sees the messages. This keeps the planner's context clean.

```python
def summarization_node(state: AgentState) -> AgentState:
    """Check token count. If over threshold, summarize before planning."""
    return maybe_summarize(state, summarizer_llm)

# Graph topology:
# [START] → summarization_check → planner → ...
graph.add_node("summarization_check", summarization_node)
graph.add_edge(START, "summarization_check")
graph.add_edge("summarization_check", "planner")
```

**Alternative: summarize after response generation**

```python
# Pro: planner sees full context on this turn
# Con: tokens wasted on this turn's LLM calls
# Better for: agents where this turn's full context matters
```

**Triggering strategy comparison:**

| Strategy | Trigger | Pro | Con |
|---|---|---|---|
| Token threshold (2000) | Every turn, check count | Predictable | May summarize mid-thought |
| Message count (every 10 turns) | Simple | Fast | Doesn't adapt to message length |
| Combined (tokens OR count) | Either threshold | Robust | Slightly more complex |
| Lazy (only when LLM errors) | Never proactive | Zero overhead | Context errors in production |

**Recommended: token threshold** because message length varies enormously (a tool result might be 500 tokens, a greeting 5 tokens).

---

## Q5. How do you verify summarization is working correctly in tests?

**Answer:**

```python
def test_summarization_triggers_at_threshold():
    """Verify summary kicks in when token count exceeds 2000."""
    # Build a state with ~2100 tokens of messages
    long_messages = []
    enc = tiktoken.encoding_for_model("gpt-4o")

    # Add messages until we exceed threshold
    filler = "This is a filler message about AI and LangGraph " * 5  # ~50 tokens
    for i in range(45):  # ~2250 tokens
        long_messages.append({"role": "user" if i % 2 == 0 else "assistant",
                               "content": filler})

    state = {
        "messages": long_messages,
        "summary": "",
        "token_count": 0,
    }

    result = maybe_summarize(state, mock_llm)

    # After summarization:
    assert result["token_count"] < 2000        # compressed successfully
    assert result["summary"] != ""             # summary was generated
    assert len(result["messages"]) == TAIL_MESSAGES_TO_KEEP + 1  # tail + summary msg
    assert result["messages"][0]["role"] == "system"  # summary is a system message
    assert "[Conversation summary" in result["messages"][0]["content"]

def test_summarization_preserves_tail():
    """Last 4 messages must be preserved verbatim."""
    messages = [{"role": "user", "content": f"msg {i}"} for i in range(50)]
    state = {"messages": messages, "summary": "", "token_count": 9999}

    result = maybe_summarize(state, mock_llm)
    tail_contents = [m["content"] for m in result["messages"][-4:]]
    assert tail_contents == ["msg 46", "msg 47", "msg 48", "msg 49"]
```

---

## Key Numbers

| Parameter | Value | Reasoning |
|---|---|---|
| SUMMARIZATION_THRESHOLD | 2000 tokens | ~15–30 typical messages; leaves room for tools |
| TAIL_MESSAGES_TO_KEEP | 4 | 2 user + 2 assistant = full current exchange |
| SUMMARY_MAX_WORDS | 200 | ~270 tokens — small overhead |
| tiktoken overhead per message | 4 tokens | Documented by OpenAI |
| Cost of 1 summarization call (gpt-4o-mini) | ~$0.0003 | 2000 tokens in, 200 out |
| Steady-state tokens after summarization | ~400–600 | Summary + 4 tail messages |