# 03 — Tools, ReAct Prompting & Memory Implementation

---

## Q1. What makes a tool schema reliable vs. unreliable for LLM selection?

**A:** The LLM selects a tool based entirely on the schema description — it never reads your implementation code. A vague description leads to wrong tool selection; a precise one leads to correct selection even on edge cases.

**The four elements of a reliable tool schema:**

| Element | What to include | Bad example | Good example |
|---|---|---|---|
| When to use | Describe the trigger condition | "searches things" | "Use when the user asks about facts, news, or any information requiring current data" |
| Input format | Exact type and format | "a query" | "a search query string — phrase it as you would type into Google" |
| Output format | What the caller receives back | "results" | "returns a text summary of top search results, max 300 words" |
| When NOT to use | Negative constraints | (missing) | "Do not use for calendar questions — use calendar_lookup instead" |

**Schema comparison:**

```python
# ❌ BAD — vague, LLM will misuse this
@tool
def search(q):
    """Search for stuff."""
    ...

# ✅ GOOD — precise, LLM will use this correctly
@tool
def web_search(query: str) -> str:
    """Search the web for current information on any topic.

    Use this tool when:
    - The user asks about facts, news, how-to guides, or documentation
    - You need information that may have changed recently
    - The user's question requires external knowledge you don't have

    Do NOT use this for:
    - Calendar or scheduling questions (use calendar_lookup)
    - Saving information (use save_note)

    Input: a search query string, phrased as a natural language question
           or keyword set. Example: "Python asyncio tutorial 2024"
    Output: a text summary of the most relevant search results
    """
```

---

## Q2. How do you implement web search with both mock and real backends?

**A:** Use a strategy pattern so you can swap mock for real without changing the tool interface.

```python
# tools/web_search.py
import os
from langchain_core.tools import tool

# ── Mock backend ────────────────────────────────────────────────
MOCK_DB = {
    "python async": "Python async/await enables non-blocking I/O using the asyncio library. Key functions: asyncio.run(), async def, await.",
    "langchain": "LangChain is a framework for building LLM applications. Core concepts: chains, agents, tools, memory, and prompts.",
    "langraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs using graph-based orchestration.",
    "chromadb": "ChromaDB is an open-source embedding database. Supports persistent storage, cosine/L2/IP similarity, and metadata filtering.",
    "openai pricing": "GPT-4o: $2.50/1M input, $10/1M output. GPT-4o-mini: $0.15/1M input, $0.60/1M output.",
}

def _mock_search(query: str) -> str:
    q = query.lower()
    for key, result in MOCK_DB.items():
        if key in q:
            return f"[Mock] Search result for '{query}':\n{result}"
    return f"[Mock] No specific results for '{query}'. A real search would return relevant web pages."

# ── Real backend (Tavily) ───────────────────────────────────────
def _tavily_search(query: str) -> str:
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        response = client.search(query=query, max_results=3)
        results = response.get("results", [])
        if not results:
            return f"No results found for '{query}'."
        lines = [f"Search results for '{query}':"]
        for r in results:
            lines.append(f"• {r['title']}: {r['content'][:200]}...")
        return "\n".join(lines)
    except Exception as e:
        return f"Search failed: {str(e)}. Falling back to general knowledge."

# ── Tool function (backend selected by env var) ─────────────────
USE_REAL_SEARCH = os.getenv("USE_REAL_SEARCH", "false").lower() == "true"

@tool
def web_search(query: str) -> str:
    """Search the web for current information on any topic.
    Use when the user asks about facts, documentation, news, or how-to guides.
    Input: a natural language search query string.
    Output: text summary of search results."""
    if USE_REAL_SEARCH:
        return _tavily_search(query)
    return _mock_search(query)
```

**Why env-var backend switching:** You develop and test with the mock (free, fast, deterministic), flip `USE_REAL_SEARCH=true` for demos and production. No code changes needed.

---

## Q3. How do you implement the ReAct pattern in a LangGraph agent?

**A:** In LangGraph, the ReAct loop is the graph itself — not a prompt trick. The Thought step is the LLM call node, the Action step is the tool execution node, and the Observation step is the tool result being injected back into messages. The loop is the `web_search → llm_call` edge.

However, you still want the LLM to verbalize its reasoning in the system prompt for two reasons: better reasoning quality, and interpretable logs.

```python
# prompts/system_prompt.py

SYSTEM_PROMPT_TEMPLATE = """\
You are a Personal Research Assistant. Help with research, note-taking, and scheduling.

## Available Tools
- web_search(query): search the web for current information
- save_note(content, title): save important information (requires human approval)
- calendar_lookup(time_period): check the user's calendar (today/tomorrow/this week)

## How to Reason

Before responding, always reason through the following steps internally:

Thought: What does the user actually need? Do I need a tool?
Action: Which tool (if any)? What exact arguments?
Observation: [tool result will appear here automatically]
Thought: What does the observation tell me? Do I need another tool?
Final Answer: Respond directly and concisely to the user.

## Rules
1. If the user asks about facts you might not know → use web_search
2. If the user asks to "save", "note", or "remember" something → use save_note
3. If the user asks about their schedule, meetings, or events → use calendar_lookup
4. Never hallucinate tool results — always actually call the tool
5. Be concise in Final Answer — the user doesn't need to see your reasoning steps

## What I Know About This User
{retrieved_facts}

## Today's Date
{current_date}
"""

def build_system_prompt(retrieved_facts: list[str]) -> str:
    from datetime import datetime
    facts = "\n".join(f"• {f}" for f in retrieved_facts) if retrieved_facts else "Nothing yet — this may be a new user."
    return SYSTEM_PROMPT_TEMPLATE.format(
        retrieved_facts=facts,
        current_date=datetime.now().strftime("%A, %B %d, %Y at %H:%M")
    )
```

**The critical difference from naive ReAct:** In the original ReAct paper, the model generates "Action: tool_name[args]" as plain text and a parser extracts it. In LangGraph with modern LLMs, the model generates a structured `tool_call` object. The prompt still uses the Thought/Action/Observation framing to guide reasoning quality, but the Action is a native tool call, not parsed text.

---

## Q4. How do you implement short-term memory with context window management?

**A:** Short-term memory is simply the `messages` list in state. The challenge is that it grows with every turn and will eventually exceed the model's context window (128K for GPT-4o), causing the oldest messages to be truncated or the call to fail.

The solution: count tokens after every turn, and trigger summarization when approaching the limit.

```python
# memory/short_term.py
import tiktoken
from openai import OpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

client = OpenAI()
encoder = tiktoken.encoding_for_model("gpt-4o")

TOKEN_LIMIT = 2000        # Trigger summarization above this
KEEP_LAST_N = 4           # Always keep the last N messages verbatim

def count_tokens(messages: list) -> int:
    total = 0
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        total += len(encoder.encode(content))
    return total

def summarize_messages(messages: list) -> str:
    """Call LLM to compress message history into a summary."""
    history_text = "\n".join([
        f"{type(m).__name__}: {m.content}"
        for m in messages
        if hasattr(m, "content")
    ])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Summarize this conversation history concisely.
Focus on: key facts stated, decisions made, topics discussed, user preferences revealed.
Keep it under 200 words. Write in third person ("The user mentioned...").

Conversation:
{history_text}"""
        }]
    )
    return response.choices[0].message.content

def maybe_compress_history(state: dict) -> dict:
    """
    Check if messages exceed token limit.
    If so, summarize all but the last KEEP_LAST_N messages.
    """
    messages = state["messages"]

    if count_tokens(messages) <= TOKEN_LIMIT:
        return {}  # No change needed

    # Keep the last N messages verbatim
    to_summarize = messages[:-KEEP_LAST_N]
    to_keep = messages[-KEEP_LAST_N:]

    if not to_summarize:
        return {}  # Nothing old enough to summarize

    summary = summarize_messages(to_summarize)

    # Replace old messages with a system summary message
    summary_msg = SystemMessage(content=f"[Conversation summary: {summary}]")
    compressed_messages = [summary_msg] + to_keep

    return {
        "messages": compressed_messages,
        "conversation_summary": summary
    }
```

**The compression node in the graph:**

```python
def compress_memory_node(state: AgentState) -> dict:
    return maybe_compress_history(state)

# Add to graph before llm_call
workflow.add_node("compress_memory", compress_memory_node)
workflow.add_edge("retrieve_memory", "compress_memory")
workflow.add_edge("compress_memory", "llm_call")
```

---

## Q5. How do you implement the full long-term memory read/write cycle?

**A:**

**Write (end of turn — extract and upsert):**

```python
# memory/long_term.py
import chromadb
import hashlib
import json
from openai import OpenAI

client = OpenAI()
chroma = chromadb.PersistentClient(path="data/chroma_db")

def _embed(text: str) -> list[float]:
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

def _get_collection(user_id: str):
    return chroma.get_or_create_collection(
        name=f"user_{user_id.replace('-', '_')}",
        metadata={"hnsw:space": "cosine"}
    )

EXTRACTION_PROMPT = """\
Extract durable facts about the user from this message.
Only extract: name, location, job, skills, preferences, goals, habits.
Skip: questions, temporary states, general knowledge.
Return JSON array only, no other text:
[{"text": "The user is a senior Python developer at a fintech startup."}]
If nothing to extract: []

Message: {message}"""

def extract_and_store(user_id: str, message: str):
    """Extract facts from message and upsert to ChromaDB."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(message=message)}]
    )

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
        facts = data if isinstance(data, list) else data.get("facts", [])
    except (json.JSONDecodeError, AttributeError):
        return

    if not facts:
        return

    collection = _get_collection(user_id)
    for fact in facts:
        text = fact.get("text", "")
        if not text:
            continue
        fact_id = hashlib.md5(text.encode()).hexdigest()[:16]
        collection.upsert(
            ids=[fact_id],
            documents=[text],
            embeddings=[_embed(text)]
        )

def retrieve_facts(user_id: str, query: str, top_k: int = 3) -> list[str]:
    """Retrieve top-k relevant facts for the current query."""
    collection = _get_collection(user_id)
    count = collection.count()

    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[_embed(query)],
        n_results=min(top_k, count)
    )
    return results["documents"][0] if results["documents"] else []
```

**The memory nodes in the graph:**

```python
def retrieve_memory_node(state: AgentState) -> dict:
    query = state["messages"][-1].content
    facts = retrieve_facts(state["user_id"], query)
    return {"retrieved_facts": facts}

def write_memory_node(state: AgentState) -> dict:
    # Extract facts from the user's message this turn
    user_messages = [m for m in state["messages"] if hasattr(m, "type") and m.type == "human"]
    if user_messages:
        latest = user_messages[-1].content
        extract_and_store(state["user_id"], latest)
    return {}  # No state change needed
```

---

## Key Numbers

| Parameter | Value |
|---|---|
| Token limit before summarization | 2000 tokens |
| Messages kept verbatim after summarization | Last 4 |
| Embedding model | text-embedding-3-small |
| Embedding dimensions | 1536 |
| Facts retrieved per turn | Top-3 |
| Fact extraction model | gpt-4o-mini (cheaper) |
| ChromaDB similarity | cosine |
| Fact ID generation | MD5 hash of fact text, first 16 chars |