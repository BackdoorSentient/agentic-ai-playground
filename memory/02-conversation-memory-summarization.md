## 2. Conversation Memory with Summarization — Managing Context Overflow via Compression

### Q1: What is the context overflow problem and what are the three main strategies to handle it?

**Answer:**

When conversation history grows beyond the model's context window, you cannot simply truncate — you lose critical context. The three main strategies are:

**1. Sliding Window (Truncation)**
Keep only the last N tokens. Simple to implement, zero latency overhead. Fatal flaw: critical information from early in the conversation (e.g., the user's stated goal, constraints, or preferences) is silently dropped. Use only for stateless tasks where early context is irrelevant.

**2. Summarization (Compression)**
When history exceeds a threshold, run an LLM call to compress the oldest turns into a dense summary, then replace those turns with the summary. Preserves semantic content at 5–10× compression ratio. Adds 200–800ms latency for the summarization call.

**3. Retrieval-Augmented Memory (RAG over history)**
Store all conversation turns in a vector DB. On each new turn, retrieve the top-K most relevant past turns and inject them. Scales to infinite history. Adds ~100–300ms retrieval latency. Requires a vector DB infrastructure dependency.

**Hybrid approach (production standard):** Summarize recent history into a rolling summary + retrieve specific past facts when relevant. This is what ChatGPT's memory system does.

---

### Q2: How do you implement a conversation summarizer, and what makes a good vs. bad summarization prompt?

**Answer:**

**Implementation pattern:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

SUMMARIZE_PROMPT = """
You are a conversation summarizer. Compress the following conversation history 
into a dense summary that preserves:
1. All user preferences and constraints stated explicitly
2. Key decisions made
3. Facts established (names, dates, quantities)
4. The current task state

Discard: pleasantries, failed attempts that were corrected, redundant clarifications.
Output a single paragraph of 100-150 words maximum.

Conversation to summarize:
{conversation}
"""

def should_summarize(messages: list, threshold: int = 2000) -> bool:
    """Trigger summarization when history exceeds token threshold."""
    total_tokens = sum(len(m.content.split()) * 1.3 for m in messages)  # rough estimate
    return total_tokens > threshold

def summarize_history(messages: list, llm: ChatOpenAI) -> str:
    conversation_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in messages
    )
    response = llm.invoke([
        HumanMessage(content=SUMMARIZE_PROMPT.format(conversation=conversation_text))
    ])
    return response.content

def manage_context(messages: list, llm: ChatOpenAI, threshold: int = 2000) -> list:
    if not should_summarize(messages, threshold):
        return messages
    
    # Keep last 2 turns verbatim (for immediate context continuity)
    recent = messages[-4:]  # last 2 user+assistant pairs
    older = messages[:-4]
    
    summary_text = summarize_history(older, llm)
    
    return [
        SystemMessage(content=f"[CONVERSATION SUMMARY]: {summary_text}"),
        *recent
    ]
```

**Good summarization prompt characteristics:**
- Explicit list of what to preserve (preferences, decisions, facts, task state)
- Explicit list of what to discard (pleasantries, corrected errors)
- Hard output length constraint (prevents summary bloat)
- Instructs the model to be dense, not conversational

**Bad summarization prompt:** "Summarize this conversation." This produces narrative summaries that retain verbose phrasing, miss key facts, and can be 3–5× longer than necessary.

---

### Q3: What is the token threshold for triggering summarization, and how do you count tokens accurately?

**Answer:**

**Accurate token counting** requires using the same tokenizer as the model:

```python
import tiktoken

def count_tokens(messages: list, model: str = "gpt-4o") -> int:
    """Count tokens the way OpenAI's API counts them."""
    enc = tiktoken.encoding_for_model(model)
    
    tokens = 0
    for message in messages:
        # OpenAI adds 4 tokens per message for role/formatting overhead
        tokens += 4
        tokens += len(enc.encode(message.get("content", "")))
        tokens += len(enc.encode(message.get("role", "")))
    
    tokens += 2  # conversation-level overhead
    return tokens
```

**For Claude models**, use Anthropic's `count_tokens` API call or the `anthropic` Python library's token counter.

**Word-based estimation (quick and dirty):** `tokens ≈ words × 1.3` for English. This is ±15% accurate — acceptable for triggering thresholds but never for billing calculations.

**Recommended thresholds:**

| Context window | Summarization trigger | Why |
|---|---|---|
| 128K (GPT-4o) | 80K tokens | Leave 48K for model's reasoning + response |
| 200K (Claude 3.7) | 120K tokens | Same 40% headroom principle |
| 8K (GPT-3.5 legacy) | 4K tokens | More aggressive — smaller window |
| **Practical chatbot** | **2,000 tokens** | Most conversations need far less than window limit |

**The 2,000-token threshold for practical chatbots:** In most consumer chatbot use cases, a 2,000-token history (roughly 10–15 conversational turns) is sufficient working memory. Summarizing beyond this keeps costs low and prevents the Lost-in-the-Middle problem.

---

### Q4: What are the failure modes of conversation summarization, and how do you mitigate them?

**Answer:**

**1. Information loss (silent hallucination into summaries)**
The summarizer model occasionally introduces small fabrications — changing a number, misattributing a preference. Mitigation: after summarization, run an LLM grading call ("Does this summary contradict anything in the original?") for high-stakes applications. Cost: one extra API call per summarization event.

**2. Summary bloat (summaries that grow toward the threshold)**
Without a hard token limit in the prompt, summaries can approach the original length. Then you summarize again, and the summary of the summary loses more. Mitigation: enforce `max_tokens` on the summarization call, and set an explicit word count constraint in the prompt.

**3. Loss of conversational register**
Summaries flatten nuance — "the user seemed frustrated" becomes "the user asked about refunds." Mitigation: preserve emotional signals if relevant: add "Also note any emotional cues or user frustration signals" to the summarization prompt.

**4. Summarization latency in real-time chat**
Adding 300–800ms for a summarization LLM call is noticeable in real-time conversations. Mitigation: trigger summarization asynchronously after the response is delivered, update the history in the background, apply it to the *next* turn. The user sees no latency increase.

**5. Context boundary artifacts**
When a conversation crosses a summarization boundary mid-topic, the summary cuts off the topic and the next turn continues it without full context. Mitigation: detect topic-in-progress heuristically (e.g., if the last user turn contains a question) and delay summarization until the topic closes.

---

### Q5: How does LangChain's built-in memory handle summarization, and what are its limitations?

**Answer:**

LangChain provides `ConversationSummaryMemory` and `ConversationSummaryBufferMemory`:

```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")  # use cheap model for summarization

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=2000,        # summarize when history exceeds this
    return_messages=True,        # return as message objects, not string
    memory_key="chat_history"    # key injected into chain prompt
)
```

**`ConversationSummaryBufferMemory`** (the production-preferred variant):
- Keeps recent turns verbatim (up to `max_token_limit`)
- Summarizes older turns into a running summary
- Best of both worlds: recent context is exact, old context is compressed

**LangChain limitations:**
1. The summarization model is called synchronously — it blocks the response chain
2. There is no built-in persistence — memory is in-RAM and lost on process restart
3. The summarization prompt is fixed and not easily customized in early versions
4. Token counting uses approximate heuristics, not the model's actual tokenizer

**Production fix:** Use LangGraph instead of LangChain memory for any non-trivial agent. LangGraph's state graph is explicit, checkpointable, and survives restarts. See topic 4.

---

### Key Numbers to Memorize

| Metric | Value |
|---|---|
| Summarization trigger threshold (practical) | 2,000 tokens |
| Compression ratio of good summarization | 5–10× |
| Summarization call latency | 200–800ms |
| Token counting overhead per message (OpenAI) | +4 tokens |
| Words to tokens ratio (English) | ×1.3 |
| `ConversationSummaryBufferMemory` token limit (recommended) | 2,000–4,000 |
| Cost of summarization call (GPT-4o-mini) | ~$0.0001–0.0005 per summarization |