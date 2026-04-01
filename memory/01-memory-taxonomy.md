## 1. Memory Taxonomy — Short-term, Long-term, Episodic, Semantic, Procedural

### Q1: What are the five memory types in AI agent systems, and how do they map to cognitive psychology?

**Answer:**

AI agent memory is modeled directly on cognitive science. The five types are:

| Memory Type | Cognitive Analogy | Storage | Lifetime | Access Pattern |
|---|---|---|---|---|
| **Short-term (Working)** | Human working memory | In-context (token buffer) | Single conversation | Direct — included in every prompt |
| **Long-term** | Human long-term memory | External DB / vector store | Persistent across sessions | Retrieval-based (semantic search or lookup) |
| **Episodic** | Episodic memory (autobiographical) | Vector DB or KV store | Persistent | Similarity-retrieved by context |
| **Semantic** | Semantic memory (general facts) | Vector DB / knowledge graph | Persistent | Similarity-retrieved by query |
| **Procedural** | Procedural memory (muscle memory) | Prompt templates, code, fine-tuned weights | Persistent / baked-in | Implicit — executed, not recalled |

**Short-term memory** is everything currently in the context window. For GPT-4o, that is 128K tokens (~96,000 words). For Claude 3 series, 200K tokens. For Gemini 1.5 Pro, 1M tokens. It is fast and perfectly accurate — but expensive (every token is paid for) and ephemeral.

**Long-term memory** requires an external store. The agent writes and reads from a database between conversations. This is where user preferences, facts, or historical interactions live.

**Episodic memory** stores specific *events* — "User asked about Python last Tuesday and preferred brief answers." Retrieving episodes requires similarity search: given the current context, find past episodes that match.

**Semantic memory** stores *general facts* — "This user is a backend engineer in Mumbai." Structured, timeless, and retrieved by relevance.

**Procedural memory** is the most underappreciated type: the agent's "how to do X" knowledge. In LLMs, this is encoded in weights during pre-training and fine-tuning. You can also encode it explicitly as few-shot examples, tool call templates, or system prompt instructions.

---

### Q2: Why is short-term memory (context window) not enough, and what is the context overflow problem?

**Answer:**

Short-term memory has four critical limitations in production agents:

**1. Token budget exhaustion:** A 200K-token context at Claude's input pricing ($3/M tokens) costs $0.60 per fully-loaded conversation. Most conversations do not need 200K tokens — but without memory management, they grow toward the limit and blow up latency and cost.

**2. Attention degradation (Lost in the Middle):** Research shows LLMs attend strongly to the *beginning* and *end* of context, with significant attention drop-off in the middle. A 100K-token context with a critical fact at position 50K may be effectively invisible to the model. This is not a bug — it is how attention mechanisms work at scale.

**3. No persistence:** When the session ends, everything in context is gone. The next conversation starts from zero.

**4. No sharing:** A user on a different device, a different agent in a multi-agent system, or a background process cannot access the in-context memory of another conversation.

**The overflow problem in numbers:**
- Average user message: ~50 tokens
- Average assistant response: ~200 tokens
- Average tool call round trip: ~400 tokens
- A 10-turn conversation: ~2,500 tokens
- A 50-turn conversation: ~12,500 tokens
- A multi-session agent with 100 past conversations: ~250K tokens — exceeds all current context windows

**Solutions:** summarization (compress old turns), retrieval (store externally, fetch relevant subset), hybrid (summarize + retrieval-augmented).

---

### Q3: How does episodic memory differ from semantic memory, and when do you choose one architecture over the other?

**Answer:**

**Episodic memory** is event-indexed: *"What happened, when?"*
- Stores: conversation turns, user actions, past agent decisions
- Example: "On March 15, the user asked for a Python FastAPI tutorial and found the response too long"
- Retrieval: given current context, find past events with similar context
- Storage: vector embeddings of events + metadata (timestamp, session ID, user ID)

**Semantic memory** is concept-indexed: *"What is true about this entity?"*
- Stores: user profile facts, domain knowledge, entity attributes
- Example: "User is a senior backend engineer. Prefers code-first explanations. Uses Python and Go."
- Retrieval: given a query or user ID, fetch the structured fact sheet
- Storage: KV store for structured facts, or vector DB for fuzzy recall

**When to choose:**

| Use case | Memory type | Why |
|---|---|---|
| "Remember this conversation for next time" | Episodic | Event-specific, time-bound |
| "Remember I prefer short answers" | Semantic | Timeless user preference |
| "Why did the agent make that decision 3 days ago?" | Episodic | Audit trail of events |
| "What does this user know about ML?" | Semantic | Entity attribute |
| Personalized recommendations | Both | Semantic for profile, episodic for past interactions |

**Production pattern:** Most production agents combine both. A user profile (semantic) is updated based on extracted facts from conversations (episodic). The profile is then retrieved with every new session.

---

### Q4: What is procedural memory in agents, and why is it the hardest to modify?

**Answer:**

Procedural memory is *implicit behavioral knowledge* — how the agent "knows" to do things without being explicitly told in every prompt. It manifests in three forms:

**1. Pre-trained weights:** The base model's knowledge of syntax, reasoning patterns, coding conventions, common-sense logic. This is baked in and cannot be modified without retraining. GPT-4o has ~1.76 trillion parameters encoding this.

**2. Fine-tuned weights:** Domain-specific procedural memory — a model fine-tuned on customer support logs "knows" how to handle refund requests. Modifying this requires another fine-tuning run.

**3. System prompt instructions:** The most modifiable layer. "Always respond in JSON. Always use bullet points. If the user asks about competitors, deflect." This is procedural memory encoded explicitly. Runtime cost: every token in the system prompt is paid on every API call.

**Why it's hard to modify:** Unlike episodic or semantic memory where you update a database row, changing procedural memory embedded in weights requires:
- Curating new examples
- Running a fine-tuning job (hours to days, $100–$10,000)
- Evaluating the retrained model against a test set
- Swapping the model endpoint in production

**Senior engineer tip:** Prefer encoding procedural knowledge in system prompts (fast to change, no retraining) over fine-tuning until you have proven the behavior at scale. Fine-tune only when: (a) the prompt-based instruction regularly fails despite rephrasing, or (b) the instruction is so long it meaningfully increases token cost.

---

### Q5: How do you design a memory architecture for a production multi-session agent?

**Answer:**

A production multi-session agent needs all five memory types working together:

```
┌─────────────────────────────────────────────────────────┐
│                  AGENT CONTEXT WINDOW                   │
│  System Prompt (procedural) + Recent turns (short-term) │
│  + Retrieved memories injected here at session start    │
└────────────────────┬────────────────────────────────────┘
                     │ reads from / writes to
        ┌────────────┴────────────┐
        │                         │
   ┌────▼──────┐           ┌──────▼──────┐
   │ Vector DB  │           │  KV / SQL   │
   │ (episodic) │           │ (semantic)  │
   │ ChromaDB / │           │ Redis /     │
   │ Pinecone   │           │ Postgres    │
   └────────────┘           └─────────────┘
```

**Session start flow:**
1. Retrieve user profile from semantic store (always)
2. Retrieve top-K relevant episodes from vector DB using current context as query
3. Inject both as compressed context at the top of the prompt (below system prompt)
4. Begin conversation with pre-loaded memory

**Session end flow:**
1. Extract new facts from the conversation (LLM extraction call or rule-based)
2. Update semantic store with new facts
3. Embed and store conversation summary in episodic vector DB

**Key numbers:**
- Top-K for retrieval: 3–5 episodes is optimal (beyond this, context bloat outweighs value)
- Embedding model cost: `text-embedding-3-small` = $0.02/1M tokens (~200× cheaper than GPT-4o)
- ChromaDB: free, local, works up to ~1M vectors before performance degrades
- Pinecone: ~$70/month for 1M vectors, handles billions at the enterprise tier
- Memory injection overhead: 200–500 tokens per session (trivial cost)

---

### Key Numbers to Memorize

| Metric | Value |
|---|---|
| GPT-4o context window | 128K tokens |
| Claude 3.7 Sonnet context window | 200K tokens |
| Gemini 1.5 Pro context window | 1M tokens |
| Lost-in-the-Middle attention drop | ~40% accuracy drop for facts in the middle of long contexts |
| text-embedding-3-small cost | $0.02 / 1M tokens |
| ChromaDB practical limit | ~1M vectors |
| Typical memory injection overhead | 200–500 tokens / session |
| Fine-tuning cost range | $100–$10,000 depending on dataset and model |