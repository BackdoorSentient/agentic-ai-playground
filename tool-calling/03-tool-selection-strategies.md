## 3. Tool Selection Strategies — Handling Small to Massive Toolsets

### Q1: What is the tool selection problem, and why does putting all tools in context fail at scale?

**Answer:**

**Tool selection** is the model's task of choosing the right tool(s) from the available set given the user's intent. For small toolsets (<10), this is trivial — the model reads all descriptions and picks correctly ~99% of the time. For large toolsets (50+), accuracy collapses because:

**1. Attention dilution:** The model's attention mechanism distributes across all tokens in context. 50 tool schemas might add 10,000–25,000 tokens to every request. The signal-to-noise ratio for any single tool description drops sharply.

**2. Semantic collision:** When tools have similar purposes (e.g., `search_orders`, `find_order`, `lookup_purchase`), the model struggles to distinguish them. Each additional similar tool pulls attention away from all others.

**3. Token cost at scale:** 50 tool schemas × 200 tokens each = 10,000 tokens per request. At GPT-4o pricing ($2.50/1M input tokens), 1M requests/month = $25,000/month in tool schema tokens alone. This is pure overhead — most requests only need 1–2 tools.

**4. Context window budget:** Tool schemas consume context that could hold conversation history, user data, or retrieved documents. This is an opportunity cost, not just a dollar cost.

**Empirical accuracy by toolset size:**

| Tools in context | GPT-4o accuracy | Claude 3.5 Sonnet accuracy |
|---|---|---|
| 1–5 | ~99% | ~99% |
| 6–10 | ~95–97% | ~96–98% |
| 11–20 | ~88–93% | ~90–94% |
| 21–50 | ~75–85% | ~78–87% |
| 50+ | <70% | <72% |

The inflection point is around 20 tools. Beyond this, you need a routing strategy.

---

### Q2: What is the categorized routing strategy, and how do you implement it for 10–50 tools?

**Answer:**

**Categorized routing** uses a fast first-pass classification step to narrow the full toolset down to a relevant subset, then runs the LLM with only that subset. The classifier is cheap (fast model, few tokens) and the main LLM call operates on a small, accurate set.

**Architecture:**
```
User query → [Classifier] → "This is a billing query"
                              → Load billing tools (3 of 40 total)
                              → LLM call with 3 tools → correct tool selected ~99%
```

**Implementation:**

```python
from enum import Enum
from langchain_openai import ChatOpenAI

# 1. Define tool categories
TOOL_REGISTRY = {
    "billing": [get_invoice, process_refund, update_payment_method, check_subscription],
    "orders": [get_order_status, track_shipment, cancel_order, modify_order],
    "account": [get_user_profile, update_email, reset_password, get_login_history],
    "support": [create_ticket, escalate_ticket, get_ticket_status, close_ticket],
    "search": [search_products, search_orders, search_knowledge_base],
}

# 2. Fast classifier (use cheap model — gpt-4o-mini or claude-haiku)
CLASSIFIER_PROMPT = """Classify the user's intent into exactly one category.
Categories: billing, orders, account, support, search, general

User message: {message}

Respond with only the category name, nothing else."""

classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def classify_intent(user_message: str) -> str:
    result = classifier_llm.invoke(
        CLASSIFIER_PROMPT.format(message=user_message)
    )
    category = result.content.strip().lower()
    return category if category in TOOL_REGISTRY else "general"

# 3. Load relevant tools and run main LLM
def run_with_routing(user_message: str) -> str:
    # Fast classification: ~50-100ms, ~$0.00002
    category = classify_intent(user_message)
    
    # Load only relevant tools (3-5 instead of 40)
    tools = TOOL_REGISTRY.get(category, [tool for tools in TOOL_REGISTRY.values() for tool in tools])
    
    # Main LLM call with small, relevant toolset: ~99% accuracy
    main_llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)
    response = main_llm.invoke([HumanMessage(content=user_message)])
    
    return response
```

**Trade-offs of categorized routing:**

| Dimension | Pro | Con |
|---|---|---|
| Latency | +50–100ms for classifier | Saved by smaller context |
| Cost | Classifier is cheap ($0.00002) | Two LLM calls instead of one |
| Accuracy | ~99% after routing | Misclassification → no valid tool available |
| Maintenance | Clear tool organization | Must update routing when adding tools |

**Failure mode:** Cross-category queries. "I want to update my email and check my order." — classifier picks one category, misses the other. **Fix:** Allow multi-category classification; load the union of relevant tools.

---

### Q3: How does RAG-based tool retrieval work for 50+ tools, and what are the implementation details?

**Answer:**

For toolsets of 50+ tools, semantic retrieval replaces categorical routing. You embed every tool description, store them in a vector database, and at query time retrieve the top-K most semantically relevant tools.

**Architecture:**
```
Offline: Embed all tool descriptions → store in vector DB

Online:  User query → embed query → semantic search → top-3 tools retrieved
         → LLM call with only those 3 tools → accurate tool selection
```

**Implementation:**

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import BaseTool
import json

class ToolRegistry:
    def __init__(self, tools: list[BaseTool]):
        self.tools = {t.name: t for t in tools}
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._build_index(tools)
    
    def _build_index(self, tools: list[BaseTool]):
        """Embed tool descriptions and store in vector DB."""
        # Create rich text for each tool (name + description + arg names)
        tool_texts = []
        tool_ids = []
        
        for tool in tools:
            # Include name, description, AND argument descriptions for better retrieval
            arg_descriptions = ""
            if hasattr(tool, "args_schema") and tool.args_schema:
                schema = tool.args_schema.schema()
                for field_name, field_info in schema.get("properties", {}).items():
                    arg_descriptions += f" {field_name}: {field_info.get('description', '')}"
            
            rich_text = f"{tool.name}: {tool.description}.{arg_descriptions}"
            tool_texts.append(rich_text)
            tool_ids.append(tool.name)
        
        self.vectorstore = Chroma.from_texts(
            texts=tool_texts,
            embedding=self.embeddings,
            metadatas=[{"tool_name": tid} for tid in tool_ids]
        )
    
    def retrieve_tools(self, query: str, k: int = 3) -> list[BaseTool]:
        """Retrieve top-k most relevant tools for a query."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        retrieved = []
        for doc, score in results:
            tool_name = doc.metadata["tool_name"]
            if score < 0.8:  # cosine distance threshold — skip irrelevant tools
                retrieved.append(self.tools[tool_name])
        
        return retrieved if retrieved else list(self.tools.values())[:3]  # fallback

# Usage
registry = ToolRegistry(all_100_tools)

def run_with_rag_routing(user_message: str) -> str:
    # Retrieve top 3-5 relevant tools (~100ms embedding + vector search)
    relevant_tools = registry.retrieve_tools(user_message, k=4)
    
    # Run LLM with small, relevant toolset
    llm = ChatOpenAI(model="gpt-4o").bind_tools(relevant_tools)
    return llm.invoke([HumanMessage(content=user_message)])
```

**Key parameters:**

| Parameter | Recommended value | Why |
|---|---|---|
| Embedding model | `text-embedding-3-small` | $0.02/1M tokens — 200× cheaper than GPT-4o |
| Top-K | 3–5 | Enough to cover the task; beyond 5, accuracy gains plateau |
| Similarity threshold | 0.7–0.85 (cosine distance) | Filter irrelevant tools below threshold |
| Re-index frequency | On every tool addition/edit | Descriptions change → embeddings go stale |

**Latency breakdown for RAG routing:**
- Embed query: ~20–50ms
- Vector search (ChromaDB, 100 tools): ~5–15ms
- Total routing overhead: ~25–65ms
- Saved context tokens: ~80–95% of tool schemas
- Net result: lower latency AND higher accuracy than full-context approach

---

### Q4: What is the hierarchical agent strategy for 100+ tools, and when do you choose it over RAG routing?

**Answer:**

When a toolset exceeds ~100 tools, RAG retrieval starts to struggle with disambiguation — many tools have similar enough descriptions that the wrong one lands in top-K. The hierarchical agent pattern solves this by organizing tools into specialist sub-agents, each with its own small toolset.

**Architecture:**
```
User query
    ↓
[Supervisor / Router Agent]
    ↓ routes to →  [Billing Specialist Agent]   ← 5 billing tools
                   [Orders Specialist Agent]    ← 6 order tools
                   [Search Specialist Agent]    ← 4 search tools
                   [Account Specialist Agent]   ← 5 account tools
```

Each specialist has ≤10 tools → ~99% selection accuracy. The supervisor never sees individual tool schemas — it only decides which specialist to engage.

**LangGraph implementation:**
```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Billing specialist — knows only billing tools
billing_tools = [get_invoice, process_refund, update_payment, check_subscription]
billing_llm = ChatAnthropic(model="claude-haiku-4-5").bind_tools(billing_tools)

def billing_agent(state):
    response = billing_llm.invoke(state["messages"])
    return {"messages": [response]}

# Orders specialist
orders_tools = [get_order_status, track_shipment, cancel_order, modify_order]
orders_llm = ChatAnthropic(model="claude-haiku-4-5").bind_tools(orders_tools)

def orders_agent(state):
    response = orders_llm.invoke(state["messages"])
    return {"messages": [response]}

# Supervisor — routes between specialists
SUPERVISOR_PROMPT = """You are a routing agent. Based on the user's message, 
decide which specialist to route to. 

Specialists:
- billing: invoice questions, refunds, payments, subscriptions
- orders: order status, shipping, cancellations, modifications  
- account: profile, email, password, login history

Respond with only the specialist name: billing, orders, or account."""

def supervisor(state):
    llm = ChatAnthropic(model="claude-haiku-4-5")
    routing_response = llm.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        state["messages"][-1]  # only the latest user message
    ])
    return routing_response.content.strip()

# Build graph
graph = StateGraph(AgentState)
graph.add_node("supervisor", lambda s: {"next": supervisor(s)})
graph.add_node("billing", billing_agent)
graph.add_node("billing_tools", ToolNode(billing_tools))
graph.add_node("orders", orders_agent)
graph.add_node("orders_tools", ToolNode(orders_tools))

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", lambda s: s["next"], {
    "billing": "billing",
    "orders": "orders",
})
```

**Routing strategy selection guide:**

| Tools | Strategy | Latency overhead | Accuracy | Infra cost |
|---|---|---|---|---|
| <10 | All in context | 0ms | ~99% | Zero |
| 10–50 | Categorized routing | +50–100ms | ~95–97% | Minimal |
| 50–200 | RAG retrieval | +25–65ms | ~93–96% | Vector DB |
| 200+ | Hierarchical agents | +100–300ms | ~97–99% | Multiple LLM calls |

---

### Q5: How do you handle cross-category queries and tool ambiguity in any routing strategy?

**Answer:**

**Cross-category queries:** "What's my refund status for order ORD-123456 and update my email?"

This query spans billing and account. Any single-category router fails on the second intent.

**Fix — multi-intent classification:**
```python
def classify_multi_intent(message: str) -> list[str]:
    prompt = """Identify ALL relevant tool categories for this query.
    Categories: billing, orders, account, support, search
    Return a JSON array of category names. Example: ["billing", "account"]
    
    Query: {message}"""
    
    result = classifier_llm.invoke(prompt.format(message=message))
    categories = json.loads(result.content)
    return categories

# Load union of all relevant tool subsets
def get_tools_for_query(message: str) -> list:
    categories = classify_multi_intent(message)
    tools = []
    for cat in categories:
        tools.extend(TOOL_REGISTRY.get(cat, []))
    return list(set(tools))  # deduplicate
```

**Tool ambiguity:** Two tools have overlapping descriptions — model picks the wrong one.

**Fix — reciprocal disambiguation in descriptions:**
- `search_orders`: "Search for orders by customer name, email, or date range. Use this for BROWSING orders. Do NOT use for checking the status of a specific order ID — use get_order_status for that."
- `get_order_status`: "Get delivery status for a SPECIFIC order by its ORD-XXXXXX ID. Do NOT use for searching across multiple orders — use search_orders for that."

This "Do NOT use for X — use Y instead" pattern is the single most effective disambiguation technique and requires zero infrastructure changes.

---

### Key Numbers to Memorize

| Metric | Value |
|---|---|
| All-in-context accuracy threshold | ≤10 tools for ~99% |
| Tool selection accuracy with 50+ in context | <70–72% |
| Classifier model cost (gpt-4o-mini) | ~$0.00002 per classification |
| RAG routing overhead | 25–65ms |
| Embedding model cost | $0.02 / 1M tokens (text-embedding-3-small) |
| Optimal top-K for tool retrieval | 3–5 |
| Hierarchical agent latency overhead | +100–300ms |
| Accuracy after hierarchical routing | ~97–99% |