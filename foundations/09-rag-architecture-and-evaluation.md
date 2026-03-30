## 9. RAG Architecture & Evaluation

### Q1: What is RAG and what problem does it solve?

**Answer:**

**RAG (Retrieval-Augmented Generation)** is the pattern of retrieving relevant documents from an external knowledge store and injecting them into the LLM's context window at inference time, before generating a response.

**Problem it solves:**

LLMs have two fundamental limitations:
1. **Knowledge cutoff:** Training data has a cutoff date. The model doesn't know about events after that date.
2. **Hallucination:** Models fabricate plausible-sounding but incorrect information when they lack knowledge.

**RAG addresses both** by grounding the model's response in retrieved, verifiable documents.

**RAG pipeline components:**

```
User Query
    ↓
[Query Embedding]  ← Embed query into vector space
    ↓
[Vector Search]  ← Find top-k similar document chunks
    ↓
[Reranker (optional)]  ← Re-score retrieved chunks for relevance
    ↓
[Context Assembly]  ← Inject chunks into prompt
    ↓
[LLM Generation]  ← Generate response grounded in context
    ↓
[Response + Citations]
```

---

### Q2: How do embeddings work and how do you choose an embedding model?

**Answer:**

**Embeddings** convert text into dense numerical vectors that capture semantic meaning. Similar texts have vectors that are close in the embedding space (measured by cosine similarity or dot product).

**How they work:** An embedding model (usually a smaller transformer like BERT or a dedicated embedding model) processes text and outputs a fixed-size vector (e.g., 768, 1536, or 3072 dimensions). The training objective ensures that semantically similar texts produce similar vectors.

**Embedding model comparison:**

| Model | Dimensions | Max Tokens | Performance (MTEB) | Cost |
|---|---|---|---|---|
| OpenAI text-embedding-3-small | 1536 | 8191 | ~62.3% | $0.02/1M tokens |
| OpenAI text-embedding-3-large | 3072 | 8191 | ~64.6% | $0.13/1M tokens |
| Cohere embed-v3 | 1024 | 512 | ~64.5% | $0.10/1M tokens |
| BGE-large-en-v1.5 (OSS) | 1024 | 512 | ~63.5% | Free (self-hosted) |
| E5-mistral-7b (OSS) | 4096 | 4096 | ~66.6% | Free (self-hosted) |

**MTEB (Massive Text Embedding Benchmark)** is the standard benchmark. Higher = better retrieval.

**Key decision factors:**
- **Latency sensitive:** Use smaller models (text-embedding-3-small)
- **Quality critical:** Use larger models or E5-mistral
- **Data privacy:** Self-host OSS models (BGE, E5)
- **Multilingual:** Use Cohere embed-v3 or multilingual-E5

---

### Q3: What is reranking and when is it necessary?

**Answer:**

**Vector search** (approximate nearest neighbor) is fast but imprecise — it finds semantically similar chunks, not necessarily the most relevant ones for answering the specific query.

**Reranking** is a second-stage scoring pass that takes the top-k retrieved chunks (e.g., k=20) and rescores them using a more expensive but more accurate model, returning the top-n (e.g., n=5) for the prompt.

**How rerankers work:** A cross-encoder model takes (query, document) as a pair and outputs a relevance score. Unlike bi-encoders (used for embeddings), cross-encoders can model the interaction between query and document — much more accurate.

**When to use reranking:**
- When retrieval quality is the bottleneck in your RAG pipeline
- When you have a large document corpus (>100K chunks)
- When queries are complex or require reasoning to determine relevance

**Reranker options:**
- **Cohere Rerank:** $2/1K searches. Easy API integration.
- **BGE-reranker-large:** Free, self-hosted, ~SOTA open-source
- **ColBERT:** Token-level late interaction, excellent for long documents

**Numbers:** In practice, adding a reranker increases RAG answer accuracy by 10–20% on benchmarks like BEIR. The cost is added latency (~100–300ms) and API cost.

---

### Q4: What is a golden dataset and how do you evaluate RAG quality?

**Answer:**

A **golden dataset** is a curated set of (question, expected answer, source document) triples that represents the ground truth for evaluating your RAG system.

**How to build one:**
1. Sample 50–200 representative queries from production (or synthetic generation)
2. Have domain experts write ideal answers and cite the relevant source document
3. Include edge cases: ambiguous queries, queries with no good answer, multi-hop reasoning queries

**RAG evaluation metrics:**

| Metric | What it Measures | Tool |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? | Ragas, TruLens |
| **Answer Relevance** | Does the answer address the question? | Ragas |
| **Context Recall** | Did retrieval find the right documents? | Ragas, custom |
| **Context Precision** | Are retrieved docs relevant (no noise)? | Ragas |
| **Answer Correctness** | Is the answer factually correct vs. golden? | ROUGE, BERTScore, LLM-judge |

**Ragas RAG triad (the essential three):**
```
Faithfulness: answer is grounded in context  (0–1)
Answer Relevance: answer addresses the question (0–1)
Context Recall: right documents were retrieved (0–1)
```

**Regression testing:** Every time you change the embedding model, chunk size, retrieval k, or prompt, run the golden dataset and compare scores. A PR that drops faithfulness by >5% should be blocked.

**Chunk size impact (empirical benchmarks):**
- Chunk size 256 tokens: High precision, may miss context
- Chunk size 512 tokens: Best balance for most domains
- Chunk size 1024 tokens: Captures more context, more noise in retrieval

---
