## 5. Pre-training

### Q1: What happens during LLM pre-training, and what does the model actually learn?

**Answer:**

Pre-training is the first and most expensive phase of building an LLM. The model learns to predict the next token given a sequence of tokens, trained on massive text corpora.

**Objective:** Next-token prediction (autoregressive language modeling). Given tokens [t1, t2, ..., t_n], predict t_{n+1}.

**Loss function:** Cross-entropy loss between predicted probability distribution and the actual next token.

**What the model learns:**
- Grammar, syntax, style across many languages
- World knowledge embedded in training data (facts, relationships, reasoning patterns)
- Code patterns across programming languages
- Long-range dependencies via attention mechanisms

**Scale of pre-training data:**
- GPT-3: 300 billion tokens (~570GB of text)
- GPT-4: Estimated 1–13 trillion tokens (undisclosed)
- Llama 3: 15 trillion tokens
- Common data sources: CommonCrawl, Wikipedia, GitHub, books, academic papers

**Compute cost (approximate):**
- GPT-3 (175B params): ~$4–12M in compute at 2020 prices
- GPT-4: Estimated $50–100M+
- Llama 3 70B: ~$1.5M on Meta infrastructure

**Result:** A **base model** that is an excellent text predictor but doesn't know how to be helpful, follow instructions, or avoid harmful outputs. That comes from fine-tuning and RLHF.

---

### Q2: What is the training lifecycle — pre-training → fine-tuning → RLHF — and what does each stage add?

**Answer:**

| Stage | Data | Objective | Result |
|---|---|---|---|
| Pre-training | Web-scale unlabeled text (trillions of tokens) | Next-token prediction | Base model: knows language and facts |
| Supervised Fine-Tuning (SFT) | High-quality instruction-response pairs (thousands to millions) | Imitate desired responses | Instruction-following model |
| RLHF / DPO | Human preference rankings between responses | Maximize human preference scores | Aligned, helpful, safe model |

**Analogy:** Pre-training is like a child reading every book ever written. SFT is like teaching them how to answer questions in a professional context. RLHF is like giving them feedback on which answers were helpful vs. annoying until they internalize the preference.

---

### Q3: What is tokenization and why does it matter for agents?

**Answer:**

**Tokenization** converts raw text into the integer token IDs the model processes. There is no single universal mapping — each model family has its own tokenizer.

**Common tokenization algorithms:**
- **BPE (Byte-Pair Encoding):** Iteratively merges the most common byte pairs. Used by GPT-2/3/4, Llama.
- **SentencePiece:** Language-agnostic subword tokenization. Used by T5, BERT, Gemma.
- **tiktoken:** OpenAI's fast BPE implementation.

**Why it matters for agents:**

1. **Cost:** You're billed per token, not per word. "ChatGPT" is 1 token. A UUID like "a4f2b7c1-9e3d" might be 8 tokens.

2. **Context limits:** Your 10,000-word document might be 13,000 tokens — you can't assume 1:1 mapping.

3. **Multilingual gotchas:** Non-English languages (especially CJK, Arabic) often tokenize very inefficiently. A 100-word Chinese sentence might be 300 tokens in a model not optimized for Chinese.

4. **Code tokenization:** Different models tokenize code differently. Python indentation spaces each count as tokens.

**Tool:** OpenAI's [Tokenizer](https://platform.openai.com/tokenizer) or the `tiktoken` library to count tokens before sending.

---
