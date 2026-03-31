## 2. Chain-of-Thought (CoT)

### Q1: What is Chain-of-Thought prompting and what is the core insight behind it?

**Answer:**

**Chain-of-Thought (CoT)** is a prompting technique where the model is guided (or prompted) to produce intermediate reasoning steps before arriving at a final answer. Instead of jumping to an output, the model "thinks out loud."

**Core insight:** LLMs are next-token predictors. If you train them to produce a correct answer token directly, they skip reasoning. If you make them generate a reasoning chain first, those intermediate tokens condition the final answer token — acting as working memory.

**Standard prompting (fails on multi-step):**
```
Q: "Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many does he have?"
A: "11" ✅ (gets lucky on simple cases)

Q: "A bat and ball cost $1.10. Bat costs $1 more than ball. How much is the ball?"
A: "$0.10" ❌ (intuitive wrong answer — correct is $0.05)
```

**CoT prompting (forces reasoning):**
```
Q: "A bat and ball cost $1.10. Bat costs $1 more than ball. How much is the ball?
Let's think step by step."

A: "Let the ball cost x cents. The bat costs x + 100 cents.
Together: x + (x + 100) = 110. So 2x + 100 = 110. 2x = 10. x = 5.
The ball costs $0.05." ✅
```

**Why this matters for agents:** Agent decisions are often multi-step — "should I call tool A or tool B, and in what order?" Without CoT, agents make brittle single-step decisions. With CoT, you get auditable reasoning you can log and debug.

---

### Q2: What are the different CoT variants and when do you use each?

**Answer:**

**1. Zero-shot CoT**
Append "Let's think step by step" or "Think through this carefully before answering."

```python
prompt = f"{user_question}\n\nLet's think step by step."
```

Best for: General reasoning, math, logic. No examples needed.

---

**2. Few-shot CoT**
Provide complete (question + reasoning trace + answer) examples.

```
Q: "If a train travels 60 mph for 2.5 hours, how far does it go?"
A: "Distance = speed × time = 60 × 2.5 = 150 miles."

Q: "If a car travels 45 mph for 3 hours and 20 minutes, how far?"
A: "3 hours 20 minutes = 3.33 hours. Distance = 45 × 3.33 = 150 miles."
```

Best for: Domain-specific reasoning, consistent format requirements, high-stakes outputs.

---

**3. Self-consistency CoT**
Generate the same problem multiple times (temperature > 0), collect N reasoning chains, take majority vote on final answer.

```python
answers = []
for _ in range(5):  # sample 5 chains
    response = llm(prompt, temperature=0.7)
    answers.append(extract_answer(response))

final_answer = Counter(answers).most_common(1)[0][0]  # majority vote
```

Best for: Math and logic where a single chain can go wrong. Wang et al. (2022) showed self-consistency improves GSM8K accuracy from 56.5% → 74.4% on PaLM 540B.

**Cost:** N× more tokens and latency. Use only when accuracy is critical and cost is secondary.

---

**4. Tree of Thought (ToT)**
Explore multiple reasoning paths simultaneously, evaluate intermediate states, backtrack and prune.

```
Root: "Plan how to solve X"
  ├── Branch A: Approach 1 → evaluate → continue if promising
  ├── Branch B: Approach 2 → evaluate → prune if bad
  └── Branch C: Approach 3 → evaluate → continue
```

Best for: Planning, complex problem-solving, game strategies. Yao et al. (2023) showed ToT solves 74% of Game of 24 puzzles vs. 4% for standard CoT.

Cost: Very expensive — multiple LLM calls per level of the tree.

---

**5. Program-of-Thought (PoT)**
Model generates code instead of natural language reasoning. Code is executed to produce the final answer.

```python
# Prompt: "How many seconds in a leap year?"
# Model generates:
days_in_year = 366
hours_per_day = 24
minutes_per_hour = 60
seconds_per_minute = 60
total_seconds = days_in_year * hours_per_day * minutes_per_hour * seconds_per_minute
print(total_seconds)
# Execute → 31,622,400
```

Best for: Precise arithmetic, combinatorics, any task where execution is more reliable than text reasoning.

---

### Q3: What is the "scratchpad" pattern and how does it work in production agents?

**Answer:**

The **scratchpad pattern** is giving the agent an explicit reasoning space — typically using XML tags — before it produces its final output. This is what Claude's extended thinking uses internally.

```python
system_prompt = """
Before responding, use <thinking> tags to reason through the problem.
After reasoning, produce your final answer outside the tags.

<thinking>
[Your step-by-step reasoning here — not shown to the user]
</thinking>

[Final answer shown to user]
"""
```

**Why XML tags work:** The model is trained to treat content inside tags as a distinct context. `<thinking>` signals "this is internal reasoning, not output." The model produces longer, more thorough reasoning inside the scratchpad because it's not constrained to sound polished.

**Production implementation at Anthropic:**
- Claude's extended thinking mode exposes the scratchpad as a `thinking` content block in the API
- The thinking block is generated with a higher token budget than the output
- Anthropic reports extended thinking improves performance on complex reasoning benchmarks by 10–30%

**Agent scratchpad example:**
```python
def agent_decide_tool(user_query, available_tools):
    prompt = f"""
    User query: {user_query}
    Available tools: {available_tools}
    
    <thinking>
    What is the user asking for?
    Which tool is most relevant?
    What parameters does it need?
    Are there any edge cases?
    </thinking>
    
    Tool to call:
    """
    return llm(prompt)
```

**Key trade-off:** Scratchpad/thinking tokens cost money but don't appear in the final response. On Claude's API, extended thinking tokens are billed at the same rate as output tokens. Budget accordingly — a 2,000-token thinking block at $15/1M output tokens (Claude Opus) costs $0.03 per call.

---

### Q4: How do you evaluate whether CoT is actually helping in your system?

**Answer:**

**Ablation test:** Run your eval dataset with and without CoT. Compare accuracy. If CoT doesn't improve accuracy by at least 5–10%, the added token cost isn't justified for simple tasks.

**Quality signals to measure:**
1. **Task accuracy:** Does CoT improve final answer correctness on your golden dataset?
2. **Reasoning faithfulness:** Does the stated reasoning actually lead to the stated conclusion? (LLM-judge this separately)
3. **Error localization:** When the model is wrong, can you identify which step in the chain failed? CoT makes debugging possible.
4. **Latency impact:** CoT adds 200–1000ms depending on chain length. Measure P95 latency with and without.

**When CoT hurts:**
- Very simple tasks: Adding CoT to "translate 'hello' to French" wastes tokens with no benefit
- Latency-critical paths: A 500ms CoT chain is unacceptable for autocomplete features
- Tasks with correct intuitive answers: CoT can sometimes "overthink" into a wrong answer on simple pattern matching

**Rule of thumb:** Use CoT when the task has more than 2 logical steps, requires arithmetic or comparison, or when wrong answers are costly (agents, financial, medical).

---