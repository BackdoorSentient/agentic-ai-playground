## 3. Top-p (Nucleus Sampling)

### Q1: What is the meaning of cumulative probability in simple words?

**Answer:**

Cumulative probability is a running total of probabilities added up in order, until they reach 100%.

Think of it like a scoreboard that keeps adding scores — you're not asking "what's the chance of this exact outcome?" but "what's the chance of this outcome **or anything before it**?"

**Dice example:**

| Roll | Probability | Cumulative Probability |
|------|-------------|----------------------|
| 1    | 17%         | 17%                  |
| 2    | 17%         | 34%                  |
| 3    | 17%         | 51%                  |
| 4    | 17%         | 68%                  |
| 5    | 17%         | 85%                  |
| 6    | 17%         | 100%                 |

**Code:**
```python
probabilities = [1/6] * 6  # equal probability for each face

cumulative = 0
for i, p in enumerate(probabilities, start=1):
    cumulative += p
    print(f"Roll {i} → P(X ≤ {i}) = {cumulative:.2f} ({cumulative*100:.0f}%)")

# Output:
# Roll 1 → P(X ≤ 1) = 0.17 (17%)
# Roll 2 → P(X ≤ 2) = 0.33 (33%)
# Roll 3 → P(X ≤ 3) = 0.50 (50%)
# Roll 4 → P(X ≤ 4) = 0.67 (67%)
# Roll 5 → P(X ≤ 5) = 0.83 (83%)
# Roll 6 → P(X ≤ 6) = 1.00 (100%)
```

So cumulative probability answers: **"What's the chance of getting THIS value or anything below it?"**

### Q2: What does "nucleus" mean in nucleus sampling?

**Answer:**

**Nucleus** means the **core group** — the small cluster of most likely tokens that together account for enough probability mass to meet the threshold `p`.

Think of it like a solar system — the nucleus is the dense center where most of the "weight" lives. Everything outside it is the long tail of low-probability tokens you don't want to sample from.

So "nucleus sampling" = **only sample from the core group of tokens that matter**, ignore everything else.

### Q3: What is nucleus sampling and how does it differ from top-k?

**Answer:**

**Top-k** selects from a fixed number of top tokens. **Top-p (nucleus sampling)** selects from the smallest set of tokens whose **cumulative probability** reaches threshold `p`.

**Algorithm:**
1. Sort tokens by probability, descending.
2. Accumulate probabilities until the running sum ≥ p.
3. Only sample from this "nucleus" set.

**Why this is smarter than top-k:**

Imagine two situations:
- The model is very confident: the top 3 tokens cover 97% probability mass. With top-k=50, you'd include 47 tokens with near-zero probability. With top-p=0.95, you'd only sample from the top 3.
- The model is uncertain: 200 tokens each have ~0.5% probability. With top-k=50, you still only sample 50. With top-p=0.95, you sample from all 200, preserving the model's genuine uncertainty.

Top-p **adapts** to the model's confidence at each step. Top-k does not.

**Typical values:**
- `top_p = 0.9` is a common default.
- OpenAI recommends: "We generally recommend altering `top_p` or `temperature` but not both."

---

### Q4: When would you set top-p = 1.0 vs top-p = 0.5?

**Answer:**

- **top-p = 1.0**: No nucleus cutoff. The model can sample from any token in the vocabulary (full distribution). Maximizes diversity. Use for creative tasks where even rare tokens are desirable.
- **top-p = 0.9**: Cuts off the bottom 10% of the probability mass. A safe default for most applications.
- **top-p = 0.5**: Very conservative. Only samples from the highest-probability tokens that together cover 50% of the mass. Tight, predictable output. Good for structured/factual tasks.

**Real-world example:** For a legal document summarizer, you might use `temperature=0.2, top_p=0.7` — you want grounded, consistent language, not creative paraphrasing that might misrepresent the original.

---

### Q5: What happens if you set both temperature and top-p to non-default values simultaneously?

**Answer:**

You can, but it's confusing and often unnecessary. The typical guidance (from OpenAI and Anthropic):

> Modify one parameter at a time. Adjusting both temperature and top-p simultaneously creates compounding effects that are difficult to reason about and tune.

**Order of operations** (OpenAI API):
1. Temperature is applied first to reshape logit probabilities.
2. Top-p is applied after, on the reshaped distribution.
3. Top-k (if set) is applied after top-p.
4. A token is sampled from the final set.

So `temperature=2.0, top_p=0.1` would: first flatten the distribution wildly, then cut it back to a tight nucleus — an incoherent combination.

**Best practice:** Set temperature for overall creativity level. Leave top-p at 0.9–1.0 unless you have a specific reason to constrain the nucleus.

---