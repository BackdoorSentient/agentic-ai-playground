## 3. Top-p (Nucleus Sampling)

### Q1: What is nucleus sampling and how does it differ from top-k?

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

### Q2: When would you set top-p = 1.0 vs top-p = 0.5?

**Answer:**

- **top-p = 1.0**: No nucleus cutoff. The model can sample from any token in the vocabulary (full distribution). Maximizes diversity. Use for creative tasks where even rare tokens are desirable.
- **top-p = 0.9**: Cuts off the bottom 10% of the probability mass. A safe default for most applications.
- **top-p = 0.5**: Very conservative. Only samples from the highest-probability tokens that together cover 50% of the mass. Tight, predictable output. Good for structured/factual tasks.

**Real-world example:** For a legal document summarizer, you might use `temperature=0.2, top_p=0.7` — you want grounded, consistent language, not creative paraphrasing that might misrepresent the original.

---

### Q3: What happens if you set both temperature and top-p to non-default values simultaneously?

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