## 7. RLHF

### Q1: What is RLHF and how does it work step by step?

**Answer:**

**RLHF (Reinforcement Learning from Human Feedback)** is the process of aligning a language model's outputs to human preferences — making it more helpful, honest, and harmless.

**Step-by-step process:**

**Step 1: Supervised Fine-Tuning (SFT)**
- Start with a pre-trained base model.
- Fine-tune it on a dataset of high-quality human-written responses to prompts.
- Output: An SFT model that can follow instructions.

**Step 2: Reward Model Training**
- Collect **comparison data**: for the same prompt, show annotators two model responses and ask which is better.
- Train a separate **reward model (RM)** to predict human preference scores.
- The RM takes (prompt, response) as input and outputs a scalar reward score.

**Step 3: PPO Fine-tuning**
- Use **Proximal Policy Optimization (PPO)** — a reinforcement learning algorithm — to fine-tune the SFT model.
- For each prompt, the policy (language model) generates a response, the RM scores it, and PPO updates the model weights to maximize reward.
- A **KL divergence penalty** prevents the model from drifting too far from the SFT model (otherwise it learns to "game" the reward model with incoherent outputs that score high).

**Formula:**
```
Objective = E[reward_model(response)] - β × KL(policy || SFT_policy)
```

Where β controls how much the model is allowed to deviate from SFT behavior.

---

### Q2: What is DPO and how does it differ from RLHF?

**Answer:**

**DPO (Direct Preference Optimization)** achieves the same alignment goal as RLHF but without training a separate reward model or using RL.

**Key insight:** The optimal RLHF policy has a closed-form solution. DPO reparameterizes the RLHF objective so you can train directly on preference pairs using a simple supervised loss.

**DPO loss:**
```
L_DPO = -log σ(β × log(π(y_w | x) / π_ref(y_w | x)) - β × log(π(y_l | x) / π_ref(y_l | x)))
```

Where:
- `y_w` = preferred (winning) response
- `y_l` = rejected (losing) response
- `π` = current policy
- `π_ref` = reference (SFT) policy

**RLHF vs DPO:**

| Aspect | RLHF | DPO |
|---|---|---|
| Complexity | High (3 stages: SFT, RM, PPO) | Low (1 stage on preference pairs) |
| Training stability | Fragile (PPO is sensitive to hyperparameters) | More stable (supervised loss) |
| Reward model | Required (separate model) | Not needed |
| Performance | State of art (GPT-4, Claude) | Competitive, simpler to implement |
| Data format | Preference rankings | Same (chosen vs rejected pairs) |

**Real-world usage:** Llama 2, Mistral, and many open-source models use DPO or variants (IPO, KTO) because of its simplicity. GPT-4 and Claude 3 reportedly use RLHF.

---

### Q3: What is Constitutional AI and how does Anthropic use it?

**Answer:**

**Constitutional AI (CAI)** is Anthropic's approach to alignment, introduced in their 2022 paper. Instead of relying on human feedback for every output, it uses a set of written principles (a "constitution") to guide AI-generated feedback.

**Process:**

1. **Critique generation:** The model is prompted to critique its own output against constitutional principles (e.g., "Is this response harmful? Is it honest?").
2. **Revision:** The model revises its output based on its own critique.
3. **AI Feedback (RLAIF):** A preference model is trained on AI-generated preference data (not just human labels), using the constitution as the rubric.

**Why it matters:**
- Scales alignment without proportional human labeling costs.
- More transparent: principles are explicit and auditable.
- Reduces sycophancy: the model learns to prioritize honesty over making users feel good.

**Example constitutional principles:**
- "Choose the response that is less likely to contain false or misleading information."
- "Choose the response that is less harmful and more ethical."
- "Choose the response that is less racist, sexist, or otherwise discriminatory."

---