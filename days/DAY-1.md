# Day 1: Understanding How LLMs Work — Senior Engineer Q&A Notes

> **Format:** Each topic has a set of questions a senior engineer should be able to answer, with deep explanations, trade-offs, real-world examples, and key numbers.

---

## Table of Contents

1. [Product AI Patterns — Assist vs Automate](#1-product-ai-patterns)
2. [Temperature](#2-temperature)
3. [Top-p (Nucleus Sampling)](#3-top-p-nucleus-sampling)
4. [Context Window](#4-context-window)
5. [Pre-training](#5-pre-training)
6. [Fine-tuning](#6-fine-tuning)
7. [RLHF](#7-rlhf)
8. [Structured Outputs](#8-structured-outputs)
9. [RAG Architecture & Evaluation](#9-rag-architecture--evaluation)
10. [FastAPI Framework](#10-fastapi-framework)
11. [Hands-On Exercises](#11-hands-on-exercises)

---

## Quick Reference: Numbers to Memorize

| Fact | Value |
|---|---|
| Tokens to words ratio | ~1 token = 0.75 words |
| Claude 3.5 Sonnet context | 200K tokens |
| GPT-4o context | 128K tokens |
| Gemini 1.5 Pro context | 1M tokens |
| Temperature for tool use | 0.0–0.2 |
| Temperature for creativity | 0.8–1.2 |
| Top-p default | 0.9 |
| LoRA typical rank | 8–64 |
| Fine-tuning minimum examples | 50–200 (style), 500+ (domain) |
| GPT-4o input price | $2.50/1M tokens |
| Gemini Flash input price | $0.075/1M tokens |
| Reranker latency overhead | 100–300ms |
| Optimal RAG chunk size | 256–512 tokens |
| Golden dataset minimum size | 50 examples |

---
