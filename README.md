---
title: Talent Lens
emoji: 🌌
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 4.36.1
app_file: sandbox/app.py
pinned: false
---

# Talent Lens — Redrob Intelligent Candidate Ranking System

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/iamshresthraj/talentlens)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Sandbox-yellow)](https://huggingface.co/spaces/iamshresthraj/talentlens-sandbox)

This repository contains our hackathon submission (**Talent Lens**) for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**. The system ranks a 100,000-candidate dataset to find the top 100 fits for a founding Senior AI Engineer role at Redrob AI, running 100% offline in under 30 seconds.

---

## 🚀 Quick Start (Reproduction Workflow)

Follow these steps to reproduce our ranking results from scratch.

### 1. Installation
Ensure Python 3.10+ is installed. Install required packages:
```bash
pip install -r requirements.txt
```

### 2. Model Setup (Network Required)
Download and cache the sentence-embedding model and tiny GGUF reasoning model locally. This is the **only** script allowed to make network requests:
```bash
python scripts/00_setup_models.py
```

### 3. Precomputation (Exempt from 5-Minute Limit)
Generate features, embeddings, and indices offline:
```bash
# Encode candidate documents to embeddings (multi-processed on CPU)
python scripts/01_build_embeddings.py

# Fit lexical TF-IDF index
python scripts/02_build_lexical_index.py

# Extract structural rules, logistics, and behavioral features
python scripts/03_build_features.py

# Encode the ideal job description vector
python scripts/04_build_jd_vector.py
```

### 4. Running the Ranker (Zero Network, CPU-Only)
Run the main ranking script to filter honeypots/disqualifiers, compute composite scores, generate fact-grounded reasoning, and output the top 100 candidates:
```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

---

## 🏗️ System Architecture & Design Decisions

Our ranker uses a **hybrid cascading approach** combining dense semantics, sparse lexical indices, strict structural disqualifiers, honeypot defenses, and behavioral availability multipliers.

### 1. Scoring Formula
All sub-scores are normalized to `[0, 1]` before linear combination:
$$\text{Linear Score} = 0.28 \cdot \text{Semantic Fit} + 0.20 \cdot \text{Skill Depth} + 0.12 \cdot \text{Lexical Fit} + 0.30 \cdot \text{Structural Fit} + 0.10 \cdot \text{Logistics Fit}$$
$$\text{Final Score} = \text{Linear Score} \cdot \text{Behavioral Multiplier} \cdot \text{Soft Disqualifier Multiplier}$$

- **Semantic Fit**: Cosine similarity between local candidate embeddings (`all-MiniLM-L6-v2`) and the Job Description query embedding.
- **Skill Depth Fit**: Scored on must-have skills from the JD, scaling by proficiency (0.25 to 1.0) and logarithmic duration, plus endorsements bonuses.
- **Lexical Fit**: Cosine similarity of candidate TF-IDF vectors against must/nice-to-have terms.
- **Structural Fit**: Experience band fit (ideal: 5–9 years) combined with production shipping evidence, minus soft-disqualifier penalties.
- **Logistics Fit**: Location relocation checks, notice period penalties, and expected salary sanity.
- **Behavioral Multiplier**: Scaled multiplier in `[0.65, 1.25]` rewarding login recency (exponential half-life decay), recruiter response rate, interview completion, open-to-work flags, and active GitHub presence.

### 2. Honeypot Defense
Honeypots are malicious profiles with impossible details. Candidates are excluded entirely if **2 or more** triggers fire:
1. Expert skills with <6 months duration (2+ expert short-duration skills triggers honeypot directly).
2. Career history months differ from profile years of experience by >30%.
3. Multiple "current" jobs or null end dates in career history.
4. Logical errors (job end_date < start_date, expected salary min > max).
5. Chronological anomalies (graduating years after starting work with no prior degree).

### 3. Disqualifiers
- **Hard Disqualifiers** (Excluded entirely):
  - *Pure Research*: Candidates whose history consists only of academic/lab titles with zero production evidence.
- **Soft Disqualifier Multipliers**:
  - LangChain/OpenAI-wrapper only with no legacy ML engineering (0.25x)
  - Pure architect/tech-lead with no coding for 18+ months (0.30x)
  - Career entirely within IT outsourcing/consulting firms (0.20x)
  - Computer vision/robotics focus only with no NLP (0.15x)
  - Title chaser/company hopper (0.20x)
  - Closed-source only with no validation (0.30x)

### 4. Fact-Grounded Reasoning
To guarantee no hallucinations in the CSV submission:
- **Template Mode (Production)**: Builds 1-2 sentence reasons utilizing candidate name, experience, current title/company, matched skills, and relocation/notice concerns. Phrasing templates are chosen deterministically using `hash(candidate_id) % N` (via MD5) to vary text style across the 100 rows.
- **Local LLM Mode (Sandbox)**: Integrates `llama-cpp-python` with `Qwen2.5-0.5B-Instruct` for on-the-fly reasoning on small datasets.

---

## ⚡ Performance Benchmarks

Measured on an 12-core CPU system:
- **Precompute Time** (Offline): ~4-5 minutes (multi-processed CPU encoding).
- **`rank.py` Query Speed**: **~10-12 seconds** (including JSONL parsed sequentially).
- **RAM Footprint**: **~1.2 GB** (comfortably under the 16 GB RAM budget).
- **Output Submission**: 100 rows, unique ranks 1–100, scores sorted, 0% honeypot rate.
