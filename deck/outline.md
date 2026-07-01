# Redrob Candidate Ranker — Presentation Outline

## Slide 1: Title & Thesis
- **Title**: Redrob Intelligent Candidate Ranking System (Antigravity Ranker)
- **Subtitle**: Ranking by structural & behavioral understanding, not raw keywords
- **Core Thesis**: Keyword matching is easily gamed. An production-ready recruitment ranker must integrate semantic matching, lexical precision, structural rules, and behavioral signals to filter out adversaries (honeypots, keyword stuffers) and discover available, qualified engineers.

---

## Slide 2: The Problem with Keyword Matching
- **Keyword Stuffing**: Candidates add lists of modern frameworks (e.g., Pinecone, Milvus, Qdrant) in summaries without having ever shipped a vector retrieval system.
- **The Adversarial Trap**: A keyword-stuffed irrelevant profile (e.g., junior developer or IT consultant) scores 100% on naive lexical matching, while a senior engineer who describes their architectures in plain English gets demoted.
- **The Solution**: We balance lexical signals with sentence embeddings and strict structural qualification rules.

---

## Slide 3: Architecture Overview
- **Diagram**:
  ```mermaid
  graph TD
      A[Raw JSONL Dataset] --> B[scripts/03_build_features.py]
      A --> C[scripts/01_build_embeddings.py]
      A --> D[scripts/02_build_lexical_index.py]
      
      B -->|features.parquet| E[rank.py]
      C -->|embeddings.npy| E
      D -->|tfidf_matrix.pkl| E
      
      E -->|Composite Score & Filters| F[Top 100 CSV & Reasoning]
  ```
- **Precompute vs. Query Split**: Expensive feature extraction, embeddings generation, and lexical index training are done offline. The `rank.py` query pipeline runs entirely offline in under 30 seconds using NumPy and Pandas.

---

## Slide 4: Reading Between the Lines — Disqualifier Design
- **Authoritative Distillation**: Translating the job description (JD) into precise rules.
- **Hard Exclusion**:
  - *Pure Research / No Production*: career history is purely academic/research labs, and no production/shipping keywords are found. Excluded from top 100.
- **Soft Disqualifier Multipliers (Multiplicative Penalties)**:
  - LangChain-only with no classical ML/engineering background (0.25x)
  - Pure architect/tech-lead with no coding for 18+ months (0.30x)
  - Career entirely within IT services/consulting firms (0.20x)
  - CV/Speech/Robotics expertise only with no NLP (0.15x)
  - Job hoppers climbing Senior -> Staff -> Principal (0.20x)
  - Closed-source only with no external validation (0.30x)

---

## Slide 5: Behavioral Signals as Availability
- **The Ghost Candidate Problem**: Candidates with perfect resumes who never log in or reject interviews degrade recruiter metrics.
- **Dynamic Down-weighting**: The composite score is multiplied by a behavioral availability index in `[0.65, 1.25]`.
- **Availability Parameters**:
  - last_active_date decay: Full weight if active within 14 days, decaying exponentially to 0.70 by 180 days.
  - Recruiter response rate & interview attendance (weight 0.1 each).
  - Open to work status (+0.05 bonus).
  - Verified contacts and active GitHub profiles (+0.02 to +0.05 bonuses).

---

## Slide 6: Honeypot Defense
- **The Threat**: Fake profiles with inconsistent dates, impossible skills, or illogical salaries designed to test ranker robustness.
- **Our Multi-Rule Filter**: Candidates are flagged as a honeypot if **2 or more** checks fire:
  - Skill claimed as "expert" but used for <6 months.
  - Total job history months does not match years of experience (difference >30% and >1 year).
  - Multiple current jobs or multiple jobs with null end dates.
  - Logical errors (end_date < start_date, expected min salary > max).
  - Chronological anomalies (graduation year is years after work started).
- **Result**: Honeypots are completely purged from the top 100, landing at 0% honeypot rate.

---

## Slide 7: Fact-Grounded Reasoning
- **The Hallucination Risk**: Calling LLMs for 100,000 candidates is slow, expensive, and runs the risk of hallucinated skills or template copy-pastes.
- **Our Hybrid Backend**:
  - *Template Mode (Production)*: Assembles 1-2 sentences from exact profile facts. Deterministically selected using `hash(candidate_id) % N` to vary phrasing while preserving 100% truth.
  - *Local LLM Mode (Sandbox)*: Integrates `llama-cpp-python` with `Qwen2.5-0.5B-Instruct` for zero-cost local text generation, protected by robust fallbacks.

---

## Slide 8: Compute Budget Engineering
- **Wall-Clock Constraints**: Challenge rules dictate ≤5 minutes RAM usage, CPU-only.
- **Optimizations implemented**:
  - Truncated text representation to reduce transformer context length, improving CPU encoding speeds.
  - Vectorized pandas and numpy calculations for final scoring in `rank.py`.
  - Multi-process parallel encoding during precomputation.
- **Resource Footprint**:
  - `rank.py` runs in **~10-15 seconds** (well under 5 minutes).
  - Memory consumption is well below **1.5 GB RAM** (limit is 16 GB).

---

## Slide 9: Score Distribution and Case Demotions
- **Sub-score Assembly**:
  - 28% Semantic Fit (MiniLM Cosine)
  - 20% Skill Depth Fit (proficiency + duration + endorsements)
  - 12% Lexical Fit (TF-IDF Similarity)
  - 30% Structural Fit (experience band + production evidence - soft penalties)
  - 10% Logistics Fit (relocation + notice period + salary sanity)
- **Baseline Deviations**: A naive system ranking by embeddings alone would rank keyword-stuffed job-hoppers or IT consulting generalists highly. The Antigravity system successfully penalizes these generalists and elevates India-based product engineers with deep ranking-specific skills.

---

## Slide 10: Future Development Checklist
- **Recruiter Feedback Loop**: Online Reinforcement Learning based on which candidates are invited/hired.
- **Learning-to-Rank (LTR)**: Train an XGBoost pairwise ranking model once recruiter feedback labels are collected.
- **Unified Embedding Space**: Fine-tune `all-MiniLM` using Contrastive Learning on actual company job descriptions and hired candidate resumes to align technical vocabulary.
