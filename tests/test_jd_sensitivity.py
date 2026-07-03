"""
Tests that the ranking genuinely follows the job description.

1. test_default_scoring_unchanged: the shared JD-aware engine reproduces the
   original hardcoded recruiter scoring EXACTLY for the default Senior AI
   Engineer configuration (so the official submission.csv is unaffected).
2. test_different_jds_produce_different_rankings: for the same candidate
   dataset, different JDs must produce different top rankings.

Runs under pytest, or standalone: python tests/test_jd_sensitivity.py
"""
import os
import sys
import json
import re
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.scoring import (
    score_candidates,
    compute_skill_depth_fit_raw,
    min_max_normalize,
    match_skill_term,
)
from src.jd_parser import parse_jd
from src.rules import (
    detect_honeypot,
    check_pure_research_no_production,
    check_recent_langchain_only_no_legacy_ml,
    check_senior_no_hands_on_code_18mo,
    check_consulting_only_career,
    check_cv_speech_robotics_only_no_nlp,
    check_title_chaser_job_hopper,
    check_closed_source_only_no_external_validation,
    check_irrelevant_role,
)

N_CANDIDATES = 800


def _load_fixture():
    with open(os.path.join(base_dir, "config", "jd_requirements.yaml"), encoding="utf-8") as f:
        jd_config = yaml.safe_load(f)

    candidates = []
    with open(os.path.join(base_dir, "data", "candidates.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
            if len(candidates) >= N_CANDIDATES:
                break

    firms = jd_config.get("consulting_firms", [])
    rows = []
    for c in candidates:
        rows.append({
            "is_honeypot": int(detect_honeypot(c)),
            "hard_disqualified": int(check_pure_research_no_production(c)),
            "soft_disq_recent_langchain": int(check_recent_langchain_only_no_legacy_ml(c)),
            "soft_disq_senior_no_code": int(check_senior_no_hands_on_code_18mo(c)),
            "soft_disq_consulting_only": int(check_consulting_only_career(c, firms)),
            "soft_disq_cv_speech_robotics": int(check_cv_speech_robotics_only_no_nlp(c)),
            "soft_disq_title_chaser": int(check_title_chaser_job_hopper(c)),
            "soft_disq_closed_source": int(check_closed_source_only_no_external_validation(c)),
        })
    features_df = pd.DataFrame(rows)

    rng = np.random.RandomState(42)
    semantic_fit = rng.rand(len(candidates))
    lexical_fit = rng.rand(len(candidates))
    return jd_config, candidates, features_df, semantic_fit, lexical_fit


def _legacy_scores(candidates, features_df, semantic_fit, skill_depth_fit, lexical_fit):
    """Verbatim replica of the ORIGINAL hardcoded scoring loop from rank.py."""

    def match_skill_term_local(term, skill_name):
        term_lower, skill_lower = term.lower(), skill_name.lower()
        if term_lower in ["c++", "c#", ".net"]:
            return term_lower in skill_lower
        if len(term_lower) <= 3:
            return bool(re.search(r"\b" + re.escape(term_lower) + r"\b", skill_lower))
        return term_lower in skill_lower

    def check_must_have_matches(cand):
        skills = cand.get("skills", [])
        history = cand.get("career_history", [])
        profile = cand.get("profile", {})
        headline = profile.get("headline", "").lower()
        summary = profile.get("summary", "").lower()
        history_text = " ".join([(j.get("title", "") + " " + j.get("description", "")).lower() for j in history])
        profile_text = f"{headline} {summary} {history_text}"
        categories = {
            "embeddings_retrieval": ["sentence-transformers", "openai embeddings", "bge", "e5", "embeddings", "dense retrieval", "semantic search"],
            "vector_db_hybrid_search": ["pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss", "hybrid search", "vector database"],
            "strong_python": ["python"],
            "eval_frameworks": ["ndcg", "mrr", "map", "a/b test", "ab testing", "offline evaluation", "online evaluation", "ranking evaluation"],
        }
        matched_count = 0
        for cat, terms in categories.items():
            has_skill_in_list = False
            for s in skills:
                name = s.get("name", "").lower()
                if any(match_skill_term_local(t, name) for t in terms):
                    in_history = any(t in history_text for t in terms)
                    long_duration = s.get("duration_months", 0) >= 12
                    has_endorsements = s.get("endorsements", 0) > 0
                    is_ml_role = any(r in profile.get("current_title", "").lower() for r in ["ml", "ai", "data", "search", "recommend", "nlp", "backend"])
                    if in_history or long_duration or has_endorsements or (cat == "strong_python" and is_ml_role):
                        has_skill_in_list = True
                        break
            has_text_evidence = False
            if any(t in profile_text for t in terms):
                if any(r in profile.get("current_title", "").lower() for r in ["engineer", "scientist", "developer", "programmer", "architect", "analyst", "mle", "lead"]):
                    has_text_evidence = True
            if has_skill_in_list or has_text_evidence:
                matched_count += 1
        return matched_count

    def check_good_to_have_matches(cand):
        good_to_haves = {
            "llm_finetuning": ["lora", "qlora", "peft", "fine-tuning", "finetuning"],
            "learning_to_rank": ["learning to rank", "ltr", "xgboost ranking", "neural ranking"],
            "hr_tech": ["hr tech", "recruiting", "talent", "marketplace"],
            "distributed_systems": ["distributed systems", "large-scale inference", "model serving at scale"],
            "open_source": ["open source", "github", "publication", "paper", "conference talk"],
        }
        count = 0
        for cat, terms in good_to_haves.items():
            for s in cand.get("skills", []):
                if any(match_skill_term_local(t, s.get("name", "").lower()) for t in terms):
                    count += 1
                    break
        return count

    final_scores = np.zeros(len(candidates))
    eligible_mask = np.ones(len(candidates), dtype=bool)
    for idx, cand in enumerate(candidates):
        row_feat = features_df.iloc[idx]
        is_hp = row_feat["is_honeypot"] == 1 or detect_honeypot(cand)
        is_irrelevant = check_irrelevant_role(cand, "ml_ai")
        is_pure_res = row_feat["hard_disqualified"] == 1 or check_pure_research_no_production(cand)
        disq_scores = []
        if is_hp: disq_scores.append(0.0)
        if is_irrelevant: disq_scores.append(0.0)
        if is_pure_res: disq_scores.append(0.05)
        if row_feat["soft_disq_consulting_only"] == 1: disq_scores.append(0.03)
        if row_feat["soft_disq_cv_speech_robotics"] == 1: disq_scores.append(0.07)
        if row_feat["soft_disq_senior_no_code"] == 1: disq_scores.append(0.08)
        if row_feat["soft_disq_recent_langchain"] == 1: disq_scores.append(0.12)
        if row_feat["soft_disq_title_chaser"] == 1: disq_scores.append(0.10)
        if row_feat["soft_disq_closed_source"] == 1: disq_scores.append(0.09)
        if disq_scores:
            final_scores[idx] = min(disq_scores)
            eligible_mask[idx] = False
            continue
        matched_must = check_must_have_matches(cand)
        if matched_must == 4:
            range_min, range_max = 0.70, 0.85
        elif matched_must == 3:
            range_min, range_max = 0.40, 0.55
        else:
            range_min, range_max = 0.10, 0.25
        s_val = 0.5 * semantic_fit[idx] + 0.3 * skill_depth_fit[idx] + 0.2 * lexical_fit[idx]
        base_score = range_min + s_val * (range_max - range_min)
        adj = 0.0
        adj += check_good_to_have_matches(cand) * 0.03
        signals = cand.get("redrob_signals", {})
        last_active_str = signals.get("last_active_date", "")
        days = 365
        if last_active_str:
            try:
                days = (datetime.strptime("2026-06-17", "%Y-%m-%d").date() - datetime.strptime(last_active_str, "%Y-%m-%d").date()).days
            except Exception:
                pass
        if days <= 30:
            adj += 0.03
        elif days >= 180:
            adj -= 0.05
        if signals.get("open_to_work_flag", False):
            adj += 0.02
        notice_days = signals.get("notice_period_days", 0)
        if notice_days <= 30:
            adj += 0.03
        elif notice_days <= 60:
            adj += 0.01
        if notice_days > 90:
            adj -= 0.02
        if notice_days > 120:
            adj -= 0.04
        final_scores[idx] = max(0.0, min(1.0, base_score + adj))
    return final_scores, eligible_mask


def _rank_ids(candidates, scores, eligible_mask, k=100):
    order = [
        {"id": candidates[i]["candidate_id"], "score": scores[i]}
        for i in np.where(eligible_mask)[0]
    ]
    order.sort(key=lambda x: (-x["score"], x["id"]))
    return [o["id"] for o in order[:k]]


def test_default_scoring_unchanged():
    jd_config, candidates, features_df, semantic_fit, lexical_fit = _load_fixture()
    must_have = jd_config["must_have_skills"]
    nice_to_have = jd_config["nice_to_have_skills"]
    skill_depth_fit = min_max_normalize(np.array([compute_skill_depth_fit_raw(c, must_have) for c in candidates]))

    new_scores, new_mask = score_candidates(
        candidates, features_df, semantic_fit, skill_depth_fit, lexical_fit,
        must_have, nice_to_have, role_type="ml_ai",
    )
    old_scores, old_mask = _legacy_scores(candidates, features_df, semantic_fit, skill_depth_fit, lexical_fit)

    assert np.array_equal(new_mask, old_mask), "eligibility changed for default JD"
    assert np.allclose(new_scores, old_scores, atol=1e-12), (
        f"default-JD scores changed; max diff {np.max(np.abs(new_scores - old_scores))}"
    )


def test_different_jds_produce_different_rankings():
    jd_config, candidates, features_df, semantic_fit, lexical_fit = _load_fixture()

    jds = {
        "ml_ai": "Senior AI engineer, 5 to 9 years experience, machine learning, embeddings, "
                 "vector database, semantic search, python, NDCG evaluation. Pune or Noida.",
        "frontend": "Frontend Engineer with 2 to 4 years experience building React and TypeScript "
                    "applications, strong CSS, state management with Redux, performance profiling. Bangalore.",
        "devops": "DevOps / SRE engineer, 6+ years, Kubernetes, Terraform, AWS, Prometheus and Grafana "
                  "observability, infrastructure as code, bash automation. Hyderabad.",
    }

    rankings = {}
    scores_by_jd = {}
    for name, jd_text in jds.items():
        parsed = parse_jd(jd_text)
        must_have = parsed["must_have_skills"]
        skill_depth_fit = min_max_normalize(np.array([compute_skill_depth_fit_raw(c, must_have) for c in candidates]))
        # Semantic/lexical fits held constant across JDs on purpose: the test
        # proves the JD rubric ALONE re-ranks candidates.
        scores, mask = score_candidates(
            candidates, features_df, semantic_fit, skill_depth_fit, lexical_fit,
            must_have, parsed["nice_to_have_skills"],
            role_type=parsed["role_type"],
            experience_band=parsed["experience_band"],
            fit_weights=(
                parsed["dimension_weights"].get("semantic_fit", 0.5),
                parsed["dimension_weights"].get("skill_depth_fit", 0.3),
                parsed["dimension_weights"].get("lexical_fit", 0.2),
            ),
            preferred_locations=parsed.get("preferred_locations") or None,
        )
        rankings[name] = _rank_ids(candidates, scores, mask, k=100)
        scores_by_jd[name] = scores

    names = list(jds)
    for a_i in range(len(names)):
        for b_i in range(a_i + 1, len(names)):
            a, b = names[a_i], names[b_i]
            assert rankings[a] != rankings[b], f"{a} and {b} JDs produced IDENTICAL rankings"
            overlap = len(set(rankings[a][:25]) & set(rankings[b][:25]))
            assert overlap < 25, f"{a} vs {b}: identical top-25 sets"
            assert not np.allclose(scores_by_jd[a], scores_by_jd[b]), f"{a} vs {b}: identical score vectors"

    # Report divergence for human inspection
    for a_i in range(len(names)):
        for b_i in range(a_i + 1, len(names)):
            a, b = names[a_i], names[b_i]
            top25_overlap = len(set(rankings[a][:25]) & set(rankings[b][:25]))
            same_order = sum(1 for x, y in zip(rankings[a], rankings[b]) if x == y)
            print(f"  {a:>8} vs {b:<8}: top-25 overlap {top25_overlap}/25, same-position ranks {same_order}/100")


if __name__ == "__main__":
    print("test_default_scoring_unchanged ...")
    test_default_scoring_unchanged()
    print("  PASSED - default JD output is unchanged")
    print("test_different_jds_produce_different_rankings ...")
    test_different_jds_produce_different_rankings()
    print("  PASSED - different JDs genuinely re-rank candidates")
