import os
import sys
import json
import yaml
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from src.scoring import (
    compute_skill_depth_fit_raw,
    compute_behavioral_multiplier,
    min_max_normalize
)
from src.reasoning import generate_reasoning


def get_memory_usage_mb():
    """Returns the RSS memory usage of the current process in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback if psutil is not installed
        try:
            if sys.platform == 'win32':
                # Windows fallback
                import ctypes
                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", ctypes.c_ulong),
                        ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)
                    ]
                GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
                GetReturnValue = ctypes.windll.kernel32.GetCurrentProcess
                counters = PROCESS_MEMORY_COUNTERS()
                GetProcessMemoryInfo(GetReturnValue(), ctypes.byref(counters), ctypes.sizeof(counters))
                return counters.WorkingSetSize / (1024 * 1024)
        except Exception:
            pass
        return -1.0


def get_concern(cand: dict, row: pd.Series) -> str:
    """Helper to determine if a candidate has a concern to report in reasoning."""
    if row.get("soft_disq_recent_langchain", 0) == 1:
        return "recent experience is focused only on LangChain/OpenAI wrappers"
    if row.get("soft_disq_senior_no_code", 0) == 1:
        return "moved to leadership and has not coded in past 18 months"
    if row.get("soft_disq_consulting_only", 0) == 1:
        return "career is entirely in IT consulting services"
    if row.get("soft_disq_cv_speech_robotics", 0) == 1:
        return "background is primarily computer vision/speech/robotics without NLP"
    if row.get("soft_disq_title_chaser", 0) == 1:
        return "frequent job changes with rapid title growth"
    if row.get("soft_disq_closed_source", 0) == 1:
        return "experience is entirely on closed-source systems without public validation"
    
    signals = cand.get("redrob_signals", {})
    days = signals.get("notice_period_days", 0)
    if days > 30:
        return f"notice period is {days} days"
        
    loc = cand.get("profile", {}).get("location", "").lower()
    willing = signals.get("willing_to_relocate", False)
    # Check if they are not in Pune or Noida
    in_pref = any(p in loc for p in ["pune", "noida"])
    if not in_pref and willing:
        return "will require relocation to Pune/Noida"
        
    resp_rate = signals.get("recruiter_response_rate", 1.0)
    if resp_rate < 0.6:
        return "has a lower response rate to recruiters"
        
    return None


def get_recency_text(cand: dict, current_date_str: str = "2026-06-17") -> str:
    """Helper to convert last active date to human text."""
    signals = cand.get("redrob_signals", {})
    last_active = signals.get("last_active_date", "")
    if not last_active:
        return "inactive for several months"
    try:
        curr_date = datetime.strptime(current_date_str, "%Y-%m-%d").date()
        act_date = datetime.strptime(last_active, "%Y-%m-%d").date()
        days = (curr_date - act_date).days
        if days <= 7:
            return "active this week"
        elif days <= 14:
            return "active last week"
        elif days <= 30:
            return "active this month"
        elif days <= 90:
            return "active 2-3 months ago"
        else:
            return f"inactive for {days // 30} months"
    except Exception:
        return "active recently"


def extract_top_matched_skills(cand: dict, must_have_skills: dict) -> list:
    """Extract top 2-3 matched must-have skills with their proficiency."""
    matched = []
    cand_skills = cand.get("skills", [])
    for cat_name, cat_def in must_have_skills.items():
        terms = cat_def.get("terms", [])
        for skill in cand_skills:
            name = skill.get("name", "").lower()
            if any(t.lower() in name for t in terms):
                matched.append((skill.get("name"), skill.get("proficiency", "intermediate")))
                break  # match found for this category, proceed to next
    return matched[:3]


def main():
    parser = argparse.ArgumentParser(description="Rank candidates for Redrob senior AI engineer role.")
    parser.add_argument("--candidates", type=str, default="data/candidates.jsonl", help="Path to candidates.jsonl")
    parser.add_argument("--out", type=str, default="submission.csv", help="Path to write ranked submission.csv")
    args = parser.parse_args()
    
    # Force offline for embedding loading just in case someone imports transformer modules later
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    start_time = time.time()
    initial_mem = get_memory_usage_mb()
    
    print("--------------------------------------------------")
    print("Starting Redrob AI Intelligent Candidate Ranker")
    print("--------------------------------------------------")
    
    # 1. Load configuration and weights
    jd_config_path = os.path.join(base_dir, "config", "jd_requirements.yaml")
    weights_path = os.path.join(base_dir, "config", "weights.yaml")
    
    with open(jd_config_path, "r", encoding="utf-8") as f:
        jd_config = yaml.safe_load(f)
    with open(weights_path, "r", encoding="utf-8") as f:
        weights = yaml.safe_load(f)
        
    must_have_skills = jd_config.get("must_have_skills", {})
    linear_w = weights.get("linear_weights", {})
    soft_disq_w = weights.get("soft_disqualifier_multipliers", {})
    
    # 2. Load precomputed artifacts
    artifacts_dir = os.path.join(base_dir, "artifacts")
    print("Loading precomputed artifacts...")
    
    embeddings_file = os.path.join(artifacts_dir, "embeddings.npy")
    ids_file = os.path.join(artifacts_dir, "ids.npy")
    jd_vector_file = os.path.join(artifacts_dir, "jd_vector.npy")
    features_file = os.path.join(artifacts_dir, "features.parquet")
    
    if not (os.path.exists(embeddings_file) and os.path.exists(ids_file) and 
            os.path.exists(jd_vector_file) and os.path.exists(features_file)):
        print("Error: Precomputed artifacts missing. Please run scripts/01_build_embeddings.py, 02_build_lexical_index.py, 03_build_features.py, and 04_build_jd_vector.py first.")
        sys.exit(1)
        
    embeddings = np.load(embeddings_file)
    ids_array = np.load(ids_file)
    jd_vector = np.load(jd_vector_file)
    features_df = pd.read_parquet(features_file)
    
    # Verify alignment
    if not np.array_equal(features_df["candidate_id"].values, ids_array):
        print("Warning: Features candidate ID ordering does not match embeddings candidate IDs. Aligning features...")
        features_df = features_df.set_index("candidate_id").loc[ids_array].reset_index()
        
    # 3. Compute semantic fit (Cosine similarity)
    print("Computing semantic fit...")
    dot_products = np.dot(embeddings, jd_vector)
    emb_norms = np.linalg.norm(embeddings, axis=1)
    jd_norm = np.linalg.norm(jd_vector)
    
    # Avoid zero-division
    emb_norms[emb_norms == 0] = 1.0
    if jd_norm == 0:
        jd_norm = 1.0
        
    similarities = dot_products / (emb_norms * jd_norm)
    semantic_fit = min_max_normalize(similarities)
    
    # 4. Compute lexical fit (TF-IDF similarity)
    print("Computing lexical fit...")
    vec_pkl = os.path.join(artifacts_dir, "tfidf_vectorizer.pkl")
    mat_pkl = os.path.join(artifacts_dir, "tfidf_matrix.pkl")
    
    with open(vec_pkl, "rb") as f:
        vectorizer = pickle.load(f)
    with open(mat_pkl, "rb") as f:
        tfidf_matrix = pickle.load(f)
        
    # Construct lexical query text from must-have and nice-to-have terms
    query_terms = []
    for s_def in must_have_skills.values():
        query_terms.extend(s_def.get("terms", []))
    for s_def in jd_config.get("nice_to_have_skills", {}).values():
        query_terms.extend(s_def.get("terms", []))
    query_text = " ".join(query_terms)
    
    query_vec = vectorizer.transform([query_text])
    lexical_raw = tfidf_matrix.dot(query_vec.T).toarray().ravel()
    lexical_fit = min_max_normalize(lexical_raw)
    
    # 5. Load candidates and compute skill depth fit
    print(f"Loading candidates from {args.candidates} and computing skill depth...")
    candidates = []
    skill_depth_raw = []
    
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            candidates.append(cand)
            # Compute skill depth fit raw score
            skill_depth_raw.append(compute_skill_depth_fit_raw(cand, must_have_skills))
            
    skill_depth_raw = np.array(skill_depth_raw)
    
    # Align precomputed artifacts with loaded candidate subset by candidate_id
    candidate_ids = [c["candidate_id"] for c in candidates]
    id_to_precomputed_idx = {cid: idx for idx, cid in enumerate(ids_array)}
    
    precomputed_indices = []
    valid_candidates = []
    valid_skill_depth_raw = []
    
    for idx, cid in enumerate(candidate_ids):
        if cid in id_to_precomputed_idx:
            precomputed_indices.append(id_to_precomputed_idx[cid])
            valid_candidates.append(candidates[idx])
            valid_skill_depth_raw.append(skill_depth_raw[idx])
            
    candidates = valid_candidates
    skill_depth_raw = np.array(valid_skill_depth_raw)
    
    embeddings = embeddings[precomputed_indices]
    ids_array = ids_array[precomputed_indices]
    features_df = features_df.iloc[precomputed_indices].reset_index(drop=True)
    tfidf_matrix = tfidf_matrix[precomputed_indices, :]
    semantic_fit = semantic_fit[precomputed_indices]
    lexical_fit = lexical_fit[precomputed_indices]
    
    skill_depth_fit = min_max_normalize(skill_depth_raw)
    
    # 6. Compute structural fit
    print("Computing structural fit...")
    exp_score = features_df["experience_band_score"].values
    prod_score = features_df["production_evidence_score"].values
    
    soft_penalties = (
        features_df["soft_disq_recent_langchain"].values * 0.15 +
        features_df["soft_disq_senior_no_code"].values * 0.10 +
        features_df["soft_disq_consulting_only"].values * 0.20 +
        features_df["soft_disq_cv_speech_robotics"].values * 0.25 +
        features_df["soft_disq_title_chaser"].values * 0.20 +
        features_df["soft_disq_closed_source"].values * 0.10
    )
    
    struct_raw = exp_score * 0.5 + prod_score * 0.5 - soft_penalties
    struct_raw = np.clip(struct_raw, 0.0, 1.0)
    structural_fit = min_max_normalize(struct_raw)
    
    # 7. Compute logistics fit
    print("Computing logistics fit...")
    loc_fit = features_df["location_fit"].values
    notice_fit = features_df["notice_period_fit"].values
    sal_fit = features_df["salary_sanity_fit"].values
    
    logistics_raw = loc_fit * 0.5 + notice_fit * 0.3 + sal_fit * 0.2
    logistics_fit = min_max_normalize(logistics_raw)
    
    # 8. Assemble linear score
    print("Assembling linear scores...")
    linear_score = (
        linear_w.get("semantic_fit", 0.28) * semantic_fit +
        linear_w.get("skill_depth_fit", 0.20) * skill_depth_fit +
        linear_w.get("lexical_fit", 0.12) * lexical_fit +
        linear_w.get("structural_fit", 0.30) * structural_fit +
        linear_w.get("logistics_fit", 0.10) * logistics_fit
    )
    
    # 9. Compute behavioral multipliers
    print("Computing behavioral multipliers...")
    # Map row parameters to compute multiplier vector
    # This is fast using apply or raw lists
    behavioral_mults = []
    # Using a fast row loop to utilize our datetime parsing logic
    for cand in candidates:
        behavioral_mults.append(compute_behavioral_multiplier(cand))
    behavioral_mults = np.array(behavioral_mults)
    
    # 10. Apply soft disqualifiers multipliers
    print("Applying soft disqualifier multipliers...")
    soft_disq_mult = np.ones(len(candidates))
    soft_disq_mult[features_df["soft_disq_recent_langchain"] == 1] *= soft_disq_w.get("recent_langchain_only_no_legacy_ml", 0.25)
    soft_disq_mult[features_df["soft_disq_senior_no_code"] == 1] *= soft_disq_w.get("senior_no_hands_on_code_18mo", 0.30)
    soft_disq_mult[features_df["soft_disq_consulting_only"] == 1] *= soft_disq_w.get("consulting_only_career", 0.20)
    soft_disq_mult[features_df["soft_disq_cv_speech_robotics"] == 1] *= soft_disq_w.get("cv_speech_robotics_only_no_nlp", 0.15)
    soft_disq_mult[features_df["soft_disq_title_chaser"] == 1] *= soft_disq_w.get("title_chaser_job_hopper", 0.20)
    soft_disq_mult[features_df["soft_disq_closed_source"] == 1] *= soft_disq_w.get("closed_source_only_no_external_validation", 0.30)
    
    # Final combined score
    final_scores = linear_score * behavioral_mults * soft_disq_mult
    
    # 11. Apply exclusions (Honeypot + Hard disqualifiers)
    print("Applying hard disqualifier and honeypot filters...")
    is_honeypot = features_df["is_honeypot"].values
    hard_disq = features_df["hard_disqualified"].values
    
    eligible_indices = np.where((is_honeypot == 0) & (hard_disq == 0))[0]
    print(f"Total candidates: {len(candidates)}")
    print(f"Eligible candidates: {len(eligible_indices)}")
    print(f"Filtered out: {len(candidates) - len(eligible_indices)} (Honeypots/Disqualified)")
    
    # 12. Sort and extract top 100
    sort_list = []
    for idx in eligible_indices:
        sort_list.append({
            "idx": idx,
            "id": ids_array[idx],
            "score": final_scores[idx]
        })
        
    # Sort by score descending, then candidate_id ascending for tie-breaking
    sort_list.sort(key=lambda x: (-x["score"], x["id"]))

    # Guarantee exactly 100 rows. If the eligible pool is smaller than 100
    # (e.g. a tiny input set where filters removed too many), backfill with the
    # filtered-out candidates ranked by their scores so the submission always
    # has the required 100 ranks. With the full candidate pool this branch is
    # never taken, but it keeps the output format-valid for any input size.
    if len(sort_list) < 100:
        eligible_set = set(int(i) for i in eligible_indices)
        backfill = [
            {"idx": idx, "id": ids_array[idx], "score": final_scores[idx]}
            for idx in range(len(candidates)) if idx not in eligible_set
        ]
        backfill.sort(key=lambda x: (-x["score"], x["id"]))
        sort_list = sort_list + backfill

    top_100_results = sort_list[:100]
    
    # 13. Generate reasoning for top 100
    print("Generating deterministic reasoning for the top 100 candidates...")
    submission_rows = []
    
    for rank_idx, item in enumerate(top_100_results, start=1):
        idx = item["idx"]
        cand = candidates[idx]
        row_feat = features_df.iloc[idx]
        
        # Build facts dictionary
        facts = {
            "current_title": cand.get("profile", {}).get("current_title"),
            "current_company": cand.get("profile", {}).get("current_company"),
            "years_of_experience": cand.get("profile", {}).get("years_of_experience"),
            "top_skills": extract_top_matched_skills(cand, must_have_skills),
            "concern": get_concern(cand, row_feat),
            "recency_text": get_recency_text(cand)
        }
        
        reasoning = generate_reasoning(cand, facts, mode="template")
        
        submission_rows.append({
            "candidate_id": item["id"],
            "rank": rank_idx,
            "score": round(float(item["score"]), 6),
            "reasoning": reasoning
        })
        
    # 14. Write submission.csv
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(args.out, index=False)
    print(f"Submission saved to {args.out}")
    
    # Print metrics
    total_time = time.time() - start_time
    final_mem = get_memory_usage_mb()
    
    print("\n--------------------------------------------------")
    print("Execution Summary:")
    print(f"Time Taken  : {total_time:.2f} seconds")
    if final_mem > 0:
        print(f"RAM Used    : {final_mem:.2f} MB (change: {(final_mem - initial_mem):.2f} MB)")
    else:
        print("RAM Used    : N/A (psutil not available)")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
