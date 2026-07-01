import os
import sys
import json
import yaml
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import re
from datetime import datetime

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from src.scoring import (
    compute_skill_depth_fit_raw,
    compute_behavioral_multiplier,
    min_max_normalize,
    match_skill_term
)
from src.reasoning import generate_reasoning
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
    check_experience_band_score,
    check_irrelevant_role
)





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
            is_match = False
            for t in terms:
                if match_skill_term(t, name):
                    is_match = True
                    break
            if is_match:
                matched.append((skill.get("name"), skill.get("proficiency", "intermediate")))
                break  # match found for this category, proceed to next
    return matched[:3]


def main():
    parser = argparse.ArgumentParser(description="Rank candidates for Redrob senior AI engineer role.")
    parser.add_argument("--candidates", type=str, default="data/candidates.jsonl", help="Path to candidates.jsonl")
    parser.add_argument("--out", type=str, default="submission.csv", help="Path to write ranked submission.csv")
    parser.add_argument("--jd-text", type=str, default=None, help="Custom Job Description text to rank against dynamically")
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
    
    # Check for custom Job Description override
    custom_jd_mode = args.jd_text is not None and args.jd_text.strip() != ""
    custom_jd_vector = None
    
    if custom_jd_mode:
        print("Custom Job Description detected. Analyzing and building dynamic rubric...")
        parsed_jd = parse_jd(args.jd_text)
        must_have_skills = parsed_jd["must_have_skills"]
        linear_w = parsed_jd["dimension_weights"]
        
        # Load experience band override
        exp_band = parsed_jd["experience_band"]
        # Update jd_config with parsed experience band
        jd_config["experience_band"] = exp_band
        
        # Load SentenceTransformer model to encode the custom JD
        from sentence_transformers import SentenceTransformer
        model_path = os.path.join(base_dir, "models", "all-MiniLM-L6-v2")
        if os.path.exists(model_path):
            model = SentenceTransformer(model_path)
        else:
            model = SentenceTransformer("all-MiniLM-L6-v2")
        custom_jd_vector = model.encode(args.jd_text, convert_to_numpy=True)
    
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
    
    if custom_jd_mode:
        jd_vector = custom_jd_vector
    else:
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
    
    # 6. Recruiter Scoring Logic
    print("Computing recruiter scores using hard disqualifiers, must-have tiering and adjustments...")
    
    # Helper to match skill terms
    def match_skill_term_local(term: str, skill_name: str) -> bool:
        term_lower = term.lower()
        skill_lower = skill_name.lower()
        if term_lower in ["c++", "c#", ".net"]:
            return term_lower in skill_lower
        if len(term_lower) <= 3:
            pattern = r'\b' + re.escape(term_lower) + r'\b'
            return bool(re.search(pattern, skill_lower))
        return term_lower in skill_lower

    # Helper to check must-have matches with supporting evidence
    def check_must_have_matches(cand):
        skills = cand.get("skills", [])
        history = cand.get("career_history", [])
        profile = cand.get("profile", {})
        
        headline = profile.get("headline", "").lower()
        summary = profile.get("summary", "").lower()
        
        # Build history text
        history_text = " ".join([
            (job.get("title", "") + " " + job.get("description", "")).lower()
            for job in history
        ])
        
        profile_text = f"{headline} {summary} {history_text}"
        
        categories = {
            "embeddings_retrieval": ["sentence-transformers", "openai embeddings", "bge", "e5", "embeddings", "dense retrieval", "semantic search"],
            "vector_db_hybrid_search": ["pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss", "hybrid search", "vector database"],
            "strong_python": ["python"],
            "eval_frameworks": ["ndcg", "mrr", "map", "a/b test", "ab testing", "offline evaluation", "online evaluation", "ranking evaluation"]
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
                is_tech_role = any(r in profile.get("current_title", "").lower() for r in ["engineer", "scientist", "developer", "programmer", "architect", "analyst", "mle", "lead"])
                if is_tech_role:
                    has_text_evidence = True
                    
            if has_skill_in_list or has_text_evidence:
                matched_count += 1
        return matched_count

    # Helper to check good-to-haves
    def check_good_to_have_matches(cand):
        skills = cand.get("skills", [])
        good_to_haves = {
            "llm_finetuning": ["lora", "qlora", "peft", "fine-tuning", "finetuning"],
            "learning_to_rank": ["learning to rank", "ltr", "xgboost ranking", "neural ranking"],
            "hr_tech": ["hr tech", "recruiting", "talent", "marketplace"],
            "distributed_systems": ["distributed systems", "large-scale inference", "model serving at scale"],
            "open_source": ["open source", "github", "publication", "paper", "conference talk"]
        }
        count = 0
        for cat, terms in good_to_haves.items():
            for s in skills:
                name = s.get("name", "").lower()
                if any(match_skill_term_local(t, name) for t in terms):
                    count += 1
                    break
        return count

    final_scores = np.zeros(len(candidates))
    is_honeypot_arr = features_df["is_honeypot"].values
    hard_disq_arr = features_df["hard_disqualified"].values
    
    role_type = "ml_ai"
    if custom_jd_mode:
        role_type = parsed_jd["role_type"]
        
    eligible_mask = np.ones(len(candidates), dtype=bool)
    
    for idx, cand in enumerate(candidates):
        row_feat = features_df.iloc[idx]
        
        # Check hard disqualifiers
        is_hp = is_honeypot_arr[idx] == 1 or detect_honeypot(cand)
        is_irrelevant = check_irrelevant_role(cand, role_type)
        is_pure_res = hard_disq_arr[idx] == 1 or check_pure_research_no_production(cand)
        
        is_consulting = row_feat["soft_disq_consulting_only"] == 1
        is_cv_robotics = row_feat["soft_disq_cv_speech_robotics"] == 1
        is_senior_nocode = row_feat["soft_disq_senior_no_code"] == 1
        is_langchain = row_feat["soft_disq_recent_langchain"] == 1
        is_hopper = row_feat["soft_disq_title_chaser"] == 1
        is_closed_source = row_feat["soft_disq_closed_source"] == 1
        
        # Determine if candidate hits any hard disqualifier or exclusion
        disq_scores = []
        if is_hp: disq_scores.append(0.0)
        if is_irrelevant: disq_scores.append(0.0)
        if is_pure_res: disq_scores.append(0.05)
        if is_consulting: disq_scores.append(0.03)
        if is_cv_robotics: disq_scores.append(0.07)
        if is_senior_nocode: disq_scores.append(0.08)
        if is_langchain: disq_scores.append(0.12)
        if is_hopper: disq_scores.append(0.10)
        if is_closed_source: disq_scores.append(0.09)
        
        if disq_scores:
            final_scores[idx] = min(disq_scores)
            eligible_mask[idx] = False
            continue
            
        # STEP 3 — MUST_HAVE scoring
        matched_must = check_must_have_matches(cand)
        
        # Base ranges
        if matched_must == 4:
            range_min, range_max = 0.70, 0.85
        elif matched_must == 3:
            range_min, range_max = 0.40, 0.55
        else:
            range_min, range_max = 0.10, 0.25
            
        # Combine fits to interpolate
        s_val = 0.5 * semantic_fit[idx] + 0.3 * skill_depth_fit[idx] + 0.2 * lexical_fit[idx]
        base_score = range_min + s_val * (range_max - range_min)
        
        # STEP 4 — Adjustments
        adj = 0.0
        
        # Confirmed good-to-haves
        gth_count = check_good_to_have_matches(cand)
        adj += gth_count * 0.03
        
        # Active on platform recently (last 30 days)
        last_active_str = row_feat["last_active_date"]
        days = 365
        if last_active_str:
            try:
                curr_date = datetime.strptime("2026-06-17", "%Y-%m-%d").date()
                act_date = datetime.strptime(last_active_str, "%Y-%m-%d").date()
                days = (curr_date - act_date).days
            except:
                pass
        
        if days <= 30:
            adj += 0.03
        elif days >= 180:
            adj -= 0.05
            
        # Open to work flag
        if row_feat["open_to_work_flag"] == 1:
            adj += 0.02
            
        # Notice period
        notice_days = cand.get("redrob_signals", {}).get("notice_period_days", 0)
        if notice_days <= 30:
            adj += 0.03
        elif notice_days <= 60:
            adj += 0.01
            
        if notice_days > 90:
            adj -= 0.02
        if notice_days > 120:
            adj -= 0.04
            
        score = base_score + adj
        final_scores[idx] = max(0.0, min(1.0, score))

    # Apply hard disqualifier, honeypot, and irrelevant role filters
    eligible_indices = np.where(eligible_mask)[0]
    print(f"Total candidates: {len(candidates)}")
    print(f"Eligible candidates: {len(eligible_indices)}")
    print(f"Filtered out: {len(candidates) - len(eligible_indices)} (Honeypots/Disqualified/Irrelevant)")
    
    # Sort and extract top 100
    sort_list = []
    for idx in eligible_indices:
        sort_list.append({
            "idx": idx,
            "id": ids_array[idx],
            "score": final_scores[idx]
        })
        
    sort_list.sort(key=lambda x: (-x["score"], x["id"]))

    if len(sort_list) < 100:
        eligible_set = set(int(i) for i in eligible_indices)
        backfill = [
            {"idx": idx, "id": ids_array[idx], "score": final_scores[idx]}
            for idx in range(len(candidates)) if idx not in eligible_set
        ]
        backfill.sort(key=lambda x: (-x["score"], x["id"]))
        sort_list = sort_list + backfill
        sort_list.sort(key=lambda x: (-x["score"], x["id"]))

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
